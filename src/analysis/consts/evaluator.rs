//! A function's constants, found by evaluating its instructions in order
//! ([`ConstEvaluator`]).

use rustc_hash::FxHashMap;

use crate::{
    BitSet,
    analysis::consts::{
        Machine, TypeQueries,
        layout::{Destination, OperandLayout},
    },
    ir::{
        function::SsaFunction, integer::IntegerBits, ops::SsaOp, value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// Cached concrete and partial integer facts for one SSA definition.
#[derive(Debug, Clone)]
pub struct ConstantFacts<T: Target> {
    /// An exact value, including noninteger constants.
    constant: Option<ConstValue<T>>,
    /// Scalar integer bits retained even when the complete value is unknown.
    bits: Option<IntegerBits>,
}

/// Evaluates SSA operations to constant values.
///
/// This provides a unified implementation of constant folding that can be
/// used by multiple passes (unflattening, decryption, SCCP, etc.).
///
/// # Features
///
/// - Caches results for efficiency
/// - Detects cycles to prevent infinite recursion
/// - Supports injecting known values from external analysis
/// - Configurable depth limit
pub struct ConstEvaluator<'a, T: Target> {
    /// Reference to the SSA function being analyzed.
    ssa: &'a SsaFunction<T>,

    /// Externally supplied bindings, retained when another binding changes.
    known: FxHashMap<SsaVarId, ConstantFacts<T>>,

    /// Derived values and partial integer facts for the current supplied bindings.
    cache: FxHashMap<SsaVarId, ConstantFacts<T>>,

    /// Variables currently being evaluated (for cycle detection).
    visiting: BitSet,

    /// Maximum recursion depth.
    max_depth: usize,

    /// How deep the evaluation in progress has recursed.
    depth: usize,

    /// The machine the function computes on.
    machine: Machine,
}

impl<'a, T: Target> ConstEvaluator<'a, T> {
    /// Default maximum recursion depth.
    const DEFAULT_MAX_DEPTH: usize = 20;

    /// Creates an evaluator of `ssa` computing as `machine` does, recursing at
    /// most twenty definitions deep ([`Self::with_max_depth`]).
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `machine` - The machine the function computes on.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, machine: Machine) -> Self {
        Self {
            ssa,
            known: FxHashMap::default(),
            cache: FxHashMap::default(),
            visiting: BitSet::new(ssa.variable_count().max(1)),
            max_depth: Self::DEFAULT_MAX_DEPTH,
            depth: 0,
            machine,
        }
    }

    /// Returns this evaluator recursing at most `max_depth` definitions deep.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Injects a known value from external analysis.
    ///
    /// This allows passes to provide values discovered through other means
    /// (e.g., from `ctx.known_values` in decryption). Injected values take
    /// precedence over computed values. Updating a binding discards derived
    /// constants, unknowns and partial facts so no result from another input
    /// context is reused. Other supplied bindings remain available.
    ///
    /// Seeding several inputs before evaluation does not rescan earlier bindings;
    /// derived entries are cleared only while that cache is nonempty.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to set.
    /// * `value` - The known constant value.
    pub fn set_known(&mut self, var: SsaVarId, value: ConstValue<T>) {
        let bits = self
            .machine
            .constant_integer_bits(&value, OperandLayout::of(self.ssa, var));
        if !self.cache.is_empty() {
            self.cache.clear();
        }
        self.known.insert(
            var,
            ConstantFacts {
                constant: Some(value),
                bits,
            },
        );
    }

    /// Evaluates a variable to a constant if possible.
    ///
    /// Definition traversal is cached, so repeated lookups are O(1) until an
    /// injected binding changes. Returning an owned constant may copy its payload.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to evaluate.
    ///
    /// # Returns
    ///
    /// The constant value if the variable can be evaluated, `None` otherwise.
    pub fn evaluate_var(&mut self, var: SsaVarId) -> Option<ConstValue<T>> {
        if let Some(known) = self.known.get(&var) {
            return known.constant.clone();
        }
        self.populate(var);
        self.cache.get(&var)?.constant.clone()
    }

    /// Computes one definition once, retaining its partial bits without cloning payloads.
    fn populate(&mut self, var: SsaVarId) {
        if self.depth > self.max_depth || self.cache.contains_key(&var) {
            return;
        }
        if var.index() < self.visiting.len() && self.visiting.contains(var.index()) {
            return;
        }
        if var.index() < self.visiting.len() {
            self.visiting.insert(var.index());
        }
        let facts = self
            .ssa
            .get_definition(var)
            .map(|op| self.evaluate_facts(op));
        if var.index() < self.visiting.len() {
            self.visiting.remove(var.index());
        }
        self.cache.insert(
            var,
            facts.unwrap_or(ConstantFacts {
                constant: None,
                bits: None,
            }),
        );
    }

    /// Reads already computed partial facts, evaluating a missing definition once.
    fn integer_bits_var(&mut self, var: SsaVarId) -> Option<IntegerBits> {
        if let Some(known) = self.known.get(&var) {
            return known.bits;
        }
        self.populate(var);
        self.cache.get(&var)?.bits
    }

    /// Evaluates an SSA operation to a constant if possible, its operands one
    /// definition deeper.
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to evaluate.
    ///
    /// # Returns
    ///
    /// The constant value if the operation can be evaluated, `None` otherwise.
    pub fn evaluate_op(&mut self, op: &SsaOp<T>) -> Option<ConstValue<T>> {
        self.evaluate_facts(op).constant
    }

    /// Evaluates concrete semantics and forward bit facts under the same depth bound.
    fn evaluate_facts(&mut self, op: &SsaOp<T>) -> ConstantFacts<T> {
        if self.depth > self.max_depth {
            return ConstantFacts {
                constant: None,
                bits: None,
            };
        }
        self.depth = self.depth.saturating_add(1);
        // A copy's cached value is its source; each typed reader applies its
        // own register interpretation through the machine.
        let machine = self.machine;
        let ssa = self.ssa;
        let dest = Destination::of_operation(ssa, op);
        let mut constant = if let SsaOp::Copy { src, .. } = op {
            self.evaluate_var(*src)
        } else {
            machine.evaluate_with_operand_layouts(
                op,
                dest,
                |var| self.evaluate_var(var),
                |var| OperandLayout::of(ssa, var),
            )
        };
        let types = TypeQueries::ordinary(machine.ptr_size);
        let bits = constant
            .as_ref()
            .and_then(|value| machine.constant_integer_bits(value, OperandLayout::Register(dest)))
            .or_else(|| {
                machine.evaluate_integer_bits(
                    op,
                    dest,
                    |var, _| self.integer_bits_var(var),
                    |var| OperandLayout::of(ssa, var),
                    &types,
                )
            });
        if constant.is_none() {
            constant = bits.and_then(|bits| machine.known_result(op, bits, dest));
        }
        self.depth = self.depth.saturating_sub(1);
        ConstantFacts { constant, bits }
    }

    /// Returns all computed constants.
    ///
    /// This consumes the evaluator and returns a map of all variables
    /// that were successfully evaluated to constants, including supplied bindings.
    /// Invalidated results from earlier input contexts are absent.
    #[must_use]
    pub fn into_results(self) -> FxHashMap<SsaVarId, ConstValue<T>> {
        self.cache
            .into_iter()
            .chain(self.known)
            .filter_map(|(var, facts)| facts.constant.map(|val| (var, val)))
            .collect()
    }

    /// Returns a reference to the SSA function being evaluated.
    #[must_use]
    pub fn ssa(&self) -> &SsaFunction<T> {
        self.ssa
    }

    /// Clears derived facts and all externally supplied bindings.
    ///
    /// The next lookup reads the SSA definition again unless another binding
    /// is supplied. This resets input assumptions as well as their consequences.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.known.clear();
    }
}
