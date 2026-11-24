//! Symbolic expression tree representation.
//!
//! This module defines [`SymbolicExpr`], an intermediate representation for
//! symbolic values that can contain typed constants, SSA variables, named
//! variables, opaque definition instances, and operation nodes. Expressions form a tree that
//! maps directly to SSA operations.
//!
//! # Expression Types
//!
//! | Variant | Meaning | Example |
//! |---------|---------|---------|
//! | `Constant(v)` | Typed constant value | `I32(42)` |
//! | `Variable(id)` | Reference to SSA variable | `v5` |
//! | `NamedVar(name)` | External symbolic input | `"state"` |
//! | `Instance(token)` | One opaque execution of a definition | `v5@2` |
//! | `Unary(op, operand)` | Unary operation | `(-x)` |
//! | `Binary(op, left, right)` | Binary operation | `(x + y)` |
//!
//! # Substitution
//!
//! Expressions support substitution of SSA variables and named variables:
//! - `substitute(var, replacement)`: Replace all occurrences of an SSA variable
//!   with another expression
//! - `substitute_named(name, value)`: Replace a named variable with a constant
//!   and simplify
//! - `substitute_named(name, replacement)`: Replace with an expression

use std::{fmt, sync::Arc};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    PointerSize,
    analysis::{consts::scalar::ScalarOperation, symbolic::instance::SymbolicInstance},
    ir::{
        evaluation::{Evaluation, NumericOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// A symbolic expression that can contain variables, constants, and operations.
///
/// This is our intermediate representation for symbolic values. It maps directly
/// to SSA operations and can be translated to Z3 for constraint solving.
///
/// Operands are shared: a relation built over an operand's expression holds it
/// without copying, so cloning an expression copies one node.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpr<T: Target> {
    /// A typed constant value preserving CIL type information.
    Constant(ConstValue<T>),

    /// A symbolic variable (identified by SSA variable ID).
    Variable(SsaVarId),

    /// One opaque definition execution, independent of later executions.
    ///
    /// Its source id is provenance, not an ordinary SSA binding. Enumeration,
    /// substitution and evaluation preserve that distinction.
    Instance(SymbolicInstance),

    /// A named symbolic variable (for external inputs like "state").
    NamedVar(String),

    /// A unary operation.
    Unary {
        /// The operation to perform.
        op: NumericOp,
        /// The operand.
        operand: Arc<SymbolicExpr<T>>,
    },

    /// A binary operation.
    Binary {
        /// The operation to perform.
        op: NumericOp,
        /// The left operand.
        left: Arc<SymbolicExpr<T>>,
        /// The right operand.
        right: Arc<SymbolicExpr<T>>,
    },

    /// A unary operation retaining its declared scalar semantics.
    TypedUnary {
        /// The complete immutable operation and machine contract.
        op: Box<ScalarOperation>,
        /// The input expression, including any definedness obligation.
        operand: Arc<SymbolicExpr<T>>,
    },

    /// A binary operation retaining its declared scalar semantics.
    TypedBinary {
        /// The complete immutable operation and machine contract.
        op: Box<ScalarOperation>,
        /// The left input expression.
        left: Arc<SymbolicExpr<T>>,
        /// The right input expression.
        right: Arc<SymbolicExpr<T>>,
    },
}

/// Maximum depth admitted to recursive simplification and display.
const MAX_RECURSIVE_EXPR_DEPTH: usize = 512;

/// Maximum node count before an expression is replaced by an opaque variable.
///
/// Depth alone does not bound an expression. `v = x op x` holds its operand
/// twice, so it *doubles* the node count while adding exactly 1 to the depth - a
/// straight-line run of such instructions reaches 2^n nodes at depth n. Shared
/// operands keep the memory small, but every walk (evaluation, simplification,
/// display, substitution) visits all 2^n nodes. The depth cap of 512 is
/// therefore unreachable in practice, and `simplify` does not collapse `x + x`,
/// so nothing shrinks the tree either.
///
/// A few thousand nodes is far past anything a real dispatch expression needs,
/// and stopping there degrades precision (the value becomes an opaque variable)
/// rather than exhausting time.
const MAX_EXPR_NODES: usize = 4096;

/// One pending leaf visit or numeric operation in iterative evaluation.
enum EvalTask<'a, T: Target> {
    Visit(&'a SymbolicExpr<T>),
    EvalUnary(NumericOp),
    EvalBinary(NumericOp),
    EvalTypedUnary(ScalarOperation),
    EvalTypedBinary(ScalarOperation),
}

/// One pending visit or reconstruction step in an iterative expression copy.
enum RebuildTask<'a, T: Target> {
    Visit(&'a SymbolicExpr<T>),
    BuildUnary(NumericOp),
    BuildBinary(NumericOp),
    BuildTypedUnary(ScalarOperation),
    BuildTypedBinary(ScalarOperation),
}

impl<T: Target> SymbolicExpr<T> {
    /// Copies the expression tree, each leaf `replace` answers for replaced by
    /// a copy of its answer, without recursing on the call stack.
    ///
    /// A recursive copy descends once per level, so an adversarially deep tree
    /// would overflow the stack before it could be copied. This walks the tree
    /// with an explicit worklist instead: visiting a node pushes its children
    /// followed by a deferred `Build*` marker, and the marker later reassembles
    /// the node from the values its children left on the `values` stack. A
    /// malformed task stream (an operator with a missing operand) degrades to a
    /// placeholder `NamedVar` rather than panicking.
    ///
    /// # Arguments
    ///
    /// * `replace` - Given a leaf (a constant or a variable), the expression it
    ///   is replaced by, or `None` to keep it.
    ///
    /// # Returns
    ///
    /// The copy.
    fn rebuilt<'r>(&self, replace: &dyn Fn(&Self) -> Option<&'r Self>) -> Self
    where
        T: 'r,
    {
        let mut tasks = vec![RebuildTask::Visit(self)];
        let mut values = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                RebuildTask::Visit(leaf @ Self::Constant(value)) => {
                    values.push(
                        replace(leaf).map_or_else(|| Self::Constant(value.clone()), Self::clone),
                    );
                }
                RebuildTask::Visit(leaf @ Self::Variable(v)) => {
                    values.push(replace(leaf).map_or_else(|| Self::Variable(*v), Self::clone));
                }
                RebuildTask::Visit(leaf @ Self::Instance(instance)) => {
                    values
                        .push(replace(leaf).map_or_else(|| Self::Instance(*instance), Self::clone));
                }
                RebuildTask::Visit(leaf @ Self::NamedVar(name)) => {
                    values.push(
                        replace(leaf).map_or_else(|| Self::NamedVar(name.clone()), Self::clone),
                    );
                }
                RebuildTask::Visit(Self::Unary { op, operand }) => {
                    tasks.push(RebuildTask::BuildUnary(*op));
                    tasks.push(RebuildTask::Visit(operand));
                }
                RebuildTask::Visit(Self::Binary { op, left, right }) => {
                    tasks.push(RebuildTask::BuildBinary(*op));
                    tasks.push(RebuildTask::Visit(right));
                    tasks.push(RebuildTask::Visit(left));
                }
                RebuildTask::Visit(Self::TypedUnary { op, operand }) => {
                    tasks.push(RebuildTask::BuildTypedUnary(**op));
                    tasks.push(RebuildTask::Visit(operand));
                }
                RebuildTask::Visit(Self::TypedBinary { op, left, right }) => {
                    tasks.push(RebuildTask::BuildTypedBinary(**op));
                    tasks.push(RebuildTask::Visit(right));
                    tasks.push(RebuildTask::Visit(left));
                }
                RebuildTask::BuildTypedUnary(op) => {
                    let Some(operand) = values.pop() else {
                        return Self::NamedVar("invalid-symbolic-expression".to_string());
                    };
                    values.push(Self::TypedUnary {
                        op: Box::new(op),
                        operand: Arc::new(operand),
                    });
                }
                RebuildTask::BuildTypedBinary(op) => {
                    let (Some(right), Some(left)) = (values.pop(), values.pop()) else {
                        return Self::NamedVar("invalid-symbolic-expression".to_string());
                    };
                    values.push(Self::TypedBinary {
                        op: Box::new(op),
                        left: Arc::new(left),
                        right: Arc::new(right),
                    });
                }
                RebuildTask::BuildUnary(op) => {
                    let Some(operand) = values.pop() else {
                        return Self::NamedVar("invalid-symbolic-expression".to_string());
                    };
                    values.push(Self::Unary {
                        op,
                        operand: Arc::new(operand),
                    });
                }
                RebuildTask::BuildBinary(op) => {
                    let Some(right) = values.pop() else {
                        return Self::NamedVar("invalid-symbolic-expression".to_string());
                    };
                    let Some(left) = values.pop() else {
                        return Self::NamedVar("invalid-symbolic-expression".to_string());
                    };
                    values.push(Self::Binary {
                        op,
                        left: Arc::new(left),
                        right: Arc::new(right),
                    });
                }
            }
        }

        values
            .pop()
            .unwrap_or_else(|| Self::NamedVar("invalid-symbolic-expression".to_string()))
    }

    /// Creates a typed unary expression, refusing a descriptor with another arity.
    ///
    /// Callers enforce construction budgets before cloning operand expressions.
    #[must_use]
    pub fn typed_unary(op: ScalarOperation, operand: Self) -> Option<Self> {
        (op.arity() == 1).then(|| Self::TypedUnary {
            op: Box::new(op),
            operand: Arc::new(operand),
        })
    }

    /// Creates a typed binary expression, refusing a descriptor with another arity.
    ///
    /// Callers enforce construction budgets before cloning operand expressions.
    #[must_use]
    pub fn typed_binary(op: ScalarOperation, left: Self, right: Self) -> Option<Self> {
        (op.arity() == 2).then(|| Self::TypedBinary {
            op: Box::new(op),
            left: Arc::new(left),
            right: Arc::new(right),
        })
    }

    /// Returns the number of nodes in this expression tree.
    ///
    /// Counts every constant, variable, and operator node. Iterative, so a deep
    /// tree cannot overflow the stack.
    #[must_use]
    pub fn node_count(&self) -> usize {
        let mut count = 0usize;
        let mut stack = vec![self];
        while let Some(expr) = stack.pop() {
            count = count.saturating_add(1);
            match expr {
                Self::Constant(_) | Self::Variable(_) | Self::NamedVar(_) | Self::Instance(_) => {}
                Self::Unary { operand, .. } | Self::TypedUnary { operand, .. } => {
                    stack.push(operand)
                }
                Self::Binary { left, right, .. } | Self::TypedBinary { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        count
    }

    /// Returns `true` when a parent over `operands` would exceed the node or
    /// depth budget, so the caller should substitute an opaque variable instead.
    ///
    /// The node budget counts every operand node together; the depth budget
    /// holds for each operand, where deeper recursive combination risks stack
    /// exhaustion. Checked on the operands rather than the result, so the
    /// oversized tree is never built. Depth alone does not bound an expression:
    /// `v = x op x` doubles the node count while adding one to the depth, so the
    /// node budget is a few thousand. One iterative walk stops at the first
    /// budget reached, so the cost is bounded by the budgets, not the operands.
    #[must_use]
    pub fn combining_reaches_limits(operands: &[&Self]) -> bool {
        if operands.iter().all(|operand| operand.is_leaf()) {
            return operands.len() >= MAX_EXPR_NODES;
        }
        let mut nodes = 0usize;
        let mut stack: Vec<(&Self, usize)> = operands.iter().map(|operand| (*operand, 0)).collect();
        while let Some((expr, depth)) = stack.pop() {
            nodes = nodes.saturating_add(1);
            if nodes >= MAX_EXPR_NODES || depth >= MAX_RECURSIVE_EXPR_DEPTH {
                return true;
            }
            let child = depth.saturating_add(1);
            match expr {
                Self::Constant(_) | Self::Variable(_) | Self::NamedVar(_) | Self::Instance(_) => {}
                Self::Unary { operand, .. } | Self::TypedUnary { operand, .. } => {
                    stack.push((operand, child));
                }
                Self::Binary { left, right, .. } | Self::TypedBinary { left, right, .. } => {
                    stack.push((right, child));
                    stack.push((left, child));
                }
            }
        }
        false
    }

    /// Returns whether this expression has no operation node.
    const fn is_leaf(&self) -> bool {
        matches!(
            self,
            Self::Constant(_) | Self::Variable(_) | Self::NamedVar(_) | Self::Instance(_)
        )
    }

    /// Returns whether partial facts may discard this input's computation.
    ///
    /// Leaves denote supplied normal values. A computed input is admitted only
    /// after it evaluates normally; unknown computations retain their obligations.
    pub fn admits_normal_input(&self, ptr_size: PointerSize) -> bool {
        if self.is_leaf() {
            return true;
        }
        let result = self.evaluate_detail_with(ptr_size, |_| Evaluation::Unknown);
        matches!(result.outcome, Evaluation::Value(_))
            || (matches!(result.outcome, Evaluation::Unknown) && result.normal_unknown)
    }

    /// Creates a constant expression from a typed `ConstValue`.
    ///
    /// This is the preferred constructor as it preserves type information.
    ///
    /// # Arguments
    ///
    /// * `value` - The typed constant value.
    ///
    /// # Returns
    ///
    /// A new [`SymbolicExpr::Constant`] containing the value.
    #[must_use]
    pub fn constant(value: ConstValue<T>) -> Self {
        Self::Constant(value)
    }

    /// Creates a constant expression from an i64 value.
    ///
    /// The value is stored as `ConstValue::I64`. For type-preserving operations,
    /// use [`constant`](Self::constant) with an explicit `ConstValue` instead.
    ///
    /// # Arguments
    ///
    /// * `value` - The integer value.
    ///
    /// # Returns
    ///
    /// A new [`SymbolicExpr::Constant`] containing the value as I64.
    #[must_use]
    pub fn constant_i64(value: i64) -> Self {
        Self::Constant(ConstValue::I64(value))
    }

    /// Creates a constant expression from an i32 value.
    ///
    /// # Arguments
    ///
    /// * `value` - The integer value.
    ///
    /// # Returns
    ///
    /// A new [`SymbolicExpr::Constant`] containing the value as I32.
    #[must_use]
    pub fn constant_i32(value: i32) -> Self {
        Self::Constant(ConstValue::I32(value))
    }

    /// Creates a variable expression from an SSA variable ID.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable identifier.
    ///
    /// # Returns
    ///
    /// A new [`SymbolicExpr::Variable`] referencing the given variable.
    #[must_use]
    pub const fn variable(var: SsaVarId) -> Self {
        Self::Variable(var)
    }

    /// Creates a named variable expression.
    ///
    /// Named variables are used for external inputs like "state" that aren't
    /// tied to a specific SSA variable ID.
    ///
    /// # Arguments
    ///
    /// * `name` - The variable name (e.g., "state").
    ///
    /// # Returns
    ///
    /// A new [`SymbolicExpr::NamedVar`] with the given name.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self::NamedVar(name.into())
    }

    /// Creates a unary operation expression.
    ///
    /// # Arguments
    ///
    /// * `op` - The unary operation (Neg or Not).
    /// * `operand` - The operand expression.
    ///
    /// # Returns
    ///
    /// A new [`SymbolicExpr::Unary`] applying the operation to the operand.
    #[must_use]
    pub fn unary(op: NumericOp, operand: Self) -> Self {
        Self::Unary {
            op,
            operand: Arc::new(operand),
        }
    }

    /// Creates a binary operation expression.
    ///
    /// # Arguments
    ///
    /// * `op` - The binary operation (Add, Sub, Mul, etc.).
    /// * `left` - The left operand expression.
    /// * `right` - The right operand expression.
    ///
    /// # Returns
    ///
    /// A new [`SymbolicExpr::Binary`] applying the operation to the operands.
    #[must_use]
    pub fn binary(op: NumericOp, left: Self, right: Self) -> Self {
        Self::Binary {
            op,
            left: Arc::new(left),
            right: Arc::new(right),
        }
    }

    /// Checks if this expression is a constant.
    ///
    /// # Returns
    ///
    /// `true` if this is a [`SymbolicExpr::Constant`].
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Checks if this expression is a variable.
    ///
    /// # Returns
    ///
    /// `true` for [`Self::Variable`], [`Self::NamedVar`] and [`Self::Instance`]
    /// leaves. These identities need not have a concrete value.
    #[must_use]
    pub const fn is_variable(&self) -> bool {
        matches!(
            self,
            Self::Variable(_) | Self::NamedVar(_) | Self::Instance(_)
        )
    }

    /// Returns the typed constant value if this is a constant expression.
    ///
    /// # Returns
    ///
    /// `Some(&ConstValue)` if this is a constant, `None` otherwise.
    #[must_use]
    pub const fn as_constant(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Constant(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the constant as i64 if this is a constant expression.
    ///
    /// This extracts the raw i64 value regardless of the underlying type.
    /// For type-preserving operations, use [`as_constant`](Self::as_constant) instead.
    ///
    /// # Returns
    ///
    /// `Some(i64)` if this is a constant with an integer value, `None` otherwise.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Constant(v) => v.as_i64(),
            _ => None,
        }
    }

    /// Returns the SSA variable ID if this is a variable expression.
    ///
    /// # Returns
    ///
    /// `Some(var_id)` if this is a [`SymbolicExpr::Variable`], `None` otherwise.
    /// Returns `None` for named inputs and opaque definition instances.
    #[must_use]
    pub const fn as_variable(&self) -> Option<SsaVarId> {
        match self {
            Self::Variable(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the opaque definition token when this expression is one instance.
    #[must_use]
    pub const fn as_instance(&self) -> Option<SymbolicInstance> {
        match self {
            Self::Instance(instance) => Some(*instance),
            _ => None,
        }
    }

    /// Calls `f` for each opaque instance leaf, preserving repeated occurrences.
    ///
    /// Instance provenance does not make it a current SSA binding. Use
    /// [`Self::for_each_variable`] for ordinary SSA-variable leaves.
    pub fn for_each_instance<F: FnMut(SymbolicInstance)>(&self, mut f: F) {
        let mut stack = vec![self];
        while let Some(expr) = stack.pop() {
            match expr {
                Self::Instance(instance) => f(*instance),
                Self::Constant(_) | Self::Variable(_) | Self::NamedVar(_) => {}
                Self::Unary { operand, .. } | Self::TypedUnary { operand, .. } => {
                    stack.push(operand)
                }
                Self::Binary { left, right, .. } | Self::TypedBinary { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
    }

    /// Calls `f` for each [`Self::Variable`] leaf.
    ///
    /// Opaque instances and named inputs are separate binding domains.
    ///
    /// Traverses the expression tree iteratively without allocating a set, so
    /// callers that only need to visit (or collect with their own dedup
    /// strategy) the variables avoid the cost of [`variables`](Self::variables).
    /// The same variable may be reported more than once.
    pub fn for_each_variable<F: FnMut(SsaVarId)>(&self, mut f: F) {
        let mut stack = vec![self];
        while let Some(expr) = stack.pop() {
            match expr {
                Self::Constant(_) | Self::NamedVar(_) | Self::Instance(_) => {}
                Self::Variable(v) => f(*v),
                Self::Unary { operand, .. } | Self::TypedUnary { operand, .. } => {
                    stack.push(operand)
                }
                Self::Binary { left, right, .. } | Self::TypedBinary { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
    }

    /// Collects all ordinary SSA-variable leaves in this expression.
    ///
    /// Traverses iteratively. Named inputs and opaque instance tokens are not
    /// current SSA bindings and do not appear in this set.
    ///
    /// # Returns
    ///
    /// The IDs appearing in [`Self::Variable`] leaves.
    #[must_use]
    pub fn variables(&self) -> FxHashSet<SsaVarId> {
        let mut vars = FxHashSet::default();
        let mut stack = vec![self];

        while let Some(expr) = stack.pop() {
            match expr {
                Self::Constant(_) | Self::NamedVar(_) | Self::Instance(_) => {}
                Self::Variable(v) => {
                    vars.insert(*v);
                }
                Self::Unary { operand, .. } | Self::TypedUnary { operand, .. } => {
                    stack.push(operand)
                }
                Self::Binary { left, right, .. } | Self::TypedBinary { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }

        vars
    }

    /// Collects all named variables referenced in this expression.
    ///
    /// Traverses the expression tree iteratively to avoid consuming call stack
    /// on adversarially deep symbolic expressions.
    ///
    /// # Returns
    ///
    /// A set of all variable names referenced in this expression.
    #[must_use]
    pub fn named_variables(&self) -> FxHashSet<String> {
        let mut vars = FxHashSet::default();
        let mut stack = vec![self];

        while let Some(expr) = stack.pop() {
            match expr {
                Self::Constant(_) | Self::Variable(_) | Self::Instance(_) => {}
                Self::NamedVar(name) => {
                    vars.insert(name.clone());
                }
                Self::Unary { operand, .. } | Self::TypedUnary { operand, .. } => {
                    stack.push(operand)
                }
                Self::Binary { left, right, .. } | Self::TypedBinary { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }

        vars
    }

    /// Evaluates this expression with ordinary SSA-variable bindings.
    ///
    /// Named inputs and opaque instances remain unbound. In particular, a
    /// binding for an instance's source id never determines that instance.
    /// Returns `None` for an unbound leaf or an operation the shared numeric
    /// kernel cannot evaluate, including invalid division.
    #[must_use]
    pub fn evaluate(
        &self,
        bindings: &FxHashMap<SsaVarId, ConstValue<T>>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        self.evaluate_with(ptr_size, |leaf| match leaf {
            Self::Variable(var) => bindings.get(var).cloned(),
            _ => None,
        })
    }

    /// Evaluates this expression with named external-input bindings.
    ///
    /// Ordinary SSA variables and opaque instances remain unbound. Returns
    /// `None` for a missing binding or an unadmitted numeric operation.
    #[must_use]
    pub fn evaluate_named(
        &self,
        bindings: &FxHashMap<&str, ConstValue<T>>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        self.evaluate_with(ptr_size, |leaf| match leaf {
            Self::NamedVar(name) => bindings.get(name.as_str()).cloned(),
            _ => None,
        })
    }

    /// Evaluates this expression with explicit bindings for every symbolic leaf.
    ///
    /// Calls `resolve` for [`Self::Variable`], [`Self::NamedVar`] and
    /// [`Self::Instance`] leaves. Constants retain their literal payloads.
    /// Distinct instances have independent bindings unless a separate proof
    /// relates them. The resolver supplies one consistent value per identity
    /// for this concrete scenario. This is evaluation under supplied inputs,
    /// rather than a proof over every possible input.
    ///
    /// The iterative walk shares one numeric kernel with the other expression
    /// evaluation APIs. `ptr_size` governs native integer arithmetic. Returns
    /// `None` if a leaf is unbound or an operation is not evaluable.
    ///
    /// # Examples
    ///
    /// ```
    /// use analyssa::{
    ///     PointerSize,
    ///     analysis::symbolic::SymbolicExpr,
    ///     ir::{evaluation::NumericOp, value::ConstValue, variable::SsaVarId},
    ///     testing::MockTarget,
    /// };
    /// let input = SsaVarId::from_index(0);
    /// let expression = SymbolicExpr::<MockTarget>::binary(
    ///     NumericOp::Add, SymbolicExpr::variable(input), SymbolicExpr::named("offset"),
    /// );
    /// let value = expression.evaluate_with(PointerSize::Bit64, |leaf| match leaf {
    ///     SymbolicExpr::Variable(var) if *var == input => Some(ConstValue::I32(6)),
    ///     SymbolicExpr::NamedVar(name) if name == "offset" => Some(ConstValue::I32(3)),
    ///     _ => None,
    /// });
    /// assert_eq!(value, Some(ConstValue::I32(9)));
    /// ```
    #[must_use]
    pub fn evaluate_with(
        &self,
        ptr_size: PointerSize,
        mut resolve: impl FnMut(&Self) -> Option<ConstValue<T>>,
    ) -> Option<ConstValue<T>> {
        self.evaluate_outcome_with(ptr_size, |leaf| {
            resolve(leaf).map_or(Evaluation::Unknown, Evaluation::Value)
        })
        .into_value()
    }

    /// Evaluates the tree while retaining faults, uncertainty and unsupported domains.
    ///
    /// Typed nodes use their captured machine; `ptr_size` applies only to legacy
    /// carrier nodes. A resolver's `Unknown` denotes a normal but unknown leaf
    /// value. A total typed operation preserves that normality; other unknown
    /// computed subtrees retain a possible failure obligation.
    /// Tree order does not establish SSA execution order: multiple failing or
    /// potentially failing children cannot prove which fault occurs first.
    pub fn evaluate_outcome_with(
        &self,
        ptr_size: PointerSize,
        resolve: impl FnMut(&Self) -> Evaluation<ConstValue<T>>,
    ) -> Evaluation<ConstValue<T>> {
        self.evaluate_detail_with(ptr_size, resolve).outcome
    }

    /// Evaluates values and normal-completion obligations in one iterative traversal.
    fn evaluate_detail_with(
        &self,
        ptr_size: PointerSize,
        mut resolve: impl FnMut(&Self) -> Evaluation<ConstValue<T>>,
    ) -> ExpressionOutcome<T> {
        let mut tasks = vec![EvalTask::Visit(self)];
        let mut values: Vec<ExpressionOutcome<T>> = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                EvalTask::Visit(Self::Constant(value)) => {
                    values.push(ExpressionOutcome::normal(Evaluation::Value(value.clone())));
                }
                EvalTask::Visit(
                    leaf @ (Self::Variable(_) | Self::NamedVar(_) | Self::Instance(_)),
                ) => {
                    values.push(ExpressionOutcome::normal(resolve(leaf)));
                }
                EvalTask::Visit(Self::Unary { op, operand }) => {
                    tasks.push(EvalTask::EvalUnary(*op));
                    tasks.push(EvalTask::Visit(operand));
                }
                EvalTask::Visit(Self::Binary { op, left, right }) => {
                    tasks.push(EvalTask::EvalBinary(*op));
                    tasks.push(EvalTask::Visit(right));
                    tasks.push(EvalTask::Visit(left));
                }
                EvalTask::Visit(Self::TypedUnary { op, operand }) => {
                    tasks.push(EvalTask::EvalTypedUnary(**op));
                    tasks.push(EvalTask::Visit(operand));
                }
                EvalTask::Visit(Self::TypedBinary { op, left, right }) => {
                    tasks.push(EvalTask::EvalTypedBinary(**op));
                    tasks.push(EvalTask::Visit(right));
                    tasks.push(EvalTask::Visit(left));
                }
                EvalTask::EvalUnary(op) => {
                    let Some(child) = values.pop() else {
                        return ExpressionOutcome::computed(Evaluation::Unsupported);
                    };
                    values.push(child.unary(false, |value| {
                        value
                            .and_then(|value| op.evaluate_unary(&value, ptr_size))
                            .map_or(Evaluation::Unknown, Evaluation::Value)
                    }));
                }
                EvalTask::EvalBinary(op) => {
                    let (Some(right), Some(left)) = (values.pop(), values.pop()) else {
                        return ExpressionOutcome::computed(Evaluation::Unsupported);
                    };
                    values.push(left.binary(right, false, |left, right| {
                        match (left, right) {
                            (Some(left), Some(right)) => op
                                .evaluate_binary(&left, &right, ptr_size)
                                .map_or(Evaluation::Unknown, Evaluation::Value),
                            _ => Evaluation::Unknown,
                        }
                    }));
                }
                EvalTask::EvalTypedUnary(op) => {
                    let Some(child) = values.pop() else {
                        return ExpressionOutcome::computed(Evaluation::Unsupported);
                    };
                    values.push(
                        child.unary(op.returns_normally_for_normal_inputs(), |value| {
                            op.evaluate(value, None)
                        }),
                    );
                }
                EvalTask::EvalTypedBinary(op) => {
                    let (Some(right), Some(left)) = (values.pop(), values.pop()) else {
                        return ExpressionOutcome::computed(Evaluation::Unsupported);
                    };
                    let count = match &right.outcome {
                        Evaluation::Value(count) => Some(count),
                        _ => None,
                    };
                    let normal = op.returns_normally_with_count(count);
                    values.push(left.binary(right, normal, |left, right| op.evaluate(left, right)));
                }
            }
        }
        values
            .pop()
            .unwrap_or_else(|| ExpressionOutcome::computed(Evaluation::Unsupported))
    }

    /// Substitutes an SSA variable with a replacement expression.
    ///
    /// Creates a new expression tree where all occurrences of `var` are
    /// replaced with `replacement`. Opaque instances with that source id stay
    /// distinct; use [`Self::substitute_instance`] to replace one execution.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable ID to replace.
    /// * `replacement` - The expression to substitute in place of the variable.
    ///
    /// # Returns
    ///
    /// A new expression with the substitution applied.
    #[must_use]
    pub fn substitute(&self, var: SsaVarId, replacement: &Self) -> Self {
        self.rebuilt(&|leaf| matches!(leaf, Self::Variable(v) if *v == var).then_some(replacement))
    }

    /// Replaces one opaque instance without changing other executions of its source.
    ///
    /// Ordinary SSA and named bindings are unaffected. The replacement is copied
    /// into each matching leaf through the iterative reconstruction path.
    #[must_use]
    pub fn substitute_instance(&self, instance: SymbolicInstance, replacement: &Self) -> Self {
        self.rebuilt(&|leaf| {
            matches!(leaf, Self::Instance(held) if *held == instance).then_some(replacement)
        })
    }

    /// Substitutes a named variable with a replacement expression.
    ///
    /// Creates a new expression tree where all occurrences of the named
    /// variable are replaced with `replacement`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the variable to replace (e.g., "state").
    /// * `replacement` - The expression to substitute.
    ///
    /// # Returns
    ///
    /// A new expression with the substitution applied.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{
    ///     analysis::symbolic::SymbolicExpr,
    ///     ir::evaluation::NumericOp,
    ///     testing::MockTarget,
    ///     PointerSize,
    /// };
    ///
    /// let expr = SymbolicExpr::<MockTarget>::binary(
    ///     NumericOp::Xor,
    ///     SymbolicExpr::named("state"),
    ///     SymbolicExpr::constant_i64(0x12345678),
    /// );
    /// let hundred = SymbolicExpr::constant_i64(100);
    ///
    /// // Substituting the only named variable folds the whole tree away.
    /// let result = expr.substitute_named("state", &hundred).simplify(PointerSize::Bit64);
    /// assert_eq!(result.as_i64(), Some(100 ^ 0x12345678));
    ///
    /// // Substituting an unrelated name leaves the expression symbolic.
    /// let untouched = expr.substitute_named("other", &hundred).simplify(PointerSize::Bit64);
    /// assert_eq!(untouched.as_i64(), None);
    /// ```
    #[must_use]
    pub fn substitute_named(&self, name: &str, replacement: &Self) -> Self {
        self.rebuilt(&|leaf| matches!(leaf, Self::NamedVar(n) if n == name).then_some(replacement))
    }

    /// Simplifies the expression by evaluating constant subexpressions.
    ///
    /// Folds constant operands through each typed node's captured machine and
    /// layouts; legacy nodes use carrier widths and the supplied pointer width.
    /// Algebraic identities cannot discard computations without proving their
    /// domain and normal completion. Failed constant operations remain in the tree.
    ///
    /// # Returns
    ///
    /// An expression with the same value and definedness for every binding at
    /// the supplied pointer width.
    #[must_use]
    pub fn simplify(&self, ptr_size: PointerSize) -> Self {
        if self.depth() > MAX_RECURSIVE_EXPR_DEPTH {
            return self.clone();
        }
        self.simplified(ptr_size).unwrap_or_else(|| self.clone())
    }

    /// Folds constant subexpressions bottom-up, or returns `None` when nothing folds.
    ///
    /// Unchanged subtrees stay shared; only the path to a folded node is
    /// rebuilt. Callers bound the depth first.
    fn simplified(&self, ptr_size: PointerSize) -> Option<Self> {
        match self {
            Self::Constant(_) | Self::Variable(_) | Self::NamedVar(_) | Self::Instance(_) => None,
            Self::TypedUnary { op, operand } => {
                let changed = operand.simplified(ptr_size);
                if let Self::Constant(value) = changed.as_ref().unwrap_or(operand)
                    && let Evaluation::Value(result) = op.evaluate(Some(value.clone()), None)
                {
                    return Some(Self::Constant(result));
                }
                changed.map(|operand| Self::TypedUnary {
                    op: op.clone(),
                    operand: Arc::new(operand),
                })
            }
            Self::TypedBinary { op, left, right } => {
                let changed = (left.simplified(ptr_size), right.simplified(ptr_size));
                if let (Self::Constant(l), Self::Constant(r)) = (
                    changed.0.as_ref().unwrap_or(left),
                    changed.1.as_ref().unwrap_or(right),
                ) && let Evaluation::Value(result) =
                    op.evaluate(Some(l.clone()), Some(r.clone()))
                {
                    return Some(Self::Constant(result));
                }
                Self::rebuilt_pair(changed, left, right, |left, right| Self::TypedBinary {
                    op: op.clone(),
                    left,
                    right,
                })
            }
            Self::Unary { op, operand } => {
                let changed = operand.simplified(ptr_size);
                if let Self::Constant(value) = changed.as_ref().unwrap_or(operand)
                    && let Some(result) = (*op).evaluate_unary(value, ptr_size)
                {
                    return Some(Self::Constant(result));
                }
                changed.map(|operand| Self::Unary {
                    op: *op,
                    operand: Arc::new(operand),
                })
            }
            Self::Binary { op, left, right } => {
                let changed = (left.simplified(ptr_size), right.simplified(ptr_size));
                if let (Self::Constant(l), Self::Constant(r)) = (
                    changed.0.as_ref().unwrap_or(left),
                    changed.1.as_ref().unwrap_or(right),
                ) && let Some(result) = (*op).evaluate_binary(l, r, ptr_size)
                {
                    return Some(Self::Constant(result));
                }
                Self::rebuilt_pair(changed, left, right, |left, right| Self::Binary {
                    op: *op,
                    left,
                    right,
                })
            }
        }
    }

    /// Rebuilds a binary node when either side changed, sharing the unchanged side.
    fn rebuilt_pair(
        changed: (Option<Self>, Option<Self>),
        left: &Arc<Self>,
        right: &Arc<Self>,
        build: impl FnOnce(Arc<Self>, Arc<Self>) -> Self,
    ) -> Option<Self> {
        match changed {
            (None, None) => None,
            (left_changed, right_changed) => Some(build(
                left_changed.map_or_else(|| Arc::clone(left), Arc::new),
                right_changed.map_or_else(|| Arc::clone(right), Arc::new),
            )),
        }
    }

    /// Writes the tree whose depth [`fmt::Display`] has already bounded.
    fn write_tree(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    where
        T::TypeRef: fmt::Display,
        T::MethodRef: fmt::Display,
        T::FieldRef: fmt::Display,
        T::SymbolRef: fmt::Display,
    {
        match self {
            Self::Constant(v) => write!(f, "{v}"),
            Self::Variable(var) => write!(f, "v{}", var.index()),
            Self::NamedVar(name) => write!(f, "{name}"),
            Self::Instance(instance) => write!(
                f,
                "v{}@{}",
                instance.source().index(),
                instance.generation()
            ),
            Self::TypedUnary { op, operand } => {
                write!(f, "({op:?} ")?;
                operand.write_tree(f)?;
                f.write_str(")")
            }
            Self::TypedBinary { op, left, right } => {
                f.write_str("(")?;
                left.write_tree(f)?;
                write!(f, " {op:?} ")?;
                right.write_tree(f)?;
                f.write_str(")")
            }
            Self::Unary { op, operand } => {
                write!(f, "({op}")?;
                operand.write_tree(f)?;
                f.write_str(")")
            }
            Self::Binary { op, left, right } => {
                f.write_str("(")?;
                left.write_tree(f)?;
                write!(f, " {op} ")?;
                right.write_tree(f)?;
                f.write_str(")")
            }
        }
    }

    /// Returns the depth of the expression tree.
    ///
    /// The depth is the length of the longest path from the root to a leaf.
    /// Constants and variables have depth 0.
    ///
    /// # Returns
    ///
    /// The maximum nesting depth of operations in this expression.
    #[must_use]
    pub fn depth(&self) -> usize {
        let mut max_depth = 0usize;
        let mut stack = vec![(self, 0usize)];

        while let Some((expr, depth)) = stack.pop() {
            max_depth = max_depth.max(depth);
            match expr {
                Self::Constant(_) | Self::Variable(_) | Self::NamedVar(_) | Self::Instance(_) => {}
                Self::Unary { operand, .. } | Self::TypedUnary { operand, .. } => {
                    stack.push((operand, depth.saturating_add(1)));
                }
                Self::Binary { left, right, .. } | Self::TypedBinary { left, right, .. } => {
                    let child_depth = depth.saturating_add(1);
                    stack.push((right, child_depth));
                    stack.push((left, child_depth));
                }
            }
        }

        max_depth
    }
}

impl<T: Target> fmt::Display for SymbolicExpr<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
    T::SymbolRef: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let depth = self.depth();
        if depth > MAX_RECURSIVE_EXPR_DEPTH {
            return write!(f, "<symbolic-expr depth {depth}>");
        }
        self.write_tree(f)
    }
}

impl<T: Target> From<ConstValue<T>> for SymbolicExpr<T> {
    fn from(value: ConstValue<T>) -> Self {
        Self::Constant(value)
    }
}

impl<T: Target> From<i32> for SymbolicExpr<T> {
    fn from(value: i32) -> Self {
        Self::Constant(ConstValue::I32(value))
    }
}

impl<T: Target> From<i64> for SymbolicExpr<T> {
    fn from(value: i64) -> Self {
        Self::Constant(ConstValue::I64(value))
    }
}

/// One evaluated subtree and whether an unknown result denotes a normal input.
struct ExpressionOutcome<T: Target> {
    outcome: Evaluation<ConstValue<T>>,
    normal_unknown: bool,
}

impl<T: Target> ExpressionOutcome<T> {
    fn normal(outcome: Evaluation<ConstValue<T>>) -> Self {
        Self {
            outcome,
            normal_unknown: true,
        }
    }

    fn computed(outcome: Evaluation<ConstValue<T>>) -> Self {
        Self {
            outcome,
            normal_unknown: false,
        }
    }

    fn with_normality(mut self, normal: bool) -> Self {
        self.normal_unknown = normal;
        self
    }

    fn unary(
        self,
        total: bool,
        evaluate: impl FnOnce(Option<ConstValue<T>>) -> Evaluation<ConstValue<T>>,
    ) -> Self {
        match self.outcome {
            Evaluation::Value(value) => Self::computed(evaluate(Some(value))).with_normality(total),
            Evaluation::Unknown if self.normal_unknown => {
                Self::computed(evaluate(None)).with_normality(total)
            }
            other => Self::computed(other),
        }
    }

    fn binary(
        self,
        right: Self,
        total: bool,
        evaluate: impl FnOnce(Option<ConstValue<T>>, Option<ConstValue<T>>) -> Evaluation<ConstValue<T>>,
    ) -> Self {
        let left_normal = matches!(&self.outcome, Evaluation::Value(_))
            || (matches!(&self.outcome, Evaluation::Unknown) && self.normal_unknown);
        let right_normal = matches!(&right.outcome, Evaluation::Value(_))
            || (matches!(&right.outcome, Evaluation::Unknown) && right.normal_unknown);
        if left_normal && right_normal {
            return Self::computed(evaluate(
                self.outcome.into_value(),
                right.outcome.into_value(),
            ))
            .with_normality(total);
        }
        let outcome = match (self.outcome, right.outcome) {
            (Evaluation::Unsupported, _) | (_, Evaluation::Unsupported) => Evaluation::Unsupported,
            (Evaluation::Trap(trap), _) if right_normal => Evaluation::Trap(trap),
            (_, Evaluation::Trap(trap)) if left_normal => Evaluation::Trap(trap),
            _ => Evaluation::Unknown,
        };
        Self::computed(outcome)
    }
}

#[cfg(test)]
mod tests {
    use std::{mem, num::NonZeroU64, sync::Arc};

    use rustc_hash::FxHashMap;

    use crate::{
        PointerSize,
        analysis::symbolic::{
            SymbolicInstance,
            expr::{MAX_EXPR_NODES, MAX_RECURSIVE_EXPR_DEPTH, SymbolicExpr},
        },
        ir::{evaluation::NumericOp, value::ConstValue, variable::SsaVarId},
        testing::MockTarget,
    };

    /// Unknown leaves cannot acquire the simplifier's preferred carrier width.
    #[test]
    fn simplification_preserves_bound_carriers_and_nested_widths() {
        for value in [ConstValue::I32(7), ConstValue::U64(7)] {
            let leaf = SymbolicExpr::<MockTarget>::named("x");
            let cancel = SymbolicExpr::binary(NumericOp::Xor, leaf.clone(), leaf.clone());
            let nested = SymbolicExpr::unary(
                NumericOp::Popcount,
                SymbolicExpr::unary(NumericOp::Not, cancel.clone()),
            );
            let zero = SymbolicExpr::binary(
                NumericOp::Mul,
                leaf,
                SymbolicExpr::constant(ConstValue::U64(0)),
            );
            let bindings = FxHashMap::from_iter([("x", value)]);
            for expr in [cancel, nested, zero] {
                assert_eq!(
                    expr.simplify(PointerSize::Bit64)
                        .evaluate_named(&bindings, PointerSize::Bit64),
                    expr.evaluate_named(&bindings, PointerSize::Bit64),
                );
            }
        }
    }

    /// Absorption and cancellation must retain an undefined child computation.
    #[test]
    fn simplification_retains_partial_subexpressions() {
        let partial = SymbolicExpr::<MockTarget>::binary(
            NumericOp::DivS,
            SymbolicExpr::constant_i32(1),
            SymbolicExpr::constant_i32(0),
        );
        let bindings = FxHashMap::default();
        for (op, right) in [
            (NumericOp::Mul, SymbolicExpr::constant_i32(0)),
            (NumericOp::And, SymbolicExpr::constant_i32(0)),
            (NumericOp::Or, SymbolicExpr::constant_i32(-1)),
            (NumericOp::Xor, partial.clone()),
            (NumericOp::Sub, partial.clone()),
        ] {
            let expr = SymbolicExpr::binary(op, partial.clone(), right);
            assert_eq!(expr.evaluate(&bindings, PointerSize::Bit64), None);
            assert_eq!(
                expr.simplify(PointerSize::Bit64)
                    .evaluate(&bindings, PointerSize::Bit64),
                None,
            );
        }
    }

    fn deep_named_add_chain(depth: usize) -> SymbolicExpr<MockTarget> {
        let mut expr = SymbolicExpr::named("state");
        for _ in 0..depth {
            expr = SymbolicExpr::binary(NumericOp::Add, expr, SymbolicExpr::constant_i32(0));
        }
        expr
    }

    /// Depth, variable collection, and `Display` all handle a 1024-deep tree
    /// without recursing on the call stack (and `Display` prints a placeholder).
    #[test]
    fn deep_expression_queries_do_not_recurse_on_call_stack() {
        let expr = deep_named_add_chain(1_024);

        assert_eq!(expr.depth(), 1_024);
        assert!(expr.named_variables().contains("state"));
        assert!(expr.variables().is_empty());
        assert_eq!(format!("{expr}"), "<symbolic-expr depth 1024>");
    }

    /// Substituting into and then evaluating a 1024-deep tree stays iterative
    /// and yields the folded constant.
    #[test]
    fn deep_expression_substitution_and_evaluation_are_iterative() {
        let expr = deep_named_add_chain(1_024);
        let substituted = expr.substitute_named("state", &SymbolicExpr::constant_i32(7));

        assert_eq!(
            substituted.evaluate(
                &FxHashMap::<SsaVarId, ConstValue<MockTarget>>::default(),
                PointerSize::Bit64
            ),
            Some(ConstValue::I32(7))
        );
    }

    /// **A copy of a tree deeper than the stack does not recurse.** A chain a
    /// million operations deep clones to an equal chain sharing its operands,
    /// where a copy that descended once per level would overflow the test
    /// thread's stack. The tree is leaked: dropping it descends through its
    /// operands.
    #[test]
    fn a_tree_deeper_than_the_stack_clones() {
        let expr = deep_named_add_chain(1_000_000);
        let copy = expr.clone();
        assert_eq!(copy.depth(), 1_000_000);
        assert_eq!(copy.node_count(), expr.node_count());
        let (SymbolicExpr::Binary { left: copied, .. }, SymbolicExpr::Binary { left, .. }) =
            (&copy, &expr)
        else {
            panic!("an add chain");
        };
        assert!(Arc::ptr_eq(copied, left));
        drop(copy);
        mem::forget(expr);
    }

    /// A copy shares its operands, and a simplification that folds nothing
    /// keeps every subtree shared rather than rebuilding it.
    #[test]
    fn copies_and_unfolded_simplification_share_operands() {
        let leaf: SymbolicExpr<MockTarget> = SymbolicExpr::named("a");
        let sum = SymbolicExpr::binary(NumericOp::Add, leaf.clone(), leaf);
        let outer = SymbolicExpr::binary(NumericOp::Mul, sum, SymbolicExpr::constant_i32(3));
        let operands = |expr: &SymbolicExpr<MockTarget>| match expr {
            SymbolicExpr::Binary { left, right, .. } => (Arc::clone(left), Arc::clone(right)),
            other => panic!("{other:?}"),
        };
        let (left, right) = operands(&outer);
        let (copied_left, copied_right) = operands(&outer.clone());
        assert!(Arc::ptr_eq(&left, &copied_left) && Arc::ptr_eq(&right, &copied_right));
        let (simplified_left, simplified_right) = operands(&outer.simplify(PointerSize::Bit64));
        assert!(Arc::ptr_eq(&left, &simplified_left));
        assert!(Arc::ptr_eq(&right, &simplified_right));
        // A fold rebuilds only its path: the constant side folds, the other stays shared.
        let folding = SymbolicExpr::binary(
            NumericOp::Add,
            SymbolicExpr::Variable(SsaVarId::from_index(0)),
            SymbolicExpr::binary(
                NumericOp::Add,
                SymbolicExpr::constant_i32(1),
                SymbolicExpr::constant_i32(2),
            ),
        );
        let (variable, _) = operands(&folding);
        let (kept, folded) = operands(&folding.simplify(PointerSize::Bit64));
        assert!(Arc::ptr_eq(&variable, &kept));
        assert_eq!(folded.as_i64(), Some(3));
    }

    /// `simplify` on a 1024-deep tree returns via the depth guard instead of
    /// descending recursively, preserving the tree's depth.
    #[test]
    fn deep_expression_simplify_returns_without_recursive_descent() {
        let expr = deep_named_add_chain(1_024);
        let simplified = expr.simplify(PointerSize::Bit64);

        assert_eq!(simplified.depth(), 1_024);
    }

    /// Depth is not a bound on size. Each `x + x` step clones both operand trees,
    /// so the node count doubles while the depth grows by one - 30 steps is a
    /// billion nodes at depth 30, far below the depth cap of 512. This is why the
    /// evaluators check `node_count` as well.
    #[test]
    fn self_doubling_grows_node_count_exponentially_at_shallow_depth() {
        let mut expr: SymbolicExpr<MockTarget> = SymbolicExpr::named("x");
        for step in 1..=12u32 {
            expr = SymbolicExpr::binary(NumericOp::Add, expr.clone(), expr);
            assert_eq!(expr.depth(), step as usize, "one step adds one level");
        }

        // 2^13 - 1 nodes at depth 12: the depth cap of 512 is nowhere near.
        assert_eq!(expr.node_count(), (1usize << 13) - 1);
        assert!(
            expr.depth() < MAX_RECURSIVE_EXPR_DEPTH,
            "the depth guard cannot see this"
        );
        assert!(
            SymbolicExpr::combining_reaches_limits(&[&expr, &SymbolicExpr::named("y")]),
            "but the size guard must, or memory is the only limit"
        );
    }

    /// `node_count` counts leaves, unary, and binary nodes exactly (1, 2, 3).
    #[test]
    fn node_count_counts_every_node() {
        let leaf: SymbolicExpr<MockTarget> = SymbolicExpr::named("a");
        assert_eq!(leaf.node_count(), 1);

        let unary = SymbolicExpr::unary(NumericOp::Neg, leaf.clone());
        assert_eq!(unary.node_count(), 2);

        let binary = SymbolicExpr::binary(NumericOp::Add, leaf.clone(), leaf);
        assert_eq!(binary.node_count(), 3);
        assert!(!SymbolicExpr::combining_reaches_limits(&[&binary, &binary]));
    }

    /// The budgets hold at their exact boundaries: the node budget over all
    /// operands together, the depth budget per operand, leaves included.
    #[test]
    fn combination_limits_hold_at_their_boundaries() {
        let leaf: SymbolicExpr<MockTarget> = SymbolicExpr::named("a");
        assert!(!SymbolicExpr::combining_reaches_limits(&[&leaf, &leaf]));
        let mut deep = leaf.clone();
        for _ in 0..MAX_RECURSIVE_EXPR_DEPTH - 1 {
            deep = SymbolicExpr::unary(NumericOp::Neg, deep);
        }
        assert_eq!(deep.depth(), MAX_RECURSIVE_EXPR_DEPTH - 1);
        assert!(!SymbolicExpr::combining_reaches_limits(&[&deep, &leaf]));
        let deeper = SymbolicExpr::unary(NumericOp::Neg, deep.clone());
        assert!(SymbolicExpr::combining_reaches_limits(&[&leaf, &deeper]));
        // Eleven doublings of a leaf hold one node fewer than the budget.
        let mut wide = leaf.clone();
        for _ in 0..11 {
            wide = SymbolicExpr::binary(NumericOp::Add, wide.clone(), wide);
        }
        assert_eq!(wide.node_count(), MAX_EXPR_NODES - 1);
        assert!(!SymbolicExpr::combining_reaches_limits(&[&wide]));
        assert!(SymbolicExpr::combining_reaches_limits(&[&wide, &leaf]));
    }
    /// Instance bindings stay independent from their source and from other executions.
    #[test]
    fn opaque_instances_keep_distinct_binding_and_substitution_domains() {
        let source = SsaVarId::from_index(0);
        let first = SymbolicInstance::new(source, NonZeroU64::new(1).unwrap());
        let second = SymbolicInstance::new(source, NonZeroU64::new(2).unwrap());
        let leaf = SymbolicExpr::<MockTarget>::Instance(first);
        assert_eq!(leaf.clone(), leaf);
        assert!(leaf.is_variable());
        assert_eq!(leaf.as_variable(), None);
        assert_eq!(leaf.as_instance(), Some(first));
        assert_eq!(first.source(), source);
        assert_eq!(first.generation(), 1);
        assert_eq!(leaf.depth(), 0);
        assert_eq!(leaf.node_count(), 1);
        assert_eq!(format!("{leaf}"), "v0@1");
        let expr =
            SymbolicExpr::binary(NumericOp::Sub, leaf.clone(), SymbolicExpr::Instance(second));
        let mut instances = Vec::new();
        expr.for_each_instance(|instance| instances.push(instance));
        assert_eq!(instances, [first, second]);
        assert!(expr.variables().is_empty());
        assert!(expr.named_variables().is_empty());
        let ssa_bindings = FxHashMap::from_iter([(source, ConstValue::I32(99))]);
        assert!(expr.evaluate(&ssa_bindings, PointerSize::Bit64).is_none());
        assert!(
            expr.evaluate_named(&FxHashMap::default(), PointerSize::Bit64)
                .is_none()
        );
        assert_eq!(
            expr.substitute(source, &SymbolicExpr::constant_i32(99)),
            expr
        );
        assert_eq!(
            expr.substitute_named("unused", &SymbolicExpr::constant_i32(99)),
            expr
        );
        assert!(!expr.simplify(PointerSize::Bit64).is_constant());
        let known = |leaf: &SymbolicExpr<MockTarget>| match leaf.as_instance() {
            Some(instance) if instance == first => Some(ConstValue::I32(6)),
            Some(instance) if instance == second => Some(ConstValue::I32(9)),
            _ => None,
        };
        assert_eq!(
            expr.evaluate_with(PointerSize::Bit64, known),
            Some(ConstValue::I32(-3))
        );
        let same = SymbolicExpr::binary(NumericOp::Sub, leaf.clone(), leaf);
        assert!(!same.simplify(PointerSize::Bit64).is_constant());
        assert_eq!(
            same.evaluate_with(PointerSize::Bit64, known),
            Some(ConstValue::I32(0))
        );
        let substituted = expr
            .substitute_instance(first, &SymbolicExpr::constant_i32(6))
            .substitute_instance(second, &SymbolicExpr::constant_i32(9));
        assert_eq!(
            substituted.evaluate(&FxHashMap::default(), PointerSize::Bit64),
            Some(ConstValue::I32(-3))
        );
        assert_eq!(
            same.substitute_instance(first, &SymbolicExpr::constant_i32(6))
                .simplify(PointerSize::Bit64)
                .as_i64(),
            Some(0)
        );
        assert!(mem::size_of::<SymbolicInstance>() <= 16);
        assert!(mem::size_of::<SymbolicExpr<MockTarget>>() <= 32);
    }

    /// A single evaluator resolves all three binding domains and never overrides literals.
    #[test]
    fn opaque_instances_share_mixed_leaf_evaluation() {
        let source = SsaVarId::from_index(0);
        let instance = SymbolicInstance::new(source, NonZeroU64::new(1).unwrap());
        let expr = SymbolicExpr::<MockTarget>::binary(
            NumericOp::Add,
            SymbolicExpr::Instance(instance),
            SymbolicExpr::binary(
                NumericOp::Add,
                SymbolicExpr::variable(source),
                SymbolicExpr::named("bias"),
            ),
        );
        assert_eq!(expr.variables(), [source].into_iter().collect());
        assert_eq!(
            expr.named_variables(),
            ["bias".to_string()].into_iter().collect()
        );
        let mut visits = 0;
        let value = expr.evaluate_with(PointerSize::Bit64, |leaf| {
            visits += 1;
            match leaf {
                SymbolicExpr::Instance(held) if *held == instance => Some(ConstValue::I32(10)),
                SymbolicExpr::Variable(var) if *var == source => Some(ConstValue::I32(2)),
                SymbolicExpr::NamedVar(name) if name == "bias" => Some(ConstValue::I32(3)),
                _ => None,
            }
        });
        assert_eq!(value, Some(ConstValue::I32(15)));
        assert_eq!(visits, 3);
        assert!(expr.evaluate_with(PointerSize::Bit64, |_| None).is_none());
        let divide = SymbolicExpr::<MockTarget>::binary(
            NumericOp::DivS,
            SymbolicExpr::constant_i32(6),
            SymbolicExpr::constant_i32(0),
        );
        assert!(
            divide
                .evaluate_with(PointerSize::Bit64, |_| panic!(
                    "literals are not symbolic bindings"
                ))
                .is_none()
        );
    }

    /// Deep instance trees retain their tokens through iterative walks, copies, evaluation and substitution.
    #[test]
    fn deep_instance_queries_and_rebuilds_stay_iterative() {
        let instance = SymbolicInstance::new(SsaVarId::from_index(0), NonZeroU64::new(1).unwrap());
        let mut expr = SymbolicExpr::<MockTarget>::Instance(instance);
        for _ in 0..1_024 {
            expr = SymbolicExpr::binary(NumericOp::Add, expr, SymbolicExpr::constant_i32(0));
        }
        let copy = expr.clone();
        assert_eq!(copy.depth(), 1_024);
        assert_eq!(copy.node_count(), 2_049);
        assert!(copy.variables().is_empty());
        assert!(copy.named_variables().is_empty());
        let mut seen = Vec::new();
        copy.for_each_instance(|held| seen.push(held));
        assert_eq!(seen, [instance]);
        assert_eq!(
            copy.evaluate_with(PointerSize::Bit64, |leaf| {
                (leaf.as_instance() == Some(instance)).then_some(ConstValue::I32(7))
            }),
            Some(ConstValue::I32(7))
        );
        let replaced = copy.substitute_instance(instance, &SymbolicExpr::constant_i32(7));
        assert_eq!(
            replaced.evaluate(&FxHashMap::default(), PointerSize::Bit64),
            Some(ConstValue::I32(7))
        );
        assert_eq!(format!("{expr}"), "<symbolic-expr depth 1024>");
    }
}
