//! Hybrid concrete/symbolic SSA evaluator for computing values from operations.
//!
//! This module provides [`SsaEvaluator`], an interpreter for SSA operations that
//! can evaluate arithmetic and logical operations given known input values. It
//! serves as the core computation engine for control flow unflattening, opaque
//! predicate detection, and symbolic execution.
//!
//! # Value Representation
//!
//! Values use a three-tier representation as [`SymbolicExpr`]:
//!
//! | Category | Representation | Meaning |
//! |----------|---------------|---------|
//! | Concrete | `SymbolicExpr::Constant(v)` | Known integer constant |
//! | Symbolic | Other `SymbolicExpr` variants | Expression tree depending on unknown inputs |
//! | Unknown | `None` (absent from value map) | Cannot be determined statically |
//!
//! # Algorithm
//!
//! ## Instruction Evaluation
//!
//! Each SSA operation is evaluated by:
//! 1. Looking up operand values in the tracked values map
//! 2. If all operands are concrete, computing a concrete result via `ConstValue` ops
//! 3. If any operand is symbolic, building a `SymbolicExpr` tree preserving the operation
//! 4. Applying simplification (constant folding, algebraic identities) to the result
//! 5. Storing the result for use by subsequent operations
//!
//! ## Path-Aware Evaluation
//!
//! Phi nodes merge values from different predecessors. For path-aware evaluation,
//! callers set the predecessor block via `set_predecessor()` before evaluating a
//! block. The evaluator then selects the phi operand from that predecessor,
//! enabling precise tracking along a specific execution path.
//!
//! Phi evaluation uses two-phase semantics ("simultaneous execution"):
//! 1. Read all source values into a temporary buffer
//! 2. Write all results from the buffer
//!
//! This ensures correct swap semantics (e.g., `v1 = phi(v2); v2 = phi(v1)`).
//!
//! ## Loop Evaluation
//!
//! Two strategies are available:
//! - **Fixed-point iteration**: `evaluate_loop_to_fixpoint` iterates until values
//!   stabilize, then widens unstable values to Unknown. Useful when loop bounds
//!   are unknown.
//! - **Bounded iteration**: `evaluate_loop_iterations` runs a known number of
//!   iterations. Useful when the loop count is a known constant.
//!
//! ## Trace Execution
//!
//! The `execute` method provides full CFG traversal: starting from a block, it
//! evaluates each block, follows control flow decisions based on computed values,
//! and records the execution trace. This is the primary tool for CFF deobfuscation
//! where the goal is to record all state transitions through the dispatcher.
//!
//! # Constraint Tracking
//!
//! When taking a branch, `apply_branch_constraint()` derives facts about the
//! condition variable (e.g., `x == 5` on the true branch of `ceq(x, 5)`).
//! These constraints can be used to detect infeasible paths and dead code.
//!
//! # Local/Argument State Tracking
//!
//! The evaluator maintains `local_state` and `arg_state` maps that mirror CIL
//! local variable and argument slots. When a variable with `VariableOrigin::Local(N)`
//! receives a value, the corresponding local slot is updated. `LoadLocal`/`LoadArg`
//! instructions read from these maps, enabling tracking across SSA variable versions.
//! A separate sparse map retains partial integer contents for both kinds of slot.
//! It stores the current facts themselves, so replacing an SSA source cannot
//! change a value already copied into a slot. Expression and bit writes share
//! invalidation and speculative undo. A value-only reset retains both mirrors.
//!
//! Field tracking captures expressions and partial facts at store time. Alias
//! writes and unanswered calls invalidate both domains. Volatile accesses,
//! atomics and fences discard field observations. Field qualifiers outside the
//! default address space at current permission are not modeled; locality and
//! alignment hints preserve admitted values. Reassigning an object
//! identity discards its cells; changing an expression dependency discards that
//! expression while retaining independent masks. A sparse dependency index
//! avoids field scans on unrelated SSA assignments. Checkpoints include tracking
//! policy and field state, so mode changes roll back with their observations.
//!
//! A slot whose address the function takes anywhere is not tracked at all: a
//! load from it is unknown. A pointer to the slot lets any call or indirect
//! store write it (`Monitor.Enter(obj, ref lockTaken)`), and once the address
//! is computed it can be used from anywhere, so no point in the function is
//! past the last write the evaluator can see. Address-of operations also
//! invalidate the tracked state for their slot, for the evaluation that got
//! there first.
//!
//! # Numeric Semantics
//!
//! Concrete evaluation and partial integer facts share the typed [`Machine`]
//! kernel. Integer operations use their declared operand and destination widths,
//! signedness and overflow policy; native integers use the configured pointer
//! width. The evaluator retains the kernel's constant representation. Floating
//! results follow the configured format, rounding and exceptional-value rules.
//! An operation outside those established contracts yields an unknown value.
//!
//! # Complexity
//!
//! Instruction evaluation: O(d) where d is the expression tree depth
//! (bounded by simplification). Block evaluation: O(n * d) for n instructions.
//! Loop fixed-point: O(i * b * n * d) where i is iterations (bounded by config).
//!
//! # Usage
//!
//! ```rust
//! use analyssa::{
//!     analysis::SsaEvaluator,
//!     ir::{value::ConstValue, SsaVarId},
//!     testing,
//! };
//!
//! // Diamond CFG: block 0 computes `v0 = true` and branches on it.
//! let ssa = testing::diamond_phi_fixture();
//! let mut eval = SsaEvaluator::new(&ssa, testing::MACHINE);
//!
//! // Evaluate a block's instructions
//! eval.evaluate_block(0);
//!
//! // Get computed result
//! let condition = SsaVarId::from_index(0);
//! let expr = eval.get(condition).unwrap();
//! assert!(expr.is_constant());
//! assert_eq!(expr.as_constant(), Some(&ConstValue::True));
//!
//! // Or use the convenience method for concrete values
//! assert_eq!(eval.get_concrete(condition), Some(&ConstValue::True));
//!
//! // Set known concrete values, or mark a variable as symbolic
//! let ssa = testing::loop_counter_fixture();
//! let mut eval = SsaEvaluator::new(&ssa, testing::MACHINE);
//! let counter = SsaVarId::from_index(1);
//! let limit = SsaVarId::from_index(4);
//!
//! eval.set_symbolic(counter, "arg0");
//! let expr = eval.get(counter).unwrap();
//! assert!(!expr.is_constant());
//! assert_eq!(expr.to_string(), "arg0");
//!
//! eval.set_concrete(limit, ConstValue::I32(42));
//! assert_eq!(eval.get_concrete(limit), Some(&ConstValue::I32(42)));
//!
//! // Variables with no assigned or computed value cannot be determined.
//! assert!(eval.get(SsaVarId::from_index(5)).is_none());
//! ```

use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
    iter,
    num::NonZeroU64,
};

use crate::{
    PointerSize,
    analysis::{
        cells::MemoryCells,
        constraints::Constraint,
        consts::{
            Destination, Machine, OperandLayout, TypeQueries, comparison_stable,
            scalar::{OperationLayout, ScalarOperation, nonreturn_terminal},
        },
        memory::MemoryLocation,
        symbolic::{SymbolicExpr, SymbolicInstance},
    },
    ir::{
        evaluation::Evaluation,
        function::{AddressTakenSlots, SsaFunction},
        integer::IntegerBits,
        ops::{
            AccessPermission, CallEffects, CallLocation, CmpKind, FloatFunctionKind, SsaEffectKind,
            SsaOp,
        },
        value::ConstValue,
        variable::{SsaVarId, VariableOrigin},
    },
    target::Target,
};

/// Result of evaluating a control flow decision.
///
/// This represents the outcome of analyzing a terminator instruction to
/// determine the next block to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlow {
    /// Continue to the specified block.
    Continue(usize),
    /// Terminal instruction - execution ends here (return, throw, etc.).
    Terminal,
    /// Cannot determine the next block - condition is unknown or symbolic.
    Unknown,
}

impl ControlFlow {
    /// Returns the target block if this is a `Continue` result.
    #[must_use]
    pub fn target(&self) -> Option<usize> {
        match self {
            Self::Continue(block) => Some(*block),
            _ => None,
        }
    }

    /// Returns `true` if this is a terminal result.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal)
    }

    /// Returns `true` if the control flow cannot be determined.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// Configuration for SSA evaluators.
///
/// This struct controls the behavior of [`SsaEvaluator`], allowing it to be
/// configured for different use cases like general evaluation, path-aware
/// analysis, or CFF deobfuscation.
#[derive(Debug, Clone, Default)]
pub struct EvaluatorConfig {
    /// Track the execution path (sequence of visited blocks).
    pub track_path: bool,
    /// Track memory state (field loads/stores).
    pub track_memory: bool,
    /// Require predecessor for phi evaluation (strict path-aware mode).
    pub strict_phi: bool,
}

impl EvaluatorConfig {
    /// Creates a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a configuration for path-aware analysis.
    ///
    /// Enables path tracking, memory tracking, and strict phi evaluation.
    #[must_use]
    pub fn path_aware() -> Self {
        Self {
            track_path: true,
            track_memory: true,
            strict_phi: true,
        }
    }

    /// Creates a configuration with memory tracking only.
    #[must_use]
    pub fn with_memory() -> Self {
        Self {
            track_path: false,
            track_memory: true,
            strict_phi: false,
        }
    }

    /// Enables path tracking.
    #[must_use]
    pub fn with_path_tracking(mut self) -> Self {
        self.track_path = true;
        self
    }

    /// Enables memory state tracking.
    #[must_use]
    pub fn with_memory_tracking(mut self) -> Self {
        self.track_memory = true;
        self
    }

    /// Enables strict phi evaluation (requires predecessor).
    #[must_use]
    pub fn with_strict_phi(mut self) -> Self {
        self.strict_phi = true;
        self
    }
}

/// Records the execution trace of SSA evaluation.
///
/// This struct tracks the sequence of blocks visited during SSA evaluation,
/// along with optional state values at each step. This is essential for
/// CFF (Control Flow Flattening) deobfuscation, where we need to record
/// the dispatcher state transitions to reconstruct the original control flow.
#[derive(Debug, Clone)]
pub struct ExecutionTrace<T: Target> {
    /// Sequence of block indices visited.
    blocks: Vec<usize>,
    /// Optional state values captured at each block (for state machines).
    states: Vec<Option<ConstValue<T>>>,
    /// Whether execution completed normally (reached terminal).
    completed: bool,
    /// Outcome of normal trace completion, distinct from a truncated trace.
    outcome: Evaluation<()>,
    /// Maximum blocks to trace before stopping (prevents infinite loops).
    limit: usize,
}

impl<T: Target> ExecutionTrace<T> {
    /// Creates a new execution trace with the given block limit.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            blocks: Vec::new(),
            states: Vec::new(),
            completed: false,
            outcome: Evaluation::Unknown,
            limit,
        }
    }

    /// Returns normal completion, an established arithmetic fault, or analysis uncertainty.
    #[must_use]
    pub fn outcome(&self) -> &Evaluation<()> {
        &self.outcome
    }

    /// Returns the blocks visited during execution.
    #[must_use]
    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    /// Returns the state values captured during execution.
    #[must_use]
    pub fn states(&self) -> &[Option<ConstValue<T>>] {
        &self.states
    }

    /// Returns `true` if execution completed (reached a terminal instruction).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Returns the number of blocks visited.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns `true` if no blocks were visited.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Returns the last visited block, if any.
    #[must_use]
    pub fn last_block(&self) -> Option<usize> {
        self.blocks.last().copied()
    }

    /// Returns `true` if the trace reached the block limit.
    #[must_use]
    pub fn hit_limit(&self) -> bool {
        self.blocks.len() >= self.limit
    }

    /// Records a block visit.
    fn record_block(&mut self, block_idx: usize, state: Option<ConstValue<T>>) {
        self.blocks.push(block_idx);
        self.states.push(state);
    }

    /// Marks execution as complete.
    fn mark_complete(&mut self) {
        self.completed = true;
        self.outcome = Evaluation::Value(());
    }
}

/// An owned field value at store time, independent of later SSA assignments.
#[derive(Debug, Clone)]
struct StoredField<T: Target> {
    /// Source identity used only to break deterministic alias-lookup ties.
    source: SsaVarId,
    /// Captured expression, forgotten when a referenced SSA identity changes.
    expression: Option<SymbolicExpr<T>>,
    /// Inline partial facts with the same store-time lifetime.
    bits: Option<IntegerBits>,
}

/// Optional tracked domains copied by an evaluator checkpoint.
#[derive(Debug, Clone)]
struct TrackedState<T: Target> {
    /// Store-time field payloads and their object identities.
    memory: MemoryCells<T, StoredField<T>>,
    /// A conservative index of SSA identities referenced by those payloads.
    dependencies: BTreeSet<SsaVarId>,
    /// Visited blocks at checkpoint time.
    path: Vec<usize>,
}

/// A position in an evaluator's undo journal.
///
/// Produced by [`SsaEvaluator::checkpoint`] and consumed by
/// [`SsaEvaluator::rollback`]. Marks are only meaningful for the evaluator that
/// issued them.
#[derive(Debug, Clone)]
pub struct EvaluatorMark<T: Target> {
    /// Ordered execution knowledge and whether any normal continuation remains.
    execution: Evaluation<()>,
    halted: bool,
    /// Journal length when the mark was taken.
    len: usize,
    /// Interpretation and tracking policy at checkpoint time.
    config: EvaluatorConfig,
    /// Memory state and execution path at mark time.
    ///
    /// The snapshot is present only when memory or path tracking is enabled;
    /// the default configuration stores neither snapshot. Tracked states are
    /// cloned at checkpoint time rather than recorded in the mutation journal.
    /// Their copy cost depends on the tracked memory and path lengths.
    tracked: Option<Box<TrackedState<T>>>,
}

/// A single reversible mutation of evaluator state.
#[derive(Debug, Clone)]
enum Undo<T: Target> {
    /// `values` entry as it was before the change.
    Value(SsaVarId, Option<SymbolicExpr<T>>),
    /// Current unknown identity before a definition, invalidation or value reset.
    ///
    /// Outer `None` means no map entry; inner `None` records identity refusal.
    Opaque(SsaVarId, Option<Option<SymbolicInstance>>),
    /// Whether unresolved inputs may use ordinary SSA bindings.
    InputSymbols(bool),
    /// Partial integer entry before a value assignment or speculative transfer.
    Bits(SsaVarId, Option<IntegerBits>),
    /// Current partial local/argument contents before a mirrored write or invalidation.
    SlotBits(VariableOrigin, Option<IntegerBits>),
    /// `local_state` entry as it was before the change.
    Local(u16, Option<SymbolicExpr<T>>),
    /// `arg_state` entry as it was before the change.
    Arg(u16, Option<SymbolicExpr<T>>),
    /// `constraints` entry as it was before the change.
    Constraints(SsaVarId, Option<Vec<Constraint<T>>>),
    /// `predecessor` as it was before the change.
    Predecessor(Option<usize>),
}

/// SSA evaluator with hybrid concrete/symbolic value tracking.
///
/// This evaluator interprets SSA operations to compute values without needing
/// full CIL emulation. Values are represented as [`SymbolicExpr`]:
///
/// - **Concrete**: `SymbolicExpr::Constant(v)` - Known integer values
/// - **Symbolic**: Other `SymbolicExpr` variants - Expressions depending on unknown inputs
/// - **Unknown**: `None` (not in the values map) - Values that cannot be determined
///
/// # Value Representation
///
/// Concrete values use [`ConstValue`] with their typed width, signedness and
/// vector geometry. Scalar integer operations also retain known bit positions
/// when the complete value is unknown. Those facts share the machine's typed
/// transfer with static execution and constant evaluation; they never require
/// constructing a symbolic expression for the remaining runtime bits.
///
/// Assignments and invalidations update both domains. Checkpoints and rollback
/// preserve their common lifetime, including facts with no tracked expression.
/// Selected phis and non-address-taken local/argument slots retain these facts.
/// Phi sources in both domains are read before any result is assigned.
/// Opaque expressions use immutable definition-instance tokens, so a repeated
/// definition cannot make an older value equal its new result. Tokens are scoped
/// to this evaluation context; clones retain shared-prefix identities and fork
/// the continuation context.
#[derive(Debug, Clone)]
pub struct SsaEvaluator<'a, T: Target> {
    /// Whether the evaluated prefix returns normally; values after uncertainty are conditional.
    execution: Evaluation<()>,
    /// A proven fault stops definitions even when its exact first exception is ambiguous.
    halted: bool,
    /// Reference to the SSA function being evaluated.
    ssa: &'a SsaFunction<T>,
    /// Tracked values for variables. Missing entries represent unknown values.
    values: BTreeMap<SsaVarId, SymbolicExpr<T>>,
    /// Current opaque identities for executed unknown definitions; `None` refuses identity.
    opaque_values: BTreeMap<SsaVarId, Option<SymbolicInstance>>,
    /// Generation high-water mark, never rewound by speculative rollback.
    next_instance: u64,
    /// Whether unresolved inputs may still use original SSA-variable bindings.
    ///
    /// A value reset disables this fallback so repeated reuse cannot alias old inputs.
    input_symbols: bool,
    /// Partial scalar facts independent of symbolic expression availability.
    integer_bits: BTreeMap<SsaVarId, IntegerBits>,
    /// Sparse partial facts for current local and argument contents.
    ///
    /// Stored values remain independent of reassigned SSA sources. Concrete
    /// slots use the expression mirrors; only partial slots allocate entries.
    slot_bits: BTreeMap<VariableOrigin, IntegerBits>,
    /// Current predecessor block for path-aware phi evaluation.
    /// When set, phi nodes will select the operand from this predecessor.
    predecessor: Option<usize>,
    /// Constraints on variable values derived from branch conditions.
    /// Used to detect dead code and propagate information after branches.
    constraints: BTreeMap<SsaVarId, Vec<Constraint<T>>>,
    /// Evaluator configuration controlling behavior.
    config: EvaluatorConfig,
    /// Execution path (sequence of visited blocks). Only populated if `config.track_path`.
    path: Vec<usize>,
    /// Memory state tracking for fields. Only used if `config.track_memory`.
    memory: MemoryCells<T, StoredField<T>>,
    /// Possible object and expression dependencies of retained field snapshots.
    ///
    /// This sparse superset avoids scanning fields for unrelated SSA writes.
    /// Entries left by alias invalidation are harmless and bounded by SSA ids.
    field_dependencies: BTreeSet<SsaVarId>,
    /// The machine the evaluated function computes on.
    machine: Machine,
    /// Expression component of current CIL locals, indexed by local index.
    /// Updated whenever a variable with `Local(N)` origin receives a value,
    /// and read by `LoadLocal` instructions.
    local_state: BTreeMap<u16, SymbolicExpr<T>>,
    /// The slots whose address the function takes, whose contents are not
    /// tracked.
    ///
    /// A pointer to such a slot lets a call or an indirect store write it
    /// where no instruction names it, so what the evaluator last saw written
    /// there says nothing about what a later load reads. A load from one
    /// yields an unknown value. Construction resolves every load from any
    /// other slot to its SSA value, so these are nearly all the loads there
    /// are.
    ///
    /// Computed on the first slot load rather than on construction: finding
    /// them walks the whole function, and callers build an evaluator per path
    /// or per branch over functions of thousands of blocks, most of which
    /// never meet a slot load.
    address_taken: Option<AddressTakenSlots>,
    /// Expression component of current CIL arguments, indexed by argument index.
    /// Updated whenever a variable with `Argument(N)` origin receives a value,
    /// and read by `LoadArg` instructions.
    arg_state: BTreeMap<u16, SymbolicExpr<T>>,
    /// Undo records for speculative evaluation, oldest first.
    ///
    /// Empty and unused until the first [`checkpoint`](Self::checkpoint); see
    /// that method for the rationale.
    journal: Vec<Undo<T>>,
    /// Whether mutations are being recorded into [`journal`](Self::journal).
    ///
    /// Latched on by the first checkpoint so that evaluators which never
    /// explore alternatives pay nothing.
    journaling: bool,
}

impl<'a, T: Target> SsaEvaluator<'a, T> {
    /// Creates an evaluator of `ssa` computing as `machine` does, with the
    /// default configuration ([`Self::with_config`]).
    ///
    /// The evaluator starts with no known values. Use the `set_*` methods
    /// to provide initial values for input variables.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `machine` - The machine the function computes on.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, machine: Machine) -> Self {
        Self {
            ssa,
            execution: Evaluation::Value(()),
            halted: false,
            values: BTreeMap::new(),
            opaque_values: BTreeMap::new(),
            next_instance: 0,
            input_symbols: true,
            integer_bits: BTreeMap::new(),
            slot_bits: BTreeMap::new(),
            predecessor: None,
            constraints: BTreeMap::new(),
            config: EvaluatorConfig::default(),
            path: Vec::new(),
            memory: MemoryCells::new(),
            field_dependencies: BTreeSet::new(),
            machine,
            local_state: BTreeMap::new(),
            address_taken: None,
            arg_state: BTreeMap::new(),
            journal: Vec::new(),
            journaling: false,
        }
    }

    /// Returns this evaluator configured by `config`.
    ///
    /// Changing memory tracking discards field snapshots and their dependency
    /// index; changing path tracking clears the recorded path. Re-enabling a
    /// domain cannot recover observations from an interval it did not track.
    /// Checkpoint rollback restores the configuration and tracked state together.
    #[must_use]
    pub fn with_config(mut self, config: EvaluatorConfig) -> Self {
        if self.config.track_memory != config.track_memory {
            self.memory.clear();
            self.field_dependencies.clear();
        }
        if self.config.track_path != config.track_path {
            self.path.clear();
        }
        self.config = config;
        self
    }

    /// Returns a reference to the underlying SSA function.
    #[must_use]
    pub fn ssa(&self) -> &SsaFunction<T> {
        self.ssa
    }

    /// Returns a reference to the evaluator configuration.
    #[must_use]
    pub fn config(&self) -> &EvaluatorConfig {
        &self.config
    }

    /// Returns the execution path if path tracking is enabled.
    #[must_use]
    pub fn path(&self) -> &[usize] {
        &self.path
    }

    /// Clears the recorded execution path.
    pub fn clear_path(&mut self) {
        self.path.clear();
    }

    /// Returns whether memory tracking is enabled.
    #[must_use]
    pub fn memory_tracking_enabled(&self) -> bool {
        self.config.track_memory
    }

    // Speculative Evaluation

    /// Marks the current state so it can be restored later with
    /// [`rollback`](Self::rollback).
    ///
    /// # Why this exists
    ///
    /// Consumers that explore alternative executions - a symbolic executor
    /// following both arms of a branch, a control-flow deobfuscator tracing
    /// every path through a dispatcher - need to evaluate from the same state
    /// more than once. Cloning the evaluator does that, but the copy is
    /// proportional to everything learned so far, and the state only grows as
    /// evaluation proceeds. An exploration that forks *f* times over a function
    /// with *v* variables therefore costs `O(f · v)` in copying alone, which
    /// comes to dominate the evaluation it exists to support.
    ///
    /// A checkpoint costs nothing, and rolling back to it costs one entry per
    /// mutation made since - the change actually being undone, not the state it
    /// was applied to.
    ///
    /// # Usage
    ///
    /// Marks nest, and rolling back to an outer mark subsumes any inner ones,
    /// so an abandoned branch does not have to be unwound level by level:
    ///
    /// ```
    /// use analyssa::{
    ///     analysis::SsaEvaluator,
    ///     ir::{value::ConstValue, variable::SsaVarId},
    ///     testing,
    /// };
    ///
    /// let ssa = testing::const_i32_return(7);
    /// let mut evaluator = SsaEvaluator::new(&ssa, testing::MACHINE);
    /// let v = SsaVarId::from_index(0);
    /// evaluator.set_concrete(v, ConstValue::I32(1));
    ///
    /// let outer = evaluator.checkpoint();
    /// evaluator.set_concrete(v, ConstValue::I32(2));
    /// let inner = evaluator.checkpoint();
    /// evaluator.set_concrete(v, ConstValue::I32(3));
    /// evaluator.rollback(inner);
    /// assert_eq!(evaluator.get_concrete(v), Some(&ConstValue::I32(2)));
    ///
    /// // Rolling back the outer mark subsumes anything taken after it.
    /// evaluator.rollback(outer);
    /// assert_eq!(evaluator.get_concrete(v), Some(&ConstValue::I32(1)));
    /// ```
    ///
    /// Marks must be rolled back in reverse order of creation (last taken,
    /// first released), which is the natural discipline for a depth-first
    /// exploration. Recording is enabled by the first checkpoint, so evaluators
    /// that never explore alternatives are unaffected.
    pub fn checkpoint(&mut self) -> EvaluatorMark<T> {
        self.journaling = true;
        EvaluatorMark {
            execution: self.execution.clone(),
            halted: self.halted,
            len: self.journal.len(),
            config: self.config.clone(),
            tracked: (self.config.track_memory || self.config.track_path).then(|| {
                Box::new(TrackedState {
                    memory: self.memory.clone(),
                    dependencies: self.field_dependencies.clone(),
                    path: self.path.clone(),
                })
            }),
        }
    }

    /// Restores the state captured by `mark`, discarding every change made
    /// since.
    ///
    /// The mark must come from this evaluator's current speculative history.
    /// Roll back nested marks in reverse order of creation. Rolling back an
    /// outer mark discards its later journal entries and invalidates marks
    /// taken inside it. Cloning a mark does not extend that validity.
    ///
    /// Restores expressions, partial variable and slot facts, origin mirrors,
    /// constraints, predecessor state and configuration. Memory and path
    /// snapshots are restored when enabled at checkpoint time. Domains disabled
    /// at the mark are cleared, including observations added after a mode change.
    /// Current opaque bindings and input naming roll back; the generation
    /// allocator remains at its high-water mark so later definitions stay fresh.
    pub fn rollback(&mut self, mark: EvaluatorMark<T>) {
        while self.journal.len() > mark.len {
            let Some(entry) = self.journal.pop() else {
                break;
            };
            match entry {
                Undo::Value(var, previous) => match previous {
                    Some(expr) => {
                        self.values.insert(var, expr);
                    }
                    None => {
                        self.values.remove(&var);
                    }
                },
                Undo::Local(idx, previous) => match previous {
                    Some(expr) => {
                        self.local_state.insert(idx, expr);
                    }
                    None => {
                        self.local_state.remove(&idx);
                    }
                },
                Undo::Arg(idx, previous) => match previous {
                    Some(expr) => {
                        self.arg_state.insert(idx, expr);
                    }
                    None => {
                        self.arg_state.remove(&idx);
                    }
                },
                Undo::Constraints(var, previous) => match previous {
                    Some(list) => {
                        self.constraints.insert(var, list);
                    }
                    None => {
                        self.constraints.remove(&var);
                    }
                },
                Undo::Bits(var, previous) => match previous {
                    Some(bits) => {
                        self.integer_bits.insert(var, bits);
                    }
                    None => {
                        self.integer_bits.remove(&var);
                    }
                },
                Undo::SlotBits(origin, previous) => match previous {
                    Some(bits) => {
                        self.slot_bits.insert(origin, bits);
                    }
                    None => {
                        self.slot_bits.remove(&origin);
                    }
                },
                Undo::Opaque(var, previous) => match previous {
                    Some(identity) => {
                        self.opaque_values.insert(var, identity);
                    }
                    None => {
                        self.opaque_values.remove(&var);
                    }
                },
                Undo::InputSymbols(previous) => self.input_symbols = previous,
                Undo::Predecessor(previous) => self.predecessor = previous,
            }
        }
        if let Some(tracked) = mark.tracked {
            self.memory = tracked.memory;
            self.field_dependencies = tracked.dependencies;
            self.path = tracked.path;
        } else {
            // Disabled domains are empty at mark time; a later mode change can
            // populate them, but that work belongs to the reverted transaction.
            self.memory.clear();
            self.field_dependencies.clear();
            self.path.clear();
        }
        self.config = mark.config;
        self.execution = mark.execution;
        self.halted = mark.halted;
    }

    /// Records the pre-change value of `var` and writes `expr` in its place.
    ///
    /// Every write to `values` goes through here so that both the journal and
    /// the current local/argument mirrors describe the same assignment.
    fn put_value(&mut self, var: SsaVarId, expr: SymbolicExpr<T>) {
        self.drop_opaque(var);
        self.invalidate_field_dependencies(var);
        self.drop_bits(var);
        self.track_origin_state(var, &expr);
        let previous = self.values.insert(var, expr);
        if self.journaling {
            self.journal.push(Undo::Value(var, previous));
        }
    }

    /// Forgets a value's expression, opaque binding, bit facts and origin mirror.
    ///
    /// Unknown definition boundaries subsequently call [`Self::set_unknown`]
    /// to install one fresh identity after evaluation declines.
    fn drop_value(&mut self, var: SsaVarId) {
        self.drop_opaque(var);
        self.forget_facts(var);
    }

    /// Forgets a value's expression, bit facts and origin mirror, keeping its
    /// opaque binding for the caller to remove or replace.
    fn forget_facts(&mut self, var: SsaVarId) {
        self.invalidate_field_dependencies(var);
        match self.ssa.variable(var).map(|variable| variable.origin()) {
            Some(VariableOrigin::Local(index)) => self.drop_local(index),
            Some(VariableOrigin::Argument(index)) => self.drop_arg(index),
            _ => {}
        }
        self.drop_bits(var);
        if let Some(previous) = self.values.remove(&var)
            && self.journaling
        {
            self.journal.push(Undo::Value(var, Some(previous)));
        }
    }

    /// Removes an obsolete current opaque binding while recording speculative undo.
    fn drop_opaque(&mut self, var: SsaVarId) {
        if let Some(previous) = self.opaque_values.remove(&var)
            && self.journaling
        {
            self.journal.push(Undo::Opaque(var, Some(previous)));
        }
    }

    /// Stores a current opaque binding, including refusal after generation exhaustion.
    fn put_opaque(&mut self, var: SsaVarId, identity: Option<SymbolicInstance>) {
        let previous = self.opaque_values.insert(var, identity);
        if self.journaling {
            self.journal.push(Undo::Opaque(var, previous));
        }
    }

    /// Allocates a fresh context-local token without names or graph traversal.
    ///
    /// Returns `None` on generation exhaustion; saturation never reuses a token.
    fn fresh_instance(&mut self, source: SsaVarId) -> Option<SymbolicInstance> {
        let generation = NonZeroU64::new(self.next_instance.checked_add(1)?)?;
        self.next_instance = generation.get();
        Some(SymbolicInstance::new(source, generation))
    }

    /// Reads a current expression or a scoped identity for an otherwise unknown value.
    ///
    /// A value no instruction has defined may name an original SSA input. Executed
    /// unknown definitions use their current token. After a reset, unresolved
    /// values get fresh tokens rather than reusing an old unversioned binding.
    fn current_symbol(&mut self, var: SsaVarId) -> Option<SymbolicExpr<T>> {
        if let Some(expression) = self.values.get(&var) {
            return Some(expression.clone());
        }
        if let Some(identity) = self.opaque_values.get(&var) {
            return identity.map(SymbolicExpr::Instance);
        }
        if self.input_symbols {
            return Some(SymbolicExpr::variable(var));
        }
        let identity = self.fresh_instance(var);
        self.put_opaque(var, identity);
        identity.map(SymbolicExpr::Instance)
    }

    /// Writes a fresh opaque expression when a computed tree cannot be retained.
    ///
    /// Returns `None` when no identity remains and `var` is forgotten instead.
    fn opaque_result(&mut self, var: SsaVarId) -> Option<()> {
        let Some(identity) = self.fresh_instance(var) else {
            self.drop_value(var);
            return None;
        };
        self.put_value(var, SymbolicExpr::Instance(identity));
        Some(())
    }

    /// Removes partial facts whenever a value is assigned or invalidated.
    fn drop_bits(&mut self, var: SsaVarId) {
        if let Some(previous) = self.integer_bits.remove(&var)
            && self.journaling
        {
            self.journal.push(Undo::Bits(var, Some(previous)));
        }
    }

    /// Stores partial facts with the same speculative lifetime as their expression.
    ///
    /// The caller has already replaced or invalidated the expression and its
    /// origin mirror. Only nonempty, nonconcrete facts enter this sparse domain.
    /// A local or argument receives a copy of the facts rather than an SSA source
    /// identity, preserving the slot contents when that source is reassigned.
    fn put_bits(&mut self, var: SsaVarId, bits: IntegerBits) {
        let previous = self.integer_bits.insert(var, bits);
        if self.journaling {
            self.journal.push(Undo::Bits(var, previous));
        }
        if let Some(origin @ (VariableOrigin::Local(_) | VariableOrigin::Argument(_))) =
            self.ssa.variable(var).map(|variable| variable.origin())
        {
            let previous = self.slot_bits.insert(origin, bits);
            if self.journaling {
                self.journal.push(Undo::SlotBits(origin, previous));
            }
        }
    }

    /// Forgets partial slot contents while preserving speculative undo information.
    fn drop_slot_bits(&mut self, origin: VariableOrigin) {
        if let Some(previous) = self.slot_bits.remove(&origin)
            && self.journaling
        {
            self.journal.push(Undo::SlotBits(origin, Some(previous)));
        }
    }

    /// Reads a non-address-taken slot at the destination's exact scalar width.
    ///
    /// Call before assigning the load destination: its origin may name the same
    /// slot, so invalidation during assignment must not erase the source fact.
    fn slot_integer_bits(&mut self, op: &SsaOp<T>, dest: Destination) -> Option<IntegerBits> {
        let origin = match op {
            SsaOp::LoadLocal { local_index, .. } if !self.slots_taken().local(*local_index) => {
                VariableOrigin::Local(*local_index)
            }
            SsaOp::LoadArg { arg_index, .. } if !self.slots_taken().argument(*arg_index) => {
                VariableOrigin::Argument(*arg_index)
            }
            _ => return None,
        };
        let bits = *self.slot_bits.get(&origin)?;
        (bits.width() == dest.scalar_integer()?.bits()).then_some(bits)
    }

    /// Invalidates runtime identities without rescanning unrelated field payloads.
    ///
    /// Reassigning an object id discards its cells. Reassigning a variable used
    /// in a captured symbolic expression forgets that expression, while retaining
    /// independent partial masks. The sparse index may contain stale entries
    /// after alias invalidation, but never omits a current dependency.
    fn invalidate_field_dependencies(&mut self, var: SsaVarId) {
        if !self.field_dependencies.remove(&var) {
            return;
        }
        self.memory.retain(|location, stored| {
            if location.base_object() == Some(var) {
                return false;
            }
            let depends = stored.expression.as_ref().is_some_and(|expr| match expr {
                SymbolicExpr::Constant(_)
                | SymbolicExpr::NamedVar(_)
                | SymbolicExpr::Instance(_) => false,
                SymbolicExpr::Variable(value) => *value == var,
                SymbolicExpr::Unary { .. }
                | SymbolicExpr::Binary { .. }
                | SymbolicExpr::TypedUnary { .. }
                | SymbolicExpr::TypedBinary { .. } => {
                    let mut found = false;
                    expr.for_each_variable(|value| found |= value == var);
                    found
                }
            });
            if depends {
                stored.expression = None;
            }
            true
        });
    }

    /// Invalidates field snapshots before an access can consume or replace them.
    ///
    /// Named fields are tracked only in the default address space at current
    /// permission. Locality and alignment describe how an admitted access is
    /// made and do not change its value. Volatile accesses, fences and atomics
    /// are observation boundaries. Other writes without a modeled field or
    /// call location may alias every tracked field and discard all snapshots.
    ///
    /// Returns `false` for a qualified field access whose value this evaluator
    /// cannot establish. Its definitions must become unknown without consulting
    /// either the expression snapshot or its independently retained bit facts.
    fn prepare_field_memory(&mut self, op: &SsaOp<T>) -> bool {
        if !self.config.track_memory {
            return true;
        }
        let unsupported_field = match op {
            SsaOp::LoadField { access, .. }
            | SsaOp::StoreField { access, .. }
            | SsaOp::LoadStaticField { access, .. }
            | SsaOp::StoreStaticField { access, .. } => {
                access.volatile
                    || access.address_space.is_some()
                    || !matches!(access.permission, AccessPermission::Current)
            }
            _ => false,
        };
        if !unsupported_field && self.memory.is_empty() {
            return true;
        }
        let effects = op.effects();
        let modeled_write = matches!(
            op,
            SsaOp::StoreField { .. } | SsaOp::StoreStaticField { .. } | SsaOp::Call(_)
        );
        if unsupported_field
            || effects.volatile
            || effects.ordering.is_some()
            || matches!(effects.kind, SsaEffectKind::Atomic | SsaEffectKind::Fence)
            || (effects.writes_memory() && !modeled_write)
        {
            self.memory.clear();
            self.field_dependencies.clear();
        }
        !unsupported_field
    }

    /// Captures a field's current expression and partial facts before any later write.
    fn store_field_value(&mut self, location: MemoryLocation<T>, value: SsaVarId) {
        let expression = self.current_symbol(value);
        if let Some(object) = location.base_object() {
            self.field_dependencies.insert(object);
        }
        if let Some(expression) = expression.as_ref() {
            match expression {
                SymbolicExpr::Constant(_)
                | SymbolicExpr::NamedVar(_)
                | SymbolicExpr::Instance(_) => {}
                SymbolicExpr::Variable(value) => {
                    self.field_dependencies.insert(*value);
                }
                SymbolicExpr::Unary { .. }
                | SymbolicExpr::Binary { .. }
                | SymbolicExpr::TypedUnary { .. }
                | SymbolicExpr::TypedBinary { .. } => {
                    expression.for_each_variable(|value| {
                        self.field_dependencies.insert(value);
                    });
                }
            }
        }
        self.memory.store(
            location,
            StoredField {
                source: value,
                expression,
                bits: self.integer_bits.get(&value).copied(),
            },
        );
    }

    /// Reads field masks before the load destination can invalidate its source cell.
    fn field_integer_bits(&self, op: &SsaOp<T>, dest: Destination) -> Option<IntegerBits> {
        if !self.config.track_memory {
            return None;
        }
        let location = match op {
            SsaOp::LoadField { object, field, .. } => {
                MemoryLocation::InstanceField(*object, field.clone())
            }
            SsaOp::LoadStaticField { field, .. } => MemoryLocation::StaticField(field.clone()),
            _ => return None,
        };
        let bits = self
            .memory
            .load(&location, |stored| Reverse(stored.source))?
            .bits?;
        (bits.width() == dest.scalar_integer()?.bits()).then_some(bits)
    }

    /// Copies a field expression without consulting its SSA source's current value.
    ///
    /// A captured symbolic input retains its identity while that input is
    /// unchanged. After a dependency changes, the load names the unknown value
    /// itself; no later source assignment retroactively determines the store.
    fn load_field_value(&mut self, dest: SsaVarId, location: MemoryLocation<T>) -> Option<()> {
        let stored = self
            .config
            .track_memory
            .then(|| {
                self.memory
                    .load(&location, |stored| Reverse(stored.source))
                    .map(|stored| stored.expression.clone())
            })
            .flatten();
        match stored {
            Some(Some(expression)) => {
                self.put_value(dest, expression);
                Some(())
            }
            Some(None) => self.opaque_result(dest),
            None => {
                self.drop_value(dest);
                None
            }
        }
    }

    /// Reads an inline bit fact or projects an already concrete operand.
    fn bits_for(&self, var: SsaVarId, layout: OperandLayout) -> Option<IntegerBits> {
        self.integer_bits.get(&var).copied().or_else(|| {
            self.machine
                .constant_integer_bits(self.get_concrete(var)?, layout)
        })
    }

    /// The function's address-taken slots, found on first use.
    fn slots_taken(&mut self) -> &AddressTakenSlots {
        let ssa = self.ssa;
        self.address_taken
            .get_or_insert_with(|| ssa.address_taken_slots())
    }

    /// Records the pre-change value of local slot `idx` and writes `expr`.
    fn put_local(&mut self, idx: u16, expr: SymbolicExpr<T>) {
        self.drop_slot_bits(VariableOrigin::Local(idx));
        if self.journaling {
            let previous = self.local_state.get(&idx).cloned();
            self.journal.push(Undo::Local(idx, previous));
        }
        self.local_state.insert(idx, expr);
    }

    /// Records the pre-change value of argument slot `idx` and writes `expr`.
    fn put_arg(&mut self, idx: u16, expr: SymbolicExpr<T>) {
        self.drop_slot_bits(VariableOrigin::Argument(idx));
        if self.journaling {
            let previous = self.arg_state.get(&idx).cloned();
            self.journal.push(Undo::Arg(idx, previous));
        }
        self.arg_state.insert(idx, expr);
    }

    /// Records the pre-change value of local slot `idx` and forgets it.
    fn drop_local(&mut self, idx: u16) {
        self.drop_slot_bits(VariableOrigin::Local(idx));
        if let Some(previous) = self.local_state.remove(&idx)
            && self.journaling
        {
            self.journal.push(Undo::Local(idx, Some(previous)));
        }
    }

    /// Records the pre-change value of argument slot `idx` and forgets it.
    fn drop_arg(&mut self, idx: u16) {
        self.drop_slot_bits(VariableOrigin::Argument(idx));
        if let Some(previous) = self.arg_state.remove(&idx)
            && self.journaling
        {
            self.journal.push(Undo::Arg(idx, Some(previous)));
        }
    }

    // Value Setting

    /// Sets a concrete (known) value for a variable.
    ///
    /// The caller is responsible for providing the correct `ConstValue` type
    /// that matches the variable's type in the SSA function.
    pub fn set_concrete(&mut self, var: SsaVarId, value: ConstValue<T>) {
        let expr = SymbolicExpr::constant(value);
        self.put_value(var, expr);
    }

    /// Sets a symbolic value for a variable using a named expression.
    ///
    /// This is useful for marking method arguments or other external inputs
    /// as symbolic with descriptive names. A name identifies one stable input
    /// within this context; [`Self::set_unknown`] models a fresh opaque definition.
    pub fn set_symbolic(&mut self, var: SsaVarId, name: impl Into<String>) {
        let expr = SymbolicExpr::named(name);
        self.put_value(var, expr);
    }

    /// Invalidates a variable's expression, partial bits and mirrored slot.
    ///
    /// Local and argument origins also forget the corresponding slot value so
    /// a later load cannot recover a constant from before this invalidation.
    /// Checkpoint rollback restores all three domains and the current opaque
    /// identity. The invalidated definition itself receives a fresh opaque token.
    pub fn set_unknown(&mut self, var: SsaVarId) {
        self.forget_facts(var);
        // The new identity replaces any older binding in one journaled write.
        let identity = self.fresh_instance(var);
        self.put_opaque(var, identity);
    }

    /// Sets an expression for a variable.
    pub fn set(&mut self, var: SsaVarId, value: SymbolicExpr<T>) {
        self.put_value(var, value);
    }

    // Value Getting

    /// Returns the tracked expression for a variable, when one is available.
    ///
    /// An unknown result may retain an opaque identity internally while returning
    /// `None` here. Expression construction at a use can expose that identity as
    /// [`SymbolicExpr::Instance`]. Use [`Self::get_concrete`] for known constants.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&SymbolicExpr<T>> {
        self.values.get(&var)
    }

    /// Gets the typed constant value for a variable, if it's a constant.
    ///
    /// Returns `None` if the variable is symbolic, unknown, or not set.
    /// Use [`ConstValue`] methods to extract specific types (e.g., `as_i64()`, `as_i32()`).
    #[must_use]
    pub fn get_concrete(&self, var: SsaVarId) -> Option<&ConstValue<T>> {
        self.values.get(&var).and_then(SymbolicExpr::as_constant)
    }

    /// Gets the symbolic expression for a variable, if it's not a constant.
    #[must_use]
    pub fn get_symbolic(&self, var: SsaVarId) -> Option<&SymbolicExpr<T>> {
        self.values.get(&var).filter(|e| !e.is_constant())
    }

    /// Checks if a variable has a concrete (constant) value.
    #[must_use]
    pub fn is_concrete(&self, var: SsaVarId) -> bool {
        self.values.get(&var).is_some_and(SymbolicExpr::is_constant)
    }

    /// Checks if a variable has a symbolic (non-constant) value.
    #[must_use]
    pub fn is_symbolic(&self, var: SsaVarId) -> bool {
        self.values.get(&var).is_some_and(|e| !e.is_constant())
    }

    /// Checks if a variable is unknown (not in the values map).
    #[must_use]
    pub fn is_unknown(&self, var: SsaVarId) -> bool {
        !self.values.contains_key(&var)
    }

    /// Returns all tracked values as expressions.
    #[must_use]
    pub fn values(&self) -> &BTreeMap<SsaVarId, SymbolicExpr<T>> {
        &self.values
    }

    /// Returns all concrete values as a map of i64 values.
    ///
    /// This is useful for compatibility with code that expects `BTreeMap<SsaVarId, i64>`.
    /// Values that can't be converted to i64 are skipped.
    #[must_use]
    pub fn concrete_values(&self) -> BTreeMap<SsaVarId, i64> {
        self.values
            .iter()
            .filter_map(|(k, v)| v.as_i64().map(|c| (*k, c)))
            .collect()
    }

    /// Returns all concrete values as typed `ConstValue`.
    #[must_use]
    pub fn const_values(&self) -> BTreeMap<SsaVarId, ConstValue<T>> {
        self.values
            .iter()
            .filter_map(|(k, v)| v.as_constant().map(|c| (*k, c.clone())))
            .collect()
    }

    /// Clears SSA values, opaque bindings, predecessor state and derived constraints.
    ///
    /// Resets the value-level state so the evaluator can be reused against the
    /// same SSA function. This clears tracked expressions and partial variable
    /// facts and current opaque bindings; it does not preserve their allocated
    /// nodes. Future unresolved values receive fresh instance identities. Memory
    /// state, the execution path and local/argument slot mirrors remain intact.
    /// Execution knowledge resets to a fresh normally reachable prefix. When journaling is
    /// active the discarded entries are recorded first, so a `clear` performed
    /// after a [`checkpoint`](Self::checkpoint) is undone by the matching
    /// [`rollback`](Self::rollback).
    pub fn clear(&mut self) {
        self.execution = Evaluation::Value(());
        self.halted = false;
        if self.journaling {
            let dropped: Vec<_> = self
                .values
                .iter()
                .map(|(k, v)| Undo::Value(*k, Some(v.clone())))
                .chain(
                    self.constraints
                        .iter()
                        .map(|(k, v)| Undo::Constraints(*k, Some(v.clone()))),
                )
                .chain(
                    self.integer_bits
                        .iter()
                        .map(|(var, bits)| Undo::Bits(*var, Some(*bits))),
                )
                .chain(
                    self.opaque_values
                        .iter()
                        .map(|(var, identity)| Undo::Opaque(*var, Some(*identity))),
                )
                .chain(iter::once(Undo::InputSymbols(self.input_symbols)))
                .chain(iter::once(Undo::Predecessor(self.predecessor)))
                .collect();
            self.journal.extend(dropped);
        }
        self.values.clear();
        self.opaque_values.clear();
        self.input_symbols = false;
        self.integer_bits.clear();
        self.predecessor = None;
        self.constraints.clear();
    }

    // Constraint Management

    /// Adds a constraint on a variable.
    ///
    /// If the constraint is an equality constraint, also sets the variable's value
    /// to concrete. This allows constraint propagation to directly affect evaluation.
    pub fn add_constraint(&mut self, var: SsaVarId, constraint: Constraint<T>) {
        // If it's an equality constraint, we can directly set the value
        if let Constraint::Equal(ref v) = constraint {
            let expr = SymbolicExpr::constant(v.clone());
            self.put_value(var, expr.clone());
        }

        if self.journaling {
            let previous = self.constraints.get(&var).cloned();
            self.journal.push(Undo::Constraints(var, previous));
        }
        self.constraints.entry(var).or_default().push(constraint);
    }

    /// Gets all constraints on a variable.
    #[must_use]
    pub fn constraints(&self, var: SsaVarId) -> &[Constraint<T>] {
        self.constraints.get(&var).map_or(&[], |v| v.as_slice())
    }

    /// Checks if a variable has any constraints.
    #[must_use]
    pub fn has_constraints(&self, var: SsaVarId) -> bool {
        self.constraints.get(&var).is_some_and(|v| !v.is_empty())
    }

    /// Clears constraints for a specific variable.
    pub fn clear_constraints(&mut self, var: SsaVarId) {
        if self.journaling {
            let previous = self.constraints.get(&var).cloned();
            self.journal.push(Undo::Constraints(var, previous));
        }
        self.constraints.remove(&var);
    }

    /// Applies constraints derived from taking a specific branch.
    ///
    /// When we know which branch was taken, we can derive facts about the condition
    /// variable. For example, if we took the true branch of `if (ceq x, 5)`, we know x == 5.
    ///
    /// # Arguments
    ///
    /// * `condition` - The variable used as the branch condition
    /// * `took_true_branch` - Whether we followed the true or false branch
    ///
    /// # Returns
    ///
    /// `true` if constraints were successfully derived, `false` otherwise.
    pub fn apply_branch_constraint(&mut self, condition: SsaVarId, took_true_branch: bool) -> bool {
        // Scan-free lookup, and `false` is the failure policy. A stale def site
        // costs one branch its constraint - the evaluator already answers
        // `false` for every condition it cannot read - where the recovering
        // form would walk the whole function on each miss, and this runs once
        // per branch *per fixed-point iteration*. Entry-defined variables make
        // that concrete: they carry no instruction index and no phi to confirm,
        // so a recovering lookup scans the function and still answers nothing.
        // Deriving a constraint from a foreign instruction, by contrast, is not
        // a weaker answer but a wrong one.
        let Some(definition) = self.ssa.recorded_definition(condition) else {
            return false;
        };

        self.derive_constraints_from_comparison(definition.op(), took_true_branch)
    }

    /// Derives constraints from a comparison operation, given which branch was
    /// taken.
    ///
    /// Backs [`apply_branch_constraint`](Self::apply_branch_constraint): once
    /// the branch condition's defining op is known, this reads the op's operands
    /// and records what the taken edge proves about the non-constant side. Only
    /// `Ceq`, `Cgt`, and `Clt` carry usable facts; for the ordered comparisons
    /// the constraint is recorded under the *same* signedness the op used, since
    /// a signed bound does not follow from an unsigned compare or vice versa. A
    /// constraint is emitted only when exactly one operand is a known constant:
    /// two symbolic operands leave nothing concrete to bound, and two constants
    /// need no constraint because the branch was already decided.
    ///
    /// # Arguments
    ///
    /// * `op` - The defining operation of the branch condition variable.
    /// * `took_true_branch` - Whether the true or false edge was followed.
    ///
    /// # Returns
    ///
    /// `true` if a usable constraint was derived (or the operands were already
    /// concrete and consistent), `false` otherwise.
    fn derive_constraints_from_comparison(
        &mut self,
        op: &SsaOp<T>,
        took_true_branch: bool,
    ) -> bool {
        if let SsaOp::Ceq { left, right, .. } = op {
            for variable in [*left, *right] {
                if let Some(value) = self.get_concrete(variable).and_then(|value| {
                    OperandLayout::of(self.ssa, variable)
                        .read_comparison_ref(value, self.machine.ptr_size)
                }) && matches!(value.as_ref(), ConstValue::Float(float)
                    if float.abs().bits() == 0 || !comparison_stable(value.as_ref(), self.machine.float))
                {
                    return false;
                }
            }
        }
        match op {
            SsaOp::Ceq { left, right, .. } => {
                // ceq: true branch means left == right, false means left != right
                let left_val = self.get(*left).cloned();
                let right_val = self.get(*right).cloned();

                if took_true_branch {
                    // left == right
                    match (&left_val, &right_val) {
                        (Some(l), None) => {
                            if let Some(v) = l.as_constant() {
                                self.add_constraint(*right, Constraint::Equal(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        (None, Some(r)) => {
                            if let Some(v) = r.as_constant() {
                                self.add_constraint(*left, Constraint::Equal(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        (Some(l), Some(r)) if l.as_constant() == r.as_constant() => {
                            // Both concrete and equal - constraint is satisfied
                            true
                        }
                        _ => false,
                    }
                } else {
                    // left != right
                    match (&left_val, &right_val) {
                        (Some(l), None) => {
                            if let Some(v) = l.as_constant() {
                                self.add_constraint(*right, Constraint::NotEqual(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        (None, Some(r)) => {
                            if let Some(v) = r.as_constant() {
                                self.add_constraint(*left, Constraint::NotEqual(v.clone()));
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    }
                }
            }

            SsaOp::Cgt {
                left,
                right,
                unsigned,
                ..
            } => {
                // cgt: true branch means left > right
                let right_val = self.get(*right).and_then(|e| e.as_constant().cloned());

                if took_true_branch {
                    // left > right
                    if let Some(v) = right_val {
                        if *unsigned {
                            self.add_constraint(*left, Constraint::GreaterThanUnsigned(v));
                        } else {
                            self.add_constraint(*left, Constraint::GreaterThan(v));
                        }
                        return true;
                    }
                } else {
                    // left <= right - under the *same* ordering the comparison
                    // used. Recording the signed form for an unsigned compare
                    // claims something the branch does not prove.
                    if let Some(v) = right_val {
                        if *unsigned {
                            self.add_constraint(*left, Constraint::LessOrEqualUnsigned(v));
                        } else {
                            self.add_constraint(*left, Constraint::LessOrEqual(v));
                        }
                        return true;
                    }
                }
                false
            }

            SsaOp::Clt {
                left,
                right,
                unsigned,
                ..
            } => {
                // clt: true branch means left < right
                let right_val = self.get(*right).and_then(|e| e.as_constant().cloned());

                if took_true_branch {
                    // left < right
                    if let Some(v) = right_val {
                        if *unsigned {
                            self.add_constraint(*left, Constraint::LessThanUnsigned(v));
                        } else {
                            self.add_constraint(*left, Constraint::LessThan(v));
                        }
                        return true;
                    }
                } else {
                    // left >= right, under the comparison's own ordering.
                    if let Some(v) = right_val {
                        if *unsigned {
                            self.add_constraint(*left, Constraint::GreaterOrEqualUnsigned(v));
                        } else {
                            self.add_constraint(*left, Constraint::GreaterOrEqual(v));
                        }
                        return true;
                    }
                }
                false
            }

            _ => false,
        }
    }

    /// Checks if the current constraints imply that a condition is always true or false.
    ///
    /// This is useful for detecting dead code after branch conditions.
    ///
    /// # Arguments
    ///
    /// * `condition` - The variable whose branch condition is being tested.
    ///
    /// # Returns
    ///
    /// - `Some(true)` if the condition is always true given current constraints
    /// - `Some(false)` if the condition is always false given current constraints
    /// - `None` if the condition cannot be determined
    #[must_use]
    pub fn evaluate_condition_with_constraints(&self, condition: SsaVarId) -> Option<bool> {
        if let Some(v) = self.get_concrete(condition) {
            // Non-zero and non-null are true; a value with no truth value
            // (a float) decides nothing.
            return v.as_bool();
        }

        // Check if constraints imply a value
        // For now, we handle the case where we have conflicting constraints
        // which would indicate dead code
        //
        // Scan-free lookup, and `None` - "cannot be determined" - is the
        // failure policy, which is what this query already answers for a
        // condition it cannot read. The alternative is not a weaker answer but
        // a wrong one: a comparison belonging to another variable would decide
        // this branch. See `apply_branch_constraint` for why the recovering
        // form is the wrong trade here.
        self.check_condition_against_constraints(self.ssa.recorded_definition(condition)?.op())
    }

    /// Checks whether a comparison's result is forced by the constraints
    /// already recorded on its operands.
    ///
    /// Backs [`evaluate_condition_with_constraints`](Self::evaluate_condition_with_constraints)
    /// for the case where the condition is not itself concrete. The right
    /// operand must be a known constant; the left operand's recorded constraints
    /// are then scanned for one that pins the comparison. For example a `left <
    /// v` bound decides `clt(left, right)` as soon as `v <= right`, and an
    /// `Equal` constraint decides any comparison outright. Bounds that only
    /// narrow the range without crossing `right` prove nothing and are skipped.
    ///
    /// The recorded bounds are signed and ordered, so they decide only a signed
    /// integer comparison: an unsigned compare, or one against a float (where
    /// a NaN makes every ordered bound false), stays undetermined. An `Equal`
    /// constraint whose value cannot be compared with `right` decides nothing.
    ///
    /// # Arguments
    ///
    /// * `op` - The comparison operation whose result is being tested.
    ///
    /// # Returns
    ///
    /// `Some(true)` or `Some(false)` if the constraints force the result, or
    /// `None` if it remains undetermined.
    fn check_condition_against_constraints(&self, op: &SsaOp<T>) -> Option<bool> {
        match op {
            SsaOp::Ceq { left, right, .. } => {
                // Check if we know both operands are equal or not equal
                let left_constraints = self.constraints(*left);
                let right_val = self.get_concrete(*right)?;

                for constraint in left_constraints {
                    match constraint {
                        Constraint::Equal(v) => {
                            // v == right_val means ceq is true
                            return self.comparison_constant(*left, v)?.ceq_at(self.comparison_constant(*right, right_val)?.as_ref(), self.machine.ptr_size).as_ref().and_then(ConstValue::as_bool);
                        }
                        Constraint::NotEqual(v)
                            // If v == right_val, then left != right_val, so ceq is false
                            if self.comparison_constant(*left, v)?.ceq_at(self.comparison_constant(*right, right_val)?.as_ref(), self.machine.ptr_size).is_some_and(|r| !r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::GreaterThan(v)
                            // left > v, so if right_val <= v, then left != right_val
                            if right_val.cgt(v, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::LessThan(v)
                            // left < v, so if right_val >= v, then left != right_val
                            if right_val.clt(v, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        _ => {}
                    }
                }
                None
            }

            SsaOp::Cgt {
                left,
                right,
                unsigned: false,
                ..
            } => {
                if matches!(OperandLayout::of(self.ssa, *right), OperandLayout::Float(_)) {
                    return None;
                }
                let left_constraints = self.constraints(*left);
                let right_val = self.get_concrete(*right).filter(|v| !v.is_float())?;

                for constraint in left_constraints {
                    match constraint {
                        Constraint::GreaterThan(v)
                            // left > v, so if v >= right_val, then left > right_val
                            if v.clt(right_val, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(true);
                        }
                        Constraint::LessOrEqual(v)
                            // left <= v, so if v <= right_val, then left <= right_val
                            if v.cgt(right_val, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::LessThan(v)
                            // left < v, so if v <= right_val, then left < right_val <= right_val
                            if v.cgt(right_val, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::Equal(v) => {
                            // left == v, so return v > right_val
                            return self.comparison_constant(*left, v)?.cgt(self.comparison_constant(*right, right_val)?.as_ref(), self.machine.ptr_size).as_ref().and_then(ConstValue::as_bool);
                        }
                        _ => {}
                    }
                }
                None
            }

            SsaOp::Clt {
                left,
                right,
                unsigned: false,
                ..
            } => {
                if matches!(OperandLayout::of(self.ssa, *right), OperandLayout::Float(_)) {
                    return None;
                }
                let left_constraints = self.constraints(*left);
                let right_val = self.get_concrete(*right).filter(|v| !v.is_float())?;

                for constraint in left_constraints {
                    match constraint {
                        Constraint::LessThan(v)
                            // left < v, so if v <= right_val, then left < right_val
                            if v.cgt(right_val, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(true);
                        }
                        Constraint::GreaterOrEqual(v)
                            // left >= v, so if v >= right_val, then left >= right_val
                            if v.clt(right_val, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::GreaterThan(v)
                            // left > v, so if v >= right_val, then left > right_val >= right_val
                            if v.clt(right_val, self.machine.ptr_size).is_some_and(|r| r.is_zero()) =>
                        {
                            return Some(false);
                        }
                        Constraint::Equal(v) => {
                            // left == v, so return v < right_val
                            return self.comparison_constant(*left, v)?.clt(self.comparison_constant(*right, right_val)?.as_ref(), self.machine.ptr_size).as_ref().and_then(ConstValue::as_bool);
                        }
                        _ => {}
                    }
                }
                None
            }

            _ => None,
        }
    }

    // Path-Aware Evaluation

    /// Sets the predecessor block for path-aware phi evaluation.
    ///
    /// When evaluating a block, phi nodes will select the operand that
    /// corresponds to this predecessor. This enables accurate evaluation
    /// when following a specific path through the CFG.
    ///
    /// # Arguments
    ///
    /// * `pred` - The predecessor block index, or `None` to clear.
    pub fn set_predecessor(&mut self, pred: Option<usize>) {
        if self.journaling {
            self.journal.push(Undo::Predecessor(self.predecessor));
        }
        self.predecessor = pred;
    }

    /// Gets the current predecessor for phi evaluation.
    #[must_use]
    pub fn predecessor(&self) -> Option<usize> {
        self.predecessor
    }

    // Block Evaluation

    /// Evaluates all phi nodes in a block.
    ///
    /// **REQUIRES** a predecessor to be set via [`set_predecessor`](Self::set_predecessor).
    /// If no predecessor is set, phi results will be `None` (removed from value map).
    ///
    /// # Phi Node Semantics
    ///
    /// Phi nodes execute "simultaneously" - all source values are read BEFORE any
    /// results are written. This is critical for correct swap semantics:
    ///
    /// ```text
    /// v1 = phi(v2 from pred)
    /// v2 = phi(v1 from pred)
    /// ```
    ///
    /// This swaps v1 and v2. If we wrote v1 before reading for v2, we'd get the
    /// wrong value. The implementation uses a two-phase approach:
    /// 1. Read all expression and partial-bit sources before any writes.
    /// 2. Assign both domains from those snapshots in phi order.
    ///
    /// Only partial sources allocate bit scratch entries. Ordinary phis retain
    /// their expression-sized scratch records. A missing predecessor or incoming
    /// edge invalidates the complete result, including any mirrored slot.
    pub fn evaluate_phis(&mut self, block_idx: usize) {
        let Some(block) = self.ssa.block(block_idx) else {
            return;
        };
        // Snapshot both domains before assigning any result. Ordinary phis keep
        // their expression-sized scratch entries; only partial sources allocate
        // entries in this separate sparse buffer.
        let mut partial_results = Vec::new();
        let phi_results: Vec<(SsaVarId, Option<SymbolicExpr<T>>)> = block
            .phi_nodes()
            .iter()
            .enumerate()
            .map(|(index, phi)| {
                let result = phi.result();
                let source = self.predecessor.and_then(|pred| {
                    phi.operands()
                        .iter()
                        .find(|operand| operand.predecessor() == pred)
                        .map(|operand| operand.value())
                });
                let value = source.and_then(|source| self.values.get(&source).cloned());
                if let Some(bits) = source
                    .and_then(|source| self.integer_bits.get(&source))
                    .copied()
                    && OperandLayout::of(self.ssa, result)
                        .scalar_integer()
                        .is_some_and(|ty| ty.bits() == bits.width())
                {
                    partial_results.push((index, bits));
                }
                (result, value)
            })
            .collect();
        let mut partial_results = partial_results.into_iter().peekable();
        for (index, (result, value)) in phi_results.into_iter().enumerate() {
            if let Some(value) = value {
                self.put_value(result, value);
            } else {
                self.set_unknown(result);
            }
            if let Some((_, bits)) = partial_results.next_if(|(position, _)| *position == index) {
                self.put_bits(result, bits);
            }
        }
    }

    /// Evaluates all instructions in a block, updating tracked values.
    ///
    /// This evaluates phi nodes first (if predecessor is set), then
    /// evaluates all other instructions in order.
    pub fn evaluate_block(&mut self, block_idx: usize) {
        if self.halted {
            return;
        }
        // Record path if tracking is enabled
        if self.config.track_path {
            self.path.push(block_idx);
        }

        // First evaluate phi nodes
        self.evaluate_phis(block_idx);

        // Then evaluate instructions
        let Some(block) = self.ssa.block(block_idx) else {
            return;
        };

        for instr in block.instructions() {
            self.apply(instr.op());
        }
    }

    /// Evaluates a block's instructions, leaving phi results untouched.
    ///
    /// Use this when the caller has already established the phi values itself -
    /// typically by seeding them symbolically with
    /// [`set_symbolic`](Self::set_symbolic) to reason about a merge abstractly.
    ///
    /// [`evaluate_block`](Self::evaluate_block) would erase those seeds:
    /// [`evaluate_phis`](Self::evaluate_phis) resolves each phi through the
    /// current predecessor, and with no predecessor set there is no operand to
    /// choose, so it *removes* the result's value. Seeding and then calling
    /// `evaluate_block` therefore leaves every phi-derived value Unknown, which
    /// is the opposite of what the seeding was for.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::{analysis::SsaEvaluator, ir::SsaVarId, testing};
    ///
    /// let ssa = testing::diamond_phi_fixture();
    /// let mut eval = SsaEvaluator::new(&ssa, testing::MACHINE);
    ///
    /// // Block 3's phi merges the two arms; treat it as an unknown symbol.
    /// let phi_result = SsaVarId::from_index(2);
    /// eval.set_symbolic(phi_result, "merged".to_string());
    /// eval.evaluate_block_instructions(3);
    ///
    /// // The seed survives, so anything derived from it stays symbolic.
    /// assert!(eval.get(phi_result).is_some());
    /// ```
    pub fn evaluate_block_instructions(&mut self, block_idx: usize) {
        if self.halted {
            return;
        }
        if self.config.track_path {
            self.path.push(block_idx);
        }
        let Some(block) = self.ssa.block(block_idx) else {
            return;
        };
        for instr in block.instructions() {
            self.apply(instr.op());
        }
    }

    /// Evaluates a sequence of blocks in order.
    ///
    /// This is useful for evaluating a path through the CFG.
    /// Note: This does not set predecessors automatically.
    pub fn evaluate_blocks(&mut self, block_indices: &[usize]) {
        for &block_idx in block_indices {
            self.evaluate_block(block_idx);
        }
    }

    /// Evaluates a sequence of blocks along a path.
    ///
    /// For each block after the first, sets the predecessor to the previous
    /// block before evaluation. This enables accurate phi node evaluation.
    pub fn evaluate_path(&mut self, path: &[usize]) {
        for (i, &block_idx) in path.iter().enumerate() {
            if i > 0
                && let Some(&prev) = path.get(i.saturating_sub(1))
            {
                self.set_predecessor(Some(prev));
            }
            self.evaluate_block(block_idx);
        }
    }

    // Fixed-Point Iteration for Loops

    /// Evaluates a loop until values reach a fixed point.
    ///
    /// This is useful for analyzing loops where variable values may change each
    /// iteration until they stabilize. The method iterates up to `max_iterations`
    /// times, or until all tracked values stop changing.
    ///
    /// # Arguments
    ///
    /// * `loop_blocks` - The blocks that form the loop body (in execution order)
    /// * `max_iterations` - Maximum number of iterations before giving up
    ///
    /// # Returns
    ///
    /// The number of iterations performed before reaching fixed point (or max).
    pub fn evaluate_loop_to_fixpoint(
        &mut self,
        loop_blocks: &[usize],
        max_iterations: usize,
    ) -> usize {
        if loop_blocks.is_empty() {
            return 0;
        }

        // Only variables defined inside the loop can change between iterations;
        // values defined outside are invariant during loop evaluation. Snapshot
        // and compare just those instead of deep-cloning the entire value map
        // every iteration.
        let mut loop_vars: Vec<SsaVarId> = Vec::new();
        for &block_idx in loop_blocks {
            if let Some(block) = self.ssa.block(block_idx) {
                for phi in block.phi_nodes() {
                    loop_vars.push(phi.result());
                }
                for instr in block.instructions() {
                    for dest in instr.op().defs() {
                        loop_vars.push(dest);
                    }
                }
            }
        }
        loop_vars.sort_unstable();
        loop_vars.dedup();

        for iteration in 0..max_iterations {
            // Snapshot only the loop-defined values.
            let snapshot: Vec<Option<SymbolicExpr<T>>> = loop_vars
                .iter()
                .map(|v| self.values.get(v).cloned())
                .collect();

            // Evaluate all loop blocks
            for (i, &block_idx) in loop_blocks.iter().enumerate() {
                if i > 0 {
                    if let Some(&prev) = loop_blocks.get(i.saturating_sub(1)) {
                        self.set_predecessor(Some(prev));
                    }
                } else if loop_blocks.len() > 1 {
                    // First block - predecessor is the last block (loop back edge)
                    if let Some(&last) = loop_blocks.last() {
                        self.set_predecessor(Some(last));
                    }
                }
                self.evaluate_block(block_idx);
            }

            // Fixed point reached when no loop-defined value changed.
            let stable = loop_vars
                .iter()
                .zip(snapshot.iter())
                .all(|(v, old)| self.values.get(v) == old.as_ref());
            if stable {
                return iteration.saturating_add(1);
            }
        }

        // Didn't reach fixed point - mark variables that changed as widened
        self.widen_unstable_values(loop_blocks);
        max_iterations
    }

    /// Widens values that didn't stabilize in a loop to Unknown.
    ///
    /// This is called when fixed-point iteration doesn't converge. Variables
    /// defined in loop blocks that still have different values are marked Unknown.
    fn widen_unstable_values(&mut self, loop_blocks: &[usize]) {
        // Find all variables defined in the loop
        for &block_idx in loop_blocks {
            let Some(block) = self.ssa.block(block_idx) else {
                continue;
            };

            // Mark phi results as unknown (they depend on loop iteration)
            for phi in block.phi_nodes() {
                self.set_unknown(phi.result());
            }

            // Check instructions for variables that might not have stabilized
            for instr in block.instructions() {
                // If this op defines variables, consider widening them
                for dest in instr.op().defs() {
                    // Keep concrete values if they're stable, widen symbolic to unknown
                    if let Some(expr) = self.values.get(&dest)
                        && !expr.is_constant()
                    {
                        // Symbolic values that didn't stabilize become unknown
                        self.set_unknown(dest);
                    }
                }
            }
        }
    }

    /// Evaluates a loop with a specific iteration count.
    ///
    /// This is useful when you know exactly how many times a loop should run
    /// (e.g., from a constant loop bound).
    pub fn evaluate_loop_iterations(&mut self, loop_blocks: &[usize], iterations: usize) {
        for _ in 0..iterations {
            for (i, &block_idx) in loop_blocks.iter().enumerate() {
                if i > 0
                    && let Some(&prev) = loop_blocks.get(i.saturating_sub(1))
                {
                    self.set_predecessor(Some(prev));
                }
                self.evaluate_block(block_idx);
            }
        }
    }

    /// Evaluates a single SSA operation, updating tracked values.
    ///
    /// This is the per-instruction core of the evaluator, driven by
    /// [`evaluate_block`](Self::evaluate_block) and the loop and trace routines.
    /// Each arm reads its operands from the value map and writes the result back
    /// through `put_value` (so speculative evaluation can undo it), or drops the
    /// destination when the result cannot be determined so a stale value is
    /// never reused. Scalar integer arithmetic, comparisons and conversions
    /// build a typed `SymbolicExpr` from their captured semantics, folding
    /// concrete operands; other domains fold only when every operand is
    /// concrete and otherwise take a fresh opaque identity. `LoadLocal` /
    /// `LoadArg` read the CIL slot mirrors, while address-of ops invalidate them. Field loads and stores are
    /// honored only when memory tracking is enabled and their access uses the
    /// default address space at current permission. Locality and alignment
    /// hints preserve values. Volatile accesses, fences, atomics and writes
    /// without modeled locations invalidate stored expressions and bit facts;
    /// unsupported field qualifiers also make the load result unknown.
    /// Calls invalidate the fields their declared effects may change, or all
    /// fields when those effects are unknown. Calls and allocations always
    /// yield an unknown result. Every definition the evaluator does not
    /// compute - the destination of an operation it does not model, or a
    /// secondary definition such as a flags output - becomes unknown.
    /// Concrete results retain the shared kernel's numeric representation.
    /// Exact bit facts materialize a constant only when that kernel supplies
    /// none; a bit proof does not replace an existing concrete result.
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to evaluate.
    ///
    /// # Returns
    ///
    /// The primary destination's expression for operations that produce a
    /// result, or `None` for result-less operations (stores, branches) and for
    /// results that cannot be determined.
    pub fn evaluate_op(&mut self, op: &SsaOp<T>) -> Option<&SymbolicExpr<T>> {
        if self.apply(op) {
            op.dest().and_then(|dest| self.values.get(&dest))
        } else {
            None
        }
    }

    /// Applies one operation's ordered transfer.
    ///
    /// Returns whether the primary destination holds an expression afterwards.
    fn apply(&mut self, op: &SsaOp<T>) -> bool {
        if self.halted {
            return false;
        }
        if let Some(outcome) = nonreturn_terminal(op) {
            self.execution = outcome;
            self.halted = true;
            return false;
        }
        let layout = OperationLayout::of(self.ssa, op, self.machine);
        let outcome = self.machine.evaluate_control_with_operand_layouts(
            op,
            layout.destination,
            |var| self.get_concrete(var).cloned(),
            |var| layout.operand(self.ssa, var),
        );
        if let Some(outcome) = outcome {
            let normal = layout.scalar.is_some_and(|(descriptor, operands)| {
                descriptor
                    .returns_normally_with_count(operands[1].and_then(|var| self.get_concrete(var)))
                    && operands
                        .into_iter()
                        .flatten()
                        .all(|var| self.admits_normal_input(var))
            });
            self.observe_execution(outcome, normal);
        } else if op.may_throw() && matches!(self.execution, Evaluation::Value(())) {
            self.execution = Evaluation::Unknown;
        }
        if self.halted {
            for def in op.defs() {
                self.drop_value(def);
            }
            return false;
        }
        self.apply_normal(op, layout)
    }

    /// Returns whether partial facts may discard `variable`'s computation.
    ///
    /// An untracked value is a supplied input, and admits.
    fn admits_normal_input(&self, variable: SsaVarId) -> bool {
        self.values
            .get(&variable)
            .is_none_or(|expression| expression.admits_normal_input(self.machine.ptr_size))
    }

    /// Returns knowledge about normal continuation after the evaluated prefix.
    ///
    /// Values built after `Unknown` or `Unsupported` remain conditional on reaching
    /// their definitions. A proved fault prevents all following definitions.
    #[must_use]
    pub fn execution_outcome(&self) -> &Evaluation<()> {
        &self.execution
    }

    /// Records ordered source outcomes without inventing an earlier exception.
    fn observe_execution(&mut self, outcome: Evaluation<ConstValue<T>>, normal: bool) {
        match outcome {
            Evaluation::Value(_) => {}
            Evaluation::Trap(trap) => {
                self.execution = if matches!(self.execution, Evaluation::Value(())) {
                    Evaluation::Trap(trap)
                } else {
                    Evaluation::Unknown
                };
                self.halted = true;
            }
            Evaluation::Unknown => {
                if !normal && matches!(self.execution, Evaluation::Value(())) {
                    self.execution = Evaluation::Unknown;
                }
            }
            _ => self.execution = Evaluation::Unsupported,
        }
    }

    /// Transfers values conditionally on normal execution reaching this definition.
    ///
    /// Returns whether the primary destination holds an expression afterwards.
    fn apply_normal(&mut self, op: &SsaOp<T>, layout: OperationLayout) -> bool {
        if !self.prepare_field_memory(op) {
            for def in op.defs() {
                self.set_unknown(def);
            }
            return false;
        }
        if op.folds_results() {
            return self.evaluate_each(op).is_some();
        }
        let dest = layout.destination;
        let types = TypeQueries::ordinary(self.machine.ptr_size);
        let bits = if matches!(op, SsaOp::LoadLocal { .. } | SsaOp::LoadArg { .. }) {
            self.slot_integer_bits(op, dest)
        } else if matches!(op, SsaOp::LoadField { .. } | SsaOp::LoadStaticField { .. }) {
            self.field_integer_bits(op, dest)
        } else {
            self.machine.evaluate_integer_bits(
                op,
                dest,
                |var, layout| self.bits_for(var, layout),
                |var| layout.operand(self.ssa, var),
                &types,
            )
        };
        // Exact bit facts describe normal results. They cannot erase an input
        // computation whose normal completion has not been established.
        let exact = bits.and_then(|bits| {
            bits.value()?;
            op.try_for_each_use(|input| {
                if self.admits_normal_input(input) {
                    Ok(())
                } else {
                    Err(())
                }
            })
            .ok()?;
            if let Some((descriptor, operands)) = layout.scalar {
                return match descriptor.evaluate(
                    operands[0].and_then(|var| self.get_concrete(var).cloned()),
                    operands[1].and_then(|var| self.get_concrete(var).cloned()),
                ) {
                    Evaluation::Value(value) => Some(value),
                    Evaluation::Unknown => self.machine.known_result(op, bits, dest),
                    Evaluation::Unsupported | Evaluation::Trap(_) => None,
                };
            }
            let concrete = match op {
                SsaOp::Copy { src, .. } => self.get_concrete(*src).cloned(),
                _ => self.machine.evaluate_with_operand_layouts(
                    op,
                    dest,
                    |var| self.get_concrete(var).cloned(),
                    |var| layout.operand(self.ssa, var),
                ),
            };
            concrete.or_else(|| self.machine.known_result(op, bits, dest))
        });
        let primary = op.dest();
        let defined = if let (Some(var), Some(value)) = (primary, exact) {
            self.put_value(var, SymbolicExpr::constant(value));
            true
        } else {
            self.evaluate_primary(op, layout).is_some()
        };
        // A declined transfer must forget the destination from an earlier visit.
        if !defined && let Some(var) = primary {
            self.set_unknown(var);
        }
        if let (Some(var), Some(bits)) = (primary, bits)
            && bits.value().is_none()
            && !bits.is_unknown()
        {
            self.put_bits(var, bits);
        }
        for def in op.defs() {
            if Some(def) != primary {
                self.set_unknown(def);
            }
        }
        defined
    }

    /// Evaluates one operation and returns its primary value's full outcome.
    ///
    /// A retained symbolic relation can have an `Unknown` outcome. This query
    /// preserves arithmetic faults but does not establish block reachability.
    pub fn evaluate_op_outcome(&mut self, op: &SsaOp<T>) -> Evaluation<ConstValue<T>> {
        let defined = self.apply(op);
        if self.halted {
            return match self.execution {
                Evaluation::Trap(trap) => Evaluation::Trap(trap),
                Evaluation::Unsupported => Evaluation::Unsupported,
                _ => Evaluation::Unknown,
            };
        }
        op.dest()
            .and_then(|dest| self.values.get(&dest))
            .filter(|_| defined)
            .map_or(Evaluation::Unknown, |expression| {
                expression.evaluate_outcome_with(self.machine.ptr_size, |_| Evaluation::Unknown)
            })
    }

    /// Reads an immutable literal definition without replacing a current input binding.
    ///
    /// A path walk may begin after a dominating constant's block. Capture that
    /// literal directly rather than retaining it as an unresolved variable in a
    /// delayed relation. Opaque execution instances and explicit bindings keep
    /// their identity; no memory read or computed definition is replayed here.
    fn scalar_input(&mut self, variable: SsaVarId) -> Option<SymbolicExpr<T>> {
        if !self.values.contains_key(&variable)
            && !self.opaque_values.contains_key(&variable)
            && let Some(op @ SsaOp::Const { value, .. }) = self
                .ssa
                .recorded_definition(variable)
                .map(|definition| definition.op())
            && (value
                .conversion_integer_pattern(self.machine.ptr_size)
                .is_some()
                || value.is_float()
                || matches!(value, ConstValue::Symbol(_)))
            && let Some(value) = self
                .machine
                .normalize_literal(value.clone(), Destination::of_operation(self.ssa, op))
        {
            return Some(SymbolicExpr::constant(value));
        }
        self.current_symbol(variable)
    }

    /// Builds one scalar relation without cloning an over-budget child tree.
    fn evaluate_scalar(
        &mut self,
        source: &SsaOp<T>,
        descriptor: ScalarOperation,
        operands: [Option<SsaVarId>; 2],
    ) -> Option<()> {
        let dest = source.dest()?;
        let first = operands[0]?;
        // Zero bit scans define an unspecified normal result, not a numeric relation.
        if matches!(
            source,
            SsaOp::BitScanForward { .. } | SsaOp::BitScanReverse { .. }
        ) && self
            .get_concrete(first)
            .and_then(|value| {
                descriptor
                    .operand(0)?
                    .read_comparison_ref(value, self.machine.ptr_size)
            })
            .and_then(|value| value.conversion_integer_pattern(self.machine.ptr_size))
            .is_some_and(|(bits, _)| bits == 0)
        {
            return None;
        }
        let first_leaf = SymbolicExpr::variable(first);
        let first_expression = self.values.get(&first).unwrap_or(&first_leaf);
        let second_leaf = operands[1].map(SymbolicExpr::variable);
        let second_expression = operands[1]
            .and_then(|id| self.values.get(&id))
            .or(second_leaf.as_ref());
        let too_large = match second_expression {
            Some(second) => SymbolicExpr::combining_reaches_limits(&[first_expression, second]),
            None => SymbolicExpr::combining_reaches_limits(&[first_expression]),
        };
        if too_large {
            return self.opaque_result(dest);
        }
        // Resolve current instance identities only after the size check. The
        // snapshots preserve reassignment and checkpoint semantics.
        let first_expression = self.scalar_input(first)?;
        if matches!(source, SsaOp::Copy { .. })
            && self.ssa.variable(first)?.var_type() == self.ssa.variable(dest)?.var_type()
        {
            self.put_value(dest, first_expression);
            return Some(());
        }
        let second_expression = match operands[1] {
            Some(second) => Some(self.scalar_input(second)?),
            None => None,
        };
        if second_expression
            .as_ref()
            .is_some_and(|second| *second == first_expression)
            && first_expression.admits_normal_input(self.machine.ptr_size)
            && let Some(value) = descriptor.identical_result()
        {
            let value = self
                .machine
                .evaluate_with_operand_layouts(
                    source,
                    descriptor.destination(),
                    |var| self.get_concrete(var).cloned(),
                    |var| OperandLayout::of(self.ssa, var),
                )
                .unwrap_or(value);
            self.put_value(dest, SymbolicExpr::constant(value));
            return Some(());
        }
        let expression = match second_expression {
            Some(second) => SymbolicExpr::typed_binary(descriptor, first_expression, second),
            None => SymbolicExpr::typed_unary(descriptor, first_expression),
        }?
        .simplify(self.machine.ptr_size);
        let expression = self.fit_to_destination(dest, expression)?;
        self.put_value(dest, expression);
        Some(())
    }

    /// Evaluates an operation whose results fold one by one
    /// ([`SsaOp::folds_results`]): each result that folds is a
    /// constant ([`Machine::evaluate_results_with_operand_layouts`]), its
    /// operands read as their types declare them, and each other is unknown.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation.
    ///
    /// # Returns
    ///
    /// `Some` when the primary result holds a constant, `None` when it is
    /// unknown.
    fn evaluate_each(&mut self, op: &SsaOp<T>) -> Option<()> {
        let results = self
            .machine
            .evaluate_results_with_operand_layouts(
                op,
                |var| {
                    self.values
                        .get(&var)
                        .and_then(|expr| expr.as_constant().cloned())
                },
                |var| OperandLayout::of(self.ssa, var),
            )
            .unwrap_or_default();
        let primary = op.dest();
        let mut result = None;
        for def in op.defs() {
            match results.iter().find(|(folded, _)| *folded == def) {
                Some((_, value)) => {
                    self.put_value(def, SymbolicExpr::constant(value.clone()));
                    if Some(def) == primary {
                        result = Some(());
                    }
                }
                None => self.set_unknown(def),
            }
        }
        result
    }

    /// Evaluates `op` into its primary destination; see
    /// [`evaluate_op`](Self::evaluate_op), which also forgets the secondary
    /// definitions.
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to evaluate.
    /// * `layout` - `op`'s captured scalar semantics and destination.
    ///
    /// # Returns
    ///
    /// `Some` when the primary destination holds a new value, `None` when it
    /// is unknown or `op` has none.
    fn evaluate_primary(&mut self, op: &SsaOp<T>, layout: OperationLayout) -> Option<()> {
        if let Some((descriptor, operands)) = layout.scalar {
            return self.evaluate_scalar(op, descriptor, operands);
        }
        match op {
            SsaOp::Const { dest, value } => {
                let Some(value) = self
                    .machine
                    .normalize_literal(value.clone(), layout.destination)
                else {
                    self.drop_value(*dest);
                    return None;
                };
                self.put_value(*dest, SymbolicExpr::constant(value));
                Some(())
            }

            SsaOp::Copy { dest, src } => {
                let value = self.values.get(src).cloned();
                if let Some(v) = value {
                    self.put_value(*dest, v);
                    Some(())
                } else {
                    self.drop_value(*dest);
                    None
                }
            }

            SsaOp::LoadLocal { dest, local_index } => {
                let value = if self.slots_taken().local(*local_index) {
                    None
                } else {
                    self.local_state.get(local_index).cloned()
                };
                if let Some(v) = value {
                    self.put_value(*dest, v);
                    Some(())
                } else {
                    self.drop_value(*dest);
                    None
                }
            }

            SsaOp::LoadArg { dest, arg_index } => {
                let value = if self.slots_taken().argument(*arg_index) {
                    None
                } else {
                    self.arg_state.get(arg_index).cloned()
                };
                if let Some(v) = value {
                    self.put_value(*dest, v);
                    Some(())
                } else {
                    self.drop_value(*dest);
                    None
                }
            }

            // Scalar integer domains took the captured descriptor above; what
            // reaches here is a pointer, float or incomplete domain.
            SsaOp::Add {
                dest, left, right, ..
            }
            | SsaOp::Sub {
                dest, left, right, ..
            }
            | SsaOp::Mul {
                dest, left, right, ..
            }
            | SsaOp::Div {
                dest, left, right, ..
            }
            | SsaOp::Rem {
                dest, left, right, ..
            }
            | SsaOp::Xor {
                dest, left, right, ..
            }
            | SsaOp::And {
                dest, left, right, ..
            }
            | SsaOp::Or {
                dest, left, right, ..
            }
            | SsaOp::Ceq {
                dest, left, right, ..
            }
            | SsaOp::Cgt {
                dest, left, right, ..
            }
            | SsaOp::Clt {
                dest, left, right, ..
            }
            | SsaOp::Shl {
                dest,
                value: left,
                amount: right,
                ..
            }
            | SsaOp::Shr {
                dest,
                value: left,
                amount: right,
                ..
            } => self.fold_or_opaque(*dest, &[*left, *right], op),

            SsaOp::Neg { dest, operand, .. } | SsaOp::Not { dest, operand, .. } => {
                self.fold_or_opaque(*dest, &[*operand], op)
            }

            // Numeric folding retains the machine's target width and flag semantics.
            SsaOp::AddOvf { dest, .. }
            | SsaOp::SubOvf { dest, .. }
            | SsaOp::MulOvf { dest, .. }
            | SsaOp::Rol { dest, .. }
            | SsaOp::Ror { dest, .. }
            | SsaOp::BSwap { dest, .. }
            | SsaOp::BRev { dest, .. }
            | SsaOp::BitScanForward { dest, .. }
            | SsaOp::BitScanReverse { dest, .. }
            | SsaOp::Popcount { dest, .. }
            | SsaOp::Parity { dest, .. }
            | SsaOp::BoolAnd { dest, .. }
            | SsaOp::BoolOr { dest, .. }
            | SsaOp::BoolXor { dest, .. }
            | SsaOp::BoolNot { dest, .. }
            | SsaOp::ReadFlags { dest, .. }
            | SsaOp::MergeFlags { dest, .. }
            | SsaOp::InsertFlag { dest, .. } => self.fold_constant(*dest, op),
            // The exact float function of one result (`fscale`) folds where
            // the scaling does not round.
            SsaOp::FloatFunction(data) if data.kind == FloatFunctionKind::Scale => {
                match data.outputs.as_slice() {
                    [dest] => self.fold_constant(*dest, op),
                    _ => {
                        for def in op.defs() {
                            self.drop_value(def);
                        }
                        None
                    }
                }
            }

            // Conditional select (pure, conditional move)
            SsaOp::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => {
                let cond = self.values.get(condition).and_then(|e| e.as_i64());
                match cond {
                    Some(c) if c != 0 => {
                        if let Some(expr) = self.values.get(true_val).cloned() {
                            self.put_value(*dest, expr);
                            Some(())
                        } else {
                            self.drop_value(*dest);
                            None
                        }
                    }
                    Some(_) => {
                        if let Some(expr) = self.values.get(false_val).cloned() {
                            self.put_value(*dest, expr);
                            Some(())
                        } else {
                            self.drop_value(*dest);
                            None
                        }
                    }
                    None => {
                        self.drop_value(*dest);
                        None
                    }
                }
            }

            // Concrete conversion semantics are shared with constant folding
            // and execution. An unresolved conversion has no identity expression.
            SsaOp::Ckfinite { dest, .. }
            | SsaOp::IntToFloat { dest, .. }
            | SsaOp::FloatConv { dest, .. }
            | SsaOp::FloatToInt { dest, .. }
            | SsaOp::IntConv { dest, .. }
            | SsaOp::IntToPtr { dest, .. }
            | SsaOp::PtrToInt { dest, .. }
            | SsaOp::FloatRaise { dest, .. } => self.fold_constant(*dest, op),

            // Operations with Option<SsaVarId> dest that produce unknown results
            SsaOp::Call(call) => {
                if let Some(d) = call.dest {
                    self.drop_value(d);
                }
                // Tracked memory the call may write holds what the call left,
                // not what was stored before it. A call nobody described may
                // write anything; one its host described writes what it says.
                if self.config.track_memory {
                    let accesses = op.stated_call_effects().and_then(CallEffects::accesses);
                    match accesses {
                        None => {
                            self.memory.clear();
                            self.field_dependencies.clear();
                        }
                        Some(accesses) => {
                            for access in accesses.iter().filter(|a| a.access.writes()) {
                                let location = match &access.location {
                                    CallLocation::Field(field) => {
                                        MemoryLocation::StaticField(field.clone())
                                    }
                                    // Resolving a pointee needs the address
                                    // decoding Memory SSA does; unresolved, it
                                    // may be any cell.
                                    CallLocation::Pointee { .. }
                                    | CallLocation::Escaped
                                    | CallLocation::Frame { .. } => MemoryLocation::Unknown,
                                };
                                self.memory.forget(&location);
                            }
                        }
                    }
                }
                None
            }

            // Field payloads capture the store's value, not a mutable SSA source.
            SsaOp::LoadStaticField { dest, field, .. } => {
                self.load_field_value(*dest, MemoryLocation::StaticField(field.clone()))
            }
            SsaOp::StoreStaticField { value, field, .. } => {
                if self.config.track_memory {
                    self.store_field_value(MemoryLocation::StaticField(field.clone()), *value);
                }
                None
            }
            SsaOp::LoadField {
                dest,
                object,
                field,
                ..
            } => {
                self.load_field_value(*dest, MemoryLocation::InstanceField(*object, field.clone()))
            }
            SsaOp::StoreField {
                object,
                field,
                value,
                ..
            } => {
                if self.config.track_memory {
                    self.store_field_value(
                        MemoryLocation::InstanceField(*object, field.clone()),
                        *value,
                    );
                }
                None
            }

            // Address-of-local/arg: the address is taken, meaning external code
            // can write to this local/arg through the pointer. Invalidate the
            // tracked state so subsequent LoadLocal/LoadArg returns Unknown
            // instead of the stale pre-address-taken value.
            //
            // This is critical for patterns like `Monitor.Enter(obj, ref bool)`
            // where the CLR writes `true` to the bool via the by-reference
            // parameter. Without invalidation, the evaluator would see the
            // initial value (false/0) and incorrectly fold branches that check
            // the lock flag.
            SsaOp::LoadLocalAddr {
                dest, local_index, ..
            } => {
                self.drop_value(*dest);
                self.drop_local(*local_index);
                None
            }
            SsaOp::LoadArgAddr {
                dest, arg_index, ..
            } => {
                self.drop_value(*dest);
                self.drop_arg(*arg_index);
                None
            }

            // Operations with SsaVarId dest that produce unknown results
            SsaOp::NewObj { dest, .. }
            | SsaOp::NewArr { dest, .. }
            | SsaOp::LoadElement { dest, .. }
            | SsaOp::LoadIndirect { dest, .. }
            | SsaOp::Box { dest, .. }
            | SsaOp::Unbox { dest, .. }
            | SsaOp::UnboxAny { dest, .. }
            | SsaOp::CastClass { dest, .. }
            | SsaOp::IsInst { dest, .. }
            | SsaOp::ArrayLength { dest, .. }
            | SsaOp::LoadToken { dest, .. }
            | SsaOp::SizeOf { dest, .. }
            | SsaOp::FpClassify { dest, .. }
            | SsaOp::LocalAlloc { dest, .. }
            | SsaOp::LoadFunctionPtr { dest, .. }
            | SsaOp::LoadVirtFunctionPtr { dest, .. }
            | SsaOp::LoadFieldAddr { dest, .. }
            | SsaOp::LoadStaticFieldAddr { dest, .. }
            | SsaOp::LoadElementAddr { dest, .. }
            | SsaOp::LoadObj { dest, .. } => {
                self.drop_value(*dest);
                None
            }

            // Everything else is not modelled: whatever it defines is unknown.
            // Operations without results (stores, branches) define nothing.
            _ => {
                for def in op.defs() {
                    self.drop_value(def);
                }
                None
            }
        }
    }

    /// Folds `op` to a constant through [`Machine::evaluate`] and stores the
    /// result in `dest`, or forgets `dest` when the operands are not all
    /// concrete or the fold declines.
    ///
    /// # Arguments
    ///
    /// * `dest` - The destination variable of `op`.
    /// * `op` - The operation to fold.
    ///
    /// # Returns
    ///
    /// The folded constant as an expression, or `None` when it is unknown.
    fn fold_constant(&mut self, dest: SsaVarId, op: &SsaOp<T>) -> Option<()> {
        let folded = self.machine.evaluate_with_operand_layouts(
            op,
            Destination::of_operation(self.ssa, op),
            |var| {
                self.values
                    .get(&var)
                    .and_then(|expr| expr.as_constant().cloned())
            },
            |var| OperandLayout::of(self.ssa, var),
        );
        if let Some(value) = folded {
            self.put_value(dest, SymbolicExpr::constant(value));
            Some(())
        } else {
            self.drop_value(dest);
            None
        }
    }

    /// Folds an unrepresented arithmetic domain when concrete, otherwise retaining identity.
    ///
    /// Admitted scalar domains use the captured descriptor before this fallback.
    /// An operand with no tracked value leaves the destination untouched for
    /// the caller to forget.
    fn fold_or_opaque(
        &mut self,
        dest: SsaVarId,
        operands: &[SsaVarId],
        source: &SsaOp<T>,
    ) -> Option<()> {
        let mut constant = true;
        for operand in operands {
            constant &= self.values.get(operand)?.as_constant().is_some();
        }
        if constant {
            return self.fold_constant(dest, source);
        }
        // Noninteger or incomplete domains reach this fallback only after the
        // scalar admission path has declined. Preserve a fresh result identity.
        self.opaque_result(dest)
    }

    /// Reads a constant at its comparison input's attested numeric layout.
    fn comparison_constant<'v>(
        &self,
        variable: SsaVarId,
        value: &'v ConstValue<T>,
    ) -> Option<Cow<'v, ConstValue<T>>> {
        OperandLayout::of(self.ssa, variable)
            .read_comparison_ref(value, self.machine.ptr_size)
            .filter(|value| comparison_stable(value.as_ref(), self.machine.float))
    }

    /// Keeps a folded constant only when it fits `dest`'s declared width.
    ///
    /// Operands of different widths fold to the wider one's width, which is the
    /// machine's result only when the destination is that wide; see
    /// [`Machine::evaluate`]. A constant that does not fit becomes the opaque
    /// value of `dest` instead, so it cannot decide a branch the machine would
    /// decide differently.
    fn fit_to_destination(
        &mut self,
        dest: SsaVarId,
        result: SymbolicExpr<T>,
    ) -> Option<SymbolicExpr<T>> {
        let dest_width = self.ssa.bit_width(dest);
        match result.as_constant() {
            Some(value) if !self.machine.fits(value, dest_width) => {
                self.fresh_instance(dest).map(SymbolicExpr::Instance)
            }
            _ => Some(result),
        }
    }

    /// Updates local/arg state when a variable with Local(N) or Argument(N)
    /// origin receives a value. This enables `LoadLocal`/`LoadArg` to read
    /// the most recently written value for the corresponding CIL local/arg.
    fn track_origin_state(&mut self, var: SsaVarId, value: &SymbolicExpr<T>) {
        if let Some(ssa_var) = self.ssa.variable(var) {
            match ssa_var.origin() {
                VariableOrigin::Local(idx) => {
                    self.put_local(idx, value.clone());
                }
                VariableOrigin::Argument(idx) => {
                    self.put_arg(idx, value.clone());
                }
                _ => {}
            }
        }
    }

    /// Tries to resolve a variable's value by tracing back through its definition.
    ///
    /// This is useful when a variable's value depends on computations that have
    /// not been evaluated yet. It recursively builds conditional value relations
    /// without asserting CFG reachability or changing the ordered execution prefix.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to resolve.
    /// * `max_depth` - Maximum recursion depth to prevent infinite loops.
    ///
    /// # Returns
    ///
    /// The resolved expression, or `None` if it cannot be determined within
    /// `max_depth`.
    pub fn resolve_with_trace(
        &mut self,
        var: SsaVarId,
        max_depth: usize,
    ) -> Option<SymbolicExpr<T>> {
        // Already known?
        if let Some(v) = self.values.get(&var) {
            return Some(v.clone());
        }

        if max_depth == 0 {
            return None;
        }

        // Find the definition of this variable.
        //
        // Scan-free lookup, and `None` - one unresolved leaf in the symbolic
        // tree - is the failure policy. This recurses through operands under
        // `max_depth`, so a recovering lookup would multiply a whole-function
        // scan by the traversal, and it bottoms out on arguments, whose entry
        // def site has no instruction index for the scan to confirm. A phi is
        // unknown here: without path context there is no single incoming op.
        let op = self.ssa.recorded_definition(var)?.op();

        // Recursively resolve operands first
        for operand in op.uses() {
            if !self.values.contains_key(&operand)
                && let Some(resolved) =
                    self.resolve_with_trace(operand, max_depth.saturating_sub(1))
            {
                self.put_value(operand, resolved);
            }
        }

        // Now evaluate this operation
        let layout = OperationLayout::of(self.ssa, op, self.machine);
        self.apply_normal(op, layout);
        self.values.get(&var).cloned()
    }

    /// Tries to evaluate a variable by tracing back through its definition.
    ///
    /// Alias for [`resolve_with_trace`](Self::resolve_with_trace) that returns
    /// `Option<i64>` for API compatibility.
    pub fn evaluate_with_trace(&mut self, var: SsaVarId, max_depth: usize) -> Option<i64> {
        self.resolve_with_trace(var, max_depth)
            .and_then(|e| e.as_i64())
    }

    /// Determines the next block to execute based on the terminator of the given block.
    ///
    /// Returns the successor conditional on reaching the block terminator. This
    /// query does not prove that the evaluated prefix completes normally; use
    /// [`Self::execution_outcome`] or [`Self::execute`] for that guarantee.
    ///
    /// # Returns
    ///
    /// - `ControlFlow::Continue(block)` - Continue to the specified block
    /// - `ControlFlow::Terminal` - No successor (return, throw, etc.)
    /// - `ControlFlow::Unknown` - Cannot determine (condition is unknown/symbolic)
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{
    ///     analysis::{ControlFlow, SsaEvaluator},
    ///     testing,
    /// };
    ///
    /// // Block 0 branches on a constant `true` to block 1; block 3 returns.
    /// let ssa = testing::diamond_phi_fixture();
    /// let mut eval = SsaEvaluator::new(&ssa, testing::MACHINE);
    /// eval.evaluate_block(0);
    ///
    /// // The condition folds to a constant, so the taken edge is known.
    /// assert_eq!(eval.next_block(0), ControlFlow::Continue(1));
    ///
    /// // The merge block ends in a return, so it has no successor.
    /// assert_eq!(eval.next_block(3), ControlFlow::Terminal);
    ///
    /// // A block that was never evaluated has an undetermined condition.
    /// let ssa = testing::loop_counter_fixture();
    /// let eval = SsaEvaluator::new(&ssa, testing::MACHINE);
    /// assert_eq!(eval.next_block(1), ControlFlow::Unknown);
    /// ```
    #[must_use]
    pub fn next_block(&self, block_idx: usize) -> ControlFlow {
        let Some(block) = self.ssa.block(block_idx) else {
            return ControlFlow::Unknown;
        };

        // Find the terminating instruction
        let terminator = block
            .instructions()
            .iter()
            .rev()
            .find(|instr| instr.op().is_terminator());

        let Some(instr) = terminator else {
            // No terminator - fall through to next block if it exists
            let next_idx = block_idx.saturating_add(1);
            if next_idx < self.ssa.block_count() {
                return ControlFlow::Continue(next_idx);
            }
            return ControlFlow::Unknown;
        };

        self.evaluate_control_flow(instr.op())
    }

    /// Resolves a terminator into its successor block using currently known
    /// values.
    ///
    /// Backs [`next_block`](Self::next_block). Unconditional jumps yield their
    /// target directly; conditional terminators are decided only when the
    /// deciding value is a known constant. A `Branch` reads its condition's
    /// truthiness (non-zero and non-null are true, per CIL), a `BranchCmp`
    /// compares its two operands via
    /// [`evaluate_comparison`](Self::evaluate_comparison), and a
    /// `Switch` uses the concrete index, falling to the default when it is out
    /// of range. Anything whose deciding value is symbolic or unknown returns
    /// `Unknown`, and the various return and throw forms are `Terminal`.
    ///
    /// # Arguments
    ///
    /// * `op` - The terminator operation to resolve.
    ///
    /// # Returns
    ///
    /// The [`ControlFlow`] outcome: a concrete successor, `Terminal`, or
    /// `Unknown`.
    fn evaluate_control_flow(&self, op: &SsaOp<T>) -> ControlFlow {
        match op {
            // Unconditional jumps
            SsaOp::Jump { target } | SsaOp::Leave { target } => ControlFlow::Continue(*target),

            // Conditional branch (bool condition)
            SsaOp::Branch {
                condition,
                true_target,
                false_target,
            } => match self.get_concrete(*condition).and_then(ConstValue::as_bool) {
                // Non-zero (and non-null) is true in CIL.
                Some(true) => ControlFlow::Continue(*true_target),
                Some(false) => ControlFlow::Continue(*false_target),
                None => ControlFlow::Unknown,
            },

            // Compare and branch
            SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            } => {
                let left_val = self.get_concrete(*left);
                let right_val = self.get_concrete(*right);

                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        let (Some(l), Some(r)) = (
                            self.comparison_constant(*left, l),
                            self.comparison_constant(*right, r),
                        ) else {
                            return ControlFlow::Unknown;
                        };
                        match self.machine.compare_values(&l, &r, *cmp, *unsigned) {
                            Some(true) => ControlFlow::Continue(*true_target),
                            Some(false) => ControlFlow::Continue(*false_target),
                            None => ControlFlow::Unknown,
                        }
                    }
                    _ => ControlFlow::Unknown,
                }
            }

            // Switch - needs a non-negative integer index
            SsaOp::Switch {
                value,
                targets,
                default,
            } => match self.get_concrete(*value).and_then(ConstValue::as_u64) {
                Some(v) => {
                    #[allow(clippy::cast_possible_truncation)]
                    let idx = v as usize;
                    if let Some(&target) = targets.get(idx) {
                        ControlFlow::Continue(target)
                    } else {
                        ControlFlow::Continue(*default)
                    }
                }
                None => ControlFlow::Unknown,
            },

            // Terminal instructions
            SsaOp::Return { .. }
            | SsaOp::Throw { .. }
            | SsaOp::Rethrow
            | SsaOp::EndFinally
            | SsaOp::EndFilter { .. }
            | SsaOp::InterruptReturn
            | SsaOp::Unreachable => ControlFlow::Terminal,

            // Not a control flow operation
            _ => ControlFlow::Unknown,
        }
    }

    /// Evaluates a comparison between two typed constant values.
    ///
    /// Uses the typed comparison methods on `ConstValue`, which apply the
    /// correct signedness from the operand types; the `unsigned` flag selects
    /// the unsigned form for the ordered relations.
    ///
    /// For floats the flag means "unordered or", as it does for CIL's `.un`
    /// branches: with a NaN operand every relation is false when the flag is
    /// clear and true when it is set. `Le` and `Ge` are therefore computed
    /// directly on floats rather than as the negation of `Gt` and `Lt`, whose
    /// complement holds only over a total order. On integers they are the
    /// negation of `Gt` and `Lt` at the same signedness.
    ///
    /// # Arguments
    ///
    /// * `left` - The left-hand constant.
    /// * `right` - The right-hand constant.
    /// * `cmp` - The comparison relation to apply.
    /// * `unsigned` - Whether to use unsigned (integer) or unordered (float)
    ///   semantics for `Lt`, `Le`, `Gt`, and `Ge`.
    /// * `ptr_size` - Target pointer size, the width of the native integers.
    ///
    /// # Returns
    ///
    /// Whether the comparison holds, or `None` when the pair is not comparable
    /// (unrelated constant kinds or vector shapes without a scalar ordering),
    /// in which case the outcome is unknown.
    #[must_use]
    pub fn evaluate_comparison(
        left: &ConstValue<T>,
        right: &ConstValue<T>,
        cmp: CmpKind,
        unsigned: bool,
        ptr_size: PointerSize,
    ) -> Option<bool> {
        if let (ConstValue::Float(l), ConstValue::Float(r)) = (left, right)
            && l.format() == r.format()
            && matches!(cmp, CmpKind::Le | CmpKind::Ge)
        {
            return Some(match l.compare(*r) {
                None => unsigned,
                Some(order) if cmp == CmpKind::Le => order != core::cmp::Ordering::Greater,
                Some(order) => order != core::cmp::Ordering::Less,
            });
        }
        left.relation(right, cmp, unsigned, ptr_size)
    }

    /// Executes the SSA function starting from a given block and records the trace.
    ///
    /// This method steps through the SSA, evaluating each block and following
    /// control flow decisions based on computed values. It records the sequence
    /// of blocks visited and optionally captures state values at each step.
    ///
    /// # Arguments
    ///
    /// * `start_block` - The block to start execution from
    /// * `state_var` - Optional variable to capture state values (for CFF analysis)
    /// * `max_steps` - Maximum number of blocks to visit (prevents infinite loops)
    ///
    /// # Returns
    ///
    /// An [`ExecutionTrace`] containing the visited blocks and state values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{
    ///     analysis::SsaEvaluator,
    ///     ir::{value::ConstValue, SsaVarId},
    ///     testing,
    /// };
    ///
    /// // A loop that increments a counter until it reaches 4.
    /// let ssa = testing::loop_counter_fixture();
    /// let mut eval = SsaEvaluator::new(&ssa, testing::MACHINE);
    ///
    /// let state_var = SsaVarId::from_index(1);
    /// eval.set_concrete(state_var, ConstValue::I32(0));
    ///
    /// let trace = eval.execute(0, Some(state_var), 1000);
    ///
    /// // Execution enters the loop header, revisits it until the counter
    /// // reaches the limit, then falls through to the exit block.
    /// assert_eq!(trace.blocks(), &[0, 1, 1, 1, 1, 2]);
    ///
    /// // The captured state values follow the counter as it increments.
    /// assert_eq!(trace.states().last(), Some(&Some(ConstValue::I32(3))));
    /// assert_eq!(trace.blocks().len(), trace.states().len());
    /// ```
    pub fn execute(
        &mut self,
        start_block: usize,
        state_var: Option<SsaVarId>,
        max_steps: usize,
    ) -> ExecutionTrace<T> {
        let mut trace = ExecutionTrace::new(max_steps);
        let mut current_block = start_block;

        loop {
            // Check if we've hit the limit
            if trace.hit_limit() {
                break;
            }

            // Record the current state before evaluation
            let state = state_var.and_then(|v| self.get_concrete(v).cloned());
            trace.record_block(current_block, state);

            // Set predecessor for phi evaluation
            if let Some(prev) = trace.blocks().iter().rev().nth(1) {
                self.set_predecessor(Some(*prev));
            }

            // Evaluate the block
            self.evaluate_block(current_block);
            if !matches!(self.execution, Evaluation::Value(())) {
                trace.outcome = self.execution.clone();
                break;
            }

            // Determine next block
            match self.next_block(current_block) {
                ControlFlow::Continue(next) => {
                    current_block = next;
                }
                ControlFlow::Terminal => {
                    let returns_normally = self
                        .ssa
                        .block(current_block)
                        .and_then(|block| {
                            block
                                .instructions()
                                .iter()
                                .rev()
                                .find(|instruction| instruction.op().is_terminator())
                        })
                        .is_some_and(|instruction| {
                            matches!(instruction.op(), SsaOp::Return { .. })
                        });
                    if returns_normally {
                        trace.mark_complete();
                    } else {
                        // EH continuation, unreachable and machine-specific exits
                        // do not establish a normal function return here.
                        self.execution = Evaluation::Unsupported;
                        trace.outcome = Evaluation::Unsupported;
                    }
                    break;
                }
                ControlFlow::Unknown => {
                    // Can't determine next block - stop execution
                    break;
                }
            }
        }

        trace
    }

    /// Executes starting from block 0 with default settings.
    ///
    /// This is a convenience method for simple cases where you want to execute
    /// from the entry block without state tracking.
    pub fn execute_from_entry(&mut self, max_steps: usize) -> ExecutionTrace<T> {
        self.execute(0, None, max_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            CallData, CallInputKey, Callee, block::SsaBlock, evaluation::ArithmeticTrap,
            instruction::SsaInstruction, variable::DefSite,
        },
        testing::{MACHINE, MockTarget, MockType, const_i32_return},
    };

    /// Current bindings roll back while future definitions never reuse a speculative token.
    #[test]
    fn unknown_conversion_normality_controls_trace_completion() {
        for checked in [false, true] {
            let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
            let input = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            );
            let narrow = ssa.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I8,
            );
            let returned = ssa.create_variable(
                VariableOrigin::Local(1),
                0,
                DefSite::instruction(0, 1),
                MockType::I32,
            );
            let mut block = SsaBlock::new(0);
            for op in [
                SsaOp::IntConv {
                    dest: narrow,
                    operand: input,
                    target: MockType::I8,
                    unsigned: false,
                    overflow_check: checked,
                },
                SsaOp::Const {
                    dest: returned,
                    value: ConstValue::I32(7),
                },
                SsaOp::returns(Some(returned)),
            ] {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            ssa.add_block(block);
            ssa.recompute_uses();
            let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
            evaluator.set_symbolic(input, "input");
            let trace = evaluator.execute(0, None, 10);
            assert_eq!(trace.is_complete(), !checked);
            assert_eq!(
                trace.outcome(),
                if checked {
                    &Evaluation::Unknown
                } else {
                    &Evaluation::Value(())
                }
            );
            assert_eq!(evaluator.next_block(0), ControlFlow::Terminal);
        }
    }

    #[test]
    fn nonreturn_terminal_does_not_prove_normal_function_completion() {
        let result = SsaVarId::from_index(0);
        for terminal in [
            SsaOp::<MockTarget>::Unreachable,
            SsaOp::EndFinally,
            SsaOp::EndFilter { result },
            SsaOp::InterruptReturn,
        ] {
            let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
            let declared = ssa.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            );
            assert_eq!(declared, result);
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: result,
                value: ConstValue::I32(0),
            }));
            block.add_instruction(SsaInstruction::synthetic(terminal));
            ssa.add_block(block);
            let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
            let trace = evaluator.execute(0, None, 10);
            assert!(!trace.is_complete());
            assert_eq!(trace.outcome(), &Evaluation::Unsupported);
        }
    }

    #[test]
    fn unsupported_terminal_halts_ordered_prefix_without_changing_successor_queries() {
        let ssa = const_i32_return(7);
        let later = SsaVarId::from_index(0);
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        let mark = evaluator.checkpoint();
        assert_eq!(
            evaluator.evaluate_op_outcome(&SsaOp::Unreachable),
            Evaluation::Unsupported
        );
        assert_eq!(evaluator.execution_outcome(), &Evaluation::Unsupported);
        evaluator.evaluate_block(0);
        assert!(evaluator.get(later).is_none());
        assert_eq!(
            evaluator.evaluate_op_outcome(&SsaOp::Const {
                dest: later,
                value: ConstValue::I32(7)
            }),
            Evaluation::Unsupported
        );
        assert_eq!(evaluator.next_block(0), ControlFlow::Terminal);
        evaluator.rollback(mark);
        assert_eq!(evaluator.execution_outcome(), &Evaluation::Value(()));
        assert!(evaluator.execute(0, None, 10).is_complete());
    }

    #[test]
    fn backward_fault_queries_do_not_mutate_ordered_execution_state() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let returned = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let left = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let right = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(1, 1),
            MockType::I32,
        );
        let dead = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(1, 2),
            MockType::I32,
        );
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: returned,
            value: ConstValue::I32(7),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(returned))));
        let mut unreachable = SsaBlock::new(1);
        for op in [
            SsaOp::Const {
                dest: left,
                value: ConstValue::I32(1),
            },
            SsaOp::Const {
                dest: right,
                value: ConstValue::I32(0),
            },
            SsaOp::Div {
                dest: dead,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            SsaOp::returns(Some(dead)),
        ] {
            unreachable.add_instruction(SsaInstruction::synthetic(op));
        }
        ssa.add_block(entry);
        ssa.add_block(unreachable);
        ssa.recompute_uses();
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        let relation = evaluator.resolve_with_trace(dead, 4).unwrap();
        assert_eq!(
            relation.evaluate_outcome_with(PointerSize::Bit64, |_| Evaluation::Unknown),
            Evaluation::Trap(ArithmeticTrap::DivideByZero)
        );
        assert_eq!(evaluator.execution_outcome(), &Evaluation::Value(()));
        assert!(evaluator.execute(0, None, 10).is_complete());
        evaluator.evaluate_op(&SsaOp::Div {
            dest: dead,
            left,
            right,
            unsigned: false,
            flags: None,
        });
        assert_eq!(
            evaluator.execution_outcome(),
            &Evaluation::Trap(ArithmeticTrap::DivideByZero)
        );
        evaluator.resolve_with_trace(returned, 4);
        assert_eq!(
            evaluator.execution_outcome(),
            &Evaluation::Trap(ArithmeticTrap::DivideByZero)
        );
        assert!(!evaluator.execute(0, None, 10).is_complete());
    }

    #[test]
    fn unused_fault_blocks_normal_trace_completion_and_rolls_back() {
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
        let returned = variable(3);
        let mut block = SsaBlock::new(0);
        for op in [
            SsaOp::Const {
                dest: left,
                value: ConstValue::I32(1),
            },
            SsaOp::Const {
                dest: right,
                value: ConstValue::I32(0),
            },
            SsaOp::Div {
                dest: dead,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            SsaOp::Const {
                dest: returned,
                value: ConstValue::I32(7),
            },
            SsaOp::returns(Some(returned)),
        ] {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        ssa.add_block(block);
        ssa.recompute_uses();
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        let mark = evaluator.checkpoint();
        let trace = evaluator.execute(0, None, 10);
        assert!(!trace.is_complete());
        assert_eq!(
            trace.outcome(),
            &Evaluation::Trap(ArithmeticTrap::DivideByZero)
        );
        assert!(evaluator.get(dead).is_none());
        assert!(evaluator.get(returned).is_none());
        assert_eq!(evaluator.next_block(0), ControlFlow::Terminal);
        evaluator.rollback(mark);
        assert_eq!(evaluator.execution_outcome(), &Evaluation::Value(()));
        evaluator.evaluate_op(&SsaOp::Const {
            dest: returned,
            value: ConstValue::I32(7),
        });
        assert_eq!(evaluator.get_concrete(returned), Some(&ConstValue::I32(7)));
    }

    /// **A call forgets what was known of every value it defines.** A value a
    /// loop's earlier trip left in a call's result, or in the value it returns
    /// beside it (NSIS's `GetIntFromParm` hands back a character in `edx`), is
    /// not what the call defines now.
    #[test]
    fn a_call_forgets_its_result_and_what_it_returns_beside_it() {
        let ssa = const_i32_return(7);
        let (result, beside) = (SsaVarId::from_index(0), SsaVarId::from_index(1));
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        for value in [result, beside] {
            evaluator.evaluate_op(&SsaOp::Const {
                dest: value,
                value: ConstValue::I32(3),
            });
        }
        evaluator.evaluate_op(&SsaOp::from(
            CallData::new(Some(result), Callee::Direct(9), vec![])
                .with_result(CallInputKey::new(2), beside),
        ));
        for value in [result, beside] {
            assert_eq!(evaluator.get_concrete(value), None, "{value}");
        }
    }

    #[test]
    fn unknown_checked_arithmetic_does_not_establish_normal_reachability() {
        let ssa = const_i32_return(7);
        let value = SsaVarId::from_index(0);
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.set_symbolic(value, "input");
        let mark = evaluator.checkpoint();
        evaluator.evaluate_op(&SsaOp::AddOvf {
            dest: value,
            left: value,
            right: value,
            unsigned: false,
            flags: None,
        });
        assert_eq!(evaluator.execution_outcome(), &Evaluation::Unknown);
        assert_eq!(evaluator.next_block(0), ControlFlow::Terminal);
        evaluator.rollback(mark);
        assert_eq!(evaluator.execution_outcome(), &Evaluation::Value(()));
    }

    #[test]
    fn opaque_bindings_rollback_without_reusing_generations() {
        let ssa = const_i32_return(1);
        let var = SsaVarId::from_index(0);
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.set_unknown(var);
        let first = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        let mark = evaluator.checkpoint();
        evaluator.set_unknown(var);
        let second = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        assert_ne!(first, second);
        evaluator.rollback(mark);
        assert_eq!(
            evaluator.current_symbol(var).unwrap().as_instance(),
            Some(first)
        );
        evaluator.set_unknown(var);
        let third = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        assert!(third.generation() > second.generation());
        assert!(evaluator.get(var).is_none());
    }

    /// Reset and copied continuation states preserve opaque identity scope without shared mutation.
    #[test]
    fn opaque_bindings_reset_and_clone_keep_contexts_separate() {
        let ssa = const_i32_return(1);
        let var = SsaVarId::from_index(0);
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        assert_eq!(
            evaluator.current_symbol(var).unwrap().as_variable(),
            Some(var)
        );
        evaluator.set_unknown(var);
        let first = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        let mut fork = evaluator.clone();
        assert_eq!(fork.current_symbol(var).unwrap().as_instance(), Some(first));
        fork.set_unknown(var);
        assert_ne!(fork.current_symbol(var).unwrap().as_instance(), Some(first));
        assert_eq!(
            evaluator.current_symbol(var).unwrap().as_instance(),
            Some(first)
        );
        let mark = evaluator.checkpoint();
        evaluator.clear();
        let after_clear = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        assert_ne!(after_clear, first);
        evaluator.rollback(mark);
        assert_eq!(
            evaluator.current_symbol(var).unwrap().as_instance(),
            Some(first)
        );
        evaluator.clear();
        let later = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        assert!(later.generation() > after_clear.generation());
    }

    /// Generation exhaustion withholds new identities while preserving known values and old snapshots.
    #[test]
    fn opaque_generation_exhaustion_refuses_new_identities() {
        let ssa = const_i32_return(1);
        let var = SsaVarId::from_index(0);
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.next_instance = u64::MAX - 1;
        evaluator.set_unknown(var);
        let final_token = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        assert_eq!(final_token.generation(), u64::MAX);
        let mark = evaluator.checkpoint();
        evaluator.set_unknown(var);
        assert!(evaluator.current_symbol(var).is_none());
        evaluator.rollback(mark);
        assert_eq!(
            evaluator.current_symbol(var).unwrap().as_instance(),
            Some(final_token)
        );
        evaluator.set_unknown(var);
        assert!(evaluator.current_symbol(var).is_none());
        evaluator.clear();
        assert!(evaluator.current_symbol(var).is_none());
        evaluator.set_concrete(var, ConstValue::I32(7));
        assert_eq!(evaluator.current_symbol(var).unwrap().as_i64(), Some(7));
    }

    /// Opaque result, carrier refusal and loop widening boundaries allocate distinct instances.
    #[test]
    fn opaque_refusals_and_loop_widening_use_fresh_instances() {
        let ssa = const_i32_return(1);
        let var = SsaVarId::from_index(0);
        let mut evaluator = SsaEvaluator::<MockTarget>::new(&ssa, MACHINE);
        evaluator.opaque_result(var).unwrap();
        let first = evaluator.get(var).unwrap().as_instance().unwrap();
        let second = evaluator
            .fit_to_destination(var, SymbolicExpr::constant_i64(1))
            .unwrap()
            .as_instance()
            .unwrap();
        assert_ne!(first, second);
        evaluator.set_symbolic(var, "first");
        evaluator.widen_unstable_values(&[0]);
        let third = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        evaluator.set_symbolic(var, "second");
        evaluator.widen_unstable_values(&[0]);
        let fourth = evaluator
            .current_symbol(var)
            .unwrap()
            .as_instance()
            .unwrap();
        assert_ne!(third, fourth);
        assert!(fourth.generation() > second.generation());
    }
}
