//! Opaque predicate detection and removal pass.
//!
//! Opaque predicates are conditional expressions that always evaluate to the same
//! value at runtime, but appear complex to static analysis. Obfuscators use them
//! to confuse decompilers and analysis tools.
//!
//! # Detection Strategies
//!
//! ## Basic Patterns
//! - **Self-comparison**: `x == x`, `x != x`, `x < x`, `x > x`
//! - **Identity operations**: `x ^ x == 0`, `x - x == 0`
//! - **Zero operations**: `x * 0`, `x & 0`, `x % 1`
//!
//! ## Number-Theoretic Predicates
//! - **Consecutive integers**: `(x * (x + 1)) % 2 == 0` (always true)
//! - **Square properties**: `x² >= 0` (always true for integers)
//! - **Modular arithmetic**: `(x² - x) % 2 == 0` (always true)
//!
//! ## Type-Based Predicates
//! - **Null checks**: `obj != null` after `newobj`, and an image object's
//!   address against zero (always true); an import's address may be null
//! - **Array length**: `arr.Length >= 0` (always true)
//!
//! ## Range-Based Predicates
//! - **Unsigned bounds**: `unsigned_x >= 0` (always true)
//! - **Correlated conditions**: a block entered along one edge re-testing
//!   the condition that edge already decided - the same flags value, the same
//!   comparison over the same operands, or the same condition variable -
//!   as in `jz L; jnz L`. Relations over *different* operands
//!   (`if (x > 5) { if (x < 3) { dead } }`) are not correlated by this rule.
//!
//! # Example
//!
//! Before:
//! ```text
//! v0 = 5
//! v1 = ceq v0, v0    // Always true
//! branch v1, B1, B2  // Always goes to B1
//! ```
//!
//! After:
//! ```text
//! v0 = 5
//! v1 = true
//! jump B1
//! ```

use std::{collections::BTreeMap, marker::PhantomData, ops::Deref};

use crate::{
    Result,
    analysis::{
        SsaCfg,
        address::AddressWalk,
        defuse::DefUseIndex,
        evaluator::SsaEvaluator,
        guards::{Definitions, GuardFacts, JoinSelect},
        range::ValueRange,
    },
    events::{EventKind, EventLog},
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        instruction::SsaInstruction,
        ops::{SsaOp, kinds::CmpKind, native::FlagCondition},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
        varstore::VarSet,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// The result of analyzing a potential opaque predicate expression.
///
/// This tri-state result captures whether a comparison or branch condition
/// can be statically resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateResult {
    /// The predicate always evaluates to a non-zero (true) value at runtime.
    AlwaysTrue,
    /// The predicate always evaluates to zero (false) at runtime.
    AlwaysFalse,
    /// The predicate's value cannot be statically determined.
    Unknown,
}

impl PredicateResult {
    /// Converts to an optional boolean.
    ///
    /// # Returns
    ///
    /// `Some(true)` for `AlwaysTrue`, `Some(false)` for `AlwaysFalse`, `None` for `Unknown`.
    #[must_use]
    pub fn as_bool(self) -> Option<bool> {
        match self {
            Self::AlwaysTrue => Some(true),
            Self::AlwaysFalse => Some(false),
            Self::Unknown => None,
        }
    }

    /// Negates the predicate result.
    ///
    /// `AlwaysTrue` becomes `AlwaysFalse` and vice versa. `Unknown` stays `Unknown`.
    #[must_use]
    pub fn negate(self) -> Self {
        match self {
            Self::AlwaysTrue => Self::AlwaysFalse,
            Self::AlwaysFalse => Self::AlwaysTrue,
            Self::Unknown => Self::Unknown,
        }
    }
}

/// Result of analyzing a comparison for algebraic simplification.
///
/// Unlike `PredicateResult` which determines if a comparison is always true/false,
/// this enum represents transformations that simplify comparisons while preserving
/// their runtime behavior.
#[derive(Debug, Clone)]
enum ComparisonSimplification<T: Target> {
    /// Replace with a simpler comparison operation.
    SimplerOp {
        new_op: SsaOp<T>,
        reason: &'static str,
    },
    /// Replace with a copy of another variable (e.g., `(cmp) == 1` → `cmp`).
    Copy {
        dest: SsaVarId,
        src: SsaVarId,
        reason: &'static str,
    },
}

/// Cached definition information for efficient predicate analysis.
///
/// The [`Definitions`] the guard facts read, augmented with specialized
/// tracking: phi-defined variables, non-null provenance, and computed
/// [`ValueRange`]s.
struct DefinitionCache<T: Target> {
    /// Each variable's defining operation, float-ness and width.
    definitions: Definitions<T>,
    /// Variables defined by phi nodes (not covered by `DefUseIndex`).
    phi_defs: VarSet,
    /// Variables known to be non-null (produced by `NewObj`, `NewArr`, `Box`,
    /// `LoadToken`, or an image object's address, a symbol constant the
    /// target places, [`Target::symbol_address`], or one the host proves
    /// bound, [`Target::symbol_bound`]).
    non_null_vars: VarSet,
    /// Computed [`ValueRange`]s for each variable. Constants get exact
    /// ranges, array lengths get non-negative ranges.
    ranges: BTreeMap<SsaVarId, ValueRange>,
}

/// The cache reads its definitions as the guard facts do.
impl<T: Target> Deref for DefinitionCache<T> {
    type Target = Definitions<T>;

    fn deref(&self) -> &Definitions<T> {
        &self.definitions
    }
}

impl<T: Target> DefinitionCache<T> {
    /// Builds the definition cache from an SSA function.
    ///
    /// Performs a single pass over all blocks to populate:
    /// - `definitions`: [`Definitions::of`] the function.
    /// - `phi_defs`: bitset of variables defined by phi nodes (not tracked by `DefUseIndex`).
    /// - `non_null_vars`: bitset of variables produced by `NewObj`, `NewArr`, `Box`, or `LoadToken`.
    /// - `ranges`: [`ValueRange::constant`] for `Const` ops, [`ValueRange::non_negative`] for
    ///   `ArrayLength` ops.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to build the cache from.
    ///
    /// # Returns
    ///
    /// A fully populated `DefinitionCache` with definition index, phi-def tracking,
    /// non-null provenance, value ranges, and the possibly-float set.
    fn build(ssa: &SsaFunction<T>) -> Self {
        // Every set here is indexed by an id read out of the IR, and a pass is
        // not the place a malformed one is discovered. `VarSet` and `VarMap`
        // are total over any id: one past the bound costs an entry, not a table.
        let bound = ssa.var_id_bound();
        let mut phi_defs = VarSet::new(bound);
        let mut non_null_vars = VarSet::new(bound);
        let mut ranges = BTreeMap::new();

        for (_block_idx, block) in ssa.iter_blocks() {
            // Process phi nodes (not covered by DefUseIndex)
            for phi in block.phi_nodes() {
                phi_defs.insert(phi.result());
            }

            // Process instructions for specialized tracking
            for instr in block.instructions() {
                let op = instr.op();
                if let Some(dest) = op.dest() {
                    // Track non-null producing operations and value ranges
                    match op {
                        SsaOp::NewObj { .. }
                        | SsaOp::NewArr { .. }
                        | SsaOp::Box { .. }
                        | SsaOp::LoadToken { .. } => {
                            // Non-null tracked separately (not a numeric range)
                            non_null_vars.insert(dest);
                        }
                        // An image object's address is never the null pointer:
                        // no loader maps a module at address zero. A symbol the
                        // image does not place (an import, which a weak one
                        // leaves null where nothing defines it) may be, unless
                        // the host proves the loader binds it.
                        SsaOp::Const {
                            value: ConstValue::Symbol(symbol),
                            ..
                        } if T::symbol_address(symbol).is_some()
                            || T::symbol_bound(ssa.facts(), symbol) =>
                        {
                            non_null_vars.insert(dest);
                        }
                        SsaOp::ArrayLength { .. } => {
                            ranges.insert(dest, ValueRange::non_negative());
                        }
                        SsaOp::Const { value, .. } => {
                            if let Some(v) = value.as_i64() {
                                ranges.insert(dest, ValueRange::constant(v));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Self {
            definitions: Definitions::of(ssa),
            phi_defs,
            non_null_vars,
            ranges,
        }
    }

    /// Resolves `var` through the definitions its low bit survives: copies,
    /// integer conversions, and the integer/pointer reinterpretations.
    ///
    /// A lift puts a conversion between a computation and the test that reads
    /// it - `(uint32_t)(v * (v + 1)) & 1` - so a detector matching on the
    /// defining op sees `IntConv` where the arithmetic is.
    ///
    /// Every fact this pass derives through the peel is about bit 0 or is an
    /// exact `0` or `1`: a value's parity, `(even) & 1 == 0`,
    /// `(odd) & 1 == 1`. Truncation keeps bit 0, and so do zero- and
    /// sign-extension; `0` converts to `0` under all three. `1` converts to
    /// `1` too, except sign-extended from a single bit, where that bit is the
    /// sign and `1` becomes `-1`. That one conversion ends the peel.
    ///
    /// Unlike the identity folds in [`crate::analysis::mba`], which need the
    /// whole value and therefore peel only same-width conversions, nothing here
    /// depends on the bits above bit 0.
    fn peeled(&self, var: SsaVarId) -> SsaVarId {
        let mut var = var;
        for _ in 0..8 {
            match self.get_definition(var) {
                Some(SsaOp::Copy { src, .. }) => var = *src,
                Some(SsaOp::IntConv {
                    operand, unsigned, ..
                }) if *unsigned || self.bit_width(*operand) != Some(1) => var = *operand,
                // A `lea` computing integer arithmetic arrives as pointer ops,
                // so the integer underneath is behind a domain change rather
                // than a width change. The bits are the same either way, which
                // is all the width-invariant rules here ask of it.
                Some(SsaOp::PtrToInt { operand, .. }) => var = *operand,
                Some(SsaOp::IntToPtr { operand, .. }) => var = *operand,
                _ => break,
            }
        }
        var
    }

    /// Checks whether a variable is defined by a phi node.
    ///
    /// # Returns
    ///
    /// `true` if `var` is the result of a phi node, `false` otherwise.
    fn is_phi_defined(&self, var: SsaVarId) -> bool {
        self.phi_defs.contains(var)
    }

    /// Checks whether a variable is known to be non-null.
    ///
    /// # Returns
    ///
    /// `true` if `var` was produced by a `NewObj`, `NewArr`, `Box`, or `LoadToken`
    /// operation, `false` otherwise.
    fn is_non_null(&self, var: SsaVarId) -> bool {
        self.non_null_vars.contains(var)
    }

    /// Looks up the computed value range for a variable.
    ///
    /// # Returns
    ///
    /// The [`ValueRange`] recorded for `var` (an exact range for a `Const`, a
    /// non-negative range for an `ArrayLength`), or `None` if none was computed.
    fn get_range(&self, var: SsaVarId) -> Option<&ValueRange> {
        self.ranges.get(&var)
    }
}

/// Opaque predicate detection and removal pass.
///
/// Detects conditional expressions that always evaluate to the same value
/// (always true or always false) and simplifies branches, comparisons,
/// and phi nodes accordingly. Handles self-comparison, identity operations,
/// number-theoretic predicates, null checks, range-based predicates, and
/// nested predicate chains.
///
/// Generic over [`Target`] and holding no state: each run analyzes the
/// function through a `PredicateAnalyzer` over that run's facts.
pub struct OpaquePredicatePass<T: Target>(PhantomData<T>);

impl<T: Target> Default for OpaquePredicatePass<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Target> OpaquePredicatePass<T> {
    /// Creates a new opaque predicate pass.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

/// Maximum recursion depth for nested predicate analysis.
///
/// Each level corresponds to one SSA instruction defining a comparison result
/// that feeds into another comparison. 16 levels handles deeply nested opaque
/// predicates from advanced obfuscators (e.g., PureLogs multi-level chains)
/// while preventing stack overflow on pathological inputs.
const MAX_PREDICATE_DEPTH: usize = 16;

/// The predicate analysis of one function: what its definitions, phis,
/// non-null values and constant ranges ([`DefinitionCache`]) prove about a
/// predicate, a branch or a comparison.
struct PredicateAnalyzer<'a, T: Target> {
    /// The facts every question is answered from.
    cache: &'a DefinitionCache<T>,
}

impl<T: Target> PredicateAnalyzer<'_, T> {
    /// Analyzes a predicate operation with full context, dispatching by `SsaOp` kind.
    ///
    /// Pattern-matching cascade:
    /// 1. **Self-comparison** (`Ceq`/`Clt`/`Cgt` where `left == right`): immediate result.
    /// 2. **Equality analysis** (`Ceq`): delegates to [`analyze_equality`](Self::analyze_equality)
    ///    for XOR==0, SUB==0, MUL*0==0, AND&0==0, number-theoretic, constant, null, and nested patterns.
    /// 3. **Less-than / greater-than** (`Clt`/`Cgt`): delegates to range-based and constant analysis.
    /// 4. **Zero-producing ops** (`Xor`/`Sub` with `left==right`): returns `Unknown` since the
    ///    result only becomes meaningful when used in a comparison (handled at that level).
    /// 5. **Remainder / Multiplication / And**: delegates to specialized analyzers.
    ///
    /// Supports recursion up to [`MAX_PREDICATE_DEPTH`] for nested
    /// predicates (e.g., `ceq(ceq(x, x), 1)`).
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to analyze (typically a comparison or arithmetic op).
    /// * `depth` - Current recursion depth (0 at the top level).
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// predicate can be statically determined, [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_predicate_with_cache(&self, op: &SsaOp<T>, depth: usize) -> PredicateResult {
        if depth > MAX_PREDICATE_DEPTH {
            return PredicateResult::Unknown;
        }

        match op {
            // Self-comparison patterns
            SsaOp::Ceq { left, right, .. } => {
                // `x == x` is *false* when `x` is NaN, so this identity needs the
                // operand to be provably non-float.
                if left == right && !self.cache.maybe_float(*left) {
                    return PredicateResult::AlwaysTrue;
                }
                self.analyze_equality(*left, *right, depth)
            }

            // `x < x` and `x > x` are false for an integer and for an ordered
            // float compare. The unsigned form of a float compare is the
            // unordered one (`clt.un`), which holds when `x` is NaN.
            SsaOp::Clt {
                left,
                right,
                unsigned,
                ..
            } => {
                if left == right && !(*unsigned && self.cache.maybe_float(*left)) {
                    return PredicateResult::AlwaysFalse;
                }
                self.analyze_less_than(*left, *right, *unsigned, depth)
            }

            SsaOp::Cgt {
                left,
                right,
                unsigned,
                ..
            } => {
                if left == right && !(*unsigned && self.cache.maybe_float(*left)) {
                    return PredicateResult::AlwaysFalse;
                }
                self.analyze_greater_than(*left, *right, *unsigned, depth)
            }

            // Operations that produce zero
            SsaOp::Xor { left, right, .. } if left == right => {
                // x ^ x = 0, handled when used in comparison
                PredicateResult::Unknown
            }

            SsaOp::Sub { left, right, .. } if left == right => {
                // x - x = 0, handled when used in comparison
                PredicateResult::Unknown
            }

            SsaOp::Rem { left, right, .. } => self.analyze_remainder(*left, *right, depth),

            SsaOp::Mul { left, right, .. } => self.analyze_multiplication(*left, *right, depth),

            // `And` is deliberately not analysed here. This function is called
            // both on branch conditions and on ordinary instructions, and a
            // non-`Unknown` result feeds `comparison_replacements`, which
            // rewrites the instruction's destination to `ConstValue::True` or
            // `False` - a boolean where an `And` produces an integer. The facts
            // live where they can be stated safely: `x & 0` is falsy in
            // `analyze_branch_op`, which is reached for branch conditions only,
            // and the low-bit tests `even & 1 == 0` / `odd & 1 == 1` in
            // `analyze_equality`, where the comparison fixes the question.
            _ => PredicateResult::Unknown,
        }
    }

    /// Decides a flag-tested branch whose flags come from an `and`.
    ///
    /// `test v, v` is how x86 asks "is `v` zero", and it lifts to
    /// `And { left: v, right: v, flags }` followed by
    /// `BranchFlags { flags, condition }`. That is the same question
    /// [`Self::analyze_compare`] already answers, wearing flag clothing: `Zero`
    /// is `v == 0` and `NotZero` is `v != 0`. This is how an x86 lift ends a
    /// test of a computed value.
    ///
    /// Every other condition answers `Unknown`, and the match is exhaustive on
    /// purpose. Nothing here models carry, overflow, sign or parity, so the
    /// honest answer for those is that the branch is not decided - and a
    /// catch-all would silently adopt whatever `FlagCondition` variant is added
    /// next.
    ///
    /// Two `and` shapes are read, and the flags of any other producer are not:
    ///
    /// - A *self*-`and`, `v & v`, is `v`, so its zero flag is exactly `v == 0`,
    ///   decided here when `v` is itself an always-zero low-bit test such as
    ///   `(x * (x + 1)) & 1`.
    /// - Any other `and` is decided only when the `and` *itself* is a low-bit
    ///   test [`Self::is_always_even_expression`] or
    ///   [`Self::is_always_odd_expression`] recognises (`and v, 1` over an
    ///   even or odd `v`): its result, and so its zero flag, is then known.
    ///   `a & b` over arbitrary distinct operands says nothing either way.
    fn analyze_flag_branch(&self, flags: SsaVarId, condition: FlagCondition) -> PredicateResult {
        let cmp = match condition {
            FlagCondition::Zero => CmpKind::Eq,
            FlagCondition::NotZero => CmpKind::Ne,
            FlagCondition::Carry
            | FlagCondition::NotCarry
            | FlagCondition::Overflow
            | FlagCondition::NotOverflow
            | FlagCondition::Negative
            | FlagCondition::Positive
            | FlagCondition::ParityEven
            | FlagCondition::ParityOdd => return PredicateResult::Unknown,
        };
        let Some(SsaOp::And { left, right, .. }) =
            self.cache.get_definition(self.cache.peeled(flags))
        else {
            return PredicateResult::Unknown;
        };
        // x86 at `-O2` sets the flags with the low-bit `and` itself - `and v7,
        // 1` with `flags`, then `branchflags zero` - rather than with a
        // separate `test`. The producer is then the question, so ask the
        // expression rules about it directly.
        if self.cache.peeled(*left) != self.cache.peeled(*right) {
            let producer = self.cache.get_definition(self.cache.peeled(flags));
            if self.is_always_even_expression(producer) {
                return match cmp {
                    CmpKind::Eq => PredicateResult::AlwaysTrue,
                    _ => PredicateResult::AlwaysFalse,
                };
            }
            if self.is_always_odd_expression(producer) {
                return match cmp {
                    CmpKind::Eq => PredicateResult::AlwaysFalse,
                    _ => PredicateResult::AlwaysTrue,
                };
            }
            return PredicateResult::Unknown;
        }
        let tested = *left;
        // The comparison wants a zero to compare against, and the IR need not
        // contain one. Ask the value rule directly instead: `(even) & 1` is
        // zero for every input, so `test` on it sets the zero flag.
        if self.is_always_even_expression(self.cache.get_definition(self.cache.peeled(tested))) {
            return match cmp {
                CmpKind::Eq => PredicateResult::AlwaysTrue,
                _ => PredicateResult::AlwaysFalse,
            };
        }
        PredicateResult::Unknown
    }

    /// Decides `left <cmp> right` through the analyses that already exist.
    ///
    /// A `BranchCmp` states its comparison inline and names no condition
    /// variable, so nothing in this pass ever saw it - yet it is the same
    /// question `Ceq`/`Clt`/`Cgt` ask, and the same three analyses answer it.
    /// `Le` and `Ge` have no analysis of their own and are the negations of
    /// `Gt` and `Lt`, the framing `analysis::evaluator::evaluate_comparison`
    /// already uses.
    ///
    /// A value compared with itself is decided by the relation alone, as the
    /// `Ceq`/`Clt`/`Cgt` arms of
    /// [`analyze_predicate_with_cache`](Self::analyze_predicate_with_cache)
    /// decide it - a loop ending in `while (p != p)` runs once. A float may be
    /// NaN, which is unordered with itself, so for one only the ordered `<` and
    /// `>` (false either way) are decided; their unsigned forms are the
    /// unordered compares, true on NaN.
    fn analyze_compare(
        &self,
        left: SsaVarId,
        right: SsaVarId,
        cmp: CmpKind,
        unsigned: bool,
    ) -> PredicateResult {
        if left == right {
            let float = self.cache.maybe_float(left);
            return match cmp {
                CmpKind::Lt | CmpKind::Gt if !(unsigned && float) => PredicateResult::AlwaysFalse,
                CmpKind::Eq | CmpKind::Le | CmpKind::Ge if !float => PredicateResult::AlwaysTrue,
                CmpKind::Ne if !float => PredicateResult::AlwaysFalse,
                _ => PredicateResult::Unknown,
            };
        }
        match cmp {
            CmpKind::Eq => self.analyze_equality(left, right, 0),
            CmpKind::Ne => self.analyze_equality(left, right, 0).negate(),
            CmpKind::Lt => self.analyze_less_than(left, right, unsigned, 0),
            CmpKind::Ge => self.analyze_less_than(left, right, unsigned, 0).negate(),
            CmpKind::Gt => self.analyze_greater_than(left, right, unsigned, 0),
            CmpKind::Le => self.analyze_greater_than(left, right, unsigned, 0).negate(),
        }
    }

    /// Analyzes an equality comparison (`Ceq`) for opaque predicate patterns.
    ///
    /// Checks the following patterns (each with symmetric left/right variants):
    /// - `(x ^ x) == 0` -- XOR self-cancellation, always true.
    /// - `(x - x) == 0` -- subtraction self-cancellation, always true.
    /// - `(x * 0) == 0` or `(0 * x) == 0` -- zero-producing multiplication, always true.
    /// - `(x & 0) == 0` or `(0 & x) == 0` -- zero-producing AND, always true.
    /// - Number-theoretic: `(x*(x+1)) % 2 == 0` and factored forms, always true.
    /// - Constant equality: both sides are constants with known values.
    /// - Null checks: non-null variable (from `NewObj` etc.) compared to null, always false.
    /// - **Nested analysis fallback**: if the left operand is itself a predicate (comparison),
    ///   recursively analyzes it. If the result is known and compared to 1, returns that result;
    ///   if compared to 0, returns the negation.
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the `Ceq`.
    /// * `right` - Right operand of the `Ceq`.
    /// * `depth` - Current recursion depth for nested predicate analysis.
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// equality can be statically determined, [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_equality(&self, left: SsaVarId, right: SsaVarId, depth: usize) -> PredicateResult {
        let left_def = self.cache.get_definition(left);
        let right_def = self.cache.get_definition(right);

        // Check for (x ^ x) == 0 pattern
        if let Some(SsaOp::Xor {
            left: xl,
            right: xr,
            ..
        }) = left_def
            && xl == xr
            && self.const_of(right).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Symmetric check
        if let Some(SsaOp::Xor {
            left: xl,
            right: xr,
            ..
        }) = right_def
            && xl == xr
            && self.const_of(left).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Check for (x - x) == 0 pattern
        if let Some(SsaOp::Sub {
            left: sl,
            right: sr,
            ..
        }) = left_def
            && sl == sr
            && self.const_of(right).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Symmetric check
        if let Some(SsaOp::Sub {
            left: sl,
            right: sr,
            ..
        }) = right_def
            && sl == sr
            && self.const_of(left).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Check for (x * 0) == 0 pattern
        if self.is_zero_producing_mul(left_def)
            && self.const_of(right).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Symmetric check
        if self.is_zero_producing_mul(right_def)
            && self.const_of(left).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Check for (x & 0) == 0 pattern
        if self.is_zero_producing_and(left_def)
            && self.const_of(right).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Symmetric check
        if self.is_zero_producing_and(right_def)
            && self.const_of(left).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Check for number-theoretic predicates that always evaluate to zero:
        //   (x * (x + 1)) % 2 == 0    - consecutive integer product is always even
        //   (x * x - x) % 2 == 0      - x²-x = x(x-1), consecutive product factored
        //
        // Peeled, because a lift puts a width conversion between the low-bit
        // test and the comparison that reads it - on MIPS and AArch64
        // `branchcmp v13 != 0` has `v13 = conv.u32 v12` over the `& 1`.
        //
        // Peeled *here* and not at `left_def` above: this identity is
        // width-invariant, since the low bit of `x * (x + 1)` is clear at every
        // width and neither truncation nor extension disturbs it. The constant
        // equality arm below is **not** - `conv.u8(300)` is `44`, so comparing
        // the value underneath a truncating conversion would decide a different
        // question from the one the program asks.
        if self.is_always_even_expression(self.cache.get_definition(self.cache.peeled(left)))
            && self.const_of(right).is_some_and(ConstValue::is_zero)
        {
            return PredicateResult::AlwaysTrue;
        }

        // And its mirror: the low bit of an odd value is always 1, which is how
        // `((2 * v + 1) & 1) == 1` is written. Peeled for
        // the same width-invariance reason as the arm above.
        if self.is_always_odd_expression(self.cache.get_definition(self.cache.peeled(left)))
            && self.const_of(right).is_some_and(ConstValue::is_one)
        {
            return PredicateResult::AlwaysTrue;
        }

        // Check constant equality
        if let (Some(lval), Some(rval)) = (self.const_of(left), self.const_of(right))
            && let (Some(l), Some(r)) = (lval.as_i64(), rval.as_i64())
        {
            return if l == r {
                PredicateResult::AlwaysTrue
            } else {
                PredicateResult::AlwaysFalse
            };
        }

        // Check non-null equality with null, which native code spells as the
        // integer zero: `cmp rax, 0` after a load of `&g`.
        if self.cache.is_non_null(left)
            && self
                .const_of(right)
                .is_some_and(|value| value.is_null() || value.is_zero())
        {
            return PredicateResult::AlwaysFalse;
        }

        if self.cache.is_non_null(right)
            && self
                .const_of(left)
                .is_some_and(|value| value.is_null() || value.is_zero())
        {
            return PredicateResult::AlwaysFalse;
        }

        // Nested analysis
        if let Some(left_op) = left_def {
            let left_result = self.analyze_predicate_with_cache(left_op, depth.saturating_add(1));
            if left_result != PredicateResult::Unknown
                && let Some(r) = self.const_of(right)
            {
                if r.is_one() {
                    return left_result;
                }
                if r.is_zero() {
                    return left_result.negate();
                }
            }
        }

        PredicateResult::Unknown
    }

    /// Analyzes a less-than comparison (`Clt`) for opaque predicate patterns.
    ///
    /// Checks in order:
    /// 1. **Constant comparison**: both operands are constants, evaluate directly (signed or unsigned).
    /// 2. **Range-based**: if both operands have known [`ValueRange`]s, checks whether
    ///    `left.max < right.min` (always true) or `left.min >= right.max` (always false).
    /// 3. **Left range vs. constant right**: uses [`ValueRange::always_less_than`].
    ///
    ///    The ranges are signed intervals, so for an unsigned comparison steps
    ///    2 and 3 apply only when both operands are provably non-negative.
    /// 4. **Unsigned bounds**: `x <.un 0` is always false (no unsigned value is less than zero).
    /// 5. **Non-negative check**: if `left` is known non-negative (e.g., `ArrayLength`),
    ///    then `left < 0` is always false.
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the comparison.
    /// * `right` - Right operand of the comparison.
    /// * `unsigned` - Whether this is an unsigned comparison (`clt.un`).
    /// * `_depth` - Unused (less-than analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// less-than comparison can be statically determined,
    /// [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_less_than(
        &self,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        _depth: usize,
    ) -> PredicateResult {
        let left_def = self.cache.get_definition(left);
        let right_def = self.cache.get_definition(right);

        // Constant comparison
        if let (Some(SsaOp::Const { value: lval, .. }), Some(SsaOp::Const { value: rval, .. })) =
            (left_def, right_def)
        {
            if unsigned {
                if let (Some(l), Some(r)) = (lval.as_u64(), rval.as_u64()) {
                    return if l < r {
                        PredicateResult::AlwaysTrue
                    } else {
                        PredicateResult::AlwaysFalse
                    };
                }
            } else if let (Some(l), Some(r)) = (lval.as_i64(), rval.as_i64()) {
                return if l < r {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::AlwaysFalse
                };
            }
        }

        // Range-based analysis. The ranges are signed intervals, and the
        // unsigned order agrees with the signed one only where both operands
        // are non-negative: `-1 <u 1` is `0xFFFFFFFF <u 1`, which is false.
        if let Some(left_range) = self.cache.get_range(left) {
            if let Some(right_range) = self.cache.get_range(right)
                && ranges_order_alike(unsigned, left_range, right_range)
            {
                // left.max < right.min => always true
                if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                    && l_max < r_min
                {
                    return PredicateResult::AlwaysTrue;
                }
                // left.min >= right.max => always false
                if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                    && l_min >= r_max
                {
                    return PredicateResult::AlwaysFalse;
                }
            }

            // Check if left < constant
            if let Some(SsaOp::Const { value: rval, .. }) = right_def
                && let Some(r) = rval.as_i64()
                && ranges_order_alike(unsigned, left_range, &ValueRange::constant(r))
                && let Some(result) = left_range.always_less_than(r)
            {
                return if result {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::AlwaysFalse
                };
            }
        }

        // Unsigned comparison: x < 0 is always false
        if unsigned
            && let Some(SsaOp::Const { value: rval, .. }) = right_def
            && rval.as_u64() == Some(0)
        {
            return PredicateResult::AlwaysFalse;
        }

        // Non-negative < 0 is always false, in either order: the unsigned
        // reading of a non-negative value is the same number.
        if let Some(left_range) = self.cache.get_range(left)
            && left_range.is_always_non_negative()
            && let Some(SsaOp::Const { value: rval, .. }) = right_def
            && rval.as_i64() == Some(0)
        {
            return PredicateResult::AlwaysFalse;
        }

        PredicateResult::Unknown
    }

    /// Analyzes a greater-than comparison (`Cgt`) for opaque predicate patterns.
    ///
    /// Checks in order:
    /// 1. **Constant comparison**: both operands are constants, evaluate directly (signed or unsigned).
    /// 2. **Range-based**: if both operands have known [`ValueRange`]s, checks whether
    ///    `left.min > right.max` (always true) or `left.max <= right.min` (always false).
    /// 3. **Left range vs. constant right**: uses [`ValueRange::always_greater_than`].
    ///
    ///    The ranges are signed intervals, so for an unsigned comparison steps
    ///    2 and 3 apply only when both operands are provably non-negative.
    /// 4. **Unsigned bounds**: `0 >.un x` is always false (zero is never greater than any unsigned value).
    /// 5. **Non-negative vs. negative** (signed only): if `left` is known
    ///    non-negative and `right` is a negative constant, returns always true.
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the comparison.
    /// * `right` - Right operand of the comparison.
    /// * `unsigned` - Whether this is an unsigned comparison (`cgt.un`).
    /// * `_depth` - Unused (greater-than analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] or [`AlwaysFalse`](PredicateResult::AlwaysFalse) if the
    /// greater-than comparison can be statically determined,
    /// [`Unknown`](PredicateResult::Unknown) otherwise.
    fn analyze_greater_than(
        &self,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
        _depth: usize,
    ) -> PredicateResult {
        let left_def = self.cache.get_definition(left);
        let right_def = self.cache.get_definition(right);

        // Constant comparison
        if let (Some(SsaOp::Const { value: lval, .. }), Some(SsaOp::Const { value: rval, .. })) =
            (left_def, right_def)
        {
            if unsigned {
                if let (Some(l), Some(r)) = (lval.as_u64(), rval.as_u64()) {
                    return if l > r {
                        PredicateResult::AlwaysTrue
                    } else {
                        PredicateResult::AlwaysFalse
                    };
                }
            } else if let (Some(l), Some(r)) = (lval.as_i64(), rval.as_i64()) {
                return if l > r {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::AlwaysFalse
                };
            }
        }

        // Range-based analysis, over signed intervals: usable for an unsigned
        // compare only where both operands are non-negative (see
        // `analyze_less_than`).
        if let Some(left_range) = self.cache.get_range(left) {
            if let Some(right_range) = self.cache.get_range(right)
                && ranges_order_alike(unsigned, left_range, right_range)
            {
                // left.min > right.max => always true
                if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                    && l_min > r_max
                {
                    return PredicateResult::AlwaysTrue;
                }
                // left.max <= right.min => always false
                if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                    && l_max <= r_min
                {
                    return PredicateResult::AlwaysFalse;
                }
            }

            // Check if left > constant
            if let Some(SsaOp::Const { value: rval, .. }) = right_def
                && let Some(r) = rval.as_i64()
                && ranges_order_alike(unsigned, left_range, &ValueRange::constant(r))
                && let Some(result) = left_range.always_greater_than(r)
            {
                return if result {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::AlwaysFalse
                };
            }
        }

        // Unsigned: 0 > x is always false
        if unsigned
            && let Some(SsaOp::Const { value: lval, .. }) = left_def
            && lval.as_u64() == Some(0)
        {
            return PredicateResult::AlwaysFalse;
        }

        // Non-negative value >= 0 is always true (x > -1 equivalent). Signed
        // only: unsigned, `-1` is the largest value and nothing exceeds it.
        if !unsigned
            && let Some(left_range) = self.cache.get_range(left)
            && left_range.is_always_non_negative()
            && let Some(SsaOp::Const { value: rval, .. }) = right_def
            && rval.as_i64().is_some_and(|r| r < 0)
        {
            return PredicateResult::AlwaysTrue;
        }

        PredicateResult::Unknown
    }

    /// Analyzes a remainder operation (`Rem`) for `x % 1` which always produces zero.
    ///
    /// Returns `Unknown` rather than `AlwaysTrue`/`AlwaysFalse` because the zero result
    /// is only meaningful when subsequently compared (handled by the equality analysis).
    ///
    /// # Arguments
    ///
    /// * `_left` - Left operand of the remainder (unused; any value mod 1 is zero).
    /// * `right` - Right operand of the remainder (checked for constant 1).
    /// * `_depth` - Unused (remainder analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// Always returns [`PredicateResult::Unknown`] because the zero result is only
    /// meaningful when used in a subsequent comparison.
    fn analyze_remainder(
        &self,
        _left: SsaVarId,
        right: SsaVarId,
        _depth: usize,
    ) -> PredicateResult {
        // x % 1 == 0 is always true
        if let Some(SsaOp::Const { value: rval, .. }) = self.cache.get_definition(right)
            && rval.as_i64() == Some(1)
        {
            // Result is always 0
            return PredicateResult::Unknown; // Handled when compared to 0
        }
        PredicateResult::Unknown
    }

    /// Analyzes a multiplication (`Mul`) for zero-producing patterns (`x * 0` or `0 * x`).
    ///
    /// Returns `Unknown` because the zero result is only meaningful when subsequently
    /// compared (handled by the equality analysis at the comparison level).
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand of the multiplication.
    /// * `right` - Right operand of the multiplication.
    /// * `_depth` - Unused (multiplication analysis is non-recursive).
    ///
    /// # Returns
    ///
    /// Always returns [`PredicateResult::Unknown`] because the zero result is only
    /// meaningful when used in a subsequent comparison.
    fn analyze_multiplication(
        &self,
        left: SsaVarId,
        right: SsaVarId,
        _depth: usize,
    ) -> PredicateResult {
        // x * 0 = 0
        if let Some(SsaOp::Const { value: lval, .. }) = self.cache.get_definition(left)
            && lval.is_zero()
        {
            return PredicateResult::Unknown; // Result is 0
        }
        if let Some(SsaOp::Const { value: rval, .. }) = self.cache.get_definition(right)
            && rval.is_zero()
        {
            return PredicateResult::Unknown; // Result is 0
        }
        PredicateResult::Unknown
    }

    /// Returns `true` if the operation is a `Mul` where either operand is a constant zero.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to check, or `None` if the variable has no definition.
    ///
    /// # Returns
    ///
    /// `true` if `op` is a `Mul` with at least one constant-zero operand, `false` otherwise.
    fn is_zero_producing_mul(&self, op: Option<&SsaOp<T>>) -> bool {
        if let Some(SsaOp::Mul { left, right, .. }) = op {
            if self.const_of(*left).is_some_and(ConstValue::is_zero) {
                return true;
            }
            if self.const_of(*right).is_some_and(ConstValue::is_zero) {
                return true;
            }
        }
        false
    }

    /// Returns `true` if the operation is an `And` where either operand is a constant zero.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to check, or `None` if the variable has no definition.
    ///
    /// # Returns
    ///
    /// `true` if `op` is an `And` with at least one constant-zero operand, `false` otherwise.
    fn is_zero_producing_and(&self, op: Option<&SsaOp<T>>) -> bool {
        if let Some(SsaOp::And { left, right, .. }) = op {
            if self.const_of(*left).is_some_and(ConstValue::is_zero) {
                return true;
            }
            if self.const_of(*right).is_some_and(ConstValue::is_zero) {
                return true;
            }
        }
        false
    }

    /// Checks if an operation is an expression modulo 2 that always evaluates to 0.
    ///
    /// Detects number-theoretic opaque predicates based on the mathematical
    /// property that the product of two consecutive integers is always even:
    ///
    /// - `(x * (x + 1)) % 2` - direct consecutive product
    /// - `(x * (x - 1)) % 2` - reversed consecutive product
    /// - `(x * x - x) % 2` - factored form: x^2-x = x(x-1)
    /// - `(x * x + x) % 2` - factored form: x^2+x = x(x+1)
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to check, or `None` if the variable has no definition.
    ///
    /// # Returns
    ///
    /// `true` if the expression is a `Rem` by 2 whose dividend is always even, `false` otherwise.
    fn is_always_even_expression(&self, op: Option<&SsaOp<T>>) -> bool {
        match op {
            // `even % 2` - the spelling the C source uses.
            Some(SsaOp::Rem { left, right, .. }) if self.is_const_int(*right, 2) => {
                self.is_always_even_value(*left)
            }
            // `even & 1` - the spelling every compiler emits for it. Both
            // extract the low bit of a value that is even for every input, so
            // both are always zero.
            Some(SsaOp::And { left, right, .. }) if self.is_const_int(*right, 1) => {
                self.is_always_even_value(*left)
            }
            Some(SsaOp::And { left, right, .. }) if self.is_const_int(*left, 1) => {
                self.is_always_even_value(*right)
            }
            _ => false,
        }
    }

    /// Whether `var`'s definition is even for every input.
    ///
    /// Two spellings of the same arithmetic fact, and the reason both hold is
    /// that one of two consecutive integers is always even:
    ///
    /// - `x * (x +/- 1)` - the consecutive product, written directly.
    /// - `(x * x) +/- x` - the same product factored, `x² + x = x(x + 1)` and
    ///   `x² - x = x(x - 1)`.
    ///
    /// Kept apart from the test that reads it ([`Self::is_always_even_expression`])
    /// so the *value* rule has one home while the low-bit test may be spelled
    /// `% 2` or `& 1`.
    fn is_always_even_value(&self, var: SsaVarId) -> bool {
        let def = self.cache.get_definition(self.cache.peeled(var));

        // x * (x +/- 1) - consecutive product
        if let Some(SsaOp::Mul { left, right, .. }) = def
            && self.is_consecutive_pair(*left, *right)
        {
            return true;
        }

        // (x * x) -/+ x - factored consecutive product
        if let Some(SsaOp::Sub { left, right, .. }) | Some(SsaOp::Add { left, right, .. }) = def
            && (self.is_self_square(*left, *right) || self.is_self_square(*right, *left))
        {
            return true;
        }

        // x + x - doubling written as a self-add, which is what x86 emits for
        // `2 * v` at `-O0` (`add v6, v6`).
        if let Some(SsaOp::Add { left, right, .. }) = def
            && self.cache.peeled(*left) == self.cache.peeled(*right)
        {
            return true;
        }

        // x << n for 1 <= n < width - the same doubling as a shift, which is
        // what aarch64 and mips emit (`shl v6, 1`). Every value with its low
        // bit shifted out is even. A count at or past the width is not a
        // shift the IR defines: the ISA decides, and x86 masks `shl eax, 32`
        // to a count of 0, leaving the value - and its low bit - unchanged. So
        // the width must be known, and the count is read exactly (see
        // `Self::const_of`), never through a truncating conversion.
        if let Some(SsaOp::Shl { dest, amount, .. }) = def
            && let Some(width) = self.cache.bit_width(*dest)
            && self
                .const_of(*amount)
                .and_then(ConstValue::as_i64)
                .is_some_and(|n| n >= 1 && n < i64::from(width))
        {
            return true;
        }

        false
    }

    /// Whether `var`'s definition is odd for every input.
    ///
    /// One rule only, and it is the even rule's counterpart: an even value plus
    /// one is odd. That covers `2 * v + 1`, the low-bit obfuscation, in every
    /// spelling [`Self::is_always_even_value`] accepts for the doubling.
    fn is_always_odd_value(&self, var: SsaVarId) -> bool {
        let def = self.cache.get_definition(self.cache.peeled(var));
        if let Some(SsaOp::Add { left, right, .. }) = def
            && ((self.is_always_even_value(*left) && self.is_const_int(*right, 1))
                || (self.is_always_even_value(*right) && self.is_const_int(*left, 1)))
        {
            return true;
        }
        false
    }

    /// Whether `op` extracts the low bit of a value that is odd for every
    /// input, and is therefore always `1`.
    ///
    /// The mirror of [`Self::is_always_even_expression`], which is always `0`.
    /// Both spellings are accepted for the same reason, with one difference:
    /// `% 2` counts only when it is unsigned. A signed remainder takes the
    /// dividend's sign, so signed `(2v + 1) % 2` is `-1` for a negative `v`.
    /// (An even dividend leaves `0` in either signedness, so the even rule
    /// takes both.)
    fn is_always_odd_expression(&self, op: Option<&SsaOp<T>>) -> bool {
        match op {
            Some(SsaOp::Rem {
                left,
                right,
                unsigned: true,
                ..
            }) if self.is_const_int(*right, 2) => self.is_always_odd_value(*left),
            Some(SsaOp::And { left, right, .. }) if self.is_const_int(*right, 1) => {
                self.is_always_odd_value(*left)
            }
            Some(SsaOp::And { left, right, .. }) if self.is_const_int(*left, 1) => {
                self.is_always_odd_value(*right)
            }
            _ => false,
        }
    }

    /// Whether `var` is defined as the integer constant `wanted`.
    fn is_const_int(&self, var: SsaVarId, wanted: i64) -> bool {
        self.const_of(var).and_then(ConstValue::as_i64) == Some(wanted)
    }

    /// The constant `var` holds, found through copies only.
    ///
    /// Not through [`DefinitionCache::peeled`]: that walk crosses width
    /// conversions, which is sound for the low-bit facts it serves and wrong
    /// for reading a *value*. `conv.u8(256)` is `0`, not `256`, and a
    /// zero-extended `-1` is `255`, so the constant under a conversion is not
    /// the operand the instruction reads.
    fn const_of(&self, var: SsaVarId) -> Option<&ConstValue<T>> {
        let mut var = var;
        for _ in 0..8 {
            match self.cache.get_definition(var)? {
                SsaOp::Copy { src, .. } => var = *src,
                SsaOp::Const { value, .. } => return Some(value),
                _ => return None,
            }
        }
        None
    }

    /// Checks if `square_var` is defined as `other * other` (i.e., `other^2`).
    ///
    /// # Arguments
    ///
    /// * `square_var` - The variable suspected to be a square.
    /// * `other` - The variable that should appear as both operands of the multiplication.
    ///
    /// # Returns
    ///
    /// `true` if `square_var` is defined as `Mul { left: other, right: other }`, `false` otherwise.
    fn is_self_square(&self, square_var: SsaVarId, other: SsaVarId) -> bool {
        let other = self.cache.peeled(other);
        matches!(
            self.cache.get_definition(self.cache.peeled(square_var)),
            Some(SsaOp::Mul { left, right, .. })
                if self.cache.peeled(*left) == other && self.cache.peeled(*right) == other
        )
    }

    /// Checks if two variables form a consecutive integer pair (`n` and `n+1`).
    ///
    /// Performs three symmetric checks:
    /// - `b = a + 1` (either operand order of the `Add`).
    /// - `a = b + 1` (symmetric: `a` is the incremented one).
    /// - `b = a - (-1)` (subtraction of -1 is equivalent to adding 1).
    ///
    /// # Arguments
    ///
    /// * `a` - First variable of the potential consecutive pair.
    /// * `b` - Second variable of the potential consecutive pair.
    ///
    /// # Returns
    ///
    /// `true` if one variable is defined as the other plus one, `false` otherwise.
    fn is_consecutive_pair(&self, a: SsaVarId, b: SsaVarId) -> bool {
        let a = self.cache.peeled(a);
        let b = self.cache.peeled(b);

        // b = a + 1, either operand order
        if let Some(SsaOp::Add { left, right, .. }) = self.cache.get_definition(b)
            && ((self.cache.peeled(*left) == a && self.is_const_int(*right, 1))
                || (self.cache.peeled(*right) == a && self.is_const_int(*left, 1)))
        {
            return true;
        }

        // a = b + 1, the symmetric case
        if let Some(SsaOp::Add { left, right, .. }) = self.cache.get_definition(a)
            && ((self.cache.peeled(*left) == b && self.is_const_int(*right, 1))
                || (self.cache.peeled(*right) == b && self.is_const_int(*left, 1)))
        {
            return true;
        }

        // b = a - (-1), which is also a + 1
        if let Some(SsaOp::Sub { left, right, .. }) = self.cache.get_definition(b)
            && self.cache.peeled(*left) == a
            && self.is_const_int(*right, -1)
        {
            return true;
        }

        // b = lea [a + 1]. x86 computes `v + 1` with an address instruction,
        // so the increment arrives as `PtrAdd` with a constant displacement and
        // no index (`ptradd v7 + 1`), where a factored form uses a real `Add`.
        if let Some(SsaOp::PtrAdd {
            base,
            index: None,
            offset: 1,
            ..
        }) = self.cache.get_definition(b)
            && self.cache.peeled(*base) == a
        {
            return true;
        }

        if let Some(SsaOp::PtrAdd {
            base,
            index: None,
            offset: 1,
            ..
        }) = self.cache.get_definition(a)
            && self.cache.peeled(*base) == b
        {
            return true;
        }

        false
    }

    /// Analyzes a branch condition variable to determine if it is an opaque predicate.
    ///
    /// Follows `Copy` chains iteratively with a [`VarSet`]-based cycle detector to handle
    /// SSA copies from phi nodes or obfuscated control flow. At each step:
    /// 1. If the variable has no definition in `DefUseIndex` but is phi-defined, checks
    ///    range info for a constant-zero equivalence (all-zero phi = always false).
    /// 2. Delegates to [`analyze_predicate_with_cache`](Self::analyze_predicate_with_cache) for
    ///    comparison operations.
    /// 3. For `Copy` ops, advances to the source variable and continues the loop.
    /// 4. For all other operations, falls back to [`analyze_branch_op`](Self::analyze_branch_op)
    ///    which checks direct truthiness of arithmetic and constant operations.
    ///
    /// The cycle detector is a caller-owned scratch set, one variable wide,
    /// shared by every branch of a run: allocating one per branch would cost
    /// O(branches x variables). It is empty on entry and left empty on return -
    /// the marks this walk set are cleared by re-walking the same copy chain.
    ///
    /// # Arguments
    ///
    /// * `condition` - The SSA variable used as the branch condition.
    /// * `visited` - Empty scratch set, sized from
    ///   [`SsaFunction::var_id_capacity`](crate::ir::function::SsaFunction::var_id_capacity).
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] if the branch always takes the true path,
    /// [`AlwaysFalse`](PredicateResult::AlwaysFalse) if it always takes the false path,
    /// [`Unknown`](PredicateResult::Unknown) if the condition cannot be statically resolved.
    fn analyze_branch(&self, condition: SsaVarId, visited: &mut VarSet) -> PredicateResult {
        let result = self.walk_branch_condition(condition, visited);
        // The walk marks a prefix of the copy chain from `condition` and
        // follows a `Copy` only out of a marked variable, so following copies
        // while the current variable is marked visits exactly those marks.
        let mut current = condition;
        while visited.contains(current) {
            visited.remove(current);
            match self.cache.get_definition(current) {
                Some(SsaOp::Copy { src, .. }) => current = *src,
                _ => break,
            }
        }
        result
    }

    /// The copy-chain walk behind [`Self::analyze_branch`], marking each
    /// variable it visits in `visited`.
    fn walk_branch_condition(&self, condition: SsaVarId, visited: &mut VarSet) -> PredicateResult {
        // Follow Copy chain iteratively with cycle detection to prevent infinite recursion.
        // This is needed because SSA can have Copy cycles (e.g., from phi nodes or
        // obfuscated control flow patterns).
        let mut current = condition;

        loop {
            // Cycle detection: if we've seen this variable before, bail out.
            // An index outside the set is treated the same way.
            if !visited.insert(current) {
                return PredicateResult::Unknown;
            }

            let Some(cond_op) = self.cache.get_definition(current) else {
                // Check if it's a phi node - analyze all operands
                if self.cache.is_phi_defined(current) {
                    // For phi nodes, we'd need to check if all operands lead to the same result
                    // This is complex, so we return Unknown for now unless we have range info
                    if let Some(range) = self.cache.get_range(current)
                        && let Some(result) = range.always_equal_to(0)
                    {
                        return if result {
                            PredicateResult::AlwaysFalse
                        } else {
                            PredicateResult::AlwaysTrue
                        };
                    }
                }
                return PredicateResult::Unknown;
            };

            // First, check if it's a direct comparison predicate
            let predicate_result = self.analyze_predicate_with_cache(cond_op, 0);
            if predicate_result != PredicateResult::Unknown {
                return predicate_result;
            }

            // Check if the condition is a Copy - trace through to the source iteratively
            if let SsaOp::Copy { src, .. } = cond_op {
                current = *src;
                continue;
            }

            // Not a Copy, break out and analyze the operation
            return self.analyze_branch_op(cond_op);
        }
    }

    /// Analyzes a non-Copy, non-comparison operation for direct truthiness in a branch.
    ///
    /// Called after the `Copy` chain has been resolved by [`analyze_branch`](Self::analyze_branch).
    /// Checks:
    /// - `x ^ x` = 0 (always false in `brtrue`).
    /// - `x - x` = 0 (always false).
    /// - `x & 0` or `x * 0` = 0 (always false).
    /// - `x | -1` = all-bits-set (always true).
    /// - `Const`: zero/null/false is always false; non-zero numeric or `true` is always true.
    ///
    /// # Arguments
    ///
    /// * `cond_op` - The resolved operation producing the branch condition value.
    ///
    /// # Returns
    ///
    /// [`PredicateResult::AlwaysTrue`] if the operation always produces a non-zero value,
    /// [`AlwaysFalse`](PredicateResult::AlwaysFalse) if it always produces zero,
    /// [`Unknown`](PredicateResult::Unknown) if the truthiness cannot be determined.
    fn analyze_branch_op(&self, cond_op: &SsaOp<T>) -> PredicateResult {
        // Check operations that produce known zero values
        match cond_op {
            // x ^ x = 0, so brtrue on this result never jumps
            SsaOp::Xor { left, right, .. } if left == right => PredicateResult::AlwaysFalse,

            // x - x = 0, so brtrue on this result never jumps - unless `x` is a
            // float, where `NaN - NaN` is NaN and therefore branches as true.
            SsaOp::Sub { left, right, .. } if left == right && !self.cache.maybe_float(*left) => {
                PredicateResult::AlwaysFalse
            }

            // x & 0 = 0, x * 0 = 0
            SsaOp::And { left, right, .. } | SsaOp::Mul { left, right, .. } => {
                let is_left_zero = self.const_of(*left).is_some_and(ConstValue::is_zero);
                let is_right_zero = self.const_of(*right).is_some_and(ConstValue::is_zero);

                if is_left_zero || is_right_zero {
                    PredicateResult::AlwaysFalse
                } else {
                    PredicateResult::Unknown
                }
            }

            // x | -1 = -1 (all bits set), so brtrue always jumps
            SsaOp::Or { left, right, .. } => {
                let is_left_minus_one = self.const_of(*left).is_some_and(ConstValue::is_minus_one);
                let is_right_minus_one =
                    self.const_of(*right).is_some_and(ConstValue::is_minus_one);

                if is_left_minus_one || is_right_minus_one {
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::Unknown
                }
            }

            // Constant values: 0/null/false is always false, non-zero is always true
            SsaOp::Const { value, .. } => {
                if value.is_zero() || value.is_null() {
                    PredicateResult::AlwaysFalse
                } else if value.as_i64().is_some() || value.as_bool().is_some() {
                    // Non-zero numeric or true boolean
                    PredicateResult::AlwaysTrue
                } else {
                    PredicateResult::Unknown
                }
            }

            // All other operations have unknown truthiness
            // Note: ArrayLength is always >= 0, but we can't prove non-empty
            _ => PredicateResult::Unknown,
        }
    }

    /// Analyzes a comparison operation for algebraic simplification opportunities.
    ///
    /// This checks for patterns like:
    /// - `(x - y) == 0` → `x == y`
    /// - `(x ^ y) == 0` → `x == y`
    /// - `(cmp) == 1` → `cmp`
    ///
    /// An order is not read through a subtraction: `(x - y) < 0` is the sign
    /// of a difference that wraps, which `x < y` is not where it overflows
    /// (`(i64::MIN + 5) - (1 << 50)` is positive).
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA comparison operation to analyze.
    ///
    /// # Returns
    ///
    /// `Some(ComparisonSimplification)` if the comparison can be algebraically simplified,
    /// `None` if no simplification applies or the operation is not a comparison.
    fn analyze_comparison_simplification(
        &self,
        op: &SsaOp<T>,
    ) -> Option<ComparisonSimplification<T>> {
        match op {
            SsaOp::Ceq {
                dest, left, right, ..
            } => self.analyze_ceq_simplification(*dest, *left, *right),
            _ => None,
        }
    }

    /// Analyzes a `Ceq` operation for algebraic simplification.
    ///
    /// Detects three patterns:
    /// - `(x - y) == 0` simplifies to `x == y` (subtraction-zero, skip self-subtraction).
    /// - `(x ^ y) == 0` simplifies to `x == y` (XOR-zero, skip self-XOR).
    ///
    /// Neither holds of floats, so both need integer operands: `inf - inf` is
    /// NaN while `inf == inf`, and `+0.0 ^ -0.0` is nonzero while `+0.0 ==
    /// -0.0`.
    /// - `(cmp) == 1` simplifies to `Copy(cmp)` when the other operand is a `Ceq`/`Clt`/`Cgt`
    ///   result, since CIL comparisons already produce 0 or 1.
    ///
    /// # Arguments
    ///
    /// * `dest` - Destination variable of the `Ceq` (preserved in the simplified op).
    /// * `left` - Left operand of the `Ceq`.
    /// * `right` - Right operand of the `Ceq`.
    ///
    /// # Returns
    ///
    /// `Some(ComparisonSimplification)` if a simplification pattern matches, `None` otherwise.
    fn analyze_ceq_simplification(
        &self,
        dest: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
    ) -> Option<ComparisonSimplification<T>> {
        // Check if comparing to zero
        let (other_var, is_comparing_to_zero) =
            if self.const_of(right).is_some_and(ConstValue::is_zero) {
                (left, true)
            } else if self.const_of(left).is_some_and(ConstValue::is_zero) {
                (right, true)
            } else {
                (left, false)
            };

        if is_comparing_to_zero && let Some(def_op) = self.cache.get_definition(other_var) {
            // Pattern: (x - y) == 0 → x == y
            if let SsaOp::Sub {
                left: sub_left,
                right: sub_right,
                ..
            } = def_op
            {
                // Skip self-subtraction - that's handled by PredicateResult (always true)
                if sub_left != sub_right
                    && !self.cache.maybe_float(*sub_left)
                    && !self.cache.maybe_float(*sub_right)
                {
                    return Some(ComparisonSimplification::SimplerOp {
                        new_op: SsaOp::Ceq {
                            dest,
                            left: *sub_left,
                            right: *sub_right,
                            signaling: false,
                        },
                        reason: "(x - y) == 0 simplified to x == y",
                    });
                }
            }

            // Pattern: (x ^ y) == 0 → x == y
            if let SsaOp::Xor {
                left: xor_left,
                right: xor_right,
                ..
            } = def_op
            {
                // Skip self-XOR - that's handled by PredicateResult (always true)
                if xor_left != xor_right
                    && !self.cache.maybe_float(*xor_left)
                    && !self.cache.maybe_float(*xor_right)
                {
                    return Some(ComparisonSimplification::SimplerOp {
                        new_op: SsaOp::Ceq {
                            dest,
                            left: *xor_left,
                            right: *xor_right,
                            signaling: false,
                        },
                        reason: "(x ^ y) == 0 simplified to x == y",
                    });
                }
            }
        }

        // Check if comparing to one (true in CIL)
        let (other_var, is_comparing_to_one) =
            if self.const_of(right).is_some_and(ConstValue::is_one) {
                (left, true)
            } else if self.const_of(left).is_some_and(ConstValue::is_one) {
                (right, true)
            } else {
                (left, false)
            };

        if is_comparing_to_one && let Some(def_op) = self.cache.get_definition(other_var) {
            // Pattern: (cmp) == 1 → copy cmp
            if matches!(
                def_op,
                SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. }
            ) {
                return Some(ComparisonSimplification::Copy {
                    dest,
                    src: other_var,
                    reason: "(cmp) == 1 simplified to cmp",
                });
            }
        }

        None
    }
}

/// Whether the signed intervals `left` and `right` order their values the
/// way the comparison does.
///
/// Always for a signed comparison. For an unsigned one only when both are
/// provably non-negative, where the two orders coincide; a negative value
/// reads as a large unsigned one, so its signed interval says nothing
/// about where it sits unsigned. The same rule `passes::ranges` applies
/// before deciding a comparison from intervals.
fn ranges_order_alike(unsigned: bool, left: &ValueRange, right: &ValueRange) -> bool {
    !unsigned || (left.is_always_non_negative() && right.is_always_non_negative())
}

/// Detects phi nodes whose value is one constant on every path.
///
/// A phi is constant when each operand is either a `Const` of that value or
/// another phi that is itself constant with it - a web of phis carrying one
/// constant around merges and loops, as a register holding an address does
/// in lifted code. The phi's own result is accepted too: `p = phi(c, p)` is
/// `c` on entry and `p` thereafter, so `c` throughout. Anything else - an
/// operand some other instruction computes, or two different constants -
/// makes the phi's value unknown.
///
/// Decided optimistically over the whole web: every phi whose operands are
/// all constants or phis starts as a candidate, and a candidate is dropped
/// as soon as an operand is not a candidate or disagrees on the value,
/// until nothing changes. Deciding each phi from its operands alone -
/// requiring a phi operand to be known constant *first* - never settles a
/// loop, where each phi of the cycle waits on the other; and a web whose
/// first phi was folded left the next one reading a phi, so the fold
/// stopped at the first merge.
///
/// These entries resolve branch conditions that depend on phi-defined
/// variables, and redirect the phis' readers to the constant.
///
/// # Arguments
///
/// * `ssa` - The SSA function whose phi nodes are analyzed.
///
/// # Returns
///
/// A map from phi result variable to the constant every path gives it.
fn analyze_phi_constants<T: Target>(ssa: &SsaFunction<T>) -> BTreeMap<SsaVarId, ConstValue<T>> {
    // Index defining ops once (O(1) lookups) rather than calling
    // `get_definition` per phi operand, whose slow path is a full-function
    // scan when def-sites are stale.
    let index = DefUseIndex::build_with_ops(ssa);

    // Each phi's operands that are other phis' results, and the constant
    // its `Const` operands agree on (`None` while it has seen none).
    let phi_results: VarSet = {
        let mut results = VarSet::new(ssa.var_id_bound());
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                results.insert(phi.result());
            }
        }
        results
    };
    let mut candidates: BTreeMap<SsaVarId, (Option<ConstValue<T>>, Vec<SsaVarId>)> =
        BTreeMap::new();
    for block in ssa.blocks() {
        'phis: for phi in block.phi_nodes() {
            if phi.operands().is_empty() {
                continue;
            }
            let mut value: Option<ConstValue<T>> = None;
            let mut phi_operands = Vec::new();
            for operand in phi.operands() {
                let var = operand.value();
                if var == phi.result() {
                    continue;
                }
                if let Some(SsaOp::Const {
                    value: constant, ..
                }) = index.def_op(var)
                {
                    match &value {
                        None => value = Some(constant.clone()),
                        Some(seen) if seen == constant => {}
                        Some(_) => continue 'phis,
                    }
                } else if phi_results.contains(var) {
                    phi_operands.push(var);
                } else {
                    continue 'phis;
                }
            }
            candidates.insert(phi.result(), (value, phi_operands));
        }
    }

    // Drop what disagrees, and let a value spread along the web, until
    // neither changes. Values only ever go from unseen to seen and
    // candidates only ever leave, so this ends.
    loop {
        let mut changed = false;
        let results: Vec<SsaVarId> = candidates.keys().copied().collect();
        for result in results {
            let Some((mut value, operands)) = candidates.get(&result).cloned() else {
                continue;
            };
            let mut keep = true;
            for operand in &operands {
                match candidates.get(operand) {
                    None => keep = false,
                    Some((Some(theirs), _)) => match &value {
                        None => value = Some(theirs.clone()),
                        Some(ours) if ours == theirs => {}
                        Some(_) => keep = false,
                    },
                    Some((None, _)) => {}
                }
                if !keep {
                    break;
                }
            }
            if !keep {
                candidates.remove(&result);
                changed = true;
            } else if let Some(entry) = candidates.get_mut(&result)
                && entry.0 != value
            {
                entry.0 = value;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // A web with no constant anywhere in it carries nothing known.
    candidates
        .into_iter()
        .filter_map(|(result, (value, _))| value.map(|value| (result, value)))
        .collect()
}

impl<T> FunctionPass<T> for OpaquePredicatePass<T>
where
    T: Target + Send + Sync,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "opaque-predicate"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Remove always-true/false conditions, simplify trivial comparisons"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Runs the opaque-predicate pass on `ssa`.
    ///
    /// Resolves opaque branch conditions and comparisons, and redirects the
    /// readers of a phi whose every operand is the same constant to that
    /// constant.
    ///
    /// Builds a `DefinitionCache` and a phi-constant map once, then makes a single
    /// read-only pass over every block to *collect* (never apply during the walk)
    /// four kinds of rewrite: branch simplifications from resolved conditions,
    /// comparison replacements for opaque predicates that fold to a constant,
    /// algebraic comparison simplifications, and phi-to-constant replacements. An
    /// unresolved branch condition falls back to a symbolic [`SsaEvaluator`] that
    /// is built and fully evaluated exactly once, then reused across branches.
    /// Because a CFG-changing branch simplification invalidates the instruction
    /// and phi indices gathered from the pre-edit graph, any collected comparison
    /// and phi rewrites are dropped whenever branch simplifications exist, to be
    /// re-derived on the scheduler's next iteration. The surviving rewrites are
    /// applied in one verified, rollback-on-failure edit, after which the internal
    /// event log is flushed to `events` in invocation order.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if anything was rewritten: a branch simplified, a comparison
    /// resolved or rewritten, or a constant phi's readers redirected to the
    /// constant.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let machine = env.context().machine();
        // Per-pass internal event collector. Flushed to `events` at the end
        // so the caller still gets every record in invocation order.
        let changes: EventLog<T> = EventLog::new();
        let method_token = env.method();

        // Build definition cache for efficient lookup
        let cache = DefinitionCache::build(ssa);
        let analyzer = PredicateAnalyzer { cache: &cache };

        // Analyze phi nodes for constant values
        let phi_constants = analyze_phi_constants(ssa);

        // Collect branches to simplify
        let mut branch_simplifications: Vec<(usize, usize, bool)> = Vec::new();

        // Collect comparison replacements (opaque predicates that become constant true/false)
        let mut comparison_replacements: Vec<(usize, usize, SsaVarId, bool)> = Vec::new();

        // Collect comparison simplifications (algebraic simplifications like (x-y)==0 → x==y)
        let mut comparison_simplifications: Vec<(usize, usize, ComparisonSimplification<T>)> =
            Vec::new();

        // Collect phi replacements
        let mut phi_replacements: Vec<(usize, SsaVarId, ConstValue<T>)> = Vec::new();

        // Symbolic evaluator for branch conditions, built lazily and evaluated
        // over every block exactly once. A fresh evaluator per unresolved branch,
        // re-evaluating blocks `0..=block_idx`, would be O(branches * blocks);
        // one forward pass suffices because SSA values are single-assignment, so
        // evaluating later blocks never changes an earlier branch condition.
        let mut branch_evaluator = None;

        // Predecessor relation and the facts guards give the blocks they enter,
        // for the correlated-condition rule below. Built once: it is a CSR pair
        // over the terminators and the walk below does not change the graph.
        let cfg = SsaCfg::from_ssa(ssa);
        let exception_blocks = ssa.exception_blocks();
        let guards = GuardFacts::of(ssa, &cfg, &exception_blocks, &cache, machine.ptr_size, None);

        // Cycle-detector scratch for `analyze_branch`, left empty by each call.
        let mut visited = VarSet::new(ssa.var_id_bound());

        // Analyze each block
        for (block_idx, block) in ssa.iter_blocks() {
            // A block knows the answers of the edges entering the blocks that
            // dominate it, so a terminator re-testing one is already decided.
            // This runs before the intra-expression analyses below because it can
            // decide branches whose condition nothing in this pass can evaluate
            // - a flags value with no computable state among them.
            if let Some((target, held)) = guards.decided_branch(ssa, block_idx, &cache) {
                branch_simplifications.push((block_idx, target, held));
                continue;
            }
            // Analyze branch terminators
            // A flag-tested branch hides the same question behind `test v,v`,
            // which is how an x86 lift ends a test of a computed value.
            if let Some(SsaOp::BranchFlags {
                flags,
                condition,
                true_target,
                false_target,
            }) = block.control_terminator()
            {
                match analyzer.analyze_flag_branch(*flags, *condition) {
                    PredicateResult::AlwaysTrue => {
                        branch_simplifications.push((block_idx, *true_target, true));
                    }
                    PredicateResult::AlwaysFalse => {
                        branch_simplifications.push((block_idx, *false_target, false));
                    }
                    PredicateResult::Unknown => {}
                }
            }

            // A compare-and-branch states its relation inline, so the
            // `Branch` arm below cannot see it. Targets without a flags
            // register end a test this way; the decision is the same one, and
            // it feeds the same simplification list.
            if let Some(SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            }) = block.control_terminator()
            {
                // Two addresses from one base compare as their displacements,
                // which no expression rule here reads.
                let outcome = match analyzer.analyze_compare(*left, *right, *cmp, *unsigned) {
                    PredicateResult::Unknown => {
                        match AddressWalk::new(ssa, machine.ptr_size)
                            .compare(*left, *right, *cmp, *unsigned)
                        {
                            Some(true) => PredicateResult::AlwaysTrue,
                            Some(false) => PredicateResult::AlwaysFalse,
                            None => PredicateResult::Unknown,
                        }
                    }
                    decided => decided,
                };
                match outcome {
                    PredicateResult::AlwaysTrue => {
                        branch_simplifications.push((block_idx, *true_target, true));
                    }
                    PredicateResult::AlwaysFalse => {
                        branch_simplifications.push((block_idx, *false_target, false));
                    }
                    PredicateResult::Unknown => {}
                }
            }

            if let Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) = block.control_terminator()
            {
                // Check phi constants first
                if let Some(const_val) = phi_constants.get(condition) {
                    let is_true = const_val.as_bool().unwrap_or(false)
                        || const_val.as_i64().is_some_and(|v| v != 0);
                    if is_true {
                        branch_simplifications.push((block_idx, *true_target, true));
                    } else {
                        branch_simplifications.push((block_idx, *false_target, false));
                    }
                    // Can't use continue with iter_blocks in a for loop, collect the data
                } else {
                    let mut result = analyzer.analyze_branch(*condition, &mut visited);

                    // If pattern matching couldn't determine the result, try the
                    // shared symbolic evaluator (built and fully evaluated once).
                    if result == PredicateResult::Unknown {
                        let evaluator = branch_evaluator.get_or_insert_with(|| {
                            let mut e = SsaEvaluator::new(ssa, machine);
                            for idx in 0..ssa.block_count() {
                                e.evaluate_block(idx);
                            }
                            e
                        });
                        result = match evaluator.get(*condition) {
                            Some(expr) if expr.is_constant() => {
                                if expr.as_constant().is_some_and(ConstValue::is_zero) {
                                    PredicateResult::AlwaysFalse
                                } else {
                                    PredicateResult::AlwaysTrue
                                }
                            }
                            Some(_) | None => PredicateResult::Unknown,
                        };
                    }

                    match result {
                        PredicateResult::AlwaysTrue => {
                            branch_simplifications.push((block_idx, *true_target, true));
                        }
                        PredicateResult::AlwaysFalse => {
                            branch_simplifications.push((block_idx, *false_target, false));
                        }
                        PredicateResult::Unknown => {}
                    }
                }
            }

            // Analyze comparison instructions
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let op = instr.op();
                // First check for opaque predicates (constant true/false)
                let result = analyzer.analyze_predicate_with_cache(op, 0);
                if let Some(value) = result.as_bool()
                    && let Some(dest) = op.dest()
                {
                    comparison_replacements.push((block_idx, instr_idx, dest, value));
                    continue; // Don't also check for simplification
                }

                // Then check for algebraic simplifications
                if let Some(simplification) = analyzer.analyze_comparison_simplification(op) {
                    comparison_simplifications.push((block_idx, instr_idx, simplification));
                }
            }

            // Check for phi nodes whose instructions can read a constant
            // instead. Only instruction uses are rewritten (see below), so a
            // phi no instruction reads has nothing to fold.
            for phi in block.phi_nodes() {
                if let Some(const_val) = phi_constants.get(&phi.result())
                    && ssa
                        .variable(phi.result())
                        .is_some_and(|var| var.uses().iter().any(|site| !site.is_phi_operand()))
                {
                    phi_replacements.push((block_idx, phi.result(), const_val.clone()));
                }
            }
        }

        // CFG-changing branch simplifications invalidate instruction and phi indices
        // collected from the pre-edit graph. Apply them alone; the scheduler's next
        // pass iteration will re-analyze comparisons and phis on the rebuilt SSA.
        if !branch_simplifications.is_empty() {
            comparison_replacements.clear();
            comparison_simplifications.clear();
            phi_replacements.clear();
        }

        // A select whose condition is decided where it is evaluated is its
        // chosen value. Kept whatever branch folds with it: folding only removes
        // ways into a block, so what every way in knew still holds.
        for (block_idx, instr_idx, dest, chosen) in guards.decided_selects(ssa, &cache) {
            comparison_simplifications.push((
                block_idx,
                instr_idx,
                ComparisonSimplification::Copy {
                    dest,
                    src: chosen,
                    reason: "select decided by its guard",
                },
            ));
        }

        // Every other read of a decided boolean, where no rule above rewrites
        // its instruction.
        let touched: Vec<(usize, usize)> = comparison_replacements
            .iter()
            .map(|(block, instr, ..)| (*block, *instr))
            .chain(
                comparison_simplifications
                    .iter()
                    .map(|(block, instr, _)| (*block, *instr)),
            )
            .collect();
        // A read of a select its guard decides, where no rule above rewrites its
        // instruction, reads the arm chosen.
        let select_reads: Vec<(usize, usize, SsaVarId, SsaVarId)> = guards
            .decided_select_reads(ssa, &cache)
            .into_iter()
            .filter(|(block, instr, ..)| !touched.contains(&(*block, *instr)))
            .collect();
        // A select at a join whose condition each way in decides is the phi of
        // the arms the ways choose.
        let join_selects: Vec<JoinSelect> = guards
            .decided_join_selects(ssa, &cfg, &cache)
            .into_iter()
            .filter(|join| !touched.contains(&(join.block, join.instruction)))
            .collect();
        let decided_reads: Vec<(usize, usize, SsaVarId, ConstValue<T>)> = guards
            .decided_reads(ssa, &cache)
            .into_iter()
            .filter(|(block, instr, ..)| !touched.contains(&(*block, *instr)))
            .filter(|(block, instr, value, _)| {
                !select_reads
                    .iter()
                    .any(|read| (read.0, read.1, read.2) == (*block, *instr, *value))
            })
            .filter(|(block, instr, ..)| {
                !join_selects
                    .iter()
                    .any(|join| (join.block, join.instruction) == (*block, *instr))
            })
            .collect();

        // A phi operand reading a select its edge decides reads the arm chosen.
        let decided_operands = guards.decided_phi_operands(ssa, &cache);

        let has_changes = !branch_simplifications.is_empty()
            || !comparison_replacements.is_empty()
            || !comparison_simplifications.is_empty()
            || !phi_replacements.is_empty()
            || !decided_reads.is_empty()
            || !select_reads.is_empty()
            || !join_selects.is_empty()
            || !decided_operands.is_empty();

        let edit_result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                // Apply branch simplifications
                for (block_idx, target, is_true) in branch_simplifications {
                    let Some(_) = editor.function().block(block_idx) else {
                        continue;
                    };
                    // See `ranges`: an edge removal cannot require a new phi, so
                    // the cheap repair suffices.
                    editor.fold_terminator_pruning_phis(block_idx, SsaOp::Jump { target })?;
                    changes
                        .record(EventKind::OpaquePredicateRemoved)
                        .at(method_token.clone(), block_idx)
                        .message(format!(
                            "removed opaque predicate (always {})",
                            if is_true { "true" } else { "false" }
                        ));
                    changes
                        .record(EventKind::BranchSimplified)
                        .at(method_token.clone(), block_idx)
                        .message(format!("simplified to unconditional branch to {target}"));
                }

                for (block_idx, phi_idx, predecessor, chosen) in decided_operands {
                    if editor.set_phi_operand(block_idx, phi_idx, predecessor, chosen)? {
                        changes
                            .record(EventKind::ConstantFolded)
                            .at(method_token.clone(), block_idx)
                            .message(format!(
                                "select on the edge from B{predecessor} decided by its branch"
                            ));
                    }
                }

                // Apply comparison replacements (opaque predicates → constant true/false)
                for (block_idx, instr_idx, dest, value) in comparison_replacements {
                    if editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.instruction(instr_idx))
                        .is_none()
                    {
                        continue;
                    }
                    let const_value = ConstValue::from_bool(value);
                    editor.replace_instruction_op(
                        block_idx,
                        instr_idx,
                        SsaOp::Const {
                            dest,
                            value: const_value,
                        },
                    )?;
                    changes
                        .record(EventKind::ConstantFolded)
                        .at(method_token.clone(), instr_idx)
                        .message(format!("opaque predicate → {value}"));
                }

                // Apply comparison simplifications (algebraic transformations)
                for (block_idx, instr_idx, simplification) in comparison_simplifications {
                    if editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.instruction(instr_idx))
                        .is_none()
                    {
                        continue;
                    }
                    match simplification {
                        ComparisonSimplification::SimplerOp { new_op, reason } => {
                            editor.replace_instruction_op(block_idx, instr_idx, new_op)?;
                            changes
                                .record(EventKind::ConstantFolded)
                                .at(method_token.clone(), instr_idx)
                                .message(reason);
                        }
                        ComparisonSimplification::Copy { dest, src, reason } => {
                            editor.replace_instruction_op(
                                block_idx,
                                instr_idx,
                                SsaOp::Copy { dest, src },
                            )?;
                            changes
                                .record(EventKind::ConstantFolded)
                                .at(method_token.clone(), instr_idx)
                                .message(reason);
                        }
                    }
                }

                // A read of a decided select reads the arm chosen. Before the
                // constants below, which insert rows and move the positions.
                for (block_idx, instr_idx, select, chosen) in select_reads {
                    let Some(mut op) = editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.instruction(instr_idx))
                        .map(|instruction| instruction.op().clone())
                    else {
                        continue;
                    };
                    op.replace_uses(select, chosen);
                    editor.replace_instruction_op(block_idx, instr_idx, op)?;
                    changes
                        .record(EventKind::ConstantFolded)
                        .at(method_token.clone(), block_idx)
                        .message("a select its guard decides read as the arm chosen");
                }

                // A read of a decided value reads its constant, materialised just
                // before the reader. Last in each block first, so the positions
                // still to come stand.
                let mut readers = decided_reads;
                readers.sort_by_key(|reader| core::cmp::Reverse((reader.0, reader.1)));
                let mut at = 0usize;
                while let Some(&(block_idx, instr_idx, ..)) = readers.get(at) {
                    let reads: Vec<(SsaVarId, ConstValue<T>)> = readers
                        .iter()
                        .skip(at)
                        .take_while(|(block, instr, ..)| (*block, *instr) == (block_idx, instr_idx))
                        .map(|(_, _, value, constant)| (*value, constant.clone()))
                        .collect();
                    at = at.saturating_add(reads.len());
                    let Some(mut op) = editor
                        .function()
                        .block(block_idx)
                        .and_then(|block| block.instruction(instr_idx))
                        .map(|instruction| instruction.op().clone())
                    else {
                        continue;
                    };
                    let mut constants = Vec::new();
                    for (value, constant) in reads {
                        let var_type = editor
                            .function()
                            .variable(value)
                            .map_or_else(T::unknown_type, |var| var.var_type().clone());
                        let dest = editor.create_fresh_variable(
                            VariableOrigin::Phi,
                            0,
                            DefSite::instruction(block_idx, instr_idx),
                            var_type,
                        );
                        op.replace_uses(value, dest);
                        constants.push(SsaOp::Const {
                            dest,
                            value: constant,
                        });
                    }
                    editor.replace_instruction_op(block_idx, instr_idx, op)?;
                    for constant in constants.into_iter().rev() {
                        editor.insert_instruction(
                            block_idx,
                            instr_idx,
                            SsaInstruction::synthetic(constant),
                        )?;
                    }
                    changes
                        .record(EventKind::ConstantFolded)
                        .at(method_token.clone(), block_idx)
                        .message("a value its guard decides read as its constant");
                }

                // A select a join's ways in decide becomes the phi of their arms,
                // found by what it defines: the constants above may have moved
                // it. An arm the join defines as a constant is stated again at the
                // end of the way that chooses it, ahead of its terminator.
                for join in join_selects {
                    let Some(position) = editor.function().block(join.block).and_then(|block| {
                        block
                            .instructions()
                            .iter()
                            .position(|instruction| instruction.op().dest() == Some(join.dest))
                    }) else {
                        continue;
                    };
                    let origin = editor
                        .function()
                        .variable(join.dest)
                        .map_or(VariableOrigin::Phi, |var| var.origin());
                    let mut phi = PhiNode::new(join.dest, origin);
                    for (pred, value) in join.arms {
                        let restated = editor
                            .function()
                            .recorded_definition(value)
                            .filter(|definition| definition.block_index() == join.block)
                            .and_then(|definition| match definition.op() {
                                SsaOp::Const {
                                    value: constant, ..
                                } => Some(constant.clone()),
                                _ => None,
                            });
                        let value = match restated {
                            Some(constant) => {
                                let at = editor.function().block(pred).map_or(0, |block| {
                                    block.instructions().len().saturating_sub(1)
                                });
                                let var_type = editor
                                    .function()
                                    .variable(value)
                                    .map_or_else(T::unknown_type, |var| var.var_type().clone());
                                let dest = editor.create_fresh_variable(
                                    VariableOrigin::Phi,
                                    0,
                                    DefSite::instruction(pred, at),
                                    var_type,
                                );
                                editor.insert_instruction(
                                    pred,
                                    at,
                                    SsaInstruction::synthetic(SsaOp::Const {
                                        dest,
                                        value: constant,
                                    }),
                                )?;
                                dest
                            }
                            None => value,
                        };
                        phi.add_operand(PhiOperand::new(value, pred));
                    }
                    editor.nop_instruction(join.block, position)?;
                    editor.add_phi(join.block, phi)?;
                    changes
                        .record(EventKind::ConstantFolded)
                        .at(method_token.clone(), join.block)
                        .message(
                            "a select each way into its join decides joined as the arms chosen",
                        );
                }

                // Apply phi replacements: phis whose every operand is the same
                // constant.
                //
                // **What reads the phi is rewritten; the phi itself is not.**
                // The constant goes into a fresh value at the top of the phi's
                // block, which dominates every instruction the phi's result
                // reaches, and those instructions read it instead. The phi is
                // left to dead-code elimination, and a phi still read by another
                // phi - a loop carrying the local, or a handler observing it -
                // stays, because those operands are the local's own flow.
                //
                // Redefining the phi's result with the constant instead - the
                // shape this had - makes a second definition of the local in
                // that block. Where the local is still live into the block, as
                // it is for a handler that observes the local anywhere in the
                // block's protected region, the next rebuild places the same
                // phi again; it folds again, and every round adds one more
                // constant the handler can observe, so none is ever dead. A
                // NetReactor method gained a thousand instructions that way and
                // never converged.
                for (block_idx, phi_result, const_value) in phi_replacements {
                    let var_type = editor
                        .function()
                        .variable(phi_result)
                        .map_or_else(T::unknown_type, |var| var.var_type().clone());
                    let dest = editor.create_fresh_variable(
                        VariableOrigin::Phi,
                        0,
                        DefSite::entry(),
                        var_type,
                    );
                    editor.insert_instruction(
                        block_idx,
                        0,
                        SsaInstruction::synthetic(SsaOp::Const {
                            dest,
                            value: const_value.clone(),
                        }),
                    )?;
                    editor.replace_uses_checked(phi_result, dest);

                    changes
                        .record(EventKind::ConstantFolded)
                        .at(method_token.clone(), block_idx)
                        .message(format!("phi with constant operands → {const_value:?}"));
                }

                Ok(())
            },
        );

        if edit_result.is_err() {
            return Ok(false);
        }

        if !changes.is_empty() {
            for ev in &changes {
                env.events().push(ev.clone());
            }
        }
        Ok(has_changes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::EventLog,
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            ops::AccessQualifiers,
            phi::{PhiNode, PhiOperand},
            variable::{DefSite, VariableOrigin},
        },
        testing::{FactfulTarget, MACHINE, MockTarget, MockType, run_mock_pass_boundary},
    };

    /// Builds `(expr) & 1 == 0` as a branch condition over a value that is
    /// **not** a constant, and reports whether the pass proved it constant.
    ///
    /// The subject is a phi joining two different constants, so no amount of
    /// constant folding can decide the test: only the parity rule can. A
    /// constant subject would make every case decidable by evaluation and the
    /// whole test vacuous.
    ///
    /// `shape` selects the spelling: `consecutive` builds `v * (v + 1)`,
    /// `factored` builds `(v * v) + v`, and `unrelated` builds `v * w` over two
    /// independent phis - the control, which is not always even.
    fn and_one_parity_is_proved(shape: &str) -> bool {
        let mut ssa = SsaFunction::new(0, 12);
        let seed_a = local_at(&mut ssa, 0, 0, 0);
        let seed_b = local_at(&mut ssa, 1, 1, 0);
        let seed_c = local_at(&mut ssa, 2, 0, 1);
        let seed_d = local_at(&mut ssa, 3, 1, 1);
        let v = ssa.create_variable(VariableOrigin::Local(4), 0, DefSite::phi(2), MockType::I32);
        let w = ssa.create_variable(VariableOrigin::Local(5), 0, DefSite::phi(2), MockType::I32);
        let one = local_at(&mut ssa, 6, 2, 0);
        let inner = local_at(&mut ssa, 7, 2, 1);
        let product = local_at(&mut ssa, 8, 2, 2);
        let mask = local_at(&mut ssa, 9, 2, 3);
        let low = local_at(&mut ssa, 10, 2, 4);
        let zero = local_at(&mut ssa, 11, 2, 5);
        let cond = local_at(&mut ssa, 12, 2, 6);

        // Two predecessors feeding different constants, so the phi is unknown.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_c,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_b,
            value: ConstValue::I32(11),
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_d,
            value: ConstValue::I32(4),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi_v = PhiNode::new(v, VariableOrigin::Local(4));
        phi_v.add_operand(PhiOperand::new(seed_a, 0));
        phi_v.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_v);
        let mut phi_w = PhiNode::new(w, VariableOrigin::Local(5));
        phi_w.add_operand(PhiOperand::new(seed_c, 0));
        phi_w.add_operand(PhiOperand::new(seed_d, 1));
        b2.add_phi(phi_w);

        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        match shape {
            "consecutive" => {
                b2.add_instruction(instr(SsaOp::Add {
                    dest: inner,
                    left: v,
                    right: one,
                    flags: None,
                }));
                b2.add_instruction(instr(SsaOp::Mul {
                    dest: product,
                    left: v,
                    right: inner,
                    flags: None,
                }));
            }
            "factored" => {
                b2.add_instruction(instr(SsaOp::Mul {
                    dest: inner,
                    left: v,
                    right: v,
                    flags: None,
                }));
                b2.add_instruction(instr(SsaOp::Add {
                    dest: product,
                    left: inner,
                    right: v,
                    flags: None,
                }));
            }
            "consecutive_conv" => {
                // The shape a lift actually produces: a width conversion sits
                // between the product and the low-bit test. `raw` is created
                // here and nowhere else - a variable registered in a shape that
                // never defines it fails the pass boundary's verifier.
                let raw = local_at(&mut ssa, 13, 2, 7);
                b2.add_instruction(instr(SsaOp::Add {
                    dest: inner,
                    left: v,
                    right: one,
                    flags: None,
                }));
                b2.add_instruction(instr(SsaOp::Mul {
                    dest: raw,
                    left: v,
                    right: inner,
                    flags: None,
                }));
                b2.add_instruction(instr(SsaOp::IntConv {
                    dest: product,
                    operand: raw,
                    target: MockType::I32,
                    overflow_check: false,
                    unsigned: true,
                }));
            }
            _ => {
                b2.add_instruction(instr(SsaOp::Mul {
                    dest: inner,
                    left: v,
                    right: w,
                    flags: None,
                }));
                b2.add_instruction(instr(SsaOp::Copy {
                    dest: product,
                    src: inner,
                }));
            }
        }
        b2.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::And {
            dest: low,
            left: product,
            right: mask,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b2.add_instruction(instr(SsaOp::Ceq {
            dest: cond,
            left: low,
            right: zero,
            signaling: false,
        }));
        b2.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        for idx in [3usize, 4] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_predicates(&mut ssa, "and-one parity", &method, &log)
    }

    /// `(v * (v + 1)) & 1 == 0` is always true: one of two consecutive integers
    /// is even, so the low bit of their product is clear.
    ///
    /// Compilers emit the low-bit test as `& 1`, not `% 2`, so the detector
    /// must accept both spellings.
    #[test]
    fn and_one_parity_of_a_consecutive_product_is_proved() {
        assert!(
            and_one_parity_is_proved("consecutive"),
            "`v * (v + 1) & 1 == 0` must be proved constant"
        );
    }

    /// The factored spelling `(v * v + v) & 1 == 0` is the same fact.
    #[test]
    fn and_one_parity_of_a_factored_product_is_proved() {
        assert!(
            and_one_parity_is_proved("factored"),
            "`(v * v + v) & 1 == 0` must be proved constant"
        );
    }

    /// The shape a real lift produces: a width conversion between the product
    /// and the low-bit test. Without peeling through it the detector sees an
    /// `IntConv` where the arithmetic is.
    #[test]
    fn and_one_parity_through_a_conversion_is_proved() {
        assert!(
            and_one_parity_is_proved("consecutive_conv"),
            "`(conv)(v * (v + 1)) & 1 == 0` must be proved constant"
        );
    }

    /// The same parity fact behind a compare-and-branch terminator, which is
    /// how targets without a flags register end a test. The relation is stated
    /// inline, so the `Branch` arm never sees it.
    #[test]
    fn a_compare_and_branch_parity_terminator_is_proved() {
        let mut ssa = SsaFunction::new(0, 8);
        let seed_a = local_at(&mut ssa, 0, 0, 0);
        let seed_b = local_at(&mut ssa, 1, 1, 0);
        let v = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let one = local_at(&mut ssa, 3, 2, 0);
        let inner = local_at(&mut ssa, 4, 2, 1);
        let product = local_at(&mut ssa, 5, 2, 2);
        let mask = local_at(&mut ssa, 6, 2, 3);
        let low = local_at(&mut ssa, 7, 2, 4);
        let zero = local_at(&mut ssa, 8, 2, 5);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_b,
            value: ConstValue::I32(11),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi_v = PhiNode::new(v, VariableOrigin::Local(2));
        phi_v.add_operand(PhiOperand::new(seed_a, 0));
        phi_v.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_v);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: v,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Mul {
            dest: product,
            left: v,
            right: inner,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::And {
            dest: low,
            left: product,
            right: mask,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: low,
            right: zero,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        for idx in [3usize, 4] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        assert!(
            run_predicates(&mut ssa, "branchcmp parity", &method, &log),
            "`branchcmp (v * (v + 1)) & 1 == 0` must be proved constant"
        );
    }

    /// Builds `branchcmp v <cmp> v` over a value of type `ty` that is no
    /// constant (a phi of two), runs the pass, and returns the terminator it
    /// leaves.
    fn self_compared(ty: MockType, cmp: CmpKind) -> Option<SsaOp<MockTarget>> {
        self_compared_as(ty, cmp, false)
    }

    /// [`self_compared`], with the comparison's `unsigned` flag: on a float,
    /// the unordered compare.
    fn self_compared_as(ty: MockType, cmp: CmpKind, unsigned: bool) -> Option<SsaOp<MockTarget>> {
        let mut ssa = SsaFunction::new(0, 8);
        let seed_a = local_at(&mut ssa, 0, 0, 0);
        let seed_b = local_at(&mut ssa, 1, 1, 0);
        let v = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), ty);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_b,
            value: ConstValue::I32(11),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        let mut phi_v = PhiNode::new(v, VariableOrigin::Local(2));
        phi_v.add_operand(PhiOperand::new(seed_a, 0));
        phi_v.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_v);
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: v,
            right: v,
            cmp,
            unsigned,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);
        for idx in [3usize, 4] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        run_predicates(&mut ssa, "self-compare", &0u32, &log);
        ssa.block(2)
            .and_then(|block| block.control_terminator())
            .cloned()
    }

    /// A compare-and-branch of two constants, `left` and `right`, ending
    /// block 0, whose arms return.
    fn constant_comparison<T: Target<Type = MockType>>(
        left: ConstValue<T>,
        right: ConstValue<T>,
        cmp: CmpKind,
        unsigned: bool,
    ) -> SsaFunction<T> {
        let mut ssa = SsaFunction::new(0, 8);
        let mut b0 = SsaBlock::new(0);
        let mut operands = Vec::new();
        for (at, value) in [left, right].into_iter().enumerate() {
            let dest = ssa.create_variable(
                VariableOrigin::Local(u16::try_from(at).expect("two")),
                0,
                DefSite::instruction(0, at),
                MockType::Ptr,
            );
            b0.add_instruction(instr(SsaOp::Const { dest, value }));
            operands.push(dest);
        }
        b0.add_instruction(instr(SsaOp::BranchCmp {
            left: operands[0],
            right: operands[1],
            cmp,
            unsigned,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        for idx in [1usize, 2] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        ssa
    }

    /// The terminator a compare-and-branch of two constants, `left` and
    /// `right`, ends block 0 with once the pass has run.
    fn constants_compared(
        left: ConstValue<MockTarget>,
        right: ConstValue<MockTarget>,
        cmp: CmpKind,
        unsigned: bool,
    ) -> SsaOp<MockTarget> {
        let mut ssa = constant_comparison(left, right, cmp, unsigned);
        let log: EventLog<MockTarget> = EventLog::new();
        run_predicates(&mut ssa, "address compare", &0u32, &log);
        ssa.block(0)
            .and_then(|block| block.control_terminator())
            .cloned()
            .expect("a terminator")
    }

    /// **Two addresses from one base are compared by their displacements**:
    /// gcc's `deregister_tm_clones` branches on `__TMC_END__ ==
    /// __TMC_LIST__`, two materialisations of one symbol in an image that
    /// lists no clones, and returns. Two symbols order unsigned by their
    /// displacements, and nothing orders them signed.
    #[test]
    fn two_addresses_from_one_base_are_compared_by_their_displacements() {
        let taken = |left, right, cmp, unsigned| match constants_compared(
            ConstValue::Symbol(left),
            ConstValue::Symbol(right),
            cmp,
            unsigned,
        ) {
            SsaOp::Jump { target: 1 } => Some(true),
            SsaOp::Jump { target: 2 } => Some(false),
            _ => None,
        };
        assert_eq!(taken(0x4030, 0x4030, CmpKind::Eq, false), Some(true));
        assert_eq!(taken(0x4030, 0x4038, CmpKind::Eq, false), Some(false));
        assert_eq!(taken(0x4030, 0x4038, CmpKind::Ne, false), Some(true));
        assert_eq!(taken(0x4030, 0x4038, CmpKind::Lt, true), Some(true));
        assert_eq!(taken(0x4030, 0x4038, CmpKind::Ge, true), Some(false));
        assert_eq!(taken(0x4030, 0x4030, CmpKind::Le, false), Some(true));
        assert_eq!(taken(0x4030, 0x4038, CmpKind::Lt, false), None);
    }

    /// **An image object's address is not the null pointer.** which's
    /// `tilde_additional_prefixes` holds `&default_prefixes`, which a fixed
    /// image serves as that address, and its readers test it against zero
    /// before walking it: either order of the operands, `==` and `!=` alike.
    /// The control compares the integer itself, decided as the number it is,
    /// and an unknown integer, which nothing decides.
    #[test]
    fn an_image_objects_address_is_not_null() {
        let taken = |left, right, cmp| match constants_compared(left, right, cmp, false) {
            SsaOp::Jump { target: 1 } => Some(true),
            SsaOp::Jump { target: 2 } => Some(false),
            _ => None,
        };
        let symbol = || ConstValue::Symbol(0x5190);
        let zero = || ConstValue::I64(0);
        assert_eq!(taken(symbol(), zero(), CmpKind::Ne), Some(true));
        assert_eq!(taken(symbol(), zero(), CmpKind::Eq), Some(false));
        assert_eq!(taken(zero(), symbol(), CmpKind::Eq), Some(false));
        assert_eq!(
            taken(ConstValue::I64(0x5190), zero(), CmpKind::Ne),
            Some(true)
        );
    }

    /// **A bound import's address is no null pointer.** An import has no
    /// address in the image, and a weak one is null where nothing defines it
    /// (`if (&__gmon_start__) __gmon_start__();` is left to run); one the host
    /// proves the loader binds is not. The controls: the same import the
    /// host says nothing of, and another import beside a bound one.
    #[test]
    fn a_bound_imports_address_is_not_null() {
        let import = FactfulTarget::IMPORTED + 1;
        let taken = |symbol: u32, bound: bool| {
            let mut ssa = constant_comparison::<FactfulTarget>(
                ConstValue::Symbol(symbol),
                ConstValue::I64(0),
                CmpKind::Ne,
                false,
            );
            if bound {
                ssa.facts_mut().bound_symbols.push(import);
            }
            let log: EventLog<FactfulTarget> = EventLog::new();
            assert!(
                OpaquePredicatePass::new()
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            match ssa.block(0).and_then(|block| block.control_terminator()) {
                Some(SsaOp::Jump { target: 1 }) => Some(true),
                Some(SsaOp::Jump { target: 2 }) => Some(false),
                _ => None,
            }
        };
        assert_eq!(taken(import, true), Some(true));
        assert_eq!(taken(import, false), None, "control: it may be null");
        assert_eq!(
            taken(FactfulTarget::IMPORTED + 2, true),
            None,
            "control: another import"
        );
    }

    /// **A value compared with itself is decided by the relation**, on the
    /// compare-and-branch a target without flags ends a test with: dash's
    /// retry loop ends `branchcmp v98 != v98` and runs once, and it stayed a
    /// loop because only the `Ceq`/`Clt`/`Cgt` path knew the rule.
    #[test]
    fn an_integer_compared_with_itself_is_decided() {
        assert!(
            matches!(
                self_compared(MockType::I64, CmpKind::Ne),
                Some(SsaOp::Jump { target: 4 })
            ),
            "`v != v` never holds"
        );
        assert!(
            matches!(
                self_compared(MockType::I64, CmpKind::Ge),
                Some(SsaOp::Jump { target: 3 })
            ),
            "`v >= v` always holds"
        );
    }

    /// A float compared with itself is decided only where NaN agrees: `v < v`
    /// fails for every float, `v != v` holds for NaN alone.
    #[test]
    fn a_float_compared_with_itself_is_decided_only_where_nan_agrees() {
        assert!(
            matches!(
                self_compared(MockType::F64, CmpKind::Lt),
                Some(SsaOp::Jump { target: 4 })
            ),
            "`v < v` fails for NaN too"
        );
        assert!(
            matches!(
                self_compared(MockType::F64, CmpKind::Ne),
                Some(SsaOp::BranchCmp { .. })
            ),
            "`v != v` is NaN's test"
        );
        // The unordered compare, `v <u v`, holds for NaN: undecided for a
        // float, false for an integer.
        assert!(
            matches!(
                self_compared_as(MockType::F64, CmpKind::Lt, true),
                Some(SsaOp::BranchCmp { .. })
            ),
            "`v <u v` holds for NaN"
        );
        assert!(
            matches!(
                self_compared_as(MockType::I64, CmpKind::Lt, true),
                Some(SsaOp::Jump { target: 4 })
            ),
            "`v <u v` fails for an integer"
        );
    }

    /// The lifted shape: the width conversion sits between the `& 1` and the
    /// comparison, not inside the product. `branchcmp v13 != 0` with
    /// `v13 = conv.u32 v12` is what MIPS and AArch64 lift to at `-O0`,
    /// and it is a different position from
    /// [`Self::and_one_parity_through_a_conversion_is_proved`]'s.
    #[test]
    fn a_parity_test_converted_before_the_comparison_is_proved() {
        let mut ssa = SsaFunction::new(0, 8);
        let seed_a = local_at(&mut ssa, 0, 0, 0);
        let seed_b = local_at(&mut ssa, 1, 1, 0);
        let v = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let one = local_at(&mut ssa, 3, 2, 0);
        let inner = local_at(&mut ssa, 4, 2, 1);
        let product = local_at(&mut ssa, 5, 2, 2);
        let mask = local_at(&mut ssa, 6, 2, 3);
        let low = local_at(&mut ssa, 7, 2, 4);
        let widened = local_at(&mut ssa, 8, 2, 5);
        let zero = local_at(&mut ssa, 9, 2, 6);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_b,
            value: ConstValue::I32(11),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi_v = PhiNode::new(v, VariableOrigin::Local(2));
        phi_v.add_operand(PhiOperand::new(seed_a, 0));
        phi_v.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_v);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: v,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Mul {
            dest: product,
            left: v,
            right: inner,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::And {
            dest: low,
            left: product,
            right: mask,
            flags: None,
        }));
        // The conversion the lift inserts before the comparison.
        b2.add_instruction(instr(SsaOp::IntConv {
            dest: widened,
            operand: low,
            target: MockType::I32,
            overflow_check: false,
            unsigned: true,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: widened,
            right: zero,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        for idx in [3usize, 4] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        assert!(
            run_predicates(&mut ssa, "converted parity", &method, &log),
            "`branchcmp conv((v * (v + 1)) & 1) != 0` must be proved constant"
        );
    }

    /// Builds `test low, low` + `branchflags <condition>` over the parity
    /// value and reports whether the pass proved it constant.
    fn flag_branch_parity_is_proved(condition: FlagCondition) -> bool {
        let mut ssa = SsaFunction::new(0, 8);
        let seed_a = local_at(&mut ssa, 0, 0, 0);
        let seed_b = local_at(&mut ssa, 1, 1, 0);
        let v = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let one = local_at(&mut ssa, 3, 2, 0);
        let inner = local_at(&mut ssa, 4, 2, 1);
        let product = local_at(&mut ssa, 5, 2, 2);
        let mask = local_at(&mut ssa, 6, 2, 3);
        let low = local_at(&mut ssa, 7, 2, 4);
        let tested = local_at(&mut ssa, 8, 2, 5);
        let flags = local_at(&mut ssa, 9, 2, 5);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_b,
            value: ConstValue::I32(11),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi_v = PhiNode::new(v, VariableOrigin::Local(2));
        phi_v.add_operand(PhiOperand::new(seed_a, 0));
        phi_v.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_v);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: v,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Mul {
            dest: product,
            left: v,
            right: inner,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::And {
            dest: low,
            left: product,
            right: mask,
            flags: None,
        }));
        // `test low, low`: a self-and whose flags the branch reads.
        b2.add_instruction(instr(SsaOp::And {
            dest: tested,
            left: low,
            right: low,
            flags: Some(flags),
        }));
        b2.add_instruction(instr(SsaOp::BranchFlags {
            flags,
            condition,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        for idx in [3usize, 4] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_predicates(&mut ssa, "flag-branch parity", &method, &log)
    }

    /// `test ((v * (v + 1)) & 1)` sets the zero flag for every input, so a
    /// branch on `Zero` or `NotZero` is decided. This is the x86 shape.
    #[test]
    fn a_flag_branch_on_a_parity_test_is_proved() {
        assert!(
            flag_branch_parity_is_proved(FlagCondition::Zero),
            "`branchflags zero` over `test (even & 1)` must be proved constant"
        );
        assert!(
            flag_branch_parity_is_proved(FlagCondition::NotZero),
            "`branchflags not_zero` over `test (even & 1)` must be proved constant"
        );
    }

    /// The control: nothing here models carry, sign, overflow or parity bits,
    /// so a branch on one of them must **not** fold however well the tested
    /// value is understood.
    #[test]
    fn a_flag_branch_on_an_unmodelled_bit_is_refused() {
        for condition in [
            FlagCondition::Carry,
            FlagCondition::Overflow,
            FlagCondition::Negative,
            FlagCondition::ParityEven,
        ] {
            assert!(
                !flag_branch_parity_is_proved(condition),
                "{condition:?} is not modelled and must not be decided"
            );
        }
    }

    /// `v + 1` computed as a `lea`: the increment arrives as `PtrAdd` with a
    /// constant displacement, crosses back through `PtrToInt`, and only then
    /// reaches the multiply. This is how x86 at `-O0` spells the consecutive
    /// product, where a factored form uses a real `Add`.
    #[test]
    fn a_pointer_shaped_increment_is_still_a_consecutive_pair() {
        let mut ssa = SsaFunction::new(0, 8);
        let seed_a = local_at(&mut ssa, 0, 0, 0);
        let seed_b = local_at(&mut ssa, 1, 1, 0);
        let v = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let as_ptr = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(2, 0),
            MockType::Ptr,
        );
        let bumped = ssa.create_variable(
            VariableOrigin::Local(4),
            0,
            DefSite::instruction(2, 1),
            MockType::Ptr,
        );
        let back = local_at(&mut ssa, 5, 2, 2);
        let product = local_at(&mut ssa, 6, 2, 3);
        let mask = local_at(&mut ssa, 7, 2, 4);
        let low = local_at(&mut ssa, 8, 2, 5);
        let zero = local_at(&mut ssa, 9, 2, 6);
        let cond = local_at(&mut ssa, 10, 2, 7);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_b,
            value: ConstValue::I32(11),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi_v = PhiNode::new(v, VariableOrigin::Local(2));
        phi_v.add_operand(PhiOperand::new(seed_a, 0));
        phi_v.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_v);
        b2.add_instruction(instr(SsaOp::IntToPtr {
            dest: as_ptr,
            operand: v,
            target: MockType::Ptr,
        }));
        // `lea [v + 1]`
        b2.add_instruction(instr(SsaOp::PtrAdd {
            dest: bumped,
            base: as_ptr,
            index: None,
            stride: 0,
            offset: 1,
            result_type: MockType::Ptr,
        }));
        b2.add_instruction(instr(SsaOp::PtrToInt {
            dest: back,
            operand: bumped,
            target: MockType::I32,
        }));
        b2.add_instruction(instr(SsaOp::Mul {
            dest: product,
            left: v,
            right: back,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::And {
            dest: low,
            left: product,
            right: mask,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b2.add_instruction(instr(SsaOp::Ceq {
            dest: cond,
            left: low,
            right: zero,
            signaling: false,
        }));
        b2.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        for idx in [3usize, 4] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        assert!(
            run_predicates(&mut ssa, "lea increment", &method, &log),
            "`(v * lea[v + 1]) & 1 == 0` must be proved constant"
        );
    }

    /// Builds `(<doubling> + 1) & 1 == 1` and reports whether it was proved.
    ///
    /// `shape` selects how `2 * v` is written: `self_add` is `v + v` (x86),
    /// `shift` is `v << 1` (aarch64, mips), and `variable_shift` is `v << w`
    /// for a non-constant `w` - the control, which proves nothing about parity.
    /// `shift_by_width` is `v << 32` on a 32-bit `v`, and `shift_by_truncated`
    /// is `v << conv.u8(256)`: both are counts the ISA defines (x86 masks a
    /// 32-bit count to five bits, so either leaves `v` unchanged), and neither
    /// proves the doubling.
    ///
    /// `signed_rem` and `unsigned_rem` keep the `v + v` doubling and spell the
    /// low-bit test `% 2` instead of `& 1`, signed and unsigned.
    fn odd_lowbit_is_proved(shape: &str) -> bool {
        let mut ssa = SsaFunction::new(0, 8);
        let seed_a = local_at(&mut ssa, 0, 0, 0);
        let seed_b = local_at(&mut ssa, 1, 1, 0);
        let v = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let w = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(2), MockType::I32);
        let doubled = local_at(&mut ssa, 4, 2, 0);
        let one = local_at(&mut ssa, 5, 2, 1);
        let odd = local_at(&mut ssa, 6, 2, 2);
        let mask = local_at(&mut ssa, 7, 2, 3);
        let low = local_at(&mut ssa, 8, 2, 4);
        let target = local_at(&mut ssa, 9, 2, 5);
        let cond = local_at(&mut ssa, 10, 2, 6);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seed_a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seed_b,
            value: ConstValue::I32(11),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi_v = PhiNode::new(v, VariableOrigin::Local(2));
        phi_v.add_operand(PhiOperand::new(seed_a, 0));
        phi_v.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_v);
        let mut phi_w = PhiNode::new(w, VariableOrigin::Local(3));
        phi_w.add_operand(PhiOperand::new(seed_a, 0));
        phi_w.add_operand(PhiOperand::new(seed_b, 1));
        b2.add_phi(phi_w);

        match shape {
            "self_add" | "signed_rem" | "unsigned_rem" | "bit_zero_extended"
            | "bit_sign_extended" => {
                b2.add_instruction(instr(SsaOp::Add {
                    dest: doubled,
                    left: v,
                    right: v,
                    flags: None,
                }));
            }
            "shift" | "shift_by_width" => {
                let amount = local_at(&mut ssa, 11, 2, 0);
                let count = if shape == "shift" { 1 } else { 32 };
                b2.add_instruction(instr(SsaOp::Const {
                    dest: amount,
                    value: ConstValue::I32(count),
                }));
                b2.add_instruction(instr(SsaOp::Shl {
                    dest: doubled,
                    value: v,
                    amount,
                    flags: None,
                }));
            }
            "shift_by_truncated" => {
                let wide = local_at(&mut ssa, 11, 2, 0);
                let amount = ssa.create_variable(
                    VariableOrigin::Local(12),
                    0,
                    DefSite::instruction(2, 1),
                    MockType::U8,
                );
                b2.add_instruction(instr(SsaOp::Const {
                    dest: wide,
                    value: ConstValue::I32(256),
                }));
                b2.add_instruction(instr(SsaOp::IntConv {
                    dest: amount,
                    operand: wide,
                    target: MockType::U8,
                    overflow_check: false,
                    unsigned: true,
                }));
                b2.add_instruction(instr(SsaOp::Shl {
                    dest: doubled,
                    value: v,
                    amount,
                    flags: None,
                }));
            }
            _ => b2.add_instruction(instr(SsaOp::Shl {
                dest: doubled,
                value: v,
                amount: w,
                flags: None,
            })),
        }
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: odd,
            left: doubled,
            right: one,
            flags: None,
        }));
        if let Some(unsigned) = match shape {
            "signed_rem" => Some(false),
            "unsigned_rem" => Some(true),
            _ => None,
        } {
            b2.add_instruction(instr(SsaOp::Const {
                dest: mask,
                value: ConstValue::I32(2),
            }));
            b2.add_instruction(instr(SsaOp::Rem {
                dest: low,
                left: odd,
                right: mask,
                unsigned,
                flags: None,
            }));
        } else {
            b2.add_instruction(instr(SsaOp::Const {
                dest: mask,
                value: ConstValue::I32(1),
            }));
            b2.add_instruction(instr(SsaOp::And {
                dest: low,
                left: odd,
                right: mask,
                flags: None,
            }));
        }
        // The low bit, narrowed to a single bit and widened back: unsigned it
        // is still `1`, signed that bit is the sign and it is `-1`.
        let tested = match shape {
            "bit_zero_extended" | "bit_sign_extended" => {
                let bit = ssa.create_variable(
                    VariableOrigin::Local(13),
                    0,
                    DefSite::instruction(2, 0),
                    MockType::I1,
                );
                let widened = local_at(&mut ssa, 14, 2, 0);
                b2.add_instruction(instr(SsaOp::IntConv {
                    dest: bit,
                    operand: low,
                    target: MockType::I1,
                    overflow_check: false,
                    unsigned: true,
                }));
                b2.add_instruction(instr(SsaOp::IntConv {
                    dest: widened,
                    operand: bit,
                    target: MockType::I32,
                    overflow_check: false,
                    unsigned: shape == "bit_zero_extended",
                }));
                widened
            }
            _ => low,
        };
        b2.add_instruction(instr(SsaOp::Const {
            dest: target,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::Ceq {
            dest: cond,
            left: tested,
            right: target,
            signaling: false,
        }));
        b2.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);
        for idx in [3usize, 4] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_predicates(&mut ssa, "odd low bit", &method, &log)
    }

    /// An odd low bit narrowed to one bit and zero-extended back is still `1`.
    #[test]
    fn an_odd_low_bit_zero_extended_from_one_bit_is_proved() {
        assert!(
            odd_lowbit_is_proved("bit_zero_extended"),
            "zero-extending the single set bit gives 1"
        );
    }

    /// Sign-extended from a single bit, the set bit is the sign: the value is
    /// `-1`, and `== 1` is false for every input.
    #[test]
    fn an_odd_low_bit_sign_extended_from_one_bit_is_not_proved_one() {
        assert!(
            !odd_lowbit_is_proved("bit_sign_extended"),
            "sign-extending the single set bit gives -1, not 1"
        );
    }

    /// `((v + v) + 1) & 1 == 1` - the x86 spelling of `2 * v + 1`. An even
    /// value plus one is odd, so its low bit is set for every input.
    #[test]
    fn an_odd_low_bit_from_a_self_add_is_proved() {
        assert!(
            odd_lowbit_is_proved("self_add"),
            "`((v + v) + 1) & 1 == 1` must be proved constant"
        );
    }

    /// `((v << 1) + 1) & 1 == 1` - the aarch64 and mips spelling of the same
    /// doubling, which a detector keyed on the self-add alone would miss.
    #[test]
    fn an_odd_low_bit_from_a_shift_is_proved() {
        assert!(
            odd_lowbit_is_proved("shift"),
            "`((v << 1) + 1) & 1 == 1` must be proved constant"
        );
    }

    /// `v << 32` on a 32-bit `v` is not a doubling: x86 masks the count to
    /// `0` and leaves `v` as it was, so the low bit is not known.
    #[test]
    fn a_shift_by_the_full_width_does_not_prove_a_low_bit() {
        assert!(
            !odd_lowbit_is_proved("shift_by_width"),
            "`((v << 32) + 1) & 1 == 1` depends on `v` and must not fold"
        );
    }

    /// `v << conv.u8(256)` shifts by `0`: the count is the truncated value,
    /// not the constant underneath the conversion.
    #[test]
    fn a_shift_count_behind_a_truncation_is_not_read_through_it() {
        assert!(
            !odd_lowbit_is_proved("shift_by_truncated"),
            "`((v << conv.u8(256)) + 1) & 1 == 1` depends on `v` and must not fold"
        );
    }

    /// Signed `(2v + 1) % 2` is `-1` for a negative `v`, so `== 1` is not
    /// decided.
    #[test]
    fn a_signed_remainder_of_an_odd_value_is_not_proved_one() {
        assert!(
            !odd_lowbit_is_proved("signed_rem"),
            "signed `((v + v) + 1) % 2 == 1` is false for negative `v` and must not fold"
        );
    }

    /// Unsigned `(2v + 1) % 2` is `1` for every input: the control for the
    /// signed case above.
    #[test]
    fn an_unsigned_remainder_of_an_odd_value_is_proved_one() {
        assert!(
            odd_lowbit_is_proved("unsigned_rem"),
            "unsigned `((v + v) + 1) % 2 == 1` must be proved constant"
        );
    }

    /// The control: `v << w` for a non-constant `w` is not known to be even -
    /// `w` may be zero - so nothing about the low bit follows.
    #[test]
    fn a_variable_shift_does_not_prove_a_low_bit() {
        assert!(
            !odd_lowbit_is_proved("variable_shift"),
            "`v << w` for unknown `w` proves no parity and must not fold"
        );
    }

    /// The control: `v * w` over two independent values is not always even, so
    /// the low-bit test must **not** fold. Without it the tests above would pass
    /// for a rule that folded every `& 1`.
    #[test]
    fn and_one_parity_of_an_unrelated_product_is_not_proved() {
        assert!(
            !and_one_parity_is_proved("unrelated"),
            "`v * w & 1` is not always zero and must not be folded"
        );
    }

    /// Verifies predicate tri-state conversions and negation.
    #[test]
    fn predicate_result_negate_and_as_bool() {
        assert_eq!(PredicateResult::AlwaysTrue.as_bool(), Some(true));
        assert_eq!(PredicateResult::AlwaysFalse.as_bool(), Some(false));
        assert_eq!(PredicateResult::Unknown.as_bool(), None);

        assert_eq!(
            PredicateResult::AlwaysTrue.negate(),
            PredicateResult::AlwaysFalse
        );
        assert_eq!(
            PredicateResult::AlwaysFalse.negate(),
            PredicateResult::AlwaysTrue
        );
        assert_eq!(PredicateResult::Unknown.negate(), PredicateResult::Unknown);
    }

    /// The values a correlated-pair test builds its two terminators from.
    ///
    /// All four operands come from `LoadArg` and both flag values from an `And`
    /// over them, so nothing here is evaluable: a fold can only come from the
    /// correlation between the two branches, never from constant propagation.
    /// That is the difference between this test and a vacuous one.
    #[derive(Clone, Copy)]
    struct PairValues {
        first: SsaVarId,
        second: SsaVarId,
        third: SsaVarId,
        fourth: SsaVarId,
        flags_of_first_pair: SsaVarId,
        flags_of_second_pair: SsaVarId,
    }

    /// Builds the `jz L; jnz L` shape and returns `b1`'s terminator after the
    /// pass, so a caller can say whether the correlated condition was folded.
    ///
    /// ```text
    /// b0: <guard>          true -> b3, false -> b1
    /// b1: <branch>         true -> b3, false -> b2   (the junk)
    /// b2: return           b3: return
    /// ```
    fn correlated_pair(
        terminators: impl FnOnce(PairValues) -> (SsaOp<MockTarget>, SsaOp<MockTarget>),
    ) -> Option<SsaOp<MockTarget>> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let first = local_at(&mut ssa, 0, 0, 0);
        let second = local_at(&mut ssa, 1, 0, 1);
        let third = local_at(&mut ssa, 2, 0, 2);
        let fourth = local_at(&mut ssa, 3, 0, 3);
        let first_and = local_at(&mut ssa, 4, 0, 4);
        let flags_of_first_pair = local_at(&mut ssa, 5, 0, 4);
        let second_and = local_at(&mut ssa, 6, 0, 5);
        let flags_of_second_pair = local_at(&mut ssa, 7, 0, 5);
        let values = PairValues {
            first,
            second,
            third,
            fourth,
            flags_of_first_pair,
            flags_of_second_pair,
        };
        let (guard, branch) = terminators(values);

        let mut b0 = SsaBlock::new(0);
        for (index, dest) in [first, second, third, fourth].into_iter().enumerate() {
            b0.add_instruction(instr(SsaOp::LoadArg {
                dest,
                arg_index: u16::try_from(index).unwrap_or(0),
            }));
        }
        b0.add_instruction(instr(SsaOp::And {
            dest: first_and,
            left: first,
            right: second,
            flags: Some(flags_of_first_pair),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: second_and,
            left: third,
            right: fourth,
            flags: Some(flags_of_second_pair),
        }));
        b0.add_instruction(instr(guard));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(branch));
        ssa.add_block(b1);

        for index in 2..=3 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        assert!(
            OpaquePredicatePass::new()
                .run(
                    &mut ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        ssa.block(1).and_then(SsaBlock::control_terminator).cloned()
    }

    /// Which condition the inner select of [`nested_select`] reads.
    #[derive(Clone, Copy)]
    enum Inner {
        /// The outer select's own condition.
        Same,
        /// Its `bool.not`.
        Negated,
        /// Another comparison, over other operands.
        Other,
        /// The outer select's condition, read through a load of what the
        /// inner select picks.
        Loaded,
    }

    /// `outer = select c, (inner + k), z` with `inner = select <condition>,
    /// x, y` and `c = ceq a, b`, run through the pass; the inner select's op
    /// afterwards. With `shared`, the function returns `inner` as well, so
    /// the inner select computes for another reader than the outer's arm.
    fn nested_select(inner: Inner, shared: bool) -> SsaOp<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 16);
        let values: Vec<SsaVarId> = (0..13u16)
            .map(|index| local_at(&mut ssa, index, 0, usize::from(index)))
            .collect();
        let [a, b, e, x, y, z, k, c, d, n, inner_value, sum, outer] = values[..] else {
            unreachable!("thirteen values");
        };
        let mut block = SsaBlock::new(0);
        for (index, dest) in [a, b, e, x, y, z, k].into_iter().enumerate() {
            block.add_instruction(instr(SsaOp::LoadArg {
                dest,
                arg_index: u16::try_from(index).unwrap_or(0),
            }));
        }
        block.add_instruction(instr(SsaOp::Ceq {
            dest: c,
            left: a,
            right: b,
            signaling: false,
        }));
        block.add_instruction(instr(SsaOp::Ceq {
            dest: d,
            left: a,
            right: e,
            signaling: false,
        }));
        block.add_instruction(instr(SsaOp::BoolNot { dest: n, value: c }));
        block.add_instruction(instr(SsaOp::Select {
            dest: inner_value,
            condition: match inner {
                Inner::Same | Inner::Loaded => c,
                Inner::Negated => n,
                Inner::Other => d,
            },
            true_val: x,
            false_val: y,
        }));
        block.add_instruction(instr(if matches!(inner, Inner::Loaded) {
            SsaOp::LoadIndirect {
                dest: sum,
                addr: inner_value,
                value_type: MockType::I32,
                access: AccessQualifiers::FLAT,
            }
        } else {
            SsaOp::Add {
                dest: sum,
                left: inner_value,
                right: k,
                flags: None,
            }
        }));
        block.add_instruction(instr(SsaOp::Select {
            dest: outer,
            condition: c,
            true_val: sum,
            false_val: z,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(if shared {
            inner_value
        } else {
            outer
        }))));
        ssa.add_block(block);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        run_predicates(&mut ssa, "nested select", &0u32, &log);
        ssa.block(0)
            .and_then(|block| block.instruction(10))
            .map(|instruction| instruction.op().clone())
            .expect("the inner select's row")
    }

    /// **A select's arm knows its condition.** `select c, (select c, x, y) +
    /// k, z` adds `k` to `x`: the arm is evaluated only where `c` holds, and
    /// the inner select computes for that arm alone. Under `bool.not c` the
    /// inner select is `y`. The controls: an inner select over another
    /// comparison, one the function also returns, which computes for a reader
    /// that knows nothing of `c`, and one an address load reads, which runs
    /// whatever `c` is, stay selects.
    #[test]
    fn a_selects_arm_knows_its_condition() {
        let copy_of = |op: SsaOp<MockTarget>| match op {
            SsaOp::Copy { src, .. } => Some(src),
            _ => None,
        };
        assert_eq!(
            copy_of(nested_select(Inner::Same, false)),
            Some(SsaVarId::from_index(3)),
            "the true arm reads x"
        );
        assert_eq!(
            copy_of(nested_select(Inner::Negated, false)),
            Some(SsaVarId::from_index(4)),
            "under its negation it reads y"
        );
        assert!(matches!(
            nested_select(Inner::Other, false),
            SsaOp::Select { .. }
        ));
        assert!(matches!(
            nested_select(Inner::Same, true),
            SsaOp::Select { .. }
        ));
        assert!(matches!(
            nested_select(Inner::Loaded, false),
            SsaOp::Select { .. }
        ));
    }

    /// **A block knows the guards of the blocks dominating it.** `b0` branches
    /// on `c = ceq a, b` to `b1`, which jumps to `b3`: every way into `b3`
    /// passed `c` true, so its `select c, x, y` is `x` and its own branch on
    /// `c` goes to `b4`. The control: `b2` jumping to `b3` too gives it a way
    /// in where `c` was false, and its branch stays; `b4` and `b5`, one
    /// entered where `c` holds and one where it fails, each return the
    /// select, so no reader alone decides it, and each reads the arm it
    /// chooses, `x` and `y`.
    #[test]
    fn a_block_knows_the_guards_of_the_blocks_dominating_it() {
        let run_with = |both_reach: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
            let a = local_at(&mut ssa, 0, 0, 0);
            let b = local_at(&mut ssa, 1, 0, 1);
            let x = local_at(&mut ssa, 2, 0, 2);
            let y = local_at(&mut ssa, 3, 0, 3);
            let c = local_at(&mut ssa, 4, 0, 4);
            let chosen = local_at(&mut ssa, 5, 3, 0);
            let mut b0 = SsaBlock::new(0);
            for (index, dest) in [a, b, x, y].into_iter().enumerate() {
                b0.add_instruction(instr(SsaOp::LoadArg {
                    dest,
                    arg_index: u16::try_from(index).unwrap_or(0),
                }));
            }
            b0.add_instruction(instr(SsaOp::Ceq {
                dest: c,
                left: a,
                right: b,
                signaling: false,
            }));
            b0.add_instruction(instr(SsaOp::Branch {
                condition: c,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);
            let mut b1 = SsaBlock::new(1);
            b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
            ssa.add_block(b1);
            let mut b2 = SsaBlock::new(2);
            b2.add_instruction(instr(if both_reach {
                SsaOp::Jump { target: 3 }
            } else {
                SsaOp::returns(None)
            }));
            ssa.add_block(b2);
            let mut b3 = SsaBlock::new(3);
            b3.add_instruction(instr(SsaOp::Select {
                dest: chosen,
                condition: c,
                true_val: x,
                false_val: y,
            }));
            b3.add_instruction(instr(SsaOp::Branch {
                condition: c,
                true_target: 4,
                false_target: 5,
            }));
            ssa.add_block(b3);
            let mut b4 = SsaBlock::new(4);
            b4.add_instruction(instr(SsaOp::returns(Some(chosen))));
            ssa.add_block(b4);
            let mut b5 = SsaBlock::new(5);
            b5.add_instruction(instr(SsaOp::returns(Some(chosen))));
            ssa.add_block(b5);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            run_predicates(&mut ssa, "dominated block", &0u32, &log);
            let returned = |at: usize| {
                ssa.block(at)
                    .and_then(|block| block.instructions().last())
                    .and_then(|instruction| instruction.op().uses().first().copied())
            };
            let returns = [returned(4), returned(5)];
            let block = ssa.block(3).expect("b3");
            (
                block
                    .instruction(0)
                    .map(|instruction| instruction.op().clone()),
                block.control_terminator().cloned(),
                returns,
            )
        };
        let (select, branch, _) = run_with(false);
        assert!(
            matches!(select, Some(SsaOp::Copy { src, .. }) if src == SsaVarId::from_index(2)),
            "{select:?}"
        );
        assert!(
            matches!(branch, Some(SsaOp::Jump { target: 4 })),
            "{branch:?}"
        );
        let (select, branch, returns) = run_with(true);
        assert!(matches!(branch, Some(SsaOp::Branch { .. })), "{branch:?}");
        // Each block after it knows its answer, so each reads the arm it
        // chooses, and nothing reads the select.
        assert!(!matches!(select, Some(SsaOp::Select { .. })), "{select:?}");
        assert_eq!(
            returns,
            [Some(SsaVarId::from_index(2)), Some(SsaVarId::from_index(3))],
            "x where c holds, y where it fails"
        );
    }

    /// **A read of a boolean its guard decides is its answer.** `b0` branches
    /// on `c = ceq a, b` to `b1`, which computes `c & x`: `b1` is entered only
    /// where `c` holds, so the `and` reads `true`. The control: a branch on the
    /// integer `a` itself says only that `a` is nonzero, which is no `1`, and
    /// `a & x` keeps reading `a`.
    #[test]
    fn a_read_of_a_boolean_its_guard_decides_is_its_answer() {
        let run_with = |integer: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
            let a = local_at(&mut ssa, 0, 0, 0);
            let b = local_at(&mut ssa, 1, 0, 1);
            let x = local_at(&mut ssa, 2, 0, 2);
            let c = local_at(&mut ssa, 3, 0, 3);
            let masked = local_at(&mut ssa, 4, 1, 0);
            let tested = if integer { a } else { c };
            let mut b0 = SsaBlock::new(0);
            for (index, dest) in [a, b, x].into_iter().enumerate() {
                b0.add_instruction(instr(SsaOp::LoadArg {
                    dest,
                    arg_index: u16::try_from(index).unwrap_or(0),
                }));
            }
            b0.add_instruction(instr(SsaOp::Ceq {
                dest: c,
                left: a,
                right: b,
                signaling: false,
            }));
            b0.add_instruction(instr(SsaOp::Branch {
                condition: tested,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);
            let mut b1 = SsaBlock::new(1);
            b1.add_instruction(instr(SsaOp::And {
                dest: masked,
                left: tested,
                right: x,
                flags: None,
            }));
            b1.add_instruction(instr(SsaOp::returns(Some(masked))));
            ssa.add_block(b1);
            let mut b2 = SsaBlock::new(2);
            b2.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(b2);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            run_predicates(&mut ssa, "guarded read", &0u32, &log);
            let block = ssa.block(1).expect("b1").clone();
            let left = block
                .instructions()
                .iter()
                .find_map(|instruction| match instruction.op() {
                    SsaOp::And { left, .. } => Some(*left),
                    _ => None,
                })
                .expect("the and");
            let constant = block.instructions().iter().any(|instruction| {
                matches!(instruction.op(), SsaOp::Const { dest, value: ConstValue::True } if *dest == left)
            });
            (left == tested, constant)
        };
        assert_eq!(run_with(false), (false, true), "c is read as true");
        assert_eq!(run_with(true), (true, false), "a is no boolean");
    }

    /// How `b0` asks about `x` before `b1` returns it ([`guarded_return`]).
    #[derive(Debug, Clone, Copy)]
    enum Guard {
        /// `branchcmp x <cmp> 7`, `b1` on the arm named.
        Compare(CmpKind, bool),
        /// `branchcmp x == -1` into `b1`.
        EqualMinusOne,
        /// `branch x`, `b1` on the arm named.
        Integer(bool),
        /// `branchflags (x - 7), zero`, `b1` on the arm named.
        SubtractedZero(bool),
        /// `branchflags (x & x), zero` into `b1`.
        TestedZero,
        /// `branch (ceq (x + 3) 7)` into `b1`.
        SumEqual,
        /// `branchflags (narrow x), zero` into `b1`, `narrow` an `i8`
        /// truncation of `x`.
        TruncatedZero,
    }

    /// `b0` loads `x`, typed `ty`, asks `guard` about it and enters `b1`,
    /// which returns `x`; `b2` returns nothing. Returns the constant the
    /// return reads after the pass, `None` where it still reads `x`.
    fn guarded_return(guard: Guard, ty: MockType) -> Option<ConstValue<MockTarget>> {
        guarded_read(guard, ty, None)
    }

    /// How `b1` passes `x` on before returning it ([`guarded_read`]).
    #[derive(Debug, Clone, Copy)]
    enum PassedOn {
        /// A copy.
        Copied,
        /// A sign extension to 64 bits.
        Widened,
    }

    /// [`guarded_return`], with `b1` passing `x` on as `passed` says and
    /// returning what it passes on. Asserts the pass-through still reads `x`:
    /// it is no reader the guard rewrites.
    fn guarded_read(
        guard: Guard,
        ty: MockType,
        passed: Option<PassedOn>,
    ) -> Option<ConstValue<MockTarget>> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
        // A value `b0`'s next instruction defines.
        let var = |ssa: &mut SsaFunction<MockTarget>, b0: &SsaBlock<MockTarget>, ty: MockType| {
            let index = u16::try_from(ssa.var_id_bound()).unwrap_or(0);
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, b0.instructions().len()),
                ty,
            )
        };
        let mut b0 = SsaBlock::new(0);
        let x = var(&mut ssa, &b0, ty);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: x,
            arg_index: 0,
        }));
        let seven = var(&mut ssa, &b0, ty);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        let targets = |enters_on: bool| if enters_on { (1, 2) } else { (2, 1) };
        let minus_one = var(&mut ssa, &b0, ty);
        b0.add_instruction(instr(SsaOp::Const {
            dest: minus_one,
            value: ConstValue::I32(-1),
        }));
        let terminator = match guard {
            Guard::Compare(cmp, enters_on) => {
                let (true_target, false_target) = targets(enters_on);
                SsaOp::BranchCmp {
                    left: x,
                    right: seven,
                    cmp,
                    unsigned: false,
                    true_target,
                    false_target,
                }
            }
            Guard::EqualMinusOne => SsaOp::BranchCmp {
                left: x,
                right: minus_one,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: 1,
                false_target: 2,
            },
            Guard::Integer(enters_on) => {
                let (true_target, false_target) = targets(enters_on);
                SsaOp::Branch {
                    condition: x,
                    true_target,
                    false_target,
                }
            }
            Guard::SubtractedZero(enters_on) => {
                let computed = var(&mut ssa, &b0, ty);
                let flags = var(&mut ssa, &b0, MockType::I32);
                b0.add_instruction(instr(SsaOp::Sub {
                    dest: computed,
                    left: x,
                    right: seven,
                    flags: Some(flags),
                }));
                let (true_target, false_target) = targets(enters_on);
                SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::Zero,
                    true_target,
                    false_target,
                }
            }
            Guard::TestedZero => {
                let computed = var(&mut ssa, &b0, ty);
                let flags = var(&mut ssa, &b0, MockType::I32);
                b0.add_instruction(instr(SsaOp::And {
                    dest: computed,
                    left: x,
                    right: x,
                    flags: Some(flags),
                }));
                SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::Zero,
                    true_target: 1,
                    false_target: 2,
                }
            }
            Guard::SumEqual => {
                let three = var(&mut ssa, &b0, ty);
                b0.add_instruction(instr(SsaOp::Const {
                    dest: three,
                    value: ConstValue::I32(3),
                }));
                let computed = var(&mut ssa, &b0, ty);
                b0.add_instruction(instr(SsaOp::Add {
                    dest: computed,
                    left: x,
                    right: three,
                    flags: None,
                }));
                let condition = var(&mut ssa, &b0, MockType::Bool);
                b0.add_instruction(instr(SsaOp::Ceq {
                    dest: condition,
                    left: computed,
                    right: seven,
                    signaling: false,
                }));
                SsaOp::Branch {
                    condition,
                    true_target: 1,
                    false_target: 2,
                }
            }
            Guard::TruncatedZero => {
                let narrow = var(&mut ssa, &b0, MockType::I8);
                b0.add_instruction(instr(SsaOp::IntConv {
                    dest: narrow,
                    operand: x,
                    target: MockType::I8,
                    overflow_check: false,
                    unsigned: true,
                }));
                let copied = var(&mut ssa, &b0, MockType::I8);
                b0.add_instruction(instr(SsaOp::Copy {
                    dest: copied,
                    src: narrow,
                }));
                let tested = var(&mut ssa, &b0, MockType::I8);
                let flags = var(&mut ssa, &b0, MockType::I32);
                b0.add_instruction(instr(SsaOp::And {
                    dest: tested,
                    left: copied,
                    right: copied,
                    flags: Some(flags),
                }));
                SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::Zero,
                    true_target: 1,
                    false_target: 2,
                }
            }
        };
        b0.add_instruction(instr(terminator));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        let returned = match passed {
            None => x,
            Some(PassedOn::Copied) => {
                let copy =
                    ssa.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(1, 0), ty);
                b1.add_instruction(instr(SsaOp::Copy { dest: copy, src: x }));
                copy
            }
            Some(PassedOn::Widened) => {
                let wide = ssa.create_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(1, 0),
                    MockType::I64,
                );
                b1.add_instruction(instr(SsaOp::IntConv {
                    dest: wide,
                    operand: x,
                    target: MockType::I64,
                    overflow_check: false,
                    unsigned: false,
                }));
                wide
            }
        };
        b1.add_instruction(instr(SsaOp::returns(Some(returned))));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        run_predicates(&mut ssa, "guarded return", &0u32, &log);
        let block = ssa.block(1)?;
        let returned =
            block
                .instructions()
                .iter()
                .find_map(|instruction| match instruction.op() {
                    SsaOp::Return { value, .. } => *value,
                    _ => None,
                })?;
        if passed.is_some() {
            let source =
                block
                    .instructions()
                    .iter()
                    .find_map(|instruction| match instruction.op() {
                        SsaOp::Copy { src, .. } | SsaOp::IntConv { operand: src, .. } => Some(*src),
                        _ => None,
                    });
            assert_eq!(source, Some(x), "the pass-through still reads x:\n{ssa}");
        }
        block
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.op() {
                SsaOp::Const { dest, value } if *dest == returned => Some(value.clone()),
                _ => None,
            })
    }

    /// **A read of a value its guard proves equal to a number is the
    /// number.** `cmp r0, #0; bne L` followed by `mov r1, r0; bl f` passes
    /// `f` the `0` the test proved, through a register that held a pointer:
    /// read as that register, the `0` is the pointer. Every way a guard
    /// proves an equality is a case - a comparison answered equal or not
    /// unequal, a branch on an integer not taken, a zero flag on a
    /// difference or on a test of the value against itself - and so is a
    /// computation undone back to the value, `(x + 3) == 7` proving `x == 4`.
    /// A copy or a conversion of the value only passes it on and keeps
    /// reading it, and what reads the copy reads the number, converted as the
    /// conversion computes it. The controls are answers that prove no single
    /// number - unequal, less than, an integer that is nonzero, a truncation
    /// that is zero - and a float, which `-0.0 == 0.0` keeps from any.
    #[test]
    fn a_read_of_a_value_its_guard_proves_equal_to_a_number_is_the_number() {
        let read = |guard| guarded_return(guard, MockType::I32);
        assert_eq!(
            read(Guard::Compare(CmpKind::Eq, true)),
            Some(ConstValue::I32(7))
        );
        assert_eq!(
            read(Guard::Compare(CmpKind::Ne, false)),
            Some(ConstValue::I32(7))
        );
        assert_eq!(read(Guard::Integer(false)), Some(ConstValue::I32(0)));
        assert_eq!(read(Guard::SubtractedZero(true)), Some(ConstValue::I32(7)));
        assert_eq!(read(Guard::TestedZero), Some(ConstValue::I32(0)));
        assert_eq!(read(Guard::SumEqual), Some(ConstValue::I32(4)));

        assert_eq!(read(Guard::Compare(CmpKind::Eq, false)), None, "unequal");
        assert_eq!(read(Guard::Compare(CmpKind::Lt, true)), None, "less than");
        assert_eq!(read(Guard::Integer(true)), None, "nonzero");
        assert_eq!(
            read(Guard::SubtractedZero(false)),
            None,
            "a nonzero difference"
        );
        assert_eq!(read(Guard::TruncatedZero), None, "a zero low byte");
        assert_eq!(
            guarded_return(Guard::Compare(CmpKind::Eq, true), MockType::F64),
            None,
            "a float"
        );
        assert_eq!(
            guarded_read(
                Guard::Compare(CmpKind::Eq, true),
                MockType::I32,
                Some(PassedOn::Copied)
            ),
            Some(ConstValue::I32(7)),
            "what reads a copy reads the number"
        );
        assert_eq!(
            guarded_read(
                Guard::SubtractedZero(true),
                MockType::I32,
                Some(PassedOn::Widened)
            ),
            Some(ConstValue::I64(7)),
            "and a widening carries it extended"
        );
        assert_eq!(
            guarded_read(Guard::EqualMinusOne, MockType::I32, Some(PassedOn::Widened)),
            Some(ConstValue::I64(-1)),
            "by its sign"
        );
    }

    /// `jz L; jnz L`: the second branch is entered knowing the first was false,
    /// so it goes to `L` unconditionally and the junk behind it is unreachable.
    ///
    /// This is a common anti-disassembly idiom, but nothing about the rule is:
    /// it is the ordinary fact that a block with one incoming edge knows its
    /// guard.
    #[test]
    fn a_complementary_flag_pair_over_one_flags_value_is_decided() {
        let folded = correlated_pair(|values| {
            (
                SsaOp::BranchFlags {
                    flags: values.flags_of_first_pair,
                    condition: FlagCondition::Zero,
                    true_target: 3,
                    false_target: 1,
                },
                SsaOp::BranchFlags {
                    flags: values.flags_of_first_pair,
                    condition: FlagCondition::NotZero,
                    true_target: 3,
                    false_target: 2,
                },
            )
        });
        assert!(
            matches!(folded, Some(SsaOp::Jump { target: 3 })),
            "entering on `!zero` makes `not_zero` true, got {folded:?}"
        );
    }

    /// The same pair with the *same* condition twice is decided the other way:
    /// entering on `!zero` makes a second `zero` test false.
    #[test]
    fn a_repeated_flag_test_takes_the_arm_the_edge_already_chose() {
        let folded = correlated_pair(|values| {
            (
                SsaOp::BranchFlags {
                    flags: values.flags_of_first_pair,
                    condition: FlagCondition::Zero,
                    true_target: 3,
                    false_target: 1,
                },
                SsaOp::BranchFlags {
                    flags: values.flags_of_first_pair,
                    condition: FlagCondition::Zero,
                    true_target: 3,
                    false_target: 2,
                },
            )
        });
        assert!(
            matches!(folded, Some(SsaOp::Jump { target: 2 })),
            "entering on `!zero` makes a second `zero` false, got {folded:?}"
        );
    }

    /// A different flags value decides nothing, however complementary the
    /// conditions look.
    ///
    /// SSA is what makes this check exact: a flags value is single-assignment,
    /// so naming the same one twice *is* the guarantee that nothing clobbered
    /// the flags in between.
    #[test]
    fn complementary_conditions_over_different_flags_are_not_correlated() {
        let folded = correlated_pair(|values| {
            (
                SsaOp::BranchFlags {
                    flags: values.flags_of_first_pair,
                    condition: FlagCondition::Zero,
                    true_target: 3,
                    false_target: 1,
                },
                SsaOp::BranchFlags {
                    flags: values.flags_of_second_pair,
                    condition: FlagCondition::NotZero,
                    true_target: 3,
                    false_target: 2,
                },
            )
        });
        assert!(
            matches!(folded, Some(SsaOp::BranchFlags { .. })),
            "two flag states are two questions, got {folded:?}"
        );
    }

    /// **The case that makes a same-target rule wrong.**
    ///
    /// Compiled code holds adjacent pairs of compare-and-branch rows with one
    /// target and complementary *relations* - `Ge` then `Lt`, `Eq` then `Ne` -
    /// over **different operand pairs**. Nothing follows from that: `a >= b`
    /// being false says nothing about `c < d`, and folding such a pair deletes
    /// live code. The operand check is the whole guard and this test is what
    /// holds it.
    #[test]
    fn complementary_relations_over_different_operands_are_not_correlated() {
        let folded = correlated_pair(|values| {
            (
                SsaOp::BranchCmp {
                    left: values.first,
                    right: values.second,
                    cmp: CmpKind::Ge,
                    unsigned: false,
                    true_target: 3,
                    false_target: 1,
                },
                SsaOp::BranchCmp {
                    left: values.third,
                    right: values.fourth,
                    cmp: CmpKind::Lt,
                    unsigned: false,
                    true_target: 3,
                    false_target: 2,
                },
            )
        });
        assert!(
            matches!(folded, Some(SsaOp::BranchCmp { .. })),
            "`a >= b` says nothing about `c < d`, got {folded:?}"
        );
    }

    /// The same relations over the *same* operands are correlated, which is the
    /// compare-and-branch spelling of the flag pair above.
    #[test]
    fn complementary_relations_over_one_operand_pair_are_decided() {
        let folded = correlated_pair(|values| {
            (
                SsaOp::BranchCmp {
                    left: values.first,
                    right: values.second,
                    cmp: CmpKind::Ge,
                    unsigned: false,
                    true_target: 3,
                    false_target: 1,
                },
                SsaOp::BranchCmp {
                    left: values.first,
                    right: values.second,
                    cmp: CmpKind::Lt,
                    unsigned: false,
                    true_target: 3,
                    false_target: 2,
                },
            )
        });
        assert!(
            matches!(folded, Some(SsaOp::Jump { target: 3 })),
            "entering on `!(a >= b)` makes `a < b` true, got {folded:?}"
        );
    }

    /// Builds a two-block chain of flag tests over one flags value and asks
    /// [`GuardFacts::decided_branch`] about `subject`.
    ///
    /// `guard_block` ends `branchflags f, zero` with its false arm to
    /// `subject`, and `subject` ends `branchflags f, not_zero`, so the
    /// terminator edge alone says `subject`'s branch holds. `handler` installs
    /// a clause whose handler range starts at `subject`, making it a block the
    /// runtime also enters on a throw.
    fn correlated_decision_for(
        guard_block: usize,
        subject: usize,
        handler: bool,
    ) -> Option<(usize, bool)> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let value = local_at(&mut ssa, 0, guard_block, 0);
        let masked = local_at(&mut ssa, 1, guard_block, 1);
        let flags = local_at(&mut ssa, 2, guard_block, 1);
        let exit = 3;
        let junk = 2;
        for index in 0..=3 {
            let mut block = SsaBlock::new(index);
            if index == guard_block {
                block.add_instruction(instr(SsaOp::LoadArg {
                    dest: value,
                    arg_index: 0,
                }));
                block.add_instruction(instr(SsaOp::And {
                    dest: masked,
                    left: value,
                    right: value,
                    flags: Some(flags),
                }));
                block.add_instruction(instr(SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::Zero,
                    true_target: exit,
                    false_target: subject,
                }));
            } else if index == subject {
                block.add_instruction(instr(SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::NotZero,
                    true_target: exit,
                    false_target: junk,
                }));
            } else {
                block.add_instruction(instr(SsaOp::returns(None)));
            }
            ssa.add_block(block);
        }
        if handler {
            ssa.set_exception_handlers(vec![SsaExceptionHandler {
                flags: 0,
                try_offset: 0,
                try_length: 1,
                handler_offset: 1,
                handler_length: 1,
                class_token_or_filter: 0,
                protected_range: BlockRange::new(guard_block, guard_block.saturating_add(1)),
                handler_range: BlockRange::new(subject, subject.saturating_add(1)),
                filter_range: None,
                selector: None,
            }]);
        }
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let exception_blocks = ssa.exception_blocks();
        let cache = DefinitionCache::build(&ssa);
        GuardFacts::of(
            &ssa,
            &cfg,
            &exception_blocks,
            &cache,
            MACHINE.ptr_size,
            None,
        )
        .decided_branch(&ssa, subject, &cache)
    }

    /// The control for the two refusals below: an ordinary block entered
    /// along its one terminator edge is decided.
    #[test]
    fn an_ordinary_single_edge_block_is_correlated() {
        assert_eq!(correlated_decision_for(0, 1, false), Some((3, true)));
    }

    /// A handler entry is also entered by the runtime on a throw from the
    /// protected region, in whatever state the flags hold, so its one
    /// terminator predecessor decides nothing about it.
    #[test]
    fn a_handler_entry_with_one_terminator_predecessor_is_not_correlated() {
        assert_eq!(correlated_decision_for(0, 1, true), None);
    }

    /// The entry block is entered by the caller, whatever its predecessor
    /// list says. Valid IR gives it none; this is the malformed case where it
    /// has exactly one, which must still decide nothing.
    #[test]
    fn the_entry_block_is_not_correlated_even_with_one_predecessor() {
        assert_eq!(correlated_decision_for(1, 0, false), None);
    }

    /// **A copy of a constant is that constant** to every probe: a copy is
    /// the identity, so `(x ^ x) == copy(0)`, `copy(3) == 3`,
    /// `ceq(x, x) == copy(1)` and a branch on `x & copy(0)` are decided as
    /// they are with the constant itself. The controls compare the same
    /// shapes against a copy of an argument, which decides nothing.
    #[test]
    fn a_copy_of_a_constant_is_that_constant() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 12);
        let var = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
            local_at(ssa, index, 0, usize::from(index))
        };
        let x = var(&mut ssa, 0);
        let xored = var(&mut ssa, 1);
        let zero = var(&mut ssa, 2);
        let zero_copy = var(&mut ssa, 3);
        let one = var(&mut ssa, 4);
        let one_copy = var(&mut ssa, 5);
        let three = var(&mut ssa, 6);
        let three_copy = var(&mut ssa, 7);
        let self_equal = var(&mut ssa, 8);
        let masked = var(&mut ssa, 9);
        let x_copy = var(&mut ssa, 10);
        let mut b0 = SsaBlock::new(0);
        for op in [
            SsaOp::LoadArg {
                dest: x,
                arg_index: 0,
            },
            SsaOp::Xor {
                dest: xored,
                left: x,
                right: x,
                flags: None,
            },
            SsaOp::Const {
                dest: zero,
                value: ConstValue::I32(0),
            },
            SsaOp::Copy {
                dest: zero_copy,
                src: zero,
            },
            SsaOp::Const {
                dest: one,
                value: ConstValue::I32(1),
            },
            SsaOp::Copy {
                dest: one_copy,
                src: one,
            },
            SsaOp::Const {
                dest: three,
                value: ConstValue::I32(3),
            },
            SsaOp::Copy {
                dest: three_copy,
                src: three,
            },
            SsaOp::Ceq {
                dest: self_equal,
                left: x,
                right: x,
                signaling: false,
            },
            SsaOp::And {
                dest: masked,
                left: x,
                right: zero_copy,
                flags: None,
            },
            SsaOp::Copy {
                dest: x_copy,
                src: x,
            },
            SsaOp::returns(None),
        ] {
            b0.add_instruction(instr(op));
        }
        ssa.add_block(b0);
        ssa.recompute_uses();
        let cache = DefinitionCache::build(&ssa);
        let analyzer = PredicateAnalyzer { cache: &cache };

        assert_eq!(
            analyzer.analyze_equality(xored, zero_copy, 0),
            PredicateResult::AlwaysTrue
        );
        assert_eq!(
            analyzer.analyze_equality(three_copy, three, 0),
            PredicateResult::AlwaysTrue
        );
        assert_eq!(
            analyzer.analyze_equality(three_copy, one_copy, 0),
            PredicateResult::AlwaysFalse
        );
        assert_eq!(
            analyzer.analyze_equality(self_equal, one_copy, 0),
            PredicateResult::AlwaysTrue
        );
        assert_eq!(
            analyzer.analyze_equality(self_equal, zero_copy, 0),
            PredicateResult::AlwaysFalse
        );
        let mut visited = VarSet::new(ssa.var_id_bound());
        assert_eq!(
            analyzer.analyze_branch(masked, &mut visited),
            PredicateResult::AlwaysFalse
        );

        // Controls: a copy of an argument is no constant.
        assert_eq!(
            analyzer.analyze_equality(xored, x_copy, 0),
            PredicateResult::Unknown
        );
        assert_eq!(
            analyzer.analyze_equality(three_copy, x_copy, 0),
            PredicateResult::Unknown
        );
    }

    /// `analyze_branch` shares one cycle-detector set across every branch, so
    /// it must hand the set back empty - after a decided chain, an undecided
    /// one, and a copy cycle alike - or a later branch would read a stale mark
    /// as a cycle.
    #[test]
    fn the_branch_walk_leaves_its_scratch_set_empty() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 6);
        let one = local_at(&mut ssa, 0, 0, 0);
        let decided = local_at(&mut ssa, 1, 0, 1);
        let arg = local_at(&mut ssa, 2, 0, 2);
        let undecided = local_at(&mut ssa, 3, 0, 3);
        let cycle_a = local_at(&mut ssa, 4, 0, 4);
        let cycle_b = local_at(&mut ssa, 5, 0, 5);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Copy {
            dest: decided,
            src: one,
        }));
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: arg,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Copy {
            dest: undecided,
            src: arg,
        }));
        b0.add_instruction(instr(SsaOp::Copy {
            dest: cycle_a,
            src: cycle_b,
        }));
        b0.add_instruction(instr(SsaOp::Copy {
            dest: cycle_b,
            src: cycle_a,
        }));
        b0.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let cache = DefinitionCache::build(&ssa);

        let analyzer = PredicateAnalyzer { cache: &cache };
        let mut visited = VarSet::new(ssa.var_id_bound());
        for (condition, expected) in [
            (decided, PredicateResult::AlwaysTrue),
            (undecided, PredicateResult::Unknown),
            (cycle_a, PredicateResult::Unknown),
        ] {
            assert_eq!(analyzer.analyze_branch(condition, &mut visited), expected);
            assert!(visited.is_empty(), "marks left behind after {condition:?}");
        }
    }

    /// A loop carrying one constant through two phis, and whether a second
    /// constant enters the header.
    ///
    /// `B1: head = phi(seven@B0, tail@B2, other@B4)`, `B2: tail = phi(head@B1)`
    /// loops back, so each phi's value hangs on the other's. `other` is `7` or
    /// `8` as `conflicting` says.
    fn constants_of_a_phi_cycle(
        conflicting: bool,
    ) -> (
        BTreeMap<SsaVarId, ConstValue<MockTarget>>,
        SsaVarId,
        SsaVarId,
    ) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 5);
        let seven = local_at(&mut ssa, 0, 0, 0);
        let condition = local_at(&mut ssa, 1, 1, 0);
        let other = local_at(&mut ssa, 2, 4, 0);
        let head = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(1), MockType::I32);
        let tail = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut head_phi = PhiNode::new(head, VariableOrigin::Local(3));
        head_phi.add_operand(PhiOperand::new(seven, 0));
        head_phi.add_operand(PhiOperand::new(tail, 2));
        head_phi.add_operand(PhiOperand::new(other, 4));
        b1.add_phi(head_phi);
        b1.add_instruction(instr(SsaOp::LoadArg {
            dest: condition,
            arg_index: 0,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut tail_phi = PhiNode::new(tail, VariableOrigin::Local(3));
        tail_phi.add_operand(PhiOperand::new(head, 1));
        b2.add_phi(tail_phi);
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(Some(head))));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Const {
            dest: other,
            value: ConstValue::I32(if conflicting { 8 } else { 7 }),
        }));
        b4.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b4);
        ssa.recompute_uses();

        (analyze_phi_constants(&ssa), head, tail)
    }

    /// A constant carried around a loop is the constant at every phi of it.
    ///
    /// Each phi of the cycle has the other as an operand, so a phi decided
    /// only once its operands are known constant waits on the other forever.
    /// Registers that hold an address across a loop in lifted code are this
    /// shape, and every access through them went unresolved.
    #[test]
    fn a_constant_carried_around_a_loop_is_seen_at_every_phi() {
        let (constants, head, tail) = constants_of_a_phi_cycle(false);
        assert_eq!(constants.get(&head), Some(&ConstValue::I32(7)));
        assert_eq!(constants.get(&tail), Some(&ConstValue::I32(7)));
    }

    /// The control: a different constant entering the cycle anywhere makes
    /// every phi of it unknown.
    #[test]
    fn a_second_constant_entering_the_loop_leaves_it_unknown() {
        let (constants, head, tail) = constants_of_a_phi_cycle(true);
        assert!(!constants.contains_key(&head), "{constants:?}");
        assert!(!constants.contains_key(&tail), "{constants:?}");
    }

    fn instr<T: Target>(op: SsaOp<T>) -> SsaInstruction<T> {
        SsaInstruction::synthetic(op)
    }

    fn local_at(
        ssa: &mut SsaFunction<MockTarget>,
        idx: u16,
        block: usize,
        instr: usize,
    ) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    fn run_predicates(
        ssa: &mut SsaFunction<MockTarget>,
        label: &str,
        method: &u32,
        log: &EventLog<MockTarget>,
    ) -> bool {
        run_mock_pass_boundary(ssa, label, |ssa| {
            OpaquePredicatePass::new().run(
                ssa,
                &PassEnv::new(method, log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        })
    }

    /// Which branch [`select_into_its_branch`] ends B0 with.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SelectGuard {
        /// `branch c`, the select's own `x == 61`, into B2 where it holds.
        Same,
        /// `branch d`, `x == 62`, into B2 where it holds.
        Other,
        /// `branch d`, `x == 62`, into B2 where it fails.
        OtherFails,
    }

    /// How B2 of [`select_into_its_branch`] reads the select.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SelectReader {
        /// B1 jumps to B2 too, and B2 returns `phi(B0: r, B1: seven)`.
        Phi,
        /// B1 and B2, each entered from B0 alone, return `r`.
        Return,
        /// B0 computes `t = r + x` for B2 alone, which returns it, and B1
        /// returns `seven`.
        Through,
    }

    /// `r = select(c, 23, x)` in B0, `c` being `x == 61`, ending in a branch
    /// `guard` names into B2 or B1; B1 sets `seven`, and B2 reads `r` as
    /// `reader` says. Returns what B2 reads for `r` once the pass ran, a copy
    /// read through: `twenty_three`, `r`, or what else it carries.
    fn select_into_its_branch(guard: SelectGuard, reader: SelectReader) -> SsaVarId {
        let mut ssa = SsaFunction::new(1, 8);
        let x = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let sixty_one = local_at(&mut ssa, 1, 0, 0);
        let c = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 1),
            MockType::Bool,
        );
        let twenty_three = local_at(&mut ssa, 3, 0, 2);
        let r = local_at(&mut ssa, 4, 0, 3);
        let sixty_two = local_at(&mut ssa, 5, 0, 4);
        let d = ssa.create_variable(
            VariableOrigin::Local(6),
            0,
            DefSite::instruction(0, 5),
            MockType::Bool,
        );
        let seven = local_at(&mut ssa, 7, 1, 0);
        let joined =
            ssa.create_variable(VariableOrigin::Local(8), 0, DefSite::phi(2), MockType::I32);
        let t = (reader == SelectReader::Through).then(|| local_at(&mut ssa, 9, 0, 6));

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: sixty_one,
            value: ConstValue::I32(61),
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: c,
            left: x,
            right: sixty_one,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: twenty_three,
            value: ConstValue::I32(23),
        }));
        b0.add_instruction(instr(SsaOp::Select {
            dest: r,
            condition: c,
            true_val: twenty_three,
            false_val: x,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: sixty_two,
            value: ConstValue::I32(62),
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: d,
            left: x,
            right: sixty_two,
            signaling: false,
        }));
        if let Some(t) = t {
            b0.add_instruction(instr(SsaOp::Add {
                dest: t,
                left: r,
                right: x,
                flags: None,
            }));
        }
        let (true_target, false_target) = if guard == SelectGuard::OtherFails {
            (1, 2)
        } else {
            (2, 1)
        };
        b0.add_instruction(instr(SsaOp::Branch {
            condition: if guard == SelectGuard::Same { c } else { d },
            true_target,
            false_target,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        let mut b2 = SsaBlock::new(2);
        match reader {
            SelectReader::Phi => {
                b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
                let mut phi = PhiNode::new(joined, VariableOrigin::Local(8));
                phi.add_operand(PhiOperand::new(r, 0));
                phi.add_operand(PhiOperand::new(seven, 1));
                b2.add_phi(phi);
                b2.add_instruction(instr(SsaOp::returns(Some(joined))));
            }
            SelectReader::Return => {
                b1.add_instruction(instr(SsaOp::returns(Some(r))));
                b2.add_instruction(instr(SsaOp::returns(Some(r))));
            }
            SelectReader::Through => {
                b1.add_instruction(instr(SsaOp::returns(Some(seven))));
                b2.add_instruction(instr(SsaOp::returns(t)));
            }
        }
        ssa.add_block(b1);
        ssa.add_block(b2);
        ssa.recompute_uses();
        let log = EventLog::new();
        run_predicates(&mut ssa, "select into its branch", &0, &log);
        let carried = match reader {
            SelectReader::Phi => ssa
                .block(2)
                .and_then(|block| block.phi_nodes().first())
                .and_then(|phi| {
                    phi.operands()
                        .iter()
                        .find(|operand| operand.predecessor() == 0)
                })
                .map(PhiOperand::value)
                .expect("the phi keeps B0's operand"),
            SelectReader::Return => ssa
                .block(2)
                .and_then(|block| block.instructions().last())
                .and_then(|instruction| instruction.op().uses().first().copied())
                .expect("B2 returns a value"),
            SelectReader::Through => match t.and_then(|t| ssa.get_definition(t)) {
                Some(SsaOp::Add { left, .. }) => *left,
                other => panic!("t is r + x: {other:?}"),
            },
        };
        let carried = match ssa.get_definition(carried) {
            Some(SsaOp::Copy { src, .. }) => *src,
            _ => carried,
        };
        let defined = ssa.get_definition(carried).cloned();
        match defined {
            Some(SsaOp::Const {
                value: ConstValue::I32(23),
                ..
            }) => twenty_three,
            Some(SsaOp::Select { .. }) => r,
            _ => carried,
        }
    }

    /// **A select a branch on its own condition reads is, on the branch's
    /// edge, the arm the edge takes.** `select(x == 61, 23, x)` carried into
    /// the block a branch on `x == 61` enters when it holds is `23` there: the
    /// value the function was entered with never arrives that way, as it does
    /// in Thumb's `it eq; moveq r0, #23; beq L`. Entered where `x == 62`
    /// holds, `x` is 62, not 61, and the select is `x`. The control: entered
    /// where `x == 62` fails, nothing is known of `x == 61`, and the select
    /// stays.
    #[test]
    fn a_select_its_branch_decides_is_the_arm_the_edge_takes() {
        let decided = select_into_its_branch(SelectGuard::Same, SelectReader::Phi);
        let other = select_into_its_branch(SelectGuard::Other, SelectReader::Phi);
        let control = select_into_its_branch(SelectGuard::OtherFails, SelectReader::Phi);
        assert_eq!(
            [decided.index(), other.index(), control.index()],
            [3, 0, 4],
            "the edge carries 23 where its branch decides the select true, x where it decides it \
             false, and the select where it decides nothing"
        );
    }

    /// **A select read where its condition is decided reads the arm
    /// chosen.** `select(x == 61, 23, x)` returned by the block a branch on
    /// `x == 61` enters when it holds returns `23`, as Thumb's `it ne; movne
    /// r4, sp; beq L` hands `sp` to the call on the path where `ne` held.
    /// Entered where `x == 62` holds, it returns `x`; the control, entered
    /// where `x == 62` fails, returns the select. Each block returning it
    /// reads its own arm. So does a computation the decided block alone
    /// reads, `r + x` computed before the branch: Thumb's `itttt eq; addeq r3,
    /// r0, #0x68; ...; addeq r3, sp; strbeq r2, [r3, #-86]` stores through
    /// it.
    #[test]
    fn a_select_read_where_its_condition_is_decided_reads_the_arm_chosen() {
        for reader in [SelectReader::Return, SelectReader::Through] {
            let decided = select_into_its_branch(SelectGuard::Same, reader);
            let other = select_into_its_branch(SelectGuard::Other, reader);
            let control = select_into_its_branch(SelectGuard::OtherFails, reader);
            assert_eq!(
                [decided.index(), other.index(), control.index()],
                [3, 0, 4],
                "the read is 23 where the guard decides the select true, x where it decides it \
                 false, and the select where it decides nothing"
            );
        }
    }

    /// Where [`join_select`] defines the value the select chooses where `c`
    /// holds.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum JoinArm {
        /// The constant `0`, in `B0`.
        Before,
        /// The constant `0`, in `B2` before the select.
        Constant,
        /// `x + p`, computed in `B2` before the select.
        Computed,
    }

    /// `B0` computes `c = x < p` and branches on it to the join `B2` where it
    /// holds, or to `B1`, which loads `v` from `p` where it fails; `B2` joins
    /// `j = phi(B0: x, B1: v)` and returns `select(c, k, j)`, `k` as `arm`
    /// says. Returns `B2`'s phis and its instructions once the pass ran.
    fn join_select(arm: JoinArm) -> (Vec<PhiNode>, Vec<SsaOp<MockTarget>>) {
        let inside = arm != JoinArm::Before;
        let mut ssa = SsaFunction::new(2, 8);
        let x = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let p = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let zero = if inside {
            local_at(&mut ssa, 2, 2, 0)
        } else {
            local_at(&mut ssa, 2, 0, 0)
        };
        let c = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(0, 1),
            MockType::Bool,
        );
        let v = local_at(&mut ssa, 4, 1, 0);
        let j = ssa.create_variable(VariableOrigin::Local(5), 0, DefSite::phi(2), MockType::I32);
        let s = local_at(&mut ssa, 6, 2, usize::from(inside));
        let mut b0 = SsaBlock::new(0);
        let zero_op = if arm == JoinArm::Computed {
            SsaOp::Add {
                dest: zero,
                left: x,
                right: p,
                flags: None,
            }
        } else {
            SsaOp::Const {
                dest: zero,
                value: ConstValue::I32(0),
            }
        };
        if !inside {
            b0.add_instruction(instr(zero_op.clone()));
        }
        b0.add_instruction(instr(SsaOp::Clt {
            dest: c,
            left: x,
            right: p,
            unsigned: false,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: c,
            true_target: 2,
            false_target: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::LoadIndirect {
            dest: v,
            addr: p,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(j, VariableOrigin::Local(5));
        phi.add_operand(PhiOperand::new(x, 0));
        phi.add_operand(PhiOperand::new(v, 1));
        b2.add_phi(phi);
        if inside {
            b2.add_instruction(instr(zero_op));
        }
        b2.add_instruction(instr(SsaOp::Select {
            dest: s,
            condition: c,
            true_val: zero,
            false_val: j,
        }));
        b2.add_instruction(instr(SsaOp::returns(Some(s))));
        ssa.add_block(b2);
        ssa.recompute_uses();
        let log = EventLog::new();
        run_predicates(&mut ssa, "join select", &0, &log);
        let block = ssa.block(2).expect("B2");
        (
            block.phi_nodes().to_vec(),
            block
                .instructions()
                .iter()
                .map(|instruction| instruction.op().clone())
                .collect(),
        )
    }

    /// **A select at a join whose condition each way in decides is the phi
    /// of the arms they choose.** Thumb's `ite lt; movlt r0, #0; ldrge r0,
    /// [r1, r0, lsl #2]` loads from an arm entered where `lt` fails and joins
    /// the way past it where `lt` holds: `select(lt, 0, phi(r0, load))` is
    /// `phi(0, load)`, the zero from the way where `lt` held and the load from
    /// the arm. A zero the join itself defines is stated again at the end of
    /// the way that chooses it, as ARM's `movlt r0, #0` past the arm does.
    /// The control: `x + p` computed in the join is not there at the end of
    /// either way in, and the select stays.
    #[test]
    fn a_select_each_way_into_its_join_decides_is_the_phi_of_their_arms() {
        let (phis, ops) = join_select(JoinArm::Before);
        let joined: Vec<Vec<(usize, u32)>> = phis
            .iter()
            .map(|phi| {
                let mut operands: Vec<(usize, u32)> = phi
                    .operands()
                    .iter()
                    .map(|operand| (operand.predecessor(), operand.value().as_u32()))
                    .collect();
                operands.sort_unstable();
                operands
            })
            .collect();
        assert!(
            joined.contains(&vec![(0, 2), (1, 4)]),
            "the zero from B0 and the load from B1: {joined:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(op, SsaOp::Select { .. })),
            "{ops:?}"
        );
        let (phis, ops) = join_select(JoinArm::Constant);
        assert!(
            !ops.iter().any(|op| matches!(op, SsaOp::Select { .. })),
            "{ops:?}"
        );
        assert!(
            phis.iter().any(|phi| {
                phi.origin() == VariableOrigin::Local(6)
                    && phi.operands().len() == 2
                    && phi
                        .operands()
                        .iter()
                        .any(|operand| operand.predecessor() == 0)
                    && phi
                        .operands()
                        .iter()
                        .any(|operand| operand.predecessor() == 1)
            }),
            "the load from B1 and a zero restated in B0: {phis:?}"
        );
        let (_, ops) = join_select(JoinArm::Computed);
        assert!(
            ops.iter().any(|op| matches!(op, SsaOp::Select { .. })),
            "the control keeps its select: {ops:?}"
        );
    }

    /// `B0` computes `c = x < p` and `n = c == false`, then branches on `c`
    /// (`compared`: `branchcmp c == false`, into `B1` where it is false) to
    /// `B1`, which returns `select(read, a, b)` for `read` the boolean
    /// `asked` names, or to `B2`. Returns what `B1` returns once the pass ran.
    fn truth_tested(compared: bool, ask_negated: bool) -> SsaVarId {
        let mut ssa = SsaFunction::new(4, 8);
        let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
            ssa.create_variable(
                VariableOrigin::Argument(index),
                0,
                DefSite::entry(),
                MockType::I32,
            )
        };
        let x = argument(&mut ssa, 0);
        let p = argument(&mut ssa, 1);
        let a = argument(&mut ssa, 2);
        let b = argument(&mut ssa, 3);
        let boolean = |ssa: &mut SsaFunction<MockTarget>, index: u16, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, at),
                MockType::Bool,
            )
        };
        let c = boolean(&mut ssa, 4, 0);
        let f = boolean(&mut ssa, 5, 1);
        let n = boolean(&mut ssa, 6, 2);
        let s = local_at(&mut ssa, 7, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Clt {
            dest: c,
            left: x,
            right: p,
            unsigned: false,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: f,
            value: ConstValue::False,
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: n,
            left: c,
            right: f,
            signaling: false,
        }));
        b0.add_instruction(instr(if compared {
            SsaOp::BranchCmp {
                left: c,
                right: f,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: 1,
                false_target: 2,
            }
        } else {
            SsaOp::Branch {
                condition: c,
                true_target: 1,
                false_target: 2,
            }
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Select {
            dest: s,
            condition: if ask_negated { n } else { c },
            true_val: a,
            false_val: b,
        }));
        b1.add_instruction(instr(SsaOp::returns(Some(s))));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);
        ssa.recompute_uses();
        let log = EventLog::new();
        run_predicates(&mut ssa, "truth tested", &0, &log);
        let returned = ssa
            .block(1)
            .and_then(|block| block.instructions().last())
            .and_then(|instruction| instruction.op().uses().first().copied())
            .expect("B1 returns a value");
        match ssa.get_definition(returned) {
            Some(SsaOp::Copy { src, .. }) => *src,
            _ => returned,
        }
    }

    /// **A boolean compared with a truth value asks the boolean's question.**
    /// `n = (x < p) == false` is `x >= p`: in the block a branch on `x < p`
    /// enters where it holds, `select(n, a, b)` is `b`, as ARM's `movlt` reads
    /// its flag condition back as a value tested against zero. So is a branch
    /// stating the test itself: `branchcmp c == false` enters where `c`
    /// fails, and `select(c, a, b)` there is `b`.
    #[test]
    fn a_boolean_compared_with_a_truth_value_asks_its_question() {
        let b = SsaVarId::from_index(3);
        assert_eq!(truth_tested(false, true), b, "n is false where c holds");
        assert_eq!(truth_tested(true, false), b, "c is false where c == false");
    }

    fn build_branch_with_opaque_predicate() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);

        // Blocks are added in index order: a block's position in the list is
        // what every terminator target and definition site names.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v1,
            left: v0,
            right: v0,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v1,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        // B1 and B2 are the branch targets.
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();
        ssa
    }

    /// A self-equality comparison `x == x` is detected and its branch simplified.
    #[test]
    fn opaque_predicate_self_equality_detected() {
        let mut ssa = build_branch_with_opaque_predicate();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 3u32;
        let changed = run_predicates(&mut ssa, "self-equality predicate", &method, &log);
        // The Ceq(v0, v0) should be detected as always true
        assert!(changed, "self-equality opaque predicate should be detected");
        assert!(log.has(EventKind::OpaquePredicateRemoved) || log.has(EventKind::BranchSimplified));
    }

    /// `(x ^ x) == 0` is recognized as always true and its branch simplified.
    #[test]
    fn xor_self_always_zero_in_predicate() {
        let mut ssa = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v_xor = local_at(&mut ssa, 1, 0, 1);
        let v_zero = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Xor {
            dest: v_xor,
            left: v0,
            right: v0,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_zero,
            left: v_xor,
            right: v0,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_zero,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "xor-self predicate", &method, &log);
        assert!(changed, "xor-self pattern should be detected");
    }

    /// The comparison `compare(dest, combine(x, y), 0)` of two arguments of
    /// type `ty`, against a zero of `zero`, once the pass has run: whatever
    /// the pass left defining the result.
    fn compared_through(
        ty: MockType,
        zero: ConstValue<MockTarget>,
        combine: fn(SsaVarId, SsaVarId, SsaVarId) -> SsaOp<MockTarget>,
        compare: fn(SsaVarId, SsaVarId, SsaVarId) -> SsaOp<MockTarget>,
    ) -> (SsaOp<MockTarget>, SsaVarId) {
        let mut ssa = SsaFunction::new(2, 3);
        let x = ssa.create_variable(VariableOrigin::Argument(0), 0, DefSite::entry(), ty);
        let y = ssa.create_variable(VariableOrigin::Argument(1), 0, DefSite::entry(), ty);
        let combined =
            ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::instruction(0, 0), ty);
        let zero_var =
            ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::instruction(0, 1), ty);
        let result = local_at(&mut ssa, 2, 0, 2);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(combine(combined, x, y)));
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero_var,
            value: zero,
        }));
        b0.add_instruction(instr(compare(result, combined, zero_var)));
        b0.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(b0);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        run_predicates(&mut ssa, "compared through", &0u32, &log);
        let op = ssa
            .get_definition(result)
            .cloned()
            .expect("the comparison stays defined");
        (op, combined)
    }

    /// **An order is not read through a subtraction, which wraps.** `(x - y)
    /// < 0` is the sign of the difference: with `x = i64::MIN + 5` and `y =
    /// 1 << 50` the difference wraps positive while `x < y`, so it stays a
    /// test of the difference, and so does `(x - y) > 0`. Equality is exact
    /// at any width, and `(x - y) == 0` still compares `x` with `y`.
    #[test]
    fn an_order_is_not_read_through_a_wrapping_subtraction() {
        let sub = |dest, left, right| SsaOp::Sub {
            dest,
            left,
            right,
            flags: None,
        };
        let lt = |dest, left, right| SsaOp::Clt {
            dest,
            left,
            right,
            unsigned: false,
            signaling: false,
        };
        let gt = |dest, left, right| SsaOp::Cgt {
            dest,
            left,
            right,
            unsigned: false,
            signaling: false,
        };
        let eq = |dest, left, right| SsaOp::Ceq {
            dest,
            left,
            right,
            signaling: false,
        };
        for compare in [lt as fn(_, _, _) -> _, gt] {
            let (op, difference) =
                compared_through(MockType::I64, ConstValue::I64(0), sub, compare);
            assert!(
                matches!(
                    op,
                    SsaOp::Clt { left, .. } | SsaOp::Cgt { left, .. } if left == difference
                ),
                "{op:?}"
            );
        }
        let (op, difference) = compared_through(MockType::I64, ConstValue::I64(0), sub, eq);
        assert!(
            matches!(op, SsaOp::Ceq { left, right, .. } if left != difference && right != difference),
            "{op:?}"
        );
    }

    /// **A float's equality is not read through a subtraction or an `xor`.**
    /// `inf - inf` is NaN, unequal to zero, while `inf == inf`; `+0.0 ^ -0.0`
    /// is nonzero while `+0.0 == -0.0`. Both comparisons stay tests of the
    /// combined value. The zero is the float's bits, as a register cleared by
    /// an `xor` folds to: a float constant is no zero the rules match.
    #[test]
    fn a_float_equality_is_not_read_through_a_subtraction_or_xor() {
        let sub = |dest, left, right| SsaOp::Sub {
            dest,
            left,
            right,
            flags: None,
        };
        let xor = |dest, left, right| SsaOp::Xor {
            dest,
            left,
            right,
            flags: None,
        };
        let eq = |dest, left, right| SsaOp::Ceq {
            dest,
            left,
            right,
            signaling: false,
        };
        for combine in [sub as fn(_, _, _) -> _, xor] {
            let (op, combined) = compared_through(MockType::F64, ConstValue::I64(0), combine, eq);
            assert!(
                matches!(op, SsaOp::Ceq { left, .. } if left == combined),
                "{op:?}"
            );
        }
    }

    /// `(x - x) == 0` is recognized as always true and its branch simplified.
    #[test]
    fn sub_self_always_zero_in_predicate() {
        let mut ssa = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v_sub = local_at(&mut ssa, 1, 0, 1);
        let v_ceq = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Sub {
            dest: v_sub,
            left: v0,
            right: v0,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_ceq,
            left: v_sub,
            right: v0,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_ceq,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "sub-self predicate", &method, &log);
        assert!(changed, "sub-self comparison should be detected");
    }

    /// Equality of two equal constants (`5 == 5`) resolves to always true.
    #[test]
    fn constant_comparison_true() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v_ceq = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(5),
        }));
        // 5 == 5 → always true
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_ceq,
            left: v0,
            right: v1,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_ceq,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "constant-true predicate", &method, &log);
        assert!(changed, "constant equality should be detected");
    }

    /// Equality of two unequal constants (`3 == 7`) resolves to always false.
    #[test]
    fn constant_comparison_false() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v_ceq = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(7),
        }));
        // 3 == 7 → always false
        b0.add_instruction(instr(SsaOp::Ceq {
            dest: v_ceq,
            left: v0,
            right: v1,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_ceq,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "constant-false predicate", &method, &log);
        assert!(changed, "constant inequality should be detected");
    }

    /// A same-variable less-than (`x < x`) resolves to always false.
    #[test]
    fn clt_same_var_always_false() {
        let mut ssa = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v_clt = local_at(&mut ssa, 1, 0, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Clt {
            dest: v_clt,
            left: v0,
            right: v0,
            unsigned: false,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_clt,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "same-var less-than predicate", &method, &log);
        assert!(changed, "x < x should be always false");
    }

    /// An unsigned `x < 0` resolves to always false.
    #[test]
    fn unsigned_less_than_zero_always_false() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let zero = local_at(&mut ssa, 1, 0, 1);
        let v_clt = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::U32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::U32(0),
        }));
        // unsigned x < 0 is always false
        b0.add_instruction(instr(SsaOp::Clt {
            dest: v_clt,
            left: v0,
            right: zero,
            unsigned: true,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v_clt,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "unsigned-less-than-zero predicate", &method, &log);
        assert!(changed, "unsigned x < 0 should be always false");
    }

    /// Runs the pass over `branchcmp.un <cmp> left, right` and returns the
    /// entry's terminator afterwards. `left` is `None` for an array length (a
    /// non-negative value of unknown size) or `Some(c)` for the constant `c`;
    /// `right` is always a constant. The true arm is `b1`, the false arm `b2`.
    fn unsigned_compare_after(
        left: Option<i32>,
        right: i32,
        cmp: CmpKind,
    ) -> Option<SsaOp<MockTarget>> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 3);
        let array = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::Ref,
        );
        let lhs = local_at(&mut ssa, 1, 0, 1);
        let rhs = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: array,
            arg_index: 0,
        }));
        b0.add_instruction(instr(match left {
            Some(value) => SsaOp::Const {
                dest: lhs,
                value: ConstValue::I32(value),
            },
            None => SsaOp::ArrayLength { dest: lhs, array },
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: rhs,
            value: ConstValue::I32(right),
        }));
        b0.add_instruction(instr(SsaOp::BranchCmp {
            left: lhs,
            right: rhs,
            cmp,
            unsigned: true,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        for index in 1..=2 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_predicates(&mut ssa, "unsigned compare", &method, &log);
        ssa.block(0).and_then(SsaBlock::control_terminator).cloned()
    }

    /// `0xFFFFFFFF <u 1` is false. `-1 < 1` is true only signed, so an
    /// unsigned compare must not borrow the signed range answer.
    #[test]
    fn an_unsigned_compare_of_a_negative_constant_is_not_decided_signed() {
        let after = unsigned_compare_after(Some(-1), 1, CmpKind::Lt);
        assert!(
            !matches!(after, Some(SsaOp::Jump { target: 1 })),
            "`-1 <u 1` is false, got {after:?}"
        );
    }

    /// An array length is below `0xFFFFFFFF` unsigned, so `len <u -1` is not
    /// false - the signed reading `len >= 0 > -1` is.
    #[test]
    fn an_array_length_below_an_unsigned_minus_one_is_not_refuted() {
        let after = unsigned_compare_after(None, -1, CmpKind::Lt);
        assert!(
            !matches!(after, Some(SsaOp::Jump { target: 2 })),
            "`len <u -1` is not always false, got {after:?}"
        );
    }

    /// Nothing is above `0xFFFFFFFF` unsigned, so `len >u -1` is not true -
    /// the signed reading `len >= 0 > -1` is.
    #[test]
    fn an_array_length_above_an_unsigned_minus_one_is_not_proved() {
        let after = unsigned_compare_after(None, -1, CmpKind::Gt);
        assert!(
            !matches!(after, Some(SsaOp::Jump { target: 1 })),
            "`len >u -1` is not always true, got {after:?}"
        );
    }

    /// The control: over non-negative operands the unsigned and signed orders
    /// agree, so `1 <u 2` is still decided.
    #[test]
    fn an_unsigned_compare_of_non_negative_constants_is_decided() {
        let after = unsigned_compare_after(Some(1), 2, CmpKind::Lt);
        assert!(
            matches!(after, Some(SsaOp::Jump { target: 1 })),
            "`1 <u 2` is true, got {after:?}"
        );
    }

    /// A phi whose operands are all the same constant is recorded as that constant.
    #[test]
    fn analyze_phi_constants_all_same() {
        let mut ssa = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 1, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(42),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v0, 0));
        phi.add_operand(PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::returns(Some(phi_var))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let constants = analyze_phi_constants(&ssa);
        // Both operands are constant I32(42) - phi should be resolved
        assert!(
            constants.contains_key(&phi_var),
            "phi with same constants should be resolved"
        );
    }

    /// A phi is only constant when *every* incoming edge carries that constant.
    /// An operand defined by another phi has no defining instruction, so its
    /// value is unknown here - it must disqualify the fold regardless of which
    /// operand slot it occupies.
    ///
    /// `B0: a = Const 99; jump B1`
    /// `B1: p1 = phi(a@B0); jump B3`
    /// `B2: c = Const 7; jump B3`
    /// `B3: p2 = phi(p1@B1, c@B2); return p2`
    ///
    /// `p2` is 99 on the `B1` edge and 7 on the `B2` edge, so it is not
    /// constant. Folding it to 7 rewrites the phi and, when such a phi feeds a
    /// `Branch`, deletes the entire opposite arm.
    #[test]
    fn phi_with_a_phi_defined_operand_is_not_constant() {
        let mut ssa = SsaFunction::new(0, 4);
        let a = local_at(&mut ssa, 0, 0, 0);
        let c = local_at(&mut ssa, 1, 2, 0);
        let p1 = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(1), MockType::I32);
        let p2 = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(3), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(99),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi1 = PhiNode::new(p1, VariableOrigin::Local(2));
        phi1.add_operand(PhiOperand::new(a, 0));
        b1.add_phi(phi1);
        b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: c,
            value: ConstValue::I32(7),
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        // `p1` first, `c` second - the order that hides the undefined operand.
        let mut phi2 = PhiNode::new(p2, VariableOrigin::Local(3));
        phi2.add_operand(PhiOperand::new(p1, 1));
        phi2.add_operand(PhiOperand::new(c, 2));
        b3.add_phi(phi2);
        b3.add_instruction(instr(SsaOp::returns(Some(p2))));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let constants = analyze_phi_constants(&ssa);

        assert!(
            !constants.contains_key(&p2),
            "p2 merges 99 and 7 - it must not be folded to a constant (got {:?})",
            constants.get(&p2)
        );
    }

    /// The self-referential loop phi `p = phi(c@entry, p@latch)` really is
    /// constant `c`, and must keep folding - the fix above must not throw this
    /// away along with the unsound case.
    #[test]
    fn self_referential_loop_phi_is_still_constant() {
        let mut ssa = SsaFunction::new(0, 3);
        let c = local_at(&mut ssa, 0, 0, 0);
        let p = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(1), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: c,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(p, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(c, 0));
        phi.add_operand(PhiOperand::new(p, 1));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(Some(p))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let constants = analyze_phi_constants(&ssa);

        assert_eq!(
            constants.get(&p),
            Some(&ConstValue::I32(5)),
            "a loop phi whose only non-self operand is a constant is that constant"
        );
    }

    /// A branch on a non-zero constant condition is simplified to its true target.
    #[test]
    fn branch_always_true_opaque_predicate() {
        let mut ssa = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        // Branch on constant 1 - always true
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v0,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "constant-branch predicate", &method, &log);
        // The branch on constant 1 should be simplified
        assert!(changed, "constant branch should be simplified");
    }

    /// An empty function is a no-op and reports no change.
    #[test]
    fn empty_function_noop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_predicates(&mut ssa, "empty predicates", &method, &log);
        assert!(!changed);
    }
}
