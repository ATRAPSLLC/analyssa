//! Strength reduction pass - replaces expensive operations with cheaper
//! equivalents.
//!
//! # Transformations
//!
//! - **Multiplication by power of 2**: `x * 2^n` → `x << n`
//! - **Unsigned division by power of 2**: `x / 2^n` → `x >> n` (unsigned)
//! - **Unsigned modulo by power of 2**: `x % 2^n` → `x & (2^n - 1)` (unsigned)
//! - **Signed division by power of 2**: same as unsigned, but only when the
//!   dividend is provably non-negative.
//! - **Signed modulo by power of 2**: same as unsigned, but only when the
//!   dividend is provably non-negative.
//!
//! # Correctness
//!
//! Signed division and modulo are NOT transformed unconditionally because
//! signed division rounds toward zero while arithmetic right-shift rounds
//! toward negative infinity:
//! - `-5 / 2 = -2` (truncation toward zero)
//! - `-5 >> 1 = -3` (round toward negative infinity)
//!
//! The proved value ranges gate these transformations: a signed division or
//! modulo becomes a shift only where the dividend cannot be negative. The
//! range analysis runs only when such an operation asks.
//!
//! # Algorithm
//!
//! 1. For each `Mul`/`Div`/`Rem`, check if one operand is a constant power
//!    of two.
//! 2. Check the constant has exactly one use (no other instruction depends
//!    on its original value).
//! 3. For signed ops, check that the dividend's range excludes negatives.
//! 4. Replace the constant with `exponent` (for shifts) or `mask` (for AND),
//!    and replace the operation with the cheaper equivalent.

use crate::{
    Result,
    analysis::{DefUseIndex, range::ValueRange},
    events::EventKind,
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
        varstore::VarSet,
    },
    passes::utils::is_power_of_two,
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// The reduction [`StrengthReductionPass`] makes, with the non-negativity
/// question answered directly.
///
/// The seam between *what is proved* and *what the reduction does with it*.
/// The public entry point reads the proof from the range analysis, which is the
/// one thing that knows; this is what the reduction's own tests drive, so they
/// exercise the rewriting rather than the fixpoint that feeds it.
///
/// # Arguments
///
/// * `ssa` - The SSA function to transform in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
/// * `is_non_negative` - Answers whether a value of the function (passed in
///   unmodified) is provably at least zero. Asked only for the dividend of a
///   signed division or modulo by a power of two, before anything is
///   rewritten.
///
/// # Returns
///
/// `true` if any operation was rewritten.
fn run_with_facts<T>(
    ssa: &mut SsaFunction<T>,
    env: &PassEnv<'_, T>,
    is_non_negative: &dyn Fn(&SsaFunction<T>, SsaVarId) -> bool,
) -> bool
where
    T: Target,
{
    let index = DefUseIndex::<T>::build_with_ops(ssa);
    let unmodified: &SsaFunction<T> = ssa;
    let candidates = find_candidates(unmodified, &index, &|var| is_non_negative(unmodified, var));
    apply_reductions(ssa, candidates, env)
}

/// Identifies an instruction by its block and index within the block.
#[derive(Debug, Clone, Copy)]
struct InstrLocation {
    /// Index of the containing block.
    block_idx: usize,
    /// Index of the instruction within the block.
    instr_idx: usize,
}

/// A detected strength-reduction opportunity.
#[derive(Debug)]
struct ReductionCandidate<T: Target> {
    /// Location of the instruction to reduce.
    location: InstrLocation,
    /// The constant variable being used as the power-of-two operand.
    const_var: SsaVarId,
    /// Block containing the constant definition.
    const_block: usize,
    /// Index of the constant definition within its block.
    const_instr: usize,
    /// The new value for the constant (exponent for shifts, mask for AND).
    new_const_value: ConstValue<T>,
    /// The replacement operation (shift or AND).
    new_op: SsaOp<T>,
    /// Human-readable description of the reduction applied.
    description: String,
}

/// Helper struct that checks for reduction opportunities on individual
/// instructions.
struct ReductionChecker<'a, T: Target> {
    /// Def-use index for looking up variable definitions.
    index: &'a DefUseIndex<T>,
    /// Bitset of constant variables already claimed by earlier reductions.
    used_constants: &'a VarSet,
}

impl<'a, T: Target> ReductionChecker<'a, T> {
    /// Create a checker over a def-use index and the set of claimed constants.
    ///
    /// # Arguments
    ///
    /// * `index` - Def-use index used to locate constant definitions and
    ///   count uses.
    /// * `used_constants` - Constants already rewritten by earlier candidates
    ///   in this run, which must not be reused.
    ///
    /// # Returns
    ///
    /// A checker bound to those borrows.
    fn new(index: &'a DefUseIndex<T>, used_constants: &'a VarSet) -> Self {
        Self {
            index,
            used_constants,
        }
    }

    /// Returns `true` when an op's `flags` definition can be discarded.
    ///
    /// `flags` is a real secondary definition (`SsaOp::defs()` yields it,
    /// `SsaOp::ReadFlags` and `SsaOp::BranchFlags` consume it), and none of the
    /// replacement ops reproduces it - nor could they correctly: `shl`'s CF/OF
    /// are not `imul`'s, so forwarding the variable through would substitute one
    /// instruction's flag semantics for another's.
    ///
    /// But *most* flag definitions are never read. A lifter emits flags for
    /// nearly every arithmetic instruction while only a few feed a conditional
    /// branch, so refusing every flag-producing op would disable this pass on
    /// essentially all native code. When nothing reads the definition, dropping
    /// it along with the op it belonged to is exactly what dead-code elimination
    /// would do anyway.
    fn flags_are_dead(&self, flags: Option<SsaVarId>) -> bool {
        flags.is_none_or(|flags| self.index.use_count(flags) == 0)
    }

    /// Build a candidate rewriting `value * 2^n` into `value << n`.
    ///
    /// Succeeds only when `const_var` is defined by a `Const` whose value is a
    /// positive power of two: the exponent becomes the shift amount, reusing
    /// the same constant variable retyped to the exponent. The constant must
    /// have exactly one use and not already be claimed by an earlier
    /// candidate - rewriting a constant shared by another instruction would corrupt that
    /// instruction, which still expects the original multiplier value.
    ///
    /// # Arguments
    ///
    /// * `dest` - Destination of the multiply, carried onto the shift.
    /// * `value_var` - The non-constant multiplicand, shifted left.
    /// * `const_var` - Candidate power-of-two operand and its `Const` def.
    /// * `location` - Location of the multiply instruction.
    ///
    /// # Returns
    ///
    /// A [`ReductionCandidate`], or `None` if the operand is not an
    /// exclusively-used power-of-two constant.
    fn try_mul_reduction(
        &self,
        dest: SsaVarId,
        value_var: SsaVarId,
        const_var: SsaVarId,
        location: InstrLocation,
    ) -> Option<ReductionCandidate<T>> {
        let (const_block, const_instr, const_op) = self.index.full_definition(const_var)?;
        let SsaOp::Const {
            value: const_value, ..
        } = const_op
        else {
            return None;
        };
        let value = const_value.as_i64()?;
        let exponent = is_power_of_two(value)?;
        let uses = self.index.use_count(const_var);
        if uses != 1 || self.used_constants.contains(const_var) {
            return None;
        }
        Some(ReductionCandidate {
            location,
            const_var,
            const_block,
            const_instr,
            new_const_value: const_value.integer_of_same_type(i64::from(exponent))?,
            new_op: SsaOp::Shl {
                dest,
                value: value_var,
                amount: const_var,
                flags: None,
            },
            description: format!("mul x, {value} → shl x, {exponent}"),
        })
    }

    /// Build a candidate rewriting `dividend / 2^n` into `dividend >> n`.
    ///
    /// The exponent becomes the shift amount, carrying the op's `unsigned`
    /// flag onto the `Shr` so a logical vs. arithmetic shift is chosen to
    /// match the division. As with [`Self::try_mul_reduction`], the divisor
    /// must be a power-of-two `Const` with exactly one unclaimed use. Signedness
    /// correctness (arithmetic shift only rounds like signed division for
    /// non-negative dividends) is enforced by the caller, not here.
    ///
    /// # Arguments
    ///
    /// * `dest` - Destination of the division, carried onto the shift.
    /// * `dividend` - The value being divided, shifted right.
    /// * `divisor_var` - Candidate power-of-two divisor and its `Const` def.
    /// * `unsigned` - Whether the division (and resulting shift) is unsigned.
    /// * `location` - Location of the division instruction.
    ///
    /// # Returns
    ///
    /// A [`ReductionCandidate`], or `None` if the divisor is not an
    /// exclusively-used power-of-two constant.
    fn try_div_reduction(
        &self,
        dest: SsaVarId,
        dividend: SsaVarId,
        divisor_var: SsaVarId,
        unsigned: bool,
        location: InstrLocation,
    ) -> Option<ReductionCandidate<T>> {
        let (const_block, const_instr, const_op) = self.index.full_definition(divisor_var)?;
        let SsaOp::Const {
            value: const_value, ..
        } = const_op
        else {
            return None;
        };
        let value = const_value.as_i64()?;
        let exponent = is_power_of_two(value)?;
        let uses = self.index.use_count(divisor_var);
        if uses != 1 || self.used_constants.contains(divisor_var) {
            return None;
        }
        let desc = if unsigned {
            format!("div.un x, {value} → shr.un x, {exponent}")
        } else {
            format!("div x, {value} → shr x, {exponent} (x >= 0)")
        };
        Some(ReductionCandidate {
            location,
            const_var: divisor_var,
            const_block,
            const_instr,
            new_const_value: const_value.integer_of_same_type(i64::from(exponent))?,
            new_op: SsaOp::Shr {
                dest,
                value: dividend,
                amount: divisor_var,
                unsigned,
                flags: None,
            },
            description: desc,
        })
    }

    /// Build a candidate rewriting `dividend % 2^n` into `dividend & (2^n - 1)`.
    ///
    /// The divisor `Const` is retyped to the mask `2^n - 1` and the op becomes
    /// a bitwise `And`. The mask is materialised via
    /// [`ConstValue::integer_of_same_type`] in the divisor's own type rather
    /// than as a bare `i64`: for a wide power of two such as `2^32` the mask
    /// exceeds a narrower type's range, and truncating it would collapse the
    /// `and` into the identity (the wrong result). Same single-unclaimed-use
    /// requirement as the other reductions; signedness correctness is the
    /// caller's responsibility.
    ///
    /// # Arguments
    ///
    /// * `dest` - Destination of the remainder, carried onto the `And`.
    /// * `dividend` - The value being reduced, masked.
    /// * `divisor_var` - Candidate power-of-two divisor and its `Const` def.
    /// * `unsigned` - Whether the remainder is unsigned (affects the description).
    /// * `location` - Location of the remainder instruction.
    ///
    /// # Returns
    ///
    /// A [`ReductionCandidate`], or `None` if the divisor is not an
    /// exclusively-used power-of-two constant or the mask cannot be
    /// represented in its type.
    fn try_rem_reduction(
        &self,
        dest: SsaVarId,
        dividend: SsaVarId,
        divisor_var: SsaVarId,
        unsigned: bool,
        location: InstrLocation,
    ) -> Option<ReductionCandidate<T>> {
        let (const_block, const_instr, const_op) = self.index.full_definition(divisor_var)?;
        let SsaOp::Const {
            value: const_value, ..
        } = const_op
        else {
            return None;
        };
        let value = const_value.as_i64()?;
        let _exponent = is_power_of_two(value)?;
        let mask = value.checked_sub(1)?;
        let uses = self.index.use_count(divisor_var);
        if uses != 1 || self.used_constants.contains(divisor_var) {
            return None;
        }
        let desc = if unsigned {
            format!("rem.un x, {value} → and x, {mask}")
        } else {
            format!("rem x, {value} → and x, {mask} (x >= 0)")
        };
        Some(ReductionCandidate {
            location,
            const_var: divisor_var,
            const_block,
            const_instr,
            new_const_value: const_value.integer_of_same_type(mask)?,
            new_op: SsaOp::And {
                dest,
                left: dividend,
                right: divisor_var,
                flags: None,
            },
            description: desc,
        })
    }
}

/// Scan every instruction for strength-reduction opportunities.
///
/// Walks the function once, building a fresh [`ReductionChecker`] per
/// instruction so it sees the running `used_constants` set. Each accepted
/// candidate claims its constant variable in that set, so a constant shared
/// by two reducible ops is rewritten for the first only - the second, now
/// seeing a claimed constant, is skipped. Collecting into a `Vec` decouples
/// analysis (immutable borrow of `ssa`) from the mutating apply phase.
///
/// # Arguments
///
/// * `ssa` - The SSA function to scan.
/// * `index` - Def-use index for definition lookup and use counting.
/// * `is_non_negative` - Predicate gating the signed division/modulo arms.
///
/// # Returns
///
/// The detected candidates in instruction order.
fn find_candidates<T: Target>(
    ssa: &SsaFunction<T>,
    index: &DefUseIndex<T>,
    is_non_negative: &dyn Fn(SsaVarId) -> bool,
) -> Vec<ReductionCandidate<T>> {
    let mut candidates = Vec::new();
    let mut used_constants = VarSet::new(ssa.var_id_bound());

    for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
        let checker = ReductionChecker::new(index, &used_constants);
        let location = InstrLocation {
            block_idx,
            instr_idx,
        };
        if let Some(candidate) = check_reduction(instr.op(), location, &checker, is_non_negative) {
            used_constants.insert(candidate.const_var);
            candidates.push(candidate);
        }
    }

    candidates
}

/// Dispatch one instruction to the matching reduction check.
///
/// Recognises `Mul`, unsigned/signed `Div`, and unsigned/signed `Rem`. Every
/// arm binds `flags` explicitly and only proceeds when
/// [`ReductionChecker::flags_are_dead`] holds, because the cheaper ops do not
/// reproduce the original instruction's flag semantics. Multiplication is
/// commutative, so both operand orderings are tried as the potential
/// power-of-two factor. The signed `Div`/`Rem` arms additionally require
/// `is_non_negative` on the dividend, since shifting/masking only matches
/// truncating division for non-negative values.
///
/// # Arguments
///
/// * `op` - The instruction operation to examine.
/// * `location` - Location of the instruction, threaded into any candidate.
/// * `checker` - Reduction checker over the current def-use and claimed set.
/// * `is_non_negative` - Predicate gating the signed arms.
///
/// # Returns
///
/// A [`ReductionCandidate`] if the op is reducible, else `None`.
fn check_reduction<T: Target>(
    op: &SsaOp<T>,
    location: InstrLocation,
    checker: &ReductionChecker<'_, T>,
    is_non_negative: &dyn Fn(SsaVarId) -> bool,
) -> Option<ReductionCandidate<T>> {
    match op {
        // Every arm binds `flags` explicitly rather than swallowing it with
        // `..`, and declines when that definition is live - see
        // `ReductionChecker::flags_are_dead` for why liveness rather than mere
        // presence is the right condition.
        SsaOp::Mul {
            dest,
            left,
            right,
            flags,
        } if checker.flags_are_dead(*flags) => {
            if let Some(candidate) = checker.try_mul_reduction(*dest, *left, *right, location) {
                return Some(candidate);
            }
            checker.try_mul_reduction(*dest, *right, *left, location)
        }
        SsaOp::Div {
            dest,
            left,
            right,
            unsigned: true,
            flags,
        } if checker.flags_are_dead(*flags) => {
            checker.try_div_reduction(*dest, *left, *right, true, location)
        }
        SsaOp::Div {
            dest,
            left,
            right,
            unsigned: false,
            flags,
        } if checker.flags_are_dead(*flags) => checker
            .try_div_reduction(*dest, *left, *right, false, location)
            .filter(|_| is_non_negative(*left)),
        SsaOp::Rem {
            dest,
            left,
            right,
            unsigned: true,
            flags,
        } if checker.flags_are_dead(*flags) => {
            checker.try_rem_reduction(*dest, *left, *right, true, location)
        }
        SsaOp::Rem {
            dest,
            left,
            right,
            unsigned: false,
            flags,
        } if checker.flags_are_dead(*flags) => checker
            .try_rem_reduction(*dest, *left, *right, false, location)
            .filter(|_| is_non_negative(*left)),
        _ => None,
    }
}

/// Apply the collected candidates to the function in a single edit session.
///
/// For each candidate it rewrites two instructions - the constant definition
/// (retyped to the exponent or mask) and the arithmetic op (replaced with the
/// shift or `And`). Both target sites are re-checked for existence first,
/// because an earlier candidate's edits may have shifted or removed them; a
/// vanished site is skipped rather than mis-addressed. A failed edit under the
/// `Never` rollback policy leaves the function mutated, so the error path
/// reports `true` to force the enclosing transaction to verify and roll back
/// rather than trusting damaged IR.
///
/// # Arguments
///
/// * `ssa` - The SSA function to mutate.
/// * `candidates` - Reductions discovered by [`find_candidates`].
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// `true` if any reduction was applied (or the edit failed mid-flight).
fn apply_reductions<T>(
    ssa: &mut SsaFunction<T>,
    candidates: Vec<ReductionCandidate<T>>,
    env: &PassEnv<'_, T>,
) -> bool
where
    T: Target,
{
    let mut changed = false;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for candidate in candidates {
            let const_exists = editor
                .function()
                .block(candidate.const_block)
                .and_then(|block| block.instruction(candidate.const_instr))
                .is_some();
            let op_exists = editor
                .function()
                .block(candidate.location.block_idx)
                .and_then(|block| block.instruction(candidate.location.instr_idx))
                .is_some();

            if !const_exists || !op_exists {
                continue;
            }

            editor.replace_instruction_op(
                candidate.const_block,
                candidate.const_instr,
                SsaOp::Const {
                    dest: candidate.const_var,
                    value: candidate.new_const_value.clone(),
                },
            )?;
            editor.replace_instruction_op(
                candidate.location.block_idx,
                candidate.location.instr_idx,
                candidate.new_op,
            )?;

            env.record(EventKind::StrengthReduced)
                .location(candidate.location.instr_idx)
                .message(candidate.description);
            changed = true;
        }
        Ok(())
    });

    if result.is_err() {
        // The session runs under `SsaRollbackPolicy::Never`, so a failed edit or
        // boundary repair leaves the edits applied - the function is mutated and
        // possibly mid-repair. Reporting "unchanged" would make the pass-group
        // transaction skip **both** verification and rollback, keeping damaged
        // IR and keeping it unchecked. Report the change so the transaction
        // verifies this function and rolls it back.
        return true;
    }

    changed
}

/// Pass that replaces expensive operations with cheaper equivalents.
///
/// Transforms:
/// - `x * 2^n` → `x << n`
/// - `x / 2^n` (unsigned) → `x >> n`
/// - `x % 2^n` (unsigned) → `x & (2^n - 1)`
///
/// Signed variants are applied where the value-range analysis proves the
/// dividend non-negative; the environment's range facts ([`PassEnv::ranges`])
/// are read only when such a candidate asks (see
/// [`StrengthReductionPass`]).
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct StrengthReductionPass;

impl StrengthReductionPass {
    /// Creates a new strength reduction pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for StrengthReductionPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "strength-reduction"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Replace expensive operations with cheaper equivalents"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run strength reduction on `ssa`.
    ///
    /// Replaces multiplication by powers of two with left shifts, unsigned
    /// division/modulo by powers of two with right shifts / bitwise AND,
    /// and signed variants when the dividend is provably non-negative.
    ///
    /// The signed arms read that proof from the function's range facts, which are
    /// asked only when a signed division or modulo by a power of two actually
    /// needs them. It is the only question the analysis answers here, and the
    /// expensive part of this pass otherwise: a method with no such operation -
    /// nearly every one - would pay a full range fixpoint, when no earlier reader
    /// had, and use none of it.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any operation was rewritten.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        // The facts the signed arms need, from the environment rather than
        // asked of the caller: every caller that was asked answered "nothing
        // is provably non-negative", so those arms never fired.
        let facts = env.ranges();
        let is_non_negative = |ssa: &SsaFunction<T>, var: SsaVarId| {
            facts.get(ssa).is_ok_and(|ranges| {
                ranges
                    .get(var)
                    .is_some_and(ValueRange::is_always_non_negative)
            })
        };
        Ok(run_with_facts(ssa, env, &is_non_negative))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            ops::FlagsMask,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType, mock_op_at, run_mock_pass_boundary},
    };

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    /// `x % 2^32` is not `x & -1`. The mask is `2^32 - 1`, which does not fit an
    /// `i32`; materialising it as one produces `I32(-1)`, and because
    /// `ConstValue::bitwise_and` sign-extends a narrower operand, the resulting
    /// `and` is the identity. The rewrite must emit the mask in the divisor's
    /// own type, or decline.
    #[test]
    fn rem_by_a_wide_power_of_two_does_not_truncate_its_mask() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local(&mut ssa, 0, 0, 0);
        let divisor = local(&mut ssa, 1, 0, 1);
        let result = local(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I64(12345),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: divisor,
            // 2^32 - a power of two whose mask needs more than 32 bits.
            value: ConstValue::I64(1i64 << 32),
        }));
        block.add_instruction(instr(SsaOp::Rem {
            dest: result,
            left: x,
            right: divisor,
            unsigned: true,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_with_facts(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            &|_, _| true,
        );

        // Whether or not the rewrite fires, the constant must never become the
        // all-ones mask of the wrong width.
        let value = ssa.blocks().iter().find_map(|block| {
            block
                .instructions()
                .iter()
                .find_map(|instruction| match instruction.op() {
                    SsaOp::Const { dest, value } if *dest == divisor => Some(value.clone()),
                    _ => None,
                })
        });
        assert_ne!(
            value,
            Some(ConstValue::I32(-1)),
            "the mask for 2^32 must not truncate to I32(-1), which makes `and` the identity"
        );
        assert_eq!(ssa.validate(), Ok(()));
    }

    /// A narrower power of two still reduces - the guard above must not cost the
    /// optimization it protects.
    #[test]
    fn rem_by_a_narrow_power_of_two_still_reduces() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local(&mut ssa, 0, 0, 0);
        let divisor = local(&mut ssa, 1, 0, 1);
        let result = local(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(12345),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: divisor,
            value: ConstValue::I32(16),
        }));
        block.add_instruction(instr(SsaOp::Rem {
            dest: result,
            left: x,
            right: divisor,
            unsigned: true,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_with_facts(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            &|_, _| true,
        );
        assert!(changed, "rem by 16 should reduce to and 15");
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::And { .. }));
    }

    /// The common native shape: the lifter emitted a flags definition because
    /// the instruction sets flags, but nothing reads it. Dropping it with the op
    /// is exactly what DCE would do, so the reduction must still fire - a lifter
    /// sets flags on nearly every arithmetic instruction, and refusing all of
    /// them would disable this pass on native code entirely.
    #[test]
    fn a_multiply_with_dead_flags_is_still_strength_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local(&mut ssa, 0, 0, 0);
        let factor = local(&mut ssa, 1, 0, 1);
        let product = local(&mut ssa, 2, 0, 2);
        let flags = local(&mut ssa, 3, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: factor,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: product,
            left: x,
            right: factor,
            // Defined, but never read by any instruction.
            flags: Some(flags),
        }));
        block.add_instruction(instr(SsaOp::returns(Some(product))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_with_facts(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            &|_, _| true,
        );

        assert!(
            changed,
            "a dead flags definition must not block the reduction"
        );
        assert!(
            matches!(mock_op_at(&ssa, 0, 2), SsaOp::Shl { .. }),
            "mul by 8 with dead flags should become shl; got {:?}",
            mock_op_at(&ssa, 0, 2)
        );
    }

    /// A flag-producing op must not be rewritten into one that drops the flags
    /// definition. Forwarding the flags through would be wrong too - `shl`'s
    /// CF/OF are not `imul`'s - so the rewrite has to be declined.
    #[test]
    fn a_flag_producing_multiply_is_not_strength_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local(&mut ssa, 0, 0, 0);
        let factor = local(&mut ssa, 1, 0, 1);
        let product = local(&mut ssa, 2, 0, 2);
        let flags = local(&mut ssa, 3, 0, 2);
        let read = local(&mut ssa, 4, 0, 3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: factor,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: product,
            left: x,
            right: factor,
            flags: Some(flags),
        }));
        block.add_instruction(instr(SsaOp::ReadFlags {
            dest: read,
            flags,
            mask: FlagsMask::CARRY,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(read))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_with_facts(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            &|_, _| true,
        );

        assert!(
            matches!(mock_op_at(&ssa, 0, 2), SsaOp::Mul { flags: Some(_), .. }),
            "a flag-setting multiply must be left alone; got {:?}",
            mock_op_at(&ssa, 0, 2)
        );
        assert_eq!(ssa.validate(), Ok(()));
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

    /// Multiplication by a power of two becomes a left shift.
    #[test]
    fn mul_by_power_of_two_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: result,
            left: x,
            right: pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed =
            run_mock_pass_boundary(&mut ssa, "mul power-of-two strength reduction", |ssa| {
                run_with_facts(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                    &|_, _| true,
                )
            });
        assert!(changed, "mul by power of two should reduce to shift");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::Shl { .. }));
    }

    /// Multiplication by a non-power-of-two is left unchanged.
    #[test]
    fn mul_by_non_power_of_two_not_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let not_pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: not_pow2,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: result,
            left: x,
            right: not_pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed =
            run_mock_pass_boundary(&mut ssa, "non-power-of-two strength reduction", |ssa| {
                run_with_facts(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                    &|_, _| true,
                )
            });
        assert!(!changed, "mul by non-power-of-two should NOT reduce");
    }

    /// Unsigned division by a power of two becomes a logical right shift.
    #[test]
    fn unsigned_div_by_power_of_two_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(instr(SsaOp::Div {
            dest: result,
            left: x,
            right: pow2,
            unsigned: true,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "unsigned div strength reduction", |ssa| {
            run_with_facts(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                &|_, _| true,
            )
        });
        assert!(changed, "unsigned div by power of two should reduce");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(
            mock_op_at(&ssa, 0, 2),
            SsaOp::Shr { unsigned: true, .. }
        ));
    }

    /// Signed division by a power of two reduces when the dividend is proven non-negative.
    #[test]
    fn signed_div_by_power_of_two_reduced_when_non_negative() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Div {
            dest: result,
            left: x,
            right: pow2,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "signed div strength reduction", |ssa| {
            run_with_facts(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                &|_, _| true,
            )
        });
        assert!(
            changed,
            "signed div by power of two should reduce when non-negative"
        );
        assert!(log.has(EventKind::StrengthReduced));
    }

    /// The sign is asked only of a dividend the rewrite would use.
    ///
    /// Answering it runs the range fixpoint, the costly part of this pass, so
    /// a signed division by something other than a power of two, and one
    /// whose power of two is shared, must decline before asking.
    #[test]
    fn the_sign_is_asked_only_for_a_reducible_divisor() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
        let x = local_at(&mut ssa, 0, 0, 0);
        let three = local_at(&mut ssa, 1, 0, 1);
        let quotient = local_at(&mut ssa, 2, 0, 2);
        let four = local_at(&mut ssa, 3, 0, 3);
        let remainder = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: three,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Div {
            dest: quotient,
            left: x,
            right: three,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: four,
            value: ConstValue::I32(4),
        }));
        // `four` is also returned, so rewriting it into a mask is not allowed.
        block.add_instruction(instr(SsaOp::Rem {
            dest: remainder,
            left: x,
            right: four,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(four))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let asked = Cell::new(0usize);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_with_facts(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            &|_, _| {
                asked.set(asked.get() + 1);
                true
            },
        );
        assert!(!changed, "neither operation is reducible");
        assert_eq!(asked.get(), 0, "no dividend's sign was needed");
    }

    /// Signed division is left alone when non-negativity cannot be proven.
    #[test]
    fn signed_div_not_reduced_when_not_proven_non_negative() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(-100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(instr(SsaOp::Div {
            dest: result,
            left: x,
            right: pow2,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(
            &mut ssa,
            "signed div unknown-sign strength reduction",
            |ssa| {
                run_with_facts(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                    &|_, _| false,
                )
            },
        );
        assert!(
            !changed,
            "signed div should NOT reduce when non-negativity not proven"
        );
    }

    /// Unsigned remainder by a power of two becomes a bitwise AND with the mask.
    #[test]
    fn unsigned_rem_by_power_of_two_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(100),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Rem {
            dest: result,
            left: x,
            right: pow2,
            unsigned: true,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "unsigned rem strength reduction", |ssa| {
            run_with_facts(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                &|_, _| true,
            )
        });
        assert!(changed, "unsigned rem by power of two should reduce to and");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::And { .. }));
    }

    /// Several independent reductions all fire in a single run.
    #[test]
    fn multiple_reductions_in_one_run() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
        let x = local_at(&mut ssa, 0, 0, 0);
        let p1 = local_at(&mut ssa, 1, 0, 1);
        let y = local_at(&mut ssa, 2, 0, 2);
        let p2 = local_at(&mut ssa, 3, 0, 3);
        let r1 = local_at(&mut ssa, 4, 0, 4);
        let r2 = local_at(&mut ssa, 5, 0, 5);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: p1,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(50),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: p2,
            value: ConstValue::I32(16),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: r1,
            left: x,
            right: p1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: r2,
            left: y,
            right: p2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(r1))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "multiple strength reductions", |ssa| {
            run_with_facts(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                &|_, _| true,
            )
        });
        assert!(changed, "multiple reductions should all fire");
        assert!(log.count_kind(EventKind::StrengthReduced) >= 2);
    }

    /// A function with no reducible ops reports no change.
    #[test]
    fn no_candidates_returns_false() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let x = local_at(&mut ssa, 0, 0, 0);
        let y = local_at(&mut ssa, 1, 0, 1);
        let result = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: result,
            left: x,
            right: y,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "no-candidate strength reduction", |ssa| {
            run_with_facts(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                &|_, _| true,
            )
        });
        assert!(!changed, "no strength-reducible ops should return false");
    }

    /// An empty function produces no reductions.
    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "empty strength reduction", |ssa| {
            run_with_facts(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                &|_, _| true,
            )
        });
        assert!(!changed);
    }

    /// A power-of-two constant used by more than one op is not rewritten.
    #[test]
    fn shared_constant_not_reduced() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let x = local_at(&mut ssa, 0, 0, 0);
        let pow2 = local_at(&mut ssa, 1, 0, 1);
        let r1 = local_at(&mut ssa, 2, 0, 2);
        let r2 = local_at(&mut ssa, 3, 0, 3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: pow2,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: r1,
            left: x,
            right: pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: r2,
            left: r1,
            right: pow2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(r2))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed =
            run_mock_pass_boundary(&mut ssa, "shared constant strength reduction", |ssa| {
                run_with_facts(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                    &|_, _| true,
                )
            });
        assert!(
            !changed,
            "shared constant should not be rewritten for strength reduction"
        );
    }
}
