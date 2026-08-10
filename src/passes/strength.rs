//! Strength reduction pass — replaces expensive operations with cheaper
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
//! The caller supplies `is_non_negative` to gate these transformations.
//! Hosts without range analysis pass `\|_\| false`.
//!
//! # Algorithm
//!
//! 1. For each `Mul`/`Div`/`Rem`, check if one operand is a constant power
//!    of two.
//! 2. Check the constant has exactly one use (no other instruction depends
//!    on its original value).
//! 3. For signed ops, check `is_non_negative` for the dividend.
//! 4. Replace the constant with `exponent` (for shifts) or `mask` (for AND),
//!    and replace the operation with the cheaper equivalent.

use crate::{
    analysis::DefUseIndex,
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
        varstore::VarSet,
    },
    passes::utils::is_power_of_two,
    target::Target,
};

/// Run strength reduction on `ssa`.
///
/// Replaces multiplication by powers of two with left shifts, unsigned
/// division/modulo by powers of two with right shifts / bitwise AND,
/// and signed variants when the dividend is provably non-negative.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::StrengthReduced`] events.
/// * `is_non_negative` — Caller-supplied predicate that returns `true`
///   if a given `SsaVarId` is provably >= 0. Hosts without range analysis
///   should pass `\|_\| false`.
///
/// # Returns
///
/// `true` if any operation was rewritten.
pub fn run<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    is_non_negative: &dyn Fn(SsaVarId) -> bool,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let index = DefUseIndex::<T>::build_with_ops(ssa);
    let candidates = find_candidates(ssa, &index, is_non_negative);
    apply_reductions(ssa, candidates, method, events)
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
    /// replacement ops reproduces it — nor could they correctly: `shl`'s CF/OF
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

fn check_reduction<T: Target>(
    op: &SsaOp<T>,
    location: InstrLocation,
    checker: &ReductionChecker<'_, T>,
    is_non_negative: &dyn Fn(SsaVarId) -> bool,
) -> Option<ReductionCandidate<T>> {
    match op {
        // Every arm binds `flags` explicitly rather than swallowing it with
        // `..`, and declines when that definition is live — see
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
        } if checker.flags_are_dead(*flags) => {
            if is_non_negative(*left) {
                checker.try_div_reduction(*dest, *left, *right, false, location)
            } else {
                None
            }
        }
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
        } if checker.flags_are_dead(*flags) => {
            if is_non_negative(*left) {
                checker.try_rem_reduction(*dest, *left, *right, false, location)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn apply_reductions<T, L>(
    ssa: &mut SsaFunction<T>,
    candidates: Vec<ReductionCandidate<T>>,
    method: &T::MethodRef,
    events: &L,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
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

            let event = crate::events::Event {
                kind: EventKind::StrengthReduced,
                method: Some(method.clone()),
                location: Some(candidate.location.instr_idx),
                message: candidate.description,
                pass: None,
            };
            events.push(event);
            changed = true;
        }
        Ok(())
    });

    if result.is_err() {
        // The session runs under `SsaRollbackPolicy::Never`, so a failed edit or
        // boundary repair leaves the edits applied — the function is mutated and
        // possibly mid-repair. Reporting "unchanged" would make the pass-group
        // transaction skip **both** verification and rollback, keeping damaged
        // IR and keeping it unchecked. Report the change so the transaction
        // verifies this function and rolls it back.
        return true;
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            ops::FlagsMask,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType, mock_op_at, run_mock_pass_boundary},
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
            // 2^32 — a power of two whose mask needs more than 32 bits.
            value: ConstValue::I64(1i64 << 32),
        }));
        block.add_instruction(instr(SsaOp::Rem {
            dest: result,
            left: x,
            right: divisor,
            unsigned: true,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run(&mut ssa, &0u32, &log, &|_| true);

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

    /// A narrower power of two still reduces — the guard above must not cost the
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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);
        assert!(changed, "rem by 16 should reduce to and 15");
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::And { .. }));
    }

    /// The common native shape: the lifter emitted a flags definition because
    /// the instruction sets flags, but nothing reads it. Dropping it with the op
    /// is exactly what DCE would do, so the reduction must still fire — a lifter
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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(product),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run(&mut ssa, &0u32, &log, &|_| true);

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
    /// definition. Forwarding the flags through would be wrong too — `shl`'s
    /// CF/OF are not `imul`'s — so the rewrite has to be declined.
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
        block.add_instruction(instr(SsaOp::Return { value: Some(read) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run(&mut ssa, &0u32, &log, &|_| true);

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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed =
            run_mock_pass_boundary(&mut ssa, "mul power-of-two strength reduction", |ssa| {
                run(ssa, &0u32, &log, &|_| true)
            });
        assert!(changed, "mul by power of two should reduce to shift");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::Shl { .. }));
    }

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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed =
            run_mock_pass_boundary(&mut ssa, "non-power-of-two strength reduction", |ssa| {
                run(ssa, &0u32, &log, &|_| true)
            });
        assert!(!changed, "mul by non-power-of-two should NOT reduce");
    }

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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "unsigned div strength reduction", |ssa| {
            run(ssa, &0u32, &log, &|_| true)
        });
        assert!(changed, "unsigned div by power of two should reduce");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(
            mock_op_at(&ssa, 0, 2),
            SsaOp::Shr { unsigned: true, .. }
        ));
    }

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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "signed div strength reduction", |ssa| {
            run(ssa, &0u32, &log, &|_| true)
        });
        assert!(
            changed,
            "signed div by power of two should reduce when non-negative"
        );
        assert!(log.has(EventKind::StrengthReduced));
    }

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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(
            &mut ssa,
            "signed div unknown-sign strength reduction",
            |ssa| run(ssa, &0u32, &log, &|_| false),
        );
        assert!(
            !changed,
            "signed div should NOT reduce when non-negativity not proven"
        );
    }

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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "unsigned rem strength reduction", |ssa| {
            run(ssa, &0u32, &log, &|_| true)
        });
        assert!(changed, "unsigned rem by power of two should reduce to and");
        assert!(log.has(EventKind::StrengthReduced));
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::And { .. }));
    }

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
        block.add_instruction(instr(SsaOp::Return { value: Some(r1) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "multiple strength reductions", |ssa| {
            run(ssa, &0u32, &log, &|_| true)
        });
        assert!(changed, "multiple reductions should all fire");
        assert!(log.count_kind(EventKind::StrengthReduced) >= 2);
    }

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
        block.add_instruction(instr(SsaOp::Return {
            value: Some(result),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "no-candidate strength reduction", |ssa| {
            run(ssa, &0u32, &log, &|_| true)
        });
        assert!(!changed, "no strength-reducible ops should return false");
    }

    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "empty strength reduction", |ssa| {
            run(ssa, &0u32, &log, &|_| true)
        });
        assert!(!changed);
    }

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
        block.add_instruction(instr(SsaOp::Return { value: Some(r2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed =
            run_mock_pass_boundary(&mut ssa, "shared constant strength reduction", |ssa| {
                run(ssa, &0u32, &log, &|_| true)
            });
        assert!(
            !changed,
            "shared constant should not be rewritten for strength reduction"
        );
    }
}
