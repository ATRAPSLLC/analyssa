//! Global Value Numbering (GVN) pass - eliminates redundant computations
//! across basic blocks.
//!
//! Detects when the same expression is computed multiple times with
//! identical operands and replaces later uses with the earlier result.
//!
//! # Algorithm
//!
//! 1. **Key construction**: For each pure operation, build a hashable value key
//!    from its opcode and operands. Binary, unary, and `LoadArg` ops get
//!    dedicated keys (a `LoadArg` only for an argument slot the function never
//!    writes); every other pure single-result op falls back to a generic key
//!    built from its operand-normalized shape. Commutative operations are
//!    normalized so that `a + b` and `b + a` produce the same key.
//!
//!    An operand that is a constant is keyed as the first copy of that
//!    constant: a lift states an immediate afresh at every instruction that
//!    reads one, and a fold that builds a shift or a mask states its own
//!    amount, so keyed by their own ids two computations of one value over two
//!    copies of one constant would key apart. A copy is the same constant when
//!    its value compares equal structurally ([`ConstValue`]'s equality is
//!    bitwise for floats, so `0.0` and `-0.0` stay apart) and its type is the
//!    same, so an integer and a pointer of one bit pattern do too. The copies
//!    themselves are never merged: each is where its own readers' evidence of
//!    what the constant is attaches - the `0` an integer starts from and the
//!    null a pointer is compared with are one bit pattern of two types.
//!
//!    Exclusions fall into two groups. **Not pure**: overflow-checked
//!    arithmetic (`AddOvf`, `SubOvf`, `MulOvf`), `Ckfinite`, and anything
//!    touching memory. **Pure but not a function of their SSA operands**:
//!    `Const` (handled by constant folding), `CallClobber` (defines fresh
//!    undefined values), and `Phi` (block-relative). Multi-result ops are excluded by the single-def gate,
//!    since GVN cannot remap their secondary definitions.
//! 2. **Hash-consing**: If the same value key was seen before, the
//!    later definition is redundant - queue it for replacement.
//! 3. **Replacement**: For each redundant result, call
//!    [`SsaFunction::replace_uses_checked`] to forward safe instruction uses
//!    to the original result. If no uses remain, nop-out the redundant
//!    instruction so a subsequent repair or DCE run removes it. Nopping
//!    (rather than leaving the instruction live) prevents ping-ponging
//!    with DCE on the next normalization iteration.
//!
//! # Scope
//!
//! GVN operates across all blocks in a single function (global within the
//! function). It does not perform interprocedural value numbering.
//!
//! # Complexity
//!
//! O(n) in the number of instructions - a single linear scan plus hash map
//! lookups.
//!
//! The generic key stores the operand-normalized [`SsaOp`] itself, so probing
//! is a structural `Eq`/`Hash` comparison with no formatting, no `String`
//! allocation, and no possibility of two distinct ops colliding onto one key.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    Result,
    analysis::cfg::SsaCfg,
    events::EventKind,
    facts::Substitution,
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        ops::{BinaryOpKind, OperandRole, SsaOp, UnaryOpKind},
        phi::PhiEdge,
        value::ConstValue,
        variable::{SsaVarId, SsaVariable, VariableOrigin},
        varstore::VarSet,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// A hashable key representing an operation's value for value numbering.
///
/// Captures the semantics of an expression - the operation kind and
/// operands, not the destination. Two operations with the same key compute
/// the same value and are candidates for elimination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ValueKey<T: Target> {
    /// A binary operation.
    ///
    /// Fields: `(kind, unsigned_flag, left_operand, right_operand)`.
    /// The unsigned flag is included for operations where signedness affects
    /// semantics (`Div`, `Rem`, `Shr`, `Clt`, `Cgt`). For other operations
    /// it is normalized to `false`. Commutative operations are normalized
    /// so that `(Add, false, a, b)` and `(Add, false, b, a)` produce the
    /// same hash.
    Binary(BinaryOpKind, bool, SsaVarId, SsaVarId),
    /// A unary operation.
    ///
    /// Fields: `(kind, operand)`. Not normalized for commutativity (unary
    /// operations are non-commutative by definition).
    Unary(UnaryOpKind, SsaVarId),
    /// Load of a method argument.
    ///
    /// Two loads of the same argument produce the same value only while the
    /// slot is not written, so [`GlobalValueNumberingPass`] drops this key for any argument the
    /// function may write (see [`written_arguments`]). The field is the
    /// zero-based argument index.
    LoadArg(usize),
    /// Any other pure, single-result operation.
    ///
    /// The key is the operation itself with its definitions normalized to a
    /// fixed sentinel, compared structurally via the op's derived `Eq`/`Hash`.
    /// Two ops are equal iff they have the same opcode, the same immediate
    /// fields, and the same use operands - i.e. they compute the same value.
    /// Because equality is derived, every field participates automatically, so
    /// this arm covers all pure ops (vector compute, conversions, bit
    /// manipulation, `Select`, …) - current and future - with no per-op code.
    Generic(Box<SsaOp<T>>),
}

impl<T: Target> ValueKey<T> {
    /// Builds a normalized value key from an SSA operation.
    ///
    /// Dispatches to a curated fast path for the common shapes - binary ops
    /// (normalized for commutativity and signedness), unary ops, and
    /// `LoadArg` - then falls back to a [`ValueKey::Generic`] key for any remaining
    /// pure, single-result op, with its definition operands rewritten to
    /// [`GVN_DEF_SENTINEL`] so the destination name does not distinguish two
    /// otherwise-identical computations. Ops that are not a pure function of
    /// their SSA operands are rejected here rather than in `run`: the
    /// may-throw arithmetic and carry-coupled rotates that `as_binary_op`/
    /// `as_unary_op` expose, plus the pure-but-not-operand-determined `Const`,
    /// `CallClobber`, and `Phi`. Multi-result ops are dropped
    /// by the single-def gate, since GVN cannot remap their secondary
    /// definitions.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to key.
    ///
    /// # Returns
    ///
    /// `Some((key, dest))` pairing the value key with the operation's single
    /// result, or `None` for operations that must not be value-numbered.
    fn from_op(op: &SsaOp<T>) -> Option<(Self, SsaVarId)> {
        if let Some(info) = op.as_binary_op() {
            // Skip overflow-checked operations: they may throw.
            if matches!(
                info.kind,
                BinaryOpKind::AddOvf | BinaryOpKind::SubOvf | BinaryOpKind::MulOvf
            ) {
                return None;
            }
            let normalized = info.normalized();
            let (kind, unsigned, left, right) = normalized.value_key();
            return Some((Self::Binary(kind, unsigned, left, right), normalized.dest));
        }

        if let Some(info) = op.as_unary_op() {
            // Skip Ckfinite (it may throw).
            if info.kind == UnaryOpKind::Ckfinite {
                return None;
            }
            return Some((Self::Unary(info.kind, info.operand), info.dest));
        }

        if let SsaOp::LoadArg { dest, arg_index } = op {
            return Some((Self::LoadArg(*arg_index as usize), *dest));
        }

        // Generic fallback: any op whose one result is a function of its
        // operands ([`SsaOp::computes_from_operands`]): pure, so it touches no
        // memory and cannot throw, single-result, since GVN cannot remap a
        // secondary definition, and none of `Const` (folded by constant
        // propagation; an operand that is one is keyed as its first copy, see
        // `run`), `CallClobber` (every clobber's operands are definitions,
        // which would all key alike) and `Phi` (the key omits its block).
        //
        // Note this gate is *not* what protects flag-producing ops: the binary
        // fast path above returns before reaching it, so a flag-setting `Add`
        // does get value-numbered. That is deliberate and sound - its `dest` is
        // the same function of the same operands whether or not it also writes
        // flags - but it makes the removal step in `run` responsible for
        // checking every definition before deleting an instruction.
        if !op.computes_from_operands() {
            return None;
        }
        let dest = op.defs().next()?;
        let mut normalized = op.clone();
        normalized.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Def | OperandRole::FlagsDef) {
                *var = GVN_DEF_SENTINEL;
            }
        });
        Some((Self::Generic(Box::new(normalized)), dest))
    }
}

/// Sentinel SSA id substituted for an operation's definition when building a
/// [`ValueKey::Generic`] key, so the destination name does not distinguish two
/// otherwise-identical computations.
///
/// Uses [`SsaVarId::PLACEHOLDER`] rather than a real index: `from_index(0)` is a
/// live variable id, so a key built from it is indistinguishable from one whose
/// destination genuinely is `v0`.
const GVN_DEF_SENTINEL: SsaVarId = SsaVarId::PLACEHOLDER;

/// Tests whether `leader`'s definition dominates `block_idx`, gating whether a
/// redundant computation there may be forwarded to it.
///
/// A value key's leader is only valid where the leader is *available*, and
/// availability is dominance. Blocks are visited in index order during the scan
/// in [`GlobalValueNumberingPass`], which says nothing about dominance, so this check prevents
/// forwarding a use to a definition on a sibling branch that never reaches it.
/// A definition trivially dominates its own block, so the same-block case is
/// short-circuited before consulting the dominator tree.
///
/// # Arguments
///
/// * `ssa` - The SSA function, used to look up `leader`'s definition site.
/// * `dominators` - Precomputed dominator tree for the function's CFG.
/// * `leader` - The candidate leader variable.
/// * `block_idx` - The block containing the redundant computation.
///
/// # Returns
///
/// `true` if `leader`'s defining block dominates `block_idx` (or is that
/// block); `false` if `leader` has no known definition site.
fn leader_dominates<T: Target>(
    ssa: &SsaFunction<T>,
    dominators: &DominatorTree,
    leader: SsaVarId,
    block_idx: usize,
) -> bool {
    let Some(def_block) = ssa.variable(leader).map(|var| var.def_site().block()) else {
        return false;
    };
    def_block == block_idx || dominators.dominates(NodeId::new(def_block), NodeId::new(block_idx))
}

/// Returns the argument slots `ssa` may write.
///
/// A slot is written by any instruction defining a variable of that
/// argument's origin (a store to the argument), and may be written through
/// memory once its address is taken with `LoadArgAddr`. A `LoadArg` itself
/// only reads the slot, so its definition does not count even when it carries
/// the argument's origin.
///
/// # Arguments
///
/// * `ssa` - The function to scan.
///
/// # Returns
///
/// The zero-based indices of every argument slot that may be written.
fn written_arguments<T: Target>(ssa: &SsaFunction<T>) -> FxHashSet<usize> {
    let mut written = FxHashSet::default();
    for block in ssa.blocks() {
        for instr in block.instructions() {
            match instr.op() {
                SsaOp::LoadArg { .. } => {}
                SsaOp::LoadArgAddr { arg_index, .. } => {
                    written.insert(usize::from(*arg_index));
                }
                op => {
                    for def in op.defs() {
                        if let Some(VariableOrigin::Argument(index)) =
                            ssa.variable(def).map(SsaVariable::origin)
                        {
                            written.insert(usize::from(index));
                        }
                    }
                }
            }
        }
    }
    written
}

/// Collects, in a single pass, the set of variables still referenced by any
/// instruction operand or phi operand.
///
/// Computed once over the post-replacement function so the removal loop in
/// [`GlobalValueNumberingPass`] can test each redundant instruction's definitions against a
/// snapshot of live uses; rescanning per redundant variable would be
/// O(redundant × instructions). Phi operands are folded in alongside
/// instruction uses because a value forwarded into a phi keeps its source live.
///
/// # Arguments
///
/// * `ssa` - The function to scan.
///
/// # Returns
///
/// A [`VarSet`], indexed by `SsaVarId::index()`, of every variable still used.
fn collect_used_vars<T: Target>(ssa: &SsaFunction<T>) -> VarSet {
    let mut used = VarSet::new(ssa.var_id_bound());
    for block in ssa.blocks() {
        for instr in block.instructions() {
            instr.op().for_each_use(|v| {
                used.insert(v);
            });
        }
        for phi in block.phi_nodes() {
            for operand in phi.operands() {
                used.insert(operand.value());
            }
        }
    }
    used
}

/// Pass that eliminates redundant computations via hash-consed value
/// numbering.
///
/// Detects when the same
/// expression is computed multiple times and replaces later uses with
/// the earlier result.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - replaces uses of variables and
/// nops out redundant instructions whose results are dead; never modifies the
/// CFG or adds instructions. The edit runs in a checked session that repairs
/// the function at its boundary, so the pass
/// [repairs its own SSA](FunctionPass::repairs_ssa).
#[derive(Debug, Default, Clone, Copy)]
pub struct GlobalValueNumberingPass;

impl GlobalValueNumberingPass {
    /// Creates a new global value numbering pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for GlobalValueNumberingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "global-value-numbering"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Eliminate redundant computations via value numbering"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Runs Global Value Numbering on `ssa`: replaces redundant computations with
    /// references to the original result, and reports the exact number of uses
    /// forwarded.
    ///
    /// Runs in two phases. The scan builds a hash-consed value key for every
    /// keyable instruction and records a redundant pair whenever a key recurs at a
    /// site its earlier leader dominates (a non-dominating recurrence is left in
    /// place - replacing the leader would only relocate the problem). The edit
    /// phase then forwards each redundant result via `replace_uses_checked_with`
    /// against a single dominator tree built once for the batch, and nops out an
    /// instruction only once *every* definition it makes is dead - a flag-producing
    /// `Add` keys on its result alone, so its live `flags` definition must be
    /// checked before removal. The rollback policy is build-dependent: debug
    /// verifies and rolls back on failure, release runs under
    /// [`SsaRollbackPolicy::Never`], which shapes what a failed edit reports.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// The number of uses replaced. On a failed edit under
    /// [`SsaRollbackPolicy::Never`], returns at least one so the pass-group
    /// transaction verifies and rolls back the possibly-damaged function rather
    /// than trusting an "unchanged" report; under debug rollback it returns zero,
    /// since the function was restored.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let mut value_map: FxHashMap<ValueKey<T>, SsaVarId> = FxHashMap::default();
        let mut redundant: Vec<(SsaVarId, SsaVarId, usize, usize)> = Vec::new();
        // Each phi redundant with an earlier phi of its block, with that phi.
        let mut redundant_phis: Vec<(SsaVarId, SsaVarId)> = Vec::new();

        // A leader is only a leader where it is *available*, and availability is
        // dominance. Blocks are visited in index order, which says nothing about
        // dominance: two sibling branches both computing `x + 1` would make the
        // lower-numbered one the leader and forward the other's uses to a
        // definition that does not reach them - a dominance violation that only a
        // verifying (debug) build would catch and roll back, so debug and release
        // would lift the same bytes differently.
        //
        // The same tree serves the edit below: replacing uses never rewrites a
        // terminator, so the CFG it was built from is the CFG the edit sees.
        let dominators = {
            let eh = SsaCfg::from_ssa(ssa);
            compute_dominators(&eh, NodeId::new(0))
        };
        let written_args = written_arguments(ssa);

        // What each operand keys as: a redundant value as its leader, so an
        // operation reading one keys as the one reading the leader, found in this
        // scan rather than the next; and a copy of a constant as the first copy of
        // the same value and type, so `x + c'` over a second copy of `c` keys as
        // `x + c`. The copies are not themselves replaced.
        let mut leaders: FxHashMap<SsaVarId, SsaVarId> = FxHashMap::default();
        let mut firsts: FxHashMap<(ConstValue<T>, T::Type), SsaVarId> = FxHashMap::default();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op()
                && let Some(variable) = ssa.variable(*dest)
            {
                let first = *firsts
                    .entry((value.clone(), variable.var_type().clone()))
                    .or_insert(*dest);
                if first != *dest {
                    leaders.insert(*dest, first);
                }
            }
        }
        // Two phis of one block that take one value on every way in are one
        // value: a phi is a function of its block and of what each edge brings,
        // so it is keyed by both. The later is redundant with the first, which is
        // defined where it is, so every reader of the one reaches the other. Its
        // readers move to the first, and the phi left with none goes with the
        // dead code; its own readers below key as reading the first.
        for block in ssa.blocks() {
            let mut joined: FxHashMap<Vec<(usize, PhiEdge, SsaVarId)>, SsaVarId> =
                FxHashMap::default();
            for phi in block.phi_nodes() {
                let mut operands: Vec<(usize, PhiEdge, SsaVarId)> = phi
                    .operands()
                    .iter()
                    .map(|operand| {
                        let value = operand.value();
                        (
                            operand.predecessor(),
                            operand.edge(),
                            leaders.get(&value).copied().unwrap_or(value),
                        )
                    })
                    .collect();
                operands.sort_unstable_by_key(|(predecessor, _, value)| (*predecessor, *value));
                match joined.get(&operands).copied() {
                    Some(first) => {
                        leaders.insert(phi.result(), first);
                        redundant_phis.push((phi.result(), first));
                    }
                    None => {
                        joined.insert(operands, phi.result());
                    }
                }
            }
        }
        for block in ssa.blocks() {
            let block_idx = block.id();
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let mut reads_redundant = false;
                instr.op().for_each_use(|var| {
                    reads_redundant |= leaders.contains_key(&var);
                });
                let led;
                let op = if reads_redundant {
                    let mut renamed = instr.op().clone();
                    renamed.visit_operands_mut(|role, var| {
                        if role == OperandRole::Use
                            && let Some(leader) = leaders.get(var)
                        {
                            *var = *leader;
                        }
                    });
                    led = renamed;
                    &led
                } else {
                    instr.op()
                };
                if let Some((key, dest)) = ValueKey::from_op(op) {
                    // A `LoadArg` reads the slot's current value, not an SSA
                    // operand; it is only a function of its index when nothing in
                    // the function can write that slot.
                    if matches!(key, ValueKey::LoadArg(index) if written_args.contains(&index)) {
                        continue;
                    }
                    match value_map.get(&key).copied() {
                        Some(original)
                            if leader_dominates(ssa, &dominators, original, block_idx) =>
                        {
                            leaders.insert(dest, original);
                            redundant.push((dest, original, block_idx, instr_idx));
                        }
                        // A recurrence of the key whose leader does not reach it.
                        // Leave the leader in place: replacing it with this one
                        // would only move the problem to whatever it does dominate.
                        Some(_) => {}
                        None => {
                            value_map.insert(key, dest);
                        }
                    }
                }
            }
        }

        let mut total_replaced: usize = 0;
        let mut removals: Vec<(usize, usize)> = Vec::new();
        let rollback = SsaRollbackPolicy::OnFailure;
        let edit_result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(rollback),
            |editor| {
                // One dominator tree for the whole batch, the one the scan built.
                // Replacing uses never rewrites a terminator, so it stays valid
                // across every forward below; the per-site check consults it only
                // for cross-block uses (same-block uses fall back to instruction
                // ordering). `replace_uses_checked` would rebuild the CFG and the
                // tree, and rescan every block, once per redundant pair.
                let forwarded = redundant_phis.iter().copied().chain(
                    redundant
                        .iter()
                        .map(|(var, original, ..)| (*var, *original)),
                );
                for (redundant_var, original_var) in forwarded {
                    let (redundant_var, original_var) = (&redundant_var, &original_var);
                    let result = editor.replace_uses_checked_with(
                        *redundant_var,
                        *original_var,
                        Some(&dominators),
                        Substitution::Gvn,
                    );
                    if result.replaced > 0 {
                        env.record(EventKind::ConstantFolded).message(format!(
                            "GVN: {redundant_var} → {original_var} ({} uses)",
                            result.replaced
                        ));
                        total_replaced = total_replaced.saturating_add(result.replaced);
                    }
                }

                // Decide removals in a single pass over the post-replacement
                // function. Rescanning per redundant variable would be
                // O(redundant * instructions).
                let used = collect_used_vars(editor.function());
                // A redundant value that is a version of a slot the function reads
                // as storage is also a write a later read through the slot sees:
                // its value is forwarded, but the write stays.
                let slots = editor.function().storage_slots();
                for (_redundant_var, _original_var, block_idx, instr_idx) in &redundant {
                    // Check *every* definition, not just the primary result. A
                    // flag-producing `Add` defines both `dest` and `flags`, and
                    // `BinaryOpInfo::value_key()` does not include `flags`, so two
                    // adds that differ only in their flags destination key
                    // identically. Forwarding the redundant `dest` is still sound -
                    // the arithmetic result is the same function of the same
                    // operands - but the instruction may only be removed once every
                    // definition it makes is dead. Removing it on the strength of a
                    // dead `dest` alone leaves a live `flags` use dangling.
                    let all_defs_dead = editor
                        .function()
                        .block(*block_idx)
                        .and_then(|block| block.instruction(*instr_idx))
                        .is_some_and(|instr| {
                            instr.op().defs().all(|def| !used.contains(def))
                                && !slots.written_by(editor.function(), instr.op())
                        });
                    if all_defs_dead {
                        removals.push((*block_idx, *instr_idx));
                    }
                }

                for (block_idx, instr_idx) in &removals {
                    editor.nop_instruction(*block_idx, *instr_idx)?;
                }
                Ok(())
            },
        );

        if edit_result.is_err() {
            // The rollback policy differs by build: debug restores the function, so
            // nothing changed and "unchanged" is the truth. Release runs under
            // `SsaRollbackPolicy::Never`, where the edits stay applied - reporting
            // no change there would make the pass-group transaction skip both
            // verification and rollback, keeping damaged IR and keeping it
            // unchecked. Report the change so the transaction handles it.
            return Ok(!matches!(rollback, SsaRollbackPolicy::OnFailure));
        }
        Ok(total_replaced > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{SsaEvaluator, context::AnalysisContext},
        events::EventLog,
        ir::{
            FlagCondition, SsaDefSpec, SsaFunctionBuilder,
            block::SsaBlock,
            float::FloatFormat,
            instruction::SsaInstruction,
            ops::RuntimeValueKind,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{
            MACHINE, MockTarget, MockType, assert_mock_valid_full, mock_op_at,
            run_mock_pass_repaired_boundary,
        },
    };

    /// Runs the function from its entry and returns the value it returns.
    fn returned(ssa: &SsaFunction<MockTarget>) -> Option<ConstValue<MockTarget>> {
        let mut evaluator = SsaEvaluator::new(ssa, MACHINE);
        let trace = evaluator.execute_from_entry(1_000);
        let last = trace.last_block()?;
        match ssa.block(last)?.control_terminator()? {
            SsaOp::Return {
                value: Some(value), ..
            } => evaluator.get_concrete(*value).cloned(),
            _ => None,
        }
    }

    /// Two identical flag-setting adds followed by a `BranchFlags` reading the
    /// *second* one's flags - the shape any lifter emits for a repeated
    /// `add eax, ebx` with a later conditional jump.
    ///
    /// `BinaryOpInfo::value_key()` does not include `flags`, so both adds key
    /// identically and GVN forwards the second `dest` to the first. That part is
    /// sound: the arithmetic result is the same function of the same operands.
    /// What is *not* sound is then deleting the instruction, because its `flags`
    /// definition is still live.
    fn two_flag_setting_adds_with_a_flags_consumer() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);

        let mut mk = |origin, block, instr| {
            ssa.create_variable(
                VariableOrigin::Local(origin),
                0,
                DefSite::instruction(block, instr),
                MockType::I32,
            )
        };
        let v0 = mk(0, 0, 0);
        let v1 = mk(1, 0, 1);
        let sum1 = mk(2, 0, 2);
        let flags1 = mk(3, 0, 2);
        let sum2 = mk(4, 0, 3);
        let flags2 = mk(5, 0, 3);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: sum1,
            left: v0,
            right: v1,
            flags: Some(flags1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: sum2,
            left: v0,
            right: v1,
            flags: Some(flags2),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::BranchFlags {
            flags: flags2,
            condition: FlagCondition::Zero,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(sum1))));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(sum2))));
        ssa.add_block(b2);

        ssa.recompute_uses();
        ssa
    }

    /// GVN may forward the redundant *value*, but it must not delete an
    /// instruction that still has a live secondary definition. Deleting it
    /// leaves `BranchFlags` reading a variable that nothing defines.
    #[test]
    fn gvn_keeps_a_redundant_add_whose_flags_are_still_live() {
        let mut ssa = two_flag_setting_adds_with_a_flags_consumer();
        let events: EventLog<MockTarget> = EventLog::new();

        assert!(
            GlobalValueNumberingPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &events, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );

        // The flags definition must survive somewhere in the function.
        let flags2 = SsaVarId::from_index(5);
        let defines_flags2 = ssa.blocks().iter().any(|block| {
            block
                .instructions()
                .iter()
                .any(|instr| instr.op().defs().any(|d| d == flags2))
        });
        assert!(
            defines_flags2,
            "GVN removed the redundant add but its `flags` definition is still \
             read by BranchFlags - the branch now reads an undefined variable"
        );
        assert_mock_valid_full(&ssa, "after GVN over flag-setting adds");
    }

    /// Commutative binary ops (`Add`) key identically under operand swap, while
    /// non-commutative ones (`Sub`) do not.
    #[test]
    fn value_key_binary_commutative() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);

        let add_op1: SsaOp<MockTarget> = SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        };
        let add_op2: SsaOp<MockTarget> = SsaOp::Add {
            dest: v3,
            left: v1,
            right: v0,
            flags: None,
        };
        let (k1, _) = ValueKey::from_op(&add_op1).expect("add should produce a value key");
        let (k2, _) = ValueKey::from_op(&add_op2).expect("add should produce a value key");
        assert_eq!(k1, k2, "Add should be commutative");

        let sub_op1: SsaOp<MockTarget> = SsaOp::Sub {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        };
        let sub_op2: SsaOp<MockTarget> = SsaOp::Sub {
            dest: v3,
            left: v1,
            right: v0,
            flags: None,
        };
        let (k3, _) = ValueKey::from_op(&sub_op1).expect("sub should produce a value key");
        let (k4, _) = ValueKey::from_op(&sub_op2).expect("sub should produce a value key");
        assert_ne!(k3, k4, "Sub should NOT be commutative");
    }

    /// Binary and unary ops produce their curated keys while `Const` produces
    /// none.
    #[test]
    fn value_key_recognizes_unary_and_skips_const() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let add_op: SsaOp<MockTarget> = SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        };
        let (key, dest) = ValueKey::from_op(&add_op).expect("add should produce a value key");
        assert_eq!(dest, v2);
        assert!(matches!(key, ValueKey::Binary(BinaryOpKind::Add, _, _, _)));

        let neg_op: SsaOp<MockTarget> = SsaOp::Neg {
            dest: v1,
            operand: v0,
            flags: None,
        };
        let (key, dest) = ValueKey::from_op(&neg_op).expect("neg should produce a value key");
        assert_eq!(dest, v1);
        assert!(matches!(key, ValueKey::Unary(UnaryOpKind::Neg, _)));

        let const_op: SsaOp<MockTarget> = SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        };
        assert!(ValueKey::from_op(&const_op).is_none());
    }

    /// **An operation over a copy of a constant keys as one over the first.**
    /// `x + 4` and `x + 4`, each reading its own copy of the constant, are one
    /// sum, and both copies stay, each where its readers' evidence attaches.
    /// The controls keep both sums: the constants differ, they are `0.0` and
    /// `-0.0` (one value to `==`, two bit patterns), or they share a bit
    /// pattern at two types.
    #[test]
    fn two_copies_of_one_constant_are_one_value() {
        let sums = |first: (ConstValue<MockTarget>, MockType),
                    second: (ConstValue<MockTarget>, MockType)| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            builder
                .in_block(0, |block| {
                    let x = block.load_arg(SsaDefSpec::tmp(first.1), 0)?;
                    let c = block.const_value(SsaDefSpec::tmp(first.1), first.0.clone())?;
                    let d = block.const_value(SsaDefSpec::tmp(second.1), second.0.clone())?;
                    let a = block.add(SsaDefSpec::tmp(first.1), x, c)?;
                    let b = block.add(SsaDefSpec::tmp(first.1), x, d)?;
                    let r = block.xor(SsaDefSpec::tmp(first.1), a, b)?;
                    block.ret(Some(r))
                })
                .expect("build block");
            let mut ssa = builder.finish().expect("finish");
            assert!(
                GlobalValueNumberingPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &EventLog::new(), AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            let constants = ssa
                .iter_instructions()
                .filter(|(_, _, instruction)| matches!(instruction.op(), SsaOp::Const { .. }))
                .count();
            assert_eq!(constants, 2, "both copies stay:\n{ssa}");
            ssa.iter_instructions()
                .filter(|(_, _, instruction)| matches!(instruction.op(), SsaOp::Add { .. }))
                .count()
        };
        let int = |value| (ConstValue::I32(value), MockType::I32);
        let double = |bits| {
            (
                ConstValue::float(FloatFormat::Double, bits).expect("a double"),
                MockType::F64,
            )
        };
        assert_eq!(sums(int(4), int(4)), 1, "one constant, one sum");
        assert_eq!(sums(int(4), int(5)), 2, "two constants");
        assert_eq!(
            sums(double(0), double(0x8000_0000_0000_0000)),
            2,
            "0.0 and -0.0"
        );
        assert_eq!(
            sums(int(4), (ConstValue::I32(4), MockType::U32)),
            2,
            "one bit pattern at two types"
        );
    }

    /// The generic key value-numbers any pure, single-result op (here `Select`,
    /// which the curated binary/unary/load arms do not cover) by structural
    /// identity, while excluding impure ops. This is what lets GVN cover the
    /// ~150 vector/native ops with no per-op code.
    #[test]
    fn generic_value_key_covers_pure_ops_and_skips_impure() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);
        let v4 = SsaVarId::from_index(4);

        // Two `Select`s with the same inputs compute the same value despite
        // different destinations -> identical generic key, distinct dests.
        let sel1: SsaOp<MockTarget> = SsaOp::Select {
            dest: v3,
            condition: v0,
            true_val: v1,
            false_val: v2,
        };
        let sel2: SsaOp<MockTarget> = SsaOp::Select {
            dest: v4,
            condition: v0,
            true_val: v1,
            false_val: v2,
        };
        let (k1, d1) = ValueKey::from_op(&sel1).expect("select is pure single-def");
        let (k2, d2) = ValueKey::from_op(&sel2).expect("select is pure single-def");
        assert!(matches!(k1, ValueKey::Generic(_)));
        assert_eq!(k1, k2, "same computation -> same key");
        assert_eq!((d1, d2), (v3, v4));

        // Differing input operand -> different key.
        let sel3: SsaOp<MockTarget> = SsaOp::Select {
            dest: v4,
            condition: v0,
            true_val: v2,
            false_val: v1,
        };
        let (k3, _) = ValueKey::from_op(&sel3).unwrap();
        assert_ne!(k1, k3);
    }

    /// `CallClobber` defines *fresh, undefined* values for caller-saved
    /// registers. Two clobbers are never the same value, but the generic key
    /// normalizes every `Def` operand to a sentinel - so all single-output
    /// clobbers in a function would key identically and GVN would forward one
    /// call's clobbered register to another's.
    #[test]
    fn value_key_does_not_number_call_clobber() {
        let v0 = SsaVarId::from_index(0);

        let clobber: SsaOp<MockTarget> = SsaOp::CallClobber {
            outputs: vec![v0],
            machine: None,
        };

        assert!(
            ValueKey::from_op(&clobber).is_none(),
            "CallClobber must never be value-numbered: it defines fresh \
             undefined values, not a function of its inputs"
        );
    }

    /// **Two values the implementation chooses are two values.** `v0` and
    /// `v1` are each [`RuntimeValueKind::Unspecified`] after a constant and
    /// `v2 = v0 - v1`: the verifier accepts them in the middle of the block,
    /// GVN leaves the subtraction reading both, and neither computes from
    /// its operands or is pure, while each is removable unread.
    #[test]
    fn two_unspecified_values_stay_two_values() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        for (index, site) in [1usize, 2, 3, 0].into_iter().enumerate() {
            ssa.create_variable(
                VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
                0,
                DefSite::instruction(0, site),
                MockType::I32,
            );
        }
        let [v0, v1, v2, v3] = [0, 1, 2, 3].map(SsaVarId::from_index);
        let unspecified = |dest| SsaOp::RuntimeValue {
            dest,
            kind: RuntimeValueKind::Unspecified,
        };
        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        for op in [
            SsaOp::Const {
                dest: v3,
                value: ConstValue::I32(1),
            },
            unspecified(v0),
            unspecified(v1),
            SsaOp::Sub {
                dest: v2,
                left: v0,
                right: v1,
                flags: None,
            },
            SsaOp::returns(Some(v2)),
        ] {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        ssa.add_block(block);
        ssa.repair_ssa();
        assert_mock_valid_full(&ssa, "unspecified values in the middle of a block");
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            GlobalValueNumberingPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        match mock_op_at(&ssa, 0, 3) {
            SsaOp::Sub { left, right, .. } => assert_ne!(left, right, "two choices merged"),
            other => panic!("expected Sub, got {other:?}"),
        }
        let op = unspecified(v0);
        assert!(!op.computes_from_operands());
        assert!(!op.effects().is_pure());
        assert!(op.effects().removable_when_unused());
    }

    /// A phi is block-relative by definition: its value depends on the block it
    /// heads, which the generic key does not encode.
    #[test]
    fn value_key_does_not_number_phi() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let phi: SsaOp<MockTarget> = SsaOp::Phi {
            dest: v2,
            operands: vec![(0, v0), (1, v1)],
        };

        assert!(
            ValueKey::from_op(&phi).is_none(),
            "Phi must never be value-numbered: the key omits the defining block"
        );
    }

    /// **Two phis of one block that take one value on every way in are one.**
    /// `3` joins `x` from `1` and `y` from `2` twice, as `p` and as `q`, and
    /// adds them: the sum reads `p` twice. A third phi, `r`, takes `z` from
    /// `2`: it is another value and stays.
    #[test]
    fn phis_of_one_block_alike_on_every_edge_are_one() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(4, 0);
        let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, at),
                MockType::I32,
            )
        };
        let condition = argument(&mut ssa, 0, 0);
        let x = argument(&mut ssa, 1, 1);
        let y = argument(&mut ssa, 2, 2);
        let z = argument(&mut ssa, 3, 3);
        let joined = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::phi(3),
                MockType::I32,
            )
        };
        let p = joined(&mut ssa, 4);
        let q = joined(&mut ssa, 5);
        let r = joined(&mut ssa, 6);
        let sum = ssa.create_variable(
            VariableOrigin::Local(7),
            0,
            DefSite::instruction(3, 0),
            MockType::I32,
        );
        let total = ssa.create_variable(
            VariableOrigin::Local(8),
            0,
            DefSite::instruction(3, 1),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        for (index, dest) in [condition, x, y, z].into_iter().enumerate() {
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
                dest,
                arg_index: u16::try_from(index).unwrap_or(0),
            }));
        }
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 2,
        }));
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        let mut b3 = SsaBlock::new(3);
        for (result, second) in [(p, y), (q, y), (r, z)] {
            let mut phi = PhiNode::new(result, VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(x, 1));
            phi.add_operand(PhiOperand::new(second, 2));
            b3.add_phi(phi);
        }
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: sum,
            left: p,
            right: q,
            flags: None,
        }));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: total,
            left: sum,
            right: r,
            flags: None,
        }));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(total))));
        for block in [b0, b1, b2, b3] {
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            GlobalValueNumberingPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ) == Ok(true)
        );
        ssa.repair_ssa();
        assert_mock_valid_full(&ssa, "phis alike after GVN repair");
        match mock_op_at(&ssa, 3, 0) {
            SsaOp::Add { left, right, .. } => assert_eq!((*left, *right), (p, p)),
            other => panic!("expected the sum, found {other:?}"),
        }
        // What the total adds is still the phi that takes `z` on `2`'s edge.
        let added = match mock_op_at(&ssa, 3, 1) {
            SsaOp::Add { right, .. } => *right,
            other => panic!("expected the total, found {other:?}"),
        };
        let from_two = ssa
            .block(3)
            .and_then(|block| block.phi_nodes().iter().find(|phi| phi.result() == added))
            .and_then(|phi| phi.operand_from(2))
            .map(PhiOperand::value);
        assert_eq!(from_two, Some(z), "{ssa:?}");
    }

    /// End-to-end smoke test: identical Add expressions should collapse.
    /// Builds the IR by hand because `SsaFunctionBuilder` is CIL-pinned.
    #[test]
    fn eliminates_identical_binop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        // 5 vars: v0, v1 (consts), v2 (add), v3 (redundant add), v4 (mul)
        for i in 0..5 {
            ssa.create_variable(
                VariableOrigin::Local(i),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);
        let v4 = SsaVarId::from_index(4);

        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v3,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Mul {
            dest: v4,
            left: v2,
            right: v3,
            flags: None,
        }));
        // Control has to leave the block through its last instruction, or the
        // successor models see a block with no edges at all - the invariant
        // `VerifierError::MissingTerminator` states. Hand-built IR is where that
        // gets forgotten, and `assert_mock_valid_full` below is what notices.
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v4))));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;
        let replaced = GlobalValueNumberingPass.run(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true);
        assert!(replaced);
        assert!(!log.is_empty());
        ssa.repair_ssa();
        assert_mock_valid_full(&ssa, "identical binop after GVN repair");

        // mul should now reference v2 twice instead of (v2, v3).
        match mock_op_at(&ssa, 0, 3) {
            SsaOp::Mul { left, right, .. } => {
                assert_eq!(*left, v2);
                assert_eq!(*right, v2);
            }
            other => panic!("expected Mul, got {other:?}"),
        }
    }

    /// A duplicate expression in a dominated successor block is eliminated
    /// (global, cross-block value numbering).
    #[test]
    fn same_expression_across_blocks_eliminated() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        // Each variable's definition site names where the block below defines
        // it: `v0`..`v2` in block 0, `v3` and `v4` in block 1.
        for (index, site) in [
            DefSite::instruction(0, 0),
            DefSite::instruction(0, 1),
            DefSite::instruction(0, 2),
            DefSite::instruction(1, 0),
            DefSite::instruction(1, 1),
        ]
        .into_iter()
        .enumerate()
        {
            ssa.create_variable(
                VariableOrigin::Local(u16::try_from(index).unwrap_or(u16::MAX)),
                0,
                site,
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);
        let v4 = SsaVarId::from_index(4);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v3,
            left: v0,
            right: v1,
            flags: None,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v4,
            left: v2,
            right: v3,
            flags: None,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v4))));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "cross-block GVN", |ssa| {
            GlobalValueNumberingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(
            changed,
            "duplicate expression across blocks should be eliminated"
        );
    }

    /// Operand-swapped non-commutative ops (`a - b` vs `b - a`) are not merged.
    #[test]
    fn non_commutative_ops_not_confused() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        for i in 0..4 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        // a - b and b - a are different
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Sub {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Sub {
            dest: v3,
            left: v1,
            right: v0,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v2))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        // v2 and v3 are different computations - GVN should NOT eliminate either
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "non-commutative GVN", |ssa| {
            GlobalValueNumberingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "swapped non-commutative ops must not be merged");
        // v3's Sub should still be present (not Nop'd)
        assert!(
            matches!(mock_op_at(&ssa, 0, 3), SsaOp::Sub { .. }),
            "non-commutative swapped sub should NOT be nop'd"
        );
    }

    /// Duplicate may-throw overflow ops (`AddOvf`) are never value-numbered.
    #[test]
    fn gvn_skips_impure_operations() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        for i in 0..4 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(4),
        }));
        // Two AddOvf ops with same operands - should NOT be eliminated (may throw)
        block.add_instruction(SsaInstruction::synthetic(SsaOp::AddOvf {
            dest: v2,
            left: v0,
            right: v1,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::AddOvf {
            dest: v3,
            left: v0,
            right: v1,
            unsigned: false,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "impure GVN", |ssa| {
            GlobalValueNumberingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "throwing overflow ops must not be value numbered");
        // Neither AddOvf should be nop'd
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::AddOvf { .. }));
        assert!(matches!(mock_op_at(&ssa, 0, 3), SsaOp::AddOvf { .. }));
    }

    /// A function with no redundant computations reports no change.
    #[test]
    fn no_duplicates_returns_false() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        for i in 0..3 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v2))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "no-duplicate GVN", |ssa| {
            GlobalValueNumberingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "no duplicates should return false");
    }

    /// `Const` and `Return` produce no value key.
    #[test]
    fn value_key_skips_const_and_return() {
        let v0 = SsaVarId::from_index(0);
        let const_op: SsaOp<MockTarget> = SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(0),
        };
        assert!(
            ValueKey::from_op(&const_op).is_none(),
            "Const should not generate a key"
        );

        let ret_op: SsaOp<MockTarget> = SsaOp::returns(None);
        assert!(
            ValueKey::from_op(&ret_op).is_none(),
            "Return should not generate a key"
        );
    }

    /// Two loads of the same argument index are recognized as the same value.
    #[test]
    fn identical_loadargs_are_value_numbered() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 2);
        let v0 = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let v2 = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: v0,
            arg_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: v1,
            arg_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v2))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "loadarg GVN", |ssa| {
            GlobalValueNumberingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "identical LoadArgs should be value numbered");
    }

    /// Two `LoadArg`s of the same slot read different values when the slot is
    /// written between them, so the second must not be forwarded to the first.
    ///
    /// `v1 = 5` defines a variable of argument 0's origin, which is a store to
    /// the slot, so the second load reads 5 while the first reads the unknown
    /// incoming argument.
    #[test]
    fn loadarg_across_an_argument_write_is_not_value_numbered() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let v0 = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = ssa.create_variable(
            VariableOrigin::Argument(0),
            1,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let v2 = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: v0,
            arg_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: v2,
            arg_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v2))));
        ssa.add_block(block);
        ssa.recompute_uses();
        assert_eq!(returned(&ssa), Some(ConstValue::I32(5)));

        let log: EventLog<MockTarget> = EventLog::new();
        let _ = GlobalValueNumberingPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
        );

        assert_eq!(
            returned(&ssa),
            Some(ConstValue::I32(5)),
            "the load after the write must still read the written value"
        );
    }

    /// A duplicate on a sibling branch that does not dominate the use is not
    /// forwarded.
    #[test]
    fn gvn_does_not_forward_from_non_dominating_block() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 6);
        let cond = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let left = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let right = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );
        let original = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let duplicate = ssa.create_variable(
            VariableOrigin::Local(4),
            0,
            DefSite::instruction(2, 0),
            MockType::I32,
        );

        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: left,
            value: ConstValue::I32(2),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: right,
            value: ConstValue::I32(3),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));

        let mut true_block = SsaBlock::new(1);
        true_block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: original,
            left,
            right,
            flags: None,
        }));
        true_block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(original))));

        let mut false_block = SsaBlock::new(2);
        false_block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: duplicate,
            left,
            right,
            flags: None,
        }));
        false_block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(duplicate))));

        ssa.add_block(entry);
        ssa.add_block(true_block);
        ssa.add_block(false_block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "non-dominating GVN", |ssa| {
            GlobalValueNumberingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });

        assert!(
            !changed,
            "GVN must not forward from a sibling branch that does not dominate the use"
        );
        assert!(matches!(mock_op_at(&ssa, 2, 0), SsaOp::Add { .. }));
    }

    /// A nested duplicate expression `(a + b) * c` collapses at every level.
    #[test]
    fn gvn_with_mul_level_expression() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 6);
        for i in 0..7 {
            ssa.create_variable(
                VariableOrigin::Local(i as u16),
                0,
                DefSite::instruction(0, i as usize),
                MockType::I32,
            );
        }
        let a = SsaVarId::from_index(0);
        let b = SsaVarId::from_index(1);
        let c = SsaVarId::from_index(2);
        let expr1 = SsaVarId::from_index(3);
        let expr2 = SsaVarId::from_index(4);
        let result1 = SsaVarId::from_index(5);
        let result2 = SsaVarId::from_index(6);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: b,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: c,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: expr1,
            left: a,
            right: b,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Mul {
            dest: result1,
            left: expr1,
            right: c,
            flags: None,
        }));
        // Duplicate of (a+b)*c
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: expr2,
            left: a,
            right: b,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Mul {
            dest: result2,
            left: expr2,
            right: c,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(result2))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "multi-level GVN", |ssa| {
            GlobalValueNumberingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "multi-level duplicates should be eliminated");
    }
}
