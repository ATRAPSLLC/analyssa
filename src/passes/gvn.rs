//! Global Value Numbering (GVN) pass — eliminates redundant computations
//! across basic blocks.
//!
//! Detects when the same expression is computed multiple times with
//! identical operands and replaces later uses with the earlier result.
//!
//! # Algorithm
//!
//! 1. **Key construction**: For each pure operation, build a hashable value key
//!    from its opcode and operands. Binary, unary, and `LoadArg` ops get
//!    dedicated keys; every other pure single-result op falls back to a generic
//!    key built from its operand-normalized shape. Commutative operations are
//!    normalized so that `a + b` and `b + a` produce the same key.
//!
//!    Exclusions fall into two groups. **Not pure**: overflow-checked
//!    arithmetic (`AddOvf`, `SubOvf`, `MulOvf`), `Ckfinite`, anything touching
//!    memory, and the carry-coupled rotates (`Rcl`/`Rcr`, which carry a hidden
//!    flag dependence). **Pure but not a function of their SSA operands**:
//!    `Const` (handled by constant folding), `ComputeFlags` (carries no opcode
//!    discriminator), `CallClobber` (defines fresh undefined values), and `Phi`
//!    (block-relative). Multi-result ops are excluded by the single-def gate,
//!    since GVN cannot remap their secondary definitions.
//! 2. **Hash-consing**: If the same value key was seen before, the
//!    later definition is redundant — queue it for replacement.
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
//! O(n) in the number of instructions — a single linear scan plus hash map
//! lookups.
//!
//! The generic key stores the operand-normalized [`SsaOp`] itself, so probing
//! is a structural `Eq`/`Hash` comparison with no formatting, no `String`
//! allocation, and no possibility of two distinct ops colliding onto one key.

use std::collections::HashMap;

use crate::{
    analysis::exceptions::EhCfg,
    events::{EventKind, EventListener},
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        ops::{BinaryOpKind, OperandRole, SsaOp, UnaryOpKind},
        variable::SsaVarId,
        varstore::VarSet,
    },
    target::Target,
};

/// Run Global Value Numbering on `ssa`.
///
/// Scans all pure instructions, builds hash-consed value keys, and
/// replaces redundant computations with references to the original result.
///
/// # Arguments
///
/// * `ssa` — The SSA function to optimize in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink receiving one [`EventKind::ConstantFolded`]
///   event per eliminated expression.
///
/// # Returns
///
/// `true` if any redundant computation was eliminated.
pub fn run<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    run_gvn(ssa, method, events) > 0
}

/// A hashable key representing an operation's value for value numbering.
///
/// Captures the semantics of an expression — the operation kind and
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
    /// Loading the same argument always produces the same value within a
    /// single function invocation. The field is the zero-based argument
    /// index.
    LoadArg(usize),
    /// Any other pure, single-result operation.
    ///
    /// The key is the operation itself with its definitions normalized to a
    /// fixed sentinel, compared structurally via the op's derived `Eq`/`Hash`.
    /// Two ops are equal iff they have the same opcode, the same immediate
    /// fields, and the same use operands — i.e. they compute the same value.
    /// Because equality is derived, every field participates automatically, so
    /// this arm covers all pure ops (vector compute, conversions, bit
    /// manipulation, `Select`, …) — current and future — with no per-op code.
    Generic(Box<SsaOp<T>>),
}

impl<T: Target> ValueKey<T> {
    /// Builds a normalized value key from an SSA operation. Returns `None`
    /// for operations that should not be value-numbered (impure operations,
    /// constants, control flow, etc.).
    fn from_op(op: &SsaOp<T>) -> Option<(Self, SsaVarId)> {
        if let Some(info) = op.as_binary_op() {
            // Skip overflow-checked operations (they may throw) and the
            // carry-coupled rotates `Rcl`/`Rcr`, which read and write the carry
            // flag — a hidden input/output not present in their SSA operands, so
            // they are not a pure function of those operands and must not be
            // value-numbered. (`as_binary_op` exposes them as binary ops, so the
            // generic purity gate below is never reached for them.)
            if matches!(
                info.kind,
                BinaryOpKind::AddOvf
                    | BinaryOpKind::SubOvf
                    | BinaryOpKind::MulOvf
                    | BinaryOpKind::Rcl
                    | BinaryOpKind::Rcr
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

        // Generic fallback: any *pure*, single-result op. Gating on
        // `effects().is_pure()` excludes everything that reads/writes memory, is
        // atomic, may throw (overflow-checked arithmetic, `Ckfinite`), or is a
        // carry-coupled rotate (`Rcl`/`Rcr`, which carry a hidden flag
        // dependence). The single-def gate below excludes wide multi-result ops
        // whose secondary definitions GVN cannot remap.
        //
        // Note this gate is *not* what protects flag-producing ops: the binary
        // fast path above returns before reaching it, so a flag-setting `Add`
        // does get value-numbered. That is deliberate and sound — its `dest` is
        // the same function of the same operands whether or not it also writes
        // flags — but it makes the removal step in `run_gvn` responsible for
        // checking every definition before deleting an instruction.
        //
        // Purity alone is not sufficient: the ops below are pure yet are still
        // not a function of their SSA operands, so the generic key cannot
        // distinguish them.
        //
        // * `Const` — constant materialization is handled by constant folding /
        //   propagation, not value numbering.
        // * `ComputeFlags` — models the flags of `bsf`/`bsr`/`popcnt`/`bt`
        //   alike but carries no opcode discriminator, so two different native
        //   flag computations over the same operands key identically.
        // * `CallClobber` — defines *fresh, undefined* values for caller-saved
        //   registers. Its operands are all `Def`s, which normalize to the
        //   sentinel below, so every single-output clobber in a function would
        //   key identically and one call's clobbered register would be
        //   forwarded to another's.
        // * `Phi` — block-relative by definition; the key omits the block the
        //   phi heads.
        if matches!(
            op,
            SsaOp::Const { .. }
                | SsaOp::ComputeFlags { .. }
                | SsaOp::CallClobber { .. }
                | SsaOp::Phi { .. }
        ) || !op.effects().is_pure()
        {
            return None;
        }
        let defs: Vec<SsaVarId> = op.defs().collect();
        let [dest] = defs[..] else {
            return None;
        };
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

/// Returns whether `leader`'s definition dominates `block_idx`, so a redundant
/// computation there can be forwarded to it.
fn leader_dominates<T: Target>(
    ssa: &SsaFunction<T>,
    dominators: &DominatorTree,
    leader: SsaVarId,
    block_idx: usize,
) -> bool {
    let Some(def_block) = ssa.variable(leader).map(|var| var.def_site().block) else {
        return false;
    };
    def_block == block_idx || dominators.dominates(NodeId::new(def_block), NodeId::new(block_idx))
}

/// Internal GVN driver. Returns the number of uses replaced (used by tests).
fn run_gvn<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut value_map: HashMap<ValueKey<T>, SsaVarId> = HashMap::new();
    let mut redundant: Vec<(SsaVarId, SsaVarId, usize, usize)> = Vec::new();

    // A leader is only a leader where it is *available*, and availability is
    // dominance. Blocks are visited in index order, which says nothing about
    // dominance: two sibling branches both computing `x + 1` would make the
    // lower-numbered one the leader and forward the other's uses to a
    // definition that does not reach them. Measured on the MIPS fixture, that
    // produced `DominanceViolation { def_block: 1, use_block: 7 }` — the last
    // remaining reason a debug and a release build lifted the same bytes
    // differently, since only debug verified the edit and rolled it back.
    let dominators = {
        let eh = EhCfg::from_ssa(ssa);
        compute_dominators(&eh, NodeId::new(0))
    };

    for block in ssa.blocks() {
        let block_idx = block.id();
        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            if let Some((key, dest)) = ValueKey::from_op(instr.op()) {
                match value_map.get(&key).copied() {
                    Some(original) if leader_dominates(ssa, &dominators, original, block_idx) => {
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
    let rollback = if cfg!(debug_assertions) {
        SsaRollbackPolicy::OnFailure
    } else {
        SsaRollbackPolicy::Never
    };
    let edit_result = ssa.edit(
        SsaEditOptions::new()
            .with_verify(cfg!(debug_assertions))
            .with_rollback(rollback),
        |editor| {
            // Build the dominator tree once for the whole batch. Replacing uses
            // never rewrites a terminator, so one tree stays valid across every
            // forward below; the per-site check consults it only for cross-block
            // uses (same-block uses fall back to instruction ordering), so a
            // single global tree is behaviour-equivalent to the previous
            // per-pair `replace_uses_checked`, which rebuilt the CFG + dominator
            // tree (and rescanned every block) once per redundant pair.
            let dominators = if editor.function().block_count() > 0 {
                let eh = EhCfg::from_ssa(editor.function());
                Some(compute_dominators(&eh, NodeId::new(0)))
            } else {
                None
            };
            for (redundant_var, original_var, _block_idx, _instr_idx) in &redundant {
                let result = editor.replace_uses_checked_with(
                    *redundant_var,
                    *original_var,
                    dominators.as_ref(),
                );
                if result.replaced > 0 {
                    let event = crate::events::Event {
                        kind: EventKind::ConstantFolded,
                        method: Some(method.clone()),
                        location: None,
                        message: format!(
                            "GVN: {redundant_var} → {original_var} ({} uses)",
                            result.replaced
                        ),
                        pass: None,
                    };
                    events.push(event);
                    total_replaced = total_replaced.saturating_add(result.replaced);
                }
            }

            // Decide removals in a single pass over the post-replacement
            // function. Rescanning per redundant variable would be
            // O(redundant * instructions).
            let used = collect_used_vars(editor.function());
            for (_redundant_var, _original_var, block_idx, instr_idx) in &redundant {
                // Check *every* definition, not just the primary result. A
                // flag-producing `Add` defines both `dest` and `flags`, and
                // `BinaryOpInfo::value_key()` does not include `flags`, so two
                // adds that differ only in their flags destination key
                // identically. Forwarding the redundant `dest` is still sound —
                // the arithmetic result is the same function of the same
                // operands — but the instruction may only be removed once every
                // definition it makes is dead. Removing it on the strength of a
                // dead `dest` alone leaves a live `flags` use dangling.
                let all_defs_dead = editor
                    .function()
                    .block(*block_idx)
                    .and_then(|block| block.instruction(*instr_idx))
                    .is_some_and(|instr| instr.op().defs().all(|def| !used.contains(def)));
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
        // nothing changed and zero is the truth. Release runs under
        // `SsaRollbackPolicy::Never`, where the edits stay applied — reporting
        // zero there would make the pass-group transaction skip both
        // verification and rollback, keeping damaged IR and keeping it
        // unchecked. Report the change so the transaction handles it.
        return if matches!(rollback, SsaRollbackPolicy::OnFailure) {
            0
        } else {
            total_replaced.max(1)
        };
    }

    total_replaced
}

/// Collects, in a single pass, the set of variables still referenced by any
/// instruction operand or phi operand (indexed by `SsaVarId::index()`).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::EventLog,
        ir::{
            FlagCondition,
            block::SsaBlock,
            instruction::SsaInstruction,
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{
            MockTarget, MockType, assert_mock_valid_full, mock_op_at,
            run_mock_pass_repaired_boundary,
        },
    };

    /// Two identical flag-setting adds followed by a `BranchFlags` reading the
    /// *second* one's flags — the shape any lifter emits for a repeated
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
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(sum1),
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(sum2),
        }));
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

        run_gvn(&mut ssa, &0u32, &events);

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
             read by BranchFlags — the branch now reads an undefined variable"
        );
        assert_mock_valid_full(&ssa, "after GVN over flag-setting adds");
    }

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

        // `Rcl` is impure (hidden carry dependence) -> never value-numbered.
        let rcl: SsaOp<MockTarget> = SsaOp::Rcl {
            dest: v2,
            value: v0,
            amount: v1,
        };
        assert!(ValueKey::from_op(&rcl).is_none());
    }

    /// `ComputeFlags` carries no opcode discriminator — it models the flags of
    /// `bsf`/`bsr`/`popcnt`/`bt` alike — so its result is *not* a function of
    /// its SSA inputs alone. Value-numbering it aliases the flags of two
    /// different native instructions that happen to share operands.
    #[test]
    fn value_key_does_not_number_compute_flags() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        // `bsf v0, v1` and `bt v0, v1` are indistinguishable at this level.
        let bsf_flags: SsaOp<MockTarget> = SsaOp::ComputeFlags {
            dest: v2,
            inputs: vec![v0, v1],
        };

        assert!(
            ValueKey::from_op(&bsf_flags).is_none(),
            "ComputeFlags must never be value-numbered: it has no opcode \
             discriminator, so distinct native flag computations collide"
        );
    }

    /// `CallClobber` defines *fresh, undefined* values for caller-saved
    /// registers. Two clobbers are never the same value, but the generic key
    /// normalizes every `Def` operand to a sentinel — so all single-output
    /// clobbers in a function would key identically and GVN would forward one
    /// call's clobbered register to another's.
    #[test]
    fn value_key_does_not_number_call_clobber() {
        let v0 = SsaVarId::from_index(0);

        let clobber: SsaOp<MockTarget> = SsaOp::CallClobber { outputs: vec![v0] };

        assert!(
            ValueKey::from_op(&clobber).is_none(),
            "CallClobber must never be value-numbered: it defines fresh \
             undefined values, not a function of its inputs"
        );
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
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;
        let replaced = run_gvn(&mut ssa, &method, &log);
        assert!(replaced > 0);
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

    #[test]
    fn same_expression_across_blocks_eliminated() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        for i in 0..5 {
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
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v4) }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "cross-block GVN", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(
            changed,
            "duplicate expression across blocks should be eliminated"
        );
    }

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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        // v2 and v3 are different computations — GVN should NOT eliminate either
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "non-commutative GVN", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(!changed, "swapped non-commutative ops must not be merged");
        // v3's Sub should still be present (not Nop'd)
        assert!(
            matches!(mock_op_at(&ssa, 0, 3), SsaOp::Sub { .. }),
            "non-commutative swapped sub should NOT be nop'd"
        );
    }

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
        // Two AddOvf ops with same operands — should NOT be eliminated (may throw)
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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_repaired_boundary(&mut ssa, "impure GVN", |ssa| run(ssa, &method, &log));
        assert!(!changed, "throwing overflow ops must not be value numbered");
        // Neither AddOvf should be nop'd
        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::AddOvf { .. }));
        assert!(matches!(mock_op_at(&ssa, 0, 3), SsaOp::AddOvf { .. }));
    }

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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "no-duplicate GVN", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(!changed, "no duplicates should return false");
    }

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

        let ret_op: SsaOp<MockTarget> = SsaOp::Return { value: None };
        assert!(
            ValueKey::from_op(&ret_op).is_none(),
            "Return should not generate a key"
        );
    }

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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_repaired_boundary(&mut ssa, "loadarg GVN", |ssa| run(ssa, &method, &log));
        assert!(changed, "identical LoadArgs should be value numbered");
    }

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
        true_block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(original),
        }));

        let mut false_block = SsaBlock::new(2);
        false_block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: duplicate,
            left,
            right,
            flags: None,
        }));
        false_block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(duplicate),
        }));

        ssa.add_block(entry);
        ssa.add_block(true_block);
        ssa.add_block(false_block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "non-dominating GVN", |ssa| {
            run(ssa, &method, &log)
        });

        assert!(
            !changed,
            "GVN must not forward from a sibling branch that does not dominate the use"
        );
        assert!(matches!(mock_op_at(&ssa, 2, 0), SsaOp::Add { .. }));
    }

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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(result2),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_repaired_boundary(&mut ssa, "multi-level GVN", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(changed, "multi-level duplicates should be eliminated");
    }
}
