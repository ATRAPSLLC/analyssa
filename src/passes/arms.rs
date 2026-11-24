//! Merging the two arms of a branch that compute the same thing.
//!
//! A test whose arms do one thing tests nothing: `if (c) { x = f(a); }
//! else { x = f(a); }` is `x = f(a)`. Such arms are seldom written, and
//! often made: a loop a compiler split on a guard, rejoined
//! ([`crate::passes::rejoin`]), takes either of the split loops' bodies by
//! the guard, and once each body's read is stated as the one operation it
//! is, the two bodies are one.
//!
//! [`ArmMergingPass`] merges the arms where they are one by construction:
//!
//! - **Each arm is entered from the test alone**, and holds no phi.
//! - **The arms are one operation for one**, each operand of the second the
//!   first's operand where that is defined outside the arms, its counterpart
//!   where the first arm defines it, or an equal constant.
//! - **They leave alike**: by one jump to one block, or by one test sending
//!   each block what the other's sends it, its comparison the same or the
//!   negation with its targets swapped (`i == n` to the exit, `i != n` back).
//! - **Where they lead takes the same from each**, under the same
//!   correspondence, and nothing past them reads what the second arm
//!   defines.
//!
//! The test jumps to the first arm then, and the second, entered from
//! nowhere, goes. Each of its rows is one the first arm runs in its place,
//! which takes in its facts
//! ([`absorb_twin`](crate::facts::InstructionFacts::absorb_twin)):
//! a host's account of where the work came from loses neither arm.

use std::collections::BTreeMap;

use crate::{
    Result,
    analysis::cfg::SsaCfg,
    events::EventKind,
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        ops::{OperandRole, SsaOp},
        variable::SsaVarId,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// A branch whose two arms compute the same thing.
#[derive(Debug, Clone, Copy)]
struct Arms {
    /// The block ending in the branch.
    test: usize,
    /// The arm kept, which the test jumps to.
    kept: usize,
    /// The arm that goes.
    dropped: usize,
}

impl Arms {
    /// The first branch of `ssa` whose arms compute the same thing.
    fn find<T: Target>(ssa: &SsaFunction<T>) -> Option<Self> {
        let cfg = SsaCfg::from_ssa(ssa);
        (0..ssa.block_count()).find_map(|test| Self::at(ssa, &cfg, test))
    }

    /// The arms of the branch ending `test`, where they compute the same
    /// thing (the module's rules).
    fn at<T: Target>(ssa: &SsaFunction<T>, cfg: &SsaCfg<'_, T>, test: usize) -> Option<Self> {
        let (kept, dropped) = match ssa.block(test)?.control_terminator()? {
            SsaOp::Branch {
                true_target,
                false_target,
                ..
            }
            | SsaOp::BranchCmp {
                true_target,
                false_target,
                ..
            } if true_target != false_target => (*true_target, *false_target),
            _ => return None,
        };
        let (first, second) = (ssa.block(kept)?, ssa.block(dropped)?);
        let entered_from_test = |arm: usize| cfg.block_predecessors(arm) == [test];
        if !entered_from_test(kept)
            || !entered_from_test(dropped)
            || !first.phi_nodes().is_empty()
            || !second.phi_nodes().is_empty()
        {
            return None;
        }
        let (rows, others) = (first.instructions(), second.instructions());
        let ((last, body), (other_last, other_body)) = (rows.split_last()?, others.split_last()?);
        if body.len() != other_body.len() {
            return None;
        }
        // What each value the first arm defines is in the second.
        let mut defined: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
        let same = |defined: &BTreeMap<SsaVarId, SsaVarId>, mine: SsaVarId, theirs: SsaVarId| {
            defined.get(&mine).map_or_else(
                || mine == theirs || equal_constants(ssa, mine, theirs),
                |own| *own == theirs,
            )
        };
        for (mine, theirs) in body.iter().zip(other_body) {
            let alike = mine
                .op()
                .same_under(theirs.op(), |role, own, their| match role {
                    OperandRole::Def | OperandRole::FlagsDef => {
                        defined.insert(own, their);
                        true
                    }
                    OperandRole::Use => same(&defined, own, their),
                });
            if !alike {
                return None;
            }
        }
        if !leave_alike(last.op(), other_last.op(), |mine, theirs| {
            same(&defined, mine, theirs)
        }) {
            return None;
        }
        // Where they lead, a throw's handler too, takes the same from each.
        let leads = cfg.block_successors(kept);
        let mut theirs_lead = cfg.block_successors(dropped).to_vec();
        let mut ours_lead = leads.to_vec();
        theirs_lead.sort_unstable();
        ours_lead.sort_unstable();
        if ours_lead != theirs_lead {
            return None;
        }
        for &target in leads {
            for phi in ssa.block(target)?.phi_nodes() {
                let mine = phi.operand_from(kept)?.value();
                let theirs = phi.operand_from(dropped)?.value();
                if !same(&defined, mine, theirs) {
                    return None;
                }
            }
        }
        // Nothing past the second arm reads what it defines, but through the
        // phis just compared.
        for theirs in defined.values() {
            let outside = ssa.variable(*theirs)?.uses().iter().any(|site| {
                site.block() != dropped && !(site.is_phi_operand() && leads.contains(&site.block()))
            });
            if outside {
                return None;
            }
        }
        Some(Self {
            test,
            kept,
            dropped,
        })
    }
}

/// Whether `mine` and `theirs` are constants alike of one type.
fn equal_constants<T: Target>(ssa: &SsaFunction<T>, mine: SsaVarId, theirs: SsaVarId) -> bool {
    let constant = |value: SsaVarId| match ssa.get_definition(value) {
        Some(SsaOp::Const { value: held, .. }) => {
            Some((held.clone(), ssa.variable(value)?.var_type().clone()))
        }
        _ => None,
    };
    matches!((constant(mine), constant(theirs)), (Some(first), Some(second)) if first == second)
}

/// Whether the arms' last operations `mine` and `theirs` leave alike: one
/// jump to one block, or one test sending each block what the other sends
/// it, its comparison the same, its operands swapped, or the negation with
/// the targets swapped. `same` says whether two operands are one.
fn leave_alike<T: Target>(
    mine: &SsaOp<T>,
    theirs: &SsaOp<T>,
    same: impl Fn(SsaVarId, SsaVarId) -> bool,
) -> bool {
    match (mine, theirs) {
        (SsaOp::Jump { target }, SsaOp::Jump { target: other }) => target == other,
        (
            SsaOp::Branch {
                condition,
                true_target,
                false_target,
            },
            SsaOp::Branch {
                condition: other,
                true_target: other_true,
                false_target: other_false,
            },
        ) => same(*condition, *other) && true_target == other_true && false_target == other_false,
        (
            SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            },
            SsaOp::BranchCmp {
                left: other_left,
                right: other_right,
                cmp: other_cmp,
                unsigned: other_unsigned,
                true_target: other_true,
                false_target: other_false,
            },
        ) => {
            if unsigned != other_unsigned {
                return false;
            }
            // Their test, written as one with the same targets as ours.
            let cmp_for_ours = if (true_target, false_target) == (other_true, other_false) {
                *other_cmp
            } else if (true_target, false_target) == (other_false, other_true) {
                other_cmp.negated()
            } else {
                return false;
            };
            (*cmp == cmp_for_ours && same(*left, *other_left) && same(*right, *other_right))
                || (*cmp == cmp_for_ours.swapped()
                    && same(*left, *other_right)
                    && same(*right, *other_left))
        }
        _ => false,
    }
}

/// Pass that merges the two arms of a branch that compute the same thing.
///
/// A branch whose arms, each
/// entered from it alone, are one operation for one and leave alike jumps to
/// the first, and the second goes.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - turns the branch into a jump and
/// removes the second arm.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArmMergingPass;

impl ArmMergingPass {
    /// Creates a new arm-merging pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for ArmMergingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "arm-merging"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Merge the two arms of a branch that compute the same thing"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Merges every branch's two arms that compute the same thing into the
    /// first.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// Whether any branch's arms were merged.
    ///
    /// # Errors
    ///
    /// Returns an error if an edit session fails.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let mut merged = false;
        // Each merge leaves a block unreached, so there are at most as many as
        // blocks.
        for _ in 0..ssa.block_count() {
            let Some(arms) = Arms::find(ssa) else {
                break;
            };
            ssa.edit(
                SsaEditOptions::new()
                    .with_verify(true)
                    .with_rollback(SsaRollbackPolicy::OnFailure),
                |editor| {
                    editor.replace_terminator_op(arms.test, SsaOp::Jump { target: arms.kept })?;
                    // The kept arm's rows stand for the dropped arm's, one for one.
                    let gone: Vec<_> = editor
                        .function()
                        .block(arms.dropped)
                        .map(|block| {
                            block
                                .instructions()
                                .iter()
                                .map(|row| row.facts().clone())
                                .collect()
                        })
                        .unwrap_or_default();
                    for (row, facts) in gone.iter().enumerate() {
                        editor.absorb_twin_facts(arms.kept, row, facts)?;
                    }
                    editor.remove_unreached_block(arms.dropped)?;
                    editor.declare_repaired();
                    Ok(())
                },
            )?;
            env.record(EventKind::ControlFlowRestructured)
                .location(arms.test)
                .message(format!(
                    "merged arms: B{}'s arms B{} and B{} compute the same, B{} goes",
                    arms.test, arms.kept, arms.dropped, arms.dropped
                ));
            merged = true;
        }
        Ok(merged)
    }
}
