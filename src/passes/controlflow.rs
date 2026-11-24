//! Control flow simplification pass - reduces CFG complexity by removing
//! redundant branches, threading through trampolines, and cleaning up dead
//! code after terminators.
//!
//! # Algorithm
//!
//! Each iteration performs three steps:
//!
//! 1. **Jump threading** - resolves block references through trampoline
//!    chains using [`SsaFunction::find_trampoline_blocks`] and
//!    [`SsaOp::redirect_target`]. Skips the entry block (handled by
//!    block merge).
//! 2. **Branch-to-same-target simplification** - detects two-way branches
//!    (`Branch`, `BranchCmp`, `BranchFlags`) and `Switch` terminators where
//!    all targets (after trampoline resolution)
//!    point to the same block, reducing them to a `Jump`. Also handles
//!    CFF-style self-loop switches that degenerate to a single non-self
//!    target.
//! 3. **Dead tail removal** - removes instructions that follow a terminator
//!    in the same block (shared with [`crate::passes::deadcode`]).
//! 4. **Truth tests** - a branch comparing a truth value extended to an
//!    integer with zero (MIPS `slt t0, a, b; bnez t0`, `(uint32_t)(a < b) !=
//!    0`) branches on the truth value itself, its targets exchanged for `== 0`.
//!
//! The outer loop iterates until convergence (no changes) or
//! `max_iterations` is reached.

use std::{collections::BTreeMap, iter};

use crate::{
    Result,
    analysis::budget::Budget,
    events::EventKind,
    ir::{
        function::{SsaEditOptions, SsaEditor, SsaFunction},
        ops::{CmpKind, SsaOp},
        variable::SsaVarId,
    },
    passes::{
        deadcode::find_dead_tails,
        redirect::{chain_end, redirect_past_trampolines},
        utils::{loop_canonical_blocks, resolve_chain},
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// The branches comparing a truth value extended to an integer with zero,
/// each with the branch on the truth value it is: `!= 0` taken where the
/// value holds, `== 0` where it does not.
///
/// A truth value is the result of a comparison or of a boolean operation, or
/// a value of the target's comparison result type; the extension is an
/// `IntConv` of it, which holds 0 or 1 (or 0 or -1 extended by its sign),
/// nonzero exactly where the value holds.
fn find_truth_tests<T: Target>(ssa: &SsaFunction<T>) -> Vec<(usize, SsaOp<T>)> {
    let truth = |var: SsaVarId| {
        matches!(
            ssa.get_definition(var),
            Some(
                SsaOp::Ceq { .. }
                    | SsaOp::Clt { .. }
                    | SsaOp::Cgt { .. }
                    | SsaOp::BoolAnd { .. }
                    | SsaOp::BoolOr { .. }
                    | SsaOp::BoolXor { .. }
                    | SsaOp::BoolNot { .. }
            )
        ) || T::comparison_result_type().is_some_and(|truth| {
            ssa.variable(var)
                .is_some_and(|variable| *variable.var_type() == truth)
        })
    };
    let zero = |var: SsaVarId| matches!(ssa.get_definition(var), Some(SsaOp::Const { value, .. }) if value.is_zero());
    let extended = |var: SsaVarId| match ssa.get_definition(var) {
        Some(SsaOp::IntConv { operand, .. }) if truth(*operand) => Some(*operand),
        _ => None,
    };
    let mut found = Vec::new();
    for (block_idx, block) in ssa.iter_blocks() {
        let Some(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            true_target,
            false_target,
            ..
        }) = block.control_terminator()
        else {
            continue;
        };
        let condition = match (extended(*left), extended(*right)) {
            (Some(condition), _) if zero(*right) => condition,
            (_, Some(condition)) if zero(*left) => condition,
            _ => continue,
        };
        let (true_target, false_target) = match cmp {
            CmpKind::Ne => (*true_target, *false_target),
            CmpKind::Eq => (*false_target, *true_target),
            _ => continue,
        };
        found.push((
            block_idx,
            SsaOp::Branch {
                condition,
                true_target,
                false_target,
            },
        ));
    }
    found
}

fn find_same_target_branches<T: Target>(
    ssa: &SsaFunction<T>,
    trampolines: &BTreeMap<usize, usize>,
) -> Vec<(usize, usize)> {
    ssa.iter_blocks()
        .filter_map(|(block_idx, block)| {
            block
                .control_terminator()
                .and_then(|op| match (op, op.two_way_targets()) {
                    // **All three two-way terminators, through the type.**
                    // `two_way_targets` answers for `Branch`, `BranchCmp` and
                    // `BranchFlags` alike, so none of them can be left out of this
                    // match: a two-way branch whose arms coincide that survived as
                    // a branch would print as an `if` with an empty body.
                    //
                    // Arms that already name one block come back as `Some((n, n))`,
                    // which this case wants as a *match*, so that is checked first.
                    (_, Some((true_target, false_target))) => {
                        if true_target == false_target {
                            return Some((block_idx, true_target));
                        }
                        let true_ultimate = resolve_chain(trampolines, true_target);
                        let false_ultimate = resolve_chain(trampolines, false_target);
                        if true_ultimate == false_ultimate {
                            Some((block_idx, true_ultimate))
                        } else {
                            None
                        }
                    }
                    (
                        SsaOp::Switch {
                            targets, default, ..
                        },
                        None,
                    ) => {
                        if targets.iter().all(|t| *t == *default) {
                            return Some((block_idx, *default));
                        }
                        let default_ultimate = resolve_chain(trampolines, *default);
                        if targets
                            .iter()
                            .all(|t| resolve_chain(trampolines, *t) == default_ultimate)
                        {
                            return Some((block_idx, default_ultimate));
                        }
                        // Self-loop elimination: residual CFF in exception
                        // handlers can leave a switch where all cases except one
                        // are self-loops. Degenerates to a jump to the single
                        // non-self target.
                        let non_self: Vec<usize> = targets
                            .iter()
                            .chain(iter::once(default))
                            .copied()
                            .filter(|&t| t != block_idx)
                            .collect();
                        if let Some(&first) = non_self.first() {
                            if non_self.iter().all(|t| *t == first) {
                                Some((block_idx, first))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
        })
        .collect()
}

/// Threads every edge into a trampoline chain to where the chain ends,
/// through the shared [`redirect_past_trampolines`], which keeps the
/// destination's phis consistent and leaves an edge whose redirect would not.
fn apply_jump_threading<T>(
    editor: &mut SsaEditor<T>,
    trampolines: &BTreeMap<usize, usize>,
    env: &PassEnv<'_, T>,
) -> Result<usize>
where
    T: Target,
{
    let redirected = redirect_past_trampolines(editor, trampolines, &|_| false)?;
    for (block_idx, old_targets, new_targets) in &redirected {
        env.record(EventKind::ControlFlowRestructured)
            .location(*block_idx)
            .message(format!("jump threaded: {old_targets:?} -> {new_targets:?}"));
    }
    Ok(redirected.len())
}

/// What `target`'s phis receive along every arm of `block_idx`'s terminator,
/// if every arm ends at `target` and they all bring the same values.
///
/// An arm reaches `target` directly or through a chain of trampolines; what it
/// brings is the phi's operand for `block_idx` or for the chain's last block.
/// Arms that loop back to `block_idx` itself are allowed when `allow_self` is
/// set; they bring nothing to `target`.
///
/// # Returns
///
/// The agreed value per phi of `target`, or `None` when an arm ends elsewhere,
/// a phi has no operand for an arm, or two arms bring different values - in
/// which case the arms are not interchangeable and the branch must stay.
fn agreed_arrivals<T: Target>(
    ssa: &SsaFunction<T>,
    block_idx: usize,
    target: usize,
    trampolines: &BTreeMap<usize, usize>,
    allow_self: bool,
) -> Option<Vec<SsaVarId>> {
    let arms = ssa.block(block_idx)?.control_terminator()?.successors();
    let mut arrivals: Vec<usize> = Vec::new();
    for arm in arms {
        if allow_self && arm == block_idx {
            continue;
        }
        let from = if arm == target {
            block_idx
        } else {
            match chain_end(ssa, trampolines, arm) {
                Some((last, ultimate)) if ultimate == target => last,
                _ => return None,
            }
        };
        if !arrivals.contains(&from) {
            arrivals.push(from);
        }
    }
    let phis = ssa.block(target)?.phi_nodes();
    let mut agreed = Vec::with_capacity(phis.len());
    for phi in phis {
        let mut value = None;
        for &from in &arrivals {
            let arriving = phi
                .normal_operands()
                .find(|operand| operand.predecessor() == from)?
                .value();
            match value {
                None => value = Some(arriving),
                Some(seen) if seen == arriving => {}
                Some(_) => return None,
            }
        }
        agreed.push(value?);
    }
    Some(agreed)
}

/// Turns each branch whose arms all end at one block into a jump to it.
///
/// Checked again here, on the function as the edits so far left it: jump
/// threading may have left one arm on its chain. The arms must all end at the
/// target and bring its phis the same values ([`agreed_arrivals`]); arms that
/// bring different values are not interchangeable, and folding the branch
/// would keep one and silently drop the others. The fold prunes the operands of
/// the edges it removes, and the jump's edge carries the agreed values.
fn simplify_same_target_branches<T>(
    editor: &mut SsaEditor<T>,
    same_target_branches: &[(usize, usize)],
    trampolines: &BTreeMap<usize, usize>,
    env: &PassEnv<'_, T>,
) -> Result<usize>
where
    T: Target,
{
    let mut simplified_count: usize = 0;

    for &(block_idx, target) in same_target_branches {
        // The arms reach `target` through forwarding blocks, and threading,
        // earlier in this session, moved them onto it where it could. Where
        // it declined - an edge that would leave a protected region it stayed
        // in, a phi the redirect would give two values on one edge - the
        // block does not name `target`, and a fold to it would add the edge
        // threading refused. Only a fold that drops edges is one.
        let names_target = editor
            .function()
            .block(block_idx)
            .is_some_and(|block| block.successors().contains(&target));
        if !names_target {
            continue;
        }
        // Residual flattening can leave a switch whose every other case loops
        // straight back to the block; those arms bring nothing to `target`.
        let allow_self = target != block_idx;
        let Some(agreed) = agreed_arrivals(
            editor.function(),
            block_idx,
            target,
            trampolines,
            allow_self,
        ) else {
            continue;
        };
        editor.fold_terminator_pruning_phis(block_idx, SsaOp::Jump { target })?;
        if let Some(block) = editor.function().block(target) {
            let missing: Vec<usize> = block
                .phi_nodes()
                .iter()
                .enumerate()
                .filter(|(_, phi)| {
                    !phi.normal_operands()
                        .any(|operand| operand.predecessor() == block_idx)
                })
                .map(|(index, _)| index)
                .collect();
            for index in missing {
                if let Some(&value) = agreed.get(index) {
                    editor.add_phi_operand(target, index, value, block_idx)?;
                }
            }
        }
        env.record(EventKind::BranchSimplified)
            .location(block_idx)
            .message(format!(
                "branch to same target simplified: B{block_idx} branch -> jump B{target}"
            ));
        simplified_count = simplified_count.saturating_add(1);
    }

    Ok(simplified_count)
}

fn remove_dead_tails<T>(
    editor: &mut SsaEditor<T>,
    dead_tails: &[(usize, usize)],
    env: &PassEnv<'_, T>,
) -> usize
where
    T: Target,
{
    let mut removed_count: usize = 0;

    for &(block_idx, start_idx) in dead_tails {
        let to_remove = editor
            .remove_instruction_tail(block_idx, start_idx)
            .unwrap_or(0);
        removed_count = removed_count.saturating_add(to_remove);
        if to_remove > 0 {
            env.record(EventKind::InstructionRemoved)
                .location(block_idx)
                .message(format!(
                    "removed {to_remove} dead instructions after terminator in B{block_idx}"
                ));
        }
    }

    removed_count
}

/// Pass that simplifies conditional branches and control flow.
///
/// Performs jump
/// threading through trampolines, branch-to-same-target simplification,
/// and dead tail removal.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - can change branch targets and
/// remove instructions.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlFlowSimplificationPass;

impl ControlFlowSimplificationPass {
    /// Single iteration of control flow simplification.
    ///
    /// Returns the number of changes made (branches simplified, instructions
    /// removed); zero means the algorithm has converged for this iteration.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to simplify.
    /// * `env` - The pass environment: where events are recorded, naming the method.
    ///
    /// # Returns
    ///
    /// Count of changes made this iteration.
    pub fn run_iteration<T: Target>(
        &self,
        ssa: &mut SsaFunction<T>,
        env: &PassEnv<'_, T>,
    ) -> usize {
        let mut total_changes: usize = 0;

        // Step 1: find jump threading targets (don't skip entry block).
        let mut trampolines = ssa.find_trampoline_blocks(false);

        // Preserve canonical loop preheaders: threading through them re-exposes a
        // loop header's multiple non-loop predecessors, which the loop canonicalizer
        // immediately re-funnels through a fresh preheader - an endless oscillation.
        //
        // Guarded, because `loop_canonical_blocks` builds the entire loop forest and
        // this runs on every fixpoint iteration, including the terminating one.
        if !trampolines.is_empty() {
            let preheaders = loop_canonical_blocks(ssa);
            trampolines.retain(|block, _| !preheaders.contains(block));
        }

        // Step 2: find branches to same target (also resolves through trampolines).
        let same_target_branches = find_same_target_branches(ssa, &trampolines);

        // Step 3: find dead tails.
        let dead_tails = find_dead_tails(ssa);

        let result = ssa.edit(SsaEditOptions::new(), |editor| {
            if !trampolines.is_empty() {
                total_changes =
                    total_changes.saturating_add(apply_jump_threading(editor, &trampolines, env)?);
            }

            if !same_target_branches.is_empty() {
                total_changes = total_changes.saturating_add(simplify_same_target_branches(
                    editor,
                    &same_target_branches,
                    &trampolines,
                    env,
                )?);
            }

            if !dead_tails.is_empty() {
                total_changes =
                    total_changes.saturating_add(remove_dead_tails(editor, &dead_tails, env));
            }

            Ok(())
        });

        if result.is_err() {
            // The session runs under [`SsaRollbackPolicy::Never`], so a failure in
            // boundary repair or rebuild leaves the edits already applied - the
            // function is mutated, and possibly mid-repair.
            //
            // Reporting zero here would say the opposite. `run` would then report
            // `changed == false` to its caller, and a pass-group transaction treats
            // "nothing changed" as "nothing to check": its `Unchanged` arm returns
            // **without verifying and without rolling back**, on the reasoning that a
            // pass which changed nothing cannot have broken anything. That reasoning
            // holds only while this return value is truthful, so a zero here keeps
            // damaged IR and hides it from the one check that would have caught it.
            //
            // Report the edits that were applied instead. `total_changes` is exact -
            // the closure itself is infallible, so any error comes from the repair
            // that runs *after* the counted mutations - and the floor of one covers
            // a mutation no counter observed. Non-zero is what makes the transaction
            // verify this function and roll it back.
            return total_changes.max(1);
        }

        // Step 4, over the graph the steps above left: the branches that test a
        // truth value through its integer.
        let truth_tests = find_truth_tests(ssa);
        if !truth_tests.is_empty() {
            let mut rewritten = 0usize;
            let result = ssa.edit(SsaEditOptions::new(), |editor| {
                for (block_idx, branch) in &truth_tests {
                    editor.replace_terminator_op(*block_idx, branch.clone())?;
                    rewritten = rewritten.saturating_add(1);
                }
                Ok(())
            });
            total_changes = total_changes.saturating_add(if result.is_err() {
                rewritten.max(1)
            } else {
                rewritten
            });
        }

        total_changes
    }
}

impl<T> FunctionPass<T> for ControlFlowSimplificationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "control-flow-simplification"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Simplify branches, eliminate unreachable code, fold constant predicates"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run control flow simplification on `ssa`.
    ///
    /// Iterates a three-step simplification until convergence or
    /// `max_iterations` is reached.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any transformation fired.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let mut changed = false;
        let mut rounds = env.context().budget().rounds(Budget::ControlFlowRounds);
        while rounds.next_round() {
            if self.run_iteration(ssa, env) == 0 {
                rounds.settle();
            } else {
                changed = true;
            }
        }
        if let Some(exhaustion) = rounds.exhausted() {
            env.exhausted(exhaustion);
        }
        Ok(changed)
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
            instruction::SsaInstruction,
            ops::{CmpKind, native::FlagCondition},
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{
            MACHINE, MockTarget, MockType, mock_terminator_at, run_mock_malformed_cleanup_boundary,
            run_mock_pass_boundary,
        },
    };

    /// An iterator's `Dispose` (mono mscorlib's `System.IDisposable.Dispose`
    /// at `0xbbe4c`): `B4` tests the state and both arms leave the outer `try`
    /// (`B3..B9`) through forwarding blocks `B5` and `B7` to `B10`, past the
    /// `finally` in `B9`; `B7` is the inner `try` (`B7..B8`), `B2` and `B6`
    /// are empty. With `regions`, the two clauses.
    fn dispose(regions: bool) -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let state = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let constant = |ssa: &mut SsaFunction<MockTarget>, block: usize, index: usize| {
            ssa.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(block, index),
                MockType::I32,
            )
        };
        let branch = |left, right, cmp, true_target, false_target| SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target,
            false_target,
        };
        let (zero, one, four) = (
            constant(&mut ssa, 0, 0),
            constant(&mut ssa, 3, 0),
            constant(&mut ssa, 4, 0),
        );
        let mut blocks: Vec<Vec<SsaOp<MockTarget>>> = vec![Vec::new(); 11];
        blocks[0] = vec![
            SsaOp::Const {
                dest: zero,
                value: ConstValue::I32(0),
            },
            branch(state, zero, CmpKind::Eq, 3, 1),
        ];
        blocks[1] = vec![SsaOp::returns(None)];
        blocks[3] = vec![
            SsaOp::Const {
                dest: one,
                value: ConstValue::I32(-4),
            },
            branch(state, one, CmpKind::Eq, 7, 4),
        ];
        blocks[4] = vec![
            SsaOp::Const {
                dest: four,
                value: ConstValue::I32(1),
            },
            branch(state, four, CmpKind::Eq, 7, 5),
        ];
        blocks[5] = vec![SsaOp::Jump { target: 10 }];
        blocks[7] = vec![SsaOp::Jump { target: 10 }];
        blocks[8] = vec![SsaOp::EndFinally];
        blocks[9] = vec![SsaOp::EndFinally];
        blocks[10] = vec![SsaOp::returns(None)];
        for (index, ops) in blocks.into_iter().enumerate() {
            let mut block = SsaBlock::new(index);
            for op in ops {
                block.add_instruction(instr(op));
            }
            ssa.add_block(block);
        }
        if regions {
            let finally =
                |protected: (usize, usize), handler: (usize, usize)| SsaExceptionHandler {
                    flags: 2,
                    try_offset: 0,
                    try_length: 0,
                    handler_offset: 0,
                    handler_length: 0,
                    class_token_or_filter: 0,
                    protected_range: BlockRange::new(protected.0, protected.1),
                    handler_range: BlockRange::new(handler.0, handler.1),
                    filter_range: None,
                    selector: None,
                };
            ssa.set_exception_handlers(vec![finally((7, 8), (8, 9)), finally((3, 9), (9, 10))]);
        }
        ssa.recompute_uses();
        ssa
    }

    /// **A branch whose arms leave a `try` through forwarding blocks is left
    /// as it is, and the pass says so.** Threading declines to move `B4`'s
    /// arms past `B5` and `B7` onto `B10`, which would leave the `try` from
    /// its middle; folding the branch to `jump B10` would add that edge, the
    /// fold refused it, and the pass reported a change it never made on every
    /// run - the normalization ran out its iterations on mscorlib's iterator
    /// `Dispose` and left it un-normalized. The control: without the regions
    /// the arms thread onto `B10` and the branch folds.
    #[test]
    fn a_branch_leaving_a_try_through_forwarders_is_kept() {
        let log: EventLog<MockTarget> = EventLog::new();
        let mut ssa = dispose(true);
        assert_eq!(
            ControlFlowSimplificationPass.run_iteration(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ),
            0,
            "{ssa}"
        );
        assert!(
            ssa.block(4)
                .and_then(SsaBlock::control_terminator)
                .is_some_and(|op| op.two_way_targets() == Some((7, 5))),
            "{ssa}"
        );

        let mut ssa = dispose(false);
        assert!(
            ControlFlowSimplificationPass.run_iteration(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ) > 0
        );
        assert!(
            ssa.block(4)
                .and_then(SsaBlock::control_terminator)
                .is_some_and(|op| op.successors() == [10]),
            "control: {ssa}"
        );
    }

    /// `B0` branches on argument 0 through trampoline `B1` or `B2` to `B3`,
    /// whose phi receives `left` (1) from `B1`, and from `B2` either `right`
    /// (2) or, when `same`, `left` again.
    fn arms_through_trampolines(same: bool) -> SsaFunction<MockTarget> {
        // Temporaries, which no rename group joins: the rebuild between
        // iterations keeps the phi's operands as they are.
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let left = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let right = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: left,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: right,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        for block in [1, 2] {
            let mut b = SsaBlock::new(block);
            b.add_instruction(instr(SsaOp::Jump { target: 3 }));
            ssa.add_block(b);
        }
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(left, 1));
        phi.add_operand(PhiOperand::new(if same { left } else { right }, 2));
        b3.add_phi(phi);
        b3.add_instruction(instr(SsaOp::returns(Some(merged))));
        ssa.add_block(b3);
        ssa.recompute_uses();
        ssa
    }

    /// Arms that reach one block with different values are not the same arm.
    ///
    /// Both arms of `B0` end at `B3`, but through trampolines that bring 1 and
    /// 2. Folding the branch into a jump kept one of them and dropped the
    /// other; the branch has to stay, and so does the value it chooses.
    #[test]
    fn arms_bringing_different_values_stay_a_branch() {
        let mut ssa = arms_through_trampolines(false);
        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "different arrivals", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(ssa.take_edit_failure().is_none());
        assert!(
            ssa.block(0)
                .and_then(SsaBlock::control_terminator)
                .is_some_and(|op| op.two_way_targets().is_some()),
            "the branch still chooses the value:\n{ssa:?}"
        );
        let phi = ssa
            .block(3)
            .and_then(|b| b.phi_nodes().first())
            .expect("the merge keeps its phi");
        let values: Vec<_> = phi
            .normal_operands()
            .filter_map(|operand| ssa.get_definition(operand.value()).cloned())
            .collect();
        assert!(
            values.iter().any(|op| matches!(
                op,
                SsaOp::Const {
                    value: ConstValue::I32(1),
                    ..
                }
            )) && values.iter().any(|op| matches!(
                op,
                SsaOp::Const {
                    value: ConstValue::I32(2),
                    ..
                }
            )),
            "both values still reach the merge:\n{ssa:?}"
        );
    }

    /// The control: arms that bring the same value fold into one jump, and the
    /// jump carries it.
    #[test]
    fn arms_bringing_the_same_value_fold_to_a_jump() {
        let mut ssa = arms_through_trampolines(true);
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_mock_pass_boundary(&mut ssa, "same arrivals", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        }));
        assert!(ssa.take_edit_failure().is_none());
        assert!(
            matches!(
                ssa.block(0).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { target: 3 })
            ),
            "{ssa:?}"
        );
    }

    /// A chain whose first link was itself threaded earlier in the same pass
    /// is followed as it now stands.
    ///
    /// `B5 -> B1 -> B2 -> B3`, with `B1` and `B2` forwarding. `B1`'s own jump is
    /// threaded to `B3` first; `B5`'s edge into `B1` then ends at `B3` through
    /// `B1`, not through `B2` as the chain read before the pass says. Reading it
    /// from before failed the bypass half way, and left it applied.
    #[test]
    fn a_chain_is_followed_as_the_pass_left_it() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let one = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let two = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(4, 0),
            MockType::I32,
        );
        let three = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(5, 0),
            MockType::I32,
        );
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: 5,
            false_target: 4,
        }));
        ssa.add_block(b0);
        for (block, target) in [(1, 2), (2, 3)] {
            let mut b = SsaBlock::new(block);
            b.add_instruction(instr(SsaOp::Jump { target }));
            ssa.add_block(b);
        }
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(one, 2));
        phi.add_operand(PhiOperand::new(two, 4));
        b3.add_phi(phi);
        b3.add_instruction(instr(SsaOp::returns(Some(merged))));
        ssa.add_block(b3);
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Const {
            dest: two,
            value: ConstValue::I32(2),
        }));
        b4.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b4);
        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::Const {
            dest: three,
            value: ConstValue::I32(3),
        }));
        b5.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b5);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_mock_pass_boundary(&mut ssa, "threaded chain", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        }));
        assert!(ssa.take_edit_failure().is_none());
        assert!(
            matches!(
                ssa.block(5).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { target: 3 })
            ),
            "B5 goes straight to the merge:\n{ssa:?}"
        );
        let phi = ssa
            .block(3)
            .and_then(|b| b.phi_nodes().first())
            .expect("the merge keeps its phi");
        let from_b5 = phi.operand_from(5).map(|operand| operand.value());
        assert!(
            from_b5.is_some_and(|value| matches!(
                ssa.get_definition(value),
                Some(SsaOp::Const {
                    value: ConstValue::I32(1),
                    ..
                })
            )),
            "and brings what the chain brought:\n{ssa:?}"
        );
    }

    /// A forwarding block that starts a `try` is the region's entry: an edge
    /// from outside threaded past it would enter the protected region in its
    /// middle.
    #[test]
    fn a_try_entry_is_not_threaded_past() {
        const PROTECTED_END: usize = 3;
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: PROTECTED_END + 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Leave {
            target: PROTECTED_END + 1,
        }));
        ssa.add_block(b2);
        let mut handler = SsaBlock::new(PROTECTED_END);
        handler.add_instruction(instr(SsaOp::EndFinally));
        ssa.add_block(handler);
        let mut exit = SsaBlock::new(PROTECTED_END + 1);
        exit.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(exit);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, PROTECTED_END),
            handler_range: BlockRange::new(PROTECTED_END, PROTECTED_END + 1),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "try entry", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let entry = ssa.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(entry, Some(SsaOp::Branch { true_target: 1, .. })),
            "the edge still enters the try at its start, got {entry:?}"
        );
    }

    /// A `leave` that runs a `finally` is not threaded past a trampoline: its
    /// target is entered from the `endfinally`, so a phi operand keyed by the
    /// `leave` would name a block that is not the target's predecessor.
    ///
    /// `B0` branches into the try `B1` or to `B4`. `B1` leaves to trampoline
    /// `B3` through the finally `B2`; `B3` and `B4` jump to `B5`, whose phi
    /// takes `a` from `B3` and `b` from `B4`. Threaded, the phi named `B1` and
    /// lost `B2`, and the method's SSA no longer verified.
    #[test]
    fn a_leave_through_a_finally_is_not_threaded() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 2);
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(5), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: b,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 4,
        }));
        ssa.add_block(b0);
        for (index, op) in [
            (1, SsaOp::Leave { target: 3 }),
            (2, SsaOp::EndFinally),
            (3, SsaOp::Jump { target: 5 }),
            (4, SsaOp::Jump { target: 5 }),
        ] {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(op));
            ssa.add_block(block);
        }
        let mut exit = SsaBlock::new(5);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(a, 3));
        phi.add_operand(PhiOperand::new(b, 4));
        exit.add_phi(phi);
        exit.add_instruction(instr(SsaOp::returns(Some(merged))));
        ssa.add_block(exit);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, 2),
            handler_range: BlockRange::new(2, 3),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "leave through finally", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let leave = ssa.block(1).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(leave, Some(SsaOp::Leave { target: 3 })),
            "the leave still goes through its trampoline, got {leave:?}"
        );
    }

    /// **A branch on a truth value's integer branches on the truth value.**
    /// MIPS `slt t0, a, b; bnez t0` compares the 0 or 1 `slt` extends into a
    /// register with zero, printed `(uint32_t)(a < b) != 0`; it is a branch on
    /// `a < b`, and `beqz t0` one on its negation, the targets exchanged. A
    /// comparison of an integer that is no truth value with zero stays.
    #[test]
    fn a_branch_on_a_truth_values_integer_branches_on_the_truth_value() {
        let build = |cmp: CmpKind, extended: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 0);
            let a = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let b = ssa.create_variable(
                VariableOrigin::Argument(1),
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let less = ssa.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, 0),
                MockType::I1,
            );
            let widened = ssa.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, 1),
                MockType::I32,
            );
            let zero = ssa.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, 2),
                MockType::I32,
            );
            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(instr(SsaOp::Clt {
                dest: less,
                left: a,
                right: b,
                unsigned: false,
                signaling: false,
            }));
            b0.add_instruction(instr(if extended {
                SsaOp::IntConv {
                    dest: widened,
                    operand: less,
                    target: MockType::I32,
                    overflow_check: false,
                    unsigned: true,
                }
            } else {
                SsaOp::Add {
                    dest: widened,
                    left: a,
                    right: b,
                    flags: None,
                }
            }));
            b0.add_instruction(instr(SsaOp::Const {
                dest: zero,
                value: ConstValue::I32(0),
            }));
            b0.add_instruction(instr(SsaOp::BranchCmp {
                left: widened,
                right: zero,
                cmp,
                unsigned: false,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);
            for index in [1, 2] {
                let mut block = SsaBlock::new(index);
                block.add_instruction(instr(SsaOp::returns(None)));
                ssa.add_block(block);
            }
            ssa.recompute_uses();
            ControlFlowSimplificationPass.run_iteration(
                &mut ssa,
                &PassEnv::new(
                    &0u32,
                    &EventLog::<MockTarget>::new(),
                    AnalysisContext::new(MACHINE),
                ),
            );
            let terminator = ssa
                .block(0)
                .and_then(|block| block.control_terminator())
                .cloned();
            (terminator, less)
        };
        let (taken, less) = build(CmpKind::Ne, true);
        assert_eq!(
            taken,
            Some(SsaOp::Branch {
                condition: less,
                true_target: 1,
                false_target: 2
            })
        );
        let (negated, less) = build(CmpKind::Eq, true);
        assert_eq!(
            negated,
            Some(SsaOp::Branch {
                condition: less,
                true_target: 2,
                false_target: 1
            })
        );
        let (kept, _) = build(CmpKind::Ne, false);
        assert!(matches!(kept, Some(SsaOp::BranchCmp { .. })), "{kept:?}");
    }

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
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

    /// A flag-tested branch whose arms name one block collapses to a jump.
    ///
    /// `BranchFlags` is the third two-way spelling, beside `Branch` and
    /// `BranchCmp`. Leaving it uncollapsed would be silent, because a branch to
    /// one target is *correct*, merely pointless - a decompiler prints it as
    /// an `if` with an empty body. `SsaOp::two_way_targets` keeps the three
    /// kinds in step.
    #[test]
    fn a_flag_branch_to_one_target_collapses_to_a_jump() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let value = local_at(&mut ssa, 0, 0, 0);
        let flags = local_at(&mut ssa, 1, 0, 1);
        let masked = local_at(&mut ssa, 2, 0, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: value,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: masked,
            left: value,
            right: value,
            flags: Some(flags),
        }));
        b0.add_instruction(instr(SsaOp::BranchFlags {
            flags,
            condition: FlagCondition::Zero,
            true_target: 1,
            false_target: 1,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "flag branch to one target", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(
            matches!(mock_terminator_at(&ssa, 0), SsaOp::Jump { target: 1 }),
            "both arms name block 1, so the condition decides nothing, got {:?}",
            mock_terminator_at(&ssa, 0)
        );
    }

    #[test]
    fn dead_tail_after_return_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::returns(None)));
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(999),
        }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_malformed_cleanup_boundary(&mut ssa, "dead tail control flow", |ssa| {
                ControlFlowSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed, "dead tail after return should be removed");
        assert!(log.has(EventKind::InstructionRemoved));
    }

    #[test]
    fn branch_to_same_target_simplified() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "same-target branch simplification", |ssa| {
                ControlFlowSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed, "branch to same target should be simplified");
        assert!(log.has(EventKind::BranchSimplified));
        // Should now be a Jump
        assert!(matches!(
            mock_terminator_at(&ssa, 0),
            SsaOp::Jump { target: 1 }
        ));
    }

    #[test]
    fn jump_thread_through_trampoline() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "jump trampoline threading", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "jump through trampoline should be threaded");
    }

    #[test]
    fn no_changes_on_well_formed_cfg() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "well-formed control flow", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "well-formed CFG should have no changes");
    }

    #[test]
    fn multiple_changes_in_one_run() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 1,
        }));
        b0.add_instruction(instr(SsaOp::Nop)); // dead tail
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_malformed_cleanup_boundary(&mut ssa, "multi-change control flow", |ssa| {
                ControlFlowSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed);
    }

    #[test]
    fn dead_tails_in_multiple_blocks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        b0.add_instruction(instr(SsaOp::Nop));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        b1.add_instruction(instr(SsaOp::Nop));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_malformed_cleanup_boundary(&mut ssa, "multi-block dead tails", |ssa| {
                ControlFlowSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed);
        assert!(log.count_kind(EventKind::InstructionRemoved) >= 2);
    }

    #[test]
    fn switch_all_targets_same() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let val = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: val,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Switch {
            value: val,
            targets: vec![1, 1, 1],
            default: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "switch simplification", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "switch with all same targets should simplify");
        assert!(matches!(
            mock_terminator_at(&ssa, 0),
            SsaOp::Jump { target: 1 }
        ));
    }

    #[test]
    fn empty_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty control flow", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
    }

    #[test]
    fn trampoline_not_skipping_entry_block() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "entry trampoline control flow", |ssa| {
            ControlFlowSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        // B0 is entry trampoline - it has no predecessors so jump threading can't redirect.
        // The entry trampoline handling is done by block merge pass, not control flow.
        assert!(
            !changed,
            "control flow pass should leave entry trampoline handling to block merge"
        );
    }
}
