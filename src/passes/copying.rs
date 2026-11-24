//! Copy propagation pass - eliminates redundant copy operations and trivial
//! phi nodes.
//!
//! # Algorithm
//!
//! 1. **Collect copies**: gather all `Copy` instructions and trivial phi
//!    nodes via [`crate::analysis::PhiAnalyzer::collect_all_copies`].
//! 2. **Protect sole local defs**: exclude copies that are the only
//!    instruction-based definition of a local or argument variable whose
//!    source is in a different rename group. This prevents information loss
//!    when the variable's value cannot be recovered from a phi node or
//!    address-taken location.
//! 3. **Protect ordering**: exclude copies whose replacement source is
//!    defined later in the same block than an existing destination use.
//! 4. **Resolve chains**: replace `dest → src` pairs with
//!    `dest → ultimate_src` using [`resolve_chain`].
//! 5. **Propagate**: call
//!    [`SsaEditor::propagate_copies`](crate::ir::function::SsaEditor::propagate_copies)
//!    to rewrite all uses of dest variables to their ultimate sources.
//! 6. **Nop out and repair**: replace fully propagated copy-defining
//!    instructions with `Nop`, strip those nops, compact the variable table,
//!    and recompute use lists so the pass does not leave orphan variables.
//! 7. **Validate**: if the repaired function is not verifier-clean, restore
//!    the pre-iteration function and report no progress for that iteration.
//! 8. **Repeat** until no more changes (converges in 1-3 iterations).
//!
//! # Host hook
//!
//! Some hosts need to perform extra work between resolving the copy chains
//! and applying the propagation, such as recovering local-variable type
//! information. [`CopyPropagationPass::propagate`] takes a closure invoked
//! once per iteration after
//! `protect_sole_local_defs` and `resolve_chains`, before `propagate_copies`
//! rewrites uses; a host with nothing to do passes one that does nothing.
//!
//! # Complexity
//!
//! - Time: O(n * m) where n is the number of variables and m is iterations.
//! - Space: O(n) for the copy map.

use std::collections::BTreeMap;

use crate::{
    Result,
    analysis::{budget::Budget, phis::PhiAnalyzer},
    bitset::BitSet,
    events::EventKind,
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        variable::SsaVarId,
        varstore::VarSet,
    },
    passes::utils::resolve_chain,
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Remove copies from the candidate set when they would lose information.
///
/// A copy is protected (excluded from propagation) when all of the
/// following hold:
/// - The destination is in a local or argument rename group.
/// - The source is in a different rename group.
/// - The destination's group has only one instruction-based definition
///   and that group appears in phi operands, or the group is address-taken.
///
/// This prevents information loss when the variable's value cannot be
/// recovered from phi nodes or address-taken locations.
///
/// # Arguments
///
/// * `ssa` - The SSA function providing variable and rename group info.
/// * `copies` - Mutable map of `dest → src` copies to filter in place.
fn protect_sole_local_defs<T: Target>(
    ssa: &SsaFunction<T>,
    copies: &mut BTreeMap<SsaVarId, SsaVarId>,
) {
    let real_local_limit = (ssa.num_args() as u32).saturating_add(ssa.num_locals() as u32);

    let group_bound = ssa
        .num_locals()
        .saturating_add(ssa.num_args())
        .saturating_add(1);
    // Single traversal collecting all three facts: per-group definition counts,
    // groups referenced by phi nodes, and address-taken groups.
    let mut group_def_count: BTreeMap<u32, usize> = BTreeMap::new();
    let mut groups_in_phis = BitSet::new(group_bound);
    // One authority for which slots are memory, so arguments whose address is
    // taken are covered alongside locals.
    let address_taken = ssa.address_taken_slots();
    for block in ssa.blocks() {
        for instr in block.instructions() {
            let op = instr.op();
            for dest in op.defs() {
                let group = ssa.rename_group(dest);
                if group < real_local_limit {
                    let counter = group_def_count.entry(group).or_insert(0);
                    *counter = counter.saturating_add(1);
                }
            }
        }
        for phi in block.phi_nodes() {
            for operand in phi.operands() {
                let group = ssa.rename_group(operand.value());
                if group < real_local_limit {
                    groups_in_phis.insert(group as usize);
                }
            }
            let result_group = ssa.rename_group(phi.result());
            if result_group < real_local_limit {
                groups_in_phis.insert(result_group as usize);
            }
        }
    }

    let mut protected = VarSet::new(ssa.var_id_bound());
    for (&dest, &src) in copies.iter() {
        let dest_group = ssa.rename_group(dest);
        if dest_group >= real_local_limit {
            continue;
        }
        if ssa.rename_group(src) == dest_group {
            continue;
        }
        let def_count = group_def_count.get(&dest_group).copied().unwrap_or(0);
        if address_taken.group(dest_group, ssa.num_args())
            || (def_count <= 1 && groups_in_phis.contains(dest_group as usize))
        {
            protected.insert(dest);
        }
    }

    if !protected.is_empty() {
        copies.retain(|dest, _| !protected.contains(*dest));
    }
}

/// Builds a `def-variable -> (block, instruction)` index in a single pass.
///
/// The result depends only on `ssa`, so callers that filter multiple copy
/// maps for the same function can build it once and reuse it across
/// [`protect_same_block_ordering`] calls.
fn build_source_defs<T: Target>(ssa: &SsaFunction<T>) -> BTreeMap<SsaVarId, (usize, usize)> {
    let mut source_defs: BTreeMap<SsaVarId, (usize, usize)> = BTreeMap::new();
    for (block_idx, block) in ssa.blocks().iter().enumerate() {
        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            for dest in instr.op().defs() {
                source_defs.insert(dest, (block_idx, instr_idx));
            }
        }
    }
    source_defs
}

/// Earliest `(block, instruction)` at which each variable is *used*.
///
/// Built in one sweep so the ordering guard below is an O(1) comparison. Scanning
/// the source block's instructions per copy instead made the guard
/// O(block_length x copies), and copy propagation runs to a fixpoint.
fn build_first_uses<T: Target>(ssa: &SsaFunction<T>) -> BTreeMap<SsaVarId, (usize, usize)> {
    let mut first_uses: BTreeMap<SsaVarId, (usize, usize)> = BTreeMap::new();
    for (block_idx, block) in ssa.blocks().iter().enumerate() {
        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            instr.op().for_each_use(|used| {
                first_uses.entry(used).or_insert((block_idx, instr_idx));
            });
        }
    }
    first_uses
}

/// Removes copies that could create same-block use-before-def ordering.
///
/// A replacement `dest -> src` is unsafe when `src` is defined by an
/// instruction in block `B` and `dest` is used by an earlier instruction in
/// the same block. Rewriting that use would place `src` before its
/// definition, violating SSA instruction order.
///
/// # Arguments
///
/// * `first_uses` - Each variable's first use ([`build_first_uses`]).
/// * `source_defs` - Each variable's definition ([`build_source_defs`]).
/// * `copies` - Mutable map of copy replacements to filter in place.
fn protect_same_block_ordering(
    first_uses: &BTreeMap<SsaVarId, (usize, usize)>,
    source_defs: &BTreeMap<SsaVarId, (usize, usize)>,
    copies: &mut BTreeMap<SsaVarId, SsaVarId>,
) {
    copies.retain(|dest, src| {
        let Some((src_block, src_instr)) = source_defs.get(src).copied() else {
            return true;
        };
        // Reject when `dest` is already read earlier in the block that defines
        // `src`: substituting `src` there would move the use above its own
        // definition. One map lookup, rather than a scan of the block prefix per
        // copy.
        let Some(&(use_block, use_instr)) = first_uses.get(dest) else {
            return true;
        };
        !(use_block == src_block && use_instr < src_instr)
    });
}

/// Pass that eliminates redundant copy operations and trivial phi nodes.
///
/// [`FunctionPass::run`] is [`Self::propagate`] with no per-iteration hook.
/// Hosts that need a post-step (e.g. CIL local-type propagation) implement
/// their own pass that calls [`Self::propagate`] with one.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - replaces instructions,
/// never changes the CFG.
#[derive(Debug, Clone, Copy, Default)]
pub struct CopyPropagationPass;

impl CopyPropagationPass {
    /// Runs copy propagation on `ssa`, with a host hook: [`FunctionPass::run`]
    /// with `on_resolved` called each round, until a round changes nothing or the
    /// context's [`Budget::CopyRounds`] runs out.
    ///
    /// Invokes `on_resolved` once per iteration after the resolved-copies map
    /// is built and before `propagate_copies` rewrites uses. The host can use
    /// the hook to perform extra per-iteration work, such as recovering type
    /// information; `|_, _| {}` does none.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to transform in place.
    /// * `env` - The pass environment: where events are recorded, naming the method.
    /// * `on_resolved` - Closure invoked with the resolved copy map before
    ///   propagation. Receives `(&mut SsaFunction<T>, &BTreeMap<SsaVarId, SsaVarId>)`.
    ///
    /// # Returns
    ///
    /// `true` if any copy was propagated.
    pub fn propagate<T, F>(
        &self,
        ssa: &mut SsaFunction<T>,
        env: &PassEnv<'_, T>,
        mut on_resolved: F,
    ) -> bool
    where
        T: Target,
        F: FnMut(&mut SsaFunction<T>, &BTreeMap<SsaVarId, SsaVarId>),
    {
        let mut changed = false;
        let mut rounds = env.context().budget().rounds(Budget::CopyRounds);
        while rounds.next_round() {
            if self.run_iteration(ssa, env, &mut on_resolved) == 0 {
                rounds.settle();
            } else {
                changed = true;
            }
        }
        if let Some(exhaustion) = rounds.exhausted() {
            env.exhausted(exhaustion);
        }
        changed
    }

    /// Run a single iteration of copy propagation.
    ///
    /// Exposed for hosts that want to drive their own fixpoint loop.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to transform in place.
    /// * `env` - The pass environment: where events are recorded, naming the method.
    /// * `on_resolved` - Closure invoked with the resolved copy map before
    ///   propagation.
    ///
    /// # Returns
    ///
    /// The number of uses replaced this iteration.
    pub fn run_iteration<T, F>(
        &self,
        ssa: &mut SsaFunction<T>,
        env: &PassEnv<'_, T>,
        mut on_resolved: F,
    ) -> usize
    where
        T: Target,
        F: FnMut(&mut SsaFunction<T>, &BTreeMap<SsaVarId, SsaVarId>),
    {
        let mut copies = PhiAnalyzer::new(ssa).collect_all_copies();
        if copies.is_empty() {
            return 0;
        }

        protect_sole_local_defs(ssa, &mut copies);
        // `source_defs` depends only on `ssa`, so build it once and reuse it for
        // both ordering-protection passes below instead of rebuilding per call.
        let source_defs = build_source_defs(ssa);
        let first_uses = build_first_uses(ssa);
        protect_same_block_ordering(&first_uses, &source_defs, &mut copies);

        let resolved: BTreeMap<SsaVarId, SsaVarId> = copies
            .iter()
            .map(|(&dest, &src)| (dest, resolve_chain(&copies, src)))
            .collect();
        let mut resolved = resolved;
        protect_same_block_ordering(&first_uses, &source_defs, &mut resolved);

        on_resolved(ssa, &resolved);

        // A copy into a version of a slot the function reads as storage is a write
        // to the slot, whatever its SSA readers: propagated, it stays.
        let slots = ssa.storage_slots();
        let mut result = None;
        let edit_result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                let propagation = editor.propagate_copies(&resolved);
                for dest in propagation.fully_propagated.iter() {
                    let writes_slot = editor
                        .function()
                        .variable(dest)
                        .is_some_and(|variable| slots.contains(variable.origin()));
                    if !writes_slot {
                        editor.nop_copy_defining(dest);
                    }
                }
                result = Some(propagation);
                Ok(())
            },
        );
        if edit_result.is_err() {
            return 0;
        }
        let Some(result) = result else {
            return 0;
        };

        let event_pairs: Vec<(SsaVarId, SsaVarId)> = result
            .fully_propagated
            .iter()
            .chain(result.partially_propagated.iter())
            .filter_map(|dest| resolved.get(&dest).copied().map(|src| (dest, src)))
            .collect();

        for (dest, src) in event_pairs {
            env.record(EventKind::CopyPropagated)
                .message(format!("{dest} → {src}"));
        }

        result.total_replaced
    }
}

impl<T> FunctionPass<T> for CopyPropagationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "copy-propagation"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Eliminate redundant copy operations and trivial phi nodes"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Runs copy propagation on one method: [`Self::propagate`] with no
    /// per-iteration hook.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to simplify in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the budget its rounds count against.
    ///
    /// # Returns
    ///
    /// `true` if any copy or trivial phi was eliminated.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        Ok(self.propagate(ssa, env, |_, _| {}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{budget::WorkBudget, context::AnalysisContext},
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{
            MACHINE, MockTarget, MockType, assert_mock_valid_full, mock_terminator_at,
            run_mock_pass_boundary,
        },
    };

    /// Wrap an op in a synthetic instruction.
    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    /// Create an `I32` local variable defined at the given block/instruction site.
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

    /// A single `Copy` of a constant is propagated away and reported.
    #[test]
    fn simple_copy_eliminated() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let src = local_at(&mut ssa, 0, 0, 0);
        let dst = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(42),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: dst, src }));
        block.add_instruction(instr(SsaOp::returns(Some(dst))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "simple copy propagation", |ssa| {
            CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                |_, _| {},
            )
        });
        assert!(changed, "simple copy should be eliminated");
        assert!(log.has(EventKind::CopyPropagated));
    }

    /// A three-link copy chain collapses back to its original source.
    #[test]
    fn three_element_chain_collapsed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 0, 2);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        block.add_instruction(instr(SsaOp::Copy { dest: c, src: b }));
        block.add_instruction(instr(SsaOp::returns(Some(c))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "copy chain propagation", |ssa| {
            CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                |_, _| {},
            )
        });
        assert!(changed, "three-element chain should be collapsed");
    }

    /// A function with no copies reports no change.
    #[test]
    fn no_copies_nothing_changed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "no-copy propagation", |ssa| {
            CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                |_, _| {},
            )
        });
        assert!(!changed, "no copies should mean no changes");
    }

    /// An empty function reports no change.
    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty copy propagation", |ssa| {
            CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                |_, _| {},
            )
        });
        assert!(!changed);
    }

    /// A zero-iteration cap makes no change even when a propagatable copy exists.
    #[test]
    fn copy_with_zero_iterations_no_change() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let src = local_at(&mut ssa, 0, 0, 0);
        let dst = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(99),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: dst, src }));
        block.add_instruction(instr(SsaOp::returns(Some(dst))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "zero-iteration copy propagation", |ssa| {
            CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, 0)),
                |_, _| {},
            )
        });
        assert!(!changed, "zero iterations should make no changes");
    }

    /// Two copies of the same source both resolve to that source.
    #[test]
    fn copy_propagation_uses_existing_value() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 0, 2);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        b0.add_instruction(instr(SsaOp::Copy { dest: c, src: a }));
        b0.add_instruction(instr(SsaOp::returns(Some(c))));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "existing-value copy propagation", |ssa| {
            CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                |_, _| {},
            )
        });
        assert!(changed, "copy propagation should rewrite existing values");
        // The return should reference v0 (a) or v2 (c) - propagation happens
        let ret = mock_terminator_at(&ssa, 0);
        assert!(matches!(ret, SsaOp::Return { .. }));
    }

    /// A single iteration reports the number of uses it replaced.
    #[test]
    fn run_iteration_returns_count() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let src = local_at(&mut ssa, 0, 0, 0);
        let dst = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: dst, src }));
        block.add_instruction(instr(SsaOp::returns(Some(dst))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert_mock_valid_full(&ssa, "copy propagation iteration before");
        let count = CopyPropagationPass.run_iteration(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            |_, _| {},
        );
        assert_mock_valid_full(&ssa, "copy propagation iteration after");
        assert!(count > 0, "should have replaced at least one use");
    }

    /// A copy chain spanning three blocks is fully collapsed.
    #[test]
    fn multiple_copies_in_different_blocks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 1, 0);
        let d = local_at(&mut ssa, 3, 2, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Copy { dest: c, src: b }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Copy { dest: d, src: c }));
        b2.add_instruction(instr(SsaOp::returns(Some(d))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "multi-block copy propagation", |ssa| {
            CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                |_, _| {},
            )
        });
        assert!(changed, "chain across blocks should be collapsed");
    }

    /// A replacement whose source is defined after an existing use of the destination is rejected.
    #[test]
    fn protect_same_block_ordering_removes_late_source_replacement() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let dest = local_at(&mut ssa, 0, 0, 0);
        let use_before_src = local_at(&mut ssa, 1, 0, 1);
        let src = local_at(&mut ssa, 2, 0, 2);
        let one = local_at(&mut ssa, 3, 0, 3);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: use_before_src,
            left: dest,
            right: dest,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(20),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::returns(Some(use_before_src))));
        ssa.add_block(block);
        ssa.recompute_uses();
        assert_mock_valid_full(&ssa, "same-block ordering protection fixture");

        let mut copies = BTreeMap::from([(dest, src)]);
        protect_same_block_ordering(
            &build_first_uses(&ssa),
            &build_source_defs(&ssa),
            &mut copies,
        );

        assert!(copies.is_empty());
    }

    /// After a full chain is propagated, no `Copy` instructions or orphan variables remain.
    #[test]
    fn propagated_copies_do_not_leave_orphan_variables() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let c = local_at(&mut ssa, 2, 0, 2);
        let out = local_at(&mut ssa, 3, 0, 3);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        block.add_instruction(instr(SsaOp::Copy { dest: c, src: b }));
        block.add_instruction(instr(SsaOp::Copy { dest: out, src: c }));
        block.add_instruction(instr(SsaOp::returns(Some(out))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        assert!(run_mock_pass_boundary(
            &mut ssa,
            "orphan-free copy propagation",
            |ssa| CopyPropagationPass.propagate(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                |_, _| {}
            )
        ));
        assert!(
            !ssa.iter_instructions()
                .any(|(_, _, instr)| matches!(instr.op(), SsaOp::Copy { .. }))
        );
    }
}
