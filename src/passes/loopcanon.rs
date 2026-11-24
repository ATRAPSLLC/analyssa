//! Loop canonicalization pass - transforms loops into well-formed
//! canonical form with single preheaders and single latches.
//!
//! # Transformations
//!
//! 1. **Preheader insertion**: When a loop header has multiple non-loop
//!    predecessors, a fresh preheader block is created. All non-loop
//!    predecessors are redirected to it, and its terminator is a `Jump`
//!    to the header. Phi nodes in the header are updated: multiple non-loop
//!    operands are merged through a new phi in the preheader; a single
//!    non-loop operand is forwarded directly.
//!
//! 2. **Latch unification**: When a loop has multiple back-edges (latches),
//!    a single unified latch block is created. All original latches are
//!    redirected to it, and its terminator is a `Jump` to the header.
//!    Phi nodes in the header are rewritten: multiple latch operands are
//!    merged through a new phi in the unified latch.
//!
//! # Impact
//!
//! Canonical loop structure is required by subsequent analyses:
//! - LICM needs a single preheader to insert hoisted code.
//! - Induction variable detection assumes a single latch.
//! - Loop interchange and unrolling benefit from canonical form.
//!
//! # Algorithm
//!
//! Repeatedly analyzes loop structure via [`SsaFunction::analyze_loops`]. Each
//! pass canonicalizes *every* loop the forest reports, innermost-first, applying
//! at most one transformation per loop (preheader insertion takes priority over
//! latch unification) so phi management stays simple; a loop needing both is
//! finished on the next pass. Because every transformation is local to its own
//! loop, the forest is re-analyzed once per pass rather than once per individual
//! transformation. After all loops are canonical, [`SsaFunction::canonicalize`]
//! is called.

use rustc_hash::FxHashMap;

use crate::{
    Result,
    analysis::{budget::Budget, cfg::SsaCfg, loopanalyzer::SsaLoopAnalysis, loops::LoopInfo},
    events::EventKind,
    facts::Derivation,
    graph::NodeId,
    ir::{
        block::SsaBlock,
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Drive loop canonicalization to a fixpoint.
///
/// Each round re-analyses the loop forest and walks its loops innermost-first
/// (deepest depth first), applying at most one transformation per loop:
/// preheader insertion takes priority over latch unification, so phi bookkeeping
/// stays simple and a loop that needs both is finished on a later round. A loop
/// whose header the runtime dispatches to (an exception entry) is skipped - a
/// preheader works by redirecting the terminators that name the header, and no
/// terminator names such a header, so the spliced block would sit between
/// nothing and the header. The predecessor relation is derived once per round
/// and shared by every loop, avoiding an O(loops × blocks) rebuild. At most
/// `rounds` rounds, because no measure provably decreases across a round.
///
/// # Arguments
///
/// * `ssa` - The SSA function to canonicalize in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// Total number of transformations applied across all rounds.
fn canonicalize_loops<T>(ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> usize
where
    T: Target,
{
    let mut total_modified: usize = 0;

    let mut rounds = env
        .context()
        .budget()
        .rounds(Budget::CanonicalizationRounds);
    while rounds.next_round() {
        let forest = ssa.analyze_loops();
        if forest.is_empty() {
            rounds.settle();
            continue;
        }

        let mut modified_this_iteration: usize = 0;

        // One predecessor relation per round, shared by every loop. Deriving it
        // per loop would be O(blocks) each, so a function with many loops would
        // pay O(loops x blocks).
        let predecessors = SsaCfg::from_ssa(ssa).to_predecessor_sets();

        // A loop whose header the runtime dispatches to cannot be canonicalised
        // by splicing a block in front of it. A preheader works by redirecting
        // the terminators that name the header, and no terminator names this
        // one -- so the preheader would sit between nothing and the header
        // while the runtime kept entering the header directly.
        //
        // Loop detection covers loops inside handlers, so this case is
        // reachable. LICM needs no matching guard: `plan_loop_hoist`
        // already returns `None` unless the preheader's terminator names the
        // header, which is false by construction for such a loop.
        let exception_entries = ssa.exception_blocks();
        let is_runtime_entry: Vec<bool> = (0..ssa.block_count())
            .map(|block| exception_entries.is_runtime_entry(block))
            .collect();
        drop(exception_entries);

        for loop_info in forest.by_depth_descending() {
            if is_runtime_entry
                .get(loop_info.header.index())
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if !loop_info.has_preheader() {
                let non_loop_preds = get_non_loop_predecessors(&predecessors, loop_info);
                if non_loop_preds.len() > 1 {
                    if !splice_respects_regions(ssa, &non_loop_preds, loop_info.header.index()) {
                        continue;
                    }
                    if insert_preheader(ssa, loop_info, &non_loop_preds, env) {
                        modified_this_iteration = modified_this_iteration.saturating_add(1);
                    }
                    continue;
                }
            }
            if !loop_info.has_single_latch() && loop_info.latches.len() > 1 {
                let latches: Vec<usize> = loop_info.latches.iter().map(|l| l.index()).collect();
                if !splice_respects_regions(ssa, &latches, loop_info.header.index()) {
                    continue;
                }
                if unify_latches(ssa, loop_info, env) {
                    modified_this_iteration = modified_this_iteration.saturating_add(1);
                }
            }
        }

        total_modified = total_modified.saturating_add(modified_this_iteration);
        if modified_this_iteration == 0 {
            rounds.settle();
        }
    }
    if let Some(exhaustion) = rounds.exhausted() {
        env.exhausted(exhaustion);
    }

    total_modified
}

/// Whether a block spliced between `sources` and `header` keeps the
/// function's exception structure.
///
/// The spliced block is appended, so no protected region, handler or filter
/// contains it. Each source's edge moves from the header to it, and it then
/// branches to the header: for a loop inside a `try`, that is an edge out of
/// the region the original stayed in and one back into its middle, neither of
/// which a clause can express. The test reads the block's index before it is
/// appended, which no clause covers either.
///
/// # Arguments
///
/// * `ssa` - The function the block would be appended to.
/// * `sources` - The blocks whose edges into `header` would go through it.
/// * `header` - The loop header it would branch to.
///
/// # Returns
///
/// `true` when every edge the splice rewrites or adds respects the clauses.
fn splice_respects_regions<T: Target>(
    ssa: &SsaFunction<T>,
    sources: &[usize],
    header: usize,
) -> bool {
    let spliced = ssa.block_count();
    sources
        .iter()
        .all(|&source| ssa.retarget_respects_regions(source, header, spliced))
        && ssa.retarget_respects_regions(spliced, header, header)
}

/// Returns the loop header's predecessors that lie outside the loop body.
///
/// Reads the precomputed predecessor relation rather than re-deriving it by
/// scanning every block's terminator, which is O(preds) instead of O(blocks).
/// These are the blocks a preheader would be spliced in front of, so each must
/// be one whose terminator really does transfer to the header - exactly what the
/// relation records. A block the runtime dispatches to carries no such transfer
/// and must not be counted here, or the preheader would be inserted for an edge
/// no terminator can be redirected along.
///
/// # Arguments
///
/// * `predecessors` - Precomputed predecessor sets, indexed by block.
/// * `loop_info` - The loop whose header predecessors are examined.
///
/// # Returns
///
/// The header's predecessor blocks that are not part of the loop body.
fn get_non_loop_predecessors(predecessors: &[Vec<usize>], loop_info: &LoopInfo) -> Vec<usize> {
    let header_idx = loop_info.header.index();
    predecessors
        .get(header_idx)
        .into_iter()
        .flatten()
        .copied()
        .filter(|block_idx| !loop_info.contains(NodeId::new(*block_idx)))
        .collect()
}

/// Insert a fresh preheader in front of a loop header that has multiple
/// non-loop predecessors.
///
/// For each header phi, the operands coming from non-loop predecessors are
/// collected. A phi with more than one such operand gets a merging phi in the
/// new preheader - keyed by the header phi's *result*, not its origin, because
/// `VariableOrigin::Phi` is a unit variant shared by every phi that merges
/// earlier phis (the ordinary shape in lifted native code); keying on the origin
/// would collapse them all onto one preheader phi and rewrite every header phi
/// to that single value, a dominance violation for all but one. The preheader
/// ends in a `Jump` to the header, every non-loop predecessor is redirected to
/// it, and each header phi's non-loop operand group is replaced by one incoming
/// edge from the preheader - the lone value forwarded directly, or the merged
/// preheader-phi result when there were several.
///
/// # Arguments
///
/// * `ssa` - The SSA function to modify in place.
/// * `loop_info` - The loop being canonicalized.
/// * `non_loop_preds` - Header predecessors outside the loop, to be rerouted
///   through the new preheader.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// `true` when the preheader was inserted; `false` when the edit failed and
/// the function was restored as it was found.
fn insert_preheader<T>(
    ssa: &mut SsaFunction<T>,
    loop_info: &LoopInfo,
    non_loop_preds: &[usize],
    env: &PassEnv<'_, T>,
) -> bool
where
    T: Target,
{
    let header_idx = loop_info.header.index();
    let preheader_idx = ssa.block_count();
    // Keyed by the header phi's *result*, not its origin. `VariableOrigin::Phi`
    // is a unit variant, so every phi that merges earlier phis - the ordinary
    // shape in lifted native code - carries the same origin. Keying on it made
    // one preheader phi stand for all of them and rewrote every header phi to
    // that single value, which is a dominance violation for all but one.
    let phi_info: Vec<(SsaVarId, VariableOrigin, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .filter_map(|phi| {
                    let non_loop_operands: Vec<_> = phi
                        .operands()
                        .iter()
                        .filter(|op| non_loop_preds.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    if non_loop_operands.len() > 1 {
                        Some((phi.result(), phi.origin(), non_loop_operands))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let header_phi_operands: Vec<(SsaVarId, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .map(|phi| {
                    let non_loop_operands = phi
                        .operands()
                        .iter()
                        .filter(|op| non_loop_preds.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    (phi.result(), non_loop_operands)
                })
                .collect()
        })
        .unwrap_or_default();

    let result = ssa.edit(
        SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure),
        |editor| {
            let mut preheader = SsaBlock::new(preheader_idx);
            let mut preheader_phi_map: FxHashMap<SsaVarId, SsaVarId> = FxHashMap::default();

            for (header_result, origin, operands) in &phi_info {
                // The header phi's value on entry, gathered where the entries now
                // meet: that phi, on those edges.
                let new_var = editor.derive_variable_for_origin(
                    *header_result,
                    *origin,
                    0,
                    DefSite::phi(preheader_idx),
                    Derivation::LoopSplit,
                );
                let mut preheader_phi = PhiNode::new(new_var, *origin);
                for op in operands {
                    preheader_phi.add_operand(*op);
                }
                preheader_phi_map.insert(*header_result, new_var);
                preheader.phi_nodes_mut().push(preheader_phi);
            }

            preheader.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
                target: header_idx,
            }));
            editor.append_block(preheader)?;

            for &pred_idx in non_loop_preds {
                editor.redirect_terminator_target(pred_idx, header_idx, preheader_idx)?;
            }

            for (header_result, non_loop_values) in &header_phi_operands {
                if non_loop_values.is_empty() {
                    continue;
                }
                let replacement = if let [single] = non_loop_values.as_slice() {
                    single.value()
                } else if let Some(&preheader_var) = preheader_phi_map.get(header_result) {
                    preheader_var
                } else {
                    continue;
                };
                editor.replace_phi_predecessor_group_for_result(
                    header_idx,
                    *header_result,
                    non_loop_preds,
                    PhiOperand::new(replacement, preheader_idx),
                )?;
            }

            // Every value the new edges carry is stated: each header phi's entry
            // operands meet in the preheader, and a value defined above the loop
            // still dominates every use, the preheader being on every entry path.
            editor.declare_repaired();
            Ok(())
        },
    );
    if result.is_err() {
        return false;
    }

    env.record(EventKind::ControlFlowRestructured)
        .location(preheader_idx)
        .message(format!(
            "Inserted preheader B{preheader_idx} for loop at B{header_idx}"
        ));
    true
}

/// Merge a loop's multiple back-edges into a single unified latch block.
///
/// For each header phi, the operands arriving from latch blocks are collected. A
/// phi with more than one latch operand gets a merging phi in the new unified
/// latch - keyed by the header phi's *result*, not its origin, for the same
/// reason as [`insert_preheader`]: `VariableOrigin::Phi` is a unit variant
/// shared by every phi that merges earlier phis, so keying on it would rewrite
/// every header phi to a single value and violate dominance for all but one. The
/// unified latch ends in a `Jump` to the header, every original latch is
/// redirected to it, and each header phi's latch operand group is replaced by
/// one incoming edge from the unified latch - the merged phi result, or the lone
/// latch value forwarded directly.
///
/// # Arguments
///
/// * `ssa` - The SSA function to modify in place.
/// * `loop_info` - The loop whose latches are unified.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// `true` when the latches were unified; `false` when the edit failed and the
/// function was restored as it was found.
fn unify_latches<T>(ssa: &mut SsaFunction<T>, loop_info: &LoopInfo, env: &PassEnv<'_, T>) -> bool
where
    T: Target,
{
    let header_idx = loop_info.header.index();
    let latches: Vec<usize> = loop_info.latches.iter().map(|n| n.index()).collect();
    let unified_latch_idx = ssa.block_count();
    // Keyed by the header phi's *result*, not its origin. `VariableOrigin::Phi`
    // is a unit variant, so every phi that merges earlier phis - the ordinary
    // shape in lifted native code - carries the same origin. Keying on it made
    // one unified-latch phi stand for all of them and rewrote every header phi
    // to that single value, which is a dominance violation for all but one.
    let phi_info: Vec<(SsaVarId, VariableOrigin, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .map(|phi| {
                    let latch_operands: Vec<_> = phi
                        .operands()
                        .iter()
                        .filter(|op| latches.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    (phi.result(), phi.origin(), latch_operands)
                })
                .collect()
        })
        .unwrap_or_default();

    let result = ssa.edit(
        SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure),
        |editor| {
            let mut unified_latch = SsaBlock::new(unified_latch_idx);
            let mut latch_phi_vars: FxHashMap<SsaVarId, SsaVarId> = FxHashMap::default();

            for (header_result, origin, latch_operands) in &phi_info {
                if latch_operands.len() > 1 {
                    // The header phi's value coming back around, gathered where the
                    // latches now meet.
                    let new_var = editor.derive_variable_for_origin(
                        *header_result,
                        *origin,
                        0,
                        DefSite::phi(unified_latch_idx),
                        Derivation::LoopSplit,
                    );
                    let mut latch_phi = PhiNode::new(new_var, *origin);
                    for op in latch_operands {
                        latch_phi.add_operand(*op);
                    }
                    latch_phi_vars.insert(*header_result, new_var);
                    unified_latch.phi_nodes_mut().push(latch_phi);
                } else if let [single] = latch_operands.as_slice() {
                    latch_phi_vars.insert(*header_result, single.value());
                }
            }

            unified_latch.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
                target: header_idx,
            }));
            editor.append_block(unified_latch)?;

            for &latch_idx in &latches {
                editor.redirect_terminator_target(latch_idx, header_idx, unified_latch_idx)?;
            }

            for (header_result, _, _) in &phi_info {
                if let Some(&var) = latch_phi_vars.get(header_result) {
                    editor.replace_phi_predecessor_group_for_result(
                        header_idx,
                        *header_result,
                        &latches,
                        PhiOperand::new(var, unified_latch_idx),
                    )?;
                }
            }

            // Every value the new edges carry is stated: each header phi's latch
            // operands meet in the unified latch, and no latch dominates a use
            // outside its own path, so nothing else reads across the new block.
            editor.declare_repaired();
            Ok(())
        },
    );
    if result.is_err() {
        return false;
    }

    env.record(EventKind::ControlFlowRestructured)
        .location(unified_latch_idx)
        .message(format!(
            "Unified {} latches into B{} for loop at B{}",
            latches.len(),
            unified_latch_idx,
            header_idx
        ));
    true
}

/// Pass that ensures every loop has a single preheader and a single latch.
///
/// Creates new preheader
/// and/or unified latch blocks when loops have multiple non-loop
/// predecessors or multiple back-edges.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - can add new blocks and
/// rewrite phi operands.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoopCanonicalizationPass;

impl LoopCanonicalizationPass {
    /// Creates a new loop-canonicalization pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for LoopCanonicalizationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "loop-canonicalization"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Ensure each loop has a single preheader and latch"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run loop canonicalization on `ssa`.
    ///
    /// Inserts missing preheaders and unifies multiple latches so every loop
    /// has well-formed canonical structure. After convergence,
    /// [`SsaFunction::canonicalize`] is called on the function.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any preheader was inserted or any latch unified.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        if ssa.block_count() < 2 {
            return Ok(false);
        }

        let modified = canonicalize_loops(ssa, env);
        if modified > 0 {
            ssa.canonicalize();
        }
        Ok(modified > 0)
    }
}

/// Re-export the predicates pass struct (defined in
/// [`crate::passes::predicates`]) and add an [`FunctionPass`] impl. The
/// struct is `OpaquePredicatePass<T>`; analyssa's blanket below makes it fit
/// any host, since a [`FunctionPass`] is not
/// bound to one.
pub use crate::passes::predicates::OpaquePredicatePass;

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
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType, assert_mock_valid_full, run_mock_pass_boundary},
    };

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    /// A loop inside a `try` keeps its two latches: the unified latch would be
    /// appended outside every region, so both latches would leave the `try`
    /// by a plain branch and the new block re-enter it at the header.
    #[test]
    fn a_loop_inside_a_try_is_not_spliced_outside_it() {
        const PROTECTED: (usize, usize) = (1, 6);
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let terminators = [
            SsaOp::Jump { target: 1 },
            SsaOp::Jump { target: 2 },
            SsaOp::Branch {
                condition,
                true_target: 3,
                false_target: 4,
            },
            SsaOp::Jump { target: 2 },
            SsaOp::Branch {
                condition,
                true_target: 2,
                false_target: 5,
            },
            SsaOp::Leave { target: 7 },
            SsaOp::EndFinally,
            SsaOp::returns(None),
        ];
        for (index, op) in terminators.into_iter().enumerate() {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(op));
            ssa.add_block(block);
        }
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(PROTECTED.0, PROTECTED.1),
            handler_range: BlockRange::new(PROTECTED.1, PROTECTED.1 + 1),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "loop inside a try", |ssa| {
            LoopCanonicalizationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert_eq!(ssa.block_count(), 8, "no block was appended:\n{ssa:?}");
        for latch in [3, 4] {
            let targets = ssa
                .block(latch)
                .and_then(SsaBlock::control_terminator)
                .map(SsaOp::successors)
                .unwrap_or_default();
            assert!(
                targets.contains(&2),
                "B{latch} still loops back to the header"
            );
        }
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

    fn phi_local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::phi(block),
            MockType::I32,
        )
    }

    fn build_loop_with_multiple_latches() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 8);
        let init = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let limit = local_at(&mut ssa, 2, 0, 2);
        let i_phi = phi_local(&mut ssa, 3, 1);
        let cond = local_at(&mut ssa, 4, 1, 0);
        let next = local_at(&mut ssa, 5, 2, 0);
        let alt_next = local_at(&mut ssa, 6, 3, 0);
        let branch_cond = local_at(&mut ssa, 7, 5, 0);

        // Preheader (B0)
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(100),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // Header (B1): both latches feed the induction phi.
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(init, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        phi.add_operand(PhiOperand::new(alt_next, 3));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
            signaling: false,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 5,
            false_target: 4,
        }));
        ssa.add_block(b1);

        // Latch 1 (B2)
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        // Latch 2 (B3)
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Add {
            dest: alt_next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b3);

        // Exit (B4)
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(Some(i_phi))));
        ssa.add_block(b4);

        // Body (B5): conditionally reaches either latch, so both are genuine
        // reachable back-edges to the header.
        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::Clt {
            dest: branch_cond,
            left: i_phi,
            right: one,
            unsigned: false,
            signaling: false,
        }));
        b5.add_instruction(instr(SsaOp::Branch {
            condition: branch_cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b5);

        ssa.recompute_uses();
        ssa
    }

    /// A loop with several back-edges gains a single unified latch block.
    #[test]
    fn multiple_latches_unified() {
        let mut ssa = build_loop_with_multiple_latches();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let original_blocks = ssa.block_count();
        let changed =
            run_mock_pass_boundary(&mut ssa, "multiple-latch loop canonicalization", |ssa| {
                LoopCanonicalizationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed, "multiple latches should be unified");
        assert!(
            ssa.block_count() > original_blocks,
            "should add a unified latch block"
        );

        assert_mock_valid_full(&ssa, "multiple-latch canonical loop");
    }

    /// A loop that already has a single latch is left unchanged.
    #[test]
    fn single_latch_already_canonical() {
        let mut ssa = SsaFunction::new(0, 5);
        let init = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let limit = local_at(&mut ssa, 2, 0, 2);
        let i_phi = phi_local(&mut ssa, 3, 1);
        let cond = local_at(&mut ssa, 4, 1, 0);
        let next = local_at(&mut ssa, 5, 2, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(init, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
            signaling: false,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(Some(i_phi))));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "single-latch loop canonicalization", |ssa| {
                LoopCanonicalizationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(!changed, "single latch should already be canonical");
    }

    /// A function with no loops produces no changes.
    #[test]
    fn no_loops_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty loop canonicalization", |ssa| {
            LoopCanonicalizationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
    }

    /// A single-block function has no loop to canonicalize.
    #[test]
    fn single_block_no_loop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "single-block loop canonicalization", |ssa| {
                LoopCanonicalizationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(!changed, "single block cannot have a loop");
    }

    /// Re-running canonicalization on already-canonical IR is a no-op.
    #[test]
    fn idempotent_after_canonicalization() {
        let mut ssa = build_loop_with_multiple_latches();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let first_changed =
            run_mock_pass_boundary(&mut ssa, "first loop canonicalization", |ssa| {
                LoopCanonicalizationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(
            first_changed,
            "first canonicalization should rewrite the loop"
        );
        let after_first = ssa.block_count();

        // Second run should be idempotent
        let changed = run_mock_pass_boundary(&mut ssa, "idempotent loop canonicalization", |ssa| {
            LoopCanonicalizationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "second canonicalization should be a no-op");
        assert_eq!(ssa.block_count(), after_first);
    }

    /// A loop needing both a preheader and latch unification is fully canonicalized.
    #[test]
    fn loop_with_preheader_insertion_and_latch_unification() {
        // Build a loop with BOTH: no preheader (B0 and B1 enter it) AND
        // multiple latches (B3 and B4 return to it).
        let mut ssa = SsaFunction::new(0, 9);
        let init_a = local_at(&mut ssa, 0, 0, 0);
        let init_b = local_at(&mut ssa, 1, 1, 0);
        let one = local_at(&mut ssa, 2, 0, 1);
        let limit = local_at(&mut ssa, 3, 0, 2);
        let i_phi = phi_local(&mut ssa, 4, 2);
        let cond = local_at(&mut ssa, 5, 2, 0);
        let next1 = local_at(&mut ssa, 6, 3, 0);
        let next2 = local_at(&mut ssa, 7, 4, 0);
        let pick = local_at(&mut ssa, 8, 0, 3);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init_a,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(50),
        }));
        b0.add_instruction(instr(SsaOp::Clt {
            dest: pick,
            left: one,
            right: limit,
            unsigned: false,
            signaling: false,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: pick,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: init_b,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(4));
        phi.add_operand(PhiOperand::new(init_a, 0));
        phi.add_operand(PhiOperand::new(init_b, 1));
        phi.add_operand(PhiOperand::new(next1, 3));
        phi.add_operand(PhiOperand::new(next2, 4));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
            signaling: false,
        }));
        b2.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 3,
            false_target: 5,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Add {
            dest: next1,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 4,
        }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Add {
            dest: next2,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b4.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b4);

        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::returns(Some(i_phi))));
        ssa.add_block(b5);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "preheader-and-latch canonicalization", |ssa| {
                LoopCanonicalizationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed, "loop with both issues should be canonicalized");
        let messages: Vec<&str> = log
            .filter_kind(EventKind::ControlFlowRestructured)
            .map(|event| event.message.as_str())
            .collect();
        assert!(
            messages.iter().any(|m| m.starts_with("Inserted preheader")),
            "the preheader was inserted: {messages:?}"
        );
        assert!(
            messages.iter().any(|m| m.starts_with("Unified 2 latches")),
            "the latches were unified: {messages:?}"
        );
        let header_predecessors = (0..ssa.block_count())
            .filter(|&block| {
                ssa.block(block)
                    .and_then(|b| b.control_terminator())
                    .is_some_and(|op| op.successors().contains(&2))
            })
            .count();
        assert_eq!(
            header_predecessors, 2,
            "the header is entered from the preheader and the one latch\n{ssa}"
        );

        assert_mock_valid_full(&ssa, "preheader-and-latch canonical loop");
    }
}
