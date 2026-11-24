//! Block merging pass - eliminates trampoline blocks and coalesces
//! single-edge block pairs in the SSA CFG.
//!
//! # Transformations
//!
//! 1. **Trampoline elimination** - removes blocks containing only an
//!    unconditional jump by redirecting predecessors to the ultimate target.
//!    Phi operands that referenced the trampoline are updated to reference
//!    the correct predecessor.
//! 2. **Block coalescing** - merges a block into its sole predecessor when
//!    the predecessor's only successor is that block. Phi nodes in the
//!    successor are converted to `Copy` instructions because they have
//!    exactly one incoming edge.
//!
//! Entry block (B0) is handled specially: when it's a trampoline, the
//! target block is inlined into B0 (if safe - single predecessor, no phis)
//! or the method is marked for code regeneration.
//!
//! # Algorithm
//!
//! Phase 1 iterates at most the budget's `BlockMergeRounds` times, each pass identifying
//! trampolines (via [`SsaFunction::find_trampoline_blocks`]), redirecting
//! all predecessors through [`redirect_target`](SsaOp::redirect_target),
//! then clearing the trampolines.
//!
//! Phase 2 handles the entry block specially (it has no predecessor, so
//! phase 1 cannot redirect through it).
//!
//! Phase 3 coalesces blocks: it computes predecessor counts, identifies
//! single-edge (A → B) pairs where B has exactly one predecessor, converts
//! B's phis to copies, appends B's instructions to A, and redirects phi
//! operands from B to A. Exception-handler boundary blocks are excluded
//! from coalescing.
//!
//! # Complexity
//!
//! O(n * r) where n is the number of blocks and r the rounds allowed.

use std::collections::BTreeMap;

use crate::{
    Result,
    analysis::{budget::Budget, cfg::SsaCfg},
    bitset::BitSet,
    events::EventKind,
    graph::algorithms::strongly_connected_components,
    ir::{
        function::{SsaEditOptions, SsaEditor, SsaFunction},
        ops::SsaOp,
    },
    passes::{redirect::redirect_past_trampolines, utils::loop_canonical_blocks},
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Blocks that must not be merged *into* - a region entry absorbing a
/// predecessor from outside would pull non-region code into the region.
///
/// Together with [`exception_region_ends`] these are the exception-region
/// boundaries. Every path that removes or bypasses a block has to respect them:
/// nothing in the crate rewrites a clause's block ranges after a block is
/// emptied, so a boundary block that is silently dropped leaves the exception
/// edge pointing at a block with no terminator, and the real handler body loses
/// its only root. The scan reads through
/// [`parts`](crate::ir::exception::SsaExceptionHandler::parts), which validates
/// nothing: a barrier has to fence for a clause that cannot be laid out just as
/// much as for one that can - more so, since that clause is the one nobody has
/// checked.
///
/// # Arguments
///
/// * `ssa` - The SSA function whose exception table is scanned.
///
/// # Returns
///
/// A [`BitSet`] over block indices with the *start* of every protected,
/// handler, and filter part set.
fn exception_region_starts<T: Target>(ssa: &SsaFunction<T>) -> BitSet {
    let mut starts = BitSet::new(ssa.block_count());
    for handler in ssa.exception_handlers() {
        for (_, range) in handler.parts() {
            starts.insert_checked(range.start());
        }
    }
    starts
}

/// Blocks that must not be merged *from* - a region end absorbing its successor
/// would extend the region past its boundary.
///
/// Every part's exclusive end, the filter's included: a filter expression is as
/// much a bounded range of blocks as the protected region is, and a merge at its
/// tail extends it just the same. Complements [`exception_region_starts`] to
/// give the full set of clause boundaries.
///
/// # Arguments
///
/// * `ssa` - The SSA function whose exception table is scanned.
///
/// # Returns
///
/// A [`BitSet`] over block indices with the exclusive *end* of every protected,
/// handler, and filter part set.
fn exception_region_ends<T: Target>(ssa: &SsaFunction<T>) -> BitSet {
    let mut ends = BitSet::new(ssa.block_count());
    for handler in ssa.exception_handlers() {
        for (_, range) in handler.parts() {
            ends.insert_checked(range.end());
        }
    }
    ends
}

/// One pass of trampoline elimination over the whole function.
///
/// Finds every trampoline block (one that is a bare forwarding jump), then
/// removes from that set the blocks that must be preserved: canonical loop
/// preheaders - so this pass does not fight the loop canonicalizer, which would
/// re-insert any preheader merged away - and exception-region boundaries, since
/// [`find_trampoline_blocks`](SsaFunction::find_trampoline_blocks) skips only
/// the entry block and a handler or filter entry that happens to be a forwarding
/// jump must keep its terminator (nothing rewrites the clause's block ranges
/// afterward). The surviving trampolines then have every predecessor redirected
/// to their ultimate target and are cleared, within one edit session.
///
/// The loop forest and exception tables are built only once a non-empty
/// trampoline set makes them worth the cost; the terminating iteration that
/// finds nothing returns early without paying for them.
///
/// # Arguments
///
/// * `ssa` - The SSA function to simplify in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// Count of changes this pass (predecessors redirected plus trampolines
/// cleared); zero means convergence. A floor of one is reported when the edit
/// session fails after applying mutations, so the pass-group transaction
/// verifies and rolls back the damaged IR rather than trusting a misleading
/// zero.
fn run_trampoline_iteration<T>(ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> usize
where
    T: Target,
{
    let mut trampolines = ssa.find_trampoline_blocks(true);
    // Nothing to filter, so do not pay for the loop forest. This runs once per
    // fixpoint iteration, including the terminating one that finds nothing -
    // and `loop_canonical_blocks` builds the whole loop forest.
    if trampolines.is_empty() {
        return 0;
    }
    // Preserve canonical loop preheaders so this pass does not fight the loop
    // canonicalizer (which re-inserts any preheader merged away here).
    let preheaders = loop_canonical_blocks(ssa);
    // ...and preserve exception-region boundaries, which `coalesce_blocks`
    // already refuses to merge across. `find_trampoline_blocks` only skips the
    // entry block, so without this a handler or filter entry that happens to be
    // a bare forwarding jump has its predecessors redirected past it and is then
    // cleared - leaving the clause's handler range pointing at an empty block.
    let region_starts = exception_region_starts(ssa);
    let region_ends = exception_region_ends(ssa);
    trampolines.retain(|block, _| {
        !preheaders.contains(block)
            && !region_starts.contains_checked(*block)
            && !region_ends.contains_checked(*block)
    });
    if trampolines.is_empty() {
        return 0;
    }
    let mut redirected = 0usize;
    let mut cleared = 0usize;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        redirected = redirect_to_ultimate_targets(editor, &trampolines, env)?;
        cleared = clear_trampolines(editor, &trampolines, env)?;
        Ok(())
    });
    if result.is_err() {
        // Left applied under the session's policy; the function records the
        // failure and the pass's runner restores it (see
        // `SsaFunction::take_edit_failure`). A floor of one keeps a caller that
        // loops on this from reading it as convergence.
        return redirected.saturating_add(cleared).max(1);
    }
    redirected.saturating_add(cleared)
}

/// Redirect every terminator that targets a trampoline to the trampoline's
/// ultimate target, through [`redirect_past_trampolines`].
///
/// A trampoline's own predecessors are not rewritten: they are redirected past
/// it, and it is cleared once nothing jumps to it.
///
/// # Arguments
///
/// * `editor` - Active edit session over the function.
/// * `trampolines` - Map from each trampoline block to its immediate jump
///   target.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// Number of blocks whose terminator was redirected.
///
/// # Errors
///
/// Propagates an editor failure.
fn redirect_to_ultimate_targets<T>(
    editor: &mut SsaEditor<T>,
    trampolines: &BTreeMap<usize, usize>,
    env: &PassEnv<'_, T>,
) -> Result<usize>
where
    T: Target,
{
    let redirected = redirect_past_trampolines(editor, trampolines, &|block| {
        trampolines.contains_key(&block)
    })?;
    for (block_idx, old_targets, new_targets) in &redirected {
        env.record(EventKind::BranchSimplified)
            .location(*block_idx)
            .message(format!(
                "redirected through trampoline: {old_targets:?} -> {new_targets:?}"
            ));
    }
    Ok(redirected.len())
}

/// Empty out each trampoline nothing jumps to any more.
///
/// A trampoline one of whose predecessors could not be redirected (see
/// [`redirect_to_ultimate_targets`]) still carries that path and stays, and so
/// does every trampoline after it in its chain. Clearing one can leave the
/// next without predecessors, so this repeats until nothing more clears. The
/// phi operands naming a cleared block are dropped with it (see
/// [`SsaEditor::remove_unreached_block`]). Emits a
/// [`EventKind::BlockRemoved`] event per block cleared.
///
/// # Arguments
///
/// * `editor` - Active edit session over the function.
/// * `trampolines` - Map whose keys are the trampoline blocks to clear.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// Number of trampoline blocks actually cleared.
///
/// # Errors
///
/// Propagates an editor failure.
fn clear_trampolines<T>(
    editor: &mut SsaEditor<T>,
    trampolines: &BTreeMap<usize, usize>,
    env: &PassEnv<'_, T>,
) -> Result<usize>
where
    T: Target,
{
    let mut cleared: usize = 0;
    loop {
        let cfg = SsaCfg::from_ssa(editor.function());
        let unreached: Vec<usize> = trampolines
            .keys()
            .copied()
            .filter(|&block| {
                editor
                    .function()
                    .block(block)
                    .is_some_and(|b| !b.is_empty())
                    && cfg.predecessor_blocks(block).next().is_none()
            })
            .collect();
        drop(cfg);
        if unreached.is_empty() {
            return Ok(cleared);
        }
        for block_idx in unreached {
            if editor.remove_unreached_block(block_idx)? {
                env.record(EventKind::BlockRemoved)
                    .location(block_idx)
                    .message(format!("cleared trampoline block B{block_idx}"));
                cleared = cleared.saturating_add(1);
            }
        }
    }
}

/// Inline B0's target when B0 is a trampoline.
///
/// Non-entry trampolines are handled by [`run_trampoline_iteration`]; B0 has no
/// predecessors, so that redirect-the-predecessors approach can never reach it.
/// The target is inlined only when B0 is its sole predecessor and it has no
/// phis. A target with any other predecessor - including its own back edge -
/// stays where it is: moving a loop header into B0 would give the entry block
/// a predecessor, which the verifier rejects
/// ([`VerifierError::EntryHasPredecessors`](crate::analysis::VerifierError::EntryHasPredecessors)).
/// Inlining is also refused when B0 or its target is an exception-table
/// boundary (a handler entry, region start, or region end), since folding the
/// two blocks erases the boundary between them.
///
/// # Arguments
///
/// * `ssa` - The SSA function to simplify in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// `true` if the target was inlined into B0, `false` if nothing changed.
fn simplify_entry_trampoline<T>(ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> bool
where
    T: Target,
{
    let target = match ssa.block(0).and_then(|b| b.is_trampoline()) {
        Some(t) => t,
        None => return false,
    };

    // Inlining the target into B0 erases the boundary between them, so refuse
    // when either is one the exception table names: a handler entry, or the
    // start or end of a protected region.
    let exception_blocks = ssa.exception_blocks();
    let is_boundary = |block: usize| {
        exception_blocks.is_runtime_entry(block)
            || exception_blocks.is_region_start(block)
            || exception_blocks.is_region_end(block)
    };
    if is_boundary(0) || is_boundary(target) {
        return false;
    }
    drop(exception_blocks);

    // Every predecessor counts, the target's own back edge included: B0 must
    // be the only way in, or the inlined code would give the entry block a
    // predecessor.
    let cfg = SsaCfg::from_ssa(ssa);
    let sole_predecessor_is_entry = {
        let mut preds = cfg.predecessor_blocks(target);
        preds.next() == Some(0) && preds.next().is_none()
    };
    drop(cfg);
    let target_has_phis = ssa.block(target).is_none_or(|b| !b.phi_nodes().is_empty());
    if !sole_predecessor_is_entry || target_has_phis {
        return false;
    }

    // The coalesce the general case uses, which is this one exactly: B0 ends
    // in a jump to a block it is the only way into. It also renames the
    // target to B0 in every successor's phis. Moving the instructions by hand
    // left those phis naming the emptied target, so the rebuild dropped the
    // operand as coming from an unreachable block; the value that reached the
    // phi through it was cut off, and the phi read the slot's undefined entry
    // value instead - a flattened dispatcher's first state, in one NetReactor
    // method, which then switched on whatever the local held.
    let mut merged = false;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        merged = editor.coalesce_unconditional_successor(0, target)?;
        Ok(())
    });
    if result.is_err() {
        // Left applied under the session's policy; the function records the
        // failure and the pass's runner restores it (see
        // `SsaFunction::take_edit_failure`).
        return true;
    }
    if !merged {
        return false;
    }
    env.record(EventKind::BlockRemoved)
        .location(0)
        .message(format!(
            "inlined entry trampoline: B0 jump to B{target} merged into B0"
        ));
    true
}

/// Coalesce single-edge block pairs: fold a successor into its sole
/// predecessor.
///
/// Iterates to a fixpoint (capped by the budget's `BlockMergeRounds`). Each pass counts
/// predecessors, then pairs a block `A` ending in an unconditional `Jump` to
/// `B` with that successor `B` whenever `B` has exactly one predecessor. A pair
/// is skipped when either endpoint is an exception-region boundary (a region
/// start must not absorb an outside predecessor; a region end must not absorb an
/// outside successor), when `B` can reach `A` - a back edge, i.e. a loop, which
/// merging would collapse - or when `B` is empty. The entry block is given a
/// phantom predecessor so it is never seen as having a single one. Each
/// surviving pair is merged by `coalesce_unconditional_successor`, which appends
/// `B` into `A` and converts `B`'s single-edge phis to copies.
///
/// # Arguments
///
/// * `ssa` - The SSA function to simplify in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// Total number of blocks coalesced across all iterations.
fn coalesce_blocks<T>(ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> usize
where
    T: Target,
{
    let mut merged: usize = 0;

    // Collect exception-handler boundary blocks.
    //
    // - Region *start* blocks must not be the merge target - absorbing a
    //   predecessor outside the region would pull non-region code in.
    // - Region *end* blocks must not be the merge source - absorbing a
    //   successor outside the region would extend the region.
    let no_merge_into = exception_region_starts(ssa);
    let no_merge_from = exception_region_ends(ssa);

    let mut rounds = env.context().budget().rounds(Budget::BlockMergeRounds);
    while rounds.next_round() {
        let mut iteration_merges: usize = 0;

        let block_count = ssa.block_count();
        // Predecessor counts (edge multiplicity included) and each block's
        // strongly connected component, from one CFG view per iteration. For an
        // edge A -> B, B reaches A exactly when the two share a component, so
        // the back-edge test is a comparison instead of a walk per pair.
        let cfg = SsaCfg::from_ssa(ssa);
        let mut pred_counts: Vec<usize> = (0..block_count)
            .map(|block| cfg.block_predecessors(block).len())
            .collect();
        let mut component: Vec<usize> = vec![usize::MAX; block_count];
        for (id, members) in strongly_connected_components(&cfg).iter().enumerate() {
            for node in members {
                if let Some(slot) = component.get_mut(node.index()) {
                    *slot = id;
                }
            }
        }
        drop(cfg);
        if let Some(c) = pred_counts.get_mut(0) {
            *c = c.saturating_add(1);
        }

        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut consumed = BitSet::new(block_count);
        for a_idx in 0..block_count {
            if consumed.contains(a_idx) {
                continue;
            }
            let b_idx = match ssa.block(a_idx).and_then(|b| b.control_terminator()) {
                Some(SsaOp::Jump { target }) => *target,
                _ => continue,
            };
            if b_idx >= block_count || b_idx == a_idx {
                continue;
            }
            if pred_counts.get(b_idx).copied().unwrap_or(0) != 1 {
                continue;
            }
            if no_merge_from.contains(a_idx) || no_merge_into.contains(b_idx) {
                continue;
            }
            if component.get(a_idx) == component.get(b_idx) {
                continue;
            }
            let b_empty = ssa.block(b_idx).is_none_or(|b| b.instructions().is_empty());
            if b_empty {
                continue;
            }
            pairs.push((a_idx, b_idx));
            consumed.insert(a_idx);
            consumed.insert(b_idx);
        }

        if !pairs.is_empty() {
            let result = ssa.edit(SsaEditOptions::new(), |editor| {
                for &(a_idx, b_idx) in &pairs {
                    if editor.coalesce_unconditional_successor(a_idx, b_idx)? {
                        env.record(EventKind::BlockRemoved)
                            .location(b_idx)
                            .message(format!("coalesced B{b_idx} into B{a_idx}"));
                        iteration_merges = iteration_merges.saturating_add(1);
                    }
                }
                Ok(())
            });
            if result.is_err() {
                // The session runs under `SsaRollbackPolicy::Never`, so a
                // failure part-way leaves the merges before it applied. Report
                // at least one so a pass-group transaction verifies this
                // function rather than keeping it unchecked, and stop: the
                // next iteration would build on a half-finished edit.
                return merged.saturating_add(iteration_merges.max(1));
            }
        }

        merged = merged.saturating_add(iteration_merges);
        if iteration_merges == 0 {
            rounds.settle();
        }
    }
    if let Some(exhaustion) = rounds.exhausted() {
        env.exhausted(exhaustion);
    }

    merged
}

/// Pass that eliminates trampoline blocks and coalesces single-edge pairs.
///
/// Runs as many rounds as the context's [`Budget::BlockMergeRounds`]
/// allows.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - can change block predecessors,
/// successors, and remove blocks.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockMergingPass;

impl<T> FunctionPass<T> for BlockMergingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "block-merging"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Eliminate trampoline (single-jump) blocks"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run block merging on `ssa`.
    ///
    /// Executes three phases: trampoline elimination, entry-trampoline
    /// simplification, and block coalescing. Each inner loop is capped by
    /// the context's [`Budget::BlockMergeRounds`].
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any block was merged or any branch redirected.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let mut changed = false;

        // Phase 1: eliminate trampoline blocks.
        let mut rounds = env.context().budget().rounds(Budget::BlockMergeRounds);
        while rounds.next_round() {
            if run_trampoline_iteration(ssa, env) == 0 {
                rounds.settle();
            } else {
                changed = true;
            }
        }
        if let Some(exhaustion) = rounds.exhausted() {
            env.exhausted(exhaustion);
        }

        // Phase 2: handle entry trampoline (B0 has no predecessors so phase 1
        // can't redirect them - instead inline the target when that is safe).
        if simplify_entry_trampoline(ssa, env) {
            changed = true;
        }

        // Phase 3: coalesce non-trivial blocks connected by a single edge.
        if coalesce_blocks(ssa, env) > 0 {
            changed = true;
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
            ops::SsaOp,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType, run_mock_pass_boundary},
    };

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

    /// A handler entry that is a bare forwarding jump must survive the
    /// trampoline pass, just as `coalesce_blocks` refuses to merge across an
    /// exception-region boundary. Without that guard the trampoline path would
    /// redirect the region's predecessors past the handler entry and clear it.
    /// Nothing in the crate rewrites the clause's handler range afterwards, so
    /// the exception edge would still point at a now-empty block and the real
    /// handler body would lose its only root.
    #[test]
    fn a_handler_entry_that_is_a_trampoline_is_not_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);

        // The handler entry: a bare forwarding jump into the handler body.
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);

        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 1,
            handler_offset: 2,
            handler_length: 1,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 1),
            handler_range: BlockRange::new(2, 3),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            BlockMergingPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );

        let handler_entry = ssa.block(2).expect("handler entry block must still exist");
        assert!(
            handler_entry.control_terminator().is_some(),
            "the handler entry must keep its terminator; the clause's handler \
             range still starts here and nothing rewrites it"
        );
        // Coalescing the entry with its body is fine - the entry keeps a
        // terminator and still roots the handler. What must never happen is the
        // entry being *emptied*, which is what the trampoline path did.
        assert!(
            !handler_entry.instructions().is_empty(),
            "the handler entry must not be cleared out from under the clause's \
             handler range"
        );
    }

    /// A filter expression's exclusive end fences a merge, exactly as a
    /// protected region's and a handler's do.
    ///
    /// A filter is a bounded range of blocks like any other clause part: a merge
    /// at its tail extends the filter over code that was never part of it, and
    /// codegen would then write the wider range back out. The barrier is
    /// computed from every part the clause maps, so there is no part it can
    /// forget.
    ///
    /// `B0 -> B1 -> B2 -> B3 -> B4`, with protected `[0, 1)`, handler `[1, 2)`
    /// and filter `[2, 3)`. Only B3, the filter's exclusive end, stands between
    /// B4 and a merge.
    #[test]
    fn a_filter_end_fences_a_merge() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        for index in 0..4 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::Jump {
                target: index.saturating_add(1),
            }));
            ssa.add_block(block);
        }
        let mut last = SsaBlock::new(4);
        last.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(last);

        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 1, // `MockTarget` reads 1 as a filter handler.
            try_offset: 0,
            try_length: 1,
            handler_offset: 1,
            handler_length: 1,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 1),
            handler_range: BlockRange::new(1, 2),
            filter_range: BlockRange::new(2, 3),
            selector: None,
        }]);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            BlockMergingPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );

        assert!(
            ssa.block(4).is_some_and(|b| !b.instructions().is_empty()),
            "B4 must not be absorbed into B3: B3 is where the filter ends"
        );
        assert!(
            ssa.block(3)
                .and_then(SsaBlock::control_terminator)
                .is_some_and(|op| matches!(op, SsaOp::Jump { target: 4 })),
            "and B3 must still jump to it"
        );
    }

    /// `SsaOp::Leave` exits a protected region. Treating it as a plain
    /// forwarding jump rewrites `pred -> Leave(target)` into
    /// `pred -> Jump(target)`, dropping the region-exit semantics.
    #[test]
    fn a_leave_only_block_is_not_a_trampoline() {
        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Leave { target: 1 }));
        assert_eq!(
            block.is_trampoline(),
            None,
            "Leave carries region-exit semantics a Jump does not"
        );
        assert_eq!(
            block.is_unconditional_transfer(),
            Some(1),
            "but it is still structurally an unconditional transfer"
        );

        let mut jump: SsaBlock<MockTarget> = SsaBlock::new(1);
        jump.add_instruction(instr(SsaOp::Jump { target: 2 }));
        assert_eq!(jump.is_trampoline(), Some(2));
    }

    /// A lone forwarding jump between two real blocks is eliminated.
    #[test]
    fn simple_trampoline_elimination() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "simple block merge", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed);
        // B1 trampoline should be eliminated
        assert!(log.has(EventKind::BranchSimplified) || log.has(EventKind::BlockRemoved));
    }

    /// A chain of consecutive forwarding jumps collapses to its ultimate target.
    #[test]
    fn chain_of_trampolines() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        for i in 1..4 {
            let mut b = SsaBlock::new(i);
            b.add_instruction(instr(SsaOp::Jump { target: i + 1 }));
            ssa.add_block(b);
        }
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "trampoline chain block merge", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "chain of trampolines should be eliminated");
    }

    /// Two blocks joined by a single edge are coalesced into one.
    #[test]
    fn coalesce_sequential_blocks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        b1.add_instruction(instr(SsaOp::returns(Some(b))));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "sequential block coalescing", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "sequential blocks should coalesce");
    }

    /// A block that absorbs its successor stands for the host blocks of both.
    #[test]
    fn a_coalesced_block_stands_for_both_host_blocks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        b0.provenance_mut().carry(0x100);
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
        b1.add_instruction(instr(SsaOp::returns(Some(b))));
        b1.provenance_mut().carry(0x108);
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_mock_pass_boundary(&mut ssa, "coalesce", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        }));
        let carried: Vec<&[u64]> = ssa
            .blocks()
            .iter()
            .filter(|block| !block.is_empty())
            .map(|block| block.provenance().carried())
            .collect();
        assert_eq!(carried, vec![&[0x100, 0x108][..]]);
    }

    /// An edge redirected past a forwarding chain records the chain's host
    /// blocks as passed through, and the chain, cleared once nothing reaches
    /// it, stands for nothing.
    #[test]
    fn a_bypassed_chain_is_passed_through() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 2);
        let ten = local_at(&mut ssa, 0, 1, 0);
        let twenty = local_at(&mut ssa, 1, 2, 0);
        let b0 = branch_on_arg(&mut ssa, 1, 2);
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: ten,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
        b1.provenance_mut().carry(0x10);
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: twenty,
            value: ConstValue::I32(20),
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 5 }));
        ssa.add_block(b2);
        for (block, target, host) in [(3, 4, 0x30), (4, 5, 0x40)] {
            let mut b = SsaBlock::new(block);
            b.add_instruction(instr(SsaOp::Jump { target }));
            b.provenance_mut().carry(host);
            ssa.add_block(b);
        }
        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b5);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_mock_pass_boundary(&mut ssa, "bypass", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        }));
        let holder = ssa
            .blocks()
            .iter()
            .find(|block| block.provenance().carried() == [0x10])
            .unwrap_or_else(|| panic!("the block holding B1's code survives"));
        assert_eq!(holder.provenance().passed(), &[0x30, 0x40]);
        assert!(
            ssa.blocks()
                .iter()
                .all(|block| !block.provenance().carried().contains(&0x30)),
            "the bypassed chain is dead and stands for nothing"
        );
    }

    /// Coalescing a single-edge pair rewrites the successor's phi operands.
    #[test]
    fn coalesce_with_phi_operand() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 1, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::returns(Some(phi_var))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "phi operand block coalescing", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "coalescing should preserve phi operands");
    }

    /// A coalesce whose edit session fails part-way has still edited the
    /// function (the session never rolls back), so the pass must report a
    /// change: "unchanged" would let a pass-group transaction keep the
    /// half-merged IR without verifying it.
    ///
    /// The failure is forced with a stale phi operand: B2's phi names both B0
    /// and B1 with different values, so retargeting B1's operand to B0 after
    /// B1 has been folded into B0 is refused, after the fold already happened.
    #[test]
    fn a_coalesce_that_fails_mid_session_reports_a_change() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 1, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(10),
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
        let before = format!("{ssa}");

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = BlockMergingPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true);

        assert_ne!(
            format!("{ssa}"),
            before,
            "the failed session left its edits"
        );
        assert!(changed, "an applied edit must be reported as a change");
    }

    /// An entry block that is a bare forwarding jump is simplified.
    #[test]
    fn entry_trampoline_is_handled() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(7),
        }));
        b1.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "entry trampoline block merge", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "entry trampoline should be handled");
    }

    /// An empty function produces no changes.
    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty block merge", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
    }

    /// A function with no trampolines is left untouched.
    #[test]
    fn no_trampoline_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "no-trampoline block merge", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "no trampolines should mean no changes");
    }

    /// Eliminating a trampoline updates phi operands in its successor.
    #[test]
    fn trampoline_with_phi_successor() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 2, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v0, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        b2.add_instruction(instr(SsaOp::returns(Some(phi_var))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "phi successor trampoline merge", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "trampoline with phi successor should be handled");
    }

    /// A two-way branch on argument 0 to `left` or `right`, as the entry.
    fn branch_on_arg(
        ssa: &mut SsaFunction<MockTarget>,
        left: usize,
        right: usize,
    ) -> SsaBlock<MockTarget> {
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: left,
            false_target: right,
        }));
        b0
    }

    /// The definition of what `phi_block`'s first phi receives from `pred`.
    fn arriving_from(
        ssa: &SsaFunction<MockTarget>,
        phi_block: usize,
        pred: usize,
    ) -> Option<SsaOp<MockTarget>> {
        let phi = ssa.block(phi_block)?.phi_nodes().first()?;
        let value = phi.operand_from(pred)?.value();
        ssa.get_definition(value).cloned()
    }

    /// A chain of trampolines is bypassed with the value its last link brought.
    ///
    /// `B1 -> B3 -> B4 -> B5`, and the phi in `B5` names `B4`, the link that
    /// jumps to it. The phi was expanded for the *first* trampoline, where it
    /// has no operand, so nothing was added for `B1`; clearing the chain then
    /// dropped the only operand that carried `B1`'s value.
    #[test]
    fn a_trampoline_chain_is_bypassed_with_its_value() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 3);
        let ten = local_at(&mut ssa, 0, 1, 0);
        let twenty = local_at(&mut ssa, 1, 2, 0);
        let merged =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(5), MockType::I32);
        let b0 = branch_on_arg(&mut ssa, 1, 2);
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: ten,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: twenty,
            value: ConstValue::I32(20),
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 5 }));
        ssa.add_block(b2);
        for (block, target) in [(3, 4), (4, 5)] {
            let mut b = SsaBlock::new(block);
            b.add_instruction(instr(SsaOp::Jump { target }));
            ssa.add_block(b);
        }
        let mut b5 = SsaBlock::new(5);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(ten, 4));
        phi.add_operand(PhiOperand::new(twenty, 2));
        b5.add_phi(phi);
        b5.add_instruction(instr(SsaOp::returns(Some(merged))));
        ssa.add_block(b5);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_mock_pass_boundary(
            &mut ssa,
            "trampoline chain",
            |ssa| {
                BlockMergingPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            }
        ));
        assert!(ssa.take_edit_failure().is_none());
        assert!(
            matches!(
                arriving_from(&ssa, 5, 1),
                Some(SsaOp::Const {
                    value: ConstValue::I32(10),
                    ..
                })
            ),
            "B1 brings 10 straight to the merge:\n{ssa:?}"
        );
    }

    /// A redirect that would put two values on one edge is not made.
    ///
    /// `B1` reaches the merge `B3` directly, carrying 2, and through the
    /// trampoline `B2`, carrying 1. Redirecting its branch past `B2` makes both
    /// arms one edge, which cannot carry both; the attempt used to fail half
    /// way through the edit and leave it applied.
    #[test]
    fn a_redirect_that_would_merge_two_values_is_skipped() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 3);
        let one = local_at(&mut ssa, 0, 1, 0);
        let two = local_at(&mut ssa, 1, 1, 1);
        let merged =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::I32);
        let inner = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let b0 = branch_on_arg(&mut ssa, 1, 4);
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: two,
            value: ConstValue::I32(2),
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: inner,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(one, 2));
        phi.add_operand(PhiOperand::new(two, 1));
        b3.add_phi(phi);
        b3.add_instruction(instr(SsaOp::returns(Some(merged))));
        ssa.add_block(b3);
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "conflicting redirect", |ssa| {
            BlockMergingPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(
            ssa.take_edit_failure().is_none(),
            "nothing was attempted that could not be done"
        );
        assert!(
            matches!(
                arriving_from(&ssa, 3, 2),
                Some(SsaOp::Const {
                    value: ConstValue::I32(1),
                    ..
                })
            ),
            "the trampoline still carries 1:\n{ssa:?}"
        );
        assert!(
            matches!(
                arriving_from(&ssa, 3, 1),
                Some(SsaOp::Const {
                    value: ConstValue::I32(2),
                    ..
                })
            ),
            "the direct edge still carries 2:\n{ssa:?}"
        );
    }

    /// Coalescing inside a protected region keeps a finally's operands for
    /// throws apart from its operand for the `leave` edge.
    ///
    /// The finally's phi has operands for throws from `B1` and, from `B2`, one
    /// for its `leave` and one for a throw. Merging `B2` into `B1` renames `B2`
    /// to `B1` in them; the existing throw operand from `B1` holds a different
    /// value, which was read as a second value for the `leave` edge and
    /// refused, half way through the edit.
    #[test]
    fn coalescing_beside_a_finally_keeps_throws_and_the_leave_apart() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let before = local_at(&mut ssa, 0, 0, 0);
        let inside = local_at(&mut ssa, 0, 1, 0);
        let observed =
            ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(3), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: before,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: inside,
            value: ConstValue::I32(1),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Leave { target: 4 }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(observed, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::exceptional(before, 1));
        phi.add_operand(PhiOperand::exceptional(inside, 1));
        phi.add_operand(PhiOperand::new(inside, 2));
        phi.add_operand(PhiOperand::exceptional(inside, 2));
        b3.add_phi(phi);
        b3.add_instruction(instr(SsaOp::EndFinally));
        ssa.add_block(b3);
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b4);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, 3),
            handler_range: BlockRange::new(3, 4),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        // Read inside the session: the boundary rebuild prunes a phi nothing
        // reads, and this one is read only by the runtime.
        let mut after: Option<PhiNode> = None;
        let result = ssa.edit(SsaEditOptions::new(), |editor| {
            assert!(editor.coalesce_unconditional_successor(1, 2)?);
            after = editor
                .function()
                .block(3)
                .and_then(|b| b.phi_nodes().first())
                .cloned();
            Ok(())
        });
        assert!(result.is_ok(), "{result:?}");
        let phi = after.expect("the finally's phi");
        assert_eq!(phi.normal_operands().count(), 1, "one `leave` edge: {phi}");
        assert!(
            phi.normal_operands()
                .all(|operand| operand.predecessor() == 1 && operand.value() == inside),
            "the leave now leaves from B1: {phi}"
        );
        assert!(
            phi.exceptional_operands()
                .all(|operand| operand.predecessor() == 1),
            "every throw is from B1 now: {phi}"
        );
        assert_eq!(
            phi.exceptional_operands().count(),
            2,
            "the throw before and the throw after the store, once each: {phi}"
        );
    }

    /// Inlining an entry trampoline into B0 erases the boundary between them,
    /// so a target the exception table names must be refused.
    ///
    /// Nothing in the terminator relation marks such a block, so the guard has
    /// to consult the table directly.
    #[test]
    fn an_entry_trampoline_targeting_a_handler_entry_is_not_inlined() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);
        let mut handler = SsaBlock::new(1);
        handler.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(handler);

        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: None,
            handler_range: BlockRange::new(1, 2),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let events = EventLog::<MockTarget>::new();
        assert!(
            !simplify_entry_trampoline(
                &mut ssa,
                &PassEnv::new(&0u32, &events, AnalysisContext::new(MACHINE))
            ),
            "B1 is a handler entry; folding it into B0 loses where the runtime dispatches"
        );
        assert!(ssa.block(1).is_some_and(|b| !b.instructions().is_empty()));
    }

    /// An entry trampoline that cannot be inlined is not a change.
    ///
    /// `while` loops compile to exactly this shape - a jump from the entry to a
    /// loop header - and reporting it as changed on every call keeps the
    /// scheduler iterating to its cap over IR that never moves.
    #[test]
    fn an_entry_trampoline_into_a_loop_reports_no_change() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
        let condition = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);
        let mut header = SsaBlock::new(1);
        header.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: condition,
            value: ConstValue::I32(1),
        }));
        header.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(header);
        let mut latch = SsaBlock::new(2);
        latch.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(latch);
        let mut exit = SsaBlock::new(3);
        exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(exit);
        ssa.recompute_uses();

        let events = EventLog::<MockTarget>::new();
        for _ in 0..3 {
            assert!(
                !(BlockMergingPass.run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &events, AnalysisContext::new(MACHINE))
                ) == Ok(true)),
                "nothing about the entry trampoline can change"
            );
        }
        assert!(events.is_empty());
    }

    /// A trampoline whose target loops back to itself is left alone.
    ///
    /// Inlining it would move the loop header into B0 and give the entry block
    /// a predecessor, so the back edge counts like any other.
    #[test]
    fn an_entry_trampoline_whose_target_self_loops_is_not_inlined() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
        let condition = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );

        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);

        let mut target = SsaBlock::new(1);
        target.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: condition,
            value: ConstValue::I32(1),
        }));
        target.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(target);

        let mut exit = SsaBlock::new(2);
        exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(exit);
        ssa.recompute_uses();

        let events = EventLog::<MockTarget>::new();
        assert!(
            !simplify_entry_trampoline(
                &mut ssa,
                &PassEnv::new(&0u32, &events, AnalysisContext::new(MACHINE))
            ),
            "inlining B1 would make B0 the loop header"
        );
        assert!(events.is_empty(), "nothing changed, so nothing is reported");
        assert!(
            ssa.validate().is_ok(),
            "the entry block still has no predecessor"
        );
    }
}
