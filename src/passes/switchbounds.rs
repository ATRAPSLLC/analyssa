//! Prunes unreachable switch cases using a proven bound on the dispatch index.
//!
//! Jump-table recovery reads the table optimistically: the entry count comes
//! from a syntactic guard when one is found nearby, and otherwise from however
//! many consecutive slots happen to land in a code section. Both over-collect,
//! so a recovered dispatch routinely carries cases the program can never reach.
//!
//! This pass removes them, using two facts that only exist together after the
//! foundational analyses landed:
//!
//! - **The index is bounded.** [`crate::passes::ranges`] proves a range for
//!   the dispatch value, refined by the guards that dominate the dispatch - so
//!   a `cmp idx, N; ja default` above the table bounds it even though the
//!   whole-function range for that value is unconstrained.
//! - **Cases are ordered.** The recovered table keeps its index order, so case
//!   `i` is `targets[i]` and a bound on `i` names exactly which entries go.
//!
//! # Soundness
//!
//! Every rewrite only ever *removes* successors, and only those the proven
//! range excludes. A case is kept whenever anything is uncertain: an
//! unconverged analysis, an unbounded index, a negative lower bound, or a
//! bound at or beyond the last case all leave the dispatch untouched. The
//! default successor is never pruned - it is where an out-of-range index goes.
//!
//! No new target is invented: this only narrows what the front end already
//! recovered.

use crate::{
    Result,
    analysis::SsaCfg,
    graph::{RootedGraph, algorithms::compute_dominators},
    ir::{SsaFunction, SsaOp},
    passes::ranges::{GuardedRanges, RangeFacts},
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// One dispatch to narrow, and the case list it should keep.
struct Prune {
    /// Block whose terminator is the dispatch.
    block: usize,
    /// Case targets that survive, in table index order.
    targets: Vec<usize>,
}

/// Returns the dispatches whose case list a proven index bound shortens.
fn plan_prunes<T: Target>(ir: &SsaFunction<T>, facts: &RangeFacts) -> Vec<Prune> {
    // Nothing below is used outside the `Switch` arm, so a function with no
    // dispatch discards the CFG, the range facts and the dominator tree
    // wholesale. The range fixpoint is a full worklist dataflow with no cheap
    // early-out of its own when no earlier reader computed it, and this pass
    // runs for every lifted function - so on a corpus with few jump tables that
    // is most of its cost. An O(blocks) terminator scan first is semantically
    // identical and skips all three.
    if !(0..ir.block_count()).any(|index| {
        matches!(
            ir.block(index).and_then(|block| block.control_terminator()),
            Some(SsaOp::Switch { .. })
        )
    }) {
        return Vec::new();
    }

    // Dominance over the exception-aware view: a handler entry has no
    // terminator edge into it, so rooting at the entry block alone would leave
    // every guarded block undominated and the range facts unusable there.
    let cfg = SsaCfg::from_ssa(ir);
    let Ok(value_ranges) = facts.get(ir) else {
        // The analysis did not reach its fixpoint, so it proved nothing.
        return Vec::new();
    };
    let dominators = compute_dominators(&cfg, cfg.entry());

    let mut plan = Vec::new();
    for block_idx in 0..ir.block_count() {
        let Some(block) = ir.block(block_idx) else {
            continue;
        };
        let Some(SsaOp::Switch { value, targets, .. }) = block.control_terminator() else {
            continue;
        };
        let range = value_ranges.range_at(ir, &dominators, *value, block_idx);
        // Only the upper bound matters. Pruning the tail of the table is sound
        // whatever the lower bound is: an index below zero is out of range and
        // takes the default, exactly as an index past the last case does. A
        // guard like `ja default` therefore prunes even though the dispatch
        // value's lower bound is unknown - which is the common shape, since the
        // index usually arrives as an argument or a load.
        let Some(high) = range.max() else {
            continue;
        };
        // A wholly negative range reaches no case at all. That is sound to act
        // on but vanishingly rare and easy to get wrong, so leave it alone.
        let Ok(last_reachable) = usize::try_from(high) else {
            continue;
        };
        // `high` is inclusive, so cases `0..=high` survive.
        let keep = last_reachable.saturating_add(1).min(targets.len());
        if keep >= targets.len() {
            continue;
        }
        let Some(kept) = targets.get(..keep) else {
            continue;
        };
        plan.push(Prune {
            block: block_idx,
            targets: kept.to_vec(),
        });
    }
    plan
}

/// Rewrites each planned dispatch, pruning the phi operands of the successors
/// that lose their edge.
fn apply_prunes<T: Target>(ir: &mut SsaFunction<T>, plan: Vec<Prune>) -> bool {
    let mut pruned = false;
    let report = ir.edit(Default::default(), |editor| {
        for prune in plan {
            let Some(block) = editor.function().block(prune.block) else {
                continue;
            };
            let Some(SsaOp::Switch { value, default, .. }) = block.control_terminator() else {
                continue;
            };
            let folded = SsaOp::Switch {
                value: *value,
                targets: prune.targets,
                default: *default,
            };
            editor.fold_terminator_pruning_phis(prune.block, folded)?;
            pruned = true;
        }
        Ok(())
    });
    report.is_ok() && pruned
}

/// Pass that prunes switch cases a proven index bound rules out.
///
/// A pruned
/// case can leave its target unreachable and its phi operands gone, which is
/// work for the structural and value passes; scheduling it inside the fixpoint
/// is what lets them do that work before the function is called normalized.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - removes switch successors and the
/// phi operands of the blocks that lose their edge.
#[derive(Debug, Default, Clone, Copy)]
pub struct SwitchBoundsPass;

impl SwitchBoundsPass {
    /// Creates a new switch-bounds pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for SwitchBoundsPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "switch-bounds"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Prune switch cases a proven index bound rules out"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Prunes switch cases a proven index bound rules out.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` when at least one case was removed; `false` means nothing was
    /// edited.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let facts = env.ranges();
        let plan = plan_prunes(ssa, facts);
        if plan.is_empty() {
            return Ok(false);
        }
        Ok(apply_prunes(ssa, plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::NullListener,
        ir::{
            ConstValue, SsaBlock, SsaInstruction,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// Builds a guarded dispatch whose recovered table over-collected:
    ///
    /// ```text
    /// b0: limit = 2; c = idx > limit; branch c ? b1 : b2
    /// b1: return                          (out-of-range default)
    /// b2: switch idx -> [b3, b4, b5, b6]  (only 0..=2 are reachable)
    /// b3..b6: return
    /// ```
    ///
    /// `idx` is a function-entry live-in with no defining instruction, so
    /// nothing in the body constrains it and only the guard bounds it, and
    /// only on the path through `b2`.
    fn guarded_dispatch(case_count: usize) -> SsaFunction<MockTarget> {
        let mut ir = SsaFunction::<MockTarget>::with_capacity(0, 0, 8, 16);
        let idx = ir.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let limit = ir.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let condition = ir.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I1,
        );

        let mut b0 = SsaBlock::with_capacity(0, 0, 3);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Cgt {
            dest: condition,
            left: idx,
            right: limit,
            unsigned: false,
            signaling: false,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 2,
        }));
        ir.add_block(b0);

        let mut b1 = SsaBlock::with_capacity(1, 0, 1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ir.add_block(b1);

        // Case bodies start at block 3.
        let targets: Vec<usize> = (0..case_count).map(|case| case.saturating_add(3)).collect();
        let mut b2 = SsaBlock::with_capacity(2, 0, 1);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Switch {
            value: idx,
            targets,
            default: 1,
        }));
        ir.add_block(b2);

        for case in 0..case_count {
            let mut body = SsaBlock::with_capacity(case.saturating_add(3), 0, 1);
            body.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ir.add_block(body);
        }

        ir.recompute_uses();
        ir
    }

    /// Returns the dispatch's surviving case targets.
    fn case_targets(ir: &SsaFunction<MockTarget>) -> Vec<usize> {
        ir.block(2)
            .and_then(|block| block.control_terminator())
            .and_then(|op| match op {
                SsaOp::Switch { targets, .. } => Some(targets.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// A dominating `idx > 2` guard proves cases 3 and beyond unreachable, so
    /// an over-collected table is narrowed to the cases the program can enter.
    #[test]
    fn a_proven_index_bound_prunes_over_collected_cases() {
        let mut ir = guarded_dispatch(6);
        assert_eq!(case_targets(&ir).len(), 6, "table starts over-collected");

        let changed = SwitchBoundsPass.run(
            &mut ir,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true);

        assert!(changed, "the guard bounds the index at 2");
        assert_eq!(
            case_targets(&ir),
            vec![3, 4, 5],
            "cases 0..=2 survive, in table order"
        );
        assert_eq!(ir.validate(), Ok(()));
    }

    /// With no guard above it, nothing bounds the index and every case stays.
    ///
    /// The negative that proves the pruning is driven by the proof rather than
    /// by the table simply being long: this fixture has the same over-collected
    /// six-case table as the positive, minus the guard.
    #[test]
    fn an_unguarded_dispatch_keeps_every_case() {
        let mut ir = guarded_dispatch(6);
        // Replace the guard with an unconditional jump into the dispatch, so
        // the index reaches it unconstrained.
        if let Some(last) = ir
            .block_mut(0)
            .and_then(|block| block.instructions_mut().last_mut())
        {
            *last = SsaInstruction::synthetic(SsaOp::Jump { target: 2 });
        }
        ir.recompute_uses();

        let changed = SwitchBoundsPass.run(
            &mut ir,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true);
        assert!(!changed, "an unbounded index rules out no case");
        assert_eq!(case_targets(&ir).len(), 6);
    }

    /// A table already matching the proven bound is left alone - the pass only
    /// ever removes cases the range excludes.
    #[test]
    fn an_exactly_bounded_table_is_untouched() {
        let mut ir = guarded_dispatch(3);
        let changed = SwitchBoundsPass.run(
            &mut ir,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true);

        assert!(!changed, "nothing to prune");
        assert_eq!(case_targets(&ir), vec![3, 4, 5]);
        assert_eq!(ir.validate(), Ok(()));
    }
}
