//! Rejoining a loop a compiler split on a guard: one loop whose trip takes
//! either body.
//!
//! A loop testing a guard each trip that, once false, stays false is split
//! by an optimiser where the guard turns: the first loop runs while the
//! guard holds, and its test leaves for a second loop, which goes on with
//! what the first had counted and never tests the guard again. GCC's `-O2`
//! reads a `va_list` that way: a register-area loop while `gp_offset <= 47`,
//! then a stack loop.
//!
//! The source wrote one loop, and the split is a fact of the guard, not of
//! the program: [`LoopRejoiningPass`] rejoins the second loop into the first, its latches
//! going back to the first loop's header, whose test, re-taken, finds the
//! guard as the second loop left it. The second loop never changes what the
//! test reads, so the test sends every later trip to the second body again,
//! and each value keeps its meaning.
//!
//! A pair is rejoined only where that holds by construction:
//!
//! - **The test leaves for the second loop alone**, along a path of blocks
//!   each entered from the one before, computing values only.
//! - **The second loop is entered from that path and its own latches alone**,
//!   and holds no block of the first loop.
//! - **Each of its phis continues one of the header's**, the value the header
//!   had when its test left, or takes a value from before the loop.
//! - **The header and the path compute values only, none from a value the
//!   second loop changes**: re-taken after a second trip, they compute what
//!   they computed when the test left.

use std::{collections::BTreeSet, iter};

use crate::{
    Result,
    analysis::{
        cfg::SsaCfg,
        loops::{LoopForest, LoopInfo, detect_loops},
    },
    events::EventKind,
    graph::{
        NodeId, Predecessors,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        ops::SsaOp,
        variable::SsaVarId,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// A loop split from another at its header's test.
#[derive(Debug)]
struct Split {
    /// The first loop's header, whose test leaves for the second.
    header: usize,
    /// The second loop's header.
    second: usize,
    /// The header's predecessors from before the loop.
    entering: Vec<usize>,
    /// The first loop's latches.
    first_latches: Vec<usize>,
    /// The second loop's latches.
    second_latches: Vec<usize>,
    /// For each of the second header's phis, the header phi it continues, or
    /// `None` for one taking a value from before the loop.
    continues: Vec<Option<SsaVarId>>,
}

impl Split {
    /// The first split loop of `ssa`, where there is one.
    fn find<T: Target>(ssa: &SsaFunction<T>) -> Option<Self> {
        let cfg = SsaCfg::from_ssa(ssa);
        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let loops = detect_loops(&cfg, &dominators);
        loops
            .iter()
            .find_map(|first| Self::of(ssa, &cfg, &dominators, &loops, first))
    }

    /// The second loop `first`'s header test leaves for, where it was split
    /// from `first` (the module's rules).
    fn of<T: Target>(
        ssa: &SsaFunction<T>,
        cfg: &SsaCfg<'_, T>,
        dominators: &DominatorTree,
        loops: &LoopForest,
        first: &LoopInfo,
    ) -> Option<Self> {
        let header = first.header.index();
        let block = ssa.block(header)?;
        let (taken, other) = match block.control_terminator()? {
            SsaOp::Branch {
                true_target,
                false_target,
                ..
            }
            | SsaOp::BranchCmp {
                true_target,
                false_target,
                ..
            } => (*true_target, *false_target),
            _ => return None,
        };
        let inside = |at: usize| first.contains(NodeId::new(at));
        let out = match (inside(taken), inside(other)) {
            (true, false) => other,
            (false, true) => taken,
            _ => return None,
        };
        let predecessors = |at: usize| -> BTreeSet<usize> {
            cfg.predecessors(NodeId::new(at))
                .map(NodeId::index)
                .collect()
        };
        // The path from the test to the second loop: blocks each entered from
        // the one before alone, holding no phi and computing values only.
        let mut path: Vec<usize> = Vec::new();
        let mut at = out;
        let second = loop {
            if path.len() > ssa.block_count() {
                return None;
            }
            if let Some(second) = loops.loop_for_header(NodeId::new(at)) {
                break second;
            }
            let block = ssa.block(at)?;
            let before = path.last().copied().unwrap_or(header);
            let SsaOp::Jump { target } = block.control_terminator()? else {
                return None;
            };
            if predecessors(at) != BTreeSet::from([before]) || !block.phi_nodes().is_empty() {
                return None;
            }
            path.push(at);
            at = *target;
        };
        let second_header = second.header.index();
        if second.contains(first.header)
            || (0..ssa.block_count())
                .any(|at| first.contains(NodeId::new(at)) && second.contains(NodeId::new(at)))
        {
            return None;
        }
        let last = path.last().copied().unwrap_or(header);
        let second_latches: Vec<usize> = second.latches.iter().map(|latch| latch.index()).collect();
        let mut expected: BTreeSet<usize> = second_latches.iter().copied().collect();
        expected.insert(last);
        if predecessors(second_header) != expected {
            return None;
        }
        // Each second-loop phi continues a header phi, once, or takes a value
        // from before the loop.
        let header_phis: BTreeSet<SsaVarId> =
            block.phi_nodes().iter().map(|phi| phi.result()).collect();
        let mut continues = Vec::new();
        let mut continued: BTreeSet<SsaVarId> = BTreeSet::new();
        for phi in ssa.block(second_header)?.phi_nodes() {
            let entering = phi.operand_from(last)?.value();
            if header_phis.contains(&entering) {
                if !continued.insert(entering) {
                    return None;
                }
                continues.push(Some(entering));
                continue;
            }
            let defined = ssa.variable(entering)?.def_site().block();
            if !dominators.strictly_dominates(NodeId::new(defined), first.header) {
                return None;
            }
            continues.push(None);
        }
        // The header and the path compute values only, and none from what
        // the second loop changes, the test included: re-taken after a
        // second trip, each computes what it computed when the test left.
        for at in iter::once(header).chain(path.iter().copied()) {
            let rows = ssa.block(at)?.instructions();
            let (terminator, body) = rows.split_last()?;
            let reads_changed =
                |op: &SsaOp<T>| op.uses().iter().any(|used| continued.contains(used));
            if body
                .iter()
                .any(|row| !row.op().is_pure() || reads_changed(row.op()))
                || reads_changed(terminator.op())
            {
                return None;
            }
        }
        let entering: Vec<usize> = predecessors(header)
            .into_iter()
            .filter(|pred| !inside(*pred))
            .collect();
        let first_latches: Vec<usize> = first.latches.iter().map(|latch| latch.index()).collect();
        Some(Self {
            header,
            second: second_header,
            entering,
            first_latches,
            second_latches,
            continues,
        })
    }
}

/// Pass that rejoins a loop a compiler split on a guard into the loop it was
/// split from.
///
/// A loop whose header's test
/// leaves for a second loop that changes nothing the test reads takes the
/// second loop's body as the other arm of its trip, the second loop's
/// latches going back to its header.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - retargets the second loop's latches
/// and moves its phis into the first loop's header.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoopRejoiningPass;

impl LoopRejoiningPass {
    /// Creates a new loop-rejoining pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for LoopRejoiningPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "loop-rejoining"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Rejoin a loop a compiler split on a guard into the loop it was split from"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Rejoins every loop a compiler split on a guard into the loop it was split
    /// from.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// Whether any loop was rejoined.
    ///
    /// # Errors
    ///
    /// Returns an error if an edit session fails.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let mut rejoined = false;
        // Each rejoin leaves one loop fewer, so there are at most as many as
        // blocks.
        for _ in 0..ssa.block_count() {
            let Some(split) = Split::find(ssa) else {
                break;
            };
            ssa.edit(
                SsaEditOptions::new()
                    .with_verify(true)
                    .with_rollback(SsaRollbackPolicy::OnFailure),
                |editor| {
                    editor.rejoin_split_loop(
                        split.header,
                        split.second,
                        &split.entering,
                        &split.first_latches,
                        &split.second_latches,
                        &split.continues,
                    )?;
                    editor.declare_repaired();
                    Ok(())
                },
            )?;
            env.record(EventKind::ControlFlowRestructured)
                .location(split.header)
                .message(format!(
                    "rejoined: B{}'s loop, split at its test, goes on in B{}'s",
                    split.second, split.header
                ));
            rejoined = true;
        }
        Ok(rejoined)
    }
}
