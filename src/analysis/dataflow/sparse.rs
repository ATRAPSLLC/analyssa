//! A worklist solver for problems with a sparse lane and a dense lane.
//!
//! Some analyses keep two kinds of fact at once. Facts about SSA values live
//! in one environment for the whole function and flow along def-use edges: a
//! value whose fact changed matters only to the blocks that read it. Facts
//! about memory hold at a program point and flow along control-flow edges: a
//! block whose outgoing state changed matters to its successors. Type
//! recovery over a lifted function is one such analysis - a value's type
//! meets what every use says, while what a store left in a stack cell is true
//! only after the store.
//!
//! [`solve_sparse_dense`] drives both lanes from one worklist ordered by
//! reverse postorder of the exception-aware CFG, so each block is visited after
//! its forward predecessors whenever it can be. The host's
//! [`SparseDenseProblem`] transfers a block, merges the states its
//! predecessors leave, and names the values whose facts changed; the solver
//! decides what runs next.
//!
//! # Termination
//!
//! A sparse lane that only descends a finite lattice settles under any fair
//! schedule. The dense lane may be gen/kill rather than monotone (a call that
//! clobbers cells removes facts), so the solver bounds its visits at 64 per
//! block plus 64 and reports a run that reached the bound as not converged
//! rather than looping.

use std::{cmp::Reverse, collections::BinaryHeap};

use rustc_hash::FxHashMap;

use crate::{
    analysis::{cfg::SsaCfg, context::AnalysisContext},
    graph::{NodeId, algorithms::reverse_postorder_all},
    interrupt::Interrupted,
    ir::{SsaFunction, variable::SsaVarId},
    target::Target,
};

/// What [`solve_sparse_dense`] needs from an analysis.
pub trait SparseDenseProblem<T: Target> {
    /// What the dense lane knows at a program point.
    type State: PartialEq;

    /// The state a block entered from several predecessors starts with: the
    /// merge of the states they leave. Only predecessors visited so far take
    /// part; an unvisited one says nothing yet.
    ///
    /// # Arguments
    ///
    /// * `leaving` - The states the visited predecessors leave with.
    fn merge<'a>(&self, leaving: impl Iterator<Item = &'a Self::State>) -> Self::State
    where
        Self::State: 'a;

    /// The state a handler entry starts with.
    ///
    /// A handler is entered from wherever in its protected region a throw
    /// happened, with memory as that point left it rather than as any block
    /// leaves it, so it enters knowing what holds everywhere in the region,
    /// whatever ordinary edges also reach it.
    fn exceptional_entry(&self) -> Self::State;

    /// Transfers block `block` from `entry`.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function.
    /// * `block` - The block to transfer.
    /// * `entry` - The state the block starts with.
    /// * `changed` - Receives every value whose sparse fact the block changed.
    ///
    /// # Returns
    ///
    /// The state the block leaves with.
    fn transfer(
        &mut self,
        ssa: &SsaFunction<T>,
        block: usize,
        entry: Self::State,
        changed: &mut Vec<SsaVarId>,
    ) -> Self::State;
}

/// How a [`solve_sparse_dense`] run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseDenseRun {
    /// Block visits made.
    pub steps: usize,
    /// `false` when the run stopped at its bound rather than at a fixpoint.
    pub reached_fixpoint: bool,
}

/// The parts of a [`solve_sparse_dense`] run that depend on the function alone:
/// its control-flow adjacency, the order blocks are visited in, and which
/// blocks read or define each value.
///
/// A host that solves the same function repeatedly - type recovery re-run as
/// the seeds it is given move - derives this once and passes it to
/// [`solve_sparse_dense`] each time. Building it is a walk over every
/// operation of the function, which is a good part of what a solve costs on a
/// function that settles in a few visits.
#[derive(Debug, Clone)]
pub struct SparseDenseSchedule {
    /// Successors of each block, exception edges included.
    successors: Vec<Vec<usize>>,
    /// Predecessors of each block, exception edges included.
    predecessors: Vec<Vec<usize>>,
    /// Position of each block in reverse postorder: the order the worklist
    /// drains in.
    rank: Vec<usize>,
    /// Blocks in reverse postorder.
    rpo: Vec<usize>,
    /// Whether each block is a handler entry, reached by the runtime rather
    /// than by a terminator.
    handler_entry: Vec<bool>,
    /// The blocks that read or define each value.
    users: FxHashMap<SsaVarId, Vec<usize>>,
}

impl SparseDenseSchedule {
    /// Derives the schedule of `ssa`.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let block_count = ssa.blocks().len();
        // The exception-aware adjacency: a handler is reached by the runtime
        // rather than by a terminator, and is scheduled and re-entered like any
        // other block.
        let cfg = SsaCfg::from_ssa(ssa);
        let successors: Vec<Vec<usize>> = cfg.to_successor_lists();
        let rpo: Vec<usize> = reverse_postorder_all(&cfg)
            .into_iter()
            .map(NodeId::index)
            .collect();
        let mut rank = vec![0usize; block_count];
        for (position, &block) in rpo.iter().enumerate() {
            if let Some(slot) = rank.get_mut(block) {
                *slot = position;
            }
        }
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); block_count];
        for (block, succs) in successors.iter().enumerate() {
            for &succ in succs {
                if let Some(preds) = predecessors.get_mut(succ) {
                    preds.push(block);
                }
            }
        }
        let handler_entry: Vec<bool> = (0..block_count)
            .map(|block| {
                cfg.predecessor_blocks(block)
                    .any(|pred| cfg.is_exception_edge(pred, block))
            })
            .collect();
        Self {
            successors,
            predecessors,
            rank,
            rpo,
            handler_entry,
            users: value_users(ssa),
        }
    }
}

/// Drives `problem` over `ssa` to a fixpoint.
///
/// Every block is visited once in reverse postorder, then again whenever a
/// value it reads or defines changed (the sparse lane: a rule may refine an
/// operation's operands from its result, so a block is a user of the values it
/// defines too), or a predecessor's outgoing state changed (the dense lane).
/// A block waits in the worklist at most once, ordered by its rank.
///
/// `schedule` is the function's [`SparseDenseSchedule`]
/// ([`SparseDenseSchedule::of`] this same `ssa`); a host that solves one
/// function more than once derives it once and keeps it.
///
/// The run asks `context`'s interrupt before every visit: a large function's
/// run is thousands of them, seconds of work inside this one call.
///
/// # Returns
///
/// The number of visits and whether the run reached a fixpoint within its
/// bound.
///
/// # Errors
///
/// [`Interrupted`] when the interrupt trips before the run ends; what the
/// problem holds then is half done.
pub fn solve_sparse_dense<T, P>(
    ssa: &SsaFunction<T>,
    schedule: &SparseDenseSchedule,
    problem: &mut P,
    context: &AnalysisContext<'_>,
) -> Result<SparseDenseRun, Interrupted>
where
    T: Target,
    P: SparseDenseProblem<T>,
{
    let block_count = ssa.blocks().len();
    if block_count == 0 {
        return Ok(SparseDenseRun {
            steps: 0,
            reached_fixpoint: true,
        });
    }
    let SparseDenseSchedule {
        successors,
        predecessors,
        rank,
        rpo,
        handler_entry,
        users,
    } = schedule;

    let mut leaving: Vec<Option<P::State>> = (0..block_count).map(|_| None).collect();
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::with_capacity(block_count);
    let mut queued = vec![false; block_count];
    for &block in rpo {
        enqueue(block, rank, &mut heap, &mut queued);
    }
    let interrupt = context.interrupt();
    let bound = block_count.saturating_mul(64).saturating_add(64);
    let mut steps = 0usize;
    let mut reached_fixpoint = true;
    let mut changed: Vec<SsaVarId> = Vec::new();

    while let Some(Reverse((_, block))) = heap.pop() {
        match queued.get_mut(block) {
            Some(flag) if *flag => *flag = false,
            _ => continue,
        }
        interrupt.check()?;
        steps = steps.saturating_add(1);
        if steps > bound {
            reached_fixpoint = false;
            break;
        }
        let entry = if handler_entry.get(block).copied().unwrap_or(false) {
            problem.exceptional_entry()
        } else {
            problem.merge(
                predecessors
                    .get(block)
                    .into_iter()
                    .flatten()
                    .filter_map(|&pred| leaving.get(pred).and_then(Option::as_ref)),
            )
        };
        changed.clear();
        let left = problem.transfer(ssa, block, entry, &mut changed);
        let state_changed = match leaving.get_mut(block) {
            Some(slot) if slot.as_ref() != Some(&left) => {
                *slot = Some(left);
                true
            }
            _ => false,
        };
        for value in &changed {
            for &user in users.get(value).into_iter().flatten() {
                enqueue(user, rank, &mut heap, &mut queued);
            }
        }
        if state_changed {
            for &succ in successors.get(block).into_iter().flatten() {
                enqueue(succ, rank, &mut heap, &mut queued);
            }
        }
    }
    Ok(SparseDenseRun {
        steps,
        reached_fixpoint,
    })
}

/// Queues `block` at its rank unless it is already waiting, so the heap holds
/// at most one entry per block.
fn enqueue(
    block: usize,
    rank: &[usize],
    heap: &mut BinaryHeap<Reverse<(usize, usize)>>,
    queued: &mut [bool],
) {
    if let Some(flag) = queued.get_mut(block)
        && !*flag
        && let Some(&at) = rank.get(block)
    {
        *flag = true;
        heap.push(Reverse((at, block)));
    }
}

/// For each value, the blocks that read it - through a phi operand or an
/// instruction operand - or define it with an instruction, ascending and once
/// each.
fn value_users<T: Target>(ssa: &SsaFunction<T>) -> FxHashMap<SsaVarId, Vec<usize>> {
    let mut users: FxHashMap<SsaVarId, Vec<usize>> = FxHashMap::default();
    for (block, contents) in ssa.blocks().iter().enumerate() {
        for phi in contents.phi_nodes() {
            for operand in phi.operands() {
                users.entry(operand.value()).or_default().push(block);
            }
        }
        for instruction in contents.instructions() {
            let op = instruction.op();
            for used in op.uses() {
                users.entry(used).or_default().push(block);
            }
            for defined in op.defs() {
                users.entry(defined).or_default().push(block);
            }
        }
    }
    for blocks in users.values_mut() {
        blocks.sort_unstable();
        blocks.dedup();
    }
    users
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{
        interrupt::Interrupt,
        ir::{
            SsaBlock, SsaInstruction,
            exception::{BlockRange, SsaExceptionHandler},
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// Records what the solver asks of it. The state a block leaves with is
    /// the number of times it was visited when `growing`, so it never settles,
    /// and the entry state otherwise.
    #[derive(Default)]
    struct Probe {
        visits: Vec<usize>,
        entries: Vec<(usize, u32)>,
        changes: Vec<(usize, SsaVarId)>,
        growing: bool,
    }

    impl SparseDenseProblem<MockTarget> for Probe {
        type State = u32;

        fn merge<'a>(&self, leaving: impl Iterator<Item = &'a u32>) -> u32 {
            leaving.copied().max().unwrap_or(0)
        }

        fn exceptional_entry(&self) -> u32 {
            u32::MAX
        }

        fn transfer(
            &mut self,
            _ssa: &SsaFunction<MockTarget>,
            block: usize,
            entry: u32,
            changed: &mut Vec<SsaVarId>,
        ) -> u32 {
            self.visits.push(block);
            self.entries.push((block, entry));
            if let Some(at) = self.changes.iter().position(|(at, _)| *at == block) {
                changed.push(self.changes.remove(at).1);
            }
            if self.growing {
                u32::try_from(self.visits.len()).unwrap_or(u32::MAX)
            } else {
                entry
            }
        }
    }

    /// `B0: v = 1; jump B1. B1: jump B2. B2: ret v`.
    fn chain() -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let v = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v))));
        ssa.add_block(b2);
        ssa.recompute_uses();
        (ssa, v)
    }

    /// With nothing changing, every block is visited once, in reverse
    /// postorder, and each enters with what its predecessor left.
    #[test]
    fn a_quiet_problem_visits_each_block_once_in_order() {
        let (ssa, _) = chain();
        let mut probe = Probe::default();
        let run = solve_sparse_dense(
            &ssa,
            &SparseDenseSchedule::of(&ssa),
            &mut probe,
            &AnalysisContext::new(MACHINE),
        )
        .expect("never interrupted");
        assert_eq!(
            run,
            SparseDenseRun {
                steps: 3,
                reached_fixpoint: true
            }
        );
        assert_eq!(probe.visits, [0, 1, 2]);
    }

    /// One schedule drives every run over the function, and each run visits
    /// what a run that derived its own schedule does.
    #[test]
    fn a_schedule_serves_every_run_over_its_function() {
        let (ssa, v) = chain();
        let schedule = SparseDenseSchedule::of(&ssa);
        let mut alone = Probe::default();
        alone.changes.push((2, v));
        let _ = solve_sparse_dense(
            &ssa,
            &SparseDenseSchedule::of(&ssa),
            &mut alone,
            &AnalysisContext::new(MACHINE),
        );
        for _ in 0..2 {
            let mut scheduled = Probe::default();
            scheduled.changes.push((2, v));
            let run = solve_sparse_dense(
                &ssa,
                &schedule,
                &mut scheduled,
                &AnalysisContext::new(MACHINE),
            )
            .expect("never interrupted");
            assert!(run.reached_fixpoint);
            assert_eq!(scheduled.visits, alone.visits);
        }
    }

    /// A changed value re-enqueues the blocks that read or define it, and only
    /// those.
    #[test]
    fn a_changed_value_revisits_its_readers() {
        let (ssa, v) = chain();
        let mut probe = Probe::default();
        // B2 changes v's fact: B0, which defines it, and B2, which reads it,
        // run again; B1 does not.
        probe.changes.push((2, v));
        let run = solve_sparse_dense(
            &ssa,
            &SparseDenseSchedule::of(&ssa),
            &mut probe,
            &AnalysisContext::new(MACHINE),
        )
        .expect("never interrupted");
        assert!(run.reached_fixpoint);
        assert_eq!(probe.visits, [0, 1, 2, 0, 2]);
    }

    /// A state that never settles stops at the bound, reported as not
    /// converged rather than looping.
    #[test]
    fn a_state_that_never_settles_stops_at_the_bound() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 0 }));
        ssa.add_block(b0);
        let mut probe = Probe {
            growing: true,
            ..Probe::default()
        };
        let run = solve_sparse_dense(
            &ssa,
            &SparseDenseSchedule::of(&ssa),
            &mut probe,
            &AnalysisContext::new(MACHINE),
        )
        .expect("never interrupted");
        assert!(!run.reached_fixpoint);
        assert_eq!(run.steps, 64 + 64 + 1);
    }

    /// Trips once it has been asked `left` times.
    struct After {
        /// How many more asks it answers no to.
        left: AtomicUsize,
    }

    impl Interrupt for After {
        fn interrupted(&self) -> bool {
            let mut left = self.left.load(Ordering::Relaxed);
            loop {
                let Some(next) = left.checked_sub(1) else {
                    return true;
                };
                match self.left.compare_exchange_weak(
                    left,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return false,
                    Err(now) => left = now,
                }
            }
        }
    }

    /// **A run that never settles stops when it is told to**, before its
    /// bound, and says it was stopped rather than that it ended: every visit
    /// asks first.
    #[test]
    fn a_run_stops_at_the_visit_it_is_told_to() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 0 }));
        ssa.add_block(b0);
        let mut probe = Probe {
            growing: true,
            ..Probe::default()
        };
        let interrupt = After {
            left: AtomicUsize::new(10),
        };
        let context = AnalysisContext::new(MACHINE).with_interrupt(&interrupt);
        let run = solve_sparse_dense(&ssa, &SparseDenseSchedule::of(&ssa), &mut probe, &context);
        assert_eq!(run, Err(Interrupted));
        assert_eq!(probe.visits.len(), 10, "ten visits, then the stop");
    }

    /// A handler entry starts from the exceptional state, not from what its
    /// protected block leaves.
    #[test]
    fn a_handler_enters_with_the_exceptional_state() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        for (index, op) in [
            SsaOp::Leave { target: 2 },
            SsaOp::Leave { target: 2 },
            SsaOp::returns(None),
        ]
        .into_iter()
        .enumerate()
        {
            let mut block = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(op));
            ssa.add_block(block);
        }
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: Default::default(),
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 1),
            handler_range: BlockRange::new(1, 2),
            filter_range: None,
            selector: None,
        }]);
        let mut probe = Probe::default();
        solve_sparse_dense(
            &ssa,
            &SparseDenseSchedule::of(&ssa),
            &mut probe,
            &AnalysisContext::new(MACHINE),
        )
        .expect("never interrupted");
        assert!(
            probe.entries.contains(&(1, u32::MAX)),
            "{:?}",
            probe.entries
        );
    }
}
