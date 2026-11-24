//! Loops that compute nothing anything reads, and end.
//!
//! A loop whose trips write nothing, call nothing and read nothing that can
//! fail, and none of whose values anything past it reads, changes nothing by
//! how many times it runs: once is as good as all of them, where it ends at
//! all. A frame too large to probe in line is reserved by such a loop:
//! `-fstack-clash-protection`'s `lea r11, [rsp - 0x10000]`, then `sub rsp,
//! 0x1000; or qword [rsp], 0; cmp rsp, r11; jne` steps the stack pointer down
//! a page a trip to the bound and touches each page, which writes it back
//! unchanged. Past it the stack pointer is the bound
//! ([`crate::passes::equalities`]), so nothing reads what the loop computed,
//! and it printed as a walk of the stack pointer doing nothing.
//!
//! The loop is left after its first trip: its one way out, a test of
//! equality, is taken. That is sound where the loop ends, which is proved, not
//! assumed: the test compares a value stepped by a constant each trip with a
//! bound the loop does not change, both measured from one base
//! ([`AddressWalk::normalize`](crate::analysis::address::AddressWalk::normalize)), and the distance between them is a whole number of
//! steps the way the value moves. A loop the test would not end, or might not,
//! is kept: deleting it would make a program that runs for ever end.
//!
//! # Scope
//!
//! A loop with one latch, which is its one way out. Its reads are of the
//! function's own stack ([`FrameEscape::derived`]), which a frame the
//! function reserved holds, and neither volatile nor ordered; every other
//! operation computes a value and can neither trap nor write.

use crate::{
    Result,
    analysis::{
        address::{AddressBase, AddressWalk},
        cfg::SsaCfg,
        frame::FrameEscape,
        loops::{LoopInfo, detect_loops},
    },
    graph::{NodeId, algorithms::compute_dominators},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::{SsaOp, kinds::CmpKind},
        variable::SsaVarId,
    },
    pointer::PointerSize,
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// The block `info`'s loop leaves by and the block it leaves to, where the
/// loop computes nothing anything past it reads and ends.
fn idle_exit<T: Target>(
    ssa: &SsaFunction<T>,
    info: &LoopInfo,
    stack: Option<&FrameEscape>,
    ptr_size: PointerSize,
) -> Option<(usize, usize)> {
    let latch = info.single_latch()?.index();
    let mut exiting = info.exiting_blocks();
    if exiting.next()?.index() != latch || exiting.next().is_some() {
        return None;
    }
    let Some(SsaOp::BranchCmp {
        left,
        right,
        cmp,
        true_target,
        false_target,
        ..
    }) = ssa.block(latch)?.control_terminator()
    else {
        return None;
    };
    // The way out is taken where the two are equal.
    let exit = match cmp {
        CmpKind::Eq if !info.body.contains(*true_target) && info.body.contains(*false_target) => {
            *true_target
        }
        CmpKind::Ne if info.body.contains(*true_target) && !info.body.contains(*false_target) => {
            *false_target
        }
        _ => return None,
    };
    let inside = |block: usize| info.body.contains(block);
    for block in info.body.iter() {
        let block_ref = ssa.block(block)?;
        for instruction in block_ref.instructions() {
            let op = instruction.op();
            let reads_the_stack = matches!(
                op,
                SsaOp::LoadIndirect { addr, access, .. }
                    if !access.volatile
                        && stack.is_some_and(|stack| stack.derived().contains(addr))
            );
            let computes = op.effects().removable_when_unused() && !op.effects().may_throw;
            let controls = matches!(
                op,
                SsaOp::Jump { .. } | SsaOp::BranchCmp { .. } | SsaOp::Branch { .. }
            );
            if !(reads_the_stack || computes || controls) {
                return None;
            }
            // Nothing past the loop reads what it computes.
            if !op.defs().all(|def| read_inside(ssa, def, &inside)) {
                return None;
            }
        }
        if !block_ref
            .phi_nodes()
            .iter()
            .all(|phi| read_inside(ssa, phi.result(), &inside))
        {
            return None;
        }
    }
    ends(ssa, info, *left, *right, ptr_size).then_some((latch, exit))
}

/// Whether every read of `value` is in the loop `inside` names: by an
/// instruction of one of its blocks, or a phi of one.
fn read_inside<T: Target>(
    ssa: &SsaFunction<T>,
    value: SsaVarId,
    inside: &impl Fn(usize) -> bool,
) -> bool {
    ssa.variable(value)
        .is_none_or(|variable| variable.uses().iter().all(|site| inside(site.block())))
}

/// Whether the loop `info` ends at its test of `left` against `right`: one is
/// a value stepped by a constant each trip, the other a bound the loop does
/// not change, both measured from one base, and the value reaches the bound
/// in a whole number of steps the way it moves.
fn ends<T: Target>(
    ssa: &SsaFunction<T>,
    info: &LoopInfo,
    left: SsaVarId,
    right: SsaVarId,
    ptr_size: PointerSize,
) -> bool {
    let bits = ptr_size.bits();
    // `value` as a base and a constant displacement in bits, with no index.
    let measured = |value: SsaVarId| {
        let expr = AddressWalk::new(ssa, ptr_size).normalize(value);
        let AddressBase::Value(base) = expr.base else {
            return None;
        };
        expr.index.is_none().then_some((base, expr.offset_bits))
    };
    let defined_outside = |value: SsaVarId| {
        ssa.variable(value)
            .is_some_and(|variable| !info.body.contains(variable.def_site().block()))
    };
    let header = info.header.index();
    let reaches = |stepped: SsaVarId, bound: SsaVarId| -> Option<bool> {
        // The compared value is the header phi `cursor` displaced by `shown`.
        let (cursor, shown) = measured(stepped)?;
        let phi = ssa
            .block(header)?
            .phi_nodes()
            .iter()
            .find(|phi| phi.result() == cursor)?;
        let mut start = None;
        let mut step = None;
        for operand in phi.operands() {
            let (base, offset) = measured(operand.value())?;
            if info.body.contains(operand.predecessor()) {
                // Each trip moves the cursor by a constant.
                (base == cursor && step.replace(offset).is_none_or(|held| held == offset))
                    .then_some(())?;
            } else {
                (start
                    .replace((base, offset))
                    .is_none_or(|held| held == (base, offset)))
                .then_some(())?;
            }
        }
        let ((origin, from), step) = (start?, step?);
        // The bound does not change in the loop and is measured from where
        // the cursor starts.
        let (bound_base, to) = measured(bound)?;
        if !defined_outside(bound) || bound_base != origin || step == 0 {
            return Some(false);
        }
        // Trips `k >= 0` with `from + k * step + shown == to`, exactly.
        let distance = i128::from(to)
            .checked_sub(i128::from(from))?
            .checked_sub(i128::from(shown))?;
        let step = i128::from(step);
        let whole = distance.checked_rem(step) == Some(0);
        let trips = distance.checked_div(step).unwrap_or(-1);
        // No wrap: the trips the value takes stay within the width it holds.
        let within = trips
            .checked_mul(step.abs())
            .is_some_and(|span| span < (1_i128 << bits.min(126)));
        Some(whole && trips >= 0 && within)
    };
    reaches(left, right).unwrap_or(false) || reaches(right, left).unwrap_or(false)
}

/// Pass that leaves, after its first trip, a loop that computes nothing
/// anything past it reads and ends.
///
/// A loop whose trips write, call
/// and fail nothing, none of whose values is read past it, and whose one way
/// out is an equality its stepped value is proved to reach, runs once as
/// well as all its trips: a frame's stack-probe loop.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - folds the loop's exit test into a
/// jump out, removing its back edge.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdleLoopPass;

impl IdleLoopPass {
    /// Creates a new idle-loop pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for IdleLoopPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "idle-loop"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Leave a loop that computes nothing read past it and ends after one trip"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Leaves, after its first trip, every loop that computes nothing anything
    /// past it reads and ends.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` when a loop was left.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let ptr_size = env.context().machine().ptr_size;
        let cfg = SsaCfg::from_ssa(ssa);
        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let loops = detect_loops(&cfg, &dominators);
        let stack = ssa
            .entry_stack_pointer()
            .map(|entry| FrameEscape::of(ssa, entry));
        let mut folds: Vec<(usize, usize)> = Vec::new();
        for info in loops.by_depth_descending() {
            if let Some(fold) = idle_exit(ssa, info, stack.as_ref(), ptr_size) {
                folds.push(fold);
            }
        }
        if folds.is_empty() {
            return Ok(false);
        }
        let mut left = false;
        let result = ssa.edit(SsaEditOptions::new(), |editor| {
            for (exiting, exit) in &folds {
                editor.fold_terminator_pruning_phis(*exiting, SsaOp::Jump { target: *exit })?;
                left = true;
            }
            Ok(())
        });
        Ok(left && result.is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::NullListener,
        ir::{SsaDefSpec, SsaFunctionBuilder},
        testing::{MACHINE, MockTarget, MockType, run_mock_pass_boundary},
    };

    /// What a loop's body does besides stepping its cursor.
    #[derive(Clone, Copy, PartialEq)]
    enum Body {
        /// Nothing.
        Idle,
        /// Stores through the cursor.
        Stores,
        /// Its cursor is read past it.
        ReadPast,
    }

    /// A loop stepping `cursor` by -16 from argument 0 until it is that less
    /// `distance` (`exit` the comparison it leaves on), its body doing
    /// `body`. Blocks: 0 the entry, 1 the loop, 2 the exit.
    fn stepping(distance: i64, exit: CmpKind, body: Body) -> SsaFunction<MockTarget> {
        let def = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
        let (start, bound) = builder
            .in_block(0, |block| {
                let start = block.load_arg(def(), 0)?;
                let length = block.const_i64(def(), distance)?;
                let bound = block.sub(def(), start, length)?;
                block.jump(1)?;
                Ok((start, bound))
            })
            .expect("entry");
        let cursor = builder
            .in_block(1, |block| block.empty_phi(def()))
            .expect("phi");
        let stepped = builder
            .in_block(1, |block| {
                let sixteen = block.const_i64(def(), 16)?;
                let stepped = block.sub(def(), cursor, sixteen)?;
                if body == Body::Stores {
                    block.store_indirect(stepped, sixteen, MockType::I64)?;
                }
                let (round, out) = match exit {
                    CmpKind::Ne => (1, 2),
                    _ => (2, 1),
                };
                block.branch_cmp(stepped, bound, exit, false, round, out)?;
                Ok(stepped)
            })
            .expect("loop");
        builder
            .add_phi_operand(1, cursor, 0, start)
            .expect("entry operand");
        builder
            .add_phi_operand(1, cursor, 1, stepped)
            .expect("loop operand");
        builder
            .in_block(2, |block| {
                let returned = if body == Body::ReadPast {
                    stepped
                } else {
                    start
                };
                block.ret(Some(returned))?;
                Ok(())
            })
            .expect("exit");
        builder.finish().expect("finish")
    }

    /// Whether the loop is left after its first trip.
    fn left(distance: i64, exit: CmpKind, body: Body) -> bool {
        let mut ssa = stepping(distance, exit, body);
        run_mock_pass_boundary(&mut ssa, "idle", |ssa| {
            IdleLoopPass.run(
                ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        })
    }

    /// **A loop that computes nothing read past it, and ends, is left after
    /// one trip.** Stepping down 16 a trip to a bound 64 below where it
    /// starts, it ends on its fourth. The controls are kept: a bound no whole
    /// number of steps reaches (the loop never ends), one the cursor moves
    /// away from, a loop that writes, one whose cursor is read past it, and
    /// one leaving where the cursor is below the bound, which says nothing
    /// of where it ends.
    #[test]
    fn a_loop_that_does_nothing_and_ends_is_left() {
        assert!(left(64, CmpKind::Ne, Body::Idle), "the bound");
        assert!(!left(60, CmpKind::Ne, Body::Idle), "never reached");
        assert!(!left(-64, CmpKind::Ne, Body::Idle), "moving away");
        assert!(!left(64, CmpKind::Ne, Body::Stores), "a store");
        assert!(!left(64, CmpKind::Ne, Body::ReadPast), "read past it");
        assert!(!left(64, CmpKind::Lt, Body::Idle), "below the bound");
    }
}
