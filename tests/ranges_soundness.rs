//! Soundness of the value-range pass under a truncated fixpoint.
//!
//! The engine is SCCP-shaped: it narrows from `Top` while discovering which CFG
//! edges are executable, and its conclusions are only sound *at* the fixpoint.
//! Stopping early leaves an incomplete executable-edge set and ranges that are
//! too narrow — which is the unsound direction, because a too-narrow range
//! proves comparisons that are not actually provable.

use analyssa::{
    events::NullListener,
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::ranges,
    testing::{MockTarget, MockType},
};

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

fn var(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instruction: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instruction),
        MockType::I32,
    )
}

/// Builds the counted loop
///
/// ```text
/// b0: i0 = 0;                jump b1
/// b1: i = phi(i0:b0, i2:b2); c = i < 10; branch c ? b2 : b3
/// b2: i2 = i + 1;            jump b1
/// b3: return i
/// ```
///
/// The loop exit in `b1` is reachable — `i` runs 0..=10 — so nothing here may
/// be folded away.
fn counted_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 12);
    let start = var(&mut ssa, 0, 0, 0);
    let limit = var(&mut ssa, 1, 1, 0);
    let counter = var(&mut ssa, 2, 1, 0);
    let condition = var(&mut ssa, 3, 1, 1);
    let step = var(&mut ssa, 4, 2, 0);
    let next = var(&mut ssa, 5, 2, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: start,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(counter, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(start, 0));
    phi.add_operand(PhiOperand::new(next, 2));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(10),
    }));
    b1.add_instruction(instr(SsaOp::Clt {
        dest: condition,
        left: counter,
        right: limit,
        unsigned: false,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: step,
        value: ConstValue::I32(1),
    }));
    b2.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: counter,
        right: step,
        flags: None,
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Return {
        value: Some(counter),
    }));
    ssa.add_block(b3);

    ssa.recompute_uses();
    ssa
}

/// Returns `true` when block 1 still ends in a conditional branch.
fn exit_branch_survives(ssa: &SsaFunction<MockTarget>) -> bool {
    ssa.block(1)
        .and_then(|block| block.terminator_op())
        .is_some_and(|op| matches!(op, SsaOp::Branch { .. }))
}

/// A counted loop's exit must survive range propagation no matter how few
/// iterations the engine is given.
///
/// The loop induction variable really does reach the limit, so `i < 10` is not
/// a provable truth and the branch is not foldable. A truncated fixpoint that
/// still believes `i` is small would fold the exit away and turn the loop
/// infinite.
/// Returns `true` while block 1 still contains the `i < limit` comparison —
/// i.e. it has not been folded to a constant.
fn comparison_survives(ssa: &SsaFunction<MockTarget>) -> bool {
    ssa.block(1).is_some_and(|block| {
        block
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.op(), SsaOp::Clt { .. }))
    })
}

/// The loop's guard is genuinely not provable — `i` reaches the limit — so the
/// comparison must survive at every iteration budget.
///
/// Regression: the analysis is optimistic, so a truncated run held a range for
/// `i` that was too narrow and folded `i < 10` to a constant `true`. Because a
/// comparison result is modelled as `[0,1]` rather than a truth value, the
/// branch survived that first run — but the folded constant then let the *next*
/// pipeline iteration delete the loop exit, turning the loop infinite. At
/// `max_iterations = 8`, which is what the native lifter uses, this reproduced.
#[test]
fn truncated_fixpoint_does_not_prove_a_loop_guard() {
    for max_iterations in [1usize, 2, 3, 4, 6, 8, 12, 16, 32, 64, 256] {
        let mut ssa = counted_loop();
        ranges::run(&mut ssa, &0u32, &NullListener, max_iterations);
        assert!(
            comparison_survives(&ssa),
            "loop guard folded to a constant at max_iterations={max_iterations}"
        );
        assert_eq!(ssa.validate(), Ok(()));
    }
}

/// Running the pass twice — as the normalization fixpoint does — must not
/// delete the loop exit. This is the end of the miscompilation path above.
#[test]
fn repeated_runs_do_not_delete_the_loop_exit() {
    for max_iterations in [1usize, 4, 8, 64] {
        let mut ssa = counted_loop();
        for _ in 0..4 {
            ranges::run(&mut ssa, &0u32, &NullListener, max_iterations);
        }
        assert!(
            exit_branch_survives(&ssa),
            "loop exit deleted at max_iterations={max_iterations}"
        );
        assert_eq!(ssa.validate(), Ok(()));
    }
}

#[test]
fn truncated_fixpoint_does_not_fold_a_live_loop_exit() {
    for max_iterations in [1usize, 2, 4, 8, 16, 64] {
        let mut ssa = counted_loop();
        ranges::run(&mut ssa, &0u32, &NullListener, max_iterations);
        assert!(
            exit_branch_survives(&ssa),
            "loop exit folded away at max_iterations={max_iterations}"
        );
        assert_eq!(ssa.validate(), Ok(()));
    }
}

/// Builds `x = a + b; c = x < 0; branch c ? b1 : b2` with all values typed
/// `i32`, and reports whether the comparison survived.
fn add_then_compare_against_zero(a: i32, b: i32) -> bool {
    let mut ssa = SsaFunction::new(0, 10);
    let left = var(&mut ssa, 0, 0, 0);
    let right = var(&mut ssa, 1, 0, 1);
    let sum = var(&mut ssa, 2, 0, 2);
    let zero = var(&mut ssa, 3, 0, 3);
    let condition = var(&mut ssa, 4, 0, 4);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(a),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(b),
    }));
    b0.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left,
        right,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Clt {
        dest: condition,
        left: sum,
        right: zero,
        unsigned: false,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Return { value: Some(sum) }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Return { value: Some(sum) }));
    ssa.add_block(b2);

    ssa.recompute_uses();
    ranges::run(&mut ssa, &0u32, &NullListener, 64);
    assert_eq!(ssa.validate(), Ok(()));
    ssa.block(0).is_some_and(|block| {
        block
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.op(), SsaOp::Clt { .. }))
    })
}

/// Arithmetic that overflows its destination width proves nothing.
///
/// Interval arithmetic runs in `i64`, but a 32-bit `add` wraps:
/// `0x7fff_ffff + 1` is `-0x8000_0000`, not `0x8000_0000`. Reading the `i64`
/// result would conclude the sum is positive and fold `sum < 0` to `false` —
/// the opposite of the truth.
#[test]
fn overflowing_arithmetic_does_not_prove_a_sign() {
    assert!(
        add_then_compare_against_zero(i32::MAX, 1),
        "an add that overflows i32 must not prove the result's sign"
    );
}

/// The control: an add that stays in range still proves its sign, so the test
/// above is not passing merely because the analysis gave up everywhere.
#[test]
fn in_range_arithmetic_still_proves_a_sign() {
    assert!(
        !add_then_compare_against_zero(2, 3),
        "an in-range add still folds the comparison"
    );
}

/// Builds a guarded dispatch:
///
/// ```text
/// b0: idx = <opaque>; limit = 4; c = idx > limit; branch c ? b1 : b2
/// b1: return idx                      (out of range)
/// b2: return idx                      (guarded: idx <= 4)
/// ```
///
/// `idx` is an argument, so the whole-function range proves nothing about it.
/// Only the path through `b2` is bounded — the fact a switch dispatch needs.
fn guarded_dispatch() -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut ssa = SsaFunction::new(1, 8);
    let idx = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let limit = var(&mut ssa, 1, 0, 0);
    let condition = var(&mut ssa, 2, 0, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(4),
    }));
    b0.add_instruction(instr(SsaOp::Cgt {
        dest: condition,
        left: idx,
        right: limit,
        unsigned: false,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Return { value: Some(idx) }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Return { value: Some(idx) }));
    ssa.add_block(b2);

    ssa.recompute_uses();
    (ssa, idx)
}

/// A dominating guard bounds the value on the arm it admits, and only there.
///
/// The whole-function range for an argument is `Top`; the bound exists only on
/// the guarded path. Without this refinement a switch index carries no bound at
/// all, which is why jump-table recovery cannot be driven from the
/// whole-function range alone.
#[test]
fn a_dominating_guard_bounds_the_value_on_its_arm() {
    use analyssa::{
        analysis::SsaCfg,
        graph::{RootedGraph, algorithms::compute_dominators},
        passes::ranges::analyze,
    };

    let (ssa, idx) = guarded_dispatch();
    let cfg = SsaCfg::from_ssa(&ssa);
    let dominators = compute_dominators(&cfg, cfg.entry());
    let converged = analyze(&ssa, 64);
    assert!(
        converged.is_some(),
        "the analysis converges on a diamond this small"
    );
    let Some(ranges) = converged else { return };

    // Whole-function: an argument is unconstrained.
    let global = ranges.get(idx).cloned().unwrap_or_default();
    assert!(global.max().is_none(), "argument is unbounded overall");

    // On the guarded arm `idx > 4` was false, so `idx <= 4`.
    let guarded = ranges.range_at(&ssa, &dominators, idx, 2);
    assert_eq!(
        guarded.max(),
        Some(4),
        "the false arm of `idx > 4` bounds idx at 4"
    );

    // On the other arm `idx > 4` held, so `idx >= 5`.
    let out_of_range = ranges.range_at(&ssa, &dominators, idx, 1);
    assert_eq!(
        out_of_range.min(),
        Some(5),
        "the true arm of `idx > 4` puts idx above the limit"
    );

    // The entry block is before the guard: nothing is proved there.
    let at_entry = ranges.range_at(&ssa, &dominators, idx, 0);
    assert!(
        at_entry.max().is_none(),
        "the guard proves nothing above itself"
    );
}

/// A range analysis driven by the *generic* dataflow solver terminates, which
/// it cannot without the widening hook.
///
/// The solver has no iteration cap — it drains its worklist and relies on the
/// lattice having finite height. `ValueRange` does not: a counted loop grows
/// its interval by one per trip, so an exact solver runs once per trip and a
/// genuinely unbounded loop never finishes. `widen` is what bounds it.
#[test]
fn the_solver_widening_hook_terminates_an_interval_analysis() {
    use analyssa::{
        analysis::{
            SsaCfg,
            dataflow::{DataFlowAnalysis, DataFlowSolver, Direction},
            range::ValueRange,
        },
        ir::{block::SsaBlock as Block, function::SsaFunction as Func},
    };

    /// Counts up by one per block visit — an intentionally unbounded ascent.
    struct Countingnalysis {
        widen_after: usize,
    }

    impl DataFlowAnalysis<MockTarget> for Countingnalysis {
        type Lattice = ValueRange;
        const DIRECTION: Direction = Direction::Forward;

        fn boundary(&self, _ssa: &Func<MockTarget>) -> ValueRange {
            ValueRange::constant(0)
        }

        fn initial(&self, _ssa: &Func<MockTarget>) -> ValueRange {
            ValueRange::bottom()
        }

        fn transfer(
            &self,
            _block_id: usize,
            _block: &Block<MockTarget>,
            input: &ValueRange,
            _ssa: &Func<MockTarget>,
        ) -> ValueRange {
            // Every pass shifts the interval up by one; nothing converges.
            input.add(&ValueRange::constant(1))
        }

        fn widen(
            &self,
            _block_id: usize,
            previous: &ValueRange,
            next: ValueRange,
            visit: usize,
        ) -> ValueRange {
            if visit > self.widen_after {
                previous.widen(&next)
            } else {
                next
            }
        }
    }

    let ssa = counted_loop();
    let cfg = SsaCfg::from_ssa(&ssa);
    let analysis = Countingnalysis { widen_after: 3 };
    let solver = DataFlowSolver::new(analysis);

    // Without widening this call does not return: the transfer function climbs
    // by one every visit and the loop's back edge keeps re-queueing the header.
    // Reaching the assertions at all is half the result.
    let results = solver.solve(&ssa, &cfg);
    assert_eq!(
        results.in_states.len(),
        ssa.block_count(),
        "the solver reached a fixpoint and produced one state per block"
    );
    // ...and the other half: widening is what stopped it, so some state must
    // have been pushed to an infinite bound rather than converging on its own.
    assert!(
        results
            .out_states
            .iter()
            .any(|state| state.max().is_none() || state.is_top()),
        "termination came from widening, not from the ascent settling"
    );
}
