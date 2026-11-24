//! A call its host described does what it says and nothing else.
//!
//! Each case runs the same program with the call's effects stated one way and
//! then left unstated - the control, which must behave as every call did before
//! a call could carry its effects: a barrier nothing crosses, kept whatever its
//! result.

use analyssa::{
    analysis::{
        AnalysisContext,
        budget::{Budget, WorkBudget},
        evaluator::{EvaluatorConfig, SsaEvaluator},
    },
    events::NullListener,
    ir::{
        CallData, Callee,
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{
            AccessMode, AccessQualifiers, CallEffects, CallLocation, Capture, PointeeExtent, SsaOp,
        },
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::{DeadCodeEliminationPass, MemoryOptimizationPass},
    scheduling::{FunctionPass, PassEnv},
    testing::{MACHINE, MockTarget, MockType},
};

/// The field the program stores to and reloads.
const HELD: u32 = 1;
/// A field the program never touches.
const OTHER: u32 = 2;

type Effects = CallEffects<MockTarget>;

fn var(ssa: &mut SsaFunction<MockTarget>, idx: u16, instr: usize, ty: MockType) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(0, instr),
        ty,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

/// A call to method 9 with `args`, carrying `effects`.
fn call(args: Vec<SsaVarId>, effects: Option<Effects>) -> SsaInstruction<MockTarget> {
    instr({
        let mut call = CallData::new(None, Callee::Direct(9), args);
        call.effects = effects;
        SsaOp::from(call)
    })
}

/// Counts the operations `keep` selects.
fn count(ssa: &SsaFunction<MockTarget>, keep: impl Fn(&SsaOp<MockTarget>) -> bool) -> usize {
    ssa.blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter(|instruction| keep(instruction.op()))
        .count()
}

/// `HELD = 7; call; w = HELD; return w`, run through the memory pass, and
/// whether the reload was forwarded the stored 7.
fn forwards_across(effects: Option<Effects>) -> bool {
    let mut ssa = SsaFunction::new(0, 8);
    let seven = var(&mut ssa, 0, 0, MockType::I32);
    let reloaded = var(&mut ssa, 1, 3, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: seven,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::StoreStaticField {
        access: AccessQualifiers::FLAT,
        field: HELD,
        value: seven,
    }));
    b0.add_instruction(call(Vec::new(), effects));
    b0.add_instruction(instr(SsaOp::LoadStaticField {
        access: AccessQualifiers::FLAT,
        dest: reloaded,
        field: HELD,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(reloaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(ssa.validate(), Ok(()));
    count(&ssa, |op| matches!(op, SsaOp::LoadStaticField { .. })) == 0
}

/// **A value crosses a call that cannot touch it, and only such a call.**
#[test]
fn a_value_crosses_only_a_call_that_leaves_it() {
    assert!(
        !forwards_across(None),
        "an undescribed call may write anything"
    );
    assert!(
        forwards_across(Some(Effects::pure())),
        "a pure call touches nothing"
    );
    let elsewhere = Effects::pure().with_access(CallLocation::Field(OTHER), AccessMode::Write);
    assert!(forwards_across(Some(elsewhere)), "it writes another field");
    let reads = Effects::pure().with_access(CallLocation::Field(HELD), AccessMode::Read);
    assert!(forwards_across(Some(reads)), "reading the field leaves it");

    let writes = Effects::pure().with_access(CallLocation::Field(HELD), AccessMode::MayWrite);
    assert!(!forwards_across(Some(writes)), "it may write the field");
    let escaped = Effects::pure().with_access(CallLocation::Escaped, AccessMode::Update);
    assert!(
        !forwards_across(Some(escaped)),
        "escaped memory may be the field"
    );
    let mut calls_back = Effects::pure();
    calls_back.calls_back = true;
    assert!(
        !forwards_across(Some(calls_back)),
        "code it runs may write anything"
    );
}

/// `HELD = 1; call; HELD = 2; return`, run through the memory pass, and whether
/// the first store survived.
fn keeps_the_overwritten_store(effects: Option<Effects>) -> bool {
    let mut ssa = SsaFunction::new(0, 8);
    let one = var(&mut ssa, 0, 0, MockType::I32);
    let two = var(&mut ssa, 1, 1, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: two,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(instr(SsaOp::StoreStaticField {
        access: AccessQualifiers::FLAT,
        field: HELD,
        value: one,
    }));
    b0.add_instruction(call(Vec::new(), effects));
    b0.add_instruction(instr(SsaOp::StoreStaticField {
        access: AccessQualifiers::FLAT,
        field: HELD,
        value: two,
    }));
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(ssa.validate(), Ok(()));
    count(&ssa, |op| matches!(op, SsaOp::StoreStaticField { .. })) == 2
}

/// **A store a call reads is live; one no call reads and a later store
/// overwrites is dead.**
#[test]
fn a_store_lives_if_the_call_reads_it() {
    assert!(
        keeps_the_overwritten_store(None),
        "an undescribed call may read it"
    );
    let reads = Effects::pure().with_access(CallLocation::Field(HELD), AccessMode::Read);
    assert!(
        keeps_the_overwritten_store(Some(reads)),
        "the call reads it"
    );

    let mut shows = Effects::pure();
    shows.world = true;
    assert!(
        !keeps_the_overwritten_store(Some(shows)),
        "a call acting only on the world never reads it"
    );
    let elsewhere = Effects::pure().with_access(CallLocation::Field(OTHER), AccessMode::Read);
    assert!(
        !keeps_the_overwritten_store(Some(elsewhere)),
        "it reads another field"
    );
}

/// **An out argument taken of a field is that field.** `p = &HELD`, a call
/// writing through `p`: the reload of `HELD` is not forwarded, the reload of
/// `OTHER` is.
#[test]
fn a_pointee_taken_of_a_field_is_the_field() {
    let forwarded = |reloaded_field: u32| {
        let mut ssa = SsaFunction::new(0, 8);
        let seven = var(&mut ssa, 0, 0, MockType::I32);
        let address = var(&mut ssa, 1, 3, MockType::Ptr);
        let reloaded = var(&mut ssa, 2, 5, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::StoreStaticField {
            access: AccessQualifiers::FLAT,
            field: HELD,
            value: seven,
        }));
        b0.add_instruction(instr(SsaOp::StoreStaticField {
            access: AccessQualifiers::FLAT,
            field: OTHER,
            value: seven,
        }));
        b0.add_instruction(instr(SsaOp::LoadStaticFieldAddr {
            dest: address,
            field: HELD,
        }));
        let out = Effects::pure().with_access(
            CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::Bytes(4),
            },
            AccessMode::Write,
        );
        b0.add_instruction(call(vec![address], Some(out)));
        b0.add_instruction(instr(SsaOp::LoadStaticField {
            access: AccessQualifiers::FLAT,
            dest: reloaded,
            field: reloaded_field,
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(reloaded))));
        ssa.add_block(b0);
        ssa.recompute_uses();
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        count(&ssa, |op| matches!(op, SsaOp::LoadStaticField { .. })) == 0
    };
    assert!(
        !forwarded(HELD),
        "the call writes the field through the address"
    );
    assert!(forwarded(OTHER), "and no other");
}

/// **A pure call whose result is unused is removed; any other call is kept.**
#[test]
fn only_a_pure_call_is_dead_code() {
    let survives = |effects: Option<Effects>| {
        let mut ssa = SsaFunction::new(0, 8);
        let result = var(&mut ssa, 0, 0, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr({
            let mut call = CallData::new(Some(result), Callee::Direct(9), Vec::new());
            call.effects = effects;
            SsaOp::from(call)
        }));
        b0.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();
        assert!(
            DeadCodeEliminationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::DeadCodeRounds, 4))
                )
                .is_ok()
        );
        count(&ssa, |op| matches!(op, SsaOp::Call(_))) == 1
    };
    assert!(survives(None), "an undescribed call may do anything");
    assert!(
        !survives(Some(Effects::pure())),
        "a pure call does nothing else"
    );
    let mut shows = Effects::pure();
    shows.world = true;
    assert!(survives(Some(shows)), "a call acting on the world is kept");
    let writes = Effects::pure().with_access(CallLocation::Field(HELD), AccessMode::Write);
    assert!(survives(Some(writes)), "a call writing memory is kept");
}

/// **Across blocks too.** `HELD = 7; call; goto b1` then `w = HELD`: Memory
/// SSA versions only what the call may write, so the store is still the
/// field's definition when the next block reads it.
#[test]
fn a_value_crosses_a_call_into_the_next_block() {
    let forwarded = |effects: Option<Effects>| {
        let mut ssa = SsaFunction::new(0, 8);
        let seven = var(&mut ssa, 0, 0, MockType::I32);
        let reloaded = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::StoreStaticField {
            access: AccessQualifiers::FLAT,
            field: HELD,
            value: seven,
        }));
        b0.add_instruction(call(Vec::new(), effects));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::LoadStaticField {
            access: AccessQualifiers::FLAT,
            dest: reloaded,
            field: HELD,
        }));
        b1.add_instruction(instr(SsaOp::returns(Some(reloaded))));
        ssa.add_block(b0);
        ssa.add_block(b1);
        ssa.recompute_uses();
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(ssa.validate(), Ok(()));
        count(&ssa, |op| matches!(op, SsaOp::LoadStaticField { .. })) == 0
    };
    assert!(!forwarded(None), "an undescribed call may write anything");
    let elsewhere = Effects::pure().with_access(CallLocation::Field(OTHER), AccessMode::Write);
    assert!(forwarded(Some(elsewhere)), "it writes another field");
    let writes = Effects::pure().with_access(CallLocation::Field(HELD), AccessMode::Write);
    assert!(!forwarded(Some(writes)), "it writes the field");
}

/// **The evaluator forgets what a call may write.** `HELD = 7; call; w =
/// HELD`: tracking memory, it carried the 7 across every call, even one that
/// may write anything.
#[test]
fn the_evaluator_forgets_what_a_call_may_write() {
    // `None`: no call at all; `Some(None)`: a call nobody described.
    let reloaded = |effects: Option<Option<Effects>>| {
        let mut ssa = SsaFunction::new(0, 8);
        let seven = var(&mut ssa, 0, 0, MockType::I32);
        let reloaded = var(&mut ssa, 1, 3, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::StoreStaticField {
            access: AccessQualifiers::FLAT,
            field: HELD,
            value: seven,
        }));
        match effects {
            Some(effects) => b0.add_instruction(call(Vec::new(), effects)),
            None => b0.add_instruction(instr(SsaOp::Nop)),
        }
        b0.add_instruction(instr(SsaOp::LoadStaticField {
            access: AccessQualifiers::FLAT,
            dest: reloaded,
            field: HELD,
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(reloaded))));
        ssa.add_block(b0);
        ssa.recompute_uses();
        let mut eval = SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
        eval.evaluate_block(0);
        eval.get_concrete(reloaded).and_then(|value| value.as_i64())
    };
    assert_eq!(
        reloaded(None),
        Some(7),
        "without a call the store reaches the load"
    );
    assert_eq!(
        reloaded(Some(None)),
        None,
        "an undescribed call may write anything"
    );
    let writes = Effects::pure().with_access(CallLocation::Field(HELD), AccessMode::MayWrite);
    assert_eq!(reloaded(Some(Some(writes))), None, "it may write the field");
    let elsewhere = Effects::pure().with_access(CallLocation::Field(OTHER), AccessMode::Write);
    assert_eq!(
        reloaded(Some(Some(elsewhere))),
        Some(7),
        "it writes another field"
    );
}

/// **A store forgets what it may overwrite.** `o1.f = 1; o2.f = 2; w = o1.f`:
/// nothing says `o1` and `o2` are different objects, so the load no longer
/// knows the 1 - the evaluator kept it, since a store replaced only its own
/// cell. The control stores to a static field instead, which no instance
/// field can be.
#[test]
fn an_evaluated_store_forgets_what_it_may_overwrite() {
    let reloaded = |instance: bool| {
        let mut ssa = SsaFunction::new(0, 8);
        let first = var(&mut ssa, 0, 0, MockType::Ptr);
        let second = var(&mut ssa, 1, 1, MockType::Ptr);
        let one = var(&mut ssa, 2, 2, MockType::I32);
        let two = var(&mut ssa, 3, 3, MockType::I32);
        let reloaded = var(&mut ssa, 4, 6, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        for (dest, value) in [(first, 0x1000), (second, 0x2000), (one, 1), (two, 2)] {
            b0.add_instruction(instr(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            }));
        }
        b0.add_instruction(instr(SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object: first,
            field: HELD,
            value: one,
        }));
        b0.add_instruction(instr(if instance {
            SsaOp::StoreField {
                access: AccessQualifiers::FLAT,
                object: second,
                field: HELD,
                value: two,
            }
        } else {
            SsaOp::StoreStaticField {
                access: AccessQualifiers::FLAT,
                field: HELD,
                value: two,
            }
        }));
        b0.add_instruction(instr(SsaOp::LoadField {
            access: AccessQualifiers::FLAT,
            dest: reloaded,
            object: first,
            field: HELD,
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(reloaded))));
        ssa.add_block(b0);
        ssa.recompute_uses();
        let mut eval = SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
        eval.evaluate_block(0);
        eval.get_concrete(reloaded).and_then(|value| value.as_i64())
    };
    assert_eq!(reloaded(false), Some(1), "a static field cannot be o1.f");
    assert_eq!(reloaded(true), None, "o2.f may be o1.f");
}

/// Capturing calls remain observable for all call forms even with no memory footprint.
#[test]
fn capturing_call_results_cannot_make_the_call_dead() {
    for kind in 0..3 {
        let mut ssa = SsaFunction::new(0, 0);
        let pointer = var(&mut ssa, 0, 0, MockType::Ptr);
        let result = var(&mut ssa, 1, 1, MockType::Ptr);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: pointer,
            value: ConstValue::U64(0x1000),
        }));
        let mut effects = Effects::pure();
        effects.captures = Capture::Any;
        let effects = Some(Box::new(effects));
        let args = vec![pointer];
        block.add_instruction(instr(match kind {
            0 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                dest: Some(result),
                callee: analyssa::ir::Callee::Direct(9),
                args,
                effects: (effects).map(|effects| *effects),
                tail: false,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: Vec::new(),
            })),
            1 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                dest: Some(result),
                callee: analyssa::ir::Callee::Virtual(9),
                args,
                effects: (effects).map(|effects| *effects),
                tail: false,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: Vec::new(),
            })),
            _ => SsaOp::Call(Box::new(analyssa::ir::CallData {
                dest: Some(result),
                callee: analyssa::ir::Callee::Indirect {
                    fptr: pointer,
                    signature: 0,
                },
                args,
                effects: (effects).map(|effects| *effects),
                tail: false,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: Vec::new(),
            })),
        }));
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();
        assert!(
            DeadCodeEliminationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::DeadCodeRounds, 4))
                )
                .is_ok()
        );
        assert_eq!(
            count(&ssa, |op| matches!(op, SsaOp::Call(_))),
            1,
            "call form {kind}"
        );
    }
}
