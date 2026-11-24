//! A slot whose address escapes is memory, and the IR treats it as such.
//!
//! `ldloca` / `ldarga` - and their equivalents on a native frame - hand out a
//! pointer to a slot. A store through that pointer changes what a later load of
//! the slot reads, and nothing in the value graph records it. Every rule that
//! reads a slot's loads as reads of its current SSA value is therefore sound
//! only for slots that never escape, and these tests pin that boundary.

use std::process;

use analyssa::{
    Endianness, PointerSize,
    analysis::{
        Machine,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
    },
    events::NullListener,
    ir::{
        FloatRules, NanEncoding,
        block::SsaBlock,
        function::{SsaConstructOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::{AccessQualifiers, SsaOp},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::{AlgebraicSimplificationPass, CopyPropagationPass, GlobalValueNumberingPass},
    scheduling::{FunctionPass, PassEnv},
    testing::{MACHINE, MockTarget, MockType},
};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn result_or_abort<T, E>(result: Result<T, E>) -> T {
    result.unwrap_or_else(|_| process::abort())
}

/// Declares a value of `origin` defined at `def`.
fn declare(
    ssa: &mut SsaFunction<MockTarget>,
    origin: VariableOrigin,
    version: u32,
    def: DefSite,
) -> SsaVarId {
    let id = ssa.create_variable(origin, version, def, MockType::I32);
    if let Some(group) = origin.canonical_rename_group(ssa.num_args()) {
        ssa.set_rename_group(id, group);
    }
    id
}

/// `local0 = 1; p = &local0; store through p; v = local0` - the shape where
/// resolving the load to the slot's SSA value reads the value from *before* the
/// store.
fn store_through_a_local_pointer() -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let initial = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
    );
    let pointer = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 1));
    let stored = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 2));
    let loaded = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(0, 4),
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: initial,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocalAddr {
        dest: pointer,
        local_index: 0,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: stored,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::StoreIndirect {
        addr: pointer,
        value: stored,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocal {
        dest: loaded,
        local_index: 0,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(loaded))));
    ssa.add_block(block);
    ssa.recompute_uses();
    (ssa, initial, loaded)
}

#[test]
fn a_load_of_an_address_taken_local_is_not_resolved_to_its_value() {
    let (mut ssa, initial, loaded) = store_through_a_local_pointer();
    assert!(ssa.address_taken_slots().local(0));

    result_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    // The load must survive as a read of memory. Collapsing it would make the
    // function return 1, the value the slot held before the store.
    let has_load = ssa.blocks().iter().any(|block| {
        block
            .instructions()
            .iter()
            .any(|instr| matches!(instr.op(), SsaOp::LoadLocal { local_index: 0, .. }))
    });
    assert!(
        has_load,
        "the load reads memory a store through the pointer changed: {ssa}"
    );
    let _ = (initial, loaded);
}

#[test]
fn a_load_of_an_ordinary_local_is_still_resolved_to_its_value() {
    // Control: the same shape without the address-taking operation keeps the
    // optimization the guard above suppresses.
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let initial = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
    );
    let loaded = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(0, 1),
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: initial,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocal {
        dest: loaded,
        local_index: 0,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(loaded))));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(!ssa.address_taken_slots().local(0));
    result_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let has_load = ssa.blocks().iter().any(|block| {
        block
            .instructions()
            .iter()
            .any(|instr| matches!(instr.op(), SsaOp::LoadLocal { .. }))
    });
    assert!(
        !has_load,
        "a slot that never escapes is a value, so its load resolves to it"
    );
}

#[test]
fn an_address_taken_argument_is_recognized() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
    let arg = declare(&mut ssa, VariableOrigin::Argument(0), 0, DefSite::entry());
    let pointer = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 0));

    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArgAddr {
        dest: pointer,
        arg_index: 0,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(pointer))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let slots = ssa.address_taken_slots();
    assert!(
        slots.argument(0),
        "an argument whose address is taken escapes exactly as a local does"
    );
    assert!(slots.group(0, ssa.num_args()));

    ssa.refresh_address_taken();
    assert!(some_or_abort(ssa.variable(arg)).is_address_taken());
}

#[test]
fn copy_propagation_leaves_an_address_taken_slot_alone() {
    let (mut ssa, _, _) = store_through_a_local_pointer();
    let method = 0u32;
    let before = format!("{ssa}");
    let _ = CopyPropagationPass.propagate(
        &mut ssa,
        &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, 4)),
        |_, _| {},
    );

    // Whatever else it does, the slot's load must still be there: forwarding a
    // copy of the pre-store value over it would change what the function
    // returns.
    let has_load = ssa.blocks().iter().any(|block| {
        block
            .instructions()
            .iter()
            .any(|instr| matches!(instr.op(), SsaOp::LoadLocal { local_index: 0, .. }))
    });
    assert!(has_load, "before: {before}\nafter: {ssa}");
}

#[test]
fn the_flag_follows_the_function() {
    let (mut ssa, _, _) = store_through_a_local_pointer();
    assert_eq!(ssa.refresh_address_taken(), 2, "both versions of local 0");

    // Removing the address-taking operation makes the slot an ordinary value
    // again, which the refresh has to notice.
    if let Some(block) = ssa.block_mut(0) {
        block.instructions_mut().retain(|instr| {
            !matches!(
                instr.op(),
                SsaOp::LoadLocalAddr { .. } | SsaOp::StoreIndirect { .. }
            )
        });
    }
    ssa.recompute_uses();
    assert_eq!(ssa.refresh_address_taken(), 0);
    assert!(!ssa.address_taken_slots().local(0));
}

/// How the second write to the slot is made.
#[derive(Clone, Copy)]
enum Reset {
    /// `local0 = 0`, the first write's constant again.
    Constant,
    /// `local0 = copy` of the first write.
    Copy,
    /// `local0 = x + y`, the first write's sum computed again.
    Sum,
}

/// `local0 = first; p = &local0; *p = 1; local0 = reset; return local0`, the
/// reset the same value as the first write.
///
/// Nothing reads the reset as an SSA value: the return reads the slot.
/// Removing it as redundant or unread drops a write the read sees, and the
/// function returns 1. Without `escapes` no address is taken and the return
/// reads the stored value, so nothing reads the slot as storage.
fn reset_after_a_store_through_a_pointer(
    reset_by: Reset,
    escapes: bool,
) -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let x = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 0));
    let y = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 1));
    let initial = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 2),
    );
    let pointer = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 3));
    let stored = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 4));
    let reset = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(0, 6),
    );
    let loaded = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 7));
    let constant = |dest, value| {
        SsaInstruction::synthetic(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let first = |dest| match reset_by {
        Reset::Constant | Reset::Copy => constant(dest, 0),
        Reset::Sum => SsaInstruction::synthetic(SsaOp::Add {
            dest,
            left: x,
            right: y,
            flags: None,
        }),
    };

    let mut block = SsaBlock::new(0);
    block.add_instruction(constant(x, 2));
    block.add_instruction(constant(y, 3));
    block.add_instruction(first(initial));
    block.add_instruction(if escapes {
        SsaInstruction::synthetic(SsaOp::LoadLocalAddr {
            dest: pointer,
            local_index: 0,
        })
    } else {
        constant(pointer, 64)
    });
    block.add_instruction(constant(stored, 1));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::StoreIndirect {
        addr: pointer,
        value: stored,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(match reset_by {
        Reset::Copy => SsaInstruction::synthetic(SsaOp::Copy {
            dest: reset,
            src: initial,
        }),
        Reset::Constant | Reset::Sum => first(reset),
    });
    if escapes {
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocal {
            dest: loaded,
            local_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(loaded))));
    } else {
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(stored))));
    }
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa
}

/// Whether the reset is still written: whether `ssa` still holds as many
/// instructions of the reset's kind as the fixture made, the first write's
/// and the reset's.
///
/// Counted by kind, not by the reset's id, since a pass that removes an
/// instruction renumbers the values after it.
fn reset_written(ssa: &SsaFunction<MockTarget>, reset_by: Reset) -> bool {
    let count = ssa
        .iter_instructions()
        .filter(|(_, _, instr)| match (reset_by, instr.op()) {
            (Reset::Constant, SsaOp::Const { value, .. }) => *value == ConstValue::I32(0),
            (Reset::Copy, SsaOp::Copy { .. }) => true,
            (Reset::Sum, SsaOp::Add { .. }) => true,
            _ => false,
        })
        .count();
    count
        == if matches!(reset_by, Reset::Copy) {
            1
        } else {
            2
        }
}

/// **A write a read through the slot sees is not removed as redundant.** GVN
/// numbers the recomputed sum as the first write's value, which it is, and
/// keeps the write; a slot nothing reads as storage loses it, which is the
/// control.
#[test]
fn value_numbering_keeps_a_write_the_slot_is_read_after() {
    let mut ssa = reset_after_a_store_through_a_pointer(Reset::Sum, true);
    assert!(ssa.storage_slots().contains(VariableOrigin::Local(0)));
    assert!(
        GlobalValueNumberingPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert!(
        reset_written(&ssa, Reset::Sum),
        "the reset is written\n{ssa}"
    );

    let mut ssa = reset_after_a_store_through_a_pointer(Reset::Sum, false);
    assert!(!ssa.storage_slots().contains(VariableOrigin::Local(0)));
    assert!(
        GlobalValueNumberingPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert!(
        !reset_written(&ssa, Reset::Sum),
        "nothing reads the slot\n{ssa}"
    );
}

/// **A copy into the slot is a write, propagated or not.** Copy propagation
/// forwards the copy's readers and keeps the copy itself.
#[test]
fn copy_propagation_keeps_a_copy_into_a_slot_read_as_storage() {
    let mut ssa = reset_after_a_store_through_a_pointer(Reset::Copy, true);
    CopyPropagationPass.run_iteration(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        |_, _| {},
    );
    assert!(
        reset_written(&ssa, Reset::Copy),
        "the reset is written\n{ssa}"
    );
}

/// **An unread write to the slot is not unread.** `remove_unread_instructions`
/// keeps it, and removes the same write to a slot nothing reads as storage.
#[test]
fn an_unread_write_to_a_slot_read_as_storage_stays() {
    let mut ssa = reset_after_a_store_through_a_pointer(Reset::Constant, true);
    assert!(ssa.remove_unread_instructions([(0, 6)]).is_empty());
    assert!(
        reset_written(&ssa, Reset::Constant),
        "the reset is written\n{ssa}"
    );

    let mut ssa = reset_after_a_store_through_a_pointer(Reset::Constant, false);
    assert_eq!(ssa.remove_unread_instructions([(0, 6)]).len(), 1);
    assert!(
        !reset_written(&ssa, Reset::Constant),
        "nothing reads the slot\n{ssa}"
    );
}

/// `local0 = x + y; local0 = 5; local0 = reset; return ldloc local0`: the
/// reset is the first write's value again, by a sum or a copy.
///
/// The middle write changes the slot, so the reset is no repetition of the
/// slot's state even though it repeats the first write's value; the read
/// through the slot sees it. Without `read_by_value` nothing reads the slot,
/// which is the control.
fn rewritten_then_reset(copied: bool, read_by_value: bool) -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let x = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 0));
    let y = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 1));
    let first = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 2),
    );
    let middle = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(0, 3),
    );
    let reset = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        2,
        DefSite::instruction(0, 4),
    );
    let negated = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 5));
    let loaded = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 6));
    let constant = |dest, value| {
        SsaInstruction::synthetic(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let sum = |dest| {
        SsaInstruction::synthetic(SsaOp::Add {
            dest,
            left: x,
            right: y,
            flags: None,
        })
    };

    let mut block = SsaBlock::new(0);
    block.add_instruction(constant(x, 2));
    block.add_instruction(constant(y, 3));
    block.add_instruction(sum(first));
    block.add_instruction(constant(middle, 5));
    block.add_instruction(if copied {
        SsaInstruction::synthetic(SsaOp::Copy {
            dest: reset,
            src: first,
        })
    } else {
        sum(reset)
    });
    // Copy propagation forwards a copy's readers before it drops the copy, so
    // the reset is read once as a value.
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Neg {
        dest: negated,
        operand: reset,
        flags: None,
    }));
    if read_by_value {
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocal {
            dest: loaded,
            local_index: 0,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(loaded))));
    } else {
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(negated))));
    }
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa
}

/// **A slot read by value is storage too.** A rewrite of the slot to a value it
/// held before is redundant as a value and not as a write: GVN and copy
/// propagation keep it where `ldloc` reads the slot, and drop it where nothing
/// does.
#[test]
fn a_write_a_load_of_the_slot_sees_is_kept() {
    for copied in [false, true] {
        let reset_by = if copied { Reset::Copy } else { Reset::Sum };
        let run = |ssa: &mut SsaFunction<MockTarget>| {
            if copied {
                CopyPropagationPass.run_iteration(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                    |_, _| {},
                );
            } else {
                assert!(
                    GlobalValueNumberingPass
                        .run(
                            ssa,
                            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                        )
                        .is_ok()
                );
            }
        };

        let mut ssa = rewritten_then_reset(copied, true);
        assert!(ssa.storage_slots().contains(VariableOrigin::Local(0)));
        assert!(!ssa.address_taken_slots().local(0));
        run(&mut ssa);
        assert!(
            reset_written(&ssa, reset_by),
            "copied={copied}: the reset is written\n{ssa}"
        );

        let mut ssa = rewritten_then_reset(copied, false);
        run(&mut ssa);
        assert!(
            !reset_written(&ssa, reset_by),
            "copied={copied}: nothing reads the slot\n{ssa}"
        );
    }
}

/// `lo = *p; hi = *(p + 4); v = (zext hi << 32) | zext lo`, the 64-bit value
/// a 32-bit target reads in two words, with `hi` loaded into local 3 and,
/// with `read_by_value`, that slot read back with `ldloc 3`.
fn two_word_load(read_by_value: bool) -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
    let at = |index| DefSite::instruction(0, index);
    let mut make = |origin, index, ty| ssa.create_variable(origin, 0, at(index), ty);
    let temp = VariableOrigin::Phi;
    let base = make(temp, 0, MockType::Ptr);
    let high_addr = make(temp, 1, MockType::Ptr);
    let low = make(temp, 2, MockType::I32);
    let high = make(VariableOrigin::Local(3), 3, MockType::I32);
    let low_wide = make(temp, 4, MockType::I64);
    let high_wide = make(temp, 5, MockType::I64);
    let thirty_two = make(temp, 6, MockType::I32);
    let shifted = make(temp, 7, MockType::I64);
    let value = make(temp, 8, MockType::I64);
    let slot = read_by_value.then(|| make(temp, 9, MockType::I32));
    let total = read_by_value.then(|| make(temp, 10, MockType::I64));
    let slot_wide = read_by_value.then(|| make(temp, 11, MockType::I64));

    let synthetic = SsaInstruction::synthetic;
    let load = |dest, addr| {
        synthetic(SsaOp::LoadIndirect {
            dest,
            addr,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        })
    };
    let widen = |dest, operand| {
        synthetic(SsaOp::IntConv {
            dest,
            operand,
            target: MockType::I64,
            overflow_check: false,
            unsigned: true,
        })
    };
    let mut block = SsaBlock::new(0);
    block.add_instruction(synthetic(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    block.add_instruction(synthetic(SsaOp::PtrAdd {
        dest: high_addr,
        base,
        index: None,
        stride: 0,
        offset: 4,
        result_type: MockType::Ptr,
    }));
    block.add_instruction(load(low, base));
    block.add_instruction(load(high, high_addr));
    block.add_instruction(widen(low_wide, low));
    block.add_instruction(widen(high_wide, high));
    block.add_instruction(synthetic(SsaOp::Const {
        dest: thirty_two,
        value: ConstValue::I32(32),
    }));
    block.add_instruction(synthetic(SsaOp::Shl {
        dest: shifted,
        value: high_wide,
        amount: thirty_two,
        flags: None,
    }));
    block.add_instruction(synthetic(SsaOp::Or {
        dest: value,
        left: shifted,
        right: low_wide,
        flags: None,
    }));
    match (slot, total, slot_wide) {
        (Some(slot), Some(total), Some(slot_wide)) => {
            block.add_instruction(synthetic(SsaOp::LoadLocal {
                dest: slot,
                local_index: 3,
            }));
            block.add_instruction(synthetic(SsaOp::Add {
                dest: total,
                left: value,
                right: slot_wide,
                flags: None,
            }));
            block.add_instruction(widen(slot_wide, slot));
            block.add_instruction(synthetic(SsaOp::returns(Some(total))));
        }
        _ => block.add_instruction(synthetic(SsaOp::returns(Some(value)))),
    }
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa
}

/// The word loads `ssa` holds.
fn word_loads(ssa: &SsaFunction<MockTarget>) -> usize {
    ssa.iter_instructions()
        .filter(|(_, _, instr)| {
            matches!(
                instr.op(),
                SsaOp::LoadIndirect {
                    value_type: MockType::I32,
                    ..
                }
            )
        })
        .count()
}

/// **A word load into a slot read as storage is not folded into a wide load.**
/// The fold removes the word loads it replaces, and one of them writes local 3,
/// which `ldloc 3` reads: the fold is declined. With no read through the slot
/// the two words fold into one load - the control.
#[test]
fn a_word_load_into_a_slot_read_as_storage_is_not_folded_away() {
    let little_endian = Machine::new(
        PointerSize::Bit32,
        Some(Endianness::Little),
        FloatRules {
            out_of_range: None,
            default_nan: None,
            nan_conversion: None,
            nan_operands: None,
            extended_nan_operands: None,

            program_flushes_subnormals: false,

            nan_encoding: NanEncoding::Ieee,
        },
    );
    let mut ssa = two_word_load(false);
    assert!(
        AlgebraicSimplificationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(little_endian))
            )
            .is_ok()
    );
    assert_eq!(
        word_loads(&ssa),
        0,
        "the two words are one wide load\n{ssa}"
    );

    let mut ssa = two_word_load(true);
    assert!(
        AlgebraicSimplificationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(little_endian))
            )
            .is_ok()
    );
    assert_eq!(
        word_loads(&ssa),
        2,
        "the load into the slot is its write\n{ssa}"
    );
}

/// `x = 1; local0 = x; p = &local0; n = -p; return`, the shape of a reflection
/// chain that stores a handle into a local and calls through its address.
fn a_slot_read_only_by_what_goes_with_it() -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let x = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 0));
    let stored = declare(
        &mut ssa,
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 1),
    );
    let pointer = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 2));
    let negated = declare(&mut ssa, VariableOrigin::Phi, 0, DefSite::instruction(0, 3));
    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: x,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
        dest: stored,
        src: x,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocalAddr {
        dest: pointer,
        local_index: 0,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Neg {
        dest: negated,
        operand: pointer,
        flags: None,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa
}

/// **A write goes with the only read of its slot.** Removed together, the write
/// to local 0 and the address taken of it leave nothing reading the slot, so
/// the write is unread like any value; with the address taken left in place
/// the slot is still read, and the write stays.
#[test]
fn a_write_goes_with_the_only_read_of_its_slot() {
    let mut ssa = a_slot_read_only_by_what_goes_with_it();
    assert_eq!(
        ssa.remove_unread_instructions([(0, 1), (0, 2), (0, 3)])
            .len(),
        3
    );

    let mut ssa = a_slot_read_only_by_what_goes_with_it();
    let removed = ssa.remove_unread_instructions([(0, 1), (0, 3)]);
    assert_eq!(
        removed.iter().map(|(b, i, _)| (*b, *i)).collect::<Vec<_>>(),
        vec![(0, 3)],
        "the address is still taken, so the write stays"
    );
}
