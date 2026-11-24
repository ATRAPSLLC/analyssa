//! Memory optimization pass: store-to-load forwarding, redundant load
//! elimination, and dead store elimination.
//!
//! Each positive case is paired with the negative that proves the alias gate or
//! barrier is doing the work - a pass that rewrote unconditionally would pass
//! every positive test here.

use std::{error::Error, num::NonZeroU16};

use analyssa::{
    analysis::{AnalysisContext, SsaVerifier, VerifyLevel},
    events::NullListener,
    ir::{
        CallData, CallInput, CallInputKey, Callee, ReturnTransfer,
        block::SsaBlock,
        exception::{BlockRange, SsaExceptionHandler},
        function::SsaFunction,
        instruction::{InstructionRole, SsaInstruction},
        ops::{
            AccessMode, AccessPermission, AccessQualifiers, AtomicAccessWidth, AtomicOrdering,
            CacheMaintenanceOp, CallAccess, CallEffects, CallLocation, CallMemory, Capture,
            FenceKind, FrameOrigin, NativeKindedData, OpKindTable, PointeeExtent, SsaOp, SysRegOp,
            SystemOpKind, kinds::CmpKind,
        },
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
    },
    passes::MemoryOptimizationPass,
    scheduling::{FunctionPass, PassEnv},
    testing::{MACHINE, MockTarget, MockType},
};

/// Declares a variable defined at `(block, instr)`.
fn var(
    ssa: &mut SsaFunction<MockTarget>,
    idx: u16,
    block: usize,
    instr: usize,
    ty: MockType,
) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        ty,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

/// Emits `dest = base + offset` as a `PtrAdd`.
fn ptr_add(dest: SsaVarId, base: SsaVarId, offset: i64) -> SsaInstruction<MockTarget> {
    instr(SsaOp::PtrAdd {
        dest,
        base,
        index: None,
        stride: 0,
        offset,
        result_type: MockType::Ptr,
    })
}

fn store(addr: SsaVarId, value: SsaVarId) -> SsaInstruction<MockTarget> {
    instr(SsaOp::StoreIndirect {
        addr,
        value,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    })
}

/// A store the host declares a save of the caller's state
/// ([`InstructionRole::CallerStateSave`]).
fn save(addr: SsaVarId, value: SsaVarId) -> SsaInstruction<MockTarget> {
    let mut save = store(addr, value);
    save.set_role(InstructionRole::CallerStateSave);
    save
}

fn load(dest: SsaVarId, addr: SsaVarId) -> SsaInstruction<MockTarget> {
    instr(SsaOp::LoadIndirect {
        dest,
        addr,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    })
}

/// Counts the loads and stores surviving in `ssa`.
fn access_counts(ssa: &SsaFunction<MockTarget>) -> (usize, usize) {
    let mut loads = 0usize;
    let mut stores = 0usize;
    for block in ssa.blocks() {
        for instruction in block.instructions() {
            match instruction.op() {
                SsaOp::LoadIndirect { .. } => loads = loads.saturating_add(1),
                SsaOp::StoreIndirect { .. } => stores = stores.saturating_add(1),
                _ => {}
            }
        }
    }
    (loads, stores)
}

/// Builds a single-block function: `*(p+a) = v; w = *(p+b); return w`.
///
/// With `a == b` the load reads exactly what was stored; with `a != b` far
/// enough apart it reads a disjoint cell.
fn store_then_load(store_offset: i64, load_offset: i64) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let store_addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let load_addr = var(&mut ssa, 3, 0, 3, MockType::Ptr);
    let loaded = var(&mut ssa, 4, 0, 5, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(store_addr, base, store_offset));
    b0.add_instruction(ptr_add(load_addr, base, load_offset));
    b0.add_instruction(store(store_addr, value));
    b0.add_instruction(load(loaded, load_addr));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

#[test]
fn store_to_load_forwarding_replaces_the_load() {
    let mut ssa = store_then_load(0, 0);
    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    assert!(changed, "an exact reload of a just-stored cell forwards");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!(loads, 0, "the load is gone");
    assert_eq!(stores, 1, "the store is live - it is still observable");
    assert_eq!(ssa.validate(), Ok(()));
}

/// The load reads a cell 8 bytes away from the store, so nothing may be
/// forwarded. This is the negative that proves the offset comparison is load
/// bearing: both accesses go through the same base, so only the decoded
/// displacement tells them apart.
#[test]
fn disjoint_offsets_are_not_forwarded() {
    let mut ssa = store_then_load(0, 8);
    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    assert!(!changed, "a disjoint cell must not be forwarded");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!((loads, stores), (1, 1));
    assert_eq!(ssa.validate(), Ok(()));
}

/// Two loads of one cell with nothing in between: the second is redundant.
#[test]
fn redundant_load_is_eliminated() {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let addr = var(&mut ssa, 1, 0, 1, MockType::Ptr);
    let first = var(&mut ssa, 2, 0, 2, MockType::I32);
    let second = var(&mut ssa, 3, 0, 3, MockType::I32);
    let sum = var(&mut ssa, 4, 0, 4, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(load(first, addr));
    b0.add_instruction(load(second, addr));
    b0.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: first,
        right: second,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "the second load of an unchanged cell is redundant");
    let (loads, _) = access_counts(&ssa);
    assert_eq!(loads, 1, "one load survives");
    assert_eq!(ssa.validate(), Ok(()));
}

/// **A redundant load into a slot read as storage is forwarded and kept.** The
/// second load's result is a version of local 3, which `ldloc 3` reads as the
/// slot: the load is also the slot's write, so its readers take the first load's
/// value and the load stays. [`redundant_load_is_eliminated`] is the control.
#[test]
fn a_redundant_load_into_a_slot_read_as_storage_stays() {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let addr = var(&mut ssa, 1, 0, 1, MockType::Ptr);
    let first = var(&mut ssa, 2, 0, 2, MockType::I32);
    let second = var(&mut ssa, 3, 0, 3, MockType::I32);
    let sum = var(&mut ssa, 4, 0, 4, MockType::I32);
    let slot = var(&mut ssa, 5, 0, 5, MockType::I32);
    let total = var(&mut ssa, 6, 0, 6, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(load(first, addr));
    b0.add_instruction(load(second, addr));
    b0.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: first,
        right: second,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::LoadLocal {
        dest: slot,
        local_index: 3,
    }));
    b0.add_instruction(instr(SsaOp::Add {
        dest: total,
        left: sum,
        right: slot,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(total))));
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
    let (loads, _) = access_counts(&ssa);
    assert_eq!(loads, 2, "the load into the slot is its write\n{ssa}");
    assert_eq!(ssa.validate(), Ok(()));
}

/// A fence between a store and a reload blocks forwarding: the barrier's memory
/// effect is not modelled precisely, so nothing may be carried across it.
#[test]
fn a_barrier_blocks_forwarding() {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let loaded = var(&mut ssa, 3, 0, 5, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(store(addr, value));
    b0.add_instruction(instr(SsaOp::Fence {
        kind: FenceKind::SeqCst,
    }));
    b0.add_instruction(load(loaded, addr));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(!changed, "a fence blocks store-to-load forwarding");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!((loads, stores), (1, 1));
    assert_eq!(ssa.validate(), Ok(()));
}

/// Two stores to one cell with no read between them: the first is dead.
#[test]
fn overwritten_store_is_removed() {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let first_value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let second_value = var(&mut ssa, 2, 0, 2, MockType::I32);
    let addr = var(&mut ssa, 3, 0, 3, MockType::Ptr);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: first_value,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: second_value,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(store(addr, first_value));
    b0.add_instruction(store(addr, second_value));
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "the overwritten store is dead");
    let (_, stores) = access_counts(&ssa);
    assert_eq!(stores, 1, "only the surviving store remains");
    assert_eq!(ssa.validate(), Ok(()));
}

/// A read between two stores makes the first observable, so it is not dead.
#[test]
fn store_read_before_being_overwritten_survives() {
    let mut ssa = SsaFunction::new(0, 10);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let first_value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let second_value = var(&mut ssa, 2, 0, 2, MockType::I32);
    let addr = var(&mut ssa, 3, 0, 3, MockType::Ptr);
    let other = var(&mut ssa, 4, 0, 4, MockType::Ptr);
    let observed = var(&mut ssa, 5, 0, 6, MockType::I32);
    let incoming = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: first_value,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: second_value,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    // An address the model cannot separate from `addr`: a pointer the function
    // was handed, which may hold `0x1000`, so a load through it may observe
    // the first store.
    b0.add_instruction(instr(SsaOp::Copy {
        dest: other,
        src: incoming,
    }));
    b0.add_instruction(store(addr, first_value));
    b0.add_instruction(load(observed, other));
    b0.add_instruction(store(addr, second_value));
    b0.add_instruction(instr(SsaOp::returns(Some(observed))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    let (_, stores) = access_counts(&ssa);
    assert_eq!(
        stores, 2,
        "a store that may have been read is not dead (changed={changed})"
    );
    assert_eq!(ssa.validate(), Ok(()));
}

/// A read of a *different* absolute address does not observe the first store.
///
/// Two addresses the program states as numbers are two cells whenever their
/// extents are disjoint: `0x2000` cannot read what was written to `0x1000`,
/// however each constant was materialised. So the overwritten store is dead.
#[test]
fn a_read_of_another_absolute_address_does_not_keep_a_store() {
    let mut ssa = SsaFunction::new(0, 10);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let first_value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let second_value = var(&mut ssa, 2, 0, 2, MockType::I32);
    let addr = var(&mut ssa, 3, 0, 3, MockType::Ptr);
    let other = var(&mut ssa, 4, 0, 4, MockType::Ptr);
    let observed = var(&mut ssa, 5, 0, 6, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: first_value,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: second_value,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(instr(SsaOp::Const {
        dest: other,
        value: ConstValue::I32(0x2000),
    }));
    b0.add_instruction(store(addr, first_value));
    b0.add_instruction(load(observed, other));
    b0.add_instruction(store(addr, second_value));
    b0.add_instruction(instr(SsaOp::returns(Some(observed))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "the store to 0x1000 is overwritten unread");
    let (_, stores) = access_counts(&ssa);
    assert_eq!(stores, 1, "a load of 0x2000 reads nothing 0x1000 holds");
    assert_eq!(ssa.validate(), Ok(()));
}

/// A forwarded load's result that is itself stored and reloaded forwards to
/// the value it was forwarded from, never to the load that was removed.
///
/// `x = *p` is served by the store before it, and `x` is then stored to `p+8`
/// and read back as `y`. Planning sees `y` served by `x`, and `x` by `v`; the
/// first rewrite removes the load defining `x`, so the second must name `v`,
/// or `y`'s readers are left reading a value nothing defines.
#[test]
fn a_chain_of_forwards_resolves_to_the_value_that_survives() {
    let mut ssa = SsaFunction::new(0, 10);
    let incoming = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    let value = var(&mut ssa, 0, 0, 0, MockType::I32);
    let first = var(&mut ssa, 1, 0, 1, MockType::Ptr);
    let second = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let x = var(&mut ssa, 3, 0, 4, MockType::I32);
    let y = var(&mut ssa, 4, 0, 6, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(first, incoming, 0));
    b0.add_instruction(ptr_add(second, incoming, 8));
    b0.add_instruction(store(first, value));
    b0.add_instruction(load(x, first));
    b0.add_instruction(store(second, x));
    b0.add_instruction(load(y, second));
    b0.add_instruction(instr(SsaOp::returns(Some(y))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "both loads are served by the stores before them");
    assert_eq!(ssa.validate(), Ok(()), "no read of a removed load survives");
    let returned = ssa.blocks().iter().find_map(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.op() {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            })
    });
    assert_eq!(returned, Some(value), "`y` is `x` is `7`");
}

/// A store in a dominating block serves a load in a dominated one.
#[test]
fn forwarding_crosses_from_a_dominating_block() {
    let mut ssa = SsaFunction::new(0, 10);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let loaded = var(&mut ssa, 3, 1, 0, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(store(addr, value));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(load(loaded, addr));
    b1.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b1);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "a dominating store serves the load");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!(loads, 0, "the dominated load is forwarded away");
    assert_eq!(stores, 1);
    assert_eq!(ssa.validate(), Ok(()));
}

/// **A forwarded load is replaced in what a throw brings a handler too.** The
/// protected `B1` loads what `B0` stored and returns it from `B2`; the handler
/// `B3` merges the loaded value for a throw in `B1`. Forwarding removes the
/// load, so the handler must read the stored value instead, or it reads a value
/// nothing defines (a Delphi x64 routine's `finally`, through analysir). The
/// merge of one value is that value, so the handler returns the store's.
#[test]
fn a_forwarded_load_is_replaced_in_a_handler_s_merge() {
    let mut ssa = SsaFunction::new(0, 10);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let loaded = var(&mut ssa, 3, 1, 0, MockType::I32);
    let seen = ssa.create_variable(VariableOrigin::Local(4), 0, DefSite::phi(3), MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(store(addr, value));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(load(loaded, addr));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b2);
    let mut b3 = SsaBlock::new(3);
    let mut phi = PhiNode::new(seen, VariableOrigin::Local(4));
    phi.add_exceptional_operand(1, loaded);
    b3.add_phi(phi);
    b3.add_instruction(instr(SsaOp::returns(Some(seen))));
    ssa.add_block(b3);
    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(1, 2),
        handler_range: BlockRange::new(3, 4),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "the store serves the load");
    assert_eq!(access_counts(&ssa), (0, 1), "the load is forwarded away");
    let reads_loaded = ssa.blocks().iter().any(|block| {
        block
            .phi_nodes()
            .iter()
            .any(|phi| phi.used_variables().any(|used| used == loaded))
            || block
                .instructions()
                .iter()
                .any(|instruction| instruction.op().uses_var(loaded))
    });
    assert!(!reads_loaded, "nothing reads the removed load:\n{ssa}");
    assert_eq!(ssa.validate(), Ok(()));
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "{errors:?}");
}

/// A load at a merge whose two predecessors store different values must not be
/// forwarded - the memory phi says two definitions reach it.
#[test]
fn merge_with_conflicting_stores_is_not_forwarded() {
    let mut ssa = SsaFunction::new(0, 12);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let cond = var(&mut ssa, 1, 0, 1, MockType::I32);
    let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let left_value = var(&mut ssa, 3, 1, 0, MockType::I32);
    let right_value = var(&mut ssa, 4, 2, 0, MockType::I32);
    let loaded = var(&mut ssa, 5, 3, 0, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: left_value,
        value: ConstValue::I32(11),
    }));
    b1.add_instruction(store(addr, left_value));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: right_value,
        value: ConstValue::I32(22),
    }));
    b2.add_instruction(store(addr, right_value));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(load(loaded, addr));
    b3.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b3);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    let (loads, stores) = access_counts(&ssa);
    assert_eq!(
        (loads, stores),
        (1, 2),
        "a merge of two stores has no single available value (changed={changed})"
    );
    assert_eq!(ssa.validate(), Ok(()));
}

/// Accesses in different address spaces name different memory, so a segmented
/// store never serves a flat load at the same numeric offset.
#[test]
fn address_spaces_are_not_forwarded_across() {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let loaded = var(&mut ssa, 3, 0, 4, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    // Store into address space 1 (an `fs:`-style access)...
    b0.add_instruction(instr(SsaOp::StoreIndirect {
        addr,
        value,
        value_type: MockType::I32,
        access: AccessQualifiers::new(NonZeroU16::new(1), AccessPermission::Current),
    }));
    // ...then read the flat space at the same address.
    b0.add_instruction(load(loaded, addr));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(
        !changed,
        "a segmented store must not serve a flat load at the same offset"
    );
    let (loads, stores) = access_counts(&ssa);
    assert_eq!((loads, stores), (1, 1));
    assert_eq!(ssa.validate(), Ok(()));
}

/// A vector access lands in its address space as a scalar one does: a flat
/// vector store never serves a vector load through `fs:` at the same offset,
/// and the same store and load in one space still forward.
#[test]
fn vector_accesses_keep_their_address_space() {
    let forwards = |load_space: AccessQualifiers| {
        let mut ssa = SsaFunction::new(0, 9);
        let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
        let value = var(&mut ssa, 1, 0, 2, MockType::V4I32);
        let addr = var(&mut ssa, 2, 0, 3, MockType::Ptr);
        let loaded = var(&mut ssa, 3, 0, 5, MockType::V4I32);
        let lane = var(&mut ssa, 4, 0, 1, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: base,
            value: ConstValue::I32(0x1000),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: lane,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::VectorSplat {
            dest: value,
            value: lane,
            vector_type: MockType::V4I32,
        }));
        b0.add_instruction(ptr_add(addr, base, 0));
        b0.add_instruction(instr(SsaOp::VectorStore {
            addr,
            value,
            vector_type: MockType::V4I32,
            access: AccessQualifiers::FLAT,
        }));
        b0.add_instruction(instr(SsaOp::VectorLoad {
            dest: loaded,
            addr,
            vector_type: MockType::V4I32,
            access: load_space,
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
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
        !ssa.iter_instructions()
            .any(|(_, _, instruction)| matches!(instruction.op(), SsaOp::VectorLoad { .. }))
    };
    assert!(forwards(AccessQualifiers::FLAT), "one space forwards");
    assert!(
        !forwards(AccessQualifiers::new(
            NonZeroU16::new(1),
            AccessPermission::Current
        )),
        "a flat vector store must not serve an fs: vector load"
    );
}

/// Builds `*(p+0) = a; <separator>; *(p+0) = b; return`, the first store made
/// as `first` says, and reports how many stores survive the pass.
fn stores_around(first: AccessQualifiers, separator: Vec<SsaInstruction<MockTarget>>) -> usize {
    let mut ssa = SsaFunction::new(0, 12);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let first_value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let second_value = var(&mut ssa, 2, 0, 2, MockType::I32);
    let addr = var(&mut ssa, 3, 0, 3, MockType::Ptr);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: first_value,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: second_value,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(instr(SsaOp::StoreIndirect {
        addr,
        value: first_value,
        value_type: MockType::I32,
        access: first,
    }));
    for extra in separator {
        b0.add_instruction(extra);
    }
    b0.add_instruction(store(addr, second_value));
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
    let (_, stores) = access_counts(&ssa);
    stores
}

/// A call between two stores to the same cell keeps the first alive.
///
/// This is the guarantee that makes dead store elimination safe for memory
/// visible outside the function: the callee may read whatever the first store
/// wrote, so the store is observable and must stay. Any call - direct, virtual,
/// or indirect - classifies as a barrier.
#[test]
fn a_call_between_stores_blocks_dead_store_elimination() {
    let call = instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(7u32),
        Vec::new(),
    )));
    assert_eq!(
        stores_around(AccessQualifiers::FLAT, vec![call]),
        2,
        "a callee may observe the first store"
    );
}

/// A volatile store a later store overwrites stays: a volatile write is
/// observable by definition (MMIO, a device register), so writing twice is two
/// distinct effects, not a redundant one.
#[test]
fn a_volatile_store_is_never_dead() {
    assert_eq!(
        stores_around(AccessQualifiers::FLAT.made_volatile(), Vec::new()),
        2,
        "a volatile access must not be coalesced away"
    );
}

/// A call touching no memory, and calling none of the program's code back,
/// still keeps the first store when its callee may throw: the exception unwinds to the caller, which may read the cell. The
/// control: the same call whose callee cannot throw leaves it dead.
#[test]
fn an_exception_a_call_may_raise_keeps_the_store_before_it() {
    let call = |may_throw: bool| {
        let effects = CallEffects {
            memory: CallMemory::Only(Vec::new()),
            calls_back: false,
            may_throw,
            ..CallEffects::unknown()
        };
        instr(SsaOp::from(
            CallData::new(None, Callee::Direct(7u32), Vec::new()).with_effects(effects),
        ))
    };
    assert_eq!(
        stores_around(AccessQualifiers::FLAT, vec![call(true)]),
        2,
        "the caller an exception unwinds to may read the first store"
    );
    assert_eq!(
        stores_around(AccessQualifiers::FLAT, vec![call(false)]),
        1,
        "a call that cannot throw neither reads nor leaves"
    );
}

/// With nothing in between, the first store really is dead - the control that
/// makes the two tests above meaningful.
#[test]
fn nothing_in_between_leaves_the_store_dead() {
    assert_eq!(stores_around(AccessQualifiers::FLAT, Vec::new()), 1);
}

/// A call between a store and a load blocks forwarding: the callee may have
/// rewritten the cell, so the stored value is not known to be there after it.
#[test]
fn a_call_blocks_forwarding() {
    let mut ssa = SsaFunction::new(0, 10);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let loaded = var(&mut ssa, 3, 0, 5, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(store(addr, value));
    b0.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(7u32),
        Vec::new(),
    ))));
    b0.add_instruction(load(loaded, addr));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(!changed, "a call may have rewritten the cell");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!((loads, stores), (1, 1));
    assert_eq!(ssa.validate(), Ok(()));
}

/// **A cache operation changes memory only where it may discard or zero it.**
/// A write-back, flush, demotion or eviction keeps every byte (`clwb`,
/// `cbo.clean`, `dc civac`), so a load after it is the value stored before it;
/// an invalidation that may drop dirty data, a zeroing and an operand-selected
/// data-cache operation do not let it through. Every one observes the store
/// before it, which stays: a flush moves the stored bytes on.
#[test]
fn a_cache_operation_changes_memory_only_where_it_may() {
    let cache = |op: CacheMaintenanceOp, addr: SsaVarId| {
        instr(SsaOp::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::CacheMaintenance(op),
            mnemonic: op.mnemonic().to_owned(),
            metadata: None,
            outputs: Vec::new(),
            inputs: vec![addr],
        })))
    };
    for op in CacheMaintenanceOp::all() {
        let mut ssa = SsaFunction::new(0, 10);
        let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
        let value = var(&mut ssa, 1, 0, 1, MockType::I32);
        let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
        let loaded = var(&mut ssa, 3, 0, 5, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: base,
            value: ConstValue::I32(0x1000),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(ptr_add(addr, base, 0));
        b0.add_instruction(store(addr, value));
        b0.add_instruction(cache(op, addr));
        b0.add_instruction(load(loaded, addr));
        b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
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
        assert_eq!(ssa.validate(), Ok(()), "{op:?}");
        let keeps = matches!(
            op,
            CacheMaintenanceOp::WriteBackInvalidateAll
                | CacheMaintenanceOp::WriteBackAll
                | CacheMaintenanceOp::FlushLine
                | CacheMaintenanceOp::WriteBackLine
                | CacheMaintenanceOp::DemoteLine
                | CacheMaintenanceOp::EvictLine
                | CacheMaintenanceOp::InstructionCacheOperation
        );
        let (loads, stores) = access_counts(&ssa);
        assert_eq!(
            (loads, stores),
            (usize::from(!keeps), 1),
            "{op:?}: the load forwards only past an operation that keeps memory"
        );
        assert_eq!(
            stores_around(AccessQualifiers::FLAT, vec![cache(op, base)]),
            2,
            "{op:?} observes the store before it"
        );
    }
}

/// Builds `b0: *(p+0) = v; <clobber>; jump b1` / `b1: w = *(p+0); return w`,
/// where the clobber sits in `b0` *after* the store.
fn cross_block_store_then_load(
    clobber: impl Fn(SsaVarId, SsaVarId) -> Vec<SsaInstruction<MockTarget>>,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let addr = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let loaded = var(&mut ssa, 3, 1, 0, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(store(addr, value));
    for extra in clobber(base, value) {
        b0.add_instruction(extra);
    }
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(load(loaded, addr));
    b1.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b1);
    ssa.recompute_uses();
    ssa
}

/// Sanity anchor: with nothing between the store and the load, cross-block
/// forwarding is legal and must still happen. Without this, a fix that simply
/// disabled cross-block seeding would look correct.
#[test]
fn cross_block_forwarding_happens_when_nothing_clobbers() {
    let mut ssa = cross_block_store_then_load(|_, _| Vec::new());

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    assert!(
        changed,
        "an unclobbered dominating store must still forward"
    );
    let (loads, _) = access_counts(&ssa);
    assert_eq!(loads, 0, "the load should have been forwarded");
    assert_eq!(ssa.validate(), Ok(()));
}

/// A fence between the store and the load, but in the *predecessor* block.
/// The fence versions only `MemoryLocation::Unknown`, so the loaded cell's
/// version at `b1`'s entry is unchanged and looks forwardable.
#[test]
fn a_fence_in_a_dominating_block_blocks_cross_block_forwarding() {
    let mut ssa = cross_block_store_then_load(|_, _| {
        vec![instr(SsaOp::Fence {
            kind: FenceKind::SeqCst,
        })]
    });

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    let (loads, stores) = access_counts(&ssa);
    assert_eq!(
        (loads, stores),
        (1, 1),
        "a fence in a dominating block must block forwarding across the edge"
    );
    assert!(!changed, "nothing may be rewritten past the fence");
    assert_eq!(ssa.validate(), Ok(()));
}

/// An overlapping store between the store and the load, again in the
/// predecessor block. `*(p+2)` as an i32 covers bits 16..48 of the cell at
/// `p+0`, so the earlier value does not describe the whole read - but it
/// versions only its own location, not the may-aliasing one being loaded.
#[test]
fn an_overlapping_store_in_a_dominating_block_blocks_cross_block_forwarding() {
    let mut ssa = cross_block_store_then_load(|base, value| {
        // Allocate the overlapping address in a scratch slot after the others.
        let overlap_addr = SsaVarId::from_index(4);
        vec![
            SsaInstruction::synthetic(SsaOp::PtrAdd {
                dest: overlap_addr,
                base,
                index: None,
                stride: 0,
                offset: 2,
                result_type: MockType::Ptr,
            }),
            store(overlap_addr, value),
        ]
    });
    // Register the scratch variable created above.
    let overlap_addr = ssa.create_variable(
        VariableOrigin::Local(4),
        0,
        DefSite::instruction(0, 4),
        MockType::Ptr,
    );
    assert_eq!(overlap_addr, SsaVarId::from_index(4));
    ssa.recompute_uses();

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    let (loads, _) = access_counts(&ssa);
    assert_eq!(
        loads, 1,
        "an overlapping store in a dominating block must block forwarding"
    );
    assert!(
        !changed,
        "the partially-overwritten cell may not be forwarded"
    );
    assert_eq!(ssa.validate(), Ok(()));
}

/// Declares the stack pointer the function is entered with, marked as such when
/// `declared` - the fact only a host can state.
fn entry_stack_pointer(ssa: &mut SsaFunction<MockTarget>, declared: bool) -> SsaVarId {
    let sp = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if declared && let Some(variable) = ssa.variable_mut(sp) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    sp
}

/// `*(sp + offset) = 7`, then `tail`, then return.
fn frame_write(
    offset: i64,
    declared: bool,
    tail: Vec<SsaInstruction<MockTarget>>,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, declared);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let slot = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(slot, sp, offset));
    b0.add_instruction(store(slot, value));
    for instruction in tail {
        b0.add_instruction(instruction);
    }
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// A write below the entry stack pointer that nothing reads before the function
/// returns is dead: the caller cannot reach that memory afterwards. This is a
/// callee-saved register save whose restore is gone.
#[test]
fn a_frame_write_nothing_reads_is_removed() {
    let mut ssa = frame_write(-8, true, Vec::new());
    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    assert!(changed, "an unread write into the frame is removed");
    assert_eq!(access_counts(&ssa).1, 0, "no store survives");
    assert_eq!(ssa.validate(), Ok(()));
}

/// **A frame word stored back unchanged is no write.** x86-64's stack probe
/// `or qword [rsp], 0` loads a word of the frame it just allocated and stores
/// the same value back; the frame's address goes to a call, so the store is
/// one the call may observe and no overwrite ends it, yet it changes nothing
/// the call could see, and goes. The controls keep it: a write through a
/// pointer that may name the word between the load and the store makes the
/// store a restore; and stored back through a pointer the function was
/// handed, the word is not the function's own (a device register written
/// back may act on it).
#[test]
fn a_frame_word_stored_back_unchanged_is_no_write() {
    // `between`: a write through the handed pointer before the store back;
    // `own`: the word is the frame's, else the handed pointer's.
    let run = |between: bool, own: bool| {
        let mut ssa = SsaFunction::new(1, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let handed = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let slot = var(&mut ssa, 1, 0, 0, MockType::Ptr);
        let loaded = var(&mut ssa, 2, 0, 1, MockType::I32);
        let seven = var(&mut ssa, 3, 0, 2, MockType::I32);
        let word = if own { slot } else { handed };
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(ptr_add(slot, sp, -8));
        b0.add_instruction(load(loaded, word));
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        if between {
            b0.add_instruction(store(handed, seven));
        }
        b0.add_instruction(store(word, loaded));
        b0.add_instruction(instr(SsaOp::from(CallData::new(
            None,
            Callee::Direct(7u32),
            vec![slot],
        ))));
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
        ssa.blocks()
            .iter()
            .flat_map(SsaBlock::instructions)
            .any(|row| matches!(row.op(), SsaOp::StoreIndirect { value, .. } if *value == loaded))
    };
    assert!(!run(false, true), "the probe's write-back goes");
    assert!(run(true, true), "a write between makes it a restore");
    assert!(
        run(false, false),
        "the handed pointer's word is not the frame's"
    );
}

/// **A word of the stack stored back unchanged in a probe loop is no
/// write.** A frame too large to probe in line is allocated by a loop
/// stepping the stack pointer down a page at a time and touching each (`sub
/// rsp, 0x1000; or qword [rsp], 0`): the touch's address is the loop's own
/// stack pointer, which no anchor of the frame places, but is the stack, an
/// address derived from the entry stack pointer, and the touch writes back
/// what it read. Stored back through a pointer the function was handed, the
/// word is not the stack's, and the store stays.
#[test]
fn a_stack_word_stored_back_in_a_probe_loop_is_no_write() {
    let run = |handed_base: bool| {
        let mut ssa = SsaFunction::new(2, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let handed = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let more = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let start = var(&mut ssa, 1, 0, 0, MockType::Ptr);
        let cursor = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::Ptr);
        let loaded = var(&mut ssa, 2, 1, 0, MockType::I32);
        let stepped = var(&mut ssa, 3, 1, 2, MockType::Ptr);
        let zero = var(&mut ssa, 4, 1, 3, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(ptr_add(start, if handed_base { handed } else { sp }, -8));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(cursor, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(start, 0));
        phi.add_operand(PhiOperand::new(stepped, 1));
        b1.add_phi(phi);
        b1.add_instruction(load(loaded, cursor));
        b1.add_instruction(store(cursor, loaded));
        b1.add_instruction(ptr_add(stepped, cursor, -16));
        b1.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: more,
            right: zero,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        for block in [b0, b1, b2] {
            ssa.add_block(block);
        }
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
        access_counts(&ssa).1
    };
    assert_eq!(run(false), 0, "the probe's write-back goes");
    assert_eq!(run(true), 1, "the handed pointer's word stays written");
}

/// **A frame word stored back after its load is forwarded keeps one store.**
/// gcc's `-O0` `OPAQUE(src)` loads a spilled pointer and stores it straight
/// back: `*slot = sym; v = *slot; *slot = v`, the address then handed to a
/// call. Once the load is forwarded, the second store writes what the first
/// put there; dropped as silent while the first was dropped as overwritten,
/// in one round, no store was left, and the call read a word nothing wrote
/// (`obfusc`'s `ob_xor_const`). Run until nothing changes, one store of the
/// value stays.
#[test]
fn a_frame_word_stored_back_after_its_load_keeps_one_store() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let slot = var(&mut ssa, 1, 0, 0, MockType::Ptr);
    let seven = var(&mut ssa, 2, 0, 1, MockType::I32);
    let loaded = var(&mut ssa, 3, 0, 3, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(instr(SsaOp::Const {
        dest: seven,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(store(slot, seven));
    b0.add_instruction(load(loaded, slot));
    b0.add_instruction(store(slot, loaded));
    b0.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(7u32),
        vec![slot],
    ))));
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    for _ in 0..8 {
        let changed = MemoryOptimizationPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true);
        assert_eq!(ssa.validate(), Ok(()));
        if !changed {
            break;
        }
    }
    let stores: Vec<SsaVarId> = ssa
        .blocks()
        .iter()
        .flat_map(SsaBlock::instructions)
        .filter_map(|row| match row.op() {
            SsaOp::StoreIndirect { value, .. } => Some(*value),
            _ => None,
        })
        .collect();
    assert_eq!(stores.len(), 1, "{ssa}");
}

/// A load nothing reads of the word at the entry stack pointer goes: it is
/// the caller's last stack word, which a realigning i386 `main` copies (`push
/// [ecx - 4]`, the return address), and reading it cannot fault. The control
/// reads one word above it, the caller's own memory, which may fault, and
/// stays.
#[test]
fn an_unread_load_of_the_entry_word_is_removed() {
    let run = |offset: i64| {
        let mut ssa = SsaFunction::new(0, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let cell = var(&mut ssa, 1, 0, 0, MockType::Ptr);
        let loaded = var(&mut ssa, 2, 0, 1, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(ptr_add(cell, sp, offset));
        b0.add_instruction(load(loaded, cell));
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
        access_counts(&ssa).0
    };
    assert_eq!(run(0), 0, "the entry word's load goes");
    assert_eq!(run(4), 0, "a part of the entry word is the entry word");
    assert_eq!(run(8), 1, "the control: the caller's memory above it stays");
}

/// **Release removes everything only the released operands kept, across the
/// renumbering each removal's repair makes.** The return hands its caller
/// `first`, `second` and `handed`, and `handed` copies a reload of a frame
/// slot. The first round removes those three; the reload is unread only then,
/// so a second round removes it and the slot's address, after the repair
/// moved the reload to an id the walk never reached. `kept`, which the return
/// returns, and `stray`, dead code the released operands never reached, stay.
#[test]
fn release_follows_what_it_reached_across_renumbering() -> Result<(), Box<dyn Error>> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    // Values of no local, which a repair renumbers once their neighbours go.
    let mut temporary =
        |index, ty| ssa.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(0, index), ty);
    let first = temporary(0, MockType::Ptr);
    let second = temporary(1, MockType::Ptr);
    let kept = temporary(2, MockType::Ptr);
    let stray = temporary(3, MockType::Ptr);
    let slot = temporary(4, MockType::Ptr);
    let reread = temporary(5, MockType::I32);
    let handed = temporary(6, MockType::I32);
    let handover = |raw, value| CallInput {
        key: CallInputKey::new(raw),
        value,
    };
    let mut b0 = SsaBlock::new(0);
    for dest in [first, second] {
        b0.add_instruction(instr(SsaOp::Copy { dest, src: sp }));
    }
    b0.add_instruction(instr(SsaOp::Copy {
        dest: kept,
        src: sp,
    }));
    b0.add_instruction(instr(SsaOp::Copy {
        dest: stray,
        src: sp,
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(load(reread, slot));
    b0.add_instruction(instr(SsaOp::Copy {
        dest: handed,
        src: reread,
    }));
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(kept),
        state: Box::default(),
        results: Default::default(),
        transfer: Some(Box::new(ReturnTransfer {
            target: None,
            handover: [handover(1, first), handover(2, second), handover(3, handed)].into(),
        })),
    }));
    ssa.add_block(b0);
    // Every instruction of a repaired function has an identity.
    ssa.repair_ssa();

    let removed = ssa.release_execution_operands(&AnalysisContext::new(MACHINE), |_| false)?;

    assert_eq!(ssa.validate(), Ok(()));
    let ops: Vec<_> = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .map(|instruction| instruction.op().clone())
        .collect();
    assert_eq!(removed, Some(5), "{ops:?}");
    assert!(
        matches!(
            ops.as_slice(),
            [
                SsaOp::Copy { .. },
                SsaOp::Copy { .. },
                SsaOp::Return {
                    value: Some(_),
                    transfer: None,
                    ..
                },
            ]
        ),
        "{ops:?}"
    );
    Ok(())
}

/// A frame load nothing reads yet stays when a later load of the same cell is
/// forwarded its result: the later load's readers read it once the forward
/// applies, and removed it would leave them reading nothing. A 32-bit x86
/// function reloads a spilled `double` twice after a merge, and only the
/// second reload was used.
#[test]
fn a_frame_load_a_forward_hands_on_is_kept() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let slot = var(&mut ssa, 1, 0, 0, MockType::Ptr);
    let first = var(&mut ssa, 2, 0, 1, MockType::I32);
    let second = var(&mut ssa, 3, 0, 2, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(load(first, slot));
    b0.add_instruction(load(second, slot));
    b0.add_instruction(instr(SsaOp::returns(Some(second))));
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
    assert_eq!(access_counts(&ssa).0, 1, "one load of the cell survives");
}

/// The control: the same write through a pointer nobody declared to be the
/// stack pointer is not known to be private, so it stays.
#[test]
fn an_undeclared_stack_pointer_makes_no_write_private() {
    let mut ssa = frame_write(-8, false, Vec::new());
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    assert_eq!(
        access_counts(&ssa).1,
        1,
        "without the declaration the base is just a pointer, and a write through it is observable"
    );
}

/// Memory at or above the entry stack pointer - the return address, stack
/// arguments, the caller's frame - outlives the function, and so does a cell
/// that only straddles the pointer.
#[test]
fn a_write_reaching_the_entry_stack_pointer_outlives_the_function() {
    for offset in [0, -2, 8] {
        let mut ssa = frame_write(offset, true, Vec::new());
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(
            access_counts(&ssa).1,
            1,
            "a 4-byte write at sp{offset:+} is not wholly inside the frame"
        );
    }
}

/// `*(sp + 0) = 7`, then a call owning `owned` with `access`, then `end`: an
/// Arm `_start`, which pops `argc` and pushes `stack_end` into the same word
/// before the call that passes it.
fn caller_word_then_call(
    owned: CallLocation<MockTarget>,
    access: AccessMode,
    end: SsaOp<MockTarget>,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(store(sp, value));
    b0.add_instruction(stated_call(vec![value], owned, access));
    b0.add_instruction(instr(end));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// A call whose argument bytes reach above the entry stack pointer owns them
/// as it owns the ones below: a write there that it carries as an operand is
/// none of its reads, and once it writes them they hold nothing of this
/// function's, for the caller either, so the write is dead whether the call
/// returns or not. The controls: a call that may leave the bytes as they were,
/// or one whose bytes end below the word, keeps the write for the caller.
#[test]
fn a_call_owning_the_caller_s_word_leaves_nothing_of_its_write() {
    let owning = |bytes: u64| CallLocation::Frame {
        origin: FrameOrigin::Entry,
        offset: -8,
        bytes,
    };
    for (bytes, access, end, kept) in [
        (12, AccessMode::Write, SsaOp::Unreachable, 0),
        (12, AccessMode::Write, SsaOp::returns(None), 0),
        (12, AccessMode::MayWrite, SsaOp::returns(None), 1),
        (8, AccessMode::Write, SsaOp::returns(None), 1),
    ] {
        let mut ssa = caller_word_then_call(owning(bytes), access, end.clone());
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(ssa.validate(), Ok(()));
        assert_eq!(
            access_counts(&ssa).1,
            kept,
            "owning {bytes} bytes from sp-8 as {access:?}, then {end:?}"
        );
    }
}

/// `*(sp + 0) = 7`, then `between`, then `*(sp + 0) = 8`, in one block or, with
/// `split`, across a jump, then return `between`'s last result when it has
/// one: a write into the caller's memory the function overwrites before it
/// leaves.
fn caller_word_overwritten(split: bool, between: Between) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let first = var(&mut ssa, 1, 0, 0, MockType::I32);
    let second = var(&mut ssa, 2, 0, 1, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: first,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: second,
        value: ConstValue::I32(8),
    }));
    b0.add_instruction(store(sp, first));
    let result = match between {
        Between::Nothing => None,
        Between::IncomingRead => {
            // A pointer the caller handed in.
            let incoming = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            let read = var(&mut ssa, 4, 0, 3, MockType::I32);
            b0.add_instruction(load(read, incoming));
            Some(read)
        }
        Between::Fault => {
            let read = var(&mut ssa, 4, 0, 3, MockType::I32);
            b0.add_instruction(instr(SsaOp::Div {
                dest: read,
                left: first,
                right: second,
                unsigned: false,
                flags: None,
            }));
            Some(read)
        }
        Between::ThrowingCall => {
            let effects = CallEffects {
                memory: CallMemory::Only(Vec::new()),
                calls_back: false,
                ..CallEffects::unknown()
            };
            b0.add_instruction(instr(SsaOp::from(
                CallData::new(None, Callee::Direct(7u32), Vec::new()).with_effects(effects),
            )));
            None
        }
    };
    let mut b1 = SsaBlock::new(1);
    let last = if split {
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        &mut b1
    } else {
        &mut b0
    };
    last.add_instruction(store(sp, second));
    last.add_instruction(instr(SsaOp::returns(result)));
    ssa.add_block(b0);
    if split {
        ssa.add_block(b1);
    }
    ssa.recompute_uses();
    ssa
}

/// What runs between the two writes of [`caller_word_overwritten`].
#[derive(Debug, Clone, Copy)]
enum Between {
    /// No instruction.
    Nothing,
    /// A load through a pointer the caller handed in, which may name the
    /// caller's word.
    IncomingRead,
    /// A division, which may fault.
    Fault,
    /// A call touching no memory, and running none of the program's code,
    /// whose callee may throw.
    ThrowingCall,
}

/// The caller's memory outlives the function, but a write the function itself
/// overwrites before it leaves is dead there too, across blocks as well. The
/// controls keep it: a read through a pointer the caller handed in may name
/// the word, and an exception a callee raises between the writes unwinds to
/// the caller with the first one in place. A fault is no exit to the caller
/// (`SsaOp::may_unwind`): outside a protected block it leaves the write dead.
#[test]
fn a_write_to_the_caller_s_word_overwritten_before_the_function_leaves_is_dead() {
    for split in [false, true] {
        for (between, kept) in [
            (Between::Nothing, 1),
            (Between::IncomingRead, 2),
            (Between::Fault, 1),
            (Between::ThrowingCall, 2),
        ] {
            let mut ssa = caller_word_overwritten(split, between);
            assert!(
                MemoryOptimizationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            assert_eq!(ssa.validate(), Ok(()));
            assert_eq!(
                access_counts(&ssa).1,
                kept,
                "with {between:?} between, split across blocks: {split}"
            );
        }
    }
}

/// A call whose callee was handed the frame address may read the write, so it
/// stays.
///
/// The address escapes as a call argument, which is the ordinary way one is
/// handed on. Everything a callee can name, it can name because something in
/// this function gave it the means.
#[test]
fn a_call_given_the_frame_address_keeps_the_write() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let slot = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(store(slot, value));
    b0.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(7u32),
        vec![slot],
    ))));
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

    assert_eq!(
        access_counts(&ssa).1,
        1,
        "the callee was given the address, so it may observe the write"
    );
}

/// A call keeps the write even when nothing escaped, because a callee reads its
/// stack arguments from the caller's frame.
///
/// The tempting relaxation, and the reason it is wrong. "Nothing escaped" makes
/// the frame private to a *pointer the function was handed* - that pointer was
/// formed before this frame existed - which is what
/// `FramePrivacy::may_name_frame` is for. It does **not** make the frame private
/// to a callee: a pushed argument sits below the entry stack pointer, in the
/// same region `own_frame_cell` claims, and the callee reads it by convention
/// without ever being given an address.
///
/// Telling the argument area from the saves needs the stack depth at the call,
/// which this pass does not have. So the barrier stays, and a store it keeps
/// alive is recoverable only by a host that states the call's memory
/// ([`CallLocation::Frame`]).
#[test]
fn a_call_keeps_the_write_because_of_its_stack_arguments() {
    let call = instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(7u32),
        Vec::new(),
    )));
    let mut ssa = frame_write(-8, true, vec![call]);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    assert_eq!(
        access_counts(&ssa).1,
        1,
        "a callee reads its stack arguments from this region"
    );
}

/// `*(phi - 4) = 7`, then return, where the phi joins `sp - 8` from one arm
/// and `sp + other` from the other. With `rewritten`, `*(sp - 12) = 8` comes
/// first and a call nobody described before the return. With `aligned`, every
/// `sp` is the entry stack pointer rounded down to 16.
fn merged_frame_write(other: i64, rewritten: bool, aligned: bool) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let entry = entry_stack_pointer(&mut ssa, true);
    let mut b0 = SsaBlock::new(0);
    let sp = if aligned {
        let mask = var(&mut ssa, 7, 0, 0, MockType::I64);
        let rounded = var(&mut ssa, 8, 0, 1, MockType::Ptr);
        b0.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I64(-16),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: rounded,
            left: entry,
            right: mask,
            flags: None,
        }));
        rounded
    } else {
        entry
    };
    let cond = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let left = var(&mut ssa, 1, 1, 0, MockType::Ptr);
    let right = var(&mut ssa, 2, 2, 0, MockType::Ptr);
    let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::Ptr);
    let value = var(&mut ssa, 3, 3, 0, MockType::I32);
    let slot = var(&mut ssa, 4, 3, 1, MockType::Ptr);
    if rewritten {
        let first_at = b0.instructions().len();
        let first = var(&mut ssa, 5, 0, first_at, MockType::I32);
        let first_slot = var(&mut ssa, 6, 0, first_at.saturating_add(1), MockType::Ptr);
        b0.add_instruction(instr(SsaOp::Const {
            dest: first,
            value: ConstValue::I32(8),
        }));
        b0.add_instruction(ptr_add(first_slot, sp, -12));
        b0.add_instruction(store(first_slot, first));
    }
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);
    for (index, dest, offset) in [(1, left, -8), (2, right, other)] {
        let mut block = SsaBlock::new(index);
        block.add_instruction(ptr_add(dest, sp, offset));
        block.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(block);
    }
    let mut b3 = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(left, 1));
    phi.add_operand(PhiOperand::new(right, 2));
    b3.phi_nodes_mut().push(phi);
    b3.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b3.add_instruction(ptr_add(slot, merged, -4));
    b3.add_instruction(store(slot, value));
    if rewritten {
        b3.add_instruction(instr(SsaOp::from(CallData::new(
            None,
            Callee::Direct(7u32),
            Vec::new(),
        ))));
    }
    b3.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b3);
    ssa.recompute_uses();
    ssa
}

/// A loop that moves the stack pointer by `step` each time round and then by
/// `back`, then `*(header - 4) = 7` and return after it: `sp - 8` enters the
/// header's phi, and the body's value comes back to it.
fn looped_frame_write(step: i64, back: i64) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let cond = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let entry = var(&mut ssa, 1, 0, 0, MockType::Ptr);
    let header = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::Ptr);
    let pushed = var(&mut ssa, 2, 2, 0, MockType::Ptr);
    let released = var(&mut ssa, 3, 2, 1, MockType::Ptr);
    let value = var(&mut ssa, 4, 3, 0, MockType::I32);
    let slot = var(&mut ssa, 5, 3, 1, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(ptr_add(entry, sp, -8));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(header, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(entry, 0));
    phi.add_operand(PhiOperand::new(released, 2));
    b1.phi_nodes_mut().push(phi);
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(ptr_add(pushed, header, step));
    b2.add_instruction(ptr_add(released, pushed, back));
    b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b2);
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b3.add_instruction(ptr_add(slot, header, -4));
    b3.add_instruction(store(slot, value));
    b3.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b3);
    ssa.recompute_uses();
    ssa
}

/// One byte of the frame is one cell whichever anchor names it: a write
/// through a stack pointer merged at one depth overwrites what a write through
/// the entry stack pointer left in the same bytes, so the call after both reads
/// only the second. The control: a merge at another depth names other bytes,
/// and the call reads both writes.
#[test]
fn a_merged_stack_pointer_overwrites_the_bytes_it_names() {
    let mut ssa = merged_frame_write(-8, true, false);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(access_counts(&ssa).1, 1, "the first write is overwritten");

    let mut ssa = merged_frame_write(-16, true, false);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(access_counts(&ssa).1, 2, "the writes name different bytes");
}

/// A phi joining addresses at one depth below a stack alignment is a place of
/// the frame measured from the alignment, as one joining addresses of the
/// entry stack pointer is measured from entry: `which-x86`'s `main` pushes every
/// call's arguments through the phi of its stack pointer. The controls are the
/// arms at different depths.
#[test]
fn a_stack_pointer_merged_below_an_alignment_addresses_the_frame() {
    let run = |other, rewritten| {
        let mut ssa = merged_frame_write(other, rewritten, true);
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(ssa.validate(), Ok(()));
        access_counts(&ssa).1
    };
    assert_eq!(run(-8, false), 0, "the write is the frame's and unread");
    assert_eq!(run(-16, false), 1, "the phi has no one place");
    assert_eq!(run(-8, true), 1, "the first write is overwritten");
    assert_eq!(run(-16, true), 2, "the writes name different bytes");
}

/// Two arms that leave the stack pointer at one depth join it in a phi, and
/// a write through the phi is a cell of the frame like a write through either
/// arm's value: nothing reads it, so it goes, as it does through the header of
/// a loop that leaves the stack pointer where it found it. The controls: arms
/// leaving it at different depths, and a loop sinking it each time round, join
/// an address of no one place, and the write through it stays.
#[test]
fn a_stack_pointer_merged_at_one_depth_addresses_the_frame() {
    let mut ssa = merged_frame_write(-8, false, false);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        access_counts(&ssa).1,
        0,
        "the write is the frame's and unread"
    );

    let mut ssa = merged_frame_write(-16, false, false);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(access_counts(&ssa).1, 1, "the phi has no one place");

    let mut ssa = looped_frame_write(-16, 16);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(access_counts(&ssa).1, 0, "a balanced loop leaves one place");

    let mut ssa = looped_frame_write(-16, 8);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa).1,
        1,
        "a loop sinking the stack has none"
    );
}

/// A call given `args` that states what it touches: escaped memory, and
/// `frame` with `access`.
fn stated_call(
    args: Vec<SsaVarId>,
    frame: CallLocation<MockTarget>,
    access: AccessMode,
) -> SsaInstruction<MockTarget> {
    let effects = CallEffects {
        memory: CallMemory::Only(vec![
            CallAccess {
                location: CallLocation::Escaped,
                access: AccessMode::Update,
            },
            CallAccess {
                location: frame,
                access,
            },
        ]),
        ..CallEffects::unknown()
    };
    instr(SsaOp::from(
        CallData::new(None, Callee::Direct(7u32), args).with_effects(effects),
    ))
}

/// A call carrying its stack arguments as operands reads none of the writes
/// that staged them: what it states it reaches of the frame is escaped memory,
/// which a frame whose address never left the function is not, and the
/// argument bytes it may overwrite. The write at `sp - 8` is the push a 32-bit
/// call is made after, and nothing else reads it.
///
/// The controls: the same call stating that it reads the bytes keeps the
/// write, and one reading the bytes just below it does not. In a frame whose
/// address escaped, the bytes the call owns are still read only as it says,
/// while the escaped local's write stays; owning other bytes, the call reads
/// the argument's write as escaped memory.
#[test]
fn a_call_carrying_its_stack_arguments_does_not_keep_their_writes() {
    let arguments = CallLocation::Frame {
        origin: FrameOrigin::Entry,
        offset: -8,
        bytes: 4,
    };
    let carried = stated_call(Vec::new(), arguments.clone(), AccessMode::MayWrite);
    let mut ssa = frame_write(-8, true, vec![carried]);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(access_counts(&ssa).1, 0, "the call carries the value");

    let read = stated_call(Vec::new(), arguments.clone(), AccessMode::Read);
    let mut ssa = frame_write(-8, true, vec![read]);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa).1,
        1,
        "the call reads the bytes it names"
    );

    let below = CallLocation::Frame {
        origin: FrameOrigin::Entry,
        offset: -16,
        bytes: 8,
    };
    let read_below = stated_call(Vec::new(), below, AccessMode::Read);
    let mut ssa = frame_write(-8, true, vec![read_below]);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa).1,
        0,
        "the bytes it reads end below the write"
    );

    for (owned, kept) in [(-8, 1), (-32, 2)] {
        let mut ssa = escaped_frame_call(CallLocation::Frame {
            origin: FrameOrigin::Entry,
            offset: owned,
            bytes: 4,
        });
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(ssa.validate(), Ok(()));
        assert_eq!(
            access_counts(&ssa).1,
            kept,
            "with the frame's address escaped, the call reads the local and the \
             argument bytes only as it says it does (owning sp{owned:+})"
        );
    }
}

/// `*(sp - 8) = 7` and `*(sp - 16) = 7`, then a call handed `sp - 16` owning
/// the 4 bytes at `sp - 8` with `access`, then another handed it owning the 4
/// at `sp - 24`, then return.
fn calls_after_an_argument(access: AccessMode) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let argument = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let local = var(&mut ssa, 3, 0, 3, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(argument, sp, -8));
    b0.add_instruction(store(argument, value));
    b0.add_instruction(ptr_add(local, sp, -16));
    b0.add_instruction(store(local, value));
    let owned = CallLocation::Frame {
        origin: FrameOrigin::Entry,
        offset: -8,
        bytes: 4,
    };
    b0.add_instruction(stated_call(vec![local], owned, access));
    let later = CallLocation::Frame {
        origin: FrameOrigin::Entry,
        offset: -24,
        bytes: 4,
    };
    b0.add_instruction(stated_call(vec![local], later, AccessMode::Write));
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// A callee's parameters are its own: a call writing its argument bytes leaves
/// nothing of this function's in them, so a later call reaching the escaped
/// frame does not read the push that staged them. The control: a call that may
/// leave them as they were hands the push on to the later call's escaped
/// memory.
#[test]
fn a_call_leaves_nothing_in_the_argument_bytes_it_writes() {
    let mut ssa = calls_after_an_argument(AccessMode::Write);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(access_counts(&ssa).1, 1, "only the local's write stays");

    let mut ssa = calls_after_an_argument(AccessMode::MayWrite);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(access_counts(&ssa).1, 2, "the later call may read the push");
}

/// `*(sp - 8) = 7` and `*(sp - 16) = 7`, then a call handed `sp - 16` that
/// states escaped memory and `owned`, which it may overwrite, then return.
fn escaped_frame_call(owned: CallLocation<MockTarget>) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let argument = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let local = var(&mut ssa, 3, 0, 3, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(argument, sp, -8));
    b0.add_instruction(store(argument, value));
    b0.add_instruction(ptr_add(local, sp, -16));
    b0.add_instruction(store(local, value));
    b0.add_instruction(stated_call(vec![local], owned, AccessMode::MayWrite));
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// `*(sp - 8) = saved`, a call, then `*(sp - 8)` read back into nothing, then
/// return: a register save, the call the function makes, and the restore. With
/// `caller_state`, the store is declared a save of the caller's state
/// ([`InstructionRole::CallerStateSave`]).
fn save_around_a_call(caller_state: bool) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let saved = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::I64,
    );
    let slot = var(&mut ssa, 2, 0, 0, MockType::Ptr);
    let restored = var(&mut ssa, 3, 0, 3, MockType::I64);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(if caller_state {
        save(slot, saved)
    } else {
        store(slot, saved)
    });
    b0.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(7u32),
        Vec::new(),
    ))));
    b0.add_instruction(load(restored, slot));
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// A register save is the caller's state, and no callee reads it: the call
/// between the save and the restore no longer keeps it, and the restore, which
/// nothing reads and which cannot fault in the function's own frame, goes too.
/// The control stores ordinary data, which a callee may read as a stack
/// argument; the dead restore still goes.
#[test]
fn a_call_does_not_keep_a_register_save() {
    let mut ssa = save_around_a_call(true);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(access_counts(&ssa), (0, 0), "the save and the restore go");
    assert_eq!(ssa.validate(), Ok(()));

    let mut ssa = save_around_a_call(false);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa),
        (0, 1),
        "the dead load goes, and the store a callee may read stays"
    );
}

/// Escaped memory holds no register save. A call stating that it reaches
/// escaped memory, in a frame whose local it was handed, reads the local and
/// not the save its restore reads at a fixed offset: the save and the restore
/// go as they do around a call nobody described.
#[test]
fn escaped_memory_does_not_hold_a_register_save() {
    let mut ssa = save_in_an_escaped_frame(SaveReader::Incoming, true);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa),
        (1, 1),
        "the read and the local's write stay; the save and its restore go"
    );
    assert_eq!(ssa.validate(), Ok(()));
}

/// What the read in [`save_in_an_escaped_frame`] goes through.
#[derive(Clone, Copy)]
enum SaveReader {
    /// A pointer the caller handed in.
    Incoming,
    /// The local's address indexed by a number the caller handed in: an
    /// element of the local.
    Element,
    /// The save's own address, a fixed offset from the entry stack pointer.
    Save,
}

/// A register save at `sp - 8`, a local at `sp - 16` whose address a call is
/// handed, a read through `reader` whose value the function returns, and a
/// restore read back into nothing.
fn save_in_an_escaped_frame(reader: SaveReader, stated: bool) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let saved = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::I64,
    );
    let incoming = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    let index = ssa.create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::I64,
    );
    let save_slot = var(&mut ssa, 3, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 4, 0, 2, MockType::I32);
    let local = var(&mut ssa, 5, 0, 3, MockType::Ptr);
    let element = var(&mut ssa, 6, 0, 6, MockType::Ptr);
    let loaded = var(&mut ssa, 7, 0, 7, MockType::I32);
    let restored = var(&mut ssa, 8, 0, 8, MockType::I64);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(ptr_add(save_slot, sp, -8));
    b0.add_instruction(save(save_slot, saved));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(local, sp, -16));
    b0.add_instruction(store(local, value));
    b0.add_instruction(if stated {
        let arguments = CallLocation::Frame {
            origin: FrameOrigin::Entry,
            offset: -24,
            bytes: 8,
        };
        stated_call(vec![local], arguments, AccessMode::MayWrite)
    } else {
        instr(SsaOp::from(CallData::new(
            None,
            Callee::Direct(7u32),
            vec![local],
        )))
    });
    b0.add_instruction(instr(SsaOp::PtrAdd {
        dest: element,
        base: local,
        index: Some(index),
        stride: 1,
        offset: 0,
        result_type: MockType::Ptr,
    }));
    b0.add_instruction(load(
        loaded,
        match reader {
            SaveReader::Incoming => incoming,
            SaveReader::Element => element,
            SaveReader::Save => save_slot,
        },
    ));
    b0.add_instruction(load(restored, save_slot));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// A frame whose local escapes keeps its local's write, which the callee may
/// read, but not its register save. A pointer the function was handed reaches
/// the frame only through the escaped address, which is the local's, and an
/// element of the local is the local's; a save is no object, and a pointer to
/// one reaches none. The control reads the save's own cell, which keeps it.
#[test]
fn an_escaped_local_does_not_keep_a_register_save() {
    for reader in [SaveReader::Incoming, SaveReader::Element] {
        let mut ssa = save_in_an_escaped_frame(reader, false);
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(
            access_counts(&ssa),
            (1, 1),
            "the read and the local's write stay; the save and its restore go"
        );
        assert_eq!(ssa.validate(), Ok(()));
    }

    let mut ssa = save_in_an_escaped_frame(SaveReader::Save, false);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa).1,
        2,
        "a read of the save's own cell keeps the save"
    );
}

/// A block copy reads its source and nothing else, so one from a pointer the
/// function was handed reads what that pointer can reach of the frame: the
/// escaped local, never the register save. The control copies from the save's
/// own address, which keeps it.
#[test]
fn a_block_copy_from_outside_does_not_keep_a_register_save() {
    for from_outside in [true, false] {
        let mut ssa = SsaFunction::new(0, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let saved = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I64,
        );
        let incoming = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let destination = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let save_slot = var(&mut ssa, 3, 0, 0, MockType::Ptr);
        let local = var(&mut ssa, 4, 0, 2, MockType::Ptr);
        let size = var(&mut ssa, 5, 0, 4, MockType::I64);
        let restored = var(&mut ssa, 6, 0, 6, MockType::I64);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(ptr_add(save_slot, sp, -8));
        b0.add_instruction(save(save_slot, saved));
        b0.add_instruction(ptr_add(local, sp, -16));
        b0.add_instruction(instr(SsaOp::from(CallData::new(
            None,
            Callee::Direct(7u32),
            vec![local],
        ))));
        b0.add_instruction(instr(SsaOp::Const {
            dest: size,
            value: ConstValue::I64(16),
        }));
        b0.add_instruction(instr(SsaOp::CopyBlk {
            dest_addr: destination,
            src_addr: if from_outside { incoming } else { save_slot },
            size,
            element_bytes: 1,
            backward: None,
            dest_access: AccessQualifiers::FLAT,
            src_access: AccessQualifiers::FLAT,
        }));
        b0.add_instruction(load(restored, save_slot));
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
        assert_eq!(
            access_counts(&ssa).1,
            usize::from(!from_outside),
            "copied from outside, the save goes; copied from itself, it stays"
        );
    }
}

/// `aligned = sp & -16`, a register save at `aligned - 8`, a call, and then a
/// read through the entry stack pointer at `read_offset`, when there is one,
/// whose value the function returns.
fn realigned_save(read_offset: Option<i64>) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let saved = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::I64,
    );
    let number = var(&mut ssa, 1, 0, 0, MockType::I64);
    let mask = var(&mut ssa, 2, 0, 1, MockType::I64);
    let rounded = var(&mut ssa, 3, 0, 2, MockType::I64);
    let aligned = var(&mut ssa, 4, 0, 3, MockType::Ptr);
    let save_slot = var(&mut ssa, 5, 0, 4, MockType::Ptr);
    let read_slot = var(&mut ssa, 6, 0, 7, MockType::Ptr);
    let loaded = var(&mut ssa, 7, 0, 8, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::PtrToInt {
        dest: number,
        operand: sp,
        target: MockType::I64,
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: mask,
        value: ConstValue::I64(-16),
    }));
    b0.add_instruction(instr(SsaOp::And {
        dest: rounded,
        left: number,
        right: mask,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::IntToPtr {
        dest: aligned,
        operand: rounded,
        target: MockType::Ptr,
    }));
    b0.add_instruction(ptr_add(save_slot, aligned, -8));
    b0.add_instruction(save(save_slot, saved));
    b0.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(7u32),
        Vec::new(),
    ))));
    match read_offset {
        Some(offset) => {
            b0.add_instruction(ptr_add(read_slot, sp, offset));
            b0.add_instruction(load(loaded, read_slot));
            b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
        }
        None => b0.add_instruction(instr(SsaOp::returns(None))),
    }
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// A frame realigned below the entry stack pointer is the function's own, and
/// its register save goes as one below the entry stack pointer does: the call
/// reads no save, and an argument read at or above the entry stack pointer
/// lies above every cell. The control reads below the entry stack pointer,
/// within the alignment of the save, where the realigned anchor may have put
/// it.
#[test]
fn a_realigned_frame_does_not_keep_its_register_save() {
    for read_offset in [None, Some(0), Some(8)] {
        let mut ssa = realigned_save(read_offset);
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(
            access_counts(&ssa).1,
            0,
            "with a read at {read_offset:?}, the save goes"
        );
        assert_eq!(ssa.validate(), Ok(()));
    }
    let mut ssa = realigned_save(Some(-16));
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa).1,
        1,
        "a read below the entry stack pointer may read the save"
    );
}

/// A pointer the function was handed cannot alias its own frame, so a read
/// through one does not keep the write.
///
/// The other access's base is a value the caller supplied, and the frame did
/// not exist when the caller formed it.
/// `IndirectLocation::may_alias` cannot know that - it is a pure function on two
/// locations, sees two different SSA roots, and must answer "may".
#[test]
fn a_read_through_an_incoming_pointer_does_not_keep_the_frame_write() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let slot = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    // No defining instruction: a pointer the caller handed in.
    let incoming = var(&mut ssa, 3, 0, 0, MockType::Ptr);
    let loaded = var(&mut ssa, 4, 0, 3, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(store(slot, value));
    b0.add_instruction(load(loaded, incoming));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
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

    assert_eq!(
        access_counts(&ssa).1,
        0,
        "the incoming pointer predates this frame, so it cannot name a cell in it"
    );
}

/// A write the function reads back is program state, however private.
#[test]
fn a_frame_write_read_back_is_kept() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let slot = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let loaded = var(&mut ssa, 3, 0, 3, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(store(slot, value));
    b0.add_instruction(load(loaded, slot));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
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

    assert_eq!(
        access_counts(&ssa).1,
        1,
        "the write is read before the return"
    );
    assert_eq!(ssa.validate(), Ok(()));
}

/// A read *earlier* in the block still observes the write when a loop brings
/// control back around to it: the walk follows the back edge.
#[test]
fn a_frame_write_read_on_a_later_iteration_is_kept() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let again = var(&mut ssa, 2, 0, 1, MockType::I32);
    let slot = var(&mut ssa, 3, 0, 2, MockType::Ptr);
    let loaded = var(&mut ssa, 4, 1, 0, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: again,
        value: ConstValue::True,
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(load(loaded, slot));
    b1.add_instruction(store(slot, value));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: again,
        true_target: 1,
        false_target: 2,
    }));
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.add_block(b1);
    ssa.add_block(b2);
    ssa.recompute_uses();
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    assert_eq!(
        access_counts(&ssa).1,
        1,
        "the next iteration's load reads what this iteration stored"
    );
}

/// Two exits, neither of which reads the cell: the write is dead on every path.
/// The positive that proves the walk crosses blocks rather than giving up at
/// the first branch.
#[test]
fn a_frame_write_unread_on_every_path_is_removed() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let either = var(&mut ssa, 2, 0, 1, MockType::I32);
    let slot = var(&mut ssa, 3, 0, 2, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: either,
        value: ConstValue::True,
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(store(slot, value));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: either,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);
    for index in [1, 2] {
        let mut exit = SsaBlock::new(index);
        exit.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(exit);
    }
    ssa.recompute_uses();
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    assert_eq!(
        access_counts(&ssa).1,
        0,
        "no path reads the cell before leaving"
    );
}

/// An indirect branch whose targets are not all known may continue anywhere,
/// including somewhere that reads the frame, so a write before it stays.
#[test]
fn a_frame_write_before_an_unresolved_indirect_branch_is_kept() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let slot = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(store(slot, value));
    b0.add_instruction(instr(SsaOp::IndirectBranch {
        target: value,
        resolved_targets: vec![1],
        state: Box::default(),
    }));
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::returns(None)));
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

    assert_eq!(
        access_counts(&ssa).1,
        1,
        "control may leave the resolved targets for code that reads the frame"
    );
}

/// A forwarded load whose only use is a phi operand leaves valid SSA.
///
/// ```text
/// b0: *(p) = 7; branch -> b1, b2
/// b1: v = *(p); jump b3          <- served by b0's store
/// b2: w = 8;    jump b3
/// b3: m = phi(v from b1, w from b2); return m
/// ```
///
/// The load is redundant: the stored `7` reaches it from the dominating block.
/// Its one use is the phi operand on the `b1` edge, which `7` also reaches.
/// A forward that replaced instruction uses only would count the replacement
/// complete and delete the load, leaving the phi reading a value nothing
/// defines, and the function would fail verification.
#[test]
fn forwarding_into_a_phi_operand_keeps_the_ssa_valid() {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let cond = var(&mut ssa, 2, 0, 2, MockType::I32);
    let addr = var(&mut ssa, 3, 0, 3, MockType::Ptr);
    let loaded = var(&mut ssa, 4, 1, 0, MockType::I32);
    let other = var(&mut ssa, 5, 2, 0, MockType::I32);
    let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(ptr_add(addr, base, 0));
    b0.add_instruction(store(addr, value));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(load(loaded, addr));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: other,
        value: ConstValue::I32(8),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    let mut b3 = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(loaded, 1));
    phi.add_operand(PhiOperand::new(other, 2));
    b3.add_phi(phi);
    b3.add_instruction(instr(SsaOp::returns(Some(merged))));
    for block in [b0, b1, b2, b3] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the function starts valid");

    let changed = MemoryOptimizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    assert!(changed, "the load is served by the dominating store");
    assert_eq!(
        ssa.validate(),
        Ok(()),
        "no operand is left naming the deleted load"
    );
    assert_eq!(access_counts(&ssa).0, 0, "the redundant load is gone");
    let merged_from_b1 = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.phi_nodes())
        .find_map(|phi| phi.operand_from(1))
        .map(PhiOperand::value);
    assert_eq!(
        merged_from_b1,
        Some(value),
        "the edge now carries the stored value"
    );
}

/// One exception clause over block ranges, with no filter.
fn clause(
    try_start: usize,
    try_end: usize,
    handler_start: usize,
    handler_end: usize,
) -> SsaExceptionHandler<MockTarget> {
    SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(try_start, try_end),
        handler_range: BlockRange::new(handler_start, handler_end),
        filter_range: None,
        selector: None,
    }
}

/// `*(sp - 8) = 7`, then whatever `tail` builds from the slot address and the
/// stored value, then return.
fn frame_write_then(
    tail: impl FnOnce(
        &mut SsaFunction<MockTarget>,
        SsaVarId,
        SsaVarId,
    ) -> Vec<SsaInstruction<MockTarget>>,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let slot = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(store(slot, value));
    for instruction in tail(&mut ssa, slot, value) {
        b0.add_instruction(instruction);
    }
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// Builds a reading operation from `(dest, slot, value)`.
type Reader = fn(SsaVarId, SsaVarId, SsaVarId) -> SsaOp<MockTarget>;

/// Builds instructions that publish the slot address and read a pointer back.
type Publish = fn(&mut SsaFunction<MockTarget>, SsaVarId) -> Vec<SsaInstruction<MockTarget>>;

/// An access Memory SSA cannot place - an atomic, a vector load, a block copy -
/// classifies to the `Unknown` location, and `Unknown` may be the frame cell.
///
/// Each of these reads the slot the write just filled (`AtomicLoad`, `movups
/// xmm0, [rsp-8]`, `rep movsb` out of the slot), so deleting the write changes
/// what they read.
#[test]
fn an_unmodelled_read_of_a_frame_cell_keeps_the_write() {
    let readers: [(&str, Reader); 4] = [
        ("atomic load", |dest, slot, _| SsaOp::AtomicLoad {
            dest,
            addr: slot,
            value_type: MockType::I32,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            reserves: false,
            access: AccessQualifiers::FLAT,
        }),
        ("vector load", |dest, slot, _| SsaOp::VectorLoad {
            dest,
            addr: slot,
            vector_type: MockType::V4I32,
            access: AccessQualifiers::FLAT,
        }),
        ("block copy", |dest, slot, size| SsaOp::CopyBlk {
            dest_addr: dest,
            src_addr: slot,
            size,
            element_bytes: 1,
            backward: None,
            dest_access: AccessQualifiers::FLAT,
            src_access: AccessQualifiers::FLAT,
        }),
        ("compare-exchange", |dest, slot, value| {
            SsaOp::AtomicCmpXchg {
                old: dest,
                success: None,
                addr: slot,
                expected: value,
                desired: value,
                success_ordering: AtomicOrdering::SeqCst,
                failure_ordering: AtomicOrdering::SeqCst,
                width: AtomicAccessWidth::Bits32,
                weak: false,
                access: AccessQualifiers::FLAT,
            }
        }),
    ];
    for (name, reader) in readers {
        let mut ssa = frame_write_then(|ssa, slot, value| {
            let dest = var(ssa, 3, 0, 3, MockType::Ptr);
            vec![instr(reader(dest, slot, value))]
        });
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(
            access_counts(&ssa).1,
            1,
            "the {name} reads the slot, so the write into it is observed"
        );
    }
}

/// Builds a function whose second protected block writes the frame and then
/// leaves by `exit`, with a handler that reads the slot back.
///
/// ```text
/// b0: slot = sp - 8; jump b1
/// b1: jump b2                  <- protected [b1, b3), the guard
/// b2: *slot = 7; <exit>        <- protected, not the guard
/// b3: r = *slot; <handler end> <- the handler
/// b4: return
/// ```
fn protected_frame_write(
    exit: impl FnOnce(&mut SsaFunction<MockTarget>, SsaVarId) -> Vec<SsaInstruction<MockTarget>>,
    handler_end: SsaOp<MockTarget>,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let slot = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let read = var(&mut ssa, 3, 3, 0, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(store(slot, value));
    for instruction in exit(&mut ssa, value) {
        b2.add_instruction(instruction);
    }
    // The handler hands what it reads to the caller, through a pointer the
    // caller passed in: a load nothing reads would be removed, and read nothing.
    let out = var(&mut ssa, 6, 0, 0, MockType::Ptr);
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(load(read, slot));
    b3.add_instruction(store(out, read));
    b3.add_instruction(instr(handler_end));
    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::returns(None)));
    for block in [b0, b1, b2, b3, b4] {
        ssa.add_block(block);
    }
    ssa.set_exception_handlers(vec![clause(1, 3, 3, 4)]);
    ssa.recompute_uses();
    ssa
}

/// A throw from *any* block of a protected region reaches its handler, not only
/// a throw from the guard that the exception-aware CFG draws the edge from.
///
/// Three ways out of the region's second block: a load that may fault, an
/// explicit `Throw`, and a `Leave` that runs the handler on the way out. The
/// handler reads the slot on every one of them.
#[test]
fn a_handler_reached_from_a_later_protected_block_observes_the_frame_write() {
    let faulting_load = protected_frame_write(
        |ssa, _| {
            let incoming = var(ssa, 4, 0, 0, MockType::Ptr);
            let loaded = var(ssa, 5, 2, 1, MockType::I32);
            vec![load(loaded, incoming), instr(SsaOp::returns(Some(loaded)))]
        },
        SsaOp::returns(None),
    );
    let throw = protected_frame_write(
        |_, value| vec![instr(SsaOp::Throw { exception: value })],
        SsaOp::returns(None),
    );
    let leave = protected_frame_write(
        |_, _| vec![instr(SsaOp::Leave { target: 4 })],
        SsaOp::EndFinally,
    );
    for (name, mut ssa) in [
        ("a faulting load", faulting_load),
        ("a throw", throw),
        ("a leave", leave),
    ] {
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(
            access_counts(&ssa).1,
            2,
            "after {name}, the handler reads the slot the region wrote"
        );
    }
}

/// An overwrite ends a path only if control actually reaches it. A fault in
/// front of it sends control to the handler with the first write still in the
/// slot.
///
/// ```text
/// b0: *slot = 1; jump b1
/// b1: x = *incoming; *slot = 2; return x   <- protected; the load may fault
/// b2: r = *slot; return                   <- the handler
/// ```
#[test]
fn a_fault_before_the_overwrite_lets_the_handler_read_the_first_write() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let first = var(&mut ssa, 1, 0, 0, MockType::I32);
    let second = var(&mut ssa, 2, 0, 1, MockType::I32);
    let slot = var(&mut ssa, 3, 0, 2, MockType::Ptr);
    let incoming = var(&mut ssa, 4, 0, 0, MockType::Ptr);
    let loaded = var(&mut ssa, 5, 1, 0, MockType::I32);
    let read = var(&mut ssa, 6, 2, 0, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: first,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: second,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(ptr_add(slot, sp, -8));
    b0.add_instruction(store(slot, first));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(load(loaded, incoming));
    b1.add_instruction(store(slot, second));
    b1.add_instruction(instr(SsaOp::returns(Some(loaded))));
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(load(read, slot));
    b2.add_instruction(instr(SsaOp::returns(Some(read))));
    for block in [b0, b1, b2] {
        ssa.add_block(block);
    }
    ssa.set_exception_handlers(vec![clause(1, 2, 2, 3)]);
    ssa.recompute_uses();
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    let first_survives = ssa.block(0).is_some_and(|block| {
        block.instructions().iter().any(|instruction| {
            matches!(instruction.op(), SsaOp::StoreIndirect { value, .. } if *value == first)
        })
    });
    assert!(
        first_survives,
        "the handler reads the first write when the load faults\n{ssa}"
    );
}

/// The block-local overwrite rule has the same obligation: inside a protected
/// block, a fault between two writes of one cell hands the first to the
/// handler.
///
/// ```text
/// b0: jump b1
/// b1: *p = 1; q = 1 / 0; *p = q; return   <- protected
/// b2: r = *p; return r                    <- the handler
/// ```
#[test]
fn a_fault_between_two_writes_in_a_protected_block_keeps_the_first() {
    let mut ssa = SsaFunction::new(0, 8);
    let pointer = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let one = var(&mut ssa, 1, 0, 0, MockType::I32);
    let zero = var(&mut ssa, 2, 0, 1, MockType::I32);
    let quotient = var(&mut ssa, 3, 1, 1, MockType::I32);
    let read = var(&mut ssa, 4, 2, 0, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(store(pointer, one));
    b1.add_instruction(instr(SsaOp::Div {
        dest: quotient,
        left: one,
        right: zero,
        unsigned: false,
        flags: None,
    }));
    b1.add_instruction(store(pointer, quotient));
    b1.add_instruction(instr(SsaOp::returns(None)));
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(load(read, pointer));
    b2.add_instruction(instr(SsaOp::returns(Some(read))));
    for block in [b0, b1, b2] {
        ssa.add_block(block);
    }
    ssa.set_exception_handlers(vec![clause(1, 2, 2, 3)]);
    ssa.recompute_uses();
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    assert_eq!(
        access_counts(&ssa).1,
        2,
        "the division may fault after the first write, and the handler reads it"
    );
}

/// A frame address handed on through any operand other than an access's own
/// address leaves the function, and a pointer read back from wherever it went
/// may name the slot.
///
/// ```text
/// *slot = 7; <publish slot>; p = <read it back>; x = *p; return
/// ```
#[test]
fn a_frame_address_leaving_through_any_operand_keeps_the_write() {
    let shapes: [(&str, Publish); 3] = [
        ("a static field", |ssa, slot| {
            let back = var(ssa, 4, 0, 4, MockType::Ptr);
            let read = var(ssa, 5, 0, 5, MockType::I32);
            vec![
                instr(SsaOp::StoreStaticField {
                    access: AccessQualifiers::FLAT,
                    field: 1u32,
                    value: slot,
                }),
                instr(SsaOp::LoadStaticField {
                    access: AccessQualifiers::FLAT,
                    dest: back,
                    field: 1u32,
                }),
                load(read, back),
            ]
        }),
        ("an instance field", |ssa, slot| {
            let object = var(ssa, 6, 0, 0, MockType::Ref);
            let back = var(ssa, 4, 0, 4, MockType::Ptr);
            let read = var(ssa, 5, 0, 5, MockType::I32);
            vec![
                instr(SsaOp::StoreField {
                    access: AccessQualifiers::FLAT,
                    object,
                    field: 1u32,
                    value: slot,
                }),
                instr(SsaOp::LoadField {
                    access: AccessQualifiers::FLAT,
                    dest: back,
                    object,
                    field: 1u32,
                }),
                load(read, back),
            ]
        }),
        ("an array element", |ssa, slot| {
            let array = var(ssa, 6, 0, 0, MockType::Ref);
            let index = var(ssa, 7, 0, 0, MockType::I32);
            let back = var(ssa, 4, 0, 4, MockType::Ptr);
            let read = var(ssa, 5, 0, 5, MockType::I32);
            vec![
                instr(SsaOp::StoreElement {
                    array,
                    index,
                    value: slot,
                    elem_type: MockType::Ptr,
                }),
                instr(SsaOp::LoadElement {
                    dest: back,
                    array,
                    index,
                    elem_type: MockType::Ptr,
                }),
                load(read, back),
            ]
        }),
    ];
    for (name, publish) in shapes {
        let mut ssa = frame_write_then(|ssa, slot, _| publish(ssa, slot));
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(
            access_counts(&ssa).1,
            1,
            "the slot's address escaped through {name}, so a pointer read back may name it"
        );
    }
}

/// `*(sp - 8) = 7; *(sp - 16) = 9; f(sp - 8, …); return`, the call stating
/// `effects` and passed the first cell's address in the positions `passed`
/// names (`true`) or another value (`false`), after a call that returns
/// twice where `jumped_back`; returns the stores left.
fn frame_cells_around_a_call(
    effects: Option<CallEffects<MockTarget>>,
    passed: &[bool],
    jumped_back: bool,
) -> usize {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let seven = var(&mut ssa, 1, 0, 0, MockType::I32);
    let nine = var(&mut ssa, 2, 0, 1, MockType::I32);
    let passed_cell = var(&mut ssa, 3, 0, 2, MockType::Ptr);
    let other_cell = var(&mut ssa, 4, 0, 3, MockType::Ptr);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: seven,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: nine,
        value: ConstValue::I32(9),
    }));
    b0.add_instruction(ptr_add(passed_cell, sp, -8));
    b0.add_instruction(ptr_add(other_cell, sp, -16));
    b0.add_instruction(store(passed_cell, seven));
    b0.add_instruction(store(other_cell, nine));
    if jumped_back {
        let mut setjmp = CallEffects::unknown();
        setjmp.memory = CallMemory::Only(vec![CallAccess {
            location: CallLocation::Escaped,
            access: AccessMode::Update,
        }]);
        setjmp.returns_twice = true;
        b0.add_instruction(instr(SsaOp::Call(Box::new(CallData {
            dest: None,
            callee: Callee::Direct(8),
            args: Vec::new(),
            implicit_inputs: Default::default(),
            results: Default::default(),
            effects: Some(setjmp),
            tail: false,
            staged: Vec::new(),
        }))));
    }
    b0.add_instruction(instr(SsaOp::Call(Box::new(CallData {
        dest: None,
        callee: Callee::Direct(9),
        args: passed
            .iter()
            .map(|cell| if *cell { passed_cell } else { seven })
            .collect(),
        implicit_inputs: Default::default(),
        results: Default::default(),
        effects,
        tail: false,
        staged: Vec::new(),
    }))));
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
    access_counts(&ssa).1
}

/// **A call that does not keep a frame address reaches only what it states
/// through it.** AAPCS64 hands a `va_list` on as the address of a copy in the
/// caller's frame, and a callee that only reads it with `va_arg` reads and
/// steps the copy while it runs: the call states that access
/// ([`CallLocation::Pointee`]) and keeps nothing it is passed there
/// ([`Capture::AnyBut`]), so the frame's other cells stay the function's own,
/// and a write to one nothing reads goes, while the copy the call reads stays.
/// The controls keep both writes: a call that may keep the address, one whose
/// memory is not listed (which could reach the frame through it in a way it
/// does not state), and one passed the address again where it may keep it.
#[test]
fn a_call_that_keeps_no_frame_address_reaches_only_what_it_states() {
    let reading = |captures: Capture, memory: bool| {
        let mut effects = CallEffects::unknown();
        if memory {
            effects.memory = CallMemory::Only(vec![
                CallAccess {
                    location: CallLocation::Pointee {
                        arg: 0,
                        extent: PointeeExtent::Bytes(4),
                    },
                    access: AccessMode::Update,
                },
                CallAccess {
                    location: CallLocation::Escaped,
                    access: AccessMode::Update,
                },
            ]);
        }
        effects.captures = captures;
        Some(effects)
    };
    assert_eq!(
        frame_cells_around_a_call(reading(Capture::Any.sparing(0), true), &[true], false),
        1,
        "the cell the call reads stays, the other goes"
    );
    assert_eq!(
        frame_cells_around_a_call(
            reading(Capture::Any.sparing(0), true),
            &[true, false],
            false
        ),
        1,
        "another argument the call may keep is no frame address"
    );
    assert_eq!(
        frame_cells_around_a_call(reading(Capture::Any, true), &[true], false),
        2,
        "a call that may keep the address"
    );
    assert_eq!(
        frame_cells_around_a_call(reading(Capture::Any.sparing(0), false), &[true], false),
        2,
        "a call whose memory is unknown"
    );
    assert_eq!(
        frame_cells_around_a_call(reading(Capture::Any.sparing(0), true), &[true, true], false),
        2,
        "the address passed again where the call may keep it"
    );
}

/// **A count a call states its reach by is the constant its argument holds.**
/// `memcpy(dst, src, n)` writes `n` bytes from `dst`, and a call passing a
/// constant count reaches that many: reading four bytes from the lower of two
/// frame cells reaches nothing of the upper one, and the write there nothing
/// reads goes. The controls keep it: a count of sixteen reaches it, and so
/// does a count no constant states, which bounds nothing.
#[test]
fn a_count_a_call_states_its_reach_by_is_its_constant_argument() {
    let stores_left = |count: Option<i32>| {
        let mut ssa = SsaFunction::new(0, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let seven = var(&mut ssa, 1, 0, 0, MockType::I32);
        let lower = var(&mut ssa, 2, 0, 1, MockType::Ptr);
        let upper = var(&mut ssa, 3, 0, 2, MockType::Ptr);
        let counted = var(&mut ssa, 4, 0, 3, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(ptr_add(lower, sp, -16));
        b0.add_instruction(ptr_add(upper, sp, -8));
        b0.add_instruction(store(lower, seven));
        b0.add_instruction(store(upper, seven));
        match count {
            Some(bytes) => b0.add_instruction(instr(SsaOp::Const {
                dest: counted,
                value: ConstValue::I32(bytes),
            })),
            None => b0.add_instruction(instr(SsaOp::LoadIndirect {
                dest: counted,
                addr: sp,
                value_type: MockType::I32,
                access: Default::default(),
            })),
        }
        let mut effects = CallEffects::unknown();
        effects.memory = CallMemory::Only(vec![
            CallAccess {
                location: CallLocation::Pointee {
                    arg: 0,
                    extent: PointeeExtent::InArgument(1),
                },
                access: AccessMode::Read,
            },
            CallAccess {
                location: CallLocation::Escaped,
                access: AccessMode::Update,
            },
        ]);
        effects.captures = Capture::Any.sparing(0);
        b0.add_instruction(instr(SsaOp::Call(Box::new(CallData {
            dest: None,
            callee: Callee::Direct(9),
            args: vec![lower, counted],
            implicit_inputs: Default::default(),
            results: Default::default(),
            effects: Some(effects),
            tail: false,
            staged: Vec::new(),
        }))));
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
        access_counts(&ssa).1
    };
    assert_eq!(
        stores_left(Some(4)),
        1,
        "four bytes reach the lower cell alone"
    );
    assert_eq!(stores_left(Some(16)), 2, "sixteen reach both");
    assert_eq!(
        stores_left(None),
        2,
        "a count no constant states bounds nothing"
    );
}

/// **A function that calls one returning twice lets its frame escape.** A
/// later call may jump back to where `setjmp` returns, and the code there
/// reads the frame as that call leaves it: dash's `main` sets its `state`
/// before each profile it reads and reads it where `setjmp` returns again.
/// No edge leads there from the call, so the frame is read by every call,
/// and a write to a cell the call is not lent stays. The control: without
/// the call that returns twice, it goes.
#[test]
fn a_function_calling_one_that_returns_twice_lets_its_frame_escape() {
    let mut lending = CallEffects::unknown();
    lending.memory = CallMemory::Only(vec![
        CallAccess {
            location: CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::Bytes(4),
            },
            access: AccessMode::Update,
        },
        CallAccess {
            location: CallLocation::Escaped,
            access: AccessMode::Update,
        },
    ]);
    lending.captures = Capture::Any.sparing(0);
    assert_eq!(
        frame_cells_around_a_call(Some(lending.clone()), &[true], true),
        2,
        "a later call may resume where setjmp returns"
    );
    assert_eq!(
        frame_cells_around_a_call(Some(lending), &[true], false),
        1,
        "the control: nothing returns twice"
    );
}

/// A value *loaded* from the frame is whatever was stored there, not an address
/// of the frame: returning it hands nothing of the frame to the caller.
///
/// ```text
/// *(sp-16) = 7; *(sp-8) = 7; x = *incoming; r = *(sp-16); return r
/// ```
///
/// The write to `sp-8` is read by nothing, and the read through `incoming`
/// cannot name it, since no frame address left the function.
#[test]
fn a_value_loaded_from_the_frame_is_not_a_frame_address() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let value = var(&mut ssa, 1, 0, 0, MockType::I32);
    let kept = var(&mut ssa, 2, 0, 1, MockType::Ptr);
    let dead = var(&mut ssa, 3, 0, 2, MockType::Ptr);
    let incoming = var(&mut ssa, 4, 0, 0, MockType::Ptr);
    let through = var(&mut ssa, 5, 0, 5, MockType::I32);
    let read = var(&mut ssa, 6, 0, 6, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(ptr_add(kept, sp, -16));
    b0.add_instruction(ptr_add(dead, sp, -8));
    b0.add_instruction(store(kept, value));
    b0.add_instruction(store(dead, value));
    b0.add_instruction(load(through, incoming));
    b0.add_instruction(load(read, kept));
    b0.add_instruction(instr(SsaOp::returns(Some(read))));
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

    assert_eq!(
        access_counts(&ssa).1,
        1,
        "only the write the function reads back survives\n{ssa}"
    );
}

/// Builds `*(&local1) = 7; *other = 9; w = *(&local1); return w`, where
/// `other` is `&local2` when `slot_other` holds and an argument - a pointer
/// nothing is known about - otherwise.
fn two_stores_then_reload(slot_other: bool) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(1, 8);
    let first = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let other = var(&mut ssa, 1, 0, 1, MockType::Ptr);
    let seven = var(&mut ssa, 2, 0, 2, MockType::I32);
    let nine = var(&mut ssa, 3, 0, 3, MockType::I32);
    let loaded = var(&mut ssa, 4, 0, 6, MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::LoadLocalAddr {
        dest: first,
        local_index: 1,
    }));
    b0.add_instruction(instr(if slot_other {
        SsaOp::LoadLocalAddr {
            dest: other,
            local_index: 2,
        }
    } else {
        SsaOp::LoadArg {
            dest: other,
            arg_index: 0,
        }
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: seven,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: nine,
        value: ConstValue::I32(9),
    }));
    b0.add_instruction(store(first, seven));
    b0.add_instruction(store(other, nine));
    b0.add_instruction(load(loaded, first));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// **Two slots are two cells.** A store to local 2 cannot change local 1, so
/// the reload of local 1 is the 7 stored to it. The control stores through an
/// argument instead, a pointer that may hold local 1's address: the reload
/// stays.
#[test]
fn a_store_to_another_slot_keeps_a_slots_value() {
    let mut ssa = two_stores_then_reload(true);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa).0,
        0,
        "the reload of local 1 is forwarded"
    );
    assert_eq!(ssa.validate(), Ok(()));

    let mut control = two_stores_then_reload(false);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut control,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&control).0,
        1,
        "control: a store through an unknown pointer may land in local 1"
    );
}

/// Builds `*(u8 *)p = v; w = *(width *)p; return w` with `v` a 32-bit value,
/// so the store writes only its low byte (BOUNDARIES D69).
fn narrow_store_then_load(load_type: MockType) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let base = var(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = var(&mut ssa, 1, 0, 1, MockType::I32);
    let loaded = var(&mut ssa, 2, 0, 3, load_type);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(0x1234_5678),
    }));
    b0.add_instruction(instr(SsaOp::StoreIndirect {
        addr: base,
        value,
        value_type: MockType::U8,
        access: AccessQualifiers::FLAT,
    }));
    b0.add_instruction(instr(SsaOp::LoadIndirect {
        dest: loaded,
        addr: base,
        value_type: load_type,
        access: AccessQualifiers::FLAT,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// **A byte store of a wider value forwards to a byte load as the value's low
/// byte.** The store writes only the bits its width names, so the load reads
/// back exactly those: a truncation of the stored value. The control reads two
/// bytes, one of which the store did not write.
#[test]
fn a_narrow_store_forwards_the_low_part_of_its_value() {
    let mut ssa = narrow_store_then_load(MockType::U8);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&ssa),
        (0, 1),
        "the load reads what the store wrote"
    );
    let narrowed = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .any(|instruction| {
            matches!(
                instruction.op(),
                SsaOp::IntConv {
                    target: MockType::U8,
                    ..
                }
            )
        });
    assert!(
        narrowed,
        "the forwarded value is the stored one narrowed to the byte read"
    );
    assert_eq!(ssa.validate(), Ok(()));

    let mut control = narrow_store_then_load(MockType::U16);
    assert!(
        MemoryOptimizationPass
            .run(
                &mut control,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert_eq!(
        access_counts(&control),
        (1, 1),
        "control: a wider read is not the byte the store wrote"
    );
}

/// A function saving a register and spilling a value into its frame, running
/// `between`, and returning: `*(sp - 8) = saved` (a save), `*(sp - 16) = 7`
/// (a spill nothing reloads), `between`, `return`. Returns the stores left.
fn frame_stores_past(between: SsaInstruction<MockTarget>, inputs: bool) -> usize {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let saved = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::I64,
    );
    let save_slot = var(&mut ssa, 2, 0, 0, MockType::Ptr);
    let spill_slot = var(&mut ssa, 3, 0, 2, MockType::Ptr);
    let seven = var(&mut ssa, 4, 0, 3, MockType::I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(ptr_add(save_slot, sp, -8));
    b0.add_instruction(save(save_slot, saved));
    b0.add_instruction(ptr_add(spill_slot, sp, -16));
    b0.add_instruction(instr(SsaOp::Const {
        dest: seven,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(store(spill_slot, seven));
    let mut between = between;
    if inputs && let SsaOp::SystemOp(data) = between.op_mut() {
        data.inputs = vec![seven];
    }
    b0.add_instruction(between);
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
    access_counts(&ssa).1
}

/// A system operation of `kind` with no operands yet.
fn system(kind: SystemOpKind) -> SsaInstruction<MockTarget> {
    instr(SsaOp::SystemOp(Box::new(NativeKindedData {
        kind,
        mnemonic: String::new(),
        metadata: None,
        outputs: Vec::new(),
        inputs: Vec::new(),
    })))
}

/// **A move into a register the machine holds keeps no frame cell alive.**
/// A Cortex-M critical section writes `basepri` (`msr basepri, r3`) between
/// its frame pointer's save and its return: the move reads no memory, so the
/// save and a spill nothing reloads both go. Read as a read and write of
/// every cell, it kept both. Every register move of the family reads none
/// (`csrrw`'s exchange, `csrrs`'s set bits); the control is a coprocessor
/// load (`ldc`), which reads the memory its address names, and keeps them.
#[test]
fn a_register_move_keeps_no_frame_cell() {
    for op in SysRegOp::all().filter(|op| !op.moves_memory()) {
        assert_eq!(
            frame_stores_past(system(SystemOpKind::ControlRegister(op)), true),
            0,
            "{op:?}"
        );
    }
    assert_eq!(
        frame_stores_past(
            system(SystemOpKind::ControlRegister(SysRegOp::CoprocessorLoad)),
            true
        ),
        2,
        "control: a coprocessor load may read either cell"
    );
}

/// **A fence keeps no frame cell alive while no address of the frame
/// escapes.** `isb; dsb` order this thread's accesses and read nothing, so
/// a spill nothing reloads before the return is dead past them, as the
/// save is. The control is a call nobody described, which may read the
/// spill as a stack argument, though not the save.
#[test]
fn a_fence_keeps_no_frame_cell() {
    assert_eq!(
        frame_stores_past(
            instr(SsaOp::Fence {
                kind: FenceKind::SeqCst,
            }),
            false
        ),
        0
    );
    assert_eq!(
        frame_stores_past(
            instr(SsaOp::Call(Box::new(CallData {
                dest: None,
                callee: Callee::Direct(9),
                args: Vec::new(),
                implicit_inputs: Default::default(),
                results: Default::default(),
                effects: None,
                tail: false,
                staged: Vec::new(),
            }))),
            false
        ),
        1,
        "control: a call may read the spill"
    );
}

/// **A cache operation on one line keeps no register save alive.**
/// `*(sp - 8) = saved` (a save), a cache operation, then `*(sp - 8)` read back
/// into nothing: `clflush [rdi+32]` in x86-64 code that saves `rbx` around
/// `cpuid`. The line operation reads the line its address names, and the
/// address is a pointer the function was handed, which reaches no register
/// save, so the save and its dead restore go. The controls keep the save: a
/// whole-cache clean names no address and may read anything, and a line
/// cleaned through the frame itself may hold the save.
#[test]
fn a_cache_line_operation_keeps_no_register_save() {
    let run = |op: CacheMaintenanceOp, through_frame: bool| {
        let mut ssa = SsaFunction::new(0, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let saved = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I64,
        );
        let handed = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let slot = var(&mut ssa, 2, 0, 0, MockType::Ptr);
        let line = var(&mut ssa, 4, 0, 2, MockType::Ptr);
        let restored = var(&mut ssa, 3, 0, 4, MockType::I64);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(ptr_add(slot, sp, -8));
        b0.add_instruction(save(slot, saved));
        b0.add_instruction(ptr_add(line, if through_frame { sp } else { handed }, 32));
        b0.add_instruction(instr(SsaOp::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::CacheMaintenance(op),
            mnemonic: op.mnemonic().to_owned(),
            metadata: None,
            outputs: Vec::new(),
            inputs: vec![line],
        }))));
        b0.add_instruction(load(restored, slot));
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
        assert_eq!(ssa.validate(), Ok(()), "{op:?}");
        access_counts(&ssa).1
    };
    for op in CacheMaintenanceOp::all().filter(|op| op.names_a_line()) {
        assert_eq!(run(op, false), 0, "{op:?} through a handed pointer");
        assert_eq!(run(op, true), 1, "control: {op:?} through the frame");
    }
    assert_eq!(
        run(CacheMaintenanceOp::WriteBackAll, false),
        1,
        "control: a whole-cache clean"
    );
}

/// **A call in an aligned frame names the bytes it owns from the alignment.**
/// The argument store below the aligned value is what the call writes, so a
/// later call reaching the escaped frame does not read it, and only the local's
/// write stays. The control names the same offset from entry, a different cell:
/// the argument store is then still handed on to the later call.
#[test]
fn a_call_owns_the_bytes_it_names_below_the_alignment() {
    let run = |origin: FrameOrigin| {
        let mut ssa = SsaFunction::new(0, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let mask = var(&mut ssa, 2, 0, 0, MockType::I64);
        let anchor = var(&mut ssa, 3, 0, 1, MockType::Ptr);
        let value = var(&mut ssa, 4, 0, 2, MockType::I32);
        let argument = var(&mut ssa, 5, 0, 3, MockType::Ptr);
        let local = var(&mut ssa, 6, 0, 5, MockType::Ptr);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(-16),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: anchor,
            left: sp,
            right: mask,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(ptr_add(argument, anchor, -8));
        b0.add_instruction(store(argument, value));
        b0.add_instruction(ptr_add(local, anchor, -16));
        b0.add_instruction(store(local, value));
        let owned = CallLocation::Frame {
            origin,
            offset: -8,
            bytes: 4,
        };
        b0.add_instruction(stated_call(vec![local], owned, AccessMode::Write));
        let later = CallLocation::Frame {
            origin: FrameOrigin::Alignment,
            offset: -24,
            bytes: 4,
        };
        b0.add_instruction(stated_call(vec![local], later, AccessMode::Write));
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
        access_counts(&ssa).1
    };
    assert_eq!(
        run(FrameOrigin::Alignment),
        1,
        "only the local's write stays"
    );
    assert_eq!(run(FrameOrigin::Entry), 2, "control: another cell");
}

/// **A call's read of the frame above its arguments keeps no register save.**
/// A callee nothing bounds may read every byte above the ones it is passed,
/// as stack arguments no one stated, so a call states that read; a register
/// save is no argument, so it keeps the save no more than a barrier does. The
/// control: ordinary data stored in the same range stays, which the callee
/// may read.
#[test]
fn a_frame_read_above_the_arguments_keeps_no_register_save() {
    let run = |caller_state: bool| {
        let mut ssa = SsaFunction::new(0, 8);
        let sp = entry_stack_pointer(&mut ssa, true);
        let saved = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I64,
        );
        let slot = var(&mut ssa, 2, 0, 0, MockType::Ptr);
        let restored = var(&mut ssa, 3, 0, 3, MockType::I64);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(ptr_add(slot, sp, -8));
        b0.add_instruction(if caller_state {
            save(slot, saved)
        } else {
            store(slot, saved)
        });
        let effects = CallEffects {
            memory: CallMemory::Only(vec![
                CallAccess {
                    location: CallLocation::Escaped,
                    access: AccessMode::Update,
                },
                CallAccess {
                    location: CallLocation::Frame {
                        origin: FrameOrigin::Entry,
                        offset: -16,
                        bytes: 1 << 28,
                    },
                    access: AccessMode::Read,
                },
                CallAccess {
                    location: CallLocation::Frame {
                        origin: FrameOrigin::Entry,
                        offset: -24,
                        bytes: 8,
                    },
                    access: AccessMode::Write,
                },
            ]),
            ..CallEffects::unknown()
        };
        b0.add_instruction(instr(SsaOp::from(
            CallData::new(None, Callee::Direct(7u32), Vec::new()).with_effects(effects),
        )));
        b0.add_instruction(load(restored, slot));
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
        access_counts(&ssa).1
    };
    assert_eq!(run(true), 0, "the save goes");
    assert_eq!(run(false), 1, "control: data the callee may read stays");
}

/// What comes between the two stores of a copy in parts
/// ([`a_value_copied_in_parts_is_stored_whole`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CopyGap {
    /// Nothing: the parts are stored one after the other.
    Nothing,
    /// A store into the bytes the copy fills.
    Write,
    /// A load of the bytes the copy fills.
    Read,
    /// The parts swapped: the low half stored high and the high half low.
    Swapped,
}

/// A 128-bit value stored at `0x1000`, read back as two 64-bit halves and the
/// halves stored at `0x2000` in the order they were read, with `between`
/// coming between the two stores, then returned: the stores that survive, as
/// the value each stores and the type it stores it as.
fn copy_in_parts(between: CopyGap) -> (SsaVarId, Vec<(SsaVarId, MockType)>) {
    let mut ssa = SsaFunction::new(0, 8);
    let wide = var(&mut ssa, 0, 0, 0, MockType::U128);
    let source = var(&mut ssa, 1, 0, 1, MockType::Ptr);
    let source_high = var(&mut ssa, 2, 0, 2, MockType::Ptr);
    let target = var(&mut ssa, 3, 0, 3, MockType::Ptr);
    let target_high = var(&mut ssa, 4, 0, 4, MockType::Ptr);
    let low = var(&mut ssa, 5, 0, 6, MockType::U64);
    let high = var(&mut ssa, 6, 0, 7, MockType::U64);
    let other = (between == CopyGap::Read).then(|| var(&mut ssa, 7, 0, 8, MockType::U64));
    let half = |addr: SsaVarId, value: SsaVarId| {
        instr(SsaOp::StoreIndirect {
            addr,
            value,
            value_type: MockType::U64,
            access: AccessQualifiers::FLAT,
        })
    };
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: wide,
        value: ConstValue::I64(7),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: source,
        value: ConstValue::I64(0x1000),
    }));
    b0.add_instruction(ptr_add(source_high, source, 8));
    b0.add_instruction(instr(SsaOp::Const {
        dest: target,
        value: ConstValue::I64(0x2000),
    }));
    b0.add_instruction(ptr_add(target_high, target, 8));
    b0.add_instruction(instr(SsaOp::StoreIndirect {
        addr: source,
        value: wide,
        value_type: MockType::U128,
        access: AccessQualifiers::FLAT,
    }));
    for (dest, addr) in [(low, source), (high, source_high)] {
        b0.add_instruction(instr(SsaOp::LoadIndirect {
            dest,
            addr,
            value_type: MockType::U64,
            access: AccessQualifiers::FLAT,
        }));
    }
    let (first, second) = match between {
        CopyGap::Swapped => ((target, high), (target_high, low)),
        _ => ((target, low), (target_high, high)),
    };
    b0.add_instruction(half(first.0, first.1));
    match between {
        CopyGap::Write => b0.add_instruction(half(target_high, low)),
        CopyGap::Read => {
            if let Some(dest) = other {
                b0.add_instruction(instr(SsaOp::LoadIndirect {
                    dest,
                    addr: target,
                    value_type: MockType::U64,
                    access: AccessQualifiers::FLAT,
                }));
            }
        }
        CopyGap::Nothing | CopyGap::Swapped => b0.add_instruction(instr(SsaOp::Nop)),
    }
    b0.add_instruction(half(second.0, second.1));
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
    let stores = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter_map(|instruction| match instruction.op() {
            SsaOp::StoreIndirect {
                addr,
                value,
                value_type,
                ..
            } if *addr != source => Some((*value, *value_type)),
            _ => None,
        })
        .collect();
    (wide, stores)
}

/// **A value copied in parts is stored whole.** A 128-bit value no load can
/// be served a part of, read back as two halves and the halves stored
/// together at another place in the order they lie, is that value stored
/// there: one store of it, as AAPCS64 copies a `va_list` in two vector moves.
/// The controls keep the parts: a store into the copy's bytes between its two
/// stores, a read of them there, and the halves stored the wrong way round,
/// which is no copy of the value.
#[test]
fn a_value_copied_in_parts_is_stored_whole() {
    let (wide, stores) = copy_in_parts(CopyGap::Nothing);
    assert_eq!(
        stores,
        vec![(wide, MockType::U128)],
        "the two halves are one store of the value"
    );
    for between in [CopyGap::Write, CopyGap::Read, CopyGap::Swapped] {
        let (wide, stores) = copy_in_parts(between);
        assert!(
            stores.len() >= 2 && stores.iter().all(|(value, _)| *value != wide),
            "control {between:?}: the parts stay ({stores:?})"
        );
    }
}

/// A vector load nothing reads of the function's own frame goes, as a scalar
/// one does: reading the frame cannot fault, and a copy whose stores became
/// one store of the value it copied leaves its loads unread.
#[test]
fn an_unread_vector_load_of_the_frame_is_removed() {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let cell = var(&mut ssa, 1, 0, 0, MockType::Ptr);
    let loaded = var(&mut ssa, 2, 0, 1, MockType::V4I32);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(ptr_add(cell, sp, -32));
    b0.add_instruction(instr(SsaOp::VectorLoad {
        dest: loaded,
        addr: cell,
        vector_type: MockType::V4I32,
        access: AccessQualifiers::FLAT,
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
    let vector_loads = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter(|instruction| matches!(instruction.op(), SsaOp::VectorLoad { .. }))
        .count();
    assert_eq!(vector_loads, 0, "the unread vector load of the frame goes");
}

/// Where [`spill_across_an_escaped_call`] keeps its spill, and what it does
/// with the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Spill {
    /// At `sp - 8`, read back in the block that spills it.
    OneBlock,
    /// At `sp - 8`, read back in the next block.
    AcrossBlocks,
    /// At `sp - 8`, whose address the call is handed and may keep.
    Escaping,
    /// At `sp + 8`, the caller's memory.
    CallerCell,
    /// At `(sp & -16) - 8`, in a realigned frame.
    Realigned,
}

/// `*slot = incoming`, a call stating that it updates escaped memory, then
/// `*slot` read back and returned, with the slot `spill` names.
fn spill_across_an_escaped_call(spill: Spill) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let sp = entry_stack_pointer(&mut ssa, true);
    let incoming = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let mut b0 = SsaBlock::new(0);
    let base = if spill == Spill::Realigned {
        let mask = var(&mut ssa, 2, 0, 0, MockType::I64);
        let aligned = var(&mut ssa, 3, 0, 1, MockType::Ptr);
        b0.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I64(-16),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: aligned,
            left: sp,
            right: mask,
            flags: None,
        }));
        aligned
    } else {
        sp
    };
    let at = b0.instructions().len();
    let slot = var(&mut ssa, 4, 0, at, MockType::Ptr);
    let offset = if spill == Spill::CallerCell { 8 } else { -8 };
    b0.add_instruction(ptr_add(slot, base, offset));
    b0.add_instruction(store(slot, incoming));
    let effects = CallEffects {
        memory: CallMemory::Only(vec![CallAccess {
            location: CallLocation::Escaped,
            access: AccessMode::Update,
        }]),
        ..CallEffects::unknown()
    };
    let args = if spill == Spill::Escaping {
        vec![slot]
    } else {
        Vec::new()
    };
    b0.add_instruction(instr(SsaOp::from(
        CallData::new(None, Callee::Direct(7u32), args).with_effects(effects),
    )));
    let (block, at) = if spill == Spill::AcrossBlocks {
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        (1, 0)
    } else {
        (0, b0.instructions().len())
    };
    let restored = var(&mut ssa, 5, block, at, MockType::I32);
    if spill == Spill::AcrossBlocks {
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(load(restored, slot));
        b1.add_instruction(instr(SsaOp::returns(Some(restored))));
        ssa.add_block(b1);
    } else {
        b0.add_instruction(load(restored, slot));
        b0.add_instruction(instr(SsaOp::returns(Some(restored))));
        ssa.add_block(b0);
    }
    ssa.recompute_uses();
    ssa
}

/// **A call that reaches only escaped memory leaves a private frame's cells
/// alone.** It touches what it was handed and what the program let escape,
/// and carries its stack arguments as operands, so a value spilled across it
/// in a frame whose address it never saw is still there: ARM's `push {r1, lr};
/// bl raise; pop {r1, pc}` gives the caller's `r1` back. Within a block, from
/// a dominating block, and in a realigned frame alike. The controls keep the
/// load: a frame whose address the call may keep, and a cell of the caller's
/// memory above the entry stack pointer.
#[test]
fn a_call_reaching_escaped_memory_leaves_a_private_frame_alone() {
    let loads = |spill: Spill| {
        let mut ssa = spill_across_an_escaped_call(spill);
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(ssa.validate(), Ok(()), "{spill:?}");
        access_counts(&ssa).0
    };
    assert_eq!(loads(Spill::OneBlock), 0, "forwarded within the block");
    assert_eq!(
        loads(Spill::AcrossBlocks),
        0,
        "forwarded from the dominator"
    );
    assert_eq!(loads(Spill::Realigned), 0, "forwarded in a realigned frame");
    assert_eq!(loads(Spill::Escaping), 1, "the call may write the frame");
    assert_eq!(
        loads(Spill::CallerCell),
        1,
        "the caller's memory may escape"
    );
}

/// **A joined value stored in one block is read back in the block it
/// dominates.** A phi defines its result as an instruction defines its
/// destination: `*p = phi(1, 2)` then a jump and `*p` read back is the phi.
/// The control stores a constant, which an instruction defines.
#[test]
fn a_joined_value_is_forwarded_from_a_dominating_store() {
    // b0: branch b1 | b2; b1, b2: const, jump b3; b3: merged = phi; *p =
    // merged (or 9); jump b4; b4: loaded = *p; return loaded.
    let joined = |stored_phi: bool| {
        let mut ssa = SsaFunction::new(1, 0);
        let cond = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let pointer = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let left = var(&mut ssa, 1, 1, 0, MockType::I32);
        let right = var(&mut ssa, 2, 2, 0, MockType::I32);
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
        let nine = var(&mut ssa, 3, 3, 0, MockType::I32);
        let loaded = var(&mut ssa, 4, 4, 0, MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        for (index, dest, value) in [(1, left, 1), (2, right, 2)] {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            }));
            block.add_instruction(instr(SsaOp::Jump { target: 3 }));
            ssa.add_block(block);
        }
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(left, 1));
        phi.add_operand(PhiOperand::new(right, 2));
        b3.phi_nodes_mut().push(phi);
        b3.add_instruction(instr(SsaOp::Const {
            dest: nine,
            value: ConstValue::I32(9),
        }));
        b3.add_instruction(store(pointer, if stored_phi { merged } else { nine }));
        b3.add_instruction(instr(SsaOp::Jump { target: 4 }));
        ssa.add_block(b3);
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(load(loaded, pointer));
        b4.add_instruction(instr(SsaOp::returns(Some(loaded))));
        ssa.add_block(b4);
        ssa.recompute_uses();
        assert!(
            MemoryOptimizationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert_eq!(ssa.validate(), Ok(()), "stored_phi {stored_phi}");
        access_counts(&ssa).0
    };
    assert_eq!(joined(true), 0, "the phi is forwarded");
    assert_eq!(joined(false), 0, "the constant is forwarded");
}
