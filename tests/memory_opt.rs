//! Memory optimization pass: store-to-load forwarding, redundant load
//! elimination, and dead store elimination.
//!
//! Each positive case is paired with the negative that proves the alias gate or
//! barrier is doing the work — a pass that rewrote unconditionally would pass
//! every positive test here.

use analyssa::{
    PointerSize,
    events::NullListener,
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{FenceKind, SsaOp},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::memory,
    testing::{MockTarget, MockType},
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
        address_space: None,
    })
}

fn load(dest: SsaVarId, addr: SsaVarId) -> SsaInstruction<MockTarget> {
    instr(SsaOp::LoadIndirect {
        dest,
        addr,
        value_type: MockType::I32,
        address_space: None,
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
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

#[test]
fn store_to_load_forwarding_replaces_the_load() {
    let mut ssa = store_then_load(0, 0);
    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);

    assert!(changed, "an exact reload of a just-stored cell forwards");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!(loads, 0, "the load is gone");
    assert_eq!(stores, 1, "the store is live — it is still observable");
    assert_eq!(ssa.validate(), Ok(()));
}

/// The load reads a cell 8 bytes away from the store, so nothing may be
/// forwarded. This is the negative that proves the offset comparison is load
/// bearing — before addresses were decoded, both accesses were simply
/// "indirect through some value".
#[test]
fn disjoint_offsets_are_not_forwarded() {
    let mut ssa = store_then_load(0, 8);
    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);

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
    b0.add_instruction(instr(SsaOp::Return { value: Some(sum) }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
    assert!(changed, "the second load of an unchanged cell is redundant");
    let (loads, _) = access_counts(&ssa);
    assert_eq!(loads, 1, "one load survives");
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
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
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
    b0.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
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
    // An address the model cannot separate from `addr`: a different base, so a
    // load through it may observe the first store.
    b0.add_instruction(instr(SsaOp::Const {
        dest: other,
        value: ConstValue::I32(0x2000),
    }));
    b0.add_instruction(store(addr, first_value));
    b0.add_instruction(load(observed, other));
    b0.add_instruction(store(addr, second_value));
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(observed),
    }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
    let (_, stores) = access_counts(&ssa);
    assert_eq!(
        stores, 2,
        "a store that may have been read is not dead (changed={changed})"
    );
    assert_eq!(ssa.validate(), Ok(()));
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
    b1.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b1);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
    assert!(changed, "a dominating store serves the load");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!(loads, 0, "the dominated load is forwarded away");
    assert_eq!(stores, 1);
    assert_eq!(ssa.validate(), Ok(()));
}

/// A load at a merge whose two predecessors store different values must not be
/// forwarded — the memory phi says two definitions reach it.
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
    b3.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b3);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
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
        address_space: Some(1),
    }));
    // ...then read the flat space at the same address.
    b0.add_instruction(load(loaded, addr));
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
    assert!(
        !changed,
        "a segmented store must not serve a flat load at the same offset"
    );
    let (loads, stores) = access_counts(&ssa);
    assert_eq!((loads, stores), (1, 1));
    assert_eq!(ssa.validate(), Ok(()));
}

/// Builds `*(p+0) = a; <separator>; *(p+0) = b; return` and reports how many
/// stores survive the pass.
fn stores_around(separator: Vec<SsaInstruction<MockTarget>>) -> usize {
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
    b0.add_instruction(store(addr, first_value));
    for extra in separator {
        b0.add_instruction(extra);
    }
    b0.add_instruction(store(addr, second_value));
    b0.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
    assert_eq!(ssa.validate(), Ok(()));
    let (_, stores) = access_counts(&ssa);
    stores
}

/// A call between two stores to the same cell keeps the first alive.
///
/// This is the guarantee that makes dead store elimination safe for memory
/// visible outside the function: the callee may read whatever the first store
/// wrote, so the store is observable and must stay. Any call — direct, virtual,
/// or indirect — classifies as a barrier.
#[test]
fn a_call_between_stores_blocks_dead_store_elimination() {
    let call = instr(SsaOp::Call {
        dest: None,
        method: 7u32,
        args: Vec::new(),
    });
    assert_eq!(
        stores_around(vec![call]),
        2,
        "a callee may observe the first store"
    );
}

/// A volatile prefix between two stores keeps the first alive: a volatile write
/// is observable by definition (MMIO, a device register), so writing twice is
/// two distinct effects, not a redundant one.
#[test]
fn a_volatile_prefix_blocks_dead_store_elimination() {
    assert_eq!(
        stores_around(vec![instr(SsaOp::Volatile)]),
        2,
        "a volatile access must not be coalesced away"
    );
}

/// With nothing in between, the first store really is dead — the control that
/// makes the two tests above meaningful.
#[test]
fn nothing_in_between_leaves_the_store_dead() {
    assert_eq!(stores_around(Vec::new()), 1);
}

/// A call between a store and a load blocks forwarding: the callee may have
/// rewritten the cell, so the stored value is no longer known to be there.
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
    b0.add_instruction(instr(SsaOp::Call {
        dest: None,
        method: 7u32,
        args: Vec::new(),
    }));
    b0.add_instruction(load(loaded, addr));
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);
    assert!(!changed, "a call may have rewritten the cell");
    let (loads, stores) = access_counts(&ssa);
    assert_eq!((loads, stores), (1, 1));
    assert_eq!(ssa.validate(), Ok(()));
}

// ---------------------------------------------------------------------------
// Cross-block forwarding.
//
// Every barrier test above places the barrier in the *same* block as the store
// and the load, which only exercises `plan_block`'s intra-block bookkeeping.
// Cross-block availability is seeded separately, from the memory-SSA version
// reaching the block entry — and memory versions are strictly per-location, so
// neither a barrier (which versions only `MemoryLocation::Unknown`) nor an
// overlapping store (which versions only its own cell) is visible to a
// version-equality test on the loaded location alone.
// ---------------------------------------------------------------------------

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
    b1.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
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

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);

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

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);

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
/// `p+0`, so the earlier value no longer describes the whole read — but it
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

    let changed = memory::run(&mut ssa, &0u32, &NullListener, PointerSize::Bit64);

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
