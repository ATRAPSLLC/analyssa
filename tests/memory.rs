//! Memory SSA and alias analysis tests.

use analyssa::{
    PointerSize,
    analysis::{
        SsaCfg,
        memory::{
            AliasResult, ArrayIndex, IndirectLocation, MemoryDefSite, MemoryLocation, MemoryOp,
            MemorySsa, MemorySsaStats, MemoryVersion, analyze_alias,
        },
    },
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{AtomicAccessWidth, AtomicOrdering, FenceKind, MemoryAccessSemantics, SsaOp},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MockTarget, MockType},
};

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        MockType::I32,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

#[test]
fn memory_location_equality_and_hash() {
    let obj = SsaVarId::from_index(0);
    let loc1 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc2 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc3 = MemoryLocation::<MockTarget>::InstanceField(obj, 2u32);

    assert_eq!(loc1, loc2);
    assert_ne!(loc1, loc3);

    let static_loc = MemoryLocation::<MockTarget>::StaticField(1u32);
    assert_ne!(loc1, static_loc);
}

#[test]
fn alias_analysis_same_location_is_must_alias() {
    let obj = SsaVarId::from_index(0);
    let loc1 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc2 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);

    assert_eq!(analyze_alias(&loc1, &loc2), AliasResult::MustAlias);
}

#[test]
fn alias_analysis_different_fields_diff_object() {
    let obj = SsaVarId::from_index(0);
    let loc1 = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);
    let loc2 = MemoryLocation::<MockTarget>::InstanceField(obj, 2u32);

    // Different fields on the same object: the analysis may return NoAlias or
    // MayAlias depending on whether it can prove non-overlap.
    let result = analyze_alias(&loc1, &loc2);
    assert!(result != AliasResult::MustAlias);

    let loc3 = MemoryLocation::<MockTarget>::InstanceField(SsaVarId::from_index(1), 1u32);
    // Same field but different known objects: can be NoAlias if objects are
    // provably distinct
    let _ = analyze_alias(&loc1, &loc3);
}

#[test]
fn alias_analysis_static_vs_instance() {
    let static_loc = MemoryLocation::<MockTarget>::StaticField(42u32);
    let instance_loc = MemoryLocation::<MockTarget>::InstanceField(SsaVarId::from_index(0), 42u32);

    assert_eq!(
        analyze_alias(&static_loc, &instance_loc),
        AliasResult::NoAlias
    );
}

#[test]
fn memory_location_array_and_indirect() {
    let array_loc = MemoryLocation::<MockTarget>::ArrayElement(
        SsaVarId::from_index(0),
        ArrayIndex::Constant(0),
    );
    let array_loc2 =
        MemoryLocation::<MockTarget>::ArrayElement(SsaVarId::from_index(0), ArrayIndex::Unknown);

    assert_eq!(
        analyze_alias(&array_loc, &array_loc2),
        AliasResult::MayAlias
    );
}

/// Builds `*(base + offset_bits)` with an `size_bits`-wide access and no
/// scaled index.
fn indirect(base: usize, offset_bits: i64, size_bits: Option<u32>) -> MemoryLocation<MockTarget> {
    MemoryLocation::Indirect(IndirectLocation::new(
        SsaVarId::from_index(base),
        None,
        0,
        offset_bits,
        size_bits,
        None,
        PointerSize::Bit64,
    ))
}

/// Builds `*(base + index*stride + offset_bits)` with an `size_bits`-wide access.
fn indirect_indexed(
    base: usize,
    index: usize,
    stride_bytes: u64,
    offset_bits: i64,
    size_bits: Option<u32>,
) -> MemoryLocation<MockTarget> {
    MemoryLocation::Indirect(IndirectLocation::new(
        SsaVarId::from_index(base),
        Some(SsaVarId::from_index(index)),
        stride_bytes,
        offset_bits,
        size_bits,
        None,
        PointerSize::Bit64,
    ))
}

#[test]
fn memory_location_indirect_and_unknown() {
    let indirect = indirect(1, 0, Some(32));
    let unknown = MemoryLocation::<MockTarget>::Unknown;

    assert_eq!(analyze_alias(&indirect, &unknown), AliasResult::MayAlias);
    assert_eq!(analyze_alias(&unknown, &indirect), AliasResult::MayAlias);
}

/// Two accesses off one base at non-overlapping offsets are provably disjoint.
///
/// This is the headline gain of decoding the address: keying on the address
/// *value id* leaves `[rbp-8]` and `[rbp-16]` unrelated-but-indistinguishable,
/// so no consumer can move a load across an unrelated store.
#[test]
fn indirect_disjoint_offsets_do_not_alias() {
    // Two adjacent 4-byte slots off one base.
    let low = indirect(1, 0, Some(32));
    let high = indirect(1, 32, Some(32));

    assert_eq!(analyze_alias(&low, &high), AliasResult::NoAlias);
    assert_eq!(analyze_alias(&high, &low), AliasResult::NoAlias);
}

/// Partially overlapping extents off one base may alias — a 4-byte access at
/// bit 0 and another at bit 16 share two bytes.
#[test]
fn indirect_overlapping_offsets_may_alias() {
    let first = indirect(1, 0, Some(32));
    let straddling = indirect(1, 16, Some(32));

    assert_eq!(analyze_alias(&first, &straddling), AliasResult::MayAlias);
    assert_eq!(analyze_alias(&straddling, &first), AliasResult::MayAlias);
}

/// Identical decoded cells must-alias, which is what lets a consumer forward a
/// store to a later load.
#[test]
fn indirect_identical_cells_must_alias() {
    let cell = indirect(1, 64, Some(32));
    assert_eq!(analyze_alias(&cell, &cell.clone()), AliasResult::MustAlias);
}

/// Distinct bases conservatively may-alias: two unrelated SSA pointers can
/// hold the same address, and the address expression alone cannot say
/// otherwise. Separating them needs a points-to oracle.
#[test]
fn indirect_distinct_bases_may_alias() {
    let from_one = indirect(1, 0, Some(32));
    let from_two = indirect(2, 0, Some(32));

    assert_eq!(analyze_alias(&from_one, &from_two), AliasResult::MayAlias);
}

/// An unknown access width overlaps anything at that base and never proves a
/// single cell — not even against an identical location.
#[test]
fn indirect_unknown_width_never_must_aliases() {
    let unsized_cell = indirect(1, 0, None);
    let sized_far_away = indirect(1, 4096, Some(32));

    assert_eq!(
        analyze_alias(&unsized_cell, &unsized_cell.clone()),
        AliasResult::MayAlias,
        "an unknown extent cannot prove one cell"
    );
    assert_eq!(
        analyze_alias(&unsized_cell, &sized_far_away),
        AliasResult::MayAlias,
        "an unknown extent may reach any offset off the same base"
    );
}

/// Accesses in different address spaces never alias, even at the same numeric
/// offset off the same base.
///
/// This is what keeps an x86 `fs:[0x30]` (the TEB/PEB idiom) separate from a
/// flat `[0x30]`: without it the two decode to one memory location, and a
/// consumer would forward a store across the segment boundary.
#[test]
fn distinct_address_spaces_never_alias() {
    let flat = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        SsaVarId::from_index(1),
        None,
        0,
        0x30 * 8,
        Some(32),
        None,
        PointerSize::Bit64,
    ));
    let segmented = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        SsaVarId::from_index(1),
        None,
        0,
        0x30 * 8,
        Some(32),
        Some(1),
        PointerSize::Bit64,
    ));

    assert_eq!(
        analyze_alias(&flat, &segmented),
        AliasResult::NoAlias,
        "a segmented access and a flat one name different memory"
    );
    // ...while two accesses in the same space still compare normally.
    assert_eq!(
        analyze_alias(&segmented, &segmented.clone()),
        AliasResult::MustAlias
    );
}

/// Offset reasoning is only valid when both addresses carry the same scaled
/// index contribution: `arr[i]` and `arr[j]` may alias (the indices could hold
/// equal values), while two disjoint fields of `arr[i]` do not.
#[test]
fn indirect_scaled_index_gates_offset_reasoning() {
    let elem_i = indirect_indexed(1, 10, 8, 0, Some(32));
    let elem_j = indirect_indexed(1, 11, 8, 0, Some(32));
    assert_eq!(
        analyze_alias(&elem_i, &elem_j),
        AliasResult::MayAlias,
        "distinct index values may hold the same number"
    );

    // Same element, two disjoint fields within it.
    let elem_i_field0 = indirect_indexed(1, 10, 8, 0, Some(32));
    let elem_i_field1 = indirect_indexed(1, 10, 8, 32, Some(32));
    assert_eq!(
        analyze_alias(&elem_i_field0, &elem_i_field1),
        AliasResult::NoAlias,
        "one index term cancels, so the constant offsets decide"
    );
    assert_eq!(
        analyze_alias(&elem_i_field0, &elem_i_field0.clone()),
        AliasResult::MustAlias
    );

    // An index on only one side: it could be zero, so nothing is provable.
    let no_index = indirect(1, 0, Some(32));
    assert_eq!(
        analyze_alias(&elem_i, &no_index),
        AliasResult::MayAlias,
        "an index that could be zero defeats offset reasoning"
    );

    // Same index value but a different stride contributes a different amount.
    let elem_i_stride4 = indirect_indexed(1, 10, 4, 0, Some(32));
    assert_eq!(
        analyze_alias(&elem_i, &elem_i_stride4),
        AliasResult::MayAlias,
        "differing strides are not comparable"
    );
}

#[test]
fn memory_location_debug_output() {
    let loc = MemoryLocation::<MockTarget>::InstanceField(SsaVarId::from_index(5), 3u32);
    let debug_str = format!("{loc:?}");
    assert!(!debug_str.is_empty());
}

#[test]
fn alias_result_equality() {
    assert_eq!(AliasResult::MustAlias, AliasResult::MustAlias);
    assert_ne!(AliasResult::MustAlias, AliasResult::NoAlias);
    assert_ne!(AliasResult::MayAlias, AliasResult::NoAlias);
}

#[test]
fn memory_ssa_empty_function_stats() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);

    let stats: MemorySsaStats = mem_ssa.stats();
    assert_eq!(stats.store_count, 0);
    assert_eq!(stats.load_count, 0);
    assert_eq!(stats.location_count, 0);
}

#[test]
fn memory_ssa_with_field_loads_and_stores() {
    let mut ssa = SsaFunction::new(0, 2);
    let obj = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);
    let loaded = local(&mut ssa, 2, 0, 2);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(100),
    }));
    b0.add_instruction(instr(SsaOp::StoreField {
        object: obj,
        field: 1u32,
        value: val,
    }));
    b0.add_instruction(instr(SsaOp::LoadField {
        dest: loaded,
        object: obj,
        field: 1u32,
    }));
    b0.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b0);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);

    let stats: MemorySsaStats = mem_ssa.stats();
    assert_eq!(stats.store_count, 1);
    assert_eq!(stats.load_count, 1);
}

#[test]
fn memory_ssa_new_is_empty() {
    let mem_ssa: MemorySsa<MockTarget> = MemorySsa::new();
    let stats = mem_ssa.stats();
    assert_eq!(stats.store_count, 0);
    assert_eq!(stats.load_count, 0);
    assert_eq!(stats.memory_phi_count, 0);
    assert_eq!(stats.version_count, 0);
}

#[test]
fn memory_ssa_classifies_atomic_and_fence_effects() {
    let mut ssa = SsaFunction::new(0, 1);
    let addr = local(&mut ssa, 0, 0, 0);
    let value = local(&mut ssa, 1, 0, 1);
    let old = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: addr,
        value: ConstValue::I32(0),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::AtomicExchange {
        dest: old,
        addr,
        value,
        ordering: AtomicOrdering::AcqRel,
        width: AtomicAccessWidth::Bits32,
        volatile: true,
    }));
    block.add_instruction(instr(SsaOp::Fence {
        kind: FenceKind::Acquire,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(old) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);
    let stats = mem_ssa.stats();

    assert_eq!(stats.store_count, 1);
    assert_eq!(stats.barrier_count, 1);
    assert!(mem_ssa.operations().iter().any(|op| {
        op.effects()
            .is_some_and(|effects| effects.memory_semantics == MemoryAccessSemantics::Atomic)
    }));
}

#[test]
fn memory_ssa_handles_store_in_branch() {
    let mut ssa = SsaFunction::new(0, 3);
    let obj = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);
    let cond = local(&mut ssa, 2, 0, 2);
    let loaded = local(&mut ssa, 3, 2, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(100),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::StoreField {
        object: obj,
        field: 5u32,
        value: val,
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::LoadField {
        dest: loaded,
        object: obj,
        field: 5u32,
    }));
    b2.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);

    let stats: MemorySsaStats = mem_ssa.stats();
    assert!(
        stats.store_count >= 1,
        "expected at least 1 store, got {}",
        stats.store_count
    );
    assert!(
        stats.load_count >= 1,
        "expected at least 1 load, got {}",
        stats.load_count
    );
    assert!(
        stats.location_count >= 1,
        "expected at least 1 location, got {}",
        stats.location_count
    );
}

/// Builds a diamond `b0 -> {b1, b2} -> b3` over one instance field, with a
/// store in each arm named by `store_in_arm2`, and a load at the merge.
///
/// `b0` dominates all three successors, so `b1` and `b2` are *sibling*
/// dominator subtrees and `b3` carries a memory phi — the shape that exposes
/// scope leakage in the rename walk.
fn diamond_over_one_field(store_in_arm2: bool) -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut ssa = SsaFunction::new(0, 5);
    let obj = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);
    let cond = local(&mut ssa, 2, 0, 2);
    let loaded = local(&mut ssa, 3, 3, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(100),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::StoreField {
        object: obj,
        field: 5u32,
        value: val,
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    if store_in_arm2 {
        b2.add_instruction(instr(SsaOp::StoreField {
            object: obj,
            field: 5u32,
            value: val,
        }));
    }
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::LoadField {
        dest: loaded,
        object: obj,
        field: 5u32,
    }));
    b3.add_instruction(instr(SsaOp::Return {
        value: Some(loaded),
    }));
    ssa.add_block(b3);

    ssa.recompute_uses();
    (ssa, obj)
}

/// A block's entry version must be the version live at the end of its
/// *dominator*, never one produced by a sibling dominator subtree.
///
/// This is the scope-restore guarantee of the Cytron rename walk. Without the
/// pop on leaving a subtree, whichever arm the traversal happens to reach first
/// leaks its store version into the other arm — silently, and in an order that
/// depends on traversal rather than on the program.
#[test]
fn sibling_subtrees_do_not_leak_memory_versions() {
    let (ssa, obj) = diamond_over_one_field(true);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);
    let loc = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    let b0_exit = mem_ssa.version_at_exit(&loc, 0);
    assert!(b0_exit.is_some(), "entry block has an exit version");

    assert_eq!(
        mem_ssa.version_at_entry(&loc, 1),
        b0_exit,
        "arm 1 must enter on its dominator's exit version"
    );
    assert_eq!(
        mem_ssa.version_at_entry(&loc, 2),
        b0_exit,
        "arm 2 must enter on its dominator's exit version"
    );

    // Each arm stores, so each leaves on a version of its own — distinct from
    // the one it entered on and from its sibling's.
    let b1_exit = mem_ssa.version_at_exit(&loc, 1);
    let b2_exit = mem_ssa.version_at_exit(&loc, 2);
    assert!(
        b1_exit.is_some() && b2_exit.is_some(),
        "both arms have exits"
    );
    assert_ne!(b1_exit, b0_exit, "arm 1's store defines a new version");
    assert_ne!(b2_exit, b0_exit, "arm 2's store defines a new version");
    assert_ne!(
        b1_exit, b2_exit,
        "the two arms' stores are distinct definitions"
    );
}

/// Each memory phi operand carries the version live at the end of the
/// corresponding predecessor, so a diamond's two storing arms contribute the
/// two distinct store versions.
#[test]
fn merge_phi_operands_track_each_predecessor_exit() {
    let (ssa, obj) = diamond_over_one_field(true);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);
    let loc = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    let phis = mem_ssa.memory_phis(3);
    let found = phis.iter().find(|phi| phi.location == loc);
    assert!(
        found.is_some(),
        "the merge block carries a memory phi for the stored location"
    );
    let Some(phi) = found else { return };

    let from_b1 = phi.operand_from(1).map(|op| op.version);
    let from_b2 = phi.operand_from(2).map(|op| op.version);
    assert!(
        from_b1.is_some() && from_b2.is_some(),
        "the phi has an operand from each predecessor"
    );

    assert_eq!(
        from_b1,
        mem_ssa.version_at_exit(&loc, 1),
        "the arm-1 operand is arm 1's exit version"
    );
    assert_eq!(
        from_b2,
        mem_ssa.version_at_exit(&loc, 2),
        "the arm-2 operand is arm 2's exit version"
    );
    assert_ne!(
        from_b1, from_b2,
        "the two arms reach the merge with different versions"
    );
}

/// A predecessor that defines no memory must contribute the version it
/// *inherited* to the merge phi — not one leaked from a sibling arm.
///
/// This is the damaging form of a missing scope restore: the store in arm 1
/// leaks across to the pass-through arm, so the phi claims the value is defined
/// on a path that never touched memory, and any consumer forwarding through
/// that phi reads the wrong store on the wrong path.
#[test]
fn non_storing_arm_contributes_its_inherited_version() {
    let (ssa, obj) = diamond_over_one_field(false);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);
    let loc = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    let b0_exit = mem_ssa.version_at_exit(&loc, 0);
    assert!(b0_exit.is_some(), "entry block has an exit version");

    // The pass-through arm neither enters nor leaves on a new version.
    assert_eq!(
        mem_ssa.version_at_entry(&loc, 2),
        b0_exit,
        "the pass-through arm inherits its dominator's version"
    );
    assert_eq!(
        mem_ssa.version_at_exit(&loc, 2),
        b0_exit,
        "the pass-through arm defines no memory, so it leaves as it entered"
    );

    let found = mem_ssa
        .memory_phis(3)
        .iter()
        .find(|phi| phi.location == loc);
    assert!(
        found.is_some(),
        "the merge block carries a memory phi for the stored location"
    );
    let Some(phi) = found else { return };

    let from_storing = phi.operand_from(1).map(|op| op.version);
    let from_passthrough = phi.operand_from(2).map(|op| op.version);

    assert_eq!(
        from_passthrough, b0_exit,
        "the pass-through arm must reach the merge on the inherited version"
    );
    assert_ne!(
        from_storing, from_passthrough,
        "the storing arm reaches the merge on its own store version"
    );
    assert_eq!(
        from_storing.and_then(|v| mem_ssa.definition(&MemoryVersion::new(loc.clone(), v))),
        Some(MemoryDefSite::Store { block: 1, instr: 0 }),
        "the storing arm's operand resolves to that arm's store"
    );
}

/// Every version the rename walk leaves live at a block boundary is traceable
/// to a definition site, and the version entering the entry block is the
/// function-entry definition.
///
/// Version *numbering* is an opaque identity, not an ordering: phi versions are
/// allocated during phi placement, so the entry version is not the lowest.
/// Consumers must match [`MemoryDefSite::Entry`] rather than compare to zero.
#[test]
fn every_live_memory_version_has_a_definition_site() {
    let (ssa, obj) = diamond_over_one_field(true);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);
    let loc = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    for block in 0..4usize {
        for version in [
            mem_ssa.version_at_entry(&loc, block),
            mem_ssa.version_at_exit(&loc, block),
        ]
        .into_iter()
        .flatten()
        {
            assert!(
                mem_ssa
                    .definition(&MemoryVersion::new(loc.clone(), version))
                    .is_some(),
                "version {version} live at block {block} must have a definition site"
            );
        }
    }

    assert_eq!(
        mem_ssa
            .version_at_entry(&loc, 0)
            .and_then(|v| mem_ssa.definition(&MemoryVersion::new(loc.clone(), v))),
        Some(MemoryDefSite::Entry),
        "the version entering the entry block is the function-entry definition"
    );
}

/// Builds a single block that computes `n` addresses off one base via
/// `PtrAdd`, each at the given byte offset, and stores 4 bytes through each.
///
/// Every `PtrAdd` gets its own `dest`, which is exactly the situation where
/// keying a memory location on the address value id goes wrong.
fn stores_through_ptradd_offsets(offsets: &[i64]) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 32);
    let base = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(7),
    }));

    let mut addrs = Vec::new();
    for (nth, offset) in offsets.iter().enumerate() {
        let addr = ssa.create_variable(
            VariableOrigin::Local(u16::try_from(nth).unwrap_or(0).saturating_add(10)),
            0,
            DefSite::instruction(0, nth.saturating_add(2)),
            MockType::Ptr,
        );
        addrs.push(addr);
        b0.add_instruction(instr(SsaOp::PtrAdd {
            dest: addr,
            base,
            index: None,
            stride: 0,
            offset: *offset,
            result_type: MockType::Ptr,
        }));
    }
    for addr in addrs {
        b0.add_instruction(instr(SsaOp::StoreIndirect {
            addr,
            value: val,
            value_type: MockType::I32,
            address_space: None,
        }));
    }
    b0.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// Two stores to different offsets off one base become two distinct memory
/// locations that provably do not alias.
///
/// Before the address was decoded these were `Indirect(addr_a)` and
/// `Indirect(addr_b)` — two unrelated ids, so no consumer could tell whether
/// they overlapped.
#[test]
fn distinct_stack_slots_get_disjoint_memory_locations() {
    let ssa = stores_through_ptradd_offsets(&[0, 8]);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);

    let locations: Vec<_> = mem_ssa
        .locations()
        .iter()
        .filter(|loc| matches!(loc, MemoryLocation::Indirect(_)))
        .cloned()
        .collect();
    assert_eq!(
        locations.len(),
        2,
        "the two offsets are two cells, got {locations:?}"
    );

    let (Some(first), Some(second)) = (locations.first(), locations.get(1)) else {
        return;
    };
    assert_eq!(
        analyze_alias(first, second),
        AliasResult::NoAlias,
        "8 bytes apart with 4-byte accesses cannot overlap"
    );

    // Both decode off the same base, at the two byte offsets, 32 bits wide.
    let mut offsets: Vec<i64> = locations
        .iter()
        .filter_map(|loc| match loc {
            MemoryLocation::Indirect(indirect) => Some(indirect.offset_bits),
            _ => None,
        })
        .collect();
    offsets.sort_unstable();
    assert_eq!(offsets, vec![0, 64], "offsets are decoded in bits");
}

/// Cell identity comes from the decoded address, so two *separate* `PtrAdd`
/// instructions computing the same address name one location.
///
/// This is what makes the model stable under GVN and LICM: those passes freely
/// re-number and hoist the pure address computation, and memory identity must
/// not move when they do.
#[test]
fn identical_addresses_from_separate_ptradds_are_one_location() {
    let ssa = stores_through_ptradd_offsets(&[16, 16]);
    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);

    let locations: Vec<_> = mem_ssa
        .locations()
        .iter()
        .filter(|loc| matches!(loc, MemoryLocation::Indirect(_)))
        .cloned()
        .collect();
    assert_eq!(
        locations.len(),
        1,
        "two PtrAdds to one address are one cell, got {locations:?}"
    );

    // ...and the two stores are both attributed to it, so the second kills the
    // first rather than being treated as an unrelated write.
    let stores = mem_ssa
        .operations()
        .iter()
        .filter(|op| matches!(op, MemoryOp::Store { .. }))
        .count();
    assert_eq!(stores, 2, "both stores are recorded");
    let Some(only) = locations.first() else {
        return;
    };
    assert_eq!(analyze_alias(only, &only.clone()), AliasResult::MustAlias);
}

/// A constant array index folds to [`ArrayIndex::Constant`], so two distinct
/// constant elements stop may-aliasing each other.
#[test]
fn constant_array_indices_fold_and_separate() {
    let mut ssa = SsaFunction::new(0, 6);
    let array = local(&mut ssa, 0, 0, 0);
    let idx0 = local(&mut ssa, 1, 0, 1);
    let idx1 = local(&mut ssa, 2, 0, 2);
    let val = local(&mut ssa, 3, 0, 3);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: array,
        value: ConstValue::I32(0x2000),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: idx0,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: idx1,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(9),
    }));
    b0.add_instruction(instr(SsaOp::StoreElement {
        array,
        index: idx0,
        value: val,
        elem_type: MockType::I32,
    }));
    b0.add_instruction(instr(SsaOp::StoreElement {
        array,
        index: idx1,
        value: val,
        elem_type: MockType::I32,
    }));
    b0.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);

    let elem0 = MemoryLocation::<MockTarget>::ArrayElement(array, ArrayIndex::Constant(0));
    let elem1 = MemoryLocation::<MockTarget>::ArrayElement(array, ArrayIndex::Constant(1));
    assert!(
        mem_ssa.locations().contains(&elem0),
        "index 0 folded to a constant, got {:?}",
        mem_ssa.locations()
    );
    assert!(mem_ssa.locations().contains(&elem1));
    assert_eq!(analyze_alias(&elem0, &elem1), AliasResult::NoAlias);
}

// ---------------------------------------------------------------------------
// Alias soundness across distinct SSA ids.
//
// `may_alias` drives the *invalidation* half of the memory pass
// (`available.retain(|k| !k.may_alias(location))` and the DSE `pending_stores`
// guard), so a false NoAlias is unsound in the direction that matters: it lets a
// stale forwarded value survive a store to the same cell, and lets DSE delete a
// store that was actually read.
// ---------------------------------------------------------------------------

/// Two distinct SSA ids routinely name one object: two un-GVN'd loads of the
/// same slot, two arguments, a value and a phi of it. Distinct ids therefore do
/// not prove distinct objects.
#[test]
fn distinct_object_ids_may_alias_the_same_field() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);

    let via_a = MemoryLocation::<MockTarget>::InstanceField(a, 3u32);
    let via_b = MemoryLocation::<MockTarget>::InstanceField(b, 3u32);

    assert!(
        via_a.may_alias(&via_b),
        "distinct SSA ids can hold the same object reference"
    );
    assert!(
        !via_a.must_alias(&via_b),
        "but they are not proved to be the same object, so forwarding stays off"
    );
}

/// Precision must survive the fix: on one object, distinct fields are disjoint.
#[test]
fn different_fields_of_one_object_still_do_not_alias() {
    let obj = SsaVarId::from_index(0);
    let f3 = MemoryLocation::<MockTarget>::InstanceField(obj, 3u32);
    let f4 = MemoryLocation::<MockTarget>::InstanceField(obj, 4u32);

    assert!(
        !f3.may_alias(&f4),
        "distinct fields of a single object are disjoint"
    );
    assert!(f3.may_alias(&f3.clone()));
    assert!(f3.must_alias(&f3.clone()));
}

/// The same argument applies to the array reference in `ArrayElement`.
#[test]
fn distinct_array_ids_may_alias_regardless_of_index() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);

    let a0 = MemoryLocation::<MockTarget>::ArrayElement(a, ArrayIndex::Constant(0));
    let b7 = MemoryLocation::<MockTarget>::ArrayElement(b, ArrayIndex::Constant(7));
    assert!(
        a0.may_alias(&b7),
        "different array ids may name one array, so the indices prove nothing"
    );

    let a7 = MemoryLocation::<MockTarget>::ArrayElement(a, ArrayIndex::Constant(7));
    assert!(
        !a0.may_alias(&a7),
        "on a single array, constant indices 0 and 7 are disjoint"
    );
}

/// Displacements are canonicalised into the signed pointer-width range, so an
/// access whose *end* runs past the top of the address space wraps around to the
/// bottom. The linear extent comparison cannot see that, and would call such an
/// access disjoint from a low-offset one it actually overlaps.
#[test]
fn an_access_wrapping_the_address_space_does_not_prove_disjointness() {
    let base = SsaVarId::from_index(0);
    // Just below the top of a 32-bit signed address space, reading 8 bytes: the
    // access runs off the top and reappears at the bottom.
    let top_bits = i64::from(i32::MAX) * 8 - 16;
    let wrapping = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        base,
        None,
        0,
        top_bits,
        Some(64),
        None,
        PointerSize::Bit32,
    ));
    // An access at the very bottom, which the wrapped tail can reach.
    let low = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        base,
        None,
        0,
        -(i64::from(i32::MAX) + 1) * 8,
        Some(64),
        None,
        PointerSize::Bit32,
    ));

    assert!(
        wrapping.may_alias(&low),
        "an access that wraps the address space must not be proved disjoint"
    );

    // Ordinary non-wrapping extents keep their precision.
    let a = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        base,
        None,
        0,
        0,
        Some(32),
        None,
        PointerSize::Bit32,
    ));
    let b = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        base,
        None,
        0,
        64,
        Some(32),
        None,
        PointerSize::Bit32,
    ));
    assert!(
        !a.may_alias(&b),
        "disjoint extents well inside the address space still do not alias"
    );
}

/// Memory SSA records an entry and an exit version for every location in every
/// block, so its retained size is `2 × blocks × locations`. For a "function"
/// that is really data misread as code, each distinct `base + offset` mints its
/// own `Indirect` location, so the location count scales with the block count
/// and the product is quadratic.
///
/// Past the budget the analysis must degrade to "nothing is known" rather than
/// exhaust memory: an empty result makes every alias query fall back to the
/// conservative answer, so the memory pass finds nothing to do — which is
/// correct, just unoptimized.
#[test]
fn a_function_past_the_memory_ssa_budget_degrades_instead_of_growing() {
    // 3000 blocks, each storing to its own distinct cell: 9M cells, past the
    // 4M budget.
    const BLOCKS: usize = 3000;

    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, BLOCKS + 1);
    let base = local(&mut ssa, 0, 0, 0);
    let value = local(&mut ssa, 1, 0, 1);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(0x1000),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    for block_id in 1..=BLOCKS {
        let addr = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(block_id, 0),
            MockType::Ptr,
        );
        let mut block = SsaBlock::new(block_id);
        // A distinct offset per block, so each store names its own location.
        block.add_instruction(instr(SsaOp::PtrAdd {
            dest: addr,
            base,
            index: None,
            stride: 0,
            offset: (block_id as i64) * 8,
            result_type: MockType::Ptr,
        }));
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr,
            value,
            value_type: MockType::I32,
            address_space: None,
        }));
        if block_id == BLOCKS {
            block.add_instruction(instr(SsaOp::Return { value: None }));
        } else {
            block.add_instruction(instr(SsaOp::Jump {
                target: block_id + 1,
            }));
        }
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &cfg, PointerSize::Bit64);

    assert!(
        mem_ssa.locations().is_empty(),
        "past the budget the analysis must report nothing rather than build \
         a quadratic structure"
    );
    // And the conservative fallback really is conservative: no version is known,
    // so nothing can be proved forwardable.
    let unknown = MemoryLocation::<MockTarget>::Unknown;
    assert_eq!(mem_ssa.version_at_entry(&unknown, 1), None);
}
