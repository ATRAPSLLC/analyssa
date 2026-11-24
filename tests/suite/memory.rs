//! Memory SSA and alias analysis tests.

use std::num::NonZeroU16;

use analyssa::{
    Endianness, PointerSize,
    analysis::{
        AnalysisContext,
        address::AddressBase,
        budget::{Budget, WorkBudget},
        cache::FunctionAnalyses,
        image::{ImageData, ImageRegion, Mutability},
        memory::{
            AliasResult, ArrayIndex, IndirectLocation, MemoryDefSite, MemoryLocation, MemoryOp,
            MemorySsa, MemorySsaStats, MemoryVersion, analyze_alias,
        },
    },
    ir::{
        AccessMode, CallAccess, CallData, CallEffects, CallLocation, CallMemory, CallTermination,
        Callee, Capture,
        block::SsaBlock,
        exception::{BlockRange, SsaExceptionHandler},
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{
            AccessQualifiers, AtomicAccessWidth, AtomicOrdering, FenceKind, MemoryAccessSemantics,
            SsaOp,
        },
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
    },
    testing::{MACHINE, MockTarget, MockType, memory_effect_fixture},
};

/// An image with one read-only extent at `0x1000` and one writable at
/// `0x2000`, as a host serves them.
#[derive(Debug)]
struct TestImage;

impl ImageData for TestImage {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        [
            (0x1000, 0x1010, Mutability::ReadOnly),
            (0x2000, 0x2010, Mutability::Writable),
        ]
        .into_iter()
        .find(|(start, end, _)| (*start..*end).contains(&address))
        .map(|(start, end, mutability)| ImageRegion {
            start,
            end,
            mutability,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        const BYTES: [u8; 0x10] = [0x2a; 0x10];
        let region = self.region(address)?;
        let start = usize::try_from(address.checked_sub(region.start)?).ok()?;
        BYTES.get(start..start.checked_add(len)?)
    }
}

/// One exception clause over block ranges, with no filter.
fn handler_clause(
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

/// Observations use the preceding state, including may-alias effects without a value edge.
#[test]
fn memory_versions_before_instructions_preserve_order_and_clobbers() {
    let ssa = memory_effect_fixture();
    let memory = MemorySsa::build(&ssa, &AnalysisContext::new(MACHINE));
    let location = MemoryLocation::InstanceField(SsaVarId::from_index(0), 1);
    // Constants at0..3, field store3, load4, indirect store5, atomic6,
    // fence7, return8. The indirect write, atomic and fence may alias the
    // field even though they are absent from its scalar def-use chain.
    for (position, expected) in [
        (0, MemoryDefSite::Entry),
        (3, MemoryDefSite::Entry),
        (4, MemoryDefSite::Store { block: 0, instr: 3 }),
        (5, MemoryDefSite::Store { block: 0, instr: 3 }),
        (6, MemoryDefSite::Store { block: 0, instr: 5 }),
        (7, MemoryDefSite::Store { block: 0, instr: 6 }),
        (8, MemoryDefSite::Store { block: 0, instr: 7 }),
        (9, MemoryDefSite::Store { block: 0, instr: 7 }),
    ] {
        assert_eq!(
            memory
                .version_before(&location, 0, position)
                .and_then(|version| {
                    memory.definition(&MemoryVersion::new(location.clone(), version))
                }),
            Some(expected),
            "before position {position}"
        );
    }
    assert_eq!(
        memory.version_before(&location, 0, 9),
        memory.version_at_exit(&location, 0)
    );
    for cell in memory.locations() {
        assert_eq!(
            memory.version_before(cell, 0, 6).and_then(|version| {
                memory.definition(&MemoryVersion::new(cell.clone(), version))
            }),
            Some(MemoryDefSite::Store { block: 0, instr: 5 }),
            "the indirect store reaches every may-alias cell in this fixture"
        );
    }
    assert_eq!(memory.version_before(&location, 0, 10), None);
    assert_eq!(memory.version_before(&location, 1, 0), None);
    assert_eq!(
        memory.version_before(&MemoryLocation::StaticField(999), 0, 0),
        None
    );
}

/// Every observation in a long block reaches its own last store, with no position cutoff.
#[test]
fn memory_versions_before_instructions_cover_long_blocks() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 2);
    let object = local(&mut ssa, 0, 0, 0);
    let value = local(&mut ssa, 1, 0, 1);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: object,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    for index in 0u16..2048 {
        let result = local(&mut ssa, index + 2, 0, 3 + usize::from(index) * 2);
        block.add_instruction(instr(SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object,
            field: 1,
            value,
        }));
        block.add_instruction(instr(SsaOp::LoadField {
            access: AccessQualifiers::FLAT,
            dest: result,
            object,
            field: 1,
        }));
    }
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();
    let memory = MemorySsa::build(&ssa, &AnalysisContext::new(MACHINE));
    let location = MemoryLocation::InstanceField(object, 1);
    for index in 0..2048 {
        assert_eq!(
            memory
                .version_before(&location, 0, 3 + index * 2)
                .and_then(|version| {
                    memory.definition(&MemoryVersion::new(location.clone(), version))
                }),
            Some(MemoryDefSite::Store {
                block: 0,
                instr: 2 + index * 2
            })
        );
    }
}

/// Builds a single cell with many ordered writes, independently of analysis policy.
fn single_cell_store_run(writes: usize) -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 2);
    let object = local(&mut ssa, 0, 0, 0);
    let value = local(&mut ssa, 1, 0, 1);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: object,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    for _ in 0..writes {
        block.add_instruction(instr(SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object,
            field: 1,
            value,
        }));
    }
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();
    (ssa, object)
}

/// A narrow block with too many writes exhausts atomically at the exact combined allowance.
#[test]
fn memory_instruction_versions_share_the_cell_budget() {
    let (ssa, object) = single_cell_store_run(3);
    let location = MemoryLocation::InstanceField(object, 1);
    for limit in [3, 4] {
        let context = AnalysisContext::new(MACHINE)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MemorySsaCells, limit));
        let memory = MemorySsa::build(&ssa, &context);
        if limit == 4 {
            assert!(memory.exhaustion().is_none());
            assert!(memory.version_before(&location, 0, 6).is_some());
        } else {
            assert_eq!(
                memory
                    .exhaustion()
                    .map(|failure| (failure.budget, failure.limit, failure.demand)),
                Some((Budget::MemorySsaCells, 3, 4))
            );
            assert!(memory.operations().is_empty() && memory.locations().is_empty());
            assert_eq!(memory.version_before(&location, 0, 0), None);
        }
    }
}

/// Definition admission stops at the first refused cell without collecting the remaining writes.
#[test]
fn memory_instruction_definition_budget_stops_at_first_refusal() {
    let (ssa, object) = single_cell_store_run(10_000);
    let location = MemoryLocation::InstanceField(object, 1);
    for limit in [0, 1, 3, 4] {
        let context = AnalysisContext::new(MACHINE)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MemorySsaCells, limit));
        let memory = MemorySsa::build(&ssa, &context);
        assert_eq!(
            memory
                .exhaustion()
                .map(|failure| (failure.limit, failure.demand)),
            Some((limit, limit + 1))
        );
        assert!(memory.operations().is_empty() && memory.locations().is_empty());
        assert_eq!(memory.version_before(&location, 0, 0), None);
    }
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

/// Partially overlapping extents off one base may alias - a 4-byte access at
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
/// single cell - not even against an identical location.
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
        NonZeroU16::new(1),
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
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

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
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 1u32,
        value: val,
    }));
    b0.add_instruction(instr(SsaOp::LoadField {
        access: AccessQualifiers::FLAT,
        dest: loaded,
        object: obj,
        field: 1u32,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);

    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

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
        access: AccessQualifiers::FLAT.made_volatile(),
    }));
    block.add_instruction(instr(SsaOp::Fence {
        kind: FenceKind::Acquire,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(old))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
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
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 5u32,
        value: val,
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::LoadField {
        access: AccessQualifiers::FLAT,
        dest: loaded,
        object: obj,
        field: 5u32,
    }));
    b2.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

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
/// dominator subtrees and `b3` carries a memory phi - the shape that exposes
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
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 5u32,
        value: val,
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    if store_in_arm2 {
        b2.add_instruction(instr(SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object: obj,
            field: 5u32,
            value: val,
        }));
    }
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::LoadField {
        access: AccessQualifiers::FLAT,
        dest: loaded,
        object: obj,
        field: 5u32,
    }));
    b3.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b3);

    ssa.recompute_uses();
    (ssa, obj)
}

/// A block's entry version must be the version live at the end of its
/// *dominator*, never one produced by a sibling dominator subtree.
///
/// This is the scope-restore guarantee of the Cytron rename walk. Without the
/// pop on leaving a subtree, whichever arm the traversal happens to reach first
/// leaks its store version into the other arm - silently, and in an order that
/// depends on traversal rather than on the program.
#[test]
fn sibling_subtrees_do_not_leak_memory_versions() {
    let (ssa, obj) = diamond_over_one_field(true);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
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

    // Each arm stores, so each leaves on a version of its own - distinct from
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
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
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

/// A merge of several cells finds each cell's phi by its interned location,
/// the one a scan of the block's phis by location finds, and nothing where the
/// block merges no cell.
#[test]
fn each_merged_cell_finds_its_own_phi() {
    let mut ssa = SsaFunction::new(0, 5);
    let obj = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);
    let cond = local(&mut ssa, 2, 0, 2);

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

    // Arm 1 stores fields 7 and 5, arm 2 field 6: the merge carries three
    // phis, and the cells are interned in an order their field numbers are
    // not.
    let store = |field: u32| {
        instr(SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object: obj,
            field,
            value: val,
        })
    };
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(store(7));
    b1.add_instruction(store(5));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(store(6));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b3);
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
    for field in [5u32, 6, 7] {
        let loc = MemoryLocation::<MockTarget>::InstanceField(obj, field);
        let id = mem_ssa.location_id(&loc);
        assert!(id.is_some(), "field {field} is interned");
        let Some(id) = id else { return };
        let phi = mem_ssa.memory_phi(3, id);
        assert_eq!(
            phi.map(|phi| &phi.location),
            Some(&loc),
            "the merge's phi for field {field} is that field's"
        );
        assert_eq!(
            phi.map(|phi| phi.result_version),
            mem_ssa.version_at_entry(&loc, 3),
            "field {field}'s phi defines the version the merge enters on"
        );
        assert!(mem_ssa.memory_phi(1, id).is_none(), "an arm merges nothing");
        assert!(
            mem_ssa.memory_phi(9, id).is_none(),
            "a block the function lacks has no phi"
        );
    }
    // The lookup agrees with a scan of every block's phis by location, for
    // every cell interned.
    for block in 0..4 {
        for location in mem_ssa.locations() {
            let Some(id) = mem_ssa.location_id(location) else {
                continue;
            };
            let scanned = mem_ssa
                .memory_phis(block)
                .iter()
                .find(|phi| &phi.location == location);
            assert_eq!(
                mem_ssa.memory_phi(block, id).map(|phi| phi.result_version),
                scanned.map(|phi| phi.result_version),
                "block {block}, {location:?}"
            );
        }
    }
}

/// A block that merges a location enters on its memory phi's version.
///
/// The phi defines the version at the block's entry, so it is what a read in
/// the block sees before any write there. Recorded ahead of the phis, the
/// entry version was the dominator's - the state before either arm stored - and
/// a caller asking what a load at the merge reads was told the function-entry
/// contents.
#[test]
fn a_merge_block_enters_on_its_phi_version() {
    let (ssa, obj) = diamond_over_one_field(true);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
    let loc = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    let merged = mem_ssa
        .memory_phis(3)
        .iter()
        .find(|phi| phi.location == loc)
        .map(|phi| phi.result_version);
    assert!(merged.is_some(), "the merge block carries a memory phi");
    assert_eq!(mem_ssa.version_at_entry(&loc, 3), merged);
    assert_eq!(mem_ssa.version_before(&loc, 3, 0), merged);
    for block in [1, 2] {
        assert_eq!(
            mem_ssa.version_before(&loc, block, 0),
            mem_ssa.version_at_entry(&loc, block)
        );
        assert_eq!(
            mem_ssa.version_before(&loc, block, 1).and_then(|version| {
                mem_ssa.definition(&MemoryVersion::new(loc.clone(), version))
            }),
            Some(MemoryDefSite::Store { block, instr: 0 }),
            "each branch observes its own write, independent of rename order"
        );
    }
    assert_eq!(
        merged.and_then(|v| mem_ssa.definition(&MemoryVersion::new(loc.clone(), v))),
        Some(MemoryDefSite::Phi { block: 3 })
    );
}

/// A predecessor that defines no memory must contribute the version it
/// *inherited* to the merge phi - not one leaked from a sibling arm.
///
/// This is the damaging form of a missing scope restore: the store in arm 1
/// leaks across to the pass-through arm, so the phi claims the value is defined
/// on a path that never touched memory, and any consumer forwarding through
/// that phi reads the wrong store on the wrong path.
#[test]
fn non_storing_arm_contributes_its_inherited_version() {
    let (ssa, obj) = diamond_over_one_field(false);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
    let loc = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    let b0_exit = mem_ssa.version_at_exit(&loc, 0);
    assert!(b0_exit.is_some(), "entry block has an exit version");

    // The pass-through arm neither enters nor leaves on a new version.
    assert_eq!(
        mem_ssa.version_at_entry(&loc, 2),
        b0_exit,
        "the pass-through arm inherits its dominator's version"
    );
    assert_eq!(mem_ssa.version_before(&loc, 2, 0), b0_exit);
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
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
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
            access: AccessQualifiers::FLAT,
        }));
    }
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// Two stores to different offsets off one base become two distinct memory
/// locations that provably do not alias.
///
/// Before the address was decoded these were `Indirect(addr_a)` and
/// `Indirect(addr_b)` - two unrelated ids, so no consumer could tell whether
/// they overlapped.
#[test]
fn distinct_stack_slots_get_disjoint_memory_locations() {
    let ssa = stores_through_ptradd_offsets(&[0, 8]);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

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

    // The base is the constant `0x1000`, so both are absolute numbers: the address
    // itself is the displacement, and the two sit one 8-byte step apart.
    let mut offsets: Vec<i64> = locations
        .iter()
        .filter_map(|loc| match loc {
            MemoryLocation::Indirect(indirect) => {
                assert_eq!(indirect.base, AddressBase::Number);
                Some(indirect.offset_bits)
            }
            _ => None,
        })
        .collect();
    offsets.sort_unstable();
    assert_eq!(
        offsets,
        vec![0x1000 * 8, 0x1000 * 8 + 64],
        "offsets are decoded in bits"
    );
}

/// Two addresses in one absolute space compare by extent; across the two
/// spaces, and against a value base, they may alias.
///
/// **A store into the frame defines no version of an image cell.** The two
/// addresses cannot be compared: the frame is addressed from the stack
/// pointer the function was entered with and the image cell by the address
/// the program states, so the address expressions must call them a
/// may-alias - a value can hold any address. Their regions say otherwise, and
/// that is the difference between a store defining one version and a store
/// defining a version of every location in the function: the image cell's
/// version is the same before and after the frame store.
///
/// The control is the third store, through a pointer the function was handed:
/// that one is in no placed region, so it keeps clobbering the image cell.
#[test]
fn a_frame_store_defines_no_version_of_an_image_cell() {
    let image = TestImage;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    // 0x2000 is writable in the test image; the frame is addressed from the
    // declared entry stack pointer.
    let cell = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        AddressBase::Symbol,
        None,
        0,
        0x2000 * 8,
        Some(32),
        None,
        PointerSize::Bit64,
    ));
    for handed_pointer in [false, true] {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 8);
        let sp = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        if let Some(variable) = ssa.variable_mut(sp) {
            variable.set_role(VariableRole::EntryStackPointer);
        }
        let value = local(&mut ssa, 1, 0, 0);
        let pointer = |ssa: &mut SsaFunction<MockTarget>, idx: u16, instr: usize| {
            ssa.create_variable(
                VariableOrigin::Local(idx),
                0,
                DefSite::instruction(0, instr),
                MockType::Ptr,
            )
        };
        let address = pointer(&mut ssa, 2, 1);
        let slot = pointer(&mut ssa, 3, 2);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: address,
            value: ConstValue::Symbol(0x2000),
        }));
        // The image cell's own store, then the frame store, then the control.
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr: address,
            value,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
        let after_image_store = block.instructions().len();
        block.add_instruction(instr(SsaOp::PtrAdd {
            dest: slot,
            base: if handed_pointer { value } else { sp },
            index: None,
            stride: 0,
            offset: -8,
            result_type: MockType::Ptr,
        }));
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr: slot,
            value,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
        let after_second_store = block.instructions().len();
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();

        let memory = MemorySsa::build(&ssa, &context);
        let before = memory.version_before(&cell, 0, after_image_store);
        let after = memory.version_before(&cell, 0, after_second_store);
        assert!(before.is_some(), "the image cell is versioned: {before:?}");
        if handed_pointer {
            assert_ne!(
                before, after,
                "a store through a pointer the function was handed may reach the image cell"
            );
        } else {
            assert_eq!(
                before,
                after,
                "a store into the frame reaches no image cell; the frame is \
                 {:?} and the cell {:?}",
                MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
                    AddressBase::Value(sp),
                    None,
                    0,
                    -64,
                    Some(32),
                    None,
                    PointerSize::Bit64,
                ))
                .region(&context, ssa.entry_stack_pointer()),
                cell.region(&context, ssa.entry_stack_pointer()),
            );
        }
    }
}

/// **A call whose effects are stated is counted apart from the barriers.**
/// It defines the locations it may write, as a store does, and the stats
/// counted it nowhere: a measure of how broad each defining operation is,
/// versions over stores and barriers, read a function of such calls as a few
/// stores clobbering hundreds of locations each.
#[test]
fn a_stated_call_is_counted_apart_from_the_barriers() {
    let (ssa, ..) = store_through_a_returned_pointer(false, false);
    let stats = MemorySsa::build(&ssa, &AnalysisContext::new(MACHINE)).stats();
    assert_eq!(stats.call_count, 1);
    assert_eq!(stats.barrier_count, 0);
}

/// One function for the footprint tests: frame byte `-8` holds a constant,
/// a call that reads and writes escaped memory follows (handed the slot's
/// address where `handed`, returning twice where `returns_twice`), and then
/// a store through the pointer the call returns.
///
/// Returns the function, the slot's location, and the positions just after
/// the call and just after the store.
fn store_through_a_returned_pointer(
    handed: bool,
    returns_twice: bool,
) -> (
    SsaFunction<MockTarget>,
    MemoryLocation<MockTarget>,
    usize,
    usize,
) {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 8);
    let sp = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = ssa.variable_mut(sp) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    let value = local(&mut ssa, 1, 0, 0);
    let slot = ssa.create_variable(
        VariableOrigin::Local(2),
        0,
        DefSite::instruction(0, 1),
        MockType::Ptr,
    );
    let returned = ssa.create_variable(
        VariableOrigin::Local(3),
        0,
        DefSite::instruction(0, 3),
        MockType::Ptr,
    );
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    block.add_instruction(instr(SsaOp::PtrAdd {
        dest: slot,
        base: sp,
        index: None,
        stride: 0,
        offset: -8,
        result_type: MockType::Ptr,
    }));
    block.add_instruction(instr(SsaOp::StoreIndirect {
        addr: slot,
        value,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    }));
    let effects = CallEffects {
        memory: CallMemory::Only(vec![CallAccess {
            location: CallLocation::Escaped,
            access: AccessMode::Update,
        }]),
        world: true,
        calls_back: true,
        captures: Capture::Any,
        termination: CallTermination::Returns,
        returns_twice,
        may_throw: false,
    };
    block.add_instruction(instr(SsaOp::from(
        CallData::new(
            Some(returned),
            Callee::Direct(7),
            if handed { vec![slot] } else { Vec::new() },
        )
        .with_effects(effects),
    )));
    let after_call = block.instructions().len();
    block.add_instruction(instr(SsaOp::StoreIndirect {
        addr: returned,
        value,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    }));
    let after_store = block.instructions().len();
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();
    let cell = MemoryLocation::Indirect(IndirectLocation::new(
        AddressBase::Value(sp),
        None,
        0,
        -64,
        Some(32),
        None,
        PointerSize::Bit64,
    ));
    (ssa, cell, after_call, after_store)
}

/// **A store through what a call returns reaches no frame byte the function
/// keeps to itself.** The call's result may point anywhere the program let
/// memory go, and no address of the frame went there, so the store defines
/// no version of the slot: the two footprints do not overlap
/// ([`analyssa::analysis::pointsto::Footprint`]). The controls: handed the
/// slot's address, the call may return it, and the store may then write the
/// slot; and a call that returns twice lets every byte of the frame go,
/// handed or not.
#[test]
fn a_store_through_what_a_call_returns_reaches_no_private_frame_byte() {
    let versioned = |handed: bool, returns_twice: bool| {
        let (ssa, cell, after_call, after_store) =
            store_through_a_returned_pointer(handed, returns_twice);
        let memory = MemorySsa::build(&ssa, &AnalysisContext::new(MACHINE));
        assert!(
            memory.locations().contains(&cell),
            "the slot is a location: {:?}",
            memory.locations()
        );
        let before = memory.version_before(&cell, 0, after_call);
        let after = memory.version_before(&cell, 0, after_store);
        assert!(before.is_some(), "the slot is versioned");
        before != after
    };
    assert!(!versioned(false, false), "the slot is the function's own");
    assert!(
        versioned(true, false),
        "the control: the call was handed it"
    );
    assert!(
        versioned(false, true),
        "the control: a call that returns twice lets the frame go"
    );
}

/// **The memory a function's analyses keep is the one its points-to
/// relation proves.** A store through what a call returns leaves the
/// function's own slot unversioned only because points-to finds no frame
/// address reaching the call: the memory the analyses build over the relation
/// they keep is the memory a fresh build makes, and says so too.
#[test]
fn the_kept_memory_reads_the_kept_points_to_relation() {
    let (ssa, cell, after_call, after_store) = store_through_a_returned_pointer(false, false);
    let context = AnalysisContext::new(MACHINE);
    let analyses = FunctionAnalyses::new(&ssa, &context);
    let kept = analyses.memory_ssa();
    // Compared before either is asked a version: the writes are indexed on
    // first ask.
    assert_eq!(
        format!("{kept:?}"),
        format!("{:?}", MemorySsa::build(&ssa, &context))
    );
    assert_eq!(
        kept.version_before(&cell, 0, after_call),
        kept.version_before(&cell, 0, after_store),
        "the store reaches no frame byte"
    );
}

/// **A frame slot and a stated absolute address are disjoint.** A slot is
/// storage of its own, and a symbol's or a bare integer's displacement is an
/// address the host or the program states - a link-time or program-stated
/// address names no slot of the function being analysed, so a store through
/// one cannot invalidate a load of the other. Two different slots were
/// already disjoint; the controls here are the pairs that must keep aliasing:
/// a slot against an address a *value* holds, since a pointer can hold the
/// slot's own address, and the two absolute spaces against each other, which
/// nothing relates.
#[test]
fn a_slot_and_a_stated_absolute_address_are_disjoint() {
    let at = |base: AddressBase, address: i64| {
        MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
            base,
            None,
            0,
            address * 8,
            Some(32),
            None,
            PointerSize::Bit64,
        ))
    };
    let local = at(AddressBase::Local(0), 0);
    let argument = at(AddressBase::Argument(1), 0);
    for slot in [local.clone(), argument.clone()] {
        for absolute in [
            at(AddressBase::Symbol, 0x1000),
            at(AddressBase::Number, 0x1000),
        ] {
            assert!(
                !slot.may_alias(&absolute),
                "a slot names no stated address: {slot:?} against {absolute:?}"
            );
            assert!(
                !absolute.may_alias(&slot),
                "and the question is symmetric: {absolute:?} against {slot:?}"
            );
        }
    }
    assert!(
        !local.may_alias(&argument),
        "two slots are each storage of their own"
    );
    let held = at(AddressBase::Value(SsaVarId::from_index(0)), 0);
    assert!(
        local.may_alias(&held),
        "an address a value holds may be the slot's own"
    );
    assert!(
        at(AddressBase::Symbol, 0x1000).may_alias(&at(AddressBase::Number, 0x1000)),
        "the two absolute spaces are not known to agree"
    );
}

/// A symbol is measured where the host measures its symbols and a bare integer
/// where the program computes, and the two need not agree - so `0x1000` stated
/// each way is not provably two cells, and not provably one.
#[test]
fn absolute_spaces_compare_within_themselves_only() {
    let at = |base: AddressBase, address: i64| {
        MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
            base,
            None,
            0,
            address * 8,
            Some(32),
            None,
            PointerSize::Bit64,
        ))
    };
    let symbol = at(AddressBase::Symbol, 0x1000);
    assert!(
        !symbol.may_alias(&at(AddressBase::Symbol, 0x1004)),
        "4 bytes apart"
    );
    assert!(symbol.must_alias(&at(AddressBase::Symbol, 0x1000)));
    assert!(!at(AddressBase::Number, 0x1000).may_alias(&at(AddressBase::Number, 0x2000)));
    assert!(
        symbol.may_alias(&at(AddressBase::Number, 0x2000)),
        "a symbol and a number are in spaces nothing relates"
    );
    assert!(!symbol.must_alias(&at(AddressBase::Number, 0x1000)));
    let value = at(AddressBase::Value(SsaVarId::from_index(0)), 0);
    assert!(symbol.may_alias(&value), "a pointer may hold any address");
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
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

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
    b0.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

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
/// conservative answer, so the memory pass finds nothing to do - which is
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
            access: AccessQualifiers::FLAT,
        }));
        if block_id == BLOCKS {
            block.add_instruction(instr(SsaOp::returns(None)));
        } else {
            block.add_instruction(instr(SsaOp::Jump {
                target: block_id + 1,
            }));
        }
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

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

/// A store inside an exception handler must get a merge point where the handler
/// rejoins the normal path.
///
/// ```text
///   B0  obj, v_try                    clause: try [1,2), handler B2..B6
///   B1  StoreField obj.f = v_try      (protected)   ->  B3
///   B2  v_catch = const               (handler entry) ->  B4
///   B4  StoreField obj.f = v_catch    ->  B5
///   B5  (nothing)                     ->  B3
///   B3  LoadField obj.f; Return
/// ```
///
/// The handler's store is deliberately **two** hops from the join. A store in a
/// block that is itself an immediate predecessor of the join still lands in the
/// join's frontier -- `compute_dominance_frontiers` inserts before it notices
/// the unreachable-node sentinel and stops -- so a depth-1 handler gets its phi
/// by accident and stays correct. That is why the existing suite is green.
///
/// One hop further and the truncated walk never reaches the join: no memory phi
/// is placed, `seed_from_dominator` finds the block's entry version is the
/// try-path store, and `passes::memory` forwards that value into the load. On
/// the path that actually threw, `obj.f` holds the handler's value.
#[test]
fn a_store_in_a_handler_gets_a_phi_at_the_rejoin() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 6);
    let obj = local(&mut ssa, 0, 0, 0);
    let try_value = local(&mut ssa, 1, 0, 1);
    let catch_value = local(&mut ssa, 2, 2, 0);
    let loaded = local(&mut ssa, 3, 3, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: try_value,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::StoreField {
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 1u32,
        value: try_value,
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: catch_value,
        value: ConstValue::I32(2),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 4 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::LoadField {
        access: AccessQualifiers::FLAT,
        dest: loaded,
        object: obj,
        field: 1u32,
    }));
    b3.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b3);

    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::StoreField {
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 1u32,
        value: catch_value,
    }));
    b4.add_instruction(instr(SsaOp::Jump { target: 5 }));
    ssa.add_block(b4);

    let mut b5 = SsaBlock::new(5);
    b5.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b5);

    ssa.set_exception_handlers(vec![handler_clause(1, 2, 2, 6)]);
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));

    assert!(
        !mem_ssa.memory_phis(3).is_empty(),
        "the rejoin block must carry a memory phi; without one the try-path \
         store is forwarded across the exception edge into the load"
    );
}
/// A store the protected region performs is **not** visible in the handler.
///
/// Block dominance says the guard ran wholly before the handler; the runtime
/// leaves it part-way. So the region's store may or may not have happened, and
/// the handler's entry version has to say "unknown" rather than name that
/// store - otherwise a later pass forwards a value the throw pre-empted.
#[test]
fn a_store_before_a_handler_is_not_visible_inside_it() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 3);
    let obj = local(&mut ssa, 0, 0, 0);
    let inside = local(&mut ssa, 1, 0, 1);
    let loaded = local(&mut ssa, 2, 2, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: inside,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    // B1 is the protected region, and its store is the one in question.
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::StoreField {
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 1u32,
        value: inside,
    }));
    b1.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::LoadField {
        access: AccessQualifiers::FLAT,
        dest: loaded,
        object: obj,
        field: 1u32,
    }));
    b2.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b2);

    ssa.set_exception_handlers(vec![handler_clause(1, 2, 2, 3)]);
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
    let location = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);

    let version = mem_ssa.version_at_entry(&location, 2);
    assert_eq!(mem_ssa.version_before(&location, 2, 0), version);
    assert!(
        version.is_some(),
        "the handler must be renamed at all -- it was unreachable before"
    );

    let definition =
        version.and_then(|version| mem_ssa.definition(&MemoryVersion::new(location, version)));
    assert!(
        matches!(definition, Some(MemoryDefSite::ExceptionEntry { block: 2 })),
        "the region's store must not be assumed to have run: the runtime \
         dispatched here from part-way through it, got {definition:?}"
    );
}

/// The precision half of the same rule: a store that completed *before* the
/// protected region started is as visible in the handler as anywhere else, and
/// must not be discarded along with the region's own writes.
#[test]
fn a_store_completed_before_the_region_is_still_visible_in_the_handler() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 3);
    let obj = local(&mut ssa, 0, 0, 0);
    let early = local(&mut ssa, 1, 0, 1);
    let loaded = local(&mut ssa, 2, 2, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: obj,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: early,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::StoreField {
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 1u32,
        value: early,
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::LoadField {
        access: AccessQualifiers::FLAT,
        dest: loaded,
        object: obj,
        field: 1u32,
    }));
    b2.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b2);

    ssa.set_exception_handlers(vec![handler_clause(1, 2, 2, 3)]);
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
    let location = MemoryLocation::<MockTarget>::InstanceField(obj, 1u32);

    let definition = mem_ssa
        .version_at_entry(&location, 2)
        .and_then(|version| mem_ssa.definition(&MemoryVersion::new(location, version)));
    assert!(
        matches!(definition, Some(MemoryDefSite::Store { block: 0, .. })),
        "a store outside the region completed, so it survives into the handler, \
         got {definition:?}"
    );
}

/// The batch form of the per-location queries answers what they answer, for
/// every block, in the order the locations first appear.
#[test]
fn entry_definitions_agree_with_the_per_location_queries() {
    let (ssa, obj) = diamond_over_one_field(true);
    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
    let loc = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    for block in 0..ssa.block_count() {
        let batch: Vec<_> = mem_ssa.entry_definitions(block).collect();
        assert!(!batch.is_empty(), "block {block} enters on some version");
        for (location, site) in &batch {
            let version = mem_ssa.version_at_entry(location, block);
            assert!(
                version.is_some(),
                "block {block}: a listed location has an entry version"
            );
            assert_eq!(
                version
                    .and_then(|version| mem_ssa
                        .definition(&MemoryVersion::new((*location).clone(), version))),
                Some(*site),
                "block {block}: {location:?}"
            );
        }
        assert!(
            batch.iter().any(|(location, _)| **location == loc),
            "block {block} lists the field"
        );
    }

    // Past the last block there is nothing to enter.
    assert_eq!(mem_ssa.entry_definitions(ssa.block_count()).count(), 0);
}

/// **A location a join does not merge enters on its dominator's version.**
/// The recorded renaming keeps, per block, only what that block itself
/// defines: a location it neither merges nor writes is read back through the
/// dominator chain. A location stored before a diamond and in neither arm has
/// no phi at the join - phis go on the iterated dominance frontier of the
/// blocks that define it, and only the head defines this one - so the join's
/// entry version is the head's exit version, reached by walking up.
///
/// The pair: the location the arms do store gets a phi at the join, so its
/// entry version is the merge's and not the head's. Both readings come out of
/// the same chain, which is what makes the sparse record equal to a dense one.
#[test]
fn a_location_no_phi_merges_enters_on_its_dominators_version() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
    let obj = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 0, 1);
    let cond = local(&mut ssa, 2, 0, 2);

    let mut b0 = SsaBlock::new(0);
    for (dest, value) in [(obj, 42i32), (val, 100), (cond, 1)] {
        b0.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
    }
    // Written before the diamond and in neither arm: no join merges it.
    b0.add_instruction(instr(SsaOp::StoreField {
        access: AccessQualifiers::FLAT,
        object: obj,
        field: 7u32,
        value: val,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    for arm in [1usize, 2] {
        let mut block = SsaBlock::new(arm);
        // Written in both arms, so the join merges it.
        block.add_instruction(instr(SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object: obj,
            field: 5u32,
            value: val,
        }));
        block.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(block);
    }

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b3);
    ssa.recompute_uses();

    let mem_ssa = MemorySsa::<MockTarget>::build(&ssa, &AnalysisContext::new(MACHINE));
    let unmerged = MemoryLocation::<MockTarget>::InstanceField(obj, 7u32);
    let merged = MemoryLocation::<MockTarget>::InstanceField(obj, 5u32);

    assert!(
        !mem_ssa
            .memory_phis(3)
            .iter()
            .any(|phi| phi.location == unmerged),
        "the join merges only what the arms define"
    );
    assert_eq!(
        mem_ssa.version_at_entry(&unmerged, 3),
        mem_ssa.version_at_exit(&unmerged, 0),
        "a location no phi merges enters the join on its dominator's version"
    );
    assert_ne!(
        mem_ssa.version_at_entry(&merged, 3),
        mem_ssa.version_at_exit(&merged, 0),
        "the location the arms store is merged at the join, not inherited"
    );
}
