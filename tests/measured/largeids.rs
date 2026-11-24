//! A variable id the variable table does not hold is tolerated, not sized for.
//!
//! Malformed or deserialized IR can name any 32-bit id. The verifier reports it
//! and the repair, construction and analysis machinery must carry on, and none of it
//! may allocate a table as large as the id on the way: their storage follows the
//! function's variable bound, and an id past it costs one entry. A container that
//! grew to hold it would ask for gigabytes - a bit set with a bit per id below
//! three billion is 375 MB, and a vector with a word per id is 24 GB.
//!
//! Two things are measured. The largest single allocation the work makes on its
//! thread stays within a megabyte, and the work gives the answer it gives for an
//! id just past the table: a far id is the same malformed shape as a near one,
//! so the two must agree.

use std::{
    collections::{BTreeMap, BTreeSet},
    process,
};

use analyssa::{
    PointerSize,
    analysis::{
        DefUseIndex, FoldForest, FoldPolicy, SsaVerifier, VerifierError, VerifyLevel,
        context::AnalysisContext, execution::StaticExecution,
    },
    events::NullListener,
    ir::{
        block::SsaBlock,
        function::{SsaConstructOptions, SsaEditOptions, SsaFunction, TrivialPhiOptions},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, UseSite, VariableOrigin},
        varstore::{VarMap, VarSet},
    },
    passes::{MemoryOptimizationPass, OpaquePredicatePass},
    scheduling::{FunctionPass, PassEnv},
    testing::{
        MACHINE, MockTarget, MockType, const_i32_return, diamond_chain_fixture,
        diamond_phi_fixture, loop_counter_fixture, scalar_rewrite_fixture, straight_line_fixture,
        try_catch_fixture,
    },
};

use crate::measuring::{assert_bounded, measured};

/// An id far beyond any variable table a test builds.
const FAR: SsaVarId = SsaVarId::from_index(3_000_000_000);

/// An id large enough that a table with a word per id below it is gigabytes,
/// small enough that a bit set sized by it stays cheap.
const LARGE: SsaVarId = SsaVarId::from_index(100_000_000);

/// An id past the one real variable's, which no allocator issued.
const NEAR: SsaVarId = SsaVarId::from_index(1_000);

/// A one-block function with one real variable, defined by a constant.
///
/// Returns the function, the open block, so a shape can add what it needs, and
/// the variable; [`finished`] closes the block and adds it.
fn base() -> (SsaFunction<MockTarget>, SsaBlock<MockTarget>, SsaVarId) {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
    let value = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(1),
    }));
    (ssa, block, value)
}

/// Adds `block` to `ssa` after closing it with a return of `value`.
fn finished(
    mut ssa: SsaFunction<MockTarget>,
    mut block: SsaBlock<MockTarget>,
    value: Option<SsaVarId>,
) -> SsaFunction<MockTarget> {
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(value)));
    ssa.add_block(block);
    ssa
}

/// An instruction reads `id`.
fn reading(id: SsaVarId) -> SsaFunction<MockTarget> {
    let (ssa, mut block, value) = base();
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
        dest: value,
        src: id,
    }));
    finished(ssa, block, Some(value))
}

/// An instruction defines `id`, twice.
fn defining(id: SsaVarId) -> SsaFunction<MockTarget> {
    let (ssa, mut block, value) = base();
    for constant in [1, 2] {
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: id,
            value: ConstValue::I32(constant),
        }));
    }
    finished(ssa, block, Some(value))
}

/// A phi operand names `id`.
fn naming_in_a_phi(id: SsaVarId) -> SsaFunction<MockTarget> {
    let (mut ssa, mut block, value) = base();
    let result = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(0), MockType::I32);
    let mut phi = PhiNode::new(result, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(id, 0));
    block.phi_nodes_mut().push(phi);
    finished(ssa, block, Some(value))
}

/// A variable-table row carries `id`, a recorded use of it, and an instruction
/// that reads it: the id is registered, so the use-list comparison buckets its
/// occurrences.
fn carrying_in_a_row(id: SsaVarId) -> SsaFunction<MockTarget> {
    let (mut ssa, block, _) = base();
    let Some(mut row) = ssa.variables().first().cloned() else {
        process::abort()
    };
    row.set_id(id);
    row.add_use(UseSite::instruction(0, 1));
    ssa.variables_mut().push(row);
    finished(ssa, block, Some(id))
}

/// Every shape over `id`, by name.
fn over(id: SsaVarId) -> Vec<(&'static str, SsaFunction<MockTarget>)> {
    vec![
        ("read", reading(id)),
        ("definition", defining(id)),
        ("phi operand", naming_in_a_phi(id)),
        ("row", carrying_in_a_row(id)),
    ]
}

/// Whether `error` reports the far id as out of range.
fn names_the_far_id(error: &VerifierError) -> bool {
    matches!(error, VerifierError::VariableIdOutOfRange { var, .. } if *var == FAR)
}

/// Whether `error` reports a variable table whose rows are not dense.
fn reports_a_sparse_table(error: &VerifierError) -> bool {
    matches!(error, VerifierError::VariableTableNotDense { .. })
}

/// A function and the error its verification must produce.
type Shape = (
    &'static str,
    SsaFunction<MockTarget>,
    fn(&VerifierError) -> bool,
);

/// Every shape, with the error each must produce.
fn reported() -> Vec<Shape> {
    over(FAR)
        .into_iter()
        .map(|(name, ssa)| {
            let expected: fn(&VerifierError) -> bool = if name == "row" {
                reports_a_sparse_table
            } else {
                names_the_far_id
            };
            (name, ssa, expected)
        })
        .collect()
}

/// Verification at every level names the far id and allocates nothing sized by it.
#[test]
fn verification_reports_a_far_id_without_sizing_a_table_for_it() {
    for (name, ssa, expected) in reported() {
        for level in [VerifyLevel::Quick, VerifyLevel::Standard, VerifyLevel::Full] {
            let (errors, largest) = measured(|| SsaVerifier::new(&ssa).verify(level));
            assert_bounded(&format!("{name} at {level:?}"), largest);
            assert!(
                errors.iter().any(expected),
                "{name} at {level:?}: expected error absent from {errors:?}"
            );
        }
    }
}

/// The index-bounds scan on its own does the same.
#[test]
fn the_index_bounds_scan_reports_a_far_id_without_sizing_a_table_for_it() {
    for (name, ssa, expected) in reported() {
        let (errors, largest) = measured(|| SsaVerifier::new(&ssa).index_bounds_only());
        assert_bounded(name, largest);
        if name != "row" {
            assert!(
                errors.iter().any(expected),
                "{name}: expected error absent from {errors:?}"
            );
        }
    }
}

/// Repair over a definition of a far id sizes nothing by it: the table of
/// instruction definitions, the bit sets and the rename groups all follow the
/// variable count, and the id costs entries.
#[test]
fn repairing_a_definition_of_a_far_id_sizes_nothing_by_it() {
    let mut ssa = defining(FAR);
    let ((), largest) = measured(|| ssa.repair_ssa());
    assert_bounded("repair_ssa", largest);
}

/// A definition of a large id, which a table with a slot per id below it would
/// make gigabytes, costs repair no more than a far one does.
#[test]
fn repairing_a_definition_of_a_large_id_sizes_nothing_by_it() {
    let mut ssa = defining(LARGE);
    let ((), largest) = measured(|| ssa.repair_ssa());
    assert_bounded("repair_ssa", largest);
}

/// The frame analysis indexes every use; a use of a large id is grouped by id
/// and does not size the index, and the dead frame write is still removed.
#[test]
fn the_frame_analysis_groups_a_use_of_a_large_id_without_sizing_for_it() {
    let mut ssa = crate::frame::frame_write_reading(LARGE);
    assert_eq!(crate::frame::stores(&ssa), 1);
    let (changed, largest) = measured(|| {
        MemoryOptimizationPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    assert_bounded("memory::run", largest);
    assert!(changed, "the dead frame write is removed");
    assert_eq!(crate::frame::stores(&ssa), 0);
}

/// The same holds for a use of a far id.
#[test]
fn the_frame_analysis_groups_a_use_of_a_far_id_without_sizing_for_it() {
    let mut ssa = crate::frame::frame_write_reading(FAR);
    let (changed, largest) = measured(|| {
        MemoryOptimizationPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    assert_bounded("memory::run", largest);
    assert!(changed, "the dead frame write is removed");
    assert_eq!(crate::frame::stores(&ssa), 0);
}

/// A dense map made for a function's variables holds a far id, and every way
/// of using it afterwards, without sizing a slot table for it: one entry in the
/// ordered part is all it costs.
#[test]
fn a_var_map_holds_a_far_id_without_sizing_for_it() {
    let (read, largest) = measured(|| {
        let mut map: VarMap<u64> = VarMap::new(8);
        map.insert(SsaVarId::from_index(2), 1);
        map.insert(FAR, 2);
        map.insert(LARGE, 3);
        map.insert(SsaVarId::PLACEHOLDER, 4);
        map.or_insert(FAR, 5);
        let found = (map.get(FAR).copied(), map.get(LARGE).copied());
        map.retain(|_, entry| *entry != 3);
        let converted = map.clone().map(|_, entry| entry + 1);
        let collected: VarMap<u64> = converted.iter().map(|(var, entry)| (var, *entry)).collect();
        (
            found,
            map.keys().collect::<Vec<_>>(),
            collected.get(FAR).copied(),
            map.remove(FAR),
        )
    });
    assert_bounded("a far id in a dense map", largest);
    assert_eq!(
        read,
        (
            (Some(2), Some(3)),
            vec![SsaVarId::from_index(2), FAR],
            Some(3),
            Some(2)
        )
    );
}

/// The measure sees a dense map's slots: a map made for a count reserves room
/// for all of them, so the test above is not bounded by an allocation the
/// measure cannot see.
#[test]
fn the_measure_sees_the_slots_a_var_map_reserves() {
    let count = 1 << 20;
    let (map, largest) = measured(|| VarMap::<u64>::new(count));
    assert!(map.is_empty());
    assert!(
        largest >= count.saturating_mul(size_of::<Option<u64>>()),
        "the measure sees the reserved slots: {largest}"
    );
}

/// A table collected from entries naming a far id sizes its slots by its
/// entries, not by the id.
#[test]
fn a_collected_var_map_does_not_size_its_slots_by_a_far_id() {
    let (collected, largest) = measured(|| {
        [(SsaVarId::from_index(1), 1u64), (FAR, 2), (LARGE, 3)]
            .into_iter()
            .collect::<VarMap<u64>>()
    });
    assert_bounded("a collected map", largest);
    assert_eq!(collected.get(FAR), Some(&2));
    assert_eq!(collected.get(LARGE), Some(&3));
    assert_eq!(collected.get(SsaVarId::from_index(1)), Some(&1));
}

/// A serialized table naming a far id reads back without a slot table sized by
/// it, the far id and the ones around it held.
#[cfg(feature = "serde")]
#[test]
fn a_deserialized_var_map_does_not_size_its_slots_by_a_far_id() {
    let text = format!(r#"{{"1":10,"{}":20,"3":30}}"#, FAR.as_u32());
    let (map, largest) = measured(|| serde_json::from_str::<VarMap<u64>>(&text).ok());
    assert_bounded("a deserialized map", largest);
    let Some(map) = map else { process::abort() };
    assert_eq!(map.get(FAR), Some(&20));
    assert_eq!(map.get(SsaVarId::from_index(3)), Some(&30));
    assert_eq!(map.len(), 3);
}

/// The `k`th far id.
const fn far(k: usize) -> SsaVarId {
    SsaVarId::from_index(3_000_000_000_usize.saturating_add(k))
}

/// The `k`th near id: just past any table a case builds, in the same order as
/// the far ones.
const fn near(k: usize) -> SsaVarId {
    SsaVarId::from_index(1_000_usize.saturating_add(k))
}

/// What the machinery does to a function, by name.
#[derive(Clone, Copy, Debug)]
enum Work {
    /// [`SsaFunction::construct_ssa`], renaming only.
    Construct,
    /// [`SsaFunction::repair_ssa`].
    Repair,
}

impl Work {
    /// Every kind of work.
    const ALL: [Self; 2] = [Self::Construct, Self::Repair];

    /// Does the work to `ssa`, and says how it went.
    fn on(self, ssa: &mut SsaFunction<MockTarget>) -> String {
        match self {
            Self::Construct => format!(
                "{:?}",
                ssa.construct_ssa(&SsaConstructOptions::renaming_only())
            ),
            Self::Repair => {
                ssa.repair_ssa();
                String::new()
            }
        }
    }
}

/// Everything observable about `ssa` and the outcome of the work done to it,
/// with the ids in `ids` as they stand.
fn fingerprint(ssa: &SsaFunction<MockTarget>, outcome: &str, ids: &[SsaVarId]) -> String {
    let mut text = format!("{outcome}\n{ssa}");
    for row in ssa.variables() {
        text.push_str(&format!("\n{row:?}"));
    }
    for index in 0..ssa.var_id_bound().saturating_add(2) {
        text.push_str(&format!(
            "\ngroup {index}: {}",
            ssa.rename_group(SsaVarId::from_index(index))
        ));
    }
    for id in ids {
        text.push_str(&format!("\ngroup of {id}: {}", ssa.rename_group(*id)));
    }
    for error in SsaVerifier::new(ssa).verify(VerifyLevel::Full) {
        text.push_str(&format!("\n{error:?}"));
    }
    text
}

/// Spells `text`, written for the far ids `0..count`, for the near ones.
fn as_near(text: &str, count: usize) -> String {
    (0..count).fold(text.to_owned(), |text, k| {
        text.replace(&far(k).index().to_string(), &near(k).index().to_string())
    })
}

/// A deterministic stream of numbers.
struct Rng(u64);

impl Rng {
    /// The next number below `bound`.
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let bound = u64::try_from(bound).unwrap_or(u64::MAX).max(1);
        usize::try_from(self.0.checked_rem(bound).unwrap_or(0)).unwrap_or(0)
    }
}

/// A fixture with a few of the malformed shapes spliced in, over the ids
/// `extras`.
///
/// The seed alone decides the structure, so two calls with the same seed and
/// ids of the same order build the same function.
fn mutated(seed: u64, extras: &[SsaVarId]) -> SsaFunction<MockTarget> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut ssa = match rng.below(6) {
        0 => const_i32_return(5),
        1 => scalar_rewrite_fixture(),
        2 => diamond_phi_fixture(),
        3 => loop_counter_fixture(),
        4 => straight_line_fixture(4),
        _ => diamond_chain_fixture(2),
    };
    for _ in 0..=rng.below(3) {
        let Some(&extra) = extras.get(rng.below(extras.len())) else {
            continue;
        };
        let block = rng.below(ssa.block_count());
        match rng.below(6) {
            // An instruction reads the id in place of its first operand.
            0 => {
                let Some(block) = ssa.block_mut(block) else {
                    continue;
                };
                let count = block.instructions().len();
                if let Some(instr) = block.instructions_mut().get_mut(rng.below(count)) {
                    let mut replaced = false;
                    instr.op_mut().replace_uses_with(|_| {
                        let first = !replaced;
                        replaced = true;
                        first.then_some(extra)
                    });
                }
            }
            // An instruction defines the id.
            1 => {
                let Some(block) = ssa.block_mut(block) else {
                    continue;
                };
                let at = rng.below(block.instructions().len());
                let value = i32::try_from(rng.below(100)).unwrap_or(0);
                block.instructions_mut().insert(
                    at,
                    SsaInstruction::synthetic(SsaOp::Const {
                        dest: extra,
                        value: ConstValue::I32(value),
                    }),
                );
            }
            // A phi operand names the id, in place of one or beside them.
            2 => {
                let predecessor = rng.below(ssa.block_count());
                let replace = rng.below(2) == 0;
                let Some(block) = ssa.block_mut(block) else {
                    continue;
                };
                let count = block.phi_nodes().len();
                if count == 0 {
                    continue;
                }
                let at = rng.below(count);
                if let Some(phi) = block.phi_nodes_mut().get_mut(at) {
                    let operands = phi.operands_mut();
                    let slot = rng.below(operands.len());
                    match operands.get_mut(slot) {
                        Some(operand) if replace => operand.set_value(extra),
                        _ => phi.add_operand(PhiOperand::new(extra, predecessor)),
                    }
                }
            }
            // A phi in the block has the id for its result.
            3 => {
                let predecessor = rng.below(ssa.block_count());
                let Some(block) = ssa.block_mut(block) else {
                    continue;
                };
                let mut phi = PhiNode::new(extra, VariableOrigin::Phi);
                phi.add_operand(PhiOperand::new(extra, predecessor));
                block.phi_nodes_mut().push(phi);
            }
            // A table row carries the id.
            4 => {
                let Some(mut row) = ssa.variables().first().cloned() else {
                    continue;
                };
                row.set_id(extra);
                ssa.variables_mut().push(row);
            }
            // The id is declared to belong to a rename group.
            _ => ssa.set_rename_group(extra, u32::try_from(rng.below(4)).unwrap_or(0)),
        }
    }
    ssa
}

/// How many cases each differential runs.
const CASES: u64 = 600;

/// The ids a case may splice in.
const EXTRAS: usize = 3;

/// A set sized for a few ids holds a far one and every way of using it
/// afterwards, as an ordered set does, without a bit for each id below it.
#[test]
fn a_var_set_holds_a_far_id_without_sizing_for_it() {
    let (read, largest) = measured(|| {
        let mut set = VarSet::new(8);
        let fresh = [
            set.insert(SsaVarId::from_index(2)),
            set.insert(FAR),
            set.insert(NEAR),
            set.insert(SsaVarId::PLACEHOLDER),
            set.insert(FAR),
        ];
        let found = [
            set.contains(SsaVarId::from_index(2)),
            set.contains(FAR),
            set.contains(NEAR),
            set.contains(SsaVarId::PLACEHOLDER),
            set.contains(far(1)),
        ];
        let held = set.iter().collect::<Vec<_>>();
        let removed = [set.remove(FAR), set.remove(FAR)];
        (
            fresh,
            found,
            held,
            removed,
            set.len(),
            set.clone().is_empty(),
        )
    });
    assert_bounded("a far id in a var set", largest);
    assert_eq!(
        read,
        (
            [true, true, true, false, false],
            [true, true, true, false, false],
            vec![SsaVarId::from_index(2), NEAR, FAR],
            [true, false],
            2,
            false
        )
    );
}

/// Any run of inserts and removals over dense, far and placeholder ids leaves a
/// set holding what an ordered set holds, in the order it iterates.
#[test]
fn a_var_set_holds_what_an_ordered_set_holds() {
    let mut set = VarSet::new(16);
    let mut model = BTreeSet::new();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for step in 0..6000 {
        let var = match rng.below(11) {
            0 => SsaVarId::PLACEHOLDER,
            1 | 2 => far(rng.below(5)),
            _ => SsaVarId::from_index(rng.below(40)),
        };
        let real = !var.is_placeholder();
        match rng.below(3) {
            0 => assert_eq!(
                set.insert(var),
                real && model.insert(var),
                "insert, step {step}"
            ),
            1 => assert_eq!(
                set.remove(var),
                real && model.remove(&var),
                "remove, step {step}"
            ),
            _ => assert_eq!(
                set.contains(var),
                model.contains(&var),
                "contains, step {step}"
            ),
        }
        assert_eq!(set.len(), model.len(), "len, step {step}");
        assert_eq!(set.is_empty(), model.is_empty(), "is_empty, step {step}");
    }
    assert!(set.iter().eq(model.iter().copied()), "iteration order");
    assert!(!model.is_empty(), "the run ends holding ids");
}

/// A construction or repair of each malformed shape allocates nothing sized by
/// the id.
#[test]
fn construction_and_repair_size_nothing_by_a_far_id() {
    for work in Work::ALL {
        for (name, mut ssa) in over(FAR) {
            let (_, largest) = measured(|| work.on(&mut ssa));
            assert_bounded(&format!("{work:?} of the {name} shape"), largest);
        }
    }
}

/// Each malformed shape gives the same result for a far id as for a near one.
#[test]
fn a_far_id_gives_the_result_a_near_id_does() {
    for work in Work::ALL {
        for ((name, mut at_far), (_, mut at_near)) in over(FAR).into_iter().zip(over(NEAR)) {
            let outcome = work.on(&mut at_far);
            let far_text = fingerprint(&at_far, &outcome, &[FAR]);
            let outcome = work.on(&mut at_near);
            let near_text = fingerprint(&at_near, &outcome, &[NEAR]);
            let far_text = far_text.replace(&FAR.index().to_string(), &NEAR.index().to_string());
            assert_eq!(far_text, near_text, "{work:?} of the {name} shape");
        }
    }
}

/// The differential holds over functions with several malformed uses spliced
/// into them, and no case allocates by an id.
#[test]
fn spliced_far_ids_give_the_results_near_ids_do() {
    let far_ids: Vec<SsaVarId> = (0..EXTRAS).map(far).collect();
    let near_ids: Vec<SsaVarId> = (0..EXTRAS).map(near).collect();
    for work in Work::ALL {
        for seed in 0..CASES {
            let mut at_far = mutated(seed, &far_ids);
            let mut at_near = mutated(seed, &near_ids);
            let (far_text, largest) = measured(|| {
                let outcome = work.on(&mut at_far);
                fingerprint(&at_far, &outcome, &far_ids)
            });
            assert_bounded(&format!("{work:?}, seed {seed}"), largest);
            let outcome = work.on(&mut at_near);
            let near_text = fingerprint(&at_near, &outcome, &near_ids);
            assert_eq!(
                as_near(&far_text, EXTRAS),
                near_text,
                "{work:?}, seed {seed}"
            );
        }
    }
}

/// The differential notices a result that differs, so a passing run means the
/// far and near results agreed.
#[test]
fn the_differential_notices_a_difference() {
    let mut at_far = mutated(7, &[far(0), far(1), far(2)]);
    let mut at_near = mutated(7, &[near(0), near(1), near(2)]);
    Work::Repair.on(&mut at_far);
    let far_text = as_near(&fingerprint(&at_far, "", &[far(0)]), EXTRAS);
    assert_eq!(
        far_text,
        {
            Work::Repair.on(&mut at_near);
            fingerprint(&at_near, "", &[near(0)])
        },
        "control: the same work agrees"
    );
    at_near.set_rename_group(near(0), 41);
    assert_ne!(far_text, fingerprint(&at_near, "", &[near(0)]));
}

/// A definition of the placeholder or of a far id is looked up without a set
/// sized by either, and only the far id is found defined.
#[test]
fn looking_up_a_definition_sizes_nothing_by_the_ids_defined() {
    let (found, largest) = measured(|| {
        let at_far = defining(FAR);
        let at_placeholder = defining(SsaVarId::PLACEHOLDER);
        (
            at_far.get_definition(FAR).is_some(),
            at_far.get_definition(far(1)).is_some(),
            at_placeholder
                .get_definition(SsaVarId::PLACEHOLDER)
                .is_some(),
            at_placeholder.get_definition(FAR).is_some(),
        )
    });
    assert_bounded("get_definition", largest);
    assert_eq!(found, (true, false, false, false));
}

/// Asking whether a far id is a parameter allocates nothing by it.
#[test]
fn asking_whether_a_far_id_is_a_parameter_sizes_nothing_by_it() {
    let ssa = reading(FAR);
    let (asked, largest) = measured(|| {
        (
            ssa.is_parameter_variable(FAR),
            ssa.is_parameter_variable(SsaVarId::from_index(0)),
        )
    });
    assert_bounded("is_parameter_variable", largest);
    assert_eq!(asked, (None, None));
}

/// Propagating a copy whose destination is a far id allocates nothing by the
/// id, and decides as it does for a near one.
#[test]
fn copy_propagation_of_a_far_destination_sizes_nothing_by_it() {
    let run = |id: SsaVarId| {
        let (ssa, mut block, value) = base();
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
            dest: id,
            src: value,
        }));
        let mut ssa = finished(ssa, block, Some(id));
        let copies = BTreeMap::from([(id, value)]);
        let mut propagated = None;
        let edited = ssa.edit(SsaEditOptions::new(), |editor| {
            let result = editor.propagate_copies(&copies);
            propagated = Some((
                result.total_replaced,
                result.fully_propagated.contains(id),
                result.partially_propagated.is_empty(),
            ));
            Ok(())
        });
        (edited.is_ok(), propagated, format!("{ssa}"))
    };
    let (at_far, largest) = measured(|| run(FAR));
    assert_bounded("propagate_copies", largest);
    let (edited, propagated, text) = at_far;
    assert!(edited, "the session succeeds");
    let (_, near_propagated, near_text) = run(NEAR);
    assert_eq!(propagated, near_propagated);
    assert_eq!(as_near(&text, 1), near_text);
}

/// Eliminating dead and trivial phis, whose results are a far id, allocates
/// nothing by it.
#[test]
fn eliminating_phis_named_by_a_far_id_sizes_nothing_by_it() {
    let build = |id: SsaVarId, read: bool| {
        let (ssa, mut block, value) = base();
        let mut phi = PhiNode::new(id, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(value, 0));
        block.phi_nodes_mut().push(phi);
        finished(ssa, block, Some(if read { id } else { value }))
    };
    for read in [false, true] {
        let mut at_far = build(FAR, read);
        let mut at_near = build(NEAR, read);
        let (counts, largest) = measured(|| {
            (
                at_far.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None }),
                at_far.eliminate_dead_phis(),
            )
        });
        assert_bounded(&format!("phi elimination, read {read}"), largest);
        assert_eq!(
            counts,
            (
                at_near.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None }),
                at_near.eliminate_dead_phis()
            ),
            "read {read}"
        );
        assert_eq!(
            at_far.block(0).map(|block| block.phi_nodes().len()),
            at_near.block(0).map(|block| block.phi_nodes().len()),
            "read {read}"
        );
    }
}

/// A handler entry that pops a far id adopts the pop as a runtime value, with
/// nothing sized by the id.
#[test]
fn adopting_a_handler_entry_value_named_by_a_far_id_sizes_nothing_by_it() {
    let build = |id: SsaVarId| {
        let mut ssa = try_catch_fixture();
        if let Some(block) = ssa.block_mut(2) {
            block
                .instructions_mut()
                .insert(0, SsaInstruction::synthetic(SsaOp::Pop { value: id }));
        }
        ssa
    };
    let mut at_far = build(FAR);
    let (adopted, largest) = measured(|| at_far.adopt_handler_entry_values());
    assert_bounded("adopt_handler_entry_values", largest);
    assert_eq!(adopted, 1);
    assert_eq!(adopted, build(NEAR).adopt_handler_entry_values());
    assert!(at_far.get_definition(FAR).is_some(), "the value is defined");
}

/// The definition and use index of each shape is built with nothing sized by
/// the id, and the real variable is still indexed.
#[test]
fn the_def_use_index_sizes_nothing_by_a_far_id() {
    for (name, ssa) in over(FAR) {
        let (index, largest) = measured(|| DefUseIndex::build_with_ops(&ssa));
        assert_bounded(&format!("the index of the {name} shape"), largest);
        assert!(
            index.def_site(SsaVarId::from_index(0)).is_some(),
            "{name}: the real variable is defined"
        );
    }
}

/// A fold policy that takes every default.
struct Folds;

impl FoldPolicy<MockTarget> for Folds {}

/// The fold decisions for each shape allocate nothing by the id.
#[test]
fn the_fold_forest_sizes_nothing_by_a_far_id() {
    for (name, ssa) in over(FAR) {
        let defuse = DefUseIndex::build(&ssa);
        let (forest, largest) =
            measured(|| FoldForest::compute(&ssa, &defuse, &Folds, PointerSize::Bit64, None));
        assert_bounded(&format!("the folds of the {name} shape"), largest);
        assert!(
            !forest.is_folded(FAR),
            "{name}: a value past the bound is never folded"
        );
    }
}

/// The opaque-predicate pass builds its definition cache for each shape with
/// nothing sized by the id, and decides as it does for a near one.
#[test]
fn the_predicate_pass_sizes_nothing_by_a_far_id() {
    for ((name, mut at_far), (_, mut at_near)) in over(FAR).into_iter().zip(over(NEAR)) {
        let (changed, largest) = measured(|| {
            OpaquePredicatePass::new().run(
                &mut at_far,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert_bounded(&format!("the predicate pass on the {name} shape"), largest);
        assert_eq!(
            changed,
            OpaquePredicatePass::new().run(
                &mut at_near,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true),
            "{name}"
        );
    }
}

/// The static execution of each shape allocates nothing by the id.
#[test]
fn static_execution_sizes_nothing_by_a_far_id() {
    for (name, ssa) in over(FAR) {
        let context = AnalysisContext::new(MACHINE);
        let (_, largest) = measured(|| StaticExecution::new(&ssa, &context).run());
        assert_bounded(&format!("the execution of the {name} shape"), largest);
    }
}

/// A rename group declared for a far id is held, read back and counted
/// without a table sized by the id, and a variable created afterwards is
/// ungrouped.
#[test]
fn a_rename_group_for_a_far_id_sizes_nothing_by_it() {
    let (mut ssa, _, value) = base();
    let ((read, created), largest) = measured(|| {
        ssa.set_rename_group(FAR, 6);
        ssa.set_rename_group(value, 2);
        let read = (
            ssa.rename_group(FAR),
            ssa.rename_group(value),
            ssa.rename_group(far(1)),
            ssa.max_rename_group(),
        );
        let created =
            ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::entry(), MockType::I32);
        (read, (created, ssa.rename_group(created)))
    });
    assert_bounded("rename groups", largest);
    assert_eq!(read, (6, 2, u32::MAX, Some(6)));
    assert_eq!(created.1, u32::MAX);
}

/// A renumbering of a table with a far id in a row maps it to its new place
/// without a table sized by the old id.
#[test]
fn reindexing_a_table_with_a_far_row_sizes_nothing_by_it() {
    let mut ssa = carrying_in_a_row(FAR);
    let (renumbered, largest) = measured(|| ssa.reindex_variables());
    assert_bounded("reindex_variables", largest);
    assert_eq!(renumbered, 1, "the far row moved to the next slot");
    assert!(
        ssa.variables()
            .iter()
            .enumerate()
            .all(|(slot, row)| row.id().index() == slot),
        "the table is dense"
    );
}
