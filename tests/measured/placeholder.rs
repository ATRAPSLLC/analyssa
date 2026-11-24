//! `SsaVarId::PLACEHOLDER` is not a variable, and nothing sizes storage by it.
//!
//! Its index is `u32::MAX - 1`. A pass that grows a table to hold it asks for
//! gigabytes, so every table the SSA machinery builds from the ids it meets must
//! leave the placeholder out. The placeholder is legitimately present between
//! phi placement and renaming, and a deserialized function can carry it
//! anywhere, so these cases are reachable from the public API.
//!
//! The measure is the largest single allocation the work makes on its thread. A
//! table sized by the placeholder is at least half a gigabyte.

use std::{collections::BTreeMap, process};

use analyssa::{
    analysis::{AnalysisContext, SsaVerifier, VerifyLevel},
    events::NullListener,
    ir::{
        block::SsaBlock,
        function::{SsaConstructOptions, SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::MemoryOptimizationPass,
    scheduling::{FunctionPass, PassEnv},
    testing::{MACHINE, MockTarget, MockType, diamond_phi_fixture},
};

use crate::measuring::{assert_bounded, measured};

/// The diamond with an extra variable-table row holding the placeholder.
fn with_placeholder_row() -> SsaFunction<MockTarget> {
    let mut ssa = diamond_phi_fixture();
    let mut row = match ssa.variables().first() {
        Some(row) => row.clone(),
        None => process::abort(),
    };
    row.set_id(SsaVarId::PLACEHOLDER);
    ssa.variables_mut().push(row);
    ssa
}

/// The diamond with an unrenamed phi, as phi placement leaves one.
fn with_placeholder_phi() -> SsaFunction<MockTarget> {
    let mut ssa = diamond_phi_fixture();
    if let Some(block) = ssa.block_mut(3) {
        block
            .phi_nodes_mut()
            .push(PhiNode::new(SsaVarId::PLACEHOLDER, VariableOrigin::Phi));
    }
    ssa
}

/// A one-block function whose copy reads the placeholder, and whose second
/// definition is the placeholder itself when `defines_placeholder` is set.
fn reading_the_placeholder(defines_placeholder: bool) -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
    let dest = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
        dest,
        src: SsaVarId::PLACEHOLDER,
    }));
    if defines_placeholder {
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: SsaVarId::PLACEHOLDER,
            value: ConstValue::I32(1),
        }));
    }
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest))));
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa
}

/// A phi that names only itself, with the placeholder as its result: trivial,
/// unread, and so a candidate for removal that has no variable to remove.
fn with_self_referential_placeholder_phi() -> SsaFunction<MockTarget> {
    let mut ssa = reading_the_placeholder(false);
    if let Some(block) = ssa.block_mut(0) {
        let mut phi = PhiNode::new(SsaVarId::PLACEHOLDER, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(SsaVarId::PLACEHOLDER, 0));
        block.phi_nodes_mut().push(phi);
    }
    ssa
}

/// Every shape of function that names the placeholder.
fn naming_the_placeholder() -> Vec<(&'static str, SsaFunction<MockTarget>)> {
    vec![
        ("placeholder row", with_placeholder_row()),
        ("placeholder phi", with_placeholder_phi()),
        (
            "placeholder self phi",
            with_self_referential_placeholder_phi(),
        ),
        ("placeholder read", reading_the_placeholder(false)),
        ("placeholder definition", reading_the_placeholder(true)),
    ]
}

/// Verification builds its tables from the function's real variables only.
#[test]
fn verification_allocates_nothing_sized_by_the_placeholder() {
    for (name, ssa) in naming_the_placeholder() {
        for level in [VerifyLevel::Quick, VerifyLevel::Standard, VerifyLevel::Full] {
            let (errors, largest) = measured(|| SsaVerifier::new(&ssa).verify(level));
            assert_bounded(&format!("{name} at {level:?}"), largest);
            if name != "placeholder row" {
                assert!(!errors.is_empty(), "{name} at {level:?} is reported");
            }
        }
    }
}

/// Repair, use recomputation and an empty edit session size nothing by it.
#[test]
fn repair_allocates_nothing_sized_by_the_placeholder() {
    for (name, mut ssa) in naming_the_placeholder() {
        let ((), largest) = measured(|| {
            ssa.recompute_uses();
            ssa.repair_ssa();
        });
        assert_bounded(name, largest);
    }
}

/// A rebuild over IR that names the placeholder finishes, and sizes nothing by
/// it: an operand with no group is given one, and a read of a value nothing
/// defines is dropped from the liveness sets rather than indexed.
#[test]
fn a_rebuild_allocates_nothing_sized_by_the_placeholder() {
    for (name, mut ssa) in naming_the_placeholder() {
        let (_, largest) = measured(|| ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
        assert_bounded(name, largest);
    }
}

/// Renumbering a function whose variable table holds the placeholder yields a
/// dense table and sizes nothing by it.
#[test]
fn a_reindex_allocates_nothing_sized_by_the_placeholder() {
    for (name, mut ssa) in [
        ("placeholder row", with_placeholder_row()),
        ("placeholder phi", with_placeholder_phi()),
    ] {
        let (_, largest) = measured(|| ssa.reindex_variables());
        assert_bounded(name, largest);
        for (index, variable) in ssa.variables().iter().enumerate() {
            assert_eq!(variable.id().index(), index, "{name}: ids are dense");
        }
    }
}

/// A rename group set for the placeholder is ignored: the placeholder names no
/// variable, so it has no group to read back and no slot to grow the table to.
#[test]
fn a_rename_group_for_the_placeholder_is_ignored() {
    let mut ssa = diamond_phi_fixture();
    let before = ssa.max_rename_group();
    let ((), largest) = measured(|| ssa.set_rename_group(SsaVarId::PLACEHOLDER, 7));
    assert_bounded("set_rename_group", largest);
    assert_eq!(ssa.rename_group(SsaVarId::PLACEHOLDER), u32::MAX);
    assert_eq!(ssa.max_rename_group(), before);
}

/// Propagating a copy whose destination is the placeholder replaces nothing:
/// the placeholder is not a variable, so no use of it can be rewritten and no
/// table slot is recorded for it.
#[test]
fn propagating_a_copy_of_the_placeholder_replaces_nothing() {
    for (name, mut ssa) in naming_the_placeholder() {
        let Some(source) = ssa.variables().first().map(|row| row.id()) else {
            process::abort()
        };
        let copies = BTreeMap::from([(SsaVarId::PLACEHOLDER, source)]);
        let (result, largest) = measured(|| {
            ssa.edit(SsaEditOptions::new(), |editor| {
                let result = editor.propagate_copies(&copies);
                assert_eq!(result.total_replaced, 0, "{name}");
                Ok(())
            })
        });
        assert_bounded(name, largest);
        assert!(result.is_ok_and(|report| !report.changed), "{name}");
    }
}

/// The frame analysis indexes every use; a read of the placeholder is not a use
/// of a value, so it sizes nothing and the dead frame write is still removed.
#[test]
fn the_frame_analysis_ignores_a_read_of_the_placeholder() {
    let mut ssa = crate::frame::frame_write_reading(SsaVarId::PLACEHOLDER);
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

/// A variable created after rows were appended to the table is found at its own
/// id. The table can be longer than the ids allocated for it, and an id drawn
/// from the allocator alone would name a row that already exists, so the new
/// variable would sit at an index other than its id.
#[test]
fn a_variable_created_after_an_appended_row_is_found_at_its_id() {
    let mut ssa = with_placeholder_row();
    let rows = ssa.variables().len();
    let id = ssa.create_variable(VariableOrigin::Local(9), 0, DefSite::entry(), MockType::I32);
    assert_eq!(
        id.index(),
        rows,
        "the new id follows every row of the table"
    );
    assert_eq!(
        ssa.variables().get(id.index()).map(|row| row.id()),
        Some(id)
    );
}
