//! Host facts survive the things that break a side table.
//!
//! A host's per-value knowledge used to live in a table beside the function,
//! keyed by `SsaVarId`, and every renumbering silently re-pointed it at another
//! value. Carried in the IR, it instead moves with the value: these tests pin
//! the three moments where one value becomes another - derivation, join,
//! substitution - plus the two where a table would simply be wrong: a rollback
//! and a round trip.

use std::process;

use analyssa::{
    Endianness, PointerSize, Target,
    analysis::{
        Machine,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
    },
    events::NullListener,
    facts::{Derivation, Join, Substitution},
    ir::{
        block::SsaBlock,
        function::{
            SsaConstructOptions, SsaDefSpec, SsaEditOptions, SsaFunction, SsaFunctionBuilder,
        },
        instruction::SsaInstruction,
        ops::{AccessQualifiers, CmpKind, SsaOp},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
    },
    passes::{
        AlgebraicSimplificationPass, CopyPropagationPass, DeadCodeEliminationPass,
        EqualityPropagationPass, GlobalValueNumberingPass, MemoryOptimizationPass,
    },
    scheduling::{FunctionPass, PassEnv, PassTransaction},
    testing::{
        FactfulTarget, MACHINE, MockOrigin, MockSource, MockTarget, MockType, diamond_phi_fixture,
    },
};
#[cfg(feature = "serde")]
use serde_json::{from_str, to_string};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn result_or_abort<T, E>(result: Result<T, E>) -> T {
    result.unwrap_or_else(|_| process::abort())
}

/// The origin recorded for `var`, or `None` when the function has no such
/// variable.
fn origin_of(ssa: &SsaFunction<FactfulTarget>, var: SsaVarId) -> Option<MockOrigin> {
    ssa.variable(var).map(|variable| *variable.facts())
}

/// Declares a local carrying `origin`.
fn local(ssa: &mut SsaFunction<FactfulTarget>, idx: u16, def: DefSite, origin: u8) -> SsaVarId {
    let id = ssa.create_variable(VariableOrigin::Local(idx), 0, def, MockType::I32);
    if let Some(var) = ssa.variable_mut(id) {
        var.set_facts(MockOrigin(Some(origin)));
    }
    id
}

/// `v0 = 1; v1 = v0` in one block, so copy propagation has something to
/// forward and dead-code elimination something to drop.
fn copy_chain() -> (SsaFunction<FactfulTarget>, SsaVarId, SsaVarId) {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 2);
    let source = local(&mut ssa, 0, DefSite::instruction(0, 0), 7);
    let copy = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, 1),
        MockType::I32,
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: source,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
        dest: copy,
        src: source,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(copy))));
    ssa.add_block(block);
    ssa.recompute_uses();
    (ssa, source, copy)
}

/// A diamond whose two arms define the same storage, merged by a phi.
fn diamond(left_origin: u8, right_origin: u8) -> (SsaFunction<FactfulTarget>, SsaVarId) {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 4);
    let cond = local(&mut ssa, 0, DefSite::instruction(0, 0), 1);
    let left = local(&mut ssa, 1, DefSite::instruction(1, 0), left_origin);
    let right = local(&mut ssa, 2, DefSite::instruction(2, 0), right_origin);
    let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    for (block_idx, dest, value) in [(1, left, 10), (2, right, 20)] {
        let mut block = SsaBlock::new(block_idx);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        ssa.add_block(block);
    }

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(left, 1));
    phi.add_operand(PhiOperand::new(right, 2));
    join.add_phi(phi);
    join.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(merged))));
    ssa.add_block(join);
    ssa.recompute_uses();
    (ssa, merged)
}

/// Two definitions of one storage slot, joined where they meet: the shape that
/// makes reconstruction mint new ids for the definitions rather than leave them
/// alone.
fn two_definitions_of_one_slot() -> (SsaFunction<FactfulTarget>, SsaVarId) {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 2);
    let cond = local(&mut ssa, 0, DefSite::instruction(0, 0), 1);
    let first = local(&mut ssa, 1, DefSite::instruction(1, 0), 4);
    let second = local(&mut ssa, 1, DefSite::instruction(2, 0), 4);
    // Both definitions are versions of local 1: the same storage, so the same
    // rename group.
    let group = 1u32;
    ssa.set_rename_group(first, group);
    ssa.set_rename_group(second, group);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    for (block_idx, dest, value) in [(1, first, 10), (2, second, 20)] {
        let mut block = SsaBlock::new(block_idx);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        ssa.add_block(block);
    }

    let mut exit = SsaBlock::new(3);
    exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(first))));
    ssa.add_block(exit);
    ssa.recompute_uses();
    (ssa, first)
}

#[test]
fn a_renamed_value_keeps_what_the_host_knew_about_it() {
    let (mut ssa, first) = two_definitions_of_one_slot();
    let before = origin_of(&ssa, first);
    assert_eq!(before, Some(MockOrigin(Some(4))));

    // Reconstruction versions the slot, minting a fresh id per definition.
    result_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let carrying = ssa
        .variables()
        .iter()
        .filter(|var| *var.facts() == MockOrigin(Some(4)))
        .count();
    assert!(
        carrying >= 2,
        "each renamed definition is still the storage the host named: {:?}",
        ssa.variables()
            .iter()
            .map(|var| (var.id(), *var.facts()))
            .collect::<Vec<_>>()
    );
}

/// A phi reconstruction clears and places again, for the same storage in the
/// same block, is the same phi, and keeps what the host knew about it - even
/// where a join of its operands would not say it.
#[test]
fn a_phi_placed_again_is_the_phi_it_replaces() {
    let (mut ssa, first) = two_definitions_of_one_slot();
    let second = some_or_abort(
        ssa.variables()
            .iter()
            .map(|var| var.id())
            .find(|id| *id != first && ssa.rename_group(*id) == ssa.rename_group(first)),
    );
    if let Some(var) = ssa.variable_mut(second) {
        var.set_facts(MockOrigin(Some(5)));
    }
    let merged = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(3), MockType::I32);
    ssa.set_rename_group(merged, 1);
    if let Some(var) = ssa.variable_mut(merged) {
        var.set_facts(MockOrigin(Some(9)));
    }
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(1));
    phi.add_operand(PhiOperand::new(first, 1));
    phi.add_operand(PhiOperand::new(second, 2));
    if let Some(exit) = ssa.block_mut(3) {
        exit.add_phi(phi);
        if let Some(ret) = exit.instructions_mut().last_mut() {
            *ret = SsaInstruction::synthetic(SsaOp::returns(Some(merged)));
        }
    }
    ssa.recompute_uses();

    result_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let placed = some_or_abort(
        ssa.block(3)
            .and_then(|exit| exit.phi_nodes().first())
            .map(|phi| phi.result()),
    );
    assert_eq!(
        origin_of(&ssa, placed),
        Some(MockOrigin(Some(9))),
        "the arms disagree, so only the phi it stands for could say this"
    );
}

#[test]
fn a_declared_role_survives_reconstruction() {
    let (mut ssa, first) = two_definitions_of_one_slot();
    if let Some(var) = ssa.variable_mut(first) {
        var.set_role(VariableRole::EntryStackPointer);
    }
    assert!(ssa.entry_stack_pointer().is_some());

    result_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    assert!(
        ssa.entry_stack_pointer().is_some(),
        "a value minted to stand for the stack pointer is still the stack pointer"
    );
}

#[test]
fn a_merge_of_agreeing_values_keeps_their_answer() {
    let (mut ssa, merged) = diamond(5, 5);
    ssa.repair_ssa();
    assert_eq!(
        origin_of(&ssa, merged),
        Some(MockOrigin(Some(5))),
        "both arms say the same thing, so the merge says it too"
    );
}

#[test]
fn a_merge_of_disagreeing_values_keeps_neither() {
    let (mut ssa, merged) = diamond(5, 6);
    ssa.repair_ssa();
    assert_eq!(
        origin_of(&ssa, merged),
        Some(MockOrigin(None)),
        "the arms disagree, so the merge knows nothing"
    );
}

#[test]
fn a_phi_that_knows_its_storage_keeps_it() {
    let (mut ssa, merged) = diamond(5, 6);
    if let Some(var) = ssa.variable_mut(merged) {
        var.set_facts(MockOrigin(Some(9)));
    }
    ssa.repair_ssa();
    assert_eq!(
        origin_of(&ssa, merged),
        Some(MockOrigin(Some(9))),
        "the host asked to keep what the placement knew, and the refresh asked it"
    );
}

#[test]
fn a_substitution_folds_the_replaced_value_into_the_survivor() {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 2);
    let known = local(&mut ssa, 0, DefSite::instruction(0, 0), 9);
    let unknown = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, 1),
        MockType::I32,
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: known,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: unknown,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(known))));
    ssa.add_block(block);
    ssa.recompute_uses();

    // The survivor knows nothing; the value it replaces does.
    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        let _ = editor.replace_uses_checked_with(known, unknown, None, Substitution::Gvn);
        Ok(())
    }));

    assert_eq!(
        origin_of(&ssa, unknown),
        Some(MockOrigin(Some(9))),
        "the survivor adopts what the replaced value knew"
    );
}

#[test]
fn copy_propagation_carries_the_origin_it_forwards() {
    let (mut ssa, source, copy) = copy_chain();
    assert_eq!(origin_of(&ssa, copy), Some(MockOrigin(None)));

    let method = 0u32;
    assert!(
        CopyPropagationPass.propagate(
            &mut ssa,
            &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, 4)),
            |_, _| {}
        )
    );

    // The copy is gone and its uses read the source, which knew where it came
    // from all along.
    assert_eq!(origin_of(&ssa, source), Some(MockOrigin(Some(7))));
}

/// `v0 = 1; v1 = v0; return v0`: a copy that knows its storage (`v1`, 7)
/// of a value that knows none, which nothing reads any more, as when an
/// earlier rewrite moved its readers onto its source.
fn unread_copy() -> (SsaFunction<FactfulTarget>, SsaVarId, SsaVarId) {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 2);
    let source = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let copy = local(&mut ssa, 1, DefSite::instruction(0, 1), 7);
    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: source,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
        dest: copy,
        src: source,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(source))));
    ssa.add_block(block);
    ssa.recompute_uses();
    (ssa, source, copy)
}

/// A copy nothing reads stands for its source as wholly as one whose readers
/// were rewritten: what it knew folds into the source rather than dying with
/// it. At `-O0` it is the local a call's result was stored to, whose readers
/// a conversion fold already read the result through.
#[test]
fn copy_propagation_carries_the_origin_of_a_copy_nothing_reads() {
    let (mut ssa, source, _) = unread_copy();
    let method = 0u32;
    let _ = CopyPropagationPass.propagate(
        &mut ssa,
        &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, 4)),
        |_, _| {},
    );
    assert_eq!(origin_of(&ssa, source), Some(MockOrigin(Some(7))));
}

/// Every substitution path takes the same view of a value nothing reads:
/// the editor's checked replacement and a resolved batch alike.
#[test]
fn every_substitution_of_an_unread_value_carries_its_facts() {
    let (mut ssa, source, copy) = unread_copy();
    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        let _ = editor.replace_uses_checked_with(copy, source, None, Substitution::Gvn);
        Ok(())
    }));
    assert_eq!(
        origin_of(&ssa, source),
        Some(MockOrigin(Some(7))),
        "checked"
    );

    let (mut ssa, source, copy) = unread_copy();
    let _ = ssa.replace_uses_resolved(&[(copy, source)]);
    assert_eq!(
        origin_of(&ssa, source),
        Some(MockOrigin(Some(7))),
        "resolved"
    );
}

/// A substitution refused at every use is none: `v1 = v0 + 1` cannot read
/// itself, so `v0` is not replaced by `v1` there, and `v1` learns nothing of
/// `v0`.
#[test]
fn a_substitution_refused_everywhere_carries_nothing() {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 2);
    let old = local(&mut ssa, 0, DefSite::instruction(0, 0), 7);
    let new = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, 1),
        MockType::I32,
    );
    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: old,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
        dest: new,
        src: old,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(new))));
    ssa.add_block(block);
    ssa.recompute_uses();
    let _ = ssa.replace_uses_resolved(&[(old, new)]);
    assert_eq!(origin_of(&ssa, new), Some(MockOrigin(None)));
}

#[test]
fn a_rolled_back_edit_restores_the_facts_too() {
    let (mut ssa, source, _) = copy_chain();
    let mut transaction: PassTransaction<FactfulTarget> = PassTransaction::new();

    let outcome = transaction.run_group(&mut ssa, |ssa| {
        if let Some(var) = ssa.variable_mut(source) {
            var.set_facts(MockOrigin(Some(99)));
        }
        // Corrupt the function so the group is rejected: an instruction after
        // the terminator.
        if let Some(block) = ssa.block_mut(0) {
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        }
        true
    });

    assert!(!outcome.changed(), "the corrupt group is rolled back");
    assert_eq!(
        origin_of(&ssa, source),
        Some(MockOrigin(Some(7))),
        "a rollback restores the host's facts with the IR"
    );
}

#[test]
fn an_instruction_fact_travels_with_a_move_and_not_with_a_copy() {
    let (mut ssa, _, _) = copy_chain();
    if let Some(instr) = ssa
        .block_mut(0)
        .and_then(|block| block.instructions_mut().first_mut())
    {
        instr.set_facts(MockSource::at(0x401000));
    }

    // A copy is a different program point, so this host's policy drops the
    // address.
    let copy_id = {
        let mut id = None;
        result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
            id = Some(editor.duplicate_instruction((0, 0), (0, 1))?);
            Ok(())
        }));
        some_or_abort(id)
    };
    let copy_at = some_or_abort(ssa.locate_instruction(copy_id));
    let copy_facts = some_or_abort(
        ssa.block(copy_at.0)
            .and_then(|block| block.instructions().get(copy_at.1))
            .map(|instr| instr.facts().clone()),
    );
    assert_eq!(
        copy_facts,
        MockSource::default(),
        "a copy is a different point"
    );

    // A move is the same instruction elsewhere, so it keeps the address.
    let source_at = some_or_abort(ssa.blocks().iter().enumerate().find_map(|(block, b)| {
        b.instructions()
            .iter()
            .position(|instr| *instr.facts() == MockSource::at(0x401000))
            .map(|idx| (block, idx))
    }));
    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        let taken = editor.detach_instruction(source_at.0, source_at.1)?;
        editor.insert_instruction(0, 0, taken)?;
        Ok(())
    }));
    assert!(
        ssa.blocks().iter().any(|block| block
            .instructions()
            .iter()
            .any(|instr| *instr.facts() == MockSource::at(0x401000))),
        "a move keeps the host's fact about the instruction"
    );
}

#[test]
fn facts_survive_the_whole_normalization_pipeline() {
    let (mut ssa, source, _) = copy_chain();
    let method = 0u32;
    let ptr = PointerSize::Bit64;

    let _ = AlgebraicSimplificationPass.run(
        &mut ssa,
        &PassEnv::new(
            &method,
            &NullListener,
            AnalysisContext::new(Machine {
                ptr_size: ptr,
                ..MACHINE
            }),
        ),
    );
    let _ = CopyPropagationPass.propagate(
        &mut ssa,
        &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, 4)),
        |_, _| {},
    );
    let _ = GlobalValueNumberingPass.run(
        &mut ssa,
        &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
    );
    let _ = DeadCodeEliminationPass.run(
        &mut ssa,
        &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::DeadCodeRounds, 4)),
    );

    assert!(
        ssa.variables()
            .iter()
            .any(|var| *var.facts() == MockOrigin(Some(7))),
        "the value's origin is still recorded somewhere in the function"
    );
    let _ = source;
}

#[test]
fn a_function_fact_is_part_of_the_function() {
    let (mut ssa, _, _) = copy_chain();
    ssa.facts_mut().name = "decrypt".to_string();

    let clone = ssa.clone();
    assert_eq!(clone.facts().name, "decrypt");

    result_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    assert_eq!(
        ssa.facts().name,
        "decrypt",
        "reconstruction rewrites values, not what the host knows about the function"
    );
}

#[test]
fn a_derivation_asks_the_host_what_its_cause_preserves() {
    let (mut ssa, source, _) = copy_chain();
    let renamed = ssa.create_variable_derived(
        source,
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(0, 0),
        MockType::I32,
        Derivation::Rename,
    );
    assert_eq!(origin_of(&ssa, renamed), Some(MockOrigin(Some(7))));

    let duplicated = ssa.create_variable_derived(
        source,
        VariableOrigin::Local(0),
        2,
        DefSite::instruction(0, 0),
        MockType::I32,
        Derivation::BlockDuplication,
    );
    assert_eq!(
        origin_of(&ssa, duplicated),
        Some(MockOrigin(None)),
        "this host says a duplicate is not the original's storage"
    );

    let joined = ssa.create_variable_joined(
        &[source, renamed],
        VariableOrigin::Phi,
        0,
        DefSite::phi(0),
        MockType::I32,
        Join::PlacedPhi,
    );
    assert_eq!(origin_of(&ssa, joined), Some(MockOrigin(Some(7))));
}

/// A loop phi whose other operands agree takes their facts: its operand that
/// is itself says only that the value is kept around the loop.
#[test]
fn a_loop_phi_takes_the_facts_of_what_enters_it() {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 2);
    let source = local(&mut ssa, 0, DefSite::instruction(0, 0), 7);
    let kept = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: source,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);
    let mut header = SsaBlock::new(1);
    let mut phi = PhiNode::new(kept, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(source, 0));
    phi.add_operand(PhiOperand::new(kept, 1));
    header.add_phi(phi);
    header.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
        condition: kept,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(header);
    let mut exit = SsaBlock::new(2);
    exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(kept))));
    ssa.add_block(exit);
    ssa.recompute_uses();

    assert_eq!(origin_of(&ssa, kept), Some(MockOrigin(None)));
    ssa.refresh_phi_facts();
    assert_eq!(origin_of(&ssa, kept), Some(MockOrigin(Some(7))));
}

#[cfg(feature = "serde")]
#[test]
fn facts_round_trip() {
    let (mut ssa, source, _) = copy_chain();
    ssa.facts_mut().name = "round".to_string();
    let encoded = result_or_abort(to_string(&ssa));
    let decoded: SsaFunction<FactfulTarget> = result_or_abort(from_str(&encoded));

    assert_eq!(decoded.facts().name, "round");
    assert_eq!(origin_of(&decoded, source), Some(MockOrigin(Some(7))));
}

#[test]
fn a_host_that_carries_nothing_pays_nothing() {
    // The unit fact type is zero-sized, so the fields cost no memory and the
    // phi-join walk is skipped outright.
    assert_eq!(size_of::<<MockTarget as Target>::VariableFacts>(), 0);
    let mut ssa = diamond_phi_fixture();
    assert_eq!(ssa.refresh_phi_facts(), 0);
}

/// The value a function stores to a cell at `offset`'s owner, read back a
/// word at a time from `offset`: `v0` the cell's address, `v1` the stored
/// doubleword (no origin of its own), `v3` the word loaded, carrying the
/// slot's origin `5` as the lift names a load of a frame slot.
fn stored_doubleword_read_at(offset: i64) -> (SsaFunction<FactfulTarget>, SsaVarId) {
    let v = SsaVarId::from_index;
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(2, 0);
    for (index, ty) in [MockType::Ptr, MockType::U64, MockType::Ptr, MockType::U32]
        .into_iter()
        .enumerate()
    {
        ssa.create_variable(
            VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
            0,
            DefSite::instruction(0, index),
            ty,
        );
    }
    if let Some(var) = ssa.variable_mut(v(3)) {
        var.set_facts(MockOrigin(Some(5)));
    }
    let mut b0 = SsaBlock::new(0);
    for op in [
        SsaOp::LoadArg {
            dest: v(0),
            arg_index: 0,
        },
        SsaOp::LoadArg {
            dest: v(1),
            arg_index: 1,
        },
        SsaOp::StoreIndirect {
            addr: v(0),
            value: v(1),
            value_type: MockType::U64,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::PtrAdd {
            dest: v(2),
            base: v(0),
            index: None,
            stride: 1,
            offset,
            result_type: MockType::Ptr,
        },
        SsaOp::LoadIndirect {
            dest: v(3),
            addr: if offset == 0 { v(0) } else { v(2) },
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::returns(Some(v(3))),
    ] {
        b0.add_instruction(SsaInstruction::synthetic(op));
    }
    ssa.add_block(b0);
    ssa.recompute_uses();
    let machine = Machine {
        byte_order: Some(Endianness::Little),
        ..MACHINE
    };
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(machine))
            )
            .is_ok()
    );
    (ssa, v(1))
}

/// **A value stored whole and read back from its start is kept where it was
/// stored.** A doubleword stored to a cell and read back a word at a time - a
/// `long long` a 32-bit target keeps in a frame slot - leaves the load the
/// part it read, and only the load knows the slot; the host is told the
/// stored value is what the slot holds. A load from further in reads a part
/// alone, and the control: it says nothing about where the whole lives.
#[test]
fn a_value_stored_whole_and_read_from_its_start_is_held_where_it_was_stored() {
    let (ssa, whole) = stored_doubleword_read_at(0);
    assert!(
        !ssa.blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .any(|instr| { matches!(instr.op(), SsaOp::LoadIndirect { .. }) }),
        "the load is answered by the stored value\n{ssa}"
    );
    assert_eq!(origin_of(&ssa, whole), Some(MockOrigin(Some(5))), "{ssa}");

    let (ssa, whole) = stored_doubleword_read_at(4);
    assert_eq!(origin_of(&ssa, whole), Some(MockOrigin(None)), "{ssa}");
}

/// `al + bl` and `ah + bh + carry`, composed into the double-width sum: the
/// four halves are arguments, and the sum's halves carry `low_origin` and
/// `high_origin`, the slots a lift names a `long long`'s two words after.
/// Returns the function after the algebraic pass's fixpoint, and the sum.
fn sum_kept_in_halves(
    low_origin: Option<u8>,
    high_origin: Option<u8>,
) -> (SsaFunction<FactfulTarget>, SsaVarId) {
    let v = SsaVarId::from_index;
    let types = [
        MockType::U32, // 0 al
        MockType::U32, // 1 ah
        MockType::U32, // 2 bl
        MockType::U32, // 3 bh
        MockType::U32, // 4 low = al + bl
        MockType::I1,  // 5 carry flag
        MockType::U32, // 6 carry
        MockType::U32, // 7 ah + bh
        MockType::U32, // 8 high
        MockType::U8,  // 9 shift
        MockType::U64, // 10 low widened
        MockType::U64, // 11 high widened
        MockType::U64, // 12 high placed
        MockType::U64, // 13 sum
    ];
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(4, 0);
    for (index, ty) in types.into_iter().enumerate() {
        ssa.create_variable(
            VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
            0,
            DefSite::instruction(0, index),
            ty,
        );
    }
    for (value, origin) in [(v(4), low_origin), (v(8), high_origin)] {
        if let Some(var) = ssa.variable_mut(value) {
            var.set_facts(MockOrigin(origin));
        }
    }
    let widen = |dest: usize, operand: usize| SsaOp::IntConv {
        dest: v(dest),
        operand: v(operand),
        target: MockType::U64,
        overflow_check: false,
        unsigned: true,
    };
    let mut b0 = SsaBlock::new(0);
    for op in [
        SsaOp::LoadArg {
            dest: v(0),
            arg_index: 0,
        },
        SsaOp::LoadArg {
            dest: v(1),
            arg_index: 1,
        },
        SsaOp::LoadArg {
            dest: v(2),
            arg_index: 2,
        },
        SsaOp::LoadArg {
            dest: v(3),
            arg_index: 3,
        },
        SsaOp::Add {
            dest: v(4),
            left: v(0),
            right: v(2),
            flags: None,
        },
        SsaOp::Clt {
            dest: v(5),
            left: v(4),
            right: v(0),
            unsigned: true,
            signaling: false,
        },
        SsaOp::IntConv {
            dest: v(6),
            operand: v(5),
            target: MockType::U32,
            overflow_check: false,
            unsigned: true,
        },
        SsaOp::Add {
            dest: v(7),
            left: v(1),
            right: v(3),
            flags: None,
        },
        SsaOp::Add {
            dest: v(8),
            left: v(7),
            right: v(6),
            flags: None,
        },
        SsaOp::Const {
            dest: v(9),
            value: ConstValue::U8(32),
        },
        widen(10, 4),
        widen(11, 8),
        SsaOp::Shl {
            dest: v(12),
            value: v(11),
            amount: v(9),
            flags: None,
        },
        SsaOp::Or {
            dest: v(13),
            left: v(12),
            right: v(10),
            flags: None,
        },
        SsaOp::returns(Some(v(13))),
    ] {
        b0.add_instruction(SsaInstruction::synthetic(op));
    }
    ssa.add_block(b0);
    ssa.recompute_uses();
    let machine = Machine {
        byte_order: Some(Endianness::Little),
        ..MACHINE
    };
    for _ in 0..8 {
        if !(AlgebraicSimplificationPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(machine)),
        ) == Ok(true))
        {
            break;
        }
    }
    (ssa, v(13))
}

/// **A double-width sum kept in its halves is kept where its halves were.**
/// The halves of `a + b` each name the slot they were kept in, and the sum
/// that replaces them names the one its low half, at the lower address, was
/// kept in. Where the high half was kept nowhere - a sign or a carry, not a
/// word of memory - the sum is not taken to be kept at all.
#[test]
fn a_sum_kept_in_halves_is_kept_where_they_were() {
    let (ssa, sum) = sum_kept_in_halves(Some(5), Some(6));
    assert!(
        ssa.blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .any(|instr| matches!(instr.op(), SsaOp::Add { dest, .. } if *dest == sum)),
        "the composition folds to the double-width sum\n{ssa}"
    );
    assert_eq!(origin_of(&ssa, sum), Some(MockOrigin(Some(5))), "{ssa}");

    let (ssa, sum) = sum_kept_in_halves(Some(5), None);
    assert_eq!(origin_of(&ssa, sum), Some(MockOrigin(None)), "{ssa}");
}

/// `a + b + c` in halves, `b` a sign-extended word and `c` a word: `a` given
/// by its halves, kept in slots `7` and `8`, and the first sum's halves kept
/// in slots `5` and `6`.
/// Returns the function after the algebraic pass's fixpoint and the whole
/// sum.
fn chain_kept_in_halves() -> (SsaFunction<FactfulTarget>, SsaVarId) {
    let v = SsaVarId::from_index;
    let types = (0..20).map(|index| match index {
        6 | 11 => MockType::I1,
        14 | 19 => MockType::U8,
        15..=18 => MockType::U64,
        _ => MockType::U32,
    });
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(5, 0);
    for (index, ty) in types.into_iter().enumerate() {
        ssa.create_variable(
            VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
            0,
            DefSite::instruction(0, index),
            ty,
        );
    }
    for (value, origin) in [(v(5), 5), (v(9), 6), (v(0), 7), (v(1), 8)] {
        if let Some(var) = ssa.variable_mut(value) {
            var.set_facts(MockOrigin(Some(origin)));
        }
    }
    let add = |dest: usize, left: usize, right: usize| SsaOp::Add {
        dest: v(dest),
        left: v(left),
        right: v(right),
        flags: None,
    };
    let below = |dest: usize, left: usize, right: usize| SsaOp::Clt {
        dest: v(dest),
        left: v(left),
        right: v(right),
        unsigned: true,
        signaling: false,
    };
    let widen = |dest: usize, operand: usize, target: MockType| SsaOp::IntConv {
        dest: v(dest),
        operand: v(operand),
        target,
        overflow_check: false,
        unsigned: true,
    };
    let mut ops: Vec<SsaOp<FactfulTarget>> = [0usize, 1, 2, 4]
        .into_iter()
        .map(|index| SsaOp::LoadArg {
            dest: v(index),
            arg_index: u16::try_from(index).unwrap_or(0),
        })
        .collect();
    ops.extend([
        SsaOp::Const {
            dest: v(19),
            value: ConstValue::U8(31),
        },
        SsaOp::Shr {
            dest: v(3),
            value: v(2),
            amount: v(19),
            unsigned: false,
            flags: None,
        },
        add(5, 0, 2),
        below(6, 5, 0),
        widen(7, 6, MockType::U32),
        add(8, 1, 3),
        add(9, 8, 7),
        add(10, 5, 4),
        below(11, 10, 5),
        widen(12, 11, MockType::U32),
        add(13, 9, 12),
        SsaOp::Const {
            dest: v(14),
            value: ConstValue::U8(32),
        },
        widen(15, 10, MockType::U64),
        widen(16, 13, MockType::U64),
        SsaOp::Shl {
            dest: v(17),
            value: v(16),
            amount: v(14),
            flags: None,
        },
        SsaOp::Or {
            dest: v(18),
            left: v(17),
            right: v(15),
            flags: None,
        },
        SsaOp::returns(Some(v(18))),
    ]);
    let mut b0 = SsaBlock::new(0);
    for op in ops {
        b0.add_instruction(SsaInstruction::synthetic(op));
    }
    ssa.add_block(b0);
    ssa.recompute_uses();
    let machine = Machine {
        byte_order: Some(Endianness::Little),
        ..MACHINE
    };
    for _ in 0..8 {
        if !(AlgebraicSimplificationPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(machine)),
        ) == Ok(true))
        {
            break;
        }
    }
    (ssa, v(18))
}

/// **Each sum along a chain kept in halves is kept where its halves were.**
/// `a + b + c`: the sum of the first two, its halves kept in slots, is kept
/// in the first's slot, and so is `a`, read from the two slots its halves
/// were kept in. `b`, a word whose high half is its sign filled, and the
/// whole sum, whose halves were kept nowhere, are not.
#[test]
fn each_sum_of_a_chain_kept_in_halves_is_kept_where_they_were() {
    let (ssa, sum) = chain_kept_in_halves();
    let definition = |var: SsaVarId| {
        ssa.blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .find(|instr| instr.op().dest() == Some(var))
            .map(|instr| instr.op().clone())
    };
    let summed = |var: SsaVarId| match definition(var) {
        Some(SsaOp::Add { left, right, .. }) => Some((left, right)),
        _ => None,
    };
    assert!(
        summed(sum).is_some(),
        "the chain folds to a double-width sum\n{ssa}"
    );
    let (partial, _) = some_or_abort(summed(sum));
    assert!(
        summed(partial).is_some(),
        "the first two operands are summed first\n{ssa}"
    );
    let (first, second) = some_or_abort(summed(partial));
    assert_eq!(origin_of(&ssa, partial), Some(MockOrigin(Some(5))), "{ssa}");
    assert_eq!(origin_of(&ssa, first), Some(MockOrigin(Some(7))), "{ssa}");
    assert_eq!(origin_of(&ssa, second), Some(MockOrigin(None)), "{ssa}");
    assert_eq!(origin_of(&ssa, sum), Some(MockOrigin(None)), "{ssa}");
}

/// **One load of two adjacent words is kept where the first was read.** Two
/// word loads composed into a doubleword become one load of it, at the
/// lower address; the loaded value takes the first word load's slot `5`.
#[test]
fn one_load_of_two_words_is_kept_where_the_first_was_read() {
    let v = SsaVarId::from_index;
    let types = [
        MockType::Ptr,
        MockType::U32,
        MockType::Ptr,
        MockType::U32,
        MockType::U8,
        MockType::U64,
        MockType::U64,
        MockType::U64,
        MockType::U64,
    ];
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(1, 0);
    for (index, ty) in types.into_iter().enumerate() {
        ssa.create_variable(
            VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
            0,
            DefSite::instruction(0, index),
            ty,
        );
    }
    for (value, origin) in [(v(1), 5), (v(3), 6)] {
        if let Some(var) = ssa.variable_mut(value) {
            var.set_facts(MockOrigin(Some(origin)));
        }
    }
    let widen = |dest: usize, operand: usize| SsaOp::IntConv {
        dest: v(dest),
        operand: v(operand),
        target: MockType::U64,
        overflow_check: false,
        unsigned: true,
    };
    let mut b0 = SsaBlock::new(0);
    for op in [
        SsaOp::LoadArg {
            dest: v(0),
            arg_index: 0,
        },
        SsaOp::LoadIndirect {
            dest: v(1),
            addr: v(0),
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::PtrAdd {
            dest: v(2),
            base: v(0),
            index: None,
            stride: 1,
            offset: 4,
            result_type: MockType::Ptr,
        },
        SsaOp::LoadIndirect {
            dest: v(3),
            addr: v(2),
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::Const {
            dest: v(4),
            value: ConstValue::U8(32),
        },
        widen(5, 1),
        widen(6, 3),
        SsaOp::Shl {
            dest: v(7),
            value: v(6),
            amount: v(4),
            flags: None,
        },
        SsaOp::Or {
            dest: v(8),
            left: v(7),
            right: v(5),
            flags: None,
        },
        SsaOp::returns(Some(v(8))),
    ] {
        b0.add_instruction(SsaInstruction::synthetic(op));
    }
    ssa.add_block(b0);
    ssa.recompute_uses();
    let machine = Machine {
        byte_order: Some(Endianness::Little),
        ..MACHINE
    };
    let _ = AlgebraicSimplificationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(machine)),
    );
    assert!(
        ssa.blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .any(|instr| matches!(
                instr.op(),
                SsaOp::LoadIndirect { dest, value_type: MockType::U64, .. } if *dest == v(8)
            )),
        "the composition is one load\n{ssa}"
    );
    assert_eq!(origin_of(&ssa, v(8)), Some(MockOrigin(Some(5))), "{ssa}");
}

/// `v0 = 4` at `0x10`, `v1 = v0 + v0` at `0x14`, `v2 = v1 + v1` at `0x18`,
/// `v3 = v0` at `0x1c`, and where `shared`, `v4 = v1` at `0x20`; `return v2`:
/// each instruction knowing the address it was lifted from.
fn sourced_chain(shared: bool) -> (SsaFunction<FactfulTarget>, [SsaVarId; 5]) {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(0, 5);
    let vars: [SsaVarId; 5] = std::array::from_fn(|index| {
        ssa.create_variable(
            VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
            0,
            DefSite::instruction(0, index),
            MockType::I32,
        )
    });
    let [v0, v1, v2, v3, v4] = vars;
    let mut block = SsaBlock::new(0);
    let mut ops = vec![
        (
            0x10,
            SsaOp::Const {
                dest: v0,
                value: ConstValue::I32(4),
            },
        ),
        (
            0x14,
            SsaOp::Add {
                dest: v1,
                left: v0,
                right: v0,
                flags: None,
            },
        ),
        (
            0x18,
            SsaOp::Add {
                dest: v2,
                left: v1,
                right: v1,
                flags: None,
            },
        ),
        (0x1c, SsaOp::Copy { dest: v3, src: v0 }),
    ];
    if shared {
        ops.push((0x20, SsaOp::Copy { dest: v4, src: v1 }));
    }
    for (at, op) in ops {
        let mut instruction = SsaInstruction::synthetic(op);
        instruction.set_facts(MockSource::at(at));
        block.add_instruction(instruction);
    }
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v2))));
    ssa.add_block(block);
    ssa.recompute_uses();
    (ssa, vars)
}

/// The addresses the instruction lifted from `at` absorbed.
fn absorbed_at(ssa: &SsaFunction<FactfulTarget>, at: u64) -> Vec<u64> {
    let facts = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .map(|instruction| instruction.facts())
        .find(|facts| facts.at == at);
    some_or_abort(facts).absorbed.iter().copied().collect()
}

/// **An instruction rewritten to stop reading an operand holds the
/// operand's computation, where nothing else reads it.** `v2 = v1 + v1`
/// folded to the constant `16` no longer reads `v1`, which nothing else
/// reads, so it takes in where `v1` came from (`0x14`), through the function
/// and through the editor alike; not `v0`'s, which `v3` still reads.
/// Rewritten to `v1 - v1`, it still reads `v1`, and where `v4 = v1` reads it
/// too, `v1`'s computation stays where it was: neither absorbs anything.
#[test]
fn an_instruction_that_stops_reading_an_operand_absorbs_its_computation() {
    let folded = |v2| SsaOp::Const {
        dest: v2,
        value: ConstValue::I32(16),
    };
    let (mut ssa, [_, _, v2, _, _]) = sourced_chain(false);
    assert!(ssa.replace_instruction_op(0, 2, folded(v2)));
    assert_eq!(absorbed_at(&ssa, 0x18), [0x14], "the function's rewrite");
    let (mut ssa, [_, _, v2, _, _]) = sourced_chain(false);
    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        editor.replace_instruction_op(0, 2, folded(v2))
    }));
    assert_eq!(absorbed_at(&ssa, 0x18), [0x14], "the editor's rewrite");
    let (mut ssa, [_, v1, v2, _, _]) = sourced_chain(false);
    assert!(ssa.replace_instruction_op(
        0,
        2,
        SsaOp::Sub {
            dest: v2,
            left: v1,
            right: v1,
            flags: None,
        },
    ));
    assert_eq!(
        absorbed_at(&ssa, 0x18),
        Vec::<u64>::new(),
        "still reading v1"
    );
    let (mut ssa, [_, _, v2, _, _]) = sourced_chain(true);
    assert!(ssa.replace_instruction_op(0, 2, folded(v2)));
    assert_eq!(
        absorbed_at(&ssa, 0x18),
        Vec::<u64>::new(),
        "v4 still reads v1"
    );
}

/// **A reader an edit session already rewrote away is no reader.** In one
/// session, `v4 = v1` (`0x20`) is rewritten to the constant `4` first, which
/// leaves `v1`'s use list naming it until the session ends; `v2 = v1 + v1`
/// folded to `16` next is then `v1`'s only reader, and takes in `v1`'s row
/// (`0x14`). The control: with `v4` still reading `v1`, it does not.
#[test]
fn a_reader_an_edit_already_rewrote_away_is_no_reader() {
    let constant = |dest, value| SsaOp::Const {
        dest,
        value: ConstValue::I32(value),
    };
    let (mut ssa, [_, _, v2, _, v4]) = sourced_chain(true);
    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        editor.replace_instruction_op(0, 4, constant(v4, 4))?;
        editor.replace_instruction_op(0, 2, constant(v2, 16))
    }));
    assert_eq!(absorbed_at(&ssa, 0x18), [0x14], "v4 no longer reads v1");
    let (mut ssa, [_, _, v2, _, _]) = sourced_chain(true);
    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        editor.replace_instruction_op(0, 2, constant(v2, 16))
    }));
    assert_eq!(
        absorbed_at(&ssa, 0x18),
        Vec::<u64>::new(),
        "control: v4 still reads v1"
    );
}

/// **A reader of a value replaced by a constant, or by a value computed
/// apart from it, holds the replaced value's computation.** Every use of
/// `v1` replaced by `v0`: `v2 = v1 + v1` (`0x18`) takes in where `v1` came
/// from (`0x14`), through the function's replacement, its resolved batch and
/// the editor's checked one, but not `v0`'s own (`0x10`), which is the
/// replacement and stays. A computation reaching through operands nothing
/// else reads goes whole: `v2` replaced by `v3`, the copy of `v0` computed
/// apart, carries `v2`'s and `v1`'s rows into the return. Replaced by `v4`,
/// which is computed from `v1`, nothing is carried.
#[test]
fn a_reader_of_a_replaced_value_absorbs_its_computation() {
    let (mut ssa, [v0, v1, _, _, _]) = sourced_chain(false);
    let _ = ssa.replace_uses(v1, v0);
    assert_eq!(absorbed_at(&ssa, 0x18), [0x14], "replace_uses");
    let (mut ssa, [v0, v1, _, _, _]) = sourced_chain(false);
    let _ = ssa.replace_uses_resolved(&[(v1, v0)]);
    assert_eq!(absorbed_at(&ssa, 0x18), [0x14], "replace_uses_resolved");
    let (mut ssa, [v0, v1, _, _, _]) = sourced_chain(false);
    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        let _ = editor.replace_uses_checked(v1, v0);
        Ok(())
    }));
    assert_eq!(
        absorbed_at(&ssa, 0x18),
        [0x14],
        "the editor's checked replacement"
    );
    let (mut ssa, [_, _, v2, v3, _]) = sourced_chain(false);
    let _ = ssa.replace_uses(v2, v3);
    let ret = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .find(|instruction| matches!(instruction.op(), SsaOp::Return { .. }))
        .map(|instruction| {
            instruction
                .facts()
                .absorbed
                .iter()
                .copied()
                .collect::<Vec<u64>>()
        });
    assert_eq!(ret, Some(vec![0x14, 0x18]), "the whole computation of v2");
    let (mut ssa, [_, v1, _, _, v4]) = sourced_chain(true);
    let _ = ssa.replace_uses(v1, v4);
    assert_eq!(absorbed_at(&ssa, 0x18), Vec::<u64>::new(), "v4 reads v1");
}

/// **A narrowed phi is held where the wide phi was.** A phi of a 32-bit
/// value's sign extension and a whole 64-bit value, read only at its low 32
/// bits, narrows to a phi of the two at 32 bits (the wide one truncated), and
/// the wide phi's readers come to read the narrow one, so what the host knew of
/// the wide phi - the storage it is held in - is the narrow phi's; and the
/// extension, read by the phi alone, dies with it, so its storage is the
/// extended value's.
#[test]
fn a_narrowed_phi_keeps_what_the_host_knew_of_the_wide_one() {
    let wide_spec = || SsaDefSpec::<FactfulTarget>::tmp(MockType::I64);
    let narrow_spec = || SsaDefSpec::<FactfulTarget>::tmp(MockType::I32);
    let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(3, 0);
    result_or_abort(builder.in_block(0, |block| {
        let flag = block.load_arg(narrow_spec(), 1)?;
        block.branch(flag, 1, 2)
    }));
    let mut extension = None;
    let mut extended = None;
    result_or_abort(builder.in_block(1, |block| {
        let x = block.load_arg(narrow_spec(), 0)?;
        extended = Some(x);
        extension = Some(block.conv(wide_spec(), x, MockType::I64, false, false)?);
        block.jump(3)
    }));
    let mut whole = None;
    result_or_abort(builder.in_block(2, |block| {
        whole = Some(block.load_arg(wide_spec(), 2)?);
        block.jump(3)
    }));
    let mut joined = None;
    result_or_abort(builder.in_block(3, |block| {
        let phi = block.phi(
            wide_spec(),
            [(1, some_or_abort(extension)), (2, some_or_abort(whole))],
        )?;
        joined = Some(phi);
        let low = block.conv(narrow_spec(), phi, MockType::I32, false, true)?;
        block.ret(Some(low))
    }));
    let mut ssa = result_or_abort(builder.finish());
    if let Some(variable) = ssa.variable_mut(some_or_abort(joined)) {
        variable.set_facts(MockOrigin(Some(9)));
    }
    if let Some(variable) = ssa.variable_mut(some_or_abort(extension)) {
        variable.set_facts(MockOrigin(Some(4)));
    }
    let _ = AlgebraicSimplificationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    );
    let narrowed = ssa
        .block(3)
        .map(|block| block.phi_nodes())
        .unwrap_or_default()
        .iter()
        .map(PhiNode::result)
        .find(|phi| ssa.bit_width(*phi) == Some(32));
    assert!(narrowed.is_some(), "the phi narrows\n{ssa}");
    let narrowed = some_or_abort(narrowed);
    assert_eq!(
        origin_of(&ssa, narrowed),
        Some(MockOrigin(Some(9))),
        "{ssa}"
    );
    assert_eq!(
        origin_of(&ssa, some_or_abort(extended)),
        Some(MockOrigin(Some(4))),
        "{ssa}"
    );
}

/// **A value a branch decides equal to another only past its edge takes
/// nothing of the other's identity.** `kept` is defined first and `later`,
/// which knows its storage (7) and its rename group, after it; the branch
/// takes block 1 only where they are equal, so there `later` reads as `kept`.
/// `kept` is still read at the branch itself, where `later` may hold anything
/// else, so naming `kept` after `later`'s storage names it so where that
/// storage holds another value: ARM dash's `str r2, [sp, #0x14]; bl f; ldr
/// r3, [sp, #0x14]; cmp r3, r0` printed the call's result as the slot it was
/// compared against. Both shapes hold: a use past the edge that the pass
/// rewrites, and none, where the rewrite is still the pass's whole answer.
#[test]
fn a_value_equal_past_an_edge_takes_nothing_of_the_other() {
    let spec = || SsaDefSpec::<FactfulTarget>::tmp(MockType::I32);
    for read_past in [true, false] {
        let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(3, 0);
        let (kept, later) = result_or_abort(builder.in_block(0, |block| {
            let kept = block.load_arg(spec(), 0)?;
            let later = block.load_arg(spec(), 1)?;
            block.branch_cmp(kept, later, CmpKind::Eq, false, 1, 2)?;
            Ok((kept, later))
        }));
        result_or_abort(builder.in_block(1, |block| {
            if read_past {
                let doubled = block.add(spec(), later, later)?;
                block.ret(Some(doubled))
            } else {
                block.ret(None)
            }
        }));
        result_or_abort(builder.in_block(2, |block| block.ret(Some(later))));
        let mut ssa = result_or_abort(builder.finish());
        if let Some(variable) = ssa.variable_mut(later) {
            variable.set_facts(MockOrigin(Some(7)));
        }
        ssa.set_rename_group(later, 5);
        let group = ssa.rename_group(kept);
        let rewritten = EqualityPropagationPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true);
        assert_eq!(rewritten, read_past, "{ssa}");
        assert_eq!(origin_of(&ssa, kept), Some(MockOrigin(None)), "{ssa}");
        assert_eq!(ssa.rename_group(kept), group, "{ssa}");
        assert_eq!(origin_of(&ssa, later), Some(MockOrigin(Some(7))), "{ssa}");
    }
}
