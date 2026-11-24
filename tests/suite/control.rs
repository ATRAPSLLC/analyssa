//! Control-flow and block utility SSA patterns.

use std::process;

use analyssa::{
    BitSet, Error,
    analysis::{
        SsaCfg, SsaVerifier, VerifyLevel, WorkBudget, context::AnalysisContext,
        loopanalyzer::SsaLoopAnalysis, verifier::VerifierError,
    },
    events::{EventLog, NullListener},
    graph::NodeId,
    ir::{
        CallData, Callee,
        block::SsaBlock,
        exception::{
            BlockRange, ClausePart, ExceptionTableError, HandlerKind, SsaExceptionHandler,
        },
        function::{
            FunctionKind, SsaConstructOptions, SsaEditOptions, SsaEditor, SsaFunction,
            SsaRollbackPolicy, ValueUpdate,
        },
        instruction::SsaInstruction,
        ops::{CmpKind, SsaOp},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::{
        BlockMergingPass, ControlFlowSimplificationPass, CopyPropagationPass,
        DeadCodeEliminationPass, GlobalValueNumberingPass, JumpThreadingPass, LicmPass,
        LoopCanonicalizationPass, OpaquePredicatePass,
    },
    scheduling::{FunctionPass, PassEnv},
    target::Target,
    testing::{self, MACHINE, MockTarget, MockType},
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

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn ok_or_abort<T, E>(value: Result<T, E>) -> T {
    value.unwrap_or_else(|_| process::abort())
}

#[test]
fn switch_and_branchcmp_successors_are_indexed_by_cfg() {
    let mut ssa = SsaFunction::new(0, 3);
    let selector = local(&mut ssa, 0, 0, 0);
    let left = local(&mut ssa, 1, 1, 0);
    let right = local(&mut ssa, 2, 1, 1);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: selector,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Switch {
        value: selector,
        targets: vec![1, 2, 3],
        default: 4,
    }));
    ssa.add_block(entry);

    let mut compare = SsaBlock::new(1);
    compare.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(7),
    }));
    compare.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(9),
    }));
    compare.add_instruction(instr(SsaOp::BranchCmp {
        left,
        right,
        cmp: CmpKind::Lt,
        unsigned: false,
        true_target: 5,
        false_target: 6,
    }));
    ssa.add_block(compare);

    for block_idx in 2..=6 {
        let mut block = SsaBlock::new(block_idx);
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(cfg.block_successors(0), &[1, 2, 3, 4]);
    assert_eq!(cfg.block_successors(1), &[5, 6]);
    assert_eq!(cfg.block_predecessors(5), &[1]);
    assert_eq!(cfg.exits().len(), 5);

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "verifier errors: {errors:?}");
}

/// A block ending in `Return` inside a protected region is still a function
/// exit.
///
/// The view used to give it a synthetic successor - the region's handler -
/// which stopped it being a sink. Every consumer that asks "does control leave
/// the function here" then got the wrong answer, post-dominance included: the
/// virtual exit is fed by sinks, so a return that is not a sink post-dominates
/// nothing.
#[test]
fn a_return_inside_a_protected_region_still_exits_the_function() {
    let mut ssa = SsaFunction::new(0, 1);
    let value = local(&mut ssa, 0, 0, 0);

    let mut try_entry = SsaBlock::new(0);
    try_entry.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(1),
    }));
    try_entry.add_instruction(instr(SsaOp::returns(Some(value))));
    ssa.add_block(try_entry);

    let mut handler = SsaBlock::new(1);
    handler.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(handler);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 1,
        handler_offset: 1,
        handler_length: 1,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 2),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert!(
        cfg.terminator_successors(0).is_empty(),
        "a return leaves the function, protected region or not"
    );
    assert!(
        cfg.exits().contains(&NodeId::new(0)),
        "so the block is an exit"
    );
    assert_eq!(
        cfg.block_successors(0),
        &[1],
        "and a throw before the return still reaches the handler"
    );
    assert!(
        cfg.terminator_predecessors(1).is_empty(),
        "the runtime enters a handler; no terminator does"
    );
    assert_eq!(cfg.exits().len(), 2, "both returns are exits");
    // The region survives for a consumer that needs it - it is simply not an
    // edge.
    assert_eq!(ssa.exception_handlers().len(), 1);
}

#[test]
fn block_utilities_detect_trampolines_and_reorder_local_dependencies() {
    let mut trampoline = SsaBlock::<MockTarget>::new(0);
    trampoline.add_instruction(instr(SsaOp::Jump { target: 9 }));
    assert_eq!(trampoline.is_trampoline(), Some(9));

    let v0 = SsaVarId::from_index(0);
    let v1 = SsaVarId::from_index(1);
    let v2 = SsaVarId::from_index(2);
    let mut block = SsaBlock::<MockTarget>::new(1);
    block.add_instruction(instr(SsaOp::Add {
        dest: v2,
        left: v1,
        right: v0,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v1,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(v2))));

    assert!(block.sort_instructions_topologically());
    assert!(
        matches!(some_or_abort(block.instruction(0)).op(), SsaOp::Const { dest, .. } if *dest == v0)
    );
    assert!(
        matches!(some_or_abort(block.instruction(1)).op(), SsaOp::Const { dest, .. } if *dest == v1)
    );
    assert!(
        matches!(some_or_abort(block.instruction(2)).op(), SsaOp::Add { dest, .. } if *dest == v2)
    );
    assert!(some_or_abort(block.instruction(3)).is_terminator());
}

#[test]
fn interrupt_handler_function_kind_defaults_to_normal() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    assert_eq!(ssa.kind(), FunctionKind::Normal);
    assert!(ssa.kind().is_normal());
    assert!(!ssa.kind().is_interrupt_handler());
}

#[test]
fn interrupt_handler_function_kind_can_be_set() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);
    assert_eq!(ssa.kind(), FunctionKind::InterruptHandler);
    assert!(ssa.kind().is_interrupt_handler());
    assert!(!ssa.kind().is_normal());
}

#[test]
fn interrupt_handler_with_interrupt_return_terminator() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert_eq!(ssa.kind(), FunctionKind::InterruptHandler);
    assert!(ssa.has_interrupt_return());
    assert!(
        some_or_abort(ssa.block(0))
            .control_terminator()
            .is_some_and(|op| { matches!(op, SsaOp::InterruptReturn) })
    );
}

#[test]
fn interrupt_return_is_terminal_no_successors() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert!(cfg.block_successors(0).is_empty());
}

#[test]
fn has_interrupt_return_returns_false_when_no_interrupt_return() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    assert!(!ssa.has_interrupt_return());
}

#[test]
fn has_interrupt_return_returns_true_when_interrupt_return_present() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();
    assert!(ssa.has_interrupt_return());
}

#[test]
fn interrupt_return_survives_canonicalization() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::InterruptReturn));
    ssa.add_block(block);
    ssa.recompute_uses();

    ssa.canonicalize();
    assert_eq!(ssa.block_count(), 1);
    assert!(ssa.has_interrupt_return());
    assert_eq!(ssa.kind(), FunctionKind::InterruptHandler);
}

#[test]
fn normal_function_without_interrupt_return_remains_normal_after_canonicalize() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa.canonicalize();

    assert_eq!(ssa.kind(), FunctionKind::Normal);
    assert!(!ssa.has_interrupt_return());
}

// Handler-kind classification and clause layout, through a host `Target`

/// `MockTarget` reads its `u32` flags as `0 = Catch, 1 = Filter, 2 = Finally,
/// 3 = Fault`. That convention is what gives a host's opaque exception kind a
/// meaning the crate can act on, and `Target::handler_kind` is the only place
/// it is read.
#[test]
fn a_host_target_classifies_its_own_exception_flags() {
    assert_eq!(MockTarget::handler_kind(&0), HandlerKind::Catch);
    assert_eq!(MockTarget::handler_kind(&1), HandlerKind::Filter);
    assert_eq!(MockTarget::handler_kind(&2), HandlerKind::Finally);
    assert_eq!(MockTarget::handler_kind(&3), HandlerKind::Fault);
}

#[test]
fn a_filter_offset_is_readable_only_for_a_filter_clause() {
    let mut handler = SsaExceptionHandler::<MockTarget> {
        flags: 0, // Catch
        try_offset: 0,
        try_length: 10,
        handler_offset: 10,
        handler_length: 20,
        class_token_or_filter: 42,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 2),
        filter_range: None,
        selector: None,
    };

    assert_eq!(handler.kind(), HandlerKind::Catch);
    assert_eq!(
        handler.filter_offset(),
        None,
        "for a catch, `class_token_or_filter` is a type token"
    );
    assert_eq!(handler.class_token_or_filter, 42);

    handler.flags = 1;
    assert_eq!(handler.kind(), HandlerKind::Filter);
    assert_eq!(
        handler.filter_offset(),
        Some(42),
        "for a filter, the same field is an IL offset"
    );
}

/// One clause over the four kinds, remapped by its members.
///
/// The three parts are laid out disjointly -- protected `[0, 2)`, handler
/// `[3, 5)`, filter `[5, 6)` -- because a filter block inside its own handler's
/// range is not a clause any table can express, and the remap has no way to
/// keep the two apart if they were.
#[test]
fn a_clause_remaps_every_part_by_its_surviving_blocks() {
    for flags in [0u32, 1, 2, 3] {
        let mut handler = SsaExceptionHandler::<MockTarget> {
            flags,
            try_offset: 0,
            try_length: 10,
            handler_offset: 10,
            handler_length: 20,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 2),
            handler_range: BlockRange::new(3, 5),
            filter_range: BlockRange::new(5, 6),
            selector: None,
        };

        assert!(handler.has_block_mapping());
        // Block 1 and block 2 go; 0 -> 0, 3 -> 1, 4 -> 2, 5 -> 3.
        handler.remap_block_indices(&[Some(0), None, None, Some(1), Some(2), Some(3)]);

        assert_eq!(handler.protected_range, BlockRange::new(0, 1));
        assert_eq!(
            handler.handler_range,
            BlockRange::new(1, 3),
            "the handler keeps its own two blocks and stops there"
        );
        assert_eq!(
            handler.filter_range,
            BlockRange::new(3, 4),
            "and the filter, which follows it, is untouched by that end"
        );
        assert!(handler.has_block_mapping());
    }
}

/// The total view and the checked view of one clause disagree on purpose.
///
/// `parts()` and `entry_blocks()` answer for a clause `layout()` refuses, because
/// the transformations that read them have to fence a malformed clause too.
#[test]
fn a_malformed_clause_still_reports_its_parts() {
    let malformed = SsaExceptionHandler::<MockTarget> {
        flags: 0, // a catch...
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 2),
        filter_range: BlockRange::new(2, 3), // ...carrying a filter range
        selector: None,
    };

    assert_eq!(
        malformed.layout(3),
        Err(ExceptionTableError::RangeWithoutFilterKind {
            kind: HandlerKind::Catch
        })
    );
    assert_eq!(
        malformed
            .parts()
            .map(|(part, _)| part)
            .collect::<Vec<ClausePart>>(),
        vec![
            ClausePart::Protected,
            ClausePart::Handler,
            ClausePart::Filter
        ]
    );
    assert_eq!(
        malformed.entry_blocks().collect::<Vec<usize>>(),
        vec![1, 2],
        "the handler entry, then the filter entry"
    );
}

#[test]
fn isr_debug_shows_function_kind() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    ssa.set_kind(FunctionKind::InterruptHandler);
    let debug = format!("{ssa:?}");
    assert!(debug.contains("kind: InterruptHandler"));

    let normal = SsaFunction::<MockTarget>::new(0, 0);
    let normal_debug = format!("{normal:?}");
    assert!(normal_debug.contains("kind: Normal"));
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

/// Builds `B0: local0 = 7; B1 (try): <body>; B2: return; B3 (handler): return local0`.
///
/// Both definitions carry rename group 0, which is how a front end says "these
/// are versions of one local" - the rebuilder derives groups from phi
/// connectivity, so two independent definitions sharing only an origin would be
/// unrelated values with nothing to merge.
///
/// `redefine_in_region` controls whether the protected block reassigns the
/// local, which is the only difference between the two cases below.
fn try_catch_reading_a_local(redefine_in_region: bool) -> SsaFunction<MockTarget> {
    const LOCAL_GROUP: u32 = 0;

    let mut ssa = SsaFunction::<MockTarget>::new(0, 1);

    let before = local(&mut ssa, 0, 0, 0);
    ssa.set_rename_group(before, LOCAL_GROUP);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: before,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    if redefine_in_region {
        let inside = local(&mut ssa, 0, 1, 0);
        ssa.set_rename_group(inside, LOCAL_GROUP);
        b1.add_instruction(instr(SsaOp::Const {
            dest: inside,
            value: ConstValue::I32(9),
        }));
    }
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);

    // The handler reads the local.
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(Some(before))));
    ssa.add_block(b3);

    ssa.set_exception_handlers(vec![clause(1, 2, 3, 4)]);
    ssa.recompute_uses();
    ssa
}

/// A loop whose header merges one constant from both ways in, all of it
/// protected by a handler that reads the local.
///
/// `local = null; loop { value = local; if arg { local = null } else break }`
/// with the handler returning `local`. The header's phi joins `null` with
/// `null`, and the handler observes the local at every point of the loop,
/// the header's entry included - so the local stays live into the header
/// whatever the header itself does.
fn constant_loop_phi_under_a_handler() -> SsaFunction<MockTarget> {
    // Group 0 is the argument's.
    const LOCAL_GROUP: u32 = 1;

    let mut ssa = SsaFunction::<MockTarget>::new(1, 3);
    let entry_value = local(&mut ssa, 0, 0, 0);
    let current = ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::phi(1), MockType::I32);
    let latch_value = local(&mut ssa, 0, 2, 0);
    for version in [entry_value, current, latch_value] {
        ssa.set_rename_group(version, LOCAL_GROUP);
    }
    let read = local(&mut ssa, 1, 1, 0);
    let condition = local(&mut ssa, 2, 1, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: entry_value,
        value: ConstValue::Null,
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(current, VariableOrigin::Local(0));
    phi.add_operand(PhiOperand::new(entry_value, 0));
    phi.add_operand(PhiOperand::new(latch_value, 2));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::Copy {
        dest: read,
        src: current,
    }));
    b1.add_instruction(instr(SsaOp::LoadArg {
        dest: condition,
        arg_index: 0,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: latch_value,
        value: ConstValue::Null,
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(Some(read))));
    ssa.add_block(b3);

    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::returns(Some(current))));
    ssa.add_block(b4);

    ssa.set_exception_handlers(vec![clause(1, 3, 4, 5)]);
    ssa.recompute_uses();
    ssa
}

/// A constant phi a handler observes is folded once, not once per rebuild.
///
/// The fold used to redefine the phi's result with the constant at the top of
/// the header. The handler keeps the local live into the header, so the next
/// rebuild placed the same phi again, the pass folded it again, and every
/// round left one more constant the handler could observe - so none was ever
/// dead. A NetReactor method gained a thousand instructions that way and the
/// pipeline never converged. Redirecting only the phi's readers settles in
/// one round.
#[test]
fn a_constant_phi_a_handler_observes_folds_once() {
    let mut ssa = constant_loop_phi_under_a_handler();
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    let header_phis = some_or_abort(ssa.block(1)).phi_nodes().len();
    assert!(header_phis > 0, "the header merges the local:\n{ssa}");

    let instructions = |ssa: &SsaFunction<MockTarget>| -> usize {
        ssa.blocks()
            .iter()
            .map(|block| block.instructions().len())
            .sum()
    };
    let _ = OpaquePredicatePass::new().run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    );
    let settled = instructions(&ssa);

    for round in 0..3 {
        let changed = OpaquePredicatePass::new().run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true);
        assert_eq!(instructions(&ssa), settled, "round {round} grew:\n{ssa}");
        assert!(!changed, "round {round} folded the phi again:\n{ssa}");
    }

    // And the fold did its job: the header's reader reads the constant.
    let read_source = some_or_abort(ssa.block(1).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Copy { src, .. } => Some(*src),
                _ => None,
            })
    }));
    assert!(
        matches!(
            ssa.get_definition(read_source),
            Some(SsaOp::Const {
                value: ConstValue::Null,
                ..
            })
        ),
        "the header reads the constant:\n{ssa}"
    );
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
}

/// The value the handler block returns after a rebuild.
fn handler_return_value(ssa: &SsaFunction<MockTarget>) -> Option<SsaVarId> {
    ssa.block(3).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            })
    })
}

/// A definition that completed before the protected region began is visible in
/// the handler, and must survive the rebuild.
///
/// Nothing the region did can have undone it: the region had not started.
#[test]
fn a_local_defined_before_a_protected_region_survives_into_the_handler() {
    let mut ssa = try_catch_reading_a_local(false);
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    // Both sides read *after* the rebuild: it renumbers every variable, so an
    // id saved beforehand would compare against a different value.
    let entry_definition = ssa
        .block(0)
        .and_then(|block| block.instructions().first())
        .and_then(|instr| match instr.op() {
            SsaOp::Const { dest, .. } => Some(*dest),
            _ => None,
        });

    assert_eq!(
        handler_return_value(&ssa),
        entry_definition,
        "the handler must read the definition that provably executed before the region"
    );
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "and the result must verify: {errors:?}");
}

/// A group the protected region reassigns has no single reaching value at the
/// handler, so the handler reads a merge over every value the region may have
/// produced before the throw.
#[test]
fn a_local_redefined_inside_a_protected_region_reads_a_merge_of_the_region() {
    let mut ssa = try_catch_reading_a_local(true);
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let before_region = ssa
        .block(0)
        .and_then(|block| block.instructions().first())
        .and_then(|instr| match instr.op() {
            SsaOp::Const { dest, .. } => Some(*dest),
            _ => None,
        });
    let inside_region = ssa
        .block(1)
        .and_then(|block| block.instructions().first())
        .and_then(|instr| match instr.op() {
            SsaOp::Const { dest, .. } => Some(*dest),
            _ => None,
        });

    let read = some_or_abort(handler_return_value(&ssa));
    assert_ne!(
        Some(read),
        before_region,
        "the region may have reassigned the local, so the pre-region value is not \
         guaranteed to reach the handler"
    );

    // What the handler reads is a phi at its entry whose operands are the
    // values a throw could have left behind: the one the region started with,
    // and the one it assigned.
    let phi = some_or_abort(
        ssa.block(3)
            .and_then(|block| block.phi_nodes().iter().find(|phi| phi.result() == read)),
    );
    let mut carried: Vec<SsaVarId> = phi.exceptional_operands().map(PhiOperand::value).collect();
    carried.sort_unstable();
    let mut expected: Vec<SsaVarId> = [before_region, inside_region]
        .into_iter()
        .flatten()
        .collect();
    expected.sort_unstable();
    assert_eq!(carried, expected, "{ssa}");
    assert!(
        phi.exceptional_operands()
            .all(|operand| operand.predecessor() == 1),
        "each names the protected block the throw would leave"
    );
    assert!(
        phi.normal_operands().next().is_none(),
        "no terminator transfers to a handler entry"
    );
}

/// A region that writes nothing needs no merge at its handler.
///
/// The criterion is what the region assigns, not that the block is a handler.
#[test]
fn no_phi_is_placed_where_the_region_writes_nothing() {
    let mut ssa = try_catch_reading_a_local(false);
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    assert!(
        ssa.block(3)
            .is_some_and(|block| block.phi_nodes().is_empty()),
        "nothing in the region can have changed the local: {ssa}"
    );
}

/// A phi at a handler entry must account for every block that can throw to it.
///
/// A phi placed there by hand, with no operands, is a merge that is missing the
/// values the region may have produced - which every analysis that meets the
/// operands would then read as narrower than the program allows.
#[test]
fn a_handler_phi_that_accounts_for_no_throw_is_reported() {
    let mut ssa = try_catch_reading_a_local(true);

    // Place a phi in the handler entry by hand, as a pass mid-pipeline might.
    let merged = local(&mut ssa, 0, 3, 0);
    ssa.set_rename_group(merged, 0);
    if let Some(handler) = ssa.block_mut(3) {
        handler
            .phi_nodes_mut()
            .push(PhiNode::new(merged, VariableOrigin::Local(0)));
    }
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.iter().any(|error| matches!(
            error,
            VerifierError::MissingExceptionalOperand {
                block: 3,
                source: 1,
                ..
            }
        )),
        "B1 can throw here and the phi says nothing about it: {errors:?}"
    );

    // Reconstruction fills it in rather than removing the phi: the merge is
    // real, it was merely unstated.
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    let after = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        !after.iter().any(|error| matches!(
            error,
            VerifierError::MissingExceptionalOperand { .. }
                | VerifierError::PhiWithoutPredecessors { .. }
        )),
        "and the result is well-formed: {after:?}\n{ssa}"
    );
}

/// A handler entry that is *also* a branch target keeps its phis: the rule is
/// about having no terminator predecessors, not about being a handler.
#[test]
fn a_handler_entry_with_real_terminator_predecessors_may_carry_a_phi() {
    let mut ssa = try_catch_reading_a_local(false);

    // Make B2 jump to the handler, giving it a real terminator predecessor.
    if let Some(block) = ssa.block_mut(2) {
        block.instructions_mut().clear();
        block.add_instruction(instr(SsaOp::Jump { target: 3 }));
    }
    ssa.recompute_uses();

    assert_eq!(ssa.demote_runtime_entry_phis(), 0, "nothing to demote");

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(
        cfg.terminator_predecessors(3),
        &[2],
        "B3 now has a terminator predecessor, so a phi there merges over edges too"
    );
    assert_eq!(
        cfg.block_predecessors(3),
        &[1, 2],
        "and control still reaches it by throwing in the protected block"
    );
}

/// A loop wholly inside an exception handler is an ordinary loop.
///
/// Back-edge detection asks whether the target dominates the source, and
/// `DominatorTree::dominates` answers `false` when either endpoint is
/// unreachable. Under a terminator-only graph both endpoints of a handler loop
/// are unreachable, so no `LoopInfo` is created at all and every loop pass
/// silently declines to act.
#[test]
fn a_loop_inside_a_handler_is_detected() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
    let condition = local(&mut ssa, 0, 2, 0);

    // B0 -> B1 (protected) -> B4 exit.  Handler B2 <-> B3 is the loop.
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Jump { target: 4 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: condition,
        value: ConstValue::I32(1),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 2,
        false_target: 4,
    }));
    ssa.add_block(b3);

    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b4);

    ssa.set_exception_handlers(vec![clause(1, 2, 2, 4)]);
    ssa.recompute_uses();

    let forest = ssa.analyze_loops();
    assert!(
        forest.loops().iter().any(|info| info.header.index() == 2),
        "the handler's back edge B3 -> B2 is a loop like any other"
    );
}

/// The loop-canonicalization pass leaves a loop whose header the runtime
/// dispatches to alone.
///
/// A preheader works by redirecting the terminators that name the header. No
/// terminator names this one, so a preheader would sit between nothing and the
/// header while the runtime kept entering it directly.
#[test]
fn loop_passes_leave_a_root_headed_loop_alone() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
    let condition = local(&mut ssa, 0, 2, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    // B2 is both the handler entry and the loop header, entered from B2 itself.
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: condition,
        value: ConstValue::I32(1),
    }));
    b2.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b3);

    ssa.set_exception_handlers(vec![clause(1, 2, 2, 3)]);
    ssa.recompute_uses();

    let before = ssa.block_count();
    let events = EventLog::<MockTarget>::new();
    assert!(
        LoopCanonicalizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &events, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    assert_eq!(
        ssa.block_count(),
        before,
        "no preheader may be spliced in front of a block no terminator names"
    );
}

/// Scoped reconstruction states an operand for a throw as an operand for a
/// throw.
///
/// A phi at a handler entry merges over the program points a throw can leave
/// from, not over an edge a terminator names. Filling one with a normal operand
/// claims the guard ran to completion, which is the one thing a throw says it
/// did not, and the verifier reports it as an operand no predecessor supplies.
#[test]
fn a_reconstructed_handler_phi_carries_an_exceptional_operand() {
    // B0 entry, B1 and B2 the two protected arms, B3 their join, B4 the
    // handler, which reads the variable. Only B1 assigns it, so the value
    // reaching the handler differs by arm and the reconstruction has to place a
    // phi there.
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
    let initial = local(&mut ssa, 0, 0, 0);
    let condition = local(&mut ssa, 1, 0, 1);
    let assigned = ssa.create_variable(
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(1, 0),
        MockType::I32,
    );
    for var in [initial, assigned] {
        ssa.set_rename_group(var, 0);
    }

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: initial,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: condition,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    let mut first = SsaBlock::new(1);
    first.add_instruction(instr(SsaOp::Const {
        dest: assigned,
        value: ConstValue::I32(2),
    }));
    first.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(first);

    let mut second = SsaBlock::new(2);
    second.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(second);

    let mut join = SsaBlock::new(3);
    join.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(join);

    let mut handler = SsaBlock::new(4);
    handler.add_instruction(instr(SsaOp::returns(Some(initial))));
    ssa.add_block(handler);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(1, 4),
        handler_range: BlockRange::new(4, 5),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    // Control can only enter a handler by a throw, so every operand of a phi
    // there names one. A normal operand would name a terminator edge that does
    // not exist.
    let handler_phis = some_or_abort(ssa.block(4)).phi_nodes().to_vec();
    assert!(
        !handler_phis.is_empty(),
        "the handler reads a value the region redefines: {ssa}"
    );
    for phi in &handler_phis {
        assert_eq!(
            phi.normal_operands().count(),
            0,
            "no terminator transfers into a handler: {ssa}"
        );
        assert!(
            phi.exceptional_operands().count() > 0,
            "the merge is over the throws that reach here: {ssa}"
        );
    }

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
}

/// Scoped reconstruction refuses a throw it cannot state exactly.
///
/// A throw leaves its block from wherever it happened, so what reaches the
/// handler is every value the block held - not the one it ends with, which is
/// all the scoped walk computes. When the block redefines the variable the two
/// differ, and stating the end value alone would give the handler a merge
/// narrower than the program allows. The reconstruction is refused instead, and
/// the caller's edit session rolls back intact.
#[test]
fn a_reconstruction_refuses_a_throw_out_of_a_block_that_redefines_the_variable() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
    let initial = local(&mut ssa, 0, 0, 0);
    let condition = local(&mut ssa, 1, 0, 1);
    let assigned = ssa.create_variable(
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(1, 0),
        MockType::I32,
    );

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: initial,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: condition,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    // B1 and B2 are both protected; B1 redefines the variable part-way
    // through, so the two arms reach the handler holding different values and
    // the handler's read needs a merge.
    let mut redefining = SsaBlock::new(1);
    redefining.add_instruction(instr(SsaOp::Const {
        dest: assigned,
        value: ConstValue::I32(2),
    }));
    redefining.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(redefining);

    let mut tail = SsaBlock::new(2);
    tail.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(tail);

    let mut handler = SsaBlock::new(3);
    handler.add_instruction(instr(SsaOp::returns(Some(initial))));
    ssa.add_block(handler);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(1, 3),
        handler_range: BlockRange::new(3, 4),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    let before = format!("{ssa}");
    let outcome = ssa.reconstruct_values(&[ValueUpdate {
        value: initial,
        definitions: vec![(1, assigned)],
    }]);
    assert!(
        outcome.is_err(),
        "the handler's merge over B1 cannot be stated from B1's end value alone: {ssa}"
    );
    assert_eq!(
        format!("{ssa}"),
        before,
        "a refused reconstruction changes nothing"
    );
}

/// A protected region with something for every pass to do.
///
/// ```text
/// B0 entry: local0 = 7; c = 1; branch  clause: try [1, 4), handler B5
/// B1 (try, loop header): branch c
/// B2 (try, body): local0 = 9; t1 = c+c
/// B3 (try, latch): t2 = c+c; t3 = copy t1; t4 = t3+t2; jump B1
/// B4 exit: return
/// B5 handler: return local0
/// ```
///
/// A loop for the loop passes, a branch on a known constant for threading, the
/// same expression twice for value numbering, an unread definition for dead
/// code - all of it inside the region, and a handler that reads a local the
/// region reassigns, so the merge those passes could break is really there.
fn try_catch_with_work_in_the_region() -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
    let outer = local(&mut ssa, 0, 0, 0);
    let condition = local(&mut ssa, 1, 0, 1);
    let inner = ssa.create_variable(
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(2, 0),
        MockType::I32,
    );
    let first = local(&mut ssa, 2, 2, 1);
    let second = local(&mut ssa, 3, 3, 0);
    let copied = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(3, 1),
        MockType::I32,
    );
    let sum = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(3, 2),
        MockType::I32,
    );
    for var in [outer, inner] {
        ssa.set_rename_group(var, 0);
    }

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: outer,
        value: ConstValue::I32(7),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: condition,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 1,
        false_target: 4,
    }));
    ssa.add_block(entry);

    let mut header = SsaBlock::new(1);
    header.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 2,
        false_target: 4,
    }));
    ssa.add_block(header);

    let mut body = SsaBlock::new(2);
    body.add_instruction(instr(SsaOp::Const {
        dest: inner,
        value: ConstValue::I32(9),
    }));
    body.add_instruction(
        instr(SsaOp::Add {
            dest: first,
            left: condition,
            right: condition,
            flags: None,
        })
        .with_result_type(MockType::I32),
    );
    body.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(body);

    let mut latch = SsaBlock::new(3);
    latch.add_instruction(
        instr(SsaOp::Add {
            dest: second,
            left: condition,
            right: condition,
            flags: None,
        })
        .with_result_type(MockType::I32),
    );
    latch.add_instruction(instr(SsaOp::Copy {
        dest: copied,
        src: first,
    }));
    latch.add_instruction(
        instr(SsaOp::Add {
            dest: sum,
            left: copied,
            right: second,
            flags: None,
        })
        .with_result_type(MockType::I32),
    );
    latch.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(latch);

    let mut exit = SsaBlock::new(4);
    exit.add_instruction(instr(SsaOp::returns(Some(condition))));
    ssa.add_block(exit);

    let mut handler = SsaBlock::new(5);
    handler.add_instruction(instr(SsaOp::returns(Some(outer))));
    ssa.add_block(handler);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(1, 4),
        handler_range: BlockRange::new(5, 6),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();
    ssa
}

/// One pass to run over the fixture: its name, how to run it, and whether the
/// fixture gives it work it must actually do.
type PassUnderTest = (&'static str, fn(&mut SsaFunction<MockTarget>) -> bool, bool);

/// Every pass leaves a function with a handler as well-formed as it found it.
///
/// A phi at a handler entry states which blocks can throw to it and what each
/// may have produced. That is a claim about the exception table and the region
/// it describes, so any pass that moves a block, retargets an edge or rewrites
/// a value can invalidate it - and a pass that does gets its whole group rolled
/// back, silently, on every method with a handler. This runs each one over a
/// function whose handler really does merge the region and demands the result
/// still verify.
/// A block that carries a phi for a group is entered holding that phi's result,
/// so that is the value a throw in it can see - not the one its predecessor
/// supplied, which never holds there at all.
///
/// A superset would still verify, so this is not about well-formedness: it is
/// about what every meet over the operands then concludes. An operand for a
/// value that cannot arrive widens the handler's answer for no reason.
#[test]
fn a_throw_sees_the_phi_a_block_starts_with_not_what_reached_it() {
    let mut ssa = try_catch_with_work_in_the_region();
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    // B1 is the loop header: it carries the group's phi, and the entry value
    // the phi merges comes from B0.
    let header_phi = some_or_abort(
        ssa.block(1)
            .and_then(|block| block.phi_nodes().first().cloned()),
    );
    let from_entry: Vec<SsaVarId> = header_phi
        .normal_operands()
        .filter(|operand| operand.predecessor() == 0)
        .map(PhiOperand::value)
        .collect();

    let handler_phi = some_or_abort(
        ssa.block(5)
            .and_then(|block| block.phi_nodes().first().cloned()),
    );
    let from_header: Vec<SsaVarId> = handler_phi
        .exceptional_operands()
        .filter(|operand| operand.predecessor() == 1)
        .map(PhiOperand::value)
        .collect();

    assert!(
        from_header.contains(&header_phi.result()),
        "the header holds its own phi from the moment it is entered: {ssa}"
    );
    for value in from_entry {
        assert!(
            !from_header.contains(&value),
            "{value:?} is what B0 supplied to the phi, and never holds inside B1: {ssa}"
        );
    }
}

/// A block a throw leaves holding the handler's own merge states it, as a loop
/// back edge does for a normal phi.
///
/// A clause with no protected range may be reached from any block, the
/// handler's own included, and the handler is entered holding its phi's
/// result. Construction dropped that operand as adding nothing, while the
/// verifier requires every block that can throw here to account for what it
/// holds, so the rebuilt function failed its own verification.
#[test]
fn a_throw_from_the_handler_itself_carries_the_handlers_merge() {
    let mut ssa = try_catch_with_work_in_the_region();
    let handlers: Vec<_> = ssa
        .exception_handlers()
        .iter()
        .cloned()
        .map(|mut handler| {
            handler.protected_range = None;
            handler
        })
        .collect();
    ssa.set_exception_handlers(handlers);
    let rebuilt = ssa.construct_ssa(&SsaConstructOptions::renaming_only());
    assert!(rebuilt.is_ok(), "{rebuilt:?}");

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "{errors:?}: {ssa}");
    let handler_phi = some_or_abort(
        ssa.block(5)
            .and_then(|block| block.phi_nodes().first().cloned()),
    );
    assert!(
        handler_phi
            .exceptional_operands()
            .any(|operand| operand.predecessor() == 5 && operand.value() == handler_phi.result()),
        "the handler throws its own merge back to itself: {ssa}"
    );
}

#[test]
fn no_pass_invalidates_a_handler_merge() {
    let method = 0u32;
    // `must_fire` marks the passes the fixture deliberately gives work to. It is
    // what makes a silent rollback visible: a pass that breaks the handler
    // merge fails its own boundary verification and restores the function, so
    // the result verifies clean and reports nothing changed. Demanding the work
    // actually happened is the difference between testing the guard and testing
    // that the guard was never reached.
    let passes: Vec<PassUnderTest> = vec![
        (
            "blockmerge",
            |ssa| {
                BlockMergingPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            },
            false,
        ),
        (
            "controlflow",
            |ssa| {
                ControlFlowSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            },
            false,
        ),
        (
            "licm",
            |ssa| {
                LicmPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            },
            true,
        ),
        (
            "loopcanon",
            |ssa| {
                LoopCanonicalizationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            },
            false,
        ),
        (
            "threading",
            |ssa| {
                JumpThreadingPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT),
                ) == Ok(true)
            },
            true,
        ),
        (
            "deadcode",
            |ssa| {
                DeadCodeEliminationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            },
            true,
        ),
        (
            "copying",
            |ssa| {
                CopyPropagationPass.propagate(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                    |_, _| {},
                )
            },
            true,
        ),
        (
            "gvn",
            |ssa| {
                GlobalValueNumberingPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            },
            true,
        ),
    ];

    for (name, run, must_fire) in passes {
        let mut ssa = try_catch_with_work_in_the_region();
        ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
        let before = format!("{ssa}");
        let baseline = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            baseline.is_empty(),
            "{name}: fixture must start clean: {baseline:?}"
        );

        let fired = run(&mut ssa);
        assert!(
            fired || !must_fire,
            "{name} had work to do here and reported none, which is what a \
             rolled-back boundary looks like:\n{before}"
        );

        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        let eh_errors: Vec<_> = errors
            .iter()
            .filter(|error| {
                matches!(
                    error,
                    VerifierError::ExceptionalOperandAtNonRoot { .. }
                        | VerifierError::ExceptionalOperandFromNonThrowingBlock { .. }
                        | VerifierError::MissingExceptionalOperand { .. }
                        | VerifierError::DuplicateExceptionalOperand { .. }
                        | VerifierError::MisplacedRuntimeValue { .. }
                        | VerifierError::PhiWithoutPredecessors { .. }
                )
            })
            .collect();
        assert!(
            eh_errors.is_empty(),
            "{name} invalidated the handler merge: {eh_errors:?}\nbefore:\n{before}\nafter:\n{ssa}"
        );

        // The clause must still describe blocks this function has, whatever the
        // pass did to them.
        assert!(
            ssa.exception_handlers()
                .iter()
                .all(|handler| handler.layout(ssa.block_count()).is_ok()),
            "{name} left the exception table describing blocks that are gone:\n{ssa}"
        );
        let _ = method;
    }
}

/// A handler's operands all name one storage location, and an operand that does
/// not is reported.
///
/// A phi at a handler entry is never materialized by moves - there is no edge
/// to put one on - so the handler reads the right value only because every
/// operand already lives in the slot it reads. That holds while the operands
/// are versions of one local. One from another group names a value living
/// somewhere else, and no lowering can repair it, so the verifier refuses it
/// rather than leaving a host to emit a wrong read.
#[test]
fn an_exceptional_operand_from_another_storage_is_reported() {
    let mut ssa = try_catch_with_work_in_the_region();
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    let clean = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(clean.is_empty(), "{clean:?}\n{ssa}");

    // The condition is local 1, a different storage from the local the handler
    // merges.
    let other_group = some_or_abort(ssa.block(0).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Const {
                    dest,
                    value: ConstValue::I32(1),
                } => Some(*dest),
                _ => None,
            })
    }));

    let handler_group = some_or_abort(
        ssa.block(5)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| ssa.rename_group(phi.result())),
    );
    // Declared as another storage: reconstruction leaves a temporary ungrouped,
    // and an ungrouped value is not the case this rule is about.
    let foreign_group = handler_group.wrapping_add(1);
    ssa.set_rename_group(other_group, foreign_group);
    assert_ne!(
        ssa.rename_group(other_group),
        handler_group,
        "the fixture must offer a value from another storage"
    );

    if let Some(block) = ssa.block_mut(5)
        && let Some(phi) = block.phi_nodes_mut().first_mut()
        && let Some(operand) = phi
            .operands_mut()
            .iter_mut()
            .find(|operand| operand.is_exceptional())
    {
        operand.set_value(other_group);
    }
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.iter().any(|error| matches!(
            error,
            VerifierError::ExceptionalOperandFromAnotherGroup { block: 5, .. }
        )),
        "{errors:?}\n{ssa}"
    );
}

/// The values a handler phi merges come back as one storage obligation.
///
/// A host assigning storage cannot discover this from the phi's shape alone:
/// an ordinary phi is lowered with a move per edge, and the rule here is that
/// there are no edges to put one on. `exceptional_webs` states it directly.
#[test]
fn a_handler_phi_reports_the_values_that_must_share_one_slot() {
    let mut ssa = try_catch_with_work_in_the_region();
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let webs = ssa.exceptional_webs();
    assert_eq!(webs.len(), 1, "one handler phi merges the region: {ssa}");
    let web = some_or_abort(webs.first());
    assert_eq!(web.block, 5);

    let phi = some_or_abort(ssa.block(5).and_then(|block| block.phi_nodes().first()));
    assert_eq!(
        web.values.first().copied(),
        Some(phi.result()),
        "the result leads, because that is what the handler reads"
    );
    for operand in phi.exceptional_operands() {
        assert!(
            web.values.contains(&operand.value()),
            "every value a throw may arrive with has to be in the same slot: {ssa}"
        );
    }
    assert_eq!(web.group, ssa.rename_group(phi.result()));

    // A function with no handler phi owes nothing.
    let mut plain = testing::diamond_phi_fixture();
    ok_or_abort(plain.construct_ssa(&SsaConstructOptions::renaming_only()));
    assert!(plain.exceptional_webs().is_empty());
}

/// `B0: local0 = 1; B1 (try): leave B3; B2 (finally): local0 = 2; endfinally;
/// B3: return local0`, with the finally clause over `[1, 2)`.
fn try_finally_writing_a_local() -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let before = local(&mut ssa, 0, 0, 0);
    let in_finally = ssa.create_variable(
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(2, 0),
        MockType::I32,
    );
    for var in [before, in_finally] {
        ssa.set_rename_group(var, 0);
    }

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: before,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    let mut guarded = SsaBlock::new(1);
    guarded.add_instruction(instr(SsaOp::Leave { target: 3 }));
    ssa.add_block(guarded);

    let mut cleanup = SsaBlock::new(2);
    cleanup.add_instruction(instr(SsaOp::Const {
        dest: in_finally,
        value: ConstValue::I32(2),
    }));
    cleanup.add_instruction(instr(SsaOp::EndFinally));
    ssa.add_block(cleanup);

    // A fresh load, which is how a front end spells "read the local here": the
    // value it yields is whatever reached this point, and reconstruction is
    // what decides that.
    let reloaded = ssa.create_variable(
        VariableOrigin::Local(0),
        2,
        DefSite::instruction(3, 0),
        MockType::I32,
    );
    ssa.set_rename_group(reloaded, 0);
    let mut after = SsaBlock::new(3);
    after.add_instruction(instr(SsaOp::LoadLocal {
        dest: reloaded,
        local_index: 0,
    }));
    after.add_instruction(instr(SsaOp::returns(Some(reloaded))));
    ssa.add_block(after);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 2,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(1, 2),
        handler_range: BlockRange::new(2, 3),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();
    ssa
}

/// A `leave` goes through the finally, and the finally resumes at the leave's
/// target.
///
/// The terminator says `leave B3`, and taking it at face value gets the graph
/// wrong twice: control does not reach B3 yet, and the `endfinally` that will
/// send it there appears to go nowhere. Everything downstream follows from
/// that - the finally's writes look dead, and code after the leave looks like
/// it bypassed the finally entirely.
#[test]
fn a_leave_runs_the_finally_before_its_target() {
    let ssa = try_finally_writing_a_local();
    let cfg = SsaCfg::from_ssa(&ssa);

    assert_eq!(
        cfg.terminator_successors(1),
        &[2],
        "the leave enters the finally, not its own target"
    );
    assert_eq!(
        cfg.terminator_successors(2),
        &[3],
        "and the endfinally is what reaches the target"
    );
    assert_eq!(
        cfg.block_predecessors(3),
        &[2],
        "so the block after the region is entered from the finally"
    );
    assert!(
        cfg.is_exception_edge(1, 2),
        "and a throw part-way through the guarded block reaches the finally too, \
         so the same edge carries both"
    );
}

/// What the finally assigns is what the code after the region reads.
///
/// The value question the edges above decide. With the leave going straight to
/// its target, the read after the region sees the value from before the try and
/// the finally's write is dead - a miscompile of every `try { } finally { x = …
/// }`.
#[test]
fn a_value_the_finally_assigns_reaches_the_code_after_it() {
    let mut ssa = try_finally_writing_a_local();
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let assigned = some_or_abort(ssa.block(2).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Const {
                    dest,
                    value: ConstValue::I32(2),
                } => Some(*dest),
                _ => None,
            })
    }));

    let returned = some_or_abort(ssa.block(3).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            })
    }));

    let reaching: Vec<SsaVarId> = ssa
        .block(3)
        .and_then(|block| {
            block
                .phi_nodes()
                .iter()
                .find(|phi| phi.result() == returned)
                .map(|phi| phi.operands().iter().map(PhiOperand::value).collect())
        })
        .unwrap_or_else(|| vec![returned]);

    assert!(
        reaching.contains(&assigned),
        "the finally ran before the target, so its assignment is what arrives: {ssa}"
    );

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
}

/// A `leave` out of nested regions runs each finally in turn, innermost first.
///
/// The chain is what makes this more than one edge: the inner `endfinally`
/// resumes into the *outer* finally, not at the leave's target, and only the
/// outermost resumes where the leave said. Getting the order wrong runs cleanup
/// in the wrong sequence, which for a `lock` inside a `using` is a real bug.
#[test]
fn nested_finallys_run_innermost_first() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let value = local(&mut ssa, 0, 0, 0);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    let mut guarded = SsaBlock::new(1);
    guarded.add_instruction(instr(SsaOp::Leave { target: 4 }));
    ssa.add_block(guarded);

    let mut inner = SsaBlock::new(2);
    inner.add_instruction(instr(SsaOp::EndFinally));
    ssa.add_block(inner);

    let mut outer = SsaBlock::new(3);
    outer.add_instruction(instr(SsaOp::EndFinally));
    ssa.add_block(outer);

    let mut after = SsaBlock::new(4);
    after.add_instruction(instr(SsaOp::returns(Some(value))));
    ssa.add_block(after);

    let finally_clause = |protected: (usize, usize), handler: (usize, usize)| SsaExceptionHandler {
        flags: 2,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(protected.0, protected.1),
        handler_range: BlockRange::new(handler.0, handler.1),
        filter_range: None,
        selector: None,
    };
    ssa.set_exception_handlers(vec![
        finally_clause((1, 2), (2, 3)),
        finally_clause((1, 3), (3, 4)),
    ]);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert_eq!(
        cfg.terminator_successors(1),
        &[2],
        "the leave enters the innermost finally"
    );
    assert_eq!(
        cfg.terminator_successors(2),
        &[3],
        "which resumes into the one enclosing it"
    );
    assert_eq!(
        cfg.terminator_successors(3),
        &[4],
        "and only the outermost resumes where the leave was going"
    );
}

/// A definition nothing reads but a handler's merge is not dead.
///
/// The only thing naming it is an operand for a throw, which is a use in the
/// one place that has no edge to hang it on. Reading only the operands a
/// terminator edge justifies would make the region's assignment look dead, and
/// deleting it leaves the handler reading a value that was never computed.
#[test]
fn a_definition_only_a_throw_carries_survives_dead_code_elimination() {
    let mut ssa = try_catch_reading_a_local(true);
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let in_region = some_or_abort(ssa.block(1).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Const { dest, .. } => Some(*dest),
                _ => None,
            })
    }));

    // Nothing but the handler's merge names it.
    let instruction_uses = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter(|instr| instr.op().uses_var(in_region))
        .count();
    assert_eq!(
        instruction_uses, 0,
        "the fixture must leave the handler's merge as the only reader: {ssa}"
    );

    let _ = DeadCodeEliminationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    );

    let survives = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .any(|instr| instr.op().defs().any(|dest| dest == in_region));
    assert!(
        survives,
        "a throw in the region can carry this value into the handler: {ssa}"
    );

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
}

/// `try_catch_reading_a_local(true)` with an empty, unreachable block ahead of
/// the protected region, which canonicalization removes and so renumbers every
/// block after it.
fn try_catch_behind_an_empty_block() -> SsaFunction<MockTarget> {
    const LOCAL_GROUP: u32 = 0;

    let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
    let before = local(&mut ssa, 0, 0, 0);
    ssa.set_rename_group(before, LOCAL_GROUP);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: before,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b0);

    ssa.add_block(SsaBlock::new(1));

    let inside = local(&mut ssa, 0, 2, 0);
    ssa.set_rename_group(inside, LOCAL_GROUP);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: inside,
        value: ConstValue::I32(9),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b3);

    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::returns(Some(before))));
    ssa.add_block(b4);

    ssa.set_exception_handlers(vec![clause(2, 3, 4, 5)]);
    ssa.recompute_uses();
    ssa
}

/// Removing a block renumbers a handler phi's exceptional operands with their
/// blocks; it does not turn them into operands for a terminator edge.
///
/// Rewritten as ordinary operands, the phi claims a terminator transfers into
/// the handler, the values a throw brings are no longer one storage location to
/// the lowering, and the handler reads a slot the protected code never wrote.
#[test]
fn canonicalization_keeps_a_handler_phi_exceptional() {
    let mut ssa = try_catch_behind_an_empty_block();
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    let before = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        before.is_empty(),
        "the rebuilt function is well formed: {before:?}\n{ssa}"
    );

    ssa.canonicalize();

    assert_eq!(ssa.block_count(), 4, "the empty block is gone:\n{ssa}");
    let phi = some_or_abort(ssa.block(3).and_then(|block| block.phi_nodes().first()));
    assert!(
        phi.normal_operands().next().is_none(),
        "no terminator transfers to the handler:\n{ssa}"
    );
    assert_eq!(
        phi.exceptional_operands().count(),
        2,
        "the throw may bring either version:\n{ssa}"
    );
    assert!(
        phi.exceptional_operands()
            .all(|operand| operand.predecessor() == 1),
        "each renumbered with the protected block it leaves from:\n{ssa}"
    );
    let after = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        after.is_empty(),
        "and the result verifies, use lists included: {after:?}\n{ssa}"
    );
}

/// Redirecting a terminator edge leaves the operands for throws alone.
///
/// The protected block both jumps somewhere and can throw into the handler.
/// Rewriting the handler's operands "for" that block as if the terminator edge
/// were the only way in drops what a throw brings and invents an edge no
/// terminator takes.
#[test]
fn expanding_a_predecessor_keeps_the_operands_for_throws() {
    let mut ssa = try_catch_reading_a_local(true);
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    let exceptional_before: Vec<PhiOperand> =
        some_or_abort(ssa.block(3).and_then(|block| block.phi_nodes().first()))
            .exceptional_operands()
            .copied()
            .collect();
    assert!(!exceptional_before.is_empty(), "{ssa}");

    // Read inside the session: a later edit in the same session reads the
    // phi as this one left it, before any repair at the boundary.
    let mut exceptional_after: Vec<PhiOperand> = Vec::new();
    let mut normal_after = 0usize;
    ok_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        editor.expand_phi_predecessor(3, 1, &[0])?;
        if let Some(phi) = editor
            .function()
            .block(3)
            .and_then(|block| block.phi_nodes().first())
        {
            exceptional_after = phi.exceptional_operands().copied().collect();
            normal_after = phi.normal_operands().count();
        }
        Ok(())
    }));

    assert_eq!(exceptional_after, exceptional_before, "{ssa}");
    assert_eq!(
        normal_after, 0,
        "the handler had no terminator edge from the block to replace:\n{ssa}"
    );
}

/// Inlining the entry trampoline keeps the value it moves reaching the phi.
///
/// A flattened dispatcher as a CIL lift leaves it: `B0` jumps to `B1`, which
/// pushes the first state and jumps to the dispatcher, whose phi merges it with
/// the states the loop writes back. The values are stack temporaries grouped by
/// their stack slot. Moving `B1` into `B0` by hand left the phi naming `B1`,
/// the rebuild dropped that operand as coming from an unreachable block, and
/// the phi read the slot's undefined entry value - so the regenerated method
/// switched on an unwritten local and always took the first case.
#[test]
fn inlining_the_entry_trampoline_keeps_the_phi_operand_defined() {
    const GROUP: u32 = 4;
    let mut ssa = SsaFunction::<MockTarget>::new(3, 1);
    // Stack temporaries, as a CIL lift makes them, grouped by their stack
    // slot; the slot also has an entry version, as the lift states one.
    let two = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(1, 0),
        MockType::I32,
    );
    let one = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(8, 0),
        MockType::I32,
    );
    let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(2), MockType::I32);
    let entry = ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
    for v in [two, one, merged, entry] {
        ssa.set_rename_group(v, GROUP);
    }
    let arg2 = ssa.create_variable(
        VariableOrigin::Argument(2),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let arg1 = ssa.create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let quotient = local(&mut ssa, 1, 10, 0);
    let blocks: Vec<Vec<SsaOp<MockTarget>>> = vec![
        vec![SsaOp::Jump { target: 1 }],
        vec![
            SsaOp::Const {
                dest: two,
                value: ConstValue::I32(2),
            },
            SsaOp::Jump { target: 2 },
        ],
        vec![SsaOp::Switch {
            value: merged,
            targets: vec![9, 10, 6],
            default: 3,
        }],
        vec![SsaOp::Jump { target: 6 }],
        vec![],
        vec![],
        vec![SsaOp::Branch {
            condition: arg2,
            true_target: 8,
            false_target: 9,
        }],
        vec![],
        vec![
            SsaOp::Const {
                dest: one,
                value: ConstValue::I32(1),
            },
            SsaOp::Jump { target: 2 },
        ],
        vec![SsaOp::returns(None)],
        vec![
            SsaOp::Div {
                dest: quotient,
                left: arg1,
                right: arg2,
                unsigned: false,
                flags: None,
            },
            SsaOp::returns(Some(quotient)),
        ],
    ];
    for (index, ops) in blocks.into_iter().enumerate() {
        let mut block = SsaBlock::new(index);
        if index == 2 {
            let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(two, 1));
            phi.add_operand(PhiOperand::new(one, 8));
            block.add_phi(phi);
        }
        for op in ops {
            block.add_instruction(instr(op));
        }
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    assert!(
        BlockMergingPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let phi = some_or_abort(ssa.block(2).and_then(|block| block.phi_nodes().first()));
    let from_entry = some_or_abort(phi.operand_from(0)).value();
    assert!(
        matches!(
            ssa.get_definition(from_entry),
            Some(SsaOp::Const {
                value: ConstValue::I32(2),
                ..
            })
        ),
        "the first state still reaches the dispatcher:\n{ssa}"
    );
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
}

/// An edit session that fails without a snapshot says so, and one with a
/// snapshot restores the function and does not.
///
/// Under the default policy a failed session leaves its edits applied, and
/// nothing guarantees the result fails verification. The record is what lets
/// the code running a pass fail it, so the transaction restores the function.
#[test]
fn an_edit_that_fails_without_a_snapshot_is_recorded() {
    let build = || {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
        let value = local(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(value))));
        ssa.add_block(b0);
        ssa.recompute_uses();
        ssa
    };
    let failing = |editor: &mut SsaEditor<'_, MockTarget>| {
        editor.remove_instruction_tail(0, 0)?;
        Err(Error::new("the edit gave up half way"))
    };

    let mut unguarded = build();
    assert!(unguarded.edit(SsaEditOptions::new(), failing).is_err());
    assert!(
        unguarded.take_edit_failure().is_some(),
        "a half-applied edit is recorded"
    );
    assert!(unguarded.take_edit_failure().is_none(), "and taken once");

    let mut guarded = build();
    let options = SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure);
    assert!(guarded.edit(options, failing).is_err());
    assert!(
        guarded.take_edit_failure().is_none(),
        "a restored function has nothing to report"
    );
    assert_eq!(some_or_abort(guarded.block(0)).instructions().len(), 2);
}

/// Pruning keeps a handler's operands for throws.
///
/// A catch whose entry is also a loop header inside the handler: `B3` is
/// entered by a throw from `B1` or `B2`, and again from `B4`, which loops back.
/// Its predecessors were read from terminators alone, where a throw is no edge,
/// so `B4` was its only predecessor, and every operand for a throw was pruned -
/// the handler then saw only what its own loop carried, whatever the throw
/// left.
#[test]
fn pruning_keeps_a_handlers_operands_for_throws() {
    let mut ssa = SsaFunction::<MockTarget>::new(1, 2);
    let before = local(&mut ssa, 0, 0, 0);
    let inside = local(&mut ssa, 0, 1, 0);
    let again = local(&mut ssa, 0, 4, 0);
    let observed = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(3), MockType::I32);
    let condition = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let blocks: Vec<Vec<SsaOp<MockTarget>>> = vec![
        vec![
            SsaOp::Const {
                dest: before,
                value: ConstValue::I32(0),
            },
            SsaOp::Jump { target: 1 },
        ],
        vec![
            SsaOp::Const {
                dest: inside,
                value: ConstValue::I32(1),
            },
            SsaOp::Jump { target: 2 },
        ],
        vec![SsaOp::Leave { target: 5 }],
        vec![SsaOp::Branch {
            condition,
            true_target: 4,
            false_target: 5,
        }],
        vec![
            SsaOp::Const {
                dest: again,
                value: ConstValue::I32(2),
            },
            SsaOp::Jump { target: 3 },
        ],
        vec![SsaOp::returns(None)],
    ];
    for (index, ops) in blocks.into_iter().enumerate() {
        let mut block = SsaBlock::new(index);
        if index == 3 {
            let mut phi = PhiNode::new(observed, VariableOrigin::Local(1));
            phi.add_operand(PhiOperand::exceptional(before, 1));
            phi.add_operand(PhiOperand::exceptional(inside, 1));
            phi.add_operand(PhiOperand::exceptional(inside, 2));
            phi.add_operand(PhiOperand::new(again, 4));
            block.add_phi(phi);
        }
        for op in ops {
            block.add_instruction(instr(op));
        }
        ssa.add_block(block);
    }
    ssa.set_exception_handlers(vec![clause(1, 3, 3, 5)]);
    ssa.recompute_uses();

    let everything = BitSet::full(6);
    let pruned = ssa.prune_phi_operands(&everything);
    let phi = some_or_abort(ssa.block(3).and_then(|block| block.phi_nodes().first()));
    assert_eq!(pruned, 0, "every operand names a way in: {phi}");
    assert_eq!(phi.exceptional_operands().count(), 3, "{phi}");
}

/// Clearing a block nothing reaches keeps a constant reachable code still
/// reads, as the rebuild's rescue did.
///
/// `B1` is unreachable but defines the constant `B2` returns - the shape a
/// CFG edit upstream can leave behind. The constant moves to the entry block,
/// and the block goes.
#[test]
fn clearing_an_unreached_block_keeps_a_constant_still_read() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
    let five = local(&mut ssa, 0, 1, 0);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: five,
        value: ConstValue::I32(5),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(five))));
    ssa.add_block(b2);
    ssa.recompute_uses();

    let mut cleared = Vec::new();
    let reachable = {
        let mut set = BitSet::new(3);
        set.insert(0);
        set.insert(2);
        set
    };
    ok_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        cleared = editor.clear_unreached_blocks(&reachable);
        Ok(())
    }));
    assert_eq!(cleared, vec![1]);
    assert!(some_or_abort(ssa.block(1)).is_empty());
    let returned = some_or_abort(ssa.block(2).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            })
    }));
    assert!(
        matches!(
            ssa.get_definition(returned),
            Some(SsaOp::Const {
                value: ConstValue::I32(5),
                ..
            })
        ),
        "the constant still reaches the return:\n{ssa}"
    );
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
}

/// A handler's phi is a definition of its variable, so the block the handler
/// leaves to merges it with what the other paths carry.
///
/// `try { call(); x = 7; } catch { } return x;` returns 7 when the try
/// completes and the `x` current at the throw when the call throws. With the
/// try at the start of the method its writes are all in the entry block, and a
/// group written only there was left out of phi placement altogether; the
/// handler's phi was then placed on its own, never counted as a definition, and
/// the join after the catch read the try's value on both paths.
#[test]
fn the_block_a_handler_leaves_to_merges_the_handler_phi() {
    // B0 the protected entry (a call that may throw, then x = 7, then leave to
    // B2), B1 the handler (leaves to B2), B2 the join, which returns x.
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    // The value x holds on entry, as a front end states it for every local.
    let on_entry =
        ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
    let assigned = ssa.create_variable(
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(0, 1),
        MockType::I32,
    );
    // What the join reads: a version of x no instruction defines, which
    // construction answers with the definition that reaches the read.
    let read = ssa.create_variable(
        VariableOrigin::Local(0),
        2,
        DefSite::instruction(2, 0),
        MockType::I32,
    );
    for var in [on_entry, assigned, read] {
        ssa.set_rename_group(var, 0);
    }

    let mut protected = SsaBlock::new(0);
    protected.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(1),
        vec![],
    ))));
    protected.add_instruction(instr(SsaOp::Const {
        dest: assigned,
        value: ConstValue::I32(7),
    }));
    protected.add_instruction(instr(SsaOp::Leave { target: 2 }));
    ssa.add_block(protected);

    let mut handler = SsaBlock::new(1);
    handler.add_instruction(instr(SsaOp::Leave { target: 2 }));
    ssa.add_block(handler);

    let mut join = SsaBlock::new(2);
    join.add_instruction(instr(SsaOp::returns(Some(read))));
    ssa.add_block(join);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 2),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::unverified()));

    let handler_phi = some_or_abort(
        some_or_abort(ssa.block(1))
            .phi_nodes()
            .first()
            .map(PhiNode::result),
    );
    let join_block = some_or_abort(ssa.block(2));
    let join_phi = some_or_abort(join_block.phi_nodes().first());
    assert_eq!(
        join_phi.operand_from(1).map(PhiOperand::value),
        Some(handler_phi),
        "the path through the catch carries the handler's x: {ssa}"
    );
    assert!(
        matches!(
            join_block.instructions().last().map(SsaInstruction::op),
            Some(SsaOp::Return { value: Some(read), .. }) if *read == join_phi.result()
        ),
        "the return reads the merge, not the try's x: {ssa}"
    );
}

/// A throw on a later iteration of a loop that re-enters a protected region
/// carries the value the variable holds then, not the one it had before the
/// loop.
///
/// The handler's phi only exists once construction has run, so liveness
/// computed before it never saw the throw read the variable. The variable then
/// looked dead at the loop head, no phi was placed there, and the exceptional
/// operand was answered with the definition before the loop.
#[test]
fn a_loop_that_reenters_a_try_merges_the_variable_at_its_head() {
    // B0 entry: jump B1. B1 loop head: jump B2. B2 protected: call (may
    // throw), x = 7, leave B4. B3 handler: return x. B4: branch back to B1 or
    // on to B5. B5: return.
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let on_entry =
        ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
    let assigned = ssa.create_variable(
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(2, 1),
        MockType::I32,
    );
    let read = ssa.create_variable(
        VariableOrigin::Local(0),
        2,
        DefSite::instruction(3, 0),
        MockType::I32,
    );
    for var in [on_entry, assigned, read] {
        ssa.set_rename_group(var, 0);
    }
    let again = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(4, 0),
        MockType::I1,
    );

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    let mut head = SsaBlock::new(1);
    head.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(head);

    let mut protected = SsaBlock::new(2);
    protected.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(1),
        vec![],
    ))));
    protected.add_instruction(instr(SsaOp::Const {
        dest: assigned,
        value: ConstValue::I32(7),
    }));
    protected.add_instruction(instr(SsaOp::Leave { target: 4 }));
    ssa.add_block(protected);

    let mut handler = SsaBlock::new(3);
    handler.add_instruction(instr(SsaOp::returns(Some(read))));
    ssa.add_block(handler);

    let mut latch = SsaBlock::new(4);
    latch.add_instruction(instr(SsaOp::Const {
        dest: again,
        value: ConstValue::True,
    }));
    latch.add_instruction(instr(SsaOp::Branch {
        condition: again,
        true_target: 1,
        false_target: 5,
    }));
    ssa.add_block(latch);

    let mut exit = SsaBlock::new(5);
    exit.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(exit);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(2, 3),
        handler_range: BlockRange::new(3, 4),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::unverified()));

    let head_phi = some_or_abort(
        some_or_abort(ssa.block(1))
            .phi_nodes()
            .first()
            .map(PhiNode::result),
    );
    let handler_phi = some_or_abort(some_or_abort(ssa.block(3)).phi_nodes().first());
    let carried: Vec<SsaVarId> = handler_phi
        .exceptional_operands()
        .map(PhiOperand::value)
        .collect();
    assert!(
        carried.contains(&head_phi),
        "a throw from the call carries the value x holds at the loop head: {ssa}"
    );
    assert!(
        !carried.contains(&on_entry),
        "the value before the loop is the head's operand, not the throw's: {ssa}"
    );
}

/// A merge on the evaluation stack is typed by what it merges, not by what
/// its stack depth held first.
///
/// A stack depth is not a variable; the values pushed there are unrelated. A
/// phi of two `int32` constants at a depth that earlier held a boolean was
/// typed as the boolean, and a code generator that gives the merge a slot of
/// that type truncates it.
#[test]
fn a_stack_merge_is_typed_by_its_operands() {
    // Group 5 is a stack depth (no arguments, one local). B0: flag (i1) at the
    // depth, branch on it. B1 and B2 push an i32 there each, and B3 reads the
    // merge.
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let depth = 5;
    let flag = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 0),
        MockType::I1,
    );
    let left = ssa.create_variable(
        VariableOrigin::Phi,
        1,
        DefSite::instruction(1, 0),
        MockType::I32,
    );
    let right = ssa.create_variable(
        VariableOrigin::Phi,
        2,
        DefSite::instruction(2, 0),
        MockType::I32,
    );
    let merged = ssa.create_variable(
        VariableOrigin::Phi,
        3,
        DefSite::instruction(3, 0),
        MockType::Unknown,
    );
    for var in [flag, left, right, merged] {
        ssa.set_rename_group(var, depth);
    }

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: flag,
        value: ConstValue::True,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: flag,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);
    for (index, dest, value) in [(1, left, -16), (2, right, -20)] {
        let mut block = SsaBlock::new(index);
        block.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
        block.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(block);
    }
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(b3);
    ssa.recompute_uses();

    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::unverified()));

    let phi = some_or_abort(
        some_or_abort(ssa.block(3))
            .phi_nodes()
            .first()
            .map(PhiNode::result),
    );
    assert_eq!(
        ssa.variable(phi).map(|variable| *variable.var_type()),
        Some(MockType::I32),
        "{ssa}"
    );
}
