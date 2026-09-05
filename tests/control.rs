//! Control-flow and block utility SSA patterns.

use analyssa::{
    analysis::{
        SsaCfg, SsaVerifier, VerifyLevel, loop_analyzer::SsaLoopAnalysis, verifier::VerifierError,
    },
    ir::{
        block::SsaBlock,
        exception::{
            BlockRange, ClausePart, ExceptionTableError, HandlerKind, SsaExceptionHandler,
        },
        function::{FunctionKind, SsaFunction},
        instruction::SsaInstruction,
        ops::{CmpKind, SsaOp},
        phi::PhiNode,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
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

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| std::process::abort())
}

fn ok_or_abort<T, E>(value: Result<T, E>) -> T {
    value.unwrap_or_else(|_| std::process::abort())
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
        block.add_instruction(instr(SsaOp::Return { value: None }));
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
/// The view used to give it a synthetic successor — the region's handler —
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
    try_entry.add_instruction(instr(SsaOp::Return { value: Some(value) }));
    ssa.add_block(try_entry);

    let mut handler = SsaBlock::new(1);
    handler.add_instruction(instr(SsaOp::Return { value: None }));
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
    }]);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    assert!(
        cfg.block_successors(0).is_empty(),
        "a return leaves the function, protected region or not"
    );
    assert!(
        cfg.block_predecessors(1).is_empty(),
        "the runtime enters a handler; no terminator does"
    );
    assert_eq!(cfg.exits().len(), 2, "both returns are exits");
    // The region survives for a consumer that needs it — it is simply not an
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v2) }));

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

// ---------------------------------------------------------------------------
// Interrupt / ISR function integration tests
// ---------------------------------------------------------------------------

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
    block.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa.canonicalize();

    assert_eq!(ssa.kind(), FunctionKind::Normal);
    assert!(!ssa.has_interrupt_return());
}

// ---------------------------------------------------------------------------
// Handler-kind classification and clause layout, through a host `Target`
// ---------------------------------------------------------------------------

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
    }
}

/// Builds `B0: local0 = 7; B1 (try): <body>; B2: return; B3 (handler): return local0`.
///
/// Both definitions carry rename group 0, which is how a front end says "these
/// are versions of one local" — the rebuilder derives groups from phi
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
    b2.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b2);

    // The handler reads the local.
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Return {
        value: Some(before),
    }));
    ssa.add_block(b3);

    ssa.set_exception_handlers(vec![clause(1, 2, 3, 4)]);
    ssa.recompute_uses();
    ssa
}

/// The value the handler block returns after a rebuild.
fn handler_return_value(ssa: &SsaFunction<MockTarget>) -> Option<SsaVarId> {
    ssa.block(3).and_then(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instr| match instr.op() {
                SsaOp::Return { value } => *value,
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
    ok_or_abort(ssa.rebuild_ssa());

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
    assert!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty(),
        "and the result must verify"
    );
}

/// A group the protected region reassigns has no single reaching value at the
/// handler, so the handler reads a merge value rather than either candidate.
#[test]
fn a_local_redefined_inside_a_protected_region_reads_a_root_entry_value() {
    let mut ssa = try_catch_reading_a_local(true);
    ok_or_abort(ssa.rebuild_ssa());

    let before_region = ssa
        .block(0)
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

    // The merge value is defined by entering the handler: a phi-shaped site at
    // the root, carried by no phi node.
    let site = ssa.variable(read).map(|variable| variable.def_site());
    assert_eq!(
        site.map(|site| (site.block, site.instruction)),
        Some((3, None)),
        "the merge value is defined at the top of the handler"
    );
    assert!(
        ssa.block(3)
            .is_some_and(|block| block.phi_nodes().iter().all(|phi| phi.result() != read)),
        "and by no phi node -- a phi would assert a known merge"
    );
}

/// No phi is placed at an exception root reached only by the runtime.
///
/// A phi's operands name terminator edges, and there are none here.
#[test]
fn no_phi_is_placed_at_an_exception_root() {
    let mut ssa = try_catch_reading_a_local(true);
    ok_or_abort(ssa.rebuild_ssa());

    assert!(
        ssa.block(3)
            .is_some_and(|block| block.phi_nodes().is_empty()),
        "the handler entry has no terminator predecessor, so it may carry no phi"
    );
}

/// A handler entry no terminator transfers to may not carry a phi, and the
/// rebuild removes one rather than emitting IR the verifier rejects.
#[test]
fn a_handler_entry_reached_only_by_the_runtime_carries_no_phi() {
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
            VerifierError::PhiWithoutPredecessors { block: 3, .. }
        )),
        "a phi whose operands could name no edge must be reported: {errors:?}"
    );

    let demoted = ssa.demote_runtime_entry_phis();
    assert_eq!(demoted, 1, "the phi is removed, not repaired in place");
    assert!(ssa.block(3).is_some_and(|b| b.phi_nodes().is_empty()));

    let after = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        !after
            .iter()
            .any(|error| matches!(error, VerifierError::PhiWithoutPredecessors { .. })),
        "and the result is representable: {after:?}"
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
        cfg.block_predecessors(3),
        &[2],
        "B3 now has a terminator predecessor, so a phi there is representable"
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
    b4.add_instruction(instr(SsaOp::Return { value: None }));
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
    b3.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(b3);

    ssa.set_exception_handlers(vec![clause(1, 2, 2, 3)]);
    ssa.recompute_uses();

    let before = ssa.block_count();
    let events = analyssa::events::EventLog::<MockTarget>::new();
    analyssa::passes::loopcanon::run(&mut ssa, &0u32, &events);

    assert_eq!(
        ssa.block_count(),
        before,
        "no preheader may be spliced in front of a block no terminator names"
    );
}
