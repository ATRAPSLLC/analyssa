//! Malformed IR is reported, never panicked on.
//!
//! Every index in the IR - a terminator target, a phi predecessor, a variable
//! id, a def site, a use site - is read by something that indexes a fixed-size
//! structure with it. A host builds functions through the raw surface
//! (`blocks_mut`, `instructions_mut`, `create_variable`), so an index that names
//! nothing is reachable rather than hypothetical, and the crate owes such a
//! function two things: [`SsaFunction::validate_indices`] names what is wrong,
//! and nothing else aborts the process on the way to finding out.
//!
//! The corruptions below are the shapes that reach a host: a branch to a block
//! that was removed, a phi operand left naming a predecessor that no longer
//! exists, an id minted past the variable table, and def/use sites that outlived
//! the instructions they name.

use std::collections::BTreeMap;

use analyssa::{
    PointerSize,
    analysis::{
        AnalysisContext, Budget, Machine, SsaVerifier, VerifierError, VerifyLevel, WorkBudget,
    },
    events::NullListener,
    ir::{
        SsaExceptionHandler,
        block::SsaBlock,
        exception::BlockRange,
        function::{SsaConstructOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, UseSite, VariableOrigin},
    },
    passes::{
        self, AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
        CopyPropagationPass, DeadCodeEliminationPass, GlobalValueNumberingPass, JumpThreadingPass,
        LicmPass, LoopCanonicalizationPass, MemoryOptimizationPass, OpaquePredicatePass,
        ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
    },
    scheduling::{FunctionPass, PassEnv},
    testing::{self, MACHINE, MockTarget, MockType},
};

/// One way of corrupting a function, with the error it must produce.
struct Corruption {
    /// What the corruption is, for failure messages.
    label: &'static str,
    /// Applies it to a freshly built fixture.
    apply: fn(&mut SsaFunction<MockTarget>),
    /// The error it must produce, or `None` when the function stays well
    /// indexed.
    expect: Option<fn(&VerifierError) -> bool>,
}

/// The fixture every corruption starts from: a loop, so the passes below have
/// phis, a back edge and a dominator tree to work with.
fn fixture() -> SsaFunction<MockTarget> {
    testing::loop_counter_fixture()
}

/// A variable id far past the function's variable table.
fn out_of_range_var(ssa: &SsaFunction<MockTarget>) -> SsaVarId {
    SsaVarId::from_index(ssa.variables().len().saturating_add(64))
}

/// Rewrites the first terminator that names a block into one naming `target`.
fn retarget_first_jump(ssa: &mut SsaFunction<MockTarget>, target: usize) {
    for block_idx in 0..ssa.block_count() {
        let Some(block) = ssa.block_mut(block_idx) else {
            continue;
        };
        let Some(instr) = block.instructions_mut().last_mut() else {
            continue;
        };
        if matches!(instr.op(), SsaOp::Jump { .. }) {
            instr.set_op(SsaOp::Jump { target });
            return;
        }
    }
}

/// The corruption table.
fn corruptions() -> Vec<Corruption> {
    vec![
        Corruption {
            label: "terminator naming a block past the end",
            apply: |ssa| {
                let target = ssa.block_count().saturating_add(9);
                retarget_first_jump(ssa, target);
            },
            expect: Some(|e| matches!(e, VerifierError::DanglingBranchTarget { .. })),
        },
        Corruption {
            label: "phi operand naming a predecessor past the end",
            apply: |ssa| {
                let beyond = ssa.block_count().saturating_add(3);
                for block_idx in 0..ssa.block_count() {
                    let Some(block) = ssa.block_mut(block_idx) else {
                        continue;
                    };
                    if let Some(phi) = block.phi_nodes_mut().first_mut() {
                        let value = phi
                            .operands()
                            .first()
                            .map_or(SsaVarId::PLACEHOLDER, |o| o.value());
                        phi.add_operand(PhiOperand::new(value, beyond));
                        return;
                    }
                }
            },
            expect: Some(|e| matches!(e, VerifierError::PhiPredecessorOutOfRange { .. })),
        },
        Corruption {
            label: "instruction operand naming no variable",
            apply: |ssa| {
                let bogus = out_of_range_var(ssa);
                if let Some(block) = ssa.block_mut(0)
                    && let Some(instr) = block.instructions_mut().first_mut()
                {
                    instr.set_op(SsaOp::returns(Some(bogus)));
                }
            },
            expect: Some(|e| matches!(e, VerifierError::VariableIdOutOfRange { .. })),
        },
        Corruption {
            label: "phi result naming no variable",
            apply: |ssa| {
                let bogus = out_of_range_var(ssa);
                for block_idx in 0..ssa.block_count() {
                    let Some(block) = ssa.block_mut(block_idx) else {
                        continue;
                    };
                    if !block.phi_nodes().is_empty() {
                        let origin = VariableOrigin::Phi;
                        let mut phi = PhiNode::new(bogus, origin);
                        phi.add_operand(PhiOperand::new(bogus, 0));
                        block.phi_nodes_mut().push(phi);
                        return;
                    }
                }
            },
            expect: Some(|e| matches!(e, VerifierError::VariableIdOutOfRange { .. })),
        },
        Corruption {
            label: "def site naming a block past the end",
            apply: |ssa| {
                let beyond = ssa.block_count().saturating_add(5);
                if let Some(var) = ssa.variables_mut().last_mut() {
                    var.set_def_site(DefSite::instruction(beyond, 0));
                }
            },
            expect: Some(|e| matches!(e, VerifierError::DefSiteOutOfRange { .. })),
        },
        Corruption {
            label: "def site naming an instruction past the end of its block",
            apply: |ssa| {
                if let Some(var) = ssa.variables_mut().last_mut() {
                    var.set_def_site(DefSite::instruction(0, 4096));
                }
            },
            expect: Some(|e| matches!(e, VerifierError::DefSiteOutOfRange { .. })),
        },
        Corruption {
            label: "use site naming an instruction past the end of its block",
            apply: |ssa| {
                if let Some(var) = ssa.variables_mut().last_mut() {
                    var.add_use(UseSite::instruction(0, 4096));
                }
            },
            expect: Some(|e| matches!(e, VerifierError::UseSiteOutOfRange { .. })),
        },
        Corruption {
            label: "use site naming a phi past the end of its block",
            apply: |ssa| {
                if let Some(var) = ssa.variables_mut().last_mut() {
                    var.add_use(UseSite::phi_operand(0, 4096));
                }
            },
            expect: Some(|e| matches!(e, VerifierError::UseSiteOutOfRange { .. })),
        },
        Corruption {
            label: "untouched fixture",
            apply: |_| {},
            expect: None,
        },
    ]
}

/// Runs every pass, every repair and every verification level over `ssa`.
///
/// The passes are run each on its own copy: a pass that rejects malformed IR by
/// leaving it alone must not hide the next pass's view of the same corruption.
fn exercise_everything(ssa: &SsaFunction<MockTarget>) {
    let method = 0_u32;
    let events = NullListener;
    let ptr = PointerSize::Bit64;

    let mut copy = ssa.clone();
    let _ = AlgebraicSimplificationPass.run(
        &mut copy,
        &PassEnv::new(
            &method,
            &events,
            AnalysisContext::new(Machine {
                ptr_size: ptr,
                ..MACHINE
            }),
        ),
    );
    let mut copy = ssa.clone();
    let _ = BlockMergingPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::BlockMergeRounds, 4)),
    );
    let mut copy = ssa.clone();
    let _ = ControlFlowSimplificationPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ControlFlowRounds, 4)),
    );
    let mut copy = ssa.clone();
    let _ = CopyPropagationPass.propagate(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, 4)),
        |_, _| {},
    );
    let mut copy = ssa.clone();
    let _ = DeadCodeEliminationPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::DeadCodeRounds, 4)),
    );
    let mut copy = ssa.clone();
    let _ = GlobalValueNumberingPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)),
    );
    let mut copy = ssa.clone();
    let _ = LicmPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)),
    );
    let mut copy = ssa.clone();
    let _ = LoopCanonicalizationPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)),
    );
    let mut copy = ssa.clone();
    let _ = MemoryOptimizationPass.run(
        &mut copy,
        &PassEnv::new(
            &method,
            &events,
            AnalysisContext::new(Machine {
                ptr_size: ptr,
                ..MACHINE
            }),
        ),
    );
    let mut copy = ssa.clone();
    let _ = OpaquePredicatePass::new().run(
        &mut copy,
        &PassEnv::new(
            &method,
            &events,
            AnalysisContext::new(Machine {
                ptr_size: ptr,
                ..MACHINE
            }),
        ),
    );
    let mut copy = ssa.clone();
    let starved = WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 4);
    let _ = ValueRangePropagationPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)).with_budget(starved),
    );
    let mut copy = ssa.clone();
    let _ = ReassociationPass.run(
        &mut copy,
        &PassEnv::new(
            &method,
            &events,
            AnalysisContext::new(Machine {
                ptr_size: ptr,
                ..MACHINE
            }),
        ),
    );
    let mut copy = ssa.clone();
    let _ = StrengthReductionPass.run(
        &mut copy,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)).with_budget(starved),
    );
    let mut copy = ssa.clone();
    let _ = JumpThreadingPass.run(
        &mut copy,
        &PassEnv::new(
            &method,
            &events,
            AnalysisContext::new(Machine {
                ptr_size: ptr,
                ..MACHINE
            }),
        )
        .with_budget(starved),
    );
    let copy = ssa.clone();
    let _ = passes::ranges::analyze(&copy, &starved, &BTreeMap::new());

    let mut copy = ssa.clone();
    copy.repair_ssa();
    let mut copy = ssa.clone();
    let _ = copy.construct_ssa(&SsaConstructOptions::renaming_only());
    let mut copy = ssa.clone();
    copy.canonicalize();
    let mut copy = ssa.clone();
    let _ = copy.ensure_entry_has_no_predecessors();
    let mut copy = ssa.clone();
    copy.recompute_uses();

    let _ = SsaVerifier::new(ssa).verify(VerifyLevel::Full);
}

#[test]
fn a_malformed_function_is_reported_not_panicked() {
    for corruption in corruptions() {
        let mut ssa = fixture();
        (corruption.apply)(&mut ssa);

        let errors: Vec<VerifierError> = SsaVerifier::new(&ssa).index_bounds_only();
        match corruption.expect {
            Some(expected) => assert!(
                errors.iter().any(expected),
                "{}: expected variant absent from {errors:?}",
                corruption.label
            ),
            None => assert!(
                errors.is_empty(),
                "{}: well-indexed function reported {errors:?}",
                corruption.label
            ),
        }
        assert_eq!(
            ssa.validate_indices().is_err(),
            corruption.expect.is_some(),
            "{}: validate_indices disagrees with index_bounds_only",
            corruption.label
        );

        // The point of the test: none of this aborts the process. A pass is
        // free to refuse the function, and construction is free to fail.
        exercise_everything(&ssa);
    }
}

#[test]
fn a_well_formed_function_names_only_indices_it_holds() {
    for (label, ssa) in [
        ("const return", testing::const_i32_return(7)),
        ("diamond phi", testing::diamond_phi_fixture()),
        ("try/catch", testing::try_catch_fixture()),
        ("loop counter", testing::loop_counter_fixture()),
        ("scalar rewrite", testing::scalar_rewrite_fixture()),
        ("memory effects", testing::memory_effect_fixture()),
        ("native effects", testing::native_effect_fixture()),
        ("vector simd", testing::vector_simd_fixture()),
    ] {
        assert!(
            ssa.validate_indices().is_ok(),
            "{label}: {:?}",
            ssa.validate_indices()
        );
    }
}

/// A value defined by several phis, one of them with no operands, in a function
/// whose third block is reached only as a handler, as the pipeline fuzzer found
/// it:
///
/// ```text
/// B0: v1 = phi()            ; ret v1
/// B1: v1 = phi(v0 from B0)  ; v9 = 65793 ; ret v1     (protected)
/// B2: v1 = phi(v1 from B0)  ; branch v0, B1, B1       (B1's handler)
/// ```
///
/// with a filter range naming blocks that do not exist, and variables whose
/// local origins run past the function's local count. Construction must not
/// index its trivial-phi set with a phi result past the variable table. It may
/// refuse the function; it may not abort.
#[test]
fn a_value_defined_by_several_phis_does_not_abort_a_rebuild() {
    let mut ssa = SsaFunction::<MockTarget>::new(145, 23);
    // The fuzzer's pool: 64 locals, most of which nothing names. Ids are
    // minted in order, so the three the function uses are `v0`, `v1` and `v9`.
    for local in 0..64u16 {
        let _ = ssa.create_variable(
            VariableOrigin::Local(local),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
    }
    let (v0, v1, v9) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(9),
    );
    let phi = |operands: &[(SsaVarId, usize)]| {
        let mut phi = PhiNode::new(v1, VariableOrigin::Phi);
        for &(value, predecessor) in operands {
            phi.add_operand(PhiOperand::new(value, predecessor));
        }
        phi
    };

    let mut entry = SsaBlock::new(0);
    entry.add_phi(phi(&[]));
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v1))));
    let mut protected = SsaBlock::new(1);
    protected.add_phi(phi(&[(v0, 0)]));
    protected.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: v9,
        value: ConstValue::I32(65793),
    }));
    protected.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v1))));
    let mut handler = SsaBlock::new(2);
    handler.add_phi(phi(&[(v1, 0)]));
    handler.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
        condition: v0,
        true_target: 1,
        false_target: 1,
    }));
    ssa.add_block(entry);
    ssa.add_block(protected);
    ssa.add_block(handler);
    // An empty range is no range: the second handler protects nothing.
    let handler = |flags, ranges: [(usize, usize); 3]| SsaExceptionHandler {
        flags,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(ranges[0].0, ranges[0].1),
        handler_range: BlockRange::new(ranges[1].0, ranges[1].1),
        filter_range: BlockRange::new(ranges[2].0, ranges[2].1),
        selector: None,
    };
    ssa.set_exception_handlers(vec![
        handler(16_776_983, [(1, 2), (2, 3), (21, 22)]),
        handler(43, [(0, 0), (0, 0), (0, 0)]),
    ]);
    ssa.recompute_uses();

    // The handler and the block it branches to are live - the table says the
    // runtime may enter B2 - and construction renames them from a tree of their
    // own, so no phi there keeps the placeholder it was placed with.
    let mut rebuilt = ssa.clone();
    let _ = rebuilt.construct_ssa(&SsaConstructOptions::renaming_only());
    let table = rebuilt.variables().len();
    for block in rebuilt.blocks() {
        for phi in block.phi_nodes() {
            assert!(
                phi.result().index() < table,
                "B{} holds a phi of {:?}, past the {table} variables",
                block.id(),
                phi.result()
            );
        }
    }
    exercise_everything(&ssa);
}
