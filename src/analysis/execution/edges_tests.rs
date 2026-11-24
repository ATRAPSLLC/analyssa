//! Actual explorer transfers and conservative edge-meet acceptance.

use std::cell::Cell;

use crate::{
    analysis::{
        context::AnalysisContext,
        execution::{
            Completion, ExecutionHost, ObservationSizeRefusal, Root, StaticExecution, Value,
            edges::{
                EdgeAllowance, EdgeByteRequest, EdgeObservationPlan, EdgeRecords,
                EdgeRequestLimits, EdgeScalarRequest, EdgeSite,
            },
            frame::Frame,
            memory::Memory,
            state::State,
            value::Address,
        },
        image::{ImageData, ImageRegion, Mutability},
        verifier::VerifyLevel,
    },
    ir::{
        CallData, Callee, SsaDefSpec, SsaFunctionBuilder,
        exception::{BlockRange, SsaExceptionHandler},
        function::SsaFunction,
        ops::{AccessQualifiers, CmpKind, SsaOp},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

struct Image;
impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, address: u64) -> Option<ImageRegion> {
        (0x1000..0x1100).contains(&address).then_some(ImageRegion {
            start: 0x1000,
            end: 0x1100,
            mutability: Mutability::Writable,
        })
    }
    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }
    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}
fn limits() -> EdgeRequestLimits {
    EdgeRequestLimits {
        sites: 8,
        requests: 16,
    }
}
fn request<'a>(
    body: &'a SsaFunction<MockTarget>,
    from: usize,
    to: usize,
    pointer: SsaVarId,
    bytes: usize,
) -> EdgeByteRequest<'a, MockTarget> {
    EdgeByteRequest {
        body,
        site: EdgeSite::of(body, from, to).unwrap(),
        pointer,
        maximum_bytes: bytes,
    }
}
fn spec(ty: MockType) -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(ty)
}
fn diamond() -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
    let mut b = SsaFunctionBuilder::new(1, 0);
    let condition = b.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let pointer = b
        .in_block(0, |b| {
            let pointer = b.const_value(spec(MockType::Ptr), ConstValue::NativeUInt(0x1000))?;
            let zero = b.const_i32(spec(MockType::I32), 0)?;
            b.branch_cmp(condition, zero, CmpKind::Eq, false, 1, 2)?;
            Ok(pointer)
        })
        .unwrap();
    for (block, value) in [(1, 0x00ff0041), (2, 0x00ff0042)] {
        b.in_block(block, |b| {
            let value = b.const_i32(spec(MockType::I32), value)?;
            b.emit_no_defs(SsaOp::StoreIndirect {
                addr: pointer,
                value,
                value_type: MockType::I32,
                access: AccessQualifiers::default(),
            })?;
            b.jump(3)
        })
        .unwrap();
    }
    let phi = b
        .in_block(3, |b| {
            let phi = b.phi(spec(MockType::Ptr), [(1, pointer), (2, pointer)])?;
            b.ret(None)?;
            Ok(phi)
        })
        .unwrap();
    (
        b.finish_verified(VerifyLevel::Full).unwrap(),
        phi,
        condition,
    )
}
#[test]
fn actual_diamond_edges_capture_after_phi_and_before_join() {
    let (body, pointer, condition) = diamond();
    let plan = EdgeObservationPlan::new(
        &[
            request(&body, 1, 3, pointer, 3),
            request(&body, 2, 3, pointer, 3),
        ],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 6)
        .run();
    assert!(facts.is_complete());
    assert!(!facts.edge_output_exhausted());
    assert!(facts.constant(condition).is_none());
    let edges = facts.take_edges();
    assert_eq!(edges.len(), 2);
    assert_eq!(edges[0].prefixes[0].bytes, [0x41, 0, 0xff]);
    assert_eq!(edges[1].prefixes[0].bytes, [0x42, 0, 0xff]);
    assert_eq!(edges[0].prefixes[0].pointer, pointer);
    assert!(
        facts
            .image_writes_within(16)
            .unwrap()
            .iter()
            .all(|range| range.address != 0x1000)
    );
}
#[test]
fn output_overflow_preserves_execution_and_is_explicit() {
    let (body, pointer, _) = diamond();
    let plan = EdgeObservationPlan::new(
        &[
            request(&body, 1, 3, pointer, 3),
            request(&body, 2, 3, pointer, 3),
        ],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let baseline = StaticExecution::new(&body, &context).run();
    for cap in [0, 5, 6] {
        let mut facts = StaticExecution::new(&body, &context)
            .with_edge_observations(&plan, cap)
            .run();
        assert!(facts.is_complete());
        assert_eq!(facts.usage().steps, baseline.usage().steps);
        assert_eq!(facts.edge_output_exhausted(), cap < 6);
        assert!(
            facts
                .take_edges()
                .iter()
                .flat_map(|edge| &edge.prefixes)
                .map(|p| p.bytes.len())
                .sum::<usize>()
                <= cap
        );
    }
}
#[test]
fn duplicate_requests_use_maximum_and_input_bounds_precede_normalization() {
    let (body, pointer, _) = diamond();
    let requests = [
        request(&body, 1, 3, pointer, 1),
        request(&body, 1, 3, pointer, 3),
    ];
    assert!(
        EdgeObservationPlan::new(
            &requests,
            MACHINE.ptr_size,
            EdgeRequestLimits {
                sites: 1,
                requests: 1
            }
        )
        .is_err()
    );
    assert!(
        EdgeObservationPlan::new(
            &requests,
            MACHINE.ptr_size,
            EdgeRequestLimits {
                sites: 0,
                requests: 2
            }
        )
        .is_err()
    );
    let plan = EdgeObservationPlan::new(
        &requests,
        MACHINE.ptr_size,
        EdgeRequestLimits {
            sites: 1,
            requests: 2,
        },
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 3)
        .run();
    let edges = facts.take_edges();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].prefixes.len(), 1);
    assert_eq!(edges[0].prefixes[0].maximum_bytes, 3);
    assert_eq!(edges[0].prefixes[0].bytes, [0x41, 0, 0xff]);
}
#[test]
fn invalid_source_pointer_and_future_definition_requests_refuse() {
    let (body, pointer, condition) = diamond();
    assert!(EdgeSite::of(&body, 1, 2).is_err());
    assert!(EdgeSite::of(&body, 3, 0).is_err());
    let mut invalid = request(&body, 1, 3, pointer, 3);
    invalid.site.instruction = usize::MAX;
    assert!(EdgeObservationPlan::new(&[invalid], MACHINE.ptr_size, limits()).is_err());
    assert!(
        EdgeObservationPlan::new(
            &[request(&body, 1, 3, condition, 3)],
            MACHINE.ptr_size,
            limits()
        )
        .is_err()
    );
    // The join phi is not available on an earlier branch edge.
    assert!(
        EdgeObservationPlan::new(
            &[request(&body, 0, 1, pointer, 3)],
            MACHINE.ptr_size,
            limits()
        )
        .is_err()
    );
}
#[test]
fn cancellation_and_incomplete_paths_withhold_edge_records() {
    let (body, pointer, _) = diamond();
    let plan = EdgeObservationPlan::new(
        &[request(&body, 1, 3, pointer, 3)],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let cancel = || true;
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 3)
        .with_cancellation(&cancel)
        .run();
    assert_eq!(facts.completion(), Completion::Cancelled);
    assert!(facts.take_edges().is_empty());
    let context = context.with_budget(
        context
            .budget()
            .with_limit(crate::analysis::budget::Budget::ExecutionSteps, 1),
    );
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(!facts.is_complete());
    assert!(facts.take_edges().is_empty());
}
#[test]
fn bounded_reader_distinguishes_unknown_end_from_output_limit() {
    let mut memory = Memory::default();
    let address = Address::new(
        crate::analysis::execution::Root::Stack,
        -8,
        MACHINE.ptr_size,
    );
    memory.store_bytes(address, &[0x41, 0, 0xff]);
    assert_eq!(
        memory.known_run_within(address, 11, 3, None),
        Some(vec![0x41, 0, 0xff])
    );
    assert_eq!(memory.known_run_within(address, 11, 2, None), None);
    assert_eq!(memory.known_run_within(address, 0, 0, None), Some(vec![]));
    let mut bytes = vec![0x41, 0, 0xff];
    memory.store_bytes(address.moved(2, MACHINE.ptr_size).unwrap(), &[0x55]);
    memory.intersect_known_run(address, &mut bytes, None);
    assert_eq!(bytes, [0x41, 0]);
}
#[test]
fn repeated_visits_never_strengthen_unknown_or_conflicting_prefixes() {
    let (body, pointer, _) = diamond();
    let plan = EdgeObservationPlan::new(
        &[request(&body, 1, 3, pointer, 3)],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE);
    for known_first in [false, true] {
        let address = Address::new(
            crate::analysis::execution::Root::Stack,
            -8,
            MACHINE.ptr_size,
        );
        let mut memory = Memory::default();
        memory.store_bytes(address, &[0x41, 0, 0xff]);
        let frame = Frame::enter_at(&body, 0, 0, address, vec![], MACHINE.ptr_size);
        let mut state = State::new(frame);
        *state.memory_mut() = memory;
        let mut records = EdgeRecords::default();
        let mut allowance = EdgeAllowance::new(3);
        for known in [known_first, !known_first, true] {
            state.frame_mut().unwrap().define(
                pointer,
                if known {
                    Value::Addr(address)
                } else {
                    Value::Unknown
                },
                1,
            );
            records.observe(&plan, &body, (1, 3), &state, &context, &mut allowance);
        }
        let records = records.into_observations();
        assert_eq!(records.len(), 1);
        assert!(records[0].prefixes[0].bytes.is_empty());
        assert!(records[0].prefixes[0].address.is_none());
        assert!(!allowance.exhausted);
    }
}

struct Host<'a>(&'a SsaFunction<MockTarget>);
impl ExecutionHost<MockTarget> for Host<'_> {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 1).then_some(self.0)
    }
}
#[test]
fn entered_call_observations_keep_the_actual_producer_domain() {
    let (body, pointer, condition) = diamond();
    let mut caller = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    caller
        .in_block(0, |b| {
            let zero = b.const_i32(spec(MockType::I32), 0)?;
            b.emit_no_defs(CallData::new(None, Callee::Direct(1), vec![zero]).into())?;
            b.ret(None)
        })
        .unwrap();
    let caller = caller.finish_verified(VerifyLevel::Full).unwrap();
    let plan = EdgeObservationPlan::with_scalars(
        &[
            request(&body, 1, 3, pointer, 3),
            request(&body, 2, 3, pointer, 3),
        ],
        &[EdgeScalarRequest {
            body: &body,
            site: EdgeSite::of(&body, 1, 3).unwrap(),
            value: condition,
        }],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let host = Host(&body);
    let mut facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_transitive_call_observations(1)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(facts.is_complete());
    assert!(facts.take_edges().is_empty());
    let mut calls = facts.take_invocations();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].parent, None);
    let edges = calls[0].facts.take_edges();
    assert_eq!(edges.len(), 1); // The untaken second branch has no visit record.
    assert_eq!(edges[0].prefixes[0].bytes, [0x41, 0, 0xff]);
    assert_eq!(edges[0].scalars[0].constant, Some(ConstValue::I32(0)));
}

#[test]
fn actual_loop_repeats_meet_and_release_discarded_payload() {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let (pointer, zero) = b
        .in_block(0, |b| {
            let pointer = b.const_value(spec(MockType::Ptr), ConstValue::NativeUInt(0x1000))?;
            let zero = b.const_i32(spec(MockType::I32), 0)?;
            b.jump(1)?;
            Ok((pointer, zero))
        })
        .unwrap();
    let i = b
        .in_block(1, |b| {
            let i = b.empty_phi(spec(MockType::I32))?;
            let base = b.const_i32(spec(MockType::I32), 0x00ff0041)?;
            let value = b.add(spec(MockType::I32), base, i)?;
            b.emit_no_defs(SsaOp::StoreIndirect {
                addr: pointer,
                value,
                value_type: MockType::I32,
                access: AccessQualifiers::default(),
            })?;
            let limit = b.const_i32(spec(MockType::I32), 2)?;
            b.branch_cmp(i, limit, CmpKind::Lt, false, 2, 3)?;
            Ok(i)
        })
        .unwrap();
    let next = b
        .in_block(2, |b| {
            let one = b.const_i32(spec(MockType::I32), 1)?;
            let next = b.add(spec(MockType::I32), i, one)?;
            b.jump(1)?;
            Ok(next)
        })
        .unwrap();
    b.in_block(3, |b| b.ret(None)).unwrap();
    b.add_phi_operand(1, i, 0, zero).unwrap();
    b.add_phi_operand(1, i, 2, next).unwrap();
    let body = b.finish_verified(VerifyLevel::Full).unwrap();
    let plan = EdgeObservationPlan::new(
        &[
            request(&body, 1, 2, pointer, 3),
            request(&body, 1, 3, pointer, 3),
        ],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(!facts.edge_output_exhausted());
    let edges = facts.take_edges();
    assert_eq!(edges.len(), 2);
    assert!(edges[0].prefixes[0].bytes.is_empty());
    assert_eq!(edges[1].prefixes[0].bytes, [0x43, 0, 0xff]);
}

#[test]
fn equal_target_unknown_branch_has_one_edge_without_condition_refinement() {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = b.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let pointer = b
        .in_block(0, |b| {
            let pointer = b.const_value(spec(MockType::Ptr), ConstValue::NativeUInt(0x1000))?;
            let value = b.const_i32(spec(MockType::I32), 0x00ff0041)?;
            b.emit_no_defs(SsaOp::StoreIndirect {
                addr: pointer,
                value,
                value_type: MockType::I32,
                access: AccessQualifiers::default(),
            })?;
            let zero = b.const_i32(spec(MockType::I32), 0)?;
            b.branch_cmp(condition, zero, CmpKind::Eq, false, 1, 1)?;
            Ok(pointer)
        })
        .unwrap();
    b.in_block(1, |b| b.ret(Some(condition))).unwrap();
    let body = b.finish_verified(VerifyLevel::Full).unwrap();
    let plan = EdgeObservationPlan::with_scalars(
        &[request(&body, 0, 1, pointer, 3)],
        &[EdgeScalarRequest {
            body: &body,
            site: EdgeSite::of(&body, 0, 1).unwrap(),
            value: condition,
        }],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(facts.is_complete());
    assert!(facts.constant(condition).is_none());
    assert!(facts.returned().and_then(Value::constant).is_none());
    let edges = facts.take_edges();
    assert_eq!(edges.len(), 1);
    assert!(edges[0].scalars[0].constant.is_none());
}

#[test]
fn a_plan_for_an_equal_but_different_body_does_not_bind_by_local_coordinates() {
    let (body, pointer, _) = diamond();
    let (other, _, _) = diamond();
    let plan = EdgeObservationPlan::new(
        &[request(&body, 1, 3, pointer, 3)],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&other, &context)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(facts.is_complete());
    assert!(facts.take_edges().is_empty());
}

#[test]
fn target_phi_chooses_distinct_pointer_payloads() {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = b.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let (left, right) = b
        .in_block(0, |b| {
            let left = b.const_value(spec(MockType::Ptr), ConstValue::NativeUInt(0x1000))?;
            let right = b.const_value(spec(MockType::Ptr), ConstValue::NativeUInt(0x1008))?;
            for (addr, word) in [(left, 0x00ff0041), (right, 0x00ff0042)] {
                let value = b.const_i32(spec(MockType::I32), word)?;
                b.emit_no_defs(SsaOp::StoreIndirect {
                    addr,
                    value,
                    value_type: MockType::I32,
                    access: AccessQualifiers::default(),
                })?;
            }
            b.branch(condition, 1, 2)?;
            Ok((left, right))
        })
        .unwrap();
    b.in_block(1, |b| b.jump(3)).unwrap();
    b.in_block(2, |b| b.jump(3)).unwrap();
    let pointer = b
        .in_block(3, |b| {
            let pointer = b.phi(spec(MockType::Ptr), [(1, left), (2, right)])?;
            b.ret(Some(pointer))?;
            Ok(pointer)
        })
        .unwrap();
    let body = b.finish_verified(VerifyLevel::Full).unwrap();
    let plan = EdgeObservationPlan::new(
        &[
            request(&body, 1, 3, pointer, 3),
            request(&body, 2, 3, pointer, 3),
        ],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 6)
        .run();
    assert!(facts.is_complete());
    let edges = facts.take_edges();
    assert_eq!(
        edges[0].prefixes[0].address.unwrap().image_address(),
        Some(0x1000)
    );
    assert_eq!(
        edges[1].prefixes[0].address.unwrap().image_address(),
        Some(0x1008)
    );
    assert_eq!(edges[0].prefixes[0].bytes, [0x41, 0, 0xff]);
    assert_eq!(edges[1].prefixes[0].bytes, [0x42, 0, 0xff]);
}

#[test]
fn an_incomplete_sibling_withholds_a_previously_observed_edge() {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = b.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let pointer = b
        .in_block(0, |b| {
            let pointer = b.const_value(spec(MockType::Ptr), ConstValue::NativeUInt(0x1000))?;
            b.branch(condition, 1, 2)?;
            Ok(pointer)
        })
        .unwrap();
    b.in_block(1, |b| {
        let value = b.const_i32(spec(MockType::I32), 0x00ff0041)?;
        b.emit_no_defs(SsaOp::StoreIndirect {
            addr: pointer,
            value,
            value_type: MockType::I32,
            access: AccessQualifiers::default(),
        })?;
        b.jump(3)
    })
    .unwrap();
    b.in_block(2, |b| {
        b.emit_no_defs(SsaOp::IndirectBranch {
            target: pointer,
            resolved_targets: vec![],
            state: Box::default(),
        })
    })
    .unwrap();
    b.in_block(3, |b| b.ret(None)).unwrap();
    let body = b.finish_verified(VerifyLevel::Full).unwrap();
    let plan = EdgeObservationPlan::new(
        &[request(&body, 1, 3, pointer, 3)],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(matches!(facts.completion(), Completion::Stuck { .. }));
    assert!(facts.take_edges().is_empty());
    assert!(EdgeSite::of(&body, 2, 3).is_err());
}

struct TailHost<'a> {
    leaf: &'a SsaFunction<MockTarget>,
    wrapper: &'a SsaFunction<MockTarget>,
}
impl ExecutionHost<MockTarget> for TailHost<'_> {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        match method {
            1 => Some(self.leaf),
            2 => Some(self.wrapper),
            _ => None,
        }
    }
}
#[test]
fn tail_callee_edges_do_not_populate_retired_caller_observations() {
    let (body, pointer, condition) = diamond();
    let mut wrapper = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    wrapper
        .in_block(0, |b| {
            let zero = b.const_i32(spec(MockType::I32), 0)?;
            let mut call = CallData::new(None, Callee::Direct(1), vec![zero]);
            call.tail = true;
            b.emit_no_defs(call.into())?;
            b.ret(None)
        })
        .unwrap();
    let wrapper = wrapper.finish_verified(VerifyLevel::Full).unwrap();
    let mut root = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    root.in_block(0, |b| {
        b.emit_no_defs(CallData::new(None, Callee::Direct(2), vec![]).into())?;
        b.ret(None)
    })
    .unwrap();
    let root = root.finish_verified(VerifyLevel::Full).unwrap();
    let host = TailHost {
        leaf: &body,
        wrapper: &wrapper,
    };
    let plan = EdgeObservationPlan::with_scalars(
        &[request(&body, 1, 3, pointer, 3)],
        &[EdgeScalarRequest {
            body: &body,
            site: EdgeSite::of(&body, 1, 3).unwrap(),
            value: condition,
        }],
        MACHINE.ptr_size,
        limits(),
    )
    .unwrap();
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_transitive_call_observations(2)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(facts.take_edges().is_empty());
    let mut calls = facts.take_invocations();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].parent, None);
    assert_eq!(calls[1].parent, Some(0));
    assert!(calls[0].facts.take_edges().is_empty());
    let edges = calls[1].facts.take_edges();
    assert_eq!(edges[0].prefixes[0].bytes, [0x41, 0, 0xff]);
    assert_eq!(edges[0].scalars[0].constant, Some(ConstValue::I32(0)));
}

/// **A raise is no edge.** Block 0 divides by the argument, which may
/// raise, then defines five and jumps to block 1, which is also the handler
/// guarding it. The raise enters block 1 before the five is defined, from the
/// middle of block 0: what it holds is no value on the edge from block 0 to
/// block 1, which the jump alone takes, holding five.
#[test]
fn a_raise_is_no_edge_its_block_takes() {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let divisor = b.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let five = b
        .in_block(0, |b| {
            let hundred = b.const_i32(spec(MockType::I32), 100)?;
            b.div(spec(MockType::I32), hundred, divisor, false)?;
            let five = b.const_i32(spec(MockType::I32), 5)?;
            b.jump(1)?;
            Ok(five)
        })
        .unwrap();
    b.in_block(1, |b| b.ret(None)).unwrap();
    let mut body = b.finish().unwrap();
    body.set_exception_handlers(vec![SsaExceptionHandler {
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
    let requests = [EdgeScalarRequest {
        body: &body,
        site: EdgeSite::of(&body, 0, 1).unwrap(),
        value: five,
    }];
    let plan =
        EdgeObservationPlan::with_scalars(&[], &requests, MACHINE.ptr_size, limits()).unwrap();
    let context = AnalysisContext::new(MACHINE);
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 0)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let edges = facts.take_edges();
    assert_eq!(edges.len(), 1);
    assert_eq!(
        edges[0]
            .scalars
            .first()
            .and_then(|scalar| scalar.constant.clone()),
        Some(ConstValue::I32(5))
    );
}

#[test]
fn borrowed_plan_preserves_source_thread_transfer_capability() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<EdgeObservationPlan<'_, MockTarget>>();
}

/// An edge's scalars are what holds as control passes along it: a phi's
/// operand from that edge, and a value the branch that led there proved,
/// zero on the edge only the equal side reaches and nothing on the other.
/// Neither is a fact about the value's definition.
#[test]
fn edge_scalars_hold_on_their_edge_and_are_not_definition_facts() {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = b.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    b.in_block(0, |b| {
        let zero = b.const_i32(spec(MockType::I32), 0)?;
        b.branch_cmp(condition, zero, CmpKind::Eq, false, 1, 2)
    })
    .unwrap();
    let a = b
        .in_block(1, |b| {
            let value = b.const_i32(spec(MockType::I32), 11)?;
            b.jump(3)?;
            Ok(value)
        })
        .unwrap();
    let z = b
        .in_block(2, |b| {
            let value = b.const_i32(spec(MockType::I32), 22)?;
            b.jump(3)?;
            Ok(value)
        })
        .unwrap();
    let phi = b
        .in_block(3, |b| {
            let value = b.phi(spec(MockType::I32), [(1, a), (2, z)])?;
            b.ret(Some(value))?;
            Ok(value)
        })
        .unwrap();
    let body = b.finish_verified(VerifyLevel::Full).unwrap();
    let requests: Vec<_> = [1, 2]
        .into_iter()
        .flat_map(|from| {
            [phi, condition].map(|value| EdgeScalarRequest {
                body: &body,
                site: EdgeSite::of(&body, from, 3).unwrap(),
                value,
            })
        })
        .collect();
    let plan =
        EdgeObservationPlan::with_scalars(&[], &requests, MACHINE.ptr_size, limits()).unwrap();
    let context = AnalysisContext::new(MACHINE);
    let baseline = StaticExecution::new(&body, &context).run();
    let mut facts = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 0)
        .run();
    assert!(facts.is_complete());
    assert_eq!(facts.usage().steps, baseline.usage().steps);
    assert!(facts.constant(phi).is_none());
    assert!(facts.constant(condition).is_none());
    assert!(!facts.edge_output_exhausted());
    let edges = facts.take_edges();
    assert_eq!(edges.len(), 2);
    for (edge, expected, proved) in edges
        .iter()
        .zip([(11, Some(ConstValue::I32(0))), (22, None)])
        .map(|(edge, (expected, proved))| (edge, expected, proved))
    {
        assert_eq!(
            edge.scalars
                .iter()
                .find(|s| s.value == phi)
                .unwrap()
                .constant,
            Some(ConstValue::I32(expected))
        );
        assert_eq!(
            edge.scalars
                .iter()
                .find(|s| s.value == condition)
                .unwrap()
                .constant,
            proved
        );
    }
    let cancelled = || true;
    let mut cancelled = StaticExecution::new(&body, &context)
        .with_edge_observations(&plan, 0)
        .with_cancellation(&cancelled)
        .run();
    assert!(!cancelled.is_complete());
    assert!(cancelled.take_edges().is_empty());
}

#[test]
fn scalar_unknown_visits_are_sticky_and_wide_payload_release_shares_byte_allowance() {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let scalar = b.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::U128,
    );
    let pointer = b.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    b.in_block(0, |b| b.jump(1)).unwrap();
    b.in_block(1, |b| b.jump(2)).unwrap();
    b.in_block(2, |b| b.ret(None)).unwrap();
    let body = b.finish_verified(VerifyLevel::Full).unwrap();
    let scalars = [EdgeScalarRequest {
        body: &body,
        site: EdgeSite::of(&body, 0, 1).unwrap(),
        value: scalar,
    }];
    let bytes = [request(&body, 1, 2, pointer, 3)];
    let plan =
        EdgeObservationPlan::with_scalars(&bytes, &scalars, MACHINE.ptr_size, limits()).unwrap();
    assert!(
        EdgeObservationPlan::with_scalars(
            &bytes,
            &scalars,
            MACHINE.ptr_size,
            EdgeRequestLimits {
                sites: 2,
                requests: 1
            }
        )
        .is_err()
    );
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let address = Address::new(Root::Symbol, 0x1000, MACHINE.ptr_size);
    for maximum in [15, 16] {
        for schedule in [
            [Some(1u128 << 100), None, Some(1u128 << 100)],
            [None, Some(1u128 << 100), Some(1u128 << 100)],
            [Some(1u128 << 100), Some(1u128 << 101), Some(1u128 << 100)],
        ] {
            let frame = Frame::enter_at(&body, 0, 0, address, vec![], MACHINE.ptr_size);
            let mut state = State::new(frame);
            state
                .frame_mut()
                .unwrap()
                .define(pointer, Value::Addr(address), 1);
            state.memory_mut().store_bytes(address, &[0x41, 0, 0xff]);
            let mut records = EdgeRecords::default();
            let mut allowance = EdgeAllowance::new(maximum);
            for (visit, pattern) in schedule.into_iter().enumerate() {
                let value = pattern.map_or(Value::Unknown, |bits| {
                    Value::Const(ConstValue::of_lane(128, bits).unwrap())
                });
                state.frame_mut().unwrap().define(scalar, value, 1);
                records.observe(&plan, &body, (0, 1), &state, &context, &mut allowance);
                let expected = usize::from(visit == 0 && pattern.is_some() && maximum == 16) * 16;
                assert_eq!(records.payload_bytes_within(16, &|| false), Ok(expected));
                if expected == 16 {
                    assert_eq!(
                        records.payload_bytes_within(15, &|| false),
                        Err(ObservationSizeRefusal::OutputLimit)
                    );
                }
                // Cancellation must still visit an unknown scalar marker.
                let checkpoints = Cell::new(0usize);
                assert_eq!(
                    records.payload_bytes_within(16, &|| {
                        checkpoints.set(checkpoints.get() + 1);
                        checkpoints.get() == 2
                    }),
                    Err(ObservationSizeRefusal::Cancelled)
                );
            }
            assert_eq!(allowance.exhausted, schedule[0].is_some() && maximum == 15);
            records.observe(&plan, &body, (1, 2), &state, &context, &mut allowance);
            let expected = if allowance.exhausted { 0 } else { 3 };
            assert_eq!(records.payload_bytes_within(3, &|| false), Ok(expected));
            if expected == 3 {
                assert_eq!(
                    records.payload_bytes_within(2, &|| false),
                    Err(ObservationSizeRefusal::OutputLimit)
                );
            }
            let records = records.into_observations();
            assert!(records[0].scalars[0].constant.is_none());
            if allowance.exhausted {
                assert!(records[1].prefixes[0].bytes.is_empty());
            } else {
                assert_eq!(records[1].prefixes[0].bytes, [0x41, 0, 0xff]);
                assert_eq!(allowance.remaining, maximum - 3);
            }
        }
    }
}
