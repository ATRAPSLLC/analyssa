//! Configured executor limits and cancellation at execution boundaries.

use std::{
    cell::Cell,
    process,
    sync::atomic::{AtomicUsize, Ordering},
};

use analyssa::{
    PointerSize,
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            CallModel, CallSite, CallStack, Completion, ExecutionHost, ModelCall,
            ParameterBindings, StaticExecution,
        },
        image::{ImageData, ImageRegion},
    },
    ir::{
        Callee, ConstValue, SsaDefSpec, SsaFunction, SsaFunctionBuilder, SsaOp,
        variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
    },
    target::{Endianness, TypeLayout},
    testing::{MACHINE, MockTarget, MockType},
};

/// Stops when a constructed fixture cannot be built.
fn abort<E, R>(_: E) -> R {
    process::abort()
}

/// Counts whether execution actually reaches the consumer.
struct Consumer(Cell<usize>);
impl CallModel<MockTarget> for Consumer {
    fn call(&self, _: &mut ModelCall<'_, MockTarget>) {
        self.0.set(self.0.get().saturating_add(1));
    }
}

/// Builds a long graph with the consumer either at entry or after all its edges.
fn graph_chain(nodes: usize, consumer_at_entry: bool) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    for node in 0..nodes {
        builder
            .in_block(node, |b| {
                if node == 0 && consumer_at_entry {
                    b.call_void(analyssa::ir::Callee::Direct(3), vec![])?;
                }
                let next = node.saturating_add(1);
                if next < nodes {
                    b.jump(next)
                } else {
                    if !consumer_at_entry {
                        b.call_void(analyssa::ir::Callee::Direct(3), vec![])?;
                    }
                    b.ret(None)
                }
            })
            .unwrap_or_else(abort);
    }
    builder.finish().unwrap_or_else(abort)
}

/// Resolves an optional large callee and one observable consumer.
struct GraphHost<'a> {
    callee: Option<&'a SsaFunction<MockTarget>>,
    consumer: &'a Consumer,
}
impl ExecutionHost<MockTarget> for GraphHost<'_> {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        if *method == 1 { self.callee } else { None }
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 3).then_some(self.consumer)
    }
}

/// The configured instruction limit stops root and callee execution before a later consumer.
#[test]
fn executor_step_limit_precedes_later_root_and_callee_consumers() {
    let large = graph_chain(2048, false);
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            b.call_void(analyssa::ir::Callee::Direct(1), vec![])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let caller = builder.finish().unwrap_or_else(abort);
    for callee in [false, true] {
        for limited in [false, true] {
            let consumer = Consumer(Cell::new(0));
            let host = GraphHost {
                callee: callee.then_some(&large),
                consumer: &consumer,
            };
            let ssa = if callee { &caller } else { &large };
            let context = AnalysisContext::new(MACHINE).with_budget(if limited {
                WorkBudget::DEFAULT.with_limit(Budget::ExecutionSteps, 512)
            } else {
                WorkBudget::DEFAULT
            });
            let facts = StaticExecution::new(ssa, &context).with_host(&host).run();
            assert_eq!(consumer.0.get(), usize::from(!limited));
            if limited {
                assert!(
                    matches!(facts.completion(),Completion::Exhausted(exhausted) if exhausted.budget==Budget::ExecutionSteps)
                );
                assert_eq!(facts.usage().steps, 512);
                assert!(
                    matches!(facts.completion(), Completion::Exhausted(exhaustion)
                    if exhaustion.limit == 512 && exhaustion.demand == 513)
                );
                assert!(facts.calls().is_empty());
            } else {
                assert!(facts.is_complete());
            }
        }
    }
}

/// Refused control transitions and simultaneous phi batches retain demand without consuming it.
#[test]
fn executor_step_usage_never_charges_a_refused_control_or_phi_batch() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let incoming = builder
        .in_block(0, |b| {
            let incoming = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(42))?;
            b.jump(1)?;
            Ok(incoming)
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let mut result = incoming;
            for _ in 0..8 {
                result = b.phi(SsaDefSpec::tmp(MockType::I64), [(0, incoming)])?;
            }
            b.call_void(Callee::Direct(3), vec![])?;
            b.ret(Some(result))
        })
        .unwrap_or_else(abort);
    let phis = builder.finish().unwrap_or_else(abort);
    assert!(SsaVerifier::new(&phis).verify(VerifyLevel::Full).is_empty());
    for ssa in [&graph_chain(4, false), &phis] {
        let consumer = Consumer(Cell::new(0));
        let host = GraphHost {
            callee: None,
            consumer: &consumer,
        };
        let context = AnalysisContext::new(MACHINE);
        let complete = StaticExecution::new(ssa, &context).with_host(&host).run();
        assert!(complete.is_complete());
        let steps = complete.usage().steps;
        for limit in 0..=steps.saturating_add(1) {
            let context = AnalysisContext::new(MACHINE)
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionSteps, limit));
            let facts = StaticExecution::new(ssa, &context).with_host(&host).run();
            assert!(facts.usage().steps <= limit, "{limit}: {:?}", facts.usage());
            assert_eq!(facts.is_complete(), limit >= steps);
            if limit < steps {
                assert!(
                    matches!(facts.completion(), Completion::Exhausted(exhaustion)
                    if exhaustion.budget == Budget::ExecutionSteps
                        && exhaustion.limit == limit && exhaustion.demand > limit)
                );
                assert!(facts.calls().is_empty());
            }
        }
    }
}

/// Cancellation during execution stops before later observable body calls.
#[test]
fn executor_cancellation_precedes_later_body_consumers() {
    let ssa = graph_chain(2048, false);
    for cancelled in [false, true] {
        let consumer = Consumer(Cell::new(0));
        let host = GraphHost {
            callee: None,
            consumer: &consumer,
        };
        let polls = Cell::new(0usize);
        let probe = || {
            polls.set(polls.get().saturating_add(1));
            cancelled && polls.get() > 16
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&host)
            .with_cancellation(&probe)
            .run();
        assert_eq!(consumer.0.get(), usize::from(!cancelled));
        if cancelled {
            assert_eq!(facts.completion(), Completion::Cancelled);
            assert!(facts.usage().steps < 2048);
            assert!(facts.calls().is_empty());
        } else {
            assert!(facts.is_complete());
        }
    }
}

/// An already-cancelled invocation executes no instructions.
#[test]
fn entry_cancellation_precedes_execution() {
    let ssa = graph_chain(2048, true);
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context)
        .with_cancellation(&|| true)
        .run();
    assert_eq!(facts.completion(), Completion::Cancelled);
    assert_eq!(facts.usage().steps, 0);
}

/// An owned literal must fit the configured buffer limit before any later model runs.
#[test]
fn literal_payload_respects_configured_buffer_limit() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let value = b.const_value(
                SsaDefSpec::tmp(MockType::I64),
                ConstValue::DecryptedString("a".repeat(4096).into_boxed_str()),
            )?;
            b.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::Copy {
                dest,
                src: value,
            })?;
            b.call_void(analyssa::ir::Callee::Direct(3), vec![])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    for limit in [0, 4095, 4096, 4097] {
        let limited = limit < 4096;
        let consumer = Consumer(Cell::new(0));
        let host = GraphHost {
            callee: None,
            consumer: &consumer,
        };
        let context = AnalysisContext::new(MACHINE)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, limit));
        let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
        assert_eq!(consumer.0.get(), usize::from(!limited));
        if limited {
            assert!(
                matches!(facts.completion(), Completion::Exhausted(exhausted) if exhausted.budget == Budget::ExecutionBufferBytes)
            );
            assert_eq!(facts.calls().len(), 0);
        } else {
            assert!(facts.is_complete());
        }
    }
}

/// An oracle that permits only the single protection query of an in-limit store.
struct StoreImage {
    permitted: bool,
    queries: AtomicUsize,
}

impl ImageData for StoreImage {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, _: u64) -> Option<ImageRegion> {
        None
    }
    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }
    fn number_offset(&self) -> Option<i64> {
        assert!(
            self.permitted,
            "oversized store resolved a numeric image address"
        );
        Some(0)
    }
    fn read_only_overlap(&self, _: u64, _: u64) -> bool {
        self.queries.fetch_add(1, Ordering::Relaxed);
        assert!(self.permitted, "oversized store queried image protection");
        false
    }
}

/// Unknown scalar stores honor exact configured boundaries before queries, effects or later calls.
#[test]
fn scalar_store_extent_limit_precedes_image_queries() {
    for (ty, bytes) in [
        (MockType::Bool, 1usize),
        (MockType::U8, 1),
        (MockType::U64, 8),
        (MockType::V2I256, 64),
    ] {
        for pointer in [0, 1, 2] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
            let value = builder.function_mut().create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                ty,
            );
            let unknown_address = builder.function_mut().create_variable(
                VariableOrigin::Argument(1),
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            let mut prefix = None;
            builder
                .in_block(0, |b| {
                    prefix = Some(b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?);
                    let address = match pointer {
                        0 => b.const_value(
                            SsaDefSpec::tmp(MockType::Ptr),
                            ConstValue::Symbol(0x1000),
                        )?,
                        1 => {
                            b.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::U64(0x1000))?
                        }
                        _ => unknown_address,
                    };
                    b.store_indirect(address, value, ty)?;
                    b.call_void(Callee::Direct(3), vec![])?;
                    b.ret(None)
                })
                .unwrap_or_else(abort);
            let ssa = builder.finish().unwrap_or_else(abort);
            assert!(
                SsaVerifier::new(&ssa)
                    .verify(VerifyLevel::Standard)
                    .is_empty()
            );
            for limit in [0, bytes.saturating_sub(1), bytes, bytes.saturating_add(1)] {
                let limited = bytes > limit;
                let image = StoreImage {
                    permitted: !limited,
                    queries: AtomicUsize::new(0),
                };
                let consumer = Consumer(Cell::new(0));
                let host = GraphHost {
                    callee: None,
                    consumer: &consumer,
                };
                let context = AnalysisContext::new(MACHINE)
                    .with_image(&image)
                    .with_budget(
                        WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, limit),
                    );
                let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
                assert_eq!(
                    image.queries.load(Ordering::Relaxed),
                    usize::from(!limited && pointer != 2)
                );
                assert_eq!(consumer.0.get(), usize::from(!limited));
                if limited {
                    assert!(
                        matches!(facts.completion(), Completion::Exhausted(exhaustion)
                        if exhaustion.budget == Budget::ExecutionBufferBytes
                        && exhaustion.limit == limit && exhaustion.demand == bytes)
                    );
                    assert!(
                        facts
                            .value(prefix.unwrap_or_else(|| process::abort()))
                            .is_none()
                    );
                    assert!(facts.calls().is_empty());
                } else {
                    assert!(facts.is_complete());
                    assert_eq!(
                        facts
                            .value(prefix.unwrap_or_else(|| process::abort()))
                            .and_then(|value| value.constant()),
                        Some(&ConstValue::U8(7))
                    );
                }
            }
        }
    }
}

/// Stages a declared extent, ignores failure, then tries a small write before callee entry.
struct StagingHost<'a> {
    body: &'a SsaFunction<MockTarget>,
    stack: SsaVarId,
    ty: MockType,
    results: Cell<(bool, bool)>,
    consumer: &'a Consumer,
}

impl ExecutionHost<MockTarget> for StagingHost<'_> {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 1).then_some(self.body)
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 3).then_some(self.consumer)
    }
    fn call_stack(&self, _: CallSite<'_, MockTarget>) -> Option<CallStack> {
        Some(CallStack {
            stack_pointer: Some(self.stack),
            adjustment: 0,
        })
    }
    fn parameters(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        _: &SsaFunction<MockTarget>,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        let layout = |ty| TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64);
        let first = bindings.stage_stack(0, 0, layout(self.ty));
        let second = bindings.stage_stack(0, 32, layout(MockType::U8));
        self.results.set((first, second));
    }
}

/// Ignored configured staging exhaustion stays latched and withholds the whole invocation.
#[test]
fn staged_store_extent_limit_is_latched_before_callee_entry() {
    let mut caller = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = caller.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    caller
        .function_mut()
        .variable_mut(stack)
        .unwrap_or_else(|| process::abort())
        .set_role(VariableRole::EntryStackPointer);
    let mut prefix = None;
    caller
        .in_block(0, |b| {
            let value = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            prefix = Some(value);
            b.call_void(Callee::Direct(1), vec![value])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let caller = caller.finish().unwrap_or_else(abort);
    assert!(
        SsaVerifier::new(&caller)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    let mut callee = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    callee
        .in_block(0, |b| {
            b.call_void(Callee::Direct(3), vec![])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let callee = callee.finish().unwrap_or_else(abort);
    assert!(
        SsaVerifier::new(&callee)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    for (ty, bytes) in [
        (MockType::U8, 1usize),
        (MockType::U128, 16),
        (MockType::V2I256, 64),
    ] {
        let consumer = Consumer(Cell::new(0));
        let host = StagingHost {
            body: &callee,
            stack,
            ty,
            results: Cell::new((false, false)),
            consumer: &consumer,
        };
        let context = AnalysisContext::new(MACHINE)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, 16));
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        let limited = bytes > 16;
        assert_eq!(host.results.get(), (!limited, !limited));
        assert_eq!(consumer.0.get(), usize::from(!limited));
        if limited {
            assert!(
                matches!(facts.completion(), Completion::Exhausted(exhaustion)
                if exhaustion.budget == Budget::ExecutionBufferBytes
                && exhaustion.limit == 16 && exhaustion.demand == bytes)
            );
            assert!(
                facts
                    .value(prefix.unwrap_or_else(|| process::abort()))
                    .is_none()
            );
            assert!(facts.calls().is_empty());
        } else {
            assert!(facts.is_complete());
        }
    }
}
