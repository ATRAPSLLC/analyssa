//! Execution regressions for typed lane computations that execute no element.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    PointerSize,
    analysis::{
        budget::{Budget, Exhaustion, WorkBudget},
        context::AnalysisContext,
        execution::{
            Address, Completion, ExecutionHost, InstructionSourceId, InstructionSources,
            MemoryRefusal, MemoryWriteGuard, Root, StaticExecution, Value,
            frame::Frame,
            state::State,
            step::{Stepper, Stop},
            trace::Trace,
        },
        image::{ImageData, ImageRegion},
        verifier::{SsaVerifier, VerifyLevel},
    },
    ir::{
        SsaDefSpec, SsaFunction, SsaFunctionBuilder,
        integer::IntegerBits,
        lanes::VectorConst,
        ops::{
            Computation, Kernel, LaneControl, SsaOp, VectorBinaryKind, VectorElement,
            VectorFaultMode, VectorPredicateLayout, VectorRounding, VectorStatus,
        },
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::{Endianness, VectorTailPolicy},
    testing::{MACHINE, MockTarget, MockType},
};

#[derive(Debug, Default)]
struct Guard {
    writes: AtomicUsize,
    reads: AtomicUsize,
}

impl MemoryWriteGuard for Guard {
    fn check_unresolved_write(
        &self,
        _: Option<Address>,
        _: Option<usize>,
    ) -> Result<(), MemoryRefusal> {
        self.writes.fetch_add(1, Ordering::Relaxed);
        Err(MemoryRefusal::UnresolvedWrite)
    }

    fn check_write(
        &self,
        _: &mut InstructionSources,
        _: Address,
        _: usize,
    ) -> Result<(), MemoryRefusal> {
        self.writes.fetch_add(1, Ordering::Relaxed);
        Err(MemoryRefusal::UnresolvedWrite)
    }
}

impl ImageData for Guard {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, _: u64) -> Option<ImageRegion> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        None
    }
    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        None
    }
    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}

impl ExecutionHost<MockTarget> for Guard {
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        Some(self)
    }
}

fn kind(kernel: Kernel) -> Computation {
    Computation {
        kernel,
        element: VectorElement::integer(32, None),
        predicate: VectorPredicateLayout::Packed,
        rounding: VectorRounding::None,
        status: VectorStatus::None,
        canonical_nan: false,
        lanes: Some(LaneControl {
            mask_tail: false,
            tail: VectorTailPolicy::Agnostic,
            inactive: None,
            no_passthrough: false,
        }),
    }
}

fn memory(load: bool, fields: u8, fault: VectorFaultMode) -> Computation {
    kind(Kernel::Memory {
        load,
        fields,
        fault,
        access_bits: 32,
        sign_extend: false,
        byte_stream: false,
        coalescible: true,
        indexed: false,
        ordered: false,
    })
}

struct Fixture {
    ssa: SsaFunction<MockTarget>,
    inputs: Vec<SsaVarId>,
    outputs: Vec<SsaVarId>,
}

fn fixture(kind: Computation, data_types: &[MockType], status_type: MockType) -> Fixture {
    assert!(kind.is_valid());
    assert_eq!(data_types.len(), kind.data_outputs());
    let mut input_types = vec![MockType::V4I32; kind.input_count()];
    for ty in input_types.iter_mut().take(kind.control_input_count()) {
        *ty = MockType::U64;
    }
    for (index, ty) in data_types.iter().take(kind.passthrough_count()).enumerate() {
        input_types[kind.control_input_count() + index] = *ty;
    }
    if matches!(kind.kernel, Kernel::Memory { .. }) {
        let at = kind.control_input_count() + kind.passthrough_count();
        input_types[at] = MockType::U64;
        input_types[at + 1] = MockType::U64;
    }
    let mut result_types = data_types.to_vec();
    if matches!(
        kind.kernel,
        Kernel::Memory {
            fault: VectorFaultMode::FaultOnlyFirst,
            ..
        }
    ) {
        result_types.push(MockType::U64);
    }
    if kind.status != VectorStatus::None {
        result_types.push(status_type);
    }
    assert_eq!(result_types.len(), kind.output_count());
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(input_types.len(), 0);
    let inputs: Vec<_> = input_types
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            builder.function_mut().create_variable(
                VariableOrigin::Argument(u16::try_from(index).expect("bounded fixture inputs")),
                0,
                DefSite::entry(),
                ty,
            )
        })
        .collect();
    let outputs = builder
        .in_block(0, |block| {
            let specs: Vec<_> = result_types.into_iter().map(SsaDefSpec::tmp).collect();
            let outputs = block.emit_many_defs(&specs, |outputs| {
                SsaOp::compute(kind, outputs.to_vec(), inputs.clone())
            })?;
            block.ret(None)?;
            Ok(outputs)
        })
        .expect("valid fixture instructions");
    let ssa = builder.finish().expect("valid fixture body");
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    Fixture {
        ssa,
        inputs,
        outputs,
    }
}

fn state(fixture: &Fixture, bound: Vec<(SsaVarId, Value<MockTarget>)>) -> State<MockTarget> {
    State::new(Frame::enter_at(
        &fixture.ssa,
        0,
        0,
        Address::new(Root::Stack, 0, PointerSize::Bit64),
        bound,
        PointerSize::Bit64,
    ))
}

fn run(
    fixture: &Fixture,
    state: &mut State<MockTarget>,
    trace: &mut Trace,
    guard: &Guard,
    buffer: usize,
    allowance: usize,
) -> (Stop, usize) {
    let context = AnalysisContext::new(MACHINE)
        .with_image(guard)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, buffer));
    let (stop, ran) = Stepper::new(&fixture.ssa, &context)
        .with_write_guard(Some(guard))
        .run(state, trace, allowance);
    (stop, ran.instructions)
}

fn zero(fixture: &Fixture) -> (SsaVarId, Value<MockTarget>) {
    (fixture.inputs[1], Value::Const(ConstValue::U64(0)))
}

#[test]
fn zero_store_performs_no_image_or_write_queries_and_charges_once() {
    let fixture = fixture(
        memory(false, 1, VectorFaultMode::Normal),
        &[],
        MockType::U32,
    );
    for reservation in [false, true] {
        let guard = Guard::default();
        let mut state = state(&fixture, vec![zero(&fixture)]);
        state.set_reservation_possible(reservation);
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        assert_eq!(
            run(&fixture, &mut state, &mut trace, &guard, 0, 0),
            (Stop::Limited, 0)
        );
        assert_eq!(
            run(&fixture, &mut state, &mut trace, &guard, 0, 1),
            (Stop::Terminator, 1)
        );
        assert_eq!(guard.writes.load(Ordering::Relaxed), 0);
        assert_eq!(guard.reads.load(Ordering::Relaxed), 0);
        assert!(trace.loops().iter().all(|visit| visit.clean));
        assert_eq!(state.reservation_possible(), reservation);
    }
}

#[test]
fn segmented_fof_copies_all_values_and_preserves_stack_and_partial_facts() {
    let fixture = fixture(
        memory(true, 3, VectorFaultMode::FaultOnlyFirst),
        &[MockType::U64, MockType::U64, MockType::U64],
        MockType::U32,
    );
    let address = Value::Addr(Address::new(
        Root::StackDerived {
            activation: 7,
            value: fixture.inputs[2],
            run: 3,
        },
        12,
        PointerSize::Bit64,
    ));
    let partial = Value::partial_integer(
        IntegerBits::new(64, 14, 1).expect("valid partial bits"),
        true,
    );
    let guard = Guard::default();
    let mut state = state(
        &fixture,
        vec![
            zero(&fixture),
            (fixture.inputs[2], address.clone()),
            (fixture.inputs[3], partial.clone()),
            (fixture.inputs[4], Value::StackDerived),
        ],
    );
    let mut trace = Trace::new();
    trace.enter_loop(0, 0);
    assert_eq!(
        run(&fixture, &mut state, &mut trace, &guard, 0, 1),
        (Stop::Terminator, 1)
    );
    assert_eq!(state.value(fixture.outputs[0]), address);
    assert_eq!(state.value(fixture.outputs[1]), partial);
    assert!(state.value(fixture.outputs[2]).may_reference_stack());
    assert_eq!(
        state.value(fixture.outputs[3]),
        Value::Const(ConstValue::U64(0))
    );
    assert!(trace.loops().iter().all(|visit| visit.clean));
    assert_eq!(guard.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn nonzero_restart_at_length_copies_fof_length_without_resolving_addresses() {
    let fixture = fixture(
        memory(true, 1, VectorFaultMode::FaultOnlyFirst),
        &[MockType::U64],
        MockType::U32,
    );
    for start in [2, 3, u64::MAX] {
        let mut state = state(
            &fixture,
            vec![
                (fixture.inputs[0], Value::Const(ConstValue::U64(start))),
                (fixture.inputs[1], Value::Const(ConstValue::U64(2))),
                (fixture.inputs[2], Value::Const(ConstValue::U64(0x1234))),
            ],
        );
        assert_eq!(
            run(
                &fixture,
                &mut state,
                &mut Trace::new(),
                &Guard::default(),
                0,
                1
            ),
            (Stop::Terminator, 1)
        );
        assert_eq!(
            state.value(fixture.outputs[0]),
            Value::Const(ConstValue::U64(0x1234))
        );
        assert_eq!(
            state.value(fixture.outputs[1]),
            Value::Const(ConstValue::U64(2))
        );
    }
}

#[test]
fn floating_and_saturation_status_use_all_declared_destinations() {
    for floating in [false, true] {
        let mut k = kind(Kernel::Binary(if floating {
            VectorBinaryKind::Add
        } else {
            VectorBinaryKind::SatAdd
        }));
        k.status = if floating {
            VectorStatus::FloatingExceptions
        } else {
            VectorStatus::Saturation
        };
        if floating {
            k.element = VectorElement::float(32);
            k.rounding = VectorRounding::FloatingEnvironment;
        }
        let fixture = fixture(
            k,
            &[MockType::U64],
            if floating {
                MockType::U32
            } else {
                MockType::Bool
            },
        );
        let old = fixture.inputs[k.control_input_count()];
        let mut state = state(
            &fixture,
            vec![
                zero(&fixture),
                (old, Value::Const(ConstValue::U64(0x7f800001))),
            ],
        );
        assert_eq!(
            run(
                &fixture,
                &mut state,
                &mut Trace::new(),
                &Guard::default(),
                0,
                1
            ),
            (Stop::Terminator, 1)
        );
        assert_eq!(
            state.value(fixture.outputs[0]),
            Value::Const(ConstValue::U64(0x7f800001))
        );
        assert_eq!(
            state.value(fixture.outputs[1]),
            Value::Const(if floating {
                ConstValue::U32(0)
            } else {
                ConstValue::False
            })
        );
    }
}

#[test]
fn unspecified_result_does_not_borrow_inactive_stack_data() {
    let mut k = kind(Kernel::Binary(VectorBinaryKind::Add));
    k.lanes.as_mut().expect("lane kind").no_passthrough = true;
    let fixture = fixture(k, &[MockType::U64], MockType::U32);
    let mut state = state(
        &fixture,
        vec![zero(&fixture), (fixture.inputs[2], Value::StackDerived)],
    );
    assert_eq!(
        run(
            &fixture,
            &mut state,
            &mut Trace::new(),
            &Guard::default(),
            0,
            1
        ),
        (Stop::Terminator, 1)
    );
    assert!(state.value(fixture.outputs[0]).constant().is_none());
    assert!(!state.value(fixture.outputs[0]).may_reference_stack());
}

#[test]
fn unknown_passthrough_keeps_its_existing_symbolic_identity() {
    let fixture = fixture(
        memory(true, 1, VectorFaultMode::Normal),
        &[MockType::U64],
        MockType::U32,
    );
    let mut state = state(&fixture, vec![zero(&fixture)]);
    let before = state.value(fixture.inputs[2]);
    assert!(before.constant().is_none());
    assert_eq!(
        run(
            &fixture,
            &mut state,
            &mut Trace::new(),
            &Guard::default(),
            0,
            1
        ),
        (Stop::Terminator, 1)
    );
    assert_eq!(state.value(fixture.outputs[0]), before);
}

#[test]
fn secondary_result_limit_is_checked_before_any_definition() {
    let fixture = fixture(
        memory(true, 2, VectorFaultMode::Normal),
        &[MockType::V4I32, MockType::V2I256],
        MockType::U32,
    );
    for limit in [0, 16, 63, 64] {
        let mut state = state(&fixture, vec![zero(&fixture)]);
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        let (stop, ran) = run(
            &fixture,
            &mut state,
            &mut trace,
            &Guard::default(),
            limit,
            1,
        );
        if limit < 64 {
            assert!(matches!(
                stop,
                Stop::Exhausted(Exhaustion {
                    budget: Budget::ExecutionBufferBytes,
                    ..
                })
            ));
            assert!(fixture.outputs.iter().all(|&v| {
                state
                    .frame()
                    .expect("running frame")
                    .held_value(v)
                    .is_none()
            }));
            // The preexisting first-result check can stop before the instruction;
            // secondary-result refusal reaches the ordinary one-step epilogue.
            assert_eq!(ran, usize::from(limit >= 16));
        } else {
            assert_eq!((stop, ran), (Stop::Terminator, 1));
        }
    }
}

#[test]
fn malformed_held_payload_is_admitted_before_cloning_any_result() {
    let fixture = fixture(
        memory(true, 2, VectorFaultMode::Normal),
        &[MockType::U64, MockType::U64],
        MockType::U32,
    );
    let malformed = Value::Const(ConstValue::DecryptedString(
        "x".repeat(4096).into_boxed_str(),
    ));
    let mut state = state(
        &fixture,
        vec![
            zero(&fixture),
            (fixture.inputs[2], Value::Const(ConstValue::U64(7))),
            (fixture.inputs[3], malformed),
        ],
    );
    assert_eq!(
        run(
            &fixture,
            &mut state,
            &mut Trace::new(),
            &Guard::default(),
            32,
            1
        ),
        (
            Stop::Exhausted(Exhaustion {
                budget: Budget::ExecutionBufferBytes,
                limit: 32,
                demand: 4096
            }),
            1
        )
    );
    assert!(fixture.outputs.iter().all(|&v| {
        state
            .frame()
            .expect("running frame")
            .held_value(v)
            .is_none()
    }));
}

#[test]
fn active_and_unknown_stores_keep_existing_refusal() {
    let fixture = fixture(
        memory(false, 1, VectorFaultMode::Normal),
        &[],
        MockType::U32,
    );
    for length in [None, Some(1)] {
        let guard = Guard::default();
        let mut bound = vec![(fixture.inputs[0], Value::Const(ConstValue::U64(0)))];
        if let Some(length) = length {
            bound.push((fixture.inputs[1], Value::Const(ConstValue::U64(length))));
        }
        let mut state = state(&fixture, bound);
        assert_eq!(
            run(&fixture, &mut state, &mut Trace::new(), &guard, 64, 1),
            (Stop::Refused(MemoryRefusal::UnresolvedWrite), 1)
        );
        assert_eq!(guard.writes.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn inactive_execution_keeps_existing_taint_reservation_and_source_history() {
    let fixture = fixture(
        memory(false, 1, VectorFaultMode::Normal),
        &[],
        MockType::U32,
    );
    let mut state = state(
        &fixture,
        vec![
            zero(&fixture),
            (fixture.inputs[2], Value::StackDerived),
            (fixture.inputs[4], Value::StackDerived),
        ],
    );
    state.memory_mut().store_bytes(
        Address::new(Root::Stack, -8, PointerSize::Bit64),
        &[1, 0, 2, 3],
    );
    state.set_reservation_possible(true);
    assert!(
        state
            .memory_mut()
            .instruction_sources_mut()
            .enter(InstructionSourceId::new(1))
    );
    state
        .memory_mut()
        .instruction_sources_mut()
        .record_write(InstructionSourceId::new(2))
        .expect("distinct unentered source");
    let memory_before = state.memory().clone();
    let mut trace = Trace::new();
    trace.enter_loop(0, 0);
    trace.taint();
    assert_eq!(
        run(&fixture, &mut state, &mut trace, &Guard::default(), 0, 1),
        (Stop::Terminator, 1)
    );
    assert!(state.reservation_possible());
    assert_eq!(state.memory(), &memory_before);
    assert!(trace.loops().iter().all(|visit| !visit.clean));
}

fn public_zero_store() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |block| {
            let start = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
            let length = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
            let stride = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(4))?;
            let value = block.const_value(
                SsaDefSpec::tmp(MockType::V4I32),
                ConstValue::Vector(VectorConst::from_lanes(32, [1, 2, 3, 4]).expect("fixed lanes")),
            )?;
            block.emit_no_defs(SsaOp::compute(
                memory(false, 1, VectorFaultMode::Normal),
                vec![],
                vec![start, length, address, stride, value],
            ))?;
            block.ret(None)
        })
        .expect("valid zero store body");
    builder.finish().expect("complete zero store body")
}

#[test]
fn guarded_zero_store_completes_through_the_public_executor() {
    let ssa = public_zero_store();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    let context = AnalysisContext::new(MACHINE);
    let host = Guard::default();
    let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
    assert_eq!(facts.completion(), Completion::Complete);
    assert_eq!(host.writes.load(Ordering::Relaxed), 0);
}

#[derive(Debug)]
struct Unstable;
impl ExecutionHost<MockTarget> for Unstable {
    fn admit_code_source(
        &self,
        _: &SsaFunction<MockTarget>,
        _: Option<&u32>,
        _: &mut InstructionSources,
    ) -> bool {
        false
    }
}

#[test]
fn inactive_compute_does_not_bypass_original_source_admission() {
    let ssa = public_zero_store();
    let context = AnalysisContext::new(MACHINE);
    assert_eq!(
        StaticExecution::new(&ssa, &context)
            .with_host(&Unstable)
            .run()
            .completion(),
        Completion::UnstableCode
    );
}
