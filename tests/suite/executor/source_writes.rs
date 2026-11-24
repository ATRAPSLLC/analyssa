//! Known writes preserve path-local instruction-source contracts across calls and models.

use std::{error::Error, num::NonZeroU16, ops::Range, ptr};

use analyssa::{
    PointerSize,
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            Address, CallModel, CallSite, Completion, ExecutionHost, InstructionSourceId,
            InstructionSources, LibraryModel, MemoryRefusal, MemoryWriteGuard, ModelCall,
            ModelControl, Root, StaticExecution, StaticFacts, Value,
        },
    },
    ir::{
        Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        lanes::VectorConst,
        ops::{
            AccessQualifiers, AtomicAccessWidth, AtomicOrdering, AtomicStatus, CallData,
            CallEffects, CallTermination, SsaOp, VectorPackKind, VectorSegmentLayout,
        },
        value::ConstValue,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    testing::{MACHINE, MockTarget, MockType},
};

type Function = SsaFunction<MockTarget>;

fn verified(builder: SsaFunctionBuilder<MockTarget>) -> Result<Function, Box<dyn Error>> {
    let function = builder.finish()?;
    let errors = SsaVerifier::new(&function).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    Ok(function)
}

#[derive(Debug)]
struct Sources;

impl MemoryWriteGuard for Sources {
    fn check_write(
        &self,
        sources: &mut InstructionSources,
        address: Address,
        len: usize,
    ) -> Result<(), MemoryRefusal> {
        let Some(at) = address.image_address() else {
            return Ok(());
        };
        let prefix = address.linear_prefix_len(len);
        let end =
            at.saturating_add(u64::try_from(prefix).map_err(|_| MemoryRefusal::AddressRange)?);
        let suffix =
            u64::try_from(len.saturating_sub(prefix)).map_err(|_| MemoryRefusal::AddressRange)?;
        for (source, start, finish) in [(1000, 0x1000, 0x1004), (3000, 0x3000, 0x3004)] {
            if (at < finish && start < end) || start < suffix {
                sources.record_write(InstructionSourceId::new(source))?;
            }
        }
        Ok(())
    }
}

struct Writer(bool);

impl CallModel<MockTarget> for Writer {
    fn only_memory(&self) -> bool {
        true
    }
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        if self.0 {
            call.write_unknown(
                Address::new(Root::Symbol, 0xffff_fffe, PointerSize::Bit32),
                0x1003,
            );
        } else {
            call.write(Address::new(Root::Symbol, 0x1001, PointerSize::Bit64), &[7]);
        }
    }
}

struct Terminate(ModelControl);

impl CallModel<MockTarget> for Terminate {
    fn only_memory(&self) -> bool {
        true
    }
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.set_control(self.0);
    }
}

struct Host {
    callee: Function,
    reject_unevaluated: bool,
}

impl ExecutionHost<MockTarget> for Host {
    fn admit_unevaluated_call(&self, _: CallSite<'_, MockTarget>) -> bool {
        !self.reject_unevaluated
    }

    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 3000).then_some(&self.callee)
    }
    fn admit_code_source(
        &self,
        _: &Function,
        method: Option<&u32>,
        sources: &mut InstructionSources,
    ) -> bool {
        sources.enter(InstructionSourceId::new(u64::from(
            method.copied().unwrap_or(1000),
        )))
    }
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        Some(&Sources)
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match method {
            5 => Some(&Writer(false)),
            6 => Some(&Writer(true)),
            10 => Some(&LibraryModel::Copy),
            11 => Some(&LibraryModel::Fill),
            12 => Some(&LibraryModel::CopyString),
            13 => Some(&Terminate(ModelControl::NoReturn)),
            14 => Some(&Terminate(ModelControl::Trap)),
            _ => None,
        }
    }
}

fn host() -> Result<Host, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let value = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
        block.ret(Some(value))
    })?;
    Ok(Host {
        callee: verified(builder)?,
        reject_unevaluated: false,
    })
}

/// Commuting an unscaled symbolic index cannot hide an actual protected instruction write.
#[test]
fn commuted_symbolic_pointer_add_retains_instruction_write_admission() -> Result<(), Box<dyn Error>>
{
    for commuted in [false, true] {
        let mut builder = SsaFunctionBuilder::new(0, 0);
        builder.in_block(0, |block| {
            let symbol =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x1000))?;
            let symbol =
                block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::PtrToInt {
                    dest,
                    operand: symbol,
                    target: MockType::I64,
                })?;
            let number = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(3))?;
            let (base, index) = if commuted {
                (number, symbol)
            } else {
                (symbol, number)
            };
            let address = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base,
                index: Some(index),
                stride: 1,
                offset: -2,
                result_type: MockType::Ptr,
            })?;
            let value = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(42))?;
            block.store_indirect(address, value, MockType::U8)?;
            block.ret(Some(value))
        })?;
        let function = verified(builder)?;
        let context = AnalysisContext::new(MACHINE);
        let host = host()?;
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .run();
        assert!(
            matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::InstructionSource,
                    ..
                }
            ),
            "{commuted}: {:?}",
            facts.completion()
        );
        assert!(facts.image_writes().is_empty());
        assert!(facts.calls().is_empty());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Trial {
    Call,
    WriteOnly,
    WriteThenCall,
    RestoreThenCall,
    ClobberThenCall,
    CallThenWrite,
    UnrelatedWrite,
    Model(u32),
}

fn program(trial: Trial) -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let marker = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
        if let Trial::Model(method) = trial {
            block.call_void(Callee::Direct(method), vec![])?;
            return block.ret(Some(marker));
        }
        if matches!(trial, Trial::CallThenWrite) {
            block.call_void(Callee::Direct(3000), vec![])?;
        }
        if !matches!(trial, Trial::Call) {
            let pointer = block.const_value(
                SsaDefSpec::tmp(MockType::Ptr),
                ConstValue::Symbol(if matches!(trial, Trial::UnrelatedWrite) {
                    0x7000
                } else {
                    0x3001
                }),
            )?;
            let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            block.store_indirect(pointer, byte, MockType::U8)?;
            if matches!(trial, Trial::RestoreThenCall) {
                let restored =
                    block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(13))?;
                block.store_indirect(pointer, restored, MockType::U8)?;
            }
        }
        if matches!(trial, Trial::ClobberThenCall) {
            block.call_void(Callee::Direct(99), vec![])?;
        }
        if !matches!(trial, Trial::WriteOnly | Trial::CallThenWrite) {
            block.call_void(Callee::Direct(3000), vec![])?;
        }
        block.ret(Some(marker))
    })?;
    verified(builder)
}

fn run(function: &Function, host: &Host) -> StaticFacts<MockTarget> {
    let context = AnalysisContext::new(MACHINE);
    StaticExecution::new(function, &context)
        .with_host(host)
        .with_method(1000)
        .with_root_call_observations(4)
        .run()
}

#[test]
fn prior_writes_restoration_and_clobbers_cannot_admit_a_stale_callee() -> Result<(), Box<dyn Error>>
{
    let host = host()?;
    for trial in [
        Trial::WriteThenCall,
        Trial::RestoreThenCall,
        Trial::ClobberThenCall,
    ] {
        let facts = run(&program(trial)?, &host);
        assert_eq!(facts.completion(), Completion::UnstableCode, "{trial:?}");
        assert!(facts.constants().is_empty());
        assert!(facts.invocations().is_empty());
        assert!(facts.returned().is_none());
    }
    Ok(())
}

#[test]
fn later_writes_and_models_withhold_earlier_observations() -> Result<(), Box<dyn Error>> {
    let host = host()?;
    for trial in [Trial::CallThenWrite, Trial::Model(5), Trial::Model(6)] {
        let facts = run(&program(trial)?, &host);
        assert!(
            matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::InstructionSource,
                    ..
                }
            ),
            "{trial:?}: {:?}",
            facts.completion()
        );
        assert!(facts.constants().is_empty());
        assert!(facts.invocations().is_empty());
        assert!(facts.calls().is_empty());
        assert!(facts.returned().is_none());
    }
    Ok(())
}

#[test]
fn unrelated_sources_and_independent_roots_remain_usable_in_either_order()
-> Result<(), Box<dyn Error>> {
    let host = host()?;
    for trials in [
        [Trial::WriteOnly, Trial::Call],
        [Trial::Call, Trial::WriteOnly],
        [Trial::UnrelatedWrite, Trial::Call],
    ] {
        for trial in trials {
            let facts = run(&program(trial)?, &host);
            assert_eq!(facts.completion(), Completion::Complete, "{trial:?}");
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&ConstValue::U32(13))
            );
        }
    }
    Ok(())
}

#[test]
fn a_join_keeps_written_and_unwritten_source_paths_distinct() -> Result<(), Box<dyn Error>> {
    for write_on_true in [false, true] {
        let mut builder = SsaFunctionBuilder::new(0, 0);
        let input = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I1,
        );
        let marker = builder.in_block(0, |block| {
            let marker = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
            block.branch(input, 1, 2)?;
            Ok(marker)
        })?;
        let mut inputs = Vec::new();
        for branch in [1, 2] {
            let written = (branch == 1) == write_on_true;
            let selected = builder.in_block(branch, |block| {
                if written {
                    let pointer = block
                        .const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x3001))?;
                    let byte =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
                    block.store_indirect(pointer, byte, MockType::U8)?;
                }
                let selected = block.const_value(
                    SsaDefSpec::tmp(MockType::I1),
                    if written {
                        ConstValue::False
                    } else {
                        ConstValue::True
                    },
                )?;
                block.jump(3)?;
                Ok(selected)
            })?;
            inputs.push((branch, selected));
        }
        builder.in_block(3, |block| {
            let selected = block.phi(SsaDefSpec::tmp(MockType::I1), inputs)?;
            block.branch(selected, 4, 5)
        })?;
        builder.in_block(4, |block| {
            let result =
                block.call(SsaDefSpec::tmp(MockType::U32), Callee::Direct(3000), vec![])?;
            block.ret(Some(result))
        })?;
        builder.in_block(5, |block| block.ret(Some(marker)))?;
        let facts = run(&verified(builder)?, &host()?);
        assert_eq!(facts.completion(), Completion::Complete);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::U32(13))
        );
    }
    Ok(())
}

fn fallback_block(
    operation: u8,
    unknown_source: bool,
    unknown_direction: bool,
    empty: bool,
    source_space: bool,
) -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U64,
    );
    let direction = builder.function_mut().create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::I1,
    );
    builder.in_block(0, |block| {
        let pointer =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x1002))?;
        let source = if unknown_source {
            let mask = block.const_value(
                SsaDefSpec::tmp(MockType::U64),
                ConstValue::U64(0xffff_ffff_ffff_fff0),
            )?;
            block.and(SsaDefSpec::tmp(MockType::U64), input, mask)?
        } else {
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x9002))?
        };
        let size = block.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(if empty { 0 } else { 4 }),
        )?;
        let value = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
        let backward = if unknown_direction {
            Some(direction)
        } else if operation == 2 {
            Some(block.const_value(SsaDefSpec::tmp(MockType::I1), ConstValue::True)?)
        } else {
            None
        };
        match operation {
            0 => block.init_blk(pointer, value, size, backward)?,
            1 | 2 => block.emit_no_defs(SsaOp::CopyBlk {
                dest_addr: pointer,
                src_addr: source,
                size,
                element_bytes: 2,
                backward,
                src_access: AccessQualifiers {
                    address_space: NonZeroU16::new(1).filter(|_| source_space),
                    ..AccessQualifiers::default()
                },
                dest_access: AccessQualifiers::default(),
            })?,
            _ => block.move_blk(pointer, source, size)?,
        }
        let answer = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
        block.ret(Some(answer))
    })?;
    verified(builder)
}

#[test]
fn unresolved_sources_directions_and_spaces_cannot_hide_known_destinations()
-> Result<(), Box<dyn Error>> {
    let host = host()?;
    for operation in 0..4 {
        for unknown_source in [false, true] {
            for unknown_direction in [false, true] {
                for source_space in [false, true] {
                    for empty in [false, true] {
                        let function = fallback_block(
                            operation,
                            unknown_source,
                            unknown_direction,
                            empty,
                            source_space,
                        )?;
                        let facts = run(&function, &host);
                        if empty {
                            assert_eq!(facts.completion(), Completion::Complete);
                        } else {
                            assert!(
                                matches!(
                                    facts.completion(),
                                    Completion::Memory {
                                        reason: MemoryRefusal::InstructionSource,
                                        ..
                                    }
                                ),
                                "operation={operation}, source={unknown_source}, direction={unknown_direction}, space={source_space}: {:?}",
                                facts.completion()
                            );
                            assert!(facts.returned().is_none());
                            assert!(facts.constants().is_empty());
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn guarded_oversized_known_writes_exhaust_before_fallback() -> Result<(), Box<dyn Error>> {
    let host = host()?;
    for operation in 0..4 {
        let function = fallback_block(operation, false, false, false, false)?;
        let context = AnalysisContext::new(MACHINE)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBlockBytes, 2));
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .with_method(1000)
            .run();
        assert!(
            matches!(facts.completion(), Completion::Exhausted(exhaustion) if exhaustion.budget == Budget::ExecutionBlockBytes && exhaustion.limit == 2 && exhaustion.demand == 4)
        );
        assert!(facts.returned().is_none());
        assert!(facts.constants().is_empty());
    }
    Ok(())
}

fn library_program(method: u32, empty: bool) -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U64,
    );
    builder.in_block(0, |block| {
        let pointer =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x1001))?;
        let mask = block.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(0xffff_ffff_ffff_fff0),
        )?;
        let source = block.and(SsaDefSpec::tmp(MockType::U64), input, mask)?;
        let count = block.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(if empty { 0 } else { 4 }),
        )?;
        block.call_void(Callee::Direct(method), vec![pointer, source, count])?;
        let answer = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
        block.ret(Some(answer))
    })?;
    verified(builder)
}

#[test]
fn builtin_models_preserve_known_write_obligations_and_limits() -> Result<(), Box<dyn Error>> {
    let host = host()?;
    for method in [10, 11, 12] {
        let function = library_program(method, false)?;
        let facts = run(&function, &host);
        assert!(
            matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::InstructionSource,
                    ..
                }
            ),
            "model={method}: {:?}",
            facts.completion()
        );
        assert!(facts.returned().is_none());
        assert!(facts.constants().is_empty());
        if method != 12 {
            for budget in [Budget::ExecutionBufferBytes, Budget::ExecutionBlockBytes] {
                let context = AnalysisContext::new(MACHINE)
                    .with_budget(WorkBudget::DEFAULT.with_limit(budget, 2));
                let facts = StaticExecution::new(&function, &context)
                    .with_host(&host)
                    .with_method(1000)
                    .run();
                if budget == Budget::ExecutionBufferBytes {
                    assert!(
                        matches!(facts.completion(), Completion::Exhausted(exhaustion) if exhaustion.budget == budget)
                    );
                } else {
                    // A library copy is bounded by its model buffer allowance, not an ISA block limit.
                    assert!(matches!(
                        facts.completion(),
                        Completion::Memory {
                            reason: MemoryRefusal::InstructionSource,
                            ..
                        }
                    ));
                }
                assert!(facts.constants().is_empty());
            }
            assert_eq!(
                run(&library_program(method, true)?, &host).completion(),
                Completion::Complete
            );
        }
    }
    Ok(())
}

#[derive(Debug)]
struct AtomicGuard;
impl MemoryWriteGuard for AtomicGuard {
    fn check_write(
        &self,
        sources: &mut InstructionSources,
        address: Address,
        len: usize,
    ) -> Result<(), MemoryRefusal> {
        assert_eq!(address.root(), Root::Stack);
        assert_eq!(address.offset(), -16);
        assert_eq!(len, 4);
        sources.record_write(InstructionSourceId::new(1))
    }
}
struct AtomicHost {
    callee: Function,
    protect_root: bool,
    guarded: bool,
}
impl ExecutionHost<MockTarget> for AtomicHost {
    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 3000).then_some(&self.callee)
    }
    fn stack_extent(&self, _: &Function, _: Option<&u32>) -> Option<Range<i64>> {
        Some(-32..0)
    }
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        self.guarded.then_some(&AtomicGuard)
    }
    fn admit_code_source(
        &self,
        _: &Function,
        method: Option<&u32>,
        sources: &mut InstructionSources,
    ) -> bool {
        sources.enter(InstructionSourceId::new(u64::from(
            self.protect_root || method == Some(&3000),
        )))
    }
}
fn atomic_program(conditional: bool, call_later: bool) -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .ok_or("entry stack")?
        .set_role(VariableRole::EntryStackPointer);
    builder.in_block(0, |block| {
        let address = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 1,
            offset: -16,
            result_type: MockType::Ptr,
        })?;
        let value = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(7))?;
        if conditional {
            block.emit_def(SsaDefSpec::tmp(MockType::U32), |status| {
                SsaOp::AtomicStoreConditional {
                    status,
                    status_encoding: AtomicStatus::ZeroOne,
                    consumes_reservation: true,
                    addr: address,
                    value,
                    value_type: MockType::U32,
                    success_ordering: AtomicOrdering::Release,
                    failure_ordering: AtomicOrdering::Relaxed,
                    width: AtomicAccessWidth::Bits32,
                    access: AccessQualifiers::default(),
                }
            })?;
        } else {
            block.emit_no_defs(SsaOp::AtomicStore {
                addr: address,
                value,
                value_type: MockType::U32,
                ordering: AtomicOrdering::SeqCst,
                width: AtomicAccessWidth::Bits32,
                access: AccessQualifiers::default(),
            })?;
        }
        if call_later {
            block.call_void(Callee::Direct(3000), vec![])?;
        }
        let answer = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
        block.ret(Some(answer))
    })?;
    verified(builder)
}

#[test]
fn conditional_atomic_writes_use_and_update_the_real_path_history() -> Result<(), Box<dyn Error>> {
    for conditional in [false, true] {
        for guarded in [false, true] {
            for protect_root in [false, true] {
                let function = atomic_program(conditional, !protect_root)?;
                let host = AtomicHost {
                    callee: host()?.callee,
                    protect_root,
                    guarded,
                };
                let context = AnalysisContext::new(MACHINE);
                let facts = StaticExecution::new(&function, &context)
                    .with_host(&host)
                    .with_method(1000)
                    .run();
                if !guarded {
                    assert_eq!(facts.completion(), Completion::Complete);
                } else if protect_root {
                    assert!(matches!(
                        facts.completion(),
                        Completion::Memory {
                            reason: MemoryRefusal::InstructionSource,
                            ..
                        }
                    ));
                } else {
                    assert_eq!(facts.completion(), Completion::UnstableCode);
                }
                if guarded {
                    assert!(facts.constants().is_empty());
                    assert!(facts.returned().is_none());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn zero_library_writes_preserve_preceding_memory_with_unknown_arguments()
-> Result<(), Box<dyn Error>> {
    for method in [10, 11] {
        for unknown_destination in [false, true] {
            let mut builder = SsaFunctionBuilder::new(0, 0);
            let input = builder.function_mut().create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::U64,
            );
            builder.in_block(0, |block| {
                let pointer = block
                    .const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x7000))?;
                let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
                block.store_indirect(pointer, byte, MockType::U8)?;
                let mask = block.const_value(
                    SsaDefSpec::tmp(MockType::U64),
                    ConstValue::U64(0xffff_ffff_ffff_fff0),
                )?;
                let unknown = block.and(SsaDefSpec::tmp(MockType::U64), input, mask)?;
                let zero = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                block.call_void(
                    Callee::Direct(method),
                    vec![
                        if unknown_destination {
                            unknown
                        } else {
                            pointer
                        },
                        unknown,
                        zero,
                    ],
                )?;
                let result =
                    block.load_indirect(SsaDefSpec::tmp(MockType::U8), pointer, MockType::U8)?;
                block.ret(Some(result))
            })?;
            let facts = run(&verified(builder)?, &host()?);
            assert_eq!(facts.completion(), Completion::Complete);
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&ConstValue::U8(7))
            );
        }
    }
    Ok(())
}

#[test]
fn vector_writes_preserve_history_before_entry_and_after_return() -> Result<(), Box<dyn Error>> {
    let host = host()?;
    let offsets = VectorConst::from_lanes(32, [0, 4, 8, 12]).ok_or("offsets")?;
    for form in 0..6 {
        for after in [false, true] {
            for (destination, empty) in [(0x3000, false), (0x7000, false), (0x3000, true)] {
                if form == 0 && empty {
                    continue;
                }
                let predicate =
                    VectorConst::from_lanes(1, [u128::from(!empty), 0, 0, 0]).ok_or("predicate")?;
                let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
                let value = builder.function_mut().create_variable(
                    VariableOrigin::Argument(0),
                    0,
                    DefSite::entry(),
                    MockType::V4I32,
                );
                builder.in_block(0, |block| {
                    if after {
                        block.call_void(Callee::Direct(3000), vec![])?;
                    }
                    let addr = block.const_value(
                        SsaDefSpec::tmp(MockType::Ptr),
                        ConstValue::Symbol(destination),
                    )?;
                    let mask = block.const_value(
                        SsaDefSpec::tmp(MockType::Mask4),
                        ConstValue::Vector(predicate),
                    )?;
                    let vector_type = MockType::V4I32;
                    let access = AccessQualifiers::default();
                    let op = match form {
                        0 => SsaOp::VectorStore {
                            addr,
                            value,
                            vector_type,
                            access,
                        },
                        1 => SsaOp::VectorMaskedStore {
                            addr,
                            value,
                            mask,
                            vector_type,
                            access,
                        },
                        2 => {
                            let indices = block.const_value(
                                SsaDefSpec::tmp(MockType::V4I32),
                                ConstValue::Vector(offsets.clone()),
                            )?;
                            SsaOp::VectorScatter {
                                base: addr,
                                indices,
                                value,
                                mask,
                                vector_type,
                                access,
                            }
                        }
                        3 => SsaOp::VectorPackStore {
                            addr,
                            value,
                            mask,
                            vector_type,
                            element_bits: 32,
                            kind: VectorPackKind::Compress,
                            access,
                        },
                        _ => SsaOp::VectorSegmentStore {
                            base: addr,
                            values: vec![value, value],
                            mask: Some(mask),
                            vector_type,
                            segments: 2,
                            layout: if form == 4 {
                                VectorSegmentLayout::Interleaved
                            } else {
                                VectorSegmentLayout::Consecutive
                            },
                            access,
                        },
                    };
                    block.emit_no_defs(op)?;
                    if !after {
                        block.call_void(Callee::Direct(3000), vec![])?;
                    }
                    let result =
                        block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
                    block.ret(Some(result))
                })?;
                let function = verified(builder)?;
                let facts = run(&function, &host);
                if destination == 0x7000 || empty {
                    assert!(
                        facts.is_complete(),
                        "form={form}, after={after}, empty={empty}: {:?}",
                        facts.completion()
                    );
                    assert_eq!(
                        facts
                            .returned()
                            .and_then(|value| value.constant())
                            .and_then(ConstValue::as_u64_bits),
                        Some(13)
                    );
                } else {
                    assert!(
                        matches!(
                            facts.completion(),
                            Completion::Memory {
                                reason: MemoryRefusal::InstructionSource,
                                ..
                            }
                        ) && after
                            || facts.completion() == Completion::UnstableCode && !after,
                        "form={form}, after={after}: {:?}",
                        facts.completion()
                    );
                    assert!(facts.returned().is_none());
                    assert!(facts.invocations().is_empty());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn completed_source_history_retains_writes_and_shares_call_evidence() -> Result<(), Box<dyn Error>>
{
    let host = host()?;
    for (trial, entered, written) in [(Trial::WriteOnly, false, true), (Trial::Call, true, false)] {
        let facts = run(&program(trial)?, &host);
        assert!(facts.is_complete());
        let history = facts
            .instruction_sources()
            .ok_or("completed source history")?;
        assert!(history.has_entered(InstructionSourceId::new(1000)));
        assert_eq!(history.has_entered(InstructionSourceId::new(3000)), entered);
        assert_eq!(history.has_written(InstructionSourceId::new(3000)), written);
        for invocation in facts.invocations() {
            assert!(ptr::eq(
                history,
                invocation
                    .facts
                    .instruction_sources()
                    .ok_or("shared call source evidence")?
            ));
        }
    }
    let unguarded =
        StaticExecution::new(&program(Trial::WriteOnly)?, &AnalysisContext::new(MACHINE)).run();
    assert!(unguarded.instruction_sources().is_none());
    Ok(())
}

#[test]
fn terminated_prefix_writes_survive_the_complete_trial_union() -> Result<(), Box<dyn Error>> {
    for (prefix, reverse) in
        (0..4).flat_map(|prefix| [false, true].map(move |reverse| (prefix, reverse)))
    {
        let mut builder = SsaFunctionBuilder::new(0, 0);
        let input = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I1,
        );
        builder.in_block(0, |block| block.branch(input, 1, 2))?;
        builder.in_block(if reverse { 2 } else { 1 }, |block| {
            let pointer =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x3001))?;
            let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            block.store_indirect(pointer, byte, MockType::U8)?;
            match prefix {
                1 => block.unreachable(),
                2 | 3 => {
                    block.call_void(Callee::Direct(if prefix == 2 { 13 } else { 14 }), vec![])?;
                    block.ret(None)
                }
                _ => block.ret(None),
            }
        })?;
        builder.in_block(if reverse { 1 } else { 2 }, |block| {
            block.call_void(Callee::Direct(3000), vec![])?;
            block.ret(None)
        })?;
        let facts = run(&verified(builder)?, &host()?);
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let history = facts.instruction_sources().ok_or("source union")?;
        let callee = InstructionSourceId::new(3000);
        assert!(history.has_entered(callee));
        assert!(
            history.has_written(callee),
            "prefix={prefix} reverse={reverse}"
        );
    }
    Ok(())
}

#[test]
fn audit_policy_refuses_unanswered_calls_without_changing_default_execution()
-> Result<(), Box<dyn Error>> {
    for indirect in [false, true] {
        let mut builder = SsaFunctionBuilder::new(0, 0);
        let pointer = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        builder.in_block(0, |block| {
            let callee = if indirect {
                Callee::Indirect {
                    fptr: pointer,
                    signature: 0,
                }
            } else {
                Callee::Direct(99)
            };
            block.call_void(callee, vec![])?;
            let marker = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
            block.ret(Some(marker))
        })?;
        let function = verified(builder)?;
        let mut host = host()?;
        let default = run(&function, &host);
        assert!(default.is_complete());
        assert_eq!(
            default.returned().and_then(Value::constant),
            Some(&ConstValue::U32(13))
        );
        host.reject_unevaluated = true;
        let refused = run(&function, &host);
        assert_eq!(refused.completion(), Completion::UnevaluatedEffects);
        assert!(refused.constants().is_empty());
        assert!(refused.invocations().is_empty());
        assert!(refused.returned().is_none());
        assert!(refused.instruction_sources().is_none());
        let executed = run(&program(Trial::Call)?, &host);
        assert!(executed.is_complete());
    }
    Ok(())
}

/// **An audit takes a call's word that it never returns only from the
/// callee's admitted body.** The callee writes and then loops, and the call
/// is stated never to return. With its body known and admitted the path ends
/// at the call, as at `exit`, and the audit publishes the other path's 13:
/// nothing the callee wrote is ever seen. The control: a call to a method
/// with no known body has only the host's word, and the audit refuses it; a
/// call that may not return resumes, and its skipped write is refused.
#[test]
fn audit_policy_admits_a_never_returning_call_only_on_its_admitted_body()
-> Result<(), Box<dyn Error>> {
    for (termination, target) in [
        (CallTermination::NeverReturns, 3000),
        (CallTermination::NeverReturns, 3001),
        (CallTermination::MayNotReturn, 3000),
    ] {
        let mut callee = SsaFunctionBuilder::new(0, 0);
        callee.in_block(0, |block| {
            let pointer =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x1001))?;
            let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            block.store_indirect(pointer, byte, MockType::U8)?;
            block.jump(1)
        })?;
        callee.in_block(1, |block| block.jump(1))?;
        let mut host = Host {
            callee: verified(callee)?,
            reject_unevaluated: false,
        };
        let mut root = SsaFunctionBuilder::new(0, 0);
        let input = root.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I1,
        );
        root.in_block(0, |block| block.branch(input, 1, 2))?;
        root.in_block(1, |block| {
            let mut effects = CallEffects::unknown();
            effects.termination = termination;
            block.emit_no_defs(
                CallData::new(None, Callee::Direct(target), vec![])
                    .with_effects(effects)
                    .into(),
            )?;
            block.jump(1)
        })?;
        root.in_block(2, |block| {
            let value = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
            block.ret(Some(value))
        })?;
        let function = verified(root)?;
        let ordinary = run(&function, &host);
        host.reject_unevaluated = true;
        let audit = run(&function, &host);
        if termination == CallTermination::NeverReturns {
            assert!(ordinary.is_complete());
            assert_eq!(
                ordinary.returned().and_then(Value::constant),
                Some(&ConstValue::U32(13))
            );
            if target == 3000 {
                assert!(audit.is_complete(), "{:?}", audit.completion());
                assert_eq!(
                    audit.returned().and_then(Value::constant),
                    Some(&ConstValue::U32(13))
                );
                continue;
            }
            assert_eq!(audit.completion(), Completion::UnevaluatedEffects);
        } else {
            for facts in [&ordinary, &audit] {
                assert!(matches!(
                    facts.completion(),
                    Completion::Memory {
                        reason: MemoryRefusal::InstructionSource,
                        ..
                    }
                ));
            }
        }
        assert!(audit.returned().is_none());
        assert!(audit.invocations().is_empty());
        assert!(audit.instruction_sources().is_none());
    }
    Ok(())
}
