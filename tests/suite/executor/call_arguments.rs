//! One ABI argument view feeds models, body entries and pre-call observations.

use std::cell::Cell;

use analyssa::{
    Error, Result,
    analysis::{
        VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            ArgumentBindings, CallModel, CallSite, Completion, ExecutionHost, LibraryModel,
            MemoryRefusal, ModelCall, ParameterBindings, StaticExecution,
        },
    },
    ir::{
        CallData, CallInput, CallInputKey, Callee, StagedValue,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{AccessMode, CallEffects, CallLocation, PointeeExtent, SsaOp},
        value::ConstValue,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::{Endianness, Target},
    testing::{MACHINE, MockTarget, MockType},
};

struct Fill;

impl ExecutionHost<MockTarget> for Fill {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 1).then_some(&LibraryModel::Fill)
    }

    fn arguments(
        &self,
        _: CallSite<'_, MockTarget>,
        _: Option<&u32>,
        _: Option<&SsaFunction<MockTarget>>,
        _: Option<&dyn CallModel<MockTarget>>,
        arguments: &mut ArgumentBindings<'_, MockTarget>,
    ) {
        arguments.begin();
        if !arguments.push_stack(None, 0, &MockType::Ptr, &MockType::Ptr) {
            return;
        }
        if !arguments.push_stack(None, 8, &MockType::U64, &MockType::U32) {
            return;
        }
        let _ = arguments.push_stack(None, 16, &MockType::U64, &MockType::U64);
    }
}

#[test]
fn actual_stack_arguments_supply_model_effects_and_pre_call_bytes() -> Result<()> {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = b.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    b.function_mut()
        .variable_mut(stack)
        .ok_or_else(|| Error::new("stack"))?
        .set_role(VariableRole::EntryStackPointer);
    b.in_block(0, |b| {
        let buffer = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 0,
            offset: 32,
            result_type: MockType::Ptr,
        })?;
        let byte_slot = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 0,
            offset: 8,
            result_type: MockType::Ptr,
        })?;
        let count_slot = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 0,
            offset: 16,
            result_type: MockType::Ptr,
        })?;
        let byte = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(9))?;
        let count = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(4))?;
        let before = b.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(0x0707070707070707),
        )?;
        b.store_indirect(buffer, before, MockType::U64)?;
        b.store_indirect(stack, buffer, MockType::Ptr)?;
        b.store_indirect(byte_slot, byte, MockType::U64)?;
        b.store_indirect(count_slot, count, MockType::U64)?;
        // The raw contract refers to source operand 1; the model's buffer is
        // logical parameter 0 after ABI binding. Its own bound must still apply.
        let mut call = CallData::new(None, Callee::Direct(1), vec![count, buffer]);
        call.effects = Some(CallEffects::pure().with_access(
            CallLocation::Pointee {
                arg: 1,
                extent: PointeeExtent::Bytes(8),
            },
            AccessMode::Update,
        ));
        b.emit_no_defs(call.into())?;
        let result = b.load_indirect(SsaDefSpec::tmp(MockType::U32), buffer, MockType::U32)?;
        b.ret(Some(result))
    })?;
    let body = b.finish_verified(VerifyLevel::Full)?;
    let call = body
        .iter_instructions()
        .find_map(|(block, index, instruction)| instruction.op().as_call().map(|_| (block, index)))
        .ok_or_else(|| Error::new("call"))?;
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&body, &context).with_host(&Fill).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(|value| value.constant())
            .and_then(ConstValue::as_u64),
        Some(0x09090909)
    );
    assert!(
        facts
            .call_arguments(call.0, call.1)
            .iter()
            .any(|argument| argument.argument == 0 && argument.bytes == [7; 4]),
        "the call retains exactly the pre-fill use, not the overwritten result"
    );
    Ok(())
}

struct Body {
    body: SsaFunction<MockTarget>,
    implicit: bool,
    tail: bool,
    promoted: bool,
}

impl ExecutionHost<MockTarget> for Body {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 2).then_some(&self.body)
    }

    fn arguments(
        &self,
        _: CallSite<'_, MockTarget>,
        _: Option<&u32>,
        _: Option<&SsaFunction<MockTarget>>,
        _: Option<&dyn CallModel<MockTarget>>,
        arguments: &mut ArgumentBindings<'_, MockTarget>,
    ) {
        arguments.begin();
        if self.promoted {
            if !arguments.stage_ordinary_stack(
                0,
                None,
                Some(0),
                MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size),
            ) {
                return;
            }
            let _ = arguments.push_stack(None, 0, &MockType::U64, &MockType::U32);
        } else if self.tail {
            let _ = arguments.push_stack(None, 0, &MockType::U32, &MockType::U32);
        } else {
            arguments.push_ordinary(
                0,
                MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size),
                &MockType::U32,
            );
        }
    }

    fn call_stack_from_entry(&self, _: CallSite<'_, MockTarget>) -> Option<i64> {
        self.promoted.then_some(0)
    }

    fn parameters(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        body: &SsaFunction<MockTarget>,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        for variable in body.variables() {
            if variable.origin() != VariableOrigin::Argument(0) {
                continue;
            }
            if self.promoted {
                if !bindings.bind_stack(variable.id(), 0, &MockType::U64) {
                    return;
                }
            } else if self.implicit {
                bindings.bind_implicit(variable.id(), CallInputKey::new(7));
            } else if self.tail {
                bindings.bind(variable.id(), 0);
            } else {
                bindings.bind_physical_register_view(variable.id(), 0, &MockType::U64, MACHINE);
            }
        }
    }
}

fn returning(ty: MockType) -> Result<SsaFunction<MockTarget>> {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let value =
        b.function_mut()
            .create_variable(VariableOrigin::Argument(0), 0, DefSite::entry(), ty);
    b.in_block(0, |b| b.ret(Some(value)))?;
    b.finish_verified(VerifyLevel::Full)
}

#[test]
fn logical_narrowing_preserves_physical_bits_and_implicit_key_ranges() -> Result<()> {
    let wide = 0x123456780000002a;
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    b.in_block(0, |b| {
        let source = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(wide))?;
        let hidden = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(15))?;
        let unused = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(9))?;
        let result = b.emit_def(SsaDefSpec::tmp(MockType::U64), |dest| {
            let mut call = CallData::new(Some(dest), Callee::Direct(2), vec![source, unused]);
            call.implicit_inputs = vec![CallInput {
                key: CallInputKey::new(7),
                value: hidden,
            }]
            .into();
            call.staged = vec![StagedValue {
                offset: 0,
                value: unused,
            }];
            call.into()
        })?;
        b.ret(Some(result))
    })?;
    let caller = b.finish_verified(VerifyLevel::Full)?;
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    for implicit in [false, true] {
        let host = Body {
            body: returning(MockType::U64)?,
            implicit,
            tail: false,
            promoted: false,
        };
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(
            facts
                .returned()
                .and_then(|value| value.constant())
                .and_then(ConstValue::as_u64),
            Some(if implicit { 15 } else { wide })
        );
    }
    Ok(())
}

#[test]
fn tail_argument_reads_live_stack_before_frame_retirement() -> Result<()> {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = b.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    b.function_mut()
        .variable_mut(stack)
        .ok_or_else(|| Error::new("stack"))?
        .set_role(VariableRole::EntryStackPointer);
    b.in_block(0, |b| {
        let value = b.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
        b.store_indirect(stack, value, MockType::U32)?;
        let result = b.emit_def(SsaDefSpec::tmp(MockType::U32), |dest| {
            let mut call = CallData::new(Some(dest), Callee::Direct(2), vec![]);
            call.tail = true;
            call.into()
        })?;
        b.ret(Some(result))
    })?;
    let caller = b.finish_verified(VerifyLevel::Full)?;
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let host = Body {
        body: returning(MockType::U32)?,
        implicit: false,
        tail: true,
        promoted: false,
    };
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(|value| value.constant())
            .and_then(ConstValue::as_u64),
        Some(42)
    );
    Ok(())
}

/// Returns precisely the model's logical argument, independently from native entry binding.
struct Echo;

impl CallModel<MockTarget> for Echo {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        if let Some(value) = call.argument(0).cloned() {
            call.return_value(value);
        }
    }
}

impl ExecutionHost<MockTarget> for Echo {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        if *method == 2 { Some(self) } else { None }
    }
    fn arguments(
        &self,
        _: CallSite<'_, MockTarget>,
        _: Option<&u32>,
        _: Option<&SsaFunction<MockTarget>>,
        _: Option<&dyn CallModel<MockTarget>>,
        arguments: &mut ArgumentBindings<'_, MockTarget>,
    ) {
        arguments.begin();
        arguments.push_ordinary(
            0,
            MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size),
            &MockType::U32,
        );
    }
}

#[test]
fn selected_model_receives_logical_low_bits_instead_of_physical_upper_bits() -> Result<()> {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    b.in_block(0, |b| {
        let wide = b.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(0x123456780000002a),
        )?;
        let result = b.call(
            SsaDefSpec::tmp(MockType::U32),
            Callee::Direct(2),
            vec![wide],
        )?;
        b.ret(Some(result))
    })?;
    let body = b.finish_verified(VerifyLevel::Full)?;
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&body, &context).with_host(&Echo).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(|value| value.constant())
            .and_then(ConstValue::as_u64),
        Some(42)
    );
    Ok(())
}

struct Unanswered {
    argument_called: Cell<bool>,
}

impl ExecutionHost<MockTarget> for Unanswered {
    fn arguments(
        &self,
        _: CallSite<'_, MockTarget>,
        _: Option<&u32>,
        _: Option<&SsaFunction<MockTarget>>,
        _: Option<&dyn CallModel<MockTarget>>,
        _: &mut ArgumentBindings<'_, MockTarget>,
    ) {
        self.argument_called.set(true);
    }
}

#[test]
fn unanswered_calls_keep_source_operand_effects() -> Result<()> {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = b.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    b.function_mut()
        .variable_mut(stack)
        .ok_or_else(|| Error::new("stack"))?
        .set_role(VariableRole::EntryStackPointer);
    b.in_block(0, |b| {
        let before = b.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
        b.store_indirect(stack, before, MockType::U32)?;
        let mut call = CallData::new(None, Callee::Direct(99), vec![stack]);
        call.effects = Some(CallEffects::pure().with_access(
            CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::Bytes(4),
            },
            AccessMode::MayWrite,
        ));
        b.emit_no_defs(call.into())?;
        let after = b.load_indirect(SsaDefSpec::tmp(MockType::U32), stack, MockType::U32)?;
        b.ret(Some(after))
    })?;
    let body = b.finish_verified(VerifyLevel::Full)?;
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let host = Unanswered {
        argument_called: Cell::new(false),
    };
    let facts = StaticExecution::new(&body, &context).with_host(&host).run();
    assert!(!host.argument_called.get());
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(
        facts
            .returned()
            .and_then(|value| value.constant())
            .is_none()
    );
    Ok(())
}

#[test]
fn source_promoted_stack_stores_preserve_physical_bits_before_logical_narrowing() -> Result<()> {
    let wide = 0x123456780000002a;
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    b.in_block(0, |b| {
        let source = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(wide))?;
        let result = b.call(
            SsaDefSpec::tmp(MockType::U64),
            Callee::Direct(2),
            vec![source],
        )?;
        b.ret(Some(result))
    })?;
    let caller = b.finish_verified(VerifyLevel::Full)?;
    let host = Body {
        body: returning(MockType::U64)?,
        implicit: false,
        tail: false,
        promoted: true,
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(|value| value.constant())
            .and_then(ConstValue::as_u64),
        Some(wide),
    );
    Ok(())
}

struct IgnoredStagingFailure {
    offset: Option<i64>,
}

impl ExecutionHost<MockTarget> for IgnoredStagingFailure {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 2).then_some(&Echo)
    }

    fn arguments(
        &self,
        _: CallSite<'_, MockTarget>,
        _: Option<&u32>,
        _: Option<&SsaFunction<MockTarget>>,
        _: Option<&dyn CallModel<MockTarget>>,
        bindings: &mut ArgumentBindings<'_, MockTarget>,
    ) {
        bindings.begin();
        let layout = MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size);
        let _ = bindings.stage_ordinary_stack(0, None, self.offset, layout);
        // Replacing the logical list cannot undo the failed memory operation.
        bindings.begin();
        bindings.push_ordinary(0, layout, &MockType::U64);
    }
}

#[test]
fn ignored_staging_refusal_and_exhaustion_withhold_model_results() -> Result<()> {
    let mut b = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    b.in_block(0, |b| {
        let source = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(42))?;
        let result = b.call(
            SsaDefSpec::tmp(MockType::U64),
            Callee::Direct(2),
            vec![source],
        )?;
        b.ret(Some(result))
    })?;
    let caller = b.finish_verified(VerifyLevel::Full)?;
    for limited in [false, true] {
        let mut machine = MACHINE;
        machine.byte_order = Some(Endianness::Little);
        let budget = if limited {
            WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, 4)
        } else {
            WorkBudget::DEFAULT
        };
        let context = AnalysisContext::new(machine).with_budget(budget);
        let host = IgnoredStagingFailure {
            offset: limited.then_some(0),
        };
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        if limited {
            assert!(
                matches!(facts.completion(), Completion::Exhausted(exhaustion)
                if exhaustion.budget == Budget::ExecutionBufferBytes
                    && exhaustion.limit == 4 && exhaustion.demand == 8)
            );
        } else {
            assert!(matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::UnresolvedWrite,
                    ..
                }
            ));
        }
        assert!(facts.returned().is_none());
    }
    Ok(())
}
