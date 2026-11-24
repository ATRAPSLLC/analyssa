//! Source branches preserve physical state without introducing a language call.

use std::{cell::Cell, process, result::Result as StdResult, sync::Arc};

use analyssa::{
    Result,
    analysis::{
        VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            BranchContinuation, BranchSite, CallModel, CallSite, CodeMemory, Completion,
            ContinuationBinding, ExecutionHost, MemoryRefusal, ModelCall, ReturnBinding,
            StaticExecution, Value,
        },
    },
    ir::{
        Callee, ConstValue, SsaOp, SsaVarId,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

type Function = SsaFunction<MockTarget>;

struct Allocate;

impl CallModel<MockTarget> for Allocate {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let address = call.allocate_block(Some(8), false, None);
        call.return_value(Value::Addr(address));
    }
}

fn input(builder: &mut SsaFunctionBuilder<MockTarget>, ty: MockType) -> SsaVarId {
    builder
        .function_mut()
        .create_variable(VariableOrigin::EntryLiveIn, 0, DefSite::entry(), ty)
}

fn stack(builder: &mut SsaFunctionBuilder<MockTarget>) -> Result<SsaVarId> {
    let id = input(builder, MockType::Ptr);
    builder
        .function_mut()
        .variable_mut(id)
        .unwrap_or_else(|| process::abort())
        .set_role(VariableRole::EntryStackPointer);
    Ok(id)
}

struct First {
    function: Function,
    slot: SsaVarId,
    heap: SsaVarId,
    marker: SsaVarId,
}

fn first() -> Result<First> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let sp = stack(&mut builder)?;
    let (slot, heap, marker) = builder.in_block(0, |block| {
        let marker = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(40))?;
        let slot = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: sp,
            index: None,
            stride: 1,
            offset: -8,
            result_type: MockType::Ptr,
        })?;
        block.store_indirect(slot, marker, MockType::I64)?;
        let heap = block.call(SsaDefSpec::tmp(MockType::Ptr), Callee::Direct(7), vec![])?;
        let one = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
        block.store_indirect(heap, one, MockType::I64)?;
        let counter =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x3000))?;
        let before = block.load_indirect(SsaDefSpec::tmp(MockType::I64), counter, MockType::I64)?;
        let next = block.add(SsaDefSpec::tmp(MockType::I64), before, one)?;
        block.store_indirect(counter, next, MockType::I64)?;
        let target = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(200))?;
        block.emit_no_defs(SsaOp::IndirectBranch {
            target,
            resolved_targets: vec![],
            state: Box::default(),
        })?;
        Ok((slot, heap, marker))
    })?;
    Ok(First {
        function: builder.finish_verified(VerifyLevel::Full)?,
        slot,
        heap,
        marker,
    })
}

struct Next {
    function: Arc<Function>,
    slot: SsaVarId,
    heap: SsaVarId,
    float: SsaVarId,
}

fn next() -> Result<Next> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let slot = input(&mut builder, MockType::Ptr);
    let heap = input(&mut builder, MockType::Ptr);
    let float = input(&mut builder, MockType::F64);
    builder.in_block(0, |block| {
        let word = block.load_indirect(SsaDefSpec::tmp(MockType::I64), slot, MockType::I64)?;
        let another = block.call(SsaDefSpec::tmp(MockType::Ptr), Callee::Direct(7), vec![])?;
        let two = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(2))?;
        block.store_indirect(another, two, MockType::I64)?;
        let prior_heap =
            block.load_indirect(SsaDefSpec::tmp(MockType::I64), heap, MockType::I64)?;
        let counter =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x3000))?;
        let count = block.load_indirect(SsaDefSpec::tmp(MockType::I64), counter, MockType::I64)?;
        let sum = block.add(SsaDefSpec::tmp(MockType::I64), word, prior_heap)?;
        let sum = block.add(SsaDefSpec::tmp(MockType::I64), sum, count)?;
        block.ret(Some(sum))
    })?;
    Ok(Next {
        function: Arc::new(builder.finish_verified(VerifyLevel::Full)?),
        slot,
        heap,
        float,
    })
}

fn next_with_missing_pointer() -> Result<Next> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let slot = input(&mut builder, MockType::Ptr);
    let heap = input(&mut builder, MockType::Ptr);
    let float = input(&mut builder, MockType::F64);
    let missing = input(&mut builder, MockType::Ptr);
    builder.in_block(0, |block| {
        let zero = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(0))?;
        block.store_indirect(missing, zero, MockType::I64)?;
        let word = block.load_indirect(SsaDefSpec::tmp(MockType::I64), slot, MockType::I64)?;
        block.ret(Some(word))
    })?;
    Ok(Next {
        function: Arc::new(builder.finish_verified(VerifyLevel::Full)?),
        slot,
        heap,
        float,
    })
}

struct Host {
    first: First,
    next: Next,
    allocate: Allocate,
    owned: bool,
    invalid: u8,
    requests: Cell<usize>,
    cancelled: Cell<bool>,
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&Function> {
        match method {
            100 => Some(&self.first.function),
            200 if !self.owned => Some(&self.next.function),
            _ => None,
        }
    }

    fn body_snapshot(
        &self,
        method: &u32,
        _: CodeMemory<'_>,
    ) -> StdResult<Option<Arc<Function>>, MemoryRefusal> {
        Ok((*method == 200 && self.owned).then(|| Arc::clone(&self.next.function)))
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 7).then_some(&self.allocate as &dyn CallModel<MockTarget>)
    }

    fn body_return_binding(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        _: &Function,
    ) -> ReturnBinding {
        ReturnBinding::Word(64)
    }

    fn branch_continuation(
        &self,
        site: BranchSite<'_, MockTarget>,
        target: &Value<MockTarget>,
        _: CodeMemory<'_>,
    ) -> StdResult<Option<BranchContinuation<MockTarget>>, MemoryRefusal> {
        self.requests.set(self.requests.get().saturating_add(1));
        assert_eq!(site.method, Some(&100));
        assert_eq!(
            target.address().and_then(|address| address.image_address()),
            Some(200)
        );
        if self.invalid == 4 {
            return Err(MemoryRefusal::InstructionSource);
        }
        if self.invalid == 5 {
            self.cancelled.set(true);
        }
        let mut bindings = vec![
            ContinuationBinding {
                from: self.first.slot,
                into: self.next.slot,
                widened: None,
            },
            ContinuationBinding {
                from: self.first.heap,
                into: self.next.heap,
                widened: None,
            },
        ];
        match self.invalid {
            1 => {
                if let Some(binding) = bindings.first().copied() {
                    bindings.push(binding);
                }
            }
            2 => bindings.push(ContinuationBinding {
                from: self.first.marker,
                into: self.next.float,
                widened: None,
            }),
            3 => bindings.push(ContinuationBinding {
                from: SsaVarId::from_index(10000),
                into: self.next.float,
                widened: None,
            }),
            _ => {}
        }
        Ok(Some(BranchContinuation {
            method: 200,
            bindings,
            stack: None,
            frame: Vec::new(),
        }))
    }
}

fn host(owned: bool, invalid: u8) -> Result<Host> {
    Ok(Host {
        first: first()?,
        next: if invalid == 6 {
            next_with_missing_pointer()?
        } else {
            next()?
        },
        allocate: Allocate,
        owned,
        invalid,
        requests: Cell::new(0),
        cancelled: Cell::new(false),
    })
}

fn root() -> Result<Function> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let counter =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x3000))?;
        let zero = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(0))?;
        block.store_indirect(counter, zero, MockType::I64)?;
        let returned = block.call(SsaDefSpec::tmp(MockType::I64), Callee::Direct(100), vec![])?;
        let one = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
        let answer = block.add(SsaDefSpec::tmp(MockType::I64), returned, one)?;
        block.ret(Some(answer))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

#[test]
fn branch_continuations_keep_stack_heap_globals_return_and_depth_without_a_prefix_rerun()
-> Result<()> {
    for owned in [false, true] {
        let host = host(owned, 0)?;
        let weak = Arc::downgrade(&host.next.function);
        let root = root()?;
        let mut machine = MACHINE;
        machine.byte_order = Some(Endianness::Little);
        let context = AnalysisContext::new(machine)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionDepth, 2));
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_call_observations(&[(0, 3)])
            .run();
        assert_eq!(facts.completion(), Completion::Complete, "owned={owned}");
        assert_eq!(
            facts
                .returned()
                .and_then(Value::constant)
                .and_then(ConstValue::as_i64),
            Some(43)
        );
        assert_eq!(host.requests.get(), 1);
        assert_eq!(
            facts.invocations().len(),
            1,
            "a source branch is not a call observation"
        );
        assert_eq!(
            facts
                .invocations()
                .first()
                .and_then(|call| call.facts.returned())
                .and_then(Value::constant)
                .and_then(ConstValue::as_i64),
            Some(42)
        );
        drop(facts);
        drop(host);
        assert!(weak.upgrade().is_none(), "source body owner released");
    }
    Ok(())
}

#[test]
fn malformed_source_bindings_and_refused_fetches_withhold_the_trial() -> Result<()> {
    for invalid in 1..=4 {
        let host = host(false, invalid)?;
        let root = root()?;
        let mut machine = MACHINE;
        machine.byte_order = Some(Endianness::Little);
        let context = AnalysisContext::new(machine);
        let facts = StaticExecution::new(&root, &context).with_host(&host).run();
        assert!(!facts.is_complete(), "invalid={invalid}");
        assert!(facts.returned().is_none(), "invalid={invalid}");
        assert!(facts.constants().is_empty(), "invalid={invalid}");
        assert!(matches!(
            facts.completion(),
            Completion::UnstableCode
                | Completion::Memory {
                    reason: MemoryRefusal::InstructionSource,
                    ..
                }
        ));
    }
    Ok(())
}

#[test]
fn cancellation_after_source_discovery_withholds_the_trial() -> Result<()> {
    let host = host(false, 5)?;
    let root = root()?;
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let cancelled = || host.cancelled.get();
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_cancellation(&cancelled)
        .run();
    assert_eq!(facts.completion(), Completion::Cancelled);
    assert!(facts.returned().is_none());
    Ok(())
}

struct Cycle(Function);

impl ExecutionHost<MockTarget> for Cycle {
    fn body(&self, _: &u32) -> Option<&Function> {
        Some(&self.0)
    }
    fn branch_continuation(
        &self,
        _: BranchSite<'_, MockTarget>,
        _: &Value<MockTarget>,
        _: CodeMemory<'_>,
    ) -> StdResult<Option<BranchContinuation<MockTarget>>, MemoryRefusal> {
        Ok(Some(BranchContinuation {
            method: 100,
            bindings: vec![],
            stack: None,
            frame: Vec::new(),
        }))
    }
}

#[test]
fn repeated_source_transfers_stop_at_the_execution_limit_without_growing_call_depth() -> Result<()>
{
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let target = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(100))?;
        block.emit_no_defs(SsaOp::IndirectBranch {
            target,
            resolved_targets: vec![],
            state: Box::default(),
        })
    })?;
    let host = Cycle(builder.finish_verified(VerifyLevel::Full)?);
    let context = AnalysisContext::new(MACHINE).with_budget(
        WorkBudget::DEFAULT
            .with_limit(Budget::ExecutionSteps, 32)
            .with_limit(Budget::ExecutionDepth, 1),
    );
    let facts = StaticExecution::new(&host.0, &context)
        .with_method(100)
        .with_host(&host)
        .run();
    assert!(
        matches!(facts.completion(), Completion::Exhausted(exhaustion) if exhaustion.budget == Budget::ExecutionSteps)
    );
    assert!(facts.returned().is_none());
    Ok(())
}

#[test]
fn missing_machine_input_cannot_exclude_an_alias_with_the_preserved_private_stack() -> Result<()> {
    let host = host(false, 6)?;
    let root = root()?;
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert_eq!(facts.completion(), Completion::Complete);
    assert!(
        facts.returned().and_then(Value::constant).is_none(),
        "an omitted register may hold the stack pointer created by the prefix: {:?}",
        facts.returned()
    );
    Ok(())
}
