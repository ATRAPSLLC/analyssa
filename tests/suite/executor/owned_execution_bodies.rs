//! Owned body versions execute inside the existing state and call continuations.

use std::{
    cell::{Cell, RefCell},
    ptr,
    sync::{Arc, Weak},
};

use analyssa::{
    Result,
    analysis::{
        VerifyLevel,
        context::AnalysisContext,
        execution::{
            Address, CallModel, CallSite, CodeMemory, Completion, ExecutionHost,
            InstructionSourceId, InstructionSources, MemoryRefusal, MemoryWriteGuard, ModelCall,
            Root, StaticExecution, Value,
        },
    },
    ir::{
        CallData, Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
        variable::{DefSite, VariableOrigin},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

const CODE: u64 = 0x2000;

fn parameter(builder: &mut SsaFunctionBuilder<MockTarget>, index: u16) -> analyssa::ir::SsaVarId {
    builder.function_mut().create_variable(
        VariableOrigin::Argument(index),
        0,
        DefSite::entry(),
        MockType::Ptr,
    )
}

fn snapshot(
    byte: u8,
    root: Root,
    overwrite_code: bool,
    tail_model: bool,
) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::new(1, 0);
    let data = parameter(&mut builder, 0);
    builder.in_block(0, |block| {
        let reached = block.load_indirect(SsaDefSpec::tmp(MockType::U8), data, MockType::U8)?;
        let decoded = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(byte))?;
        if overwrite_code {
            let code = block.const_value(
                SsaDefSpec::tmp(MockType::Ptr),
                match root {
                    Root::Symbol => ConstValue::Symbol(CODE as u32),
                    _ => ConstValue::NativeUInt(CODE),
                },
            )?;
            let patch = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0xff))?;
            block.store_indirect(code, patch, MockType::U8)?;
        }
        let result = block.add(SsaDefSpec::tmp(MockType::U8), decoded, reached)?;
        if tail_model {
            let result = block.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| {
                CallData::new(Some(dest), Callee::Direct(88), vec![])
                    .made_tail()
                    .into()
            })?;
            block.ret(Some(result))
        } else {
            block.ret(Some(result))
        }
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

struct Host {
    wrapper: SsaFunction<MockTarget>,
    prefix_runs: Cell<usize>,
    probes: RefCell<Vec<usize>>,
    retired: Cell<usize>,
    overwrite_code: bool,
    tail_model: bool,
    snapshots: RefCell<Vec<(u8, Weak<SsaFunction<MockTarget>>)>>,
}

impl Host {
    fn source(&self, body: &SsaFunction<MockTarget>) -> Option<InstructionSourceId> {
        self.snapshots
            .borrow()
            .iter()
            .position(|(_, known)| ptr::eq(known.as_ptr(), body))
            .and_then(|index| u64::try_from(index).ok())
            .map(InstructionSourceId::versioned)
    }
}

#[derive(Debug)]
struct CodeGuard;

impl MemoryWriteGuard for CodeGuard {
    fn check_write(
        &self,
        sources: &mut InstructionSources,
        address: Address,
        len: usize,
    ) -> std::result::Result<(), MemoryRefusal> {
        if matches!(address.root(), Root::Number | Root::Symbol) {
            let first = address.linear_prefix_len(len);
            let overlaps = CODE
                .checked_sub(address.offset().cast_unsigned())
                .is_some_and(|delta| delta < first as u64)
                || (len > first && CODE < len.saturating_sub(first) as u64);
            if overlaps
                && [0, 1]
                    .into_iter()
                    .any(|id| sources.is_active(InstructionSourceId::versioned(id)))
            {
                return Err(MemoryRefusal::InstructionSource);
            }
        }
        Ok(())
    }
}

impl ExecutionHost<MockTarget> for Host {
    fn method_for_call(
        &self,
        site: CallSite<'_, MockTarget>,
        method: &u32,
        memory: CodeMemory<'_>,
    ) -> Option<u32> {
        if *method != 10 {
            return Some(*method);
        }
        let declared = site
            .caller
            .block(site.block)
            .and_then(|block| block.instruction(site.instruction))
            .and_then(|instruction| instruction.op().as_call())
            .and_then(|call| match &call.callee {
                Callee::Direct(method) => Some(*method),
                _ => None,
            });
        assert_eq!(
            declared,
            Some(10),
            "lookup receives the actual source call site"
        );
        let bytes = memory
            .read(Address::new(Root::Number, CODE as i64, MACHINE.ptr_size), 1)
            .ok()??;
        match bytes.first().copied()? {
            40 => Some(1),
            2 => Some(4),
            _ => None,
        }
    }

    fn admit_code_source(
        &self,
        body: &SsaFunction<MockTarget>,
        method: Option<&u32>,
        sources: &mut InstructionSources,
    ) -> bool {
        if method.is_some_and(|method| matches!(*method, 1 | 3 | 4)) {
            self.source(body)
                .is_some_and(|source| sources.enter_active(source))
        } else {
            true // Independently supplied semantic caller and wrapper.
        }
    }

    fn retire_code_source(
        &self,
        body: &SsaFunction<MockTarget>,
        method: Option<&u32>,
        sources: &mut InstructionSources,
    ) -> bool {
        if method.is_some_and(|method| matches!(*method, 1 | 3 | 4)) {
            let retired = self
                .source(body)
                .is_some_and(|source| sources.retire_active(source));
            if retired {
                self.retired.set(self.retired.get().saturating_add(1));
            }
            retired
        } else {
            true
        }
    }

    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        Some(&CodeGuard)
    }

    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 2).then_some(&self.wrapper)
    }

    fn body_snapshot(
        &self,
        method: &u32,
        memory: CodeMemory<'_>,
    ) -> std::result::Result<Option<Arc<SsaFunction<MockTarget>>>, MemoryRefusal> {
        let root = match *method {
            1 | 4 => analyssa::analysis::execution::Root::Number,
            3 => analyssa::analysis::execution::Root::Symbol,
            _ => return Ok(None),
        };
        let address = Address::new(root, CODE as i64, MACHINE.ptr_size);
        let inspected = memory.inspect_prefix(address, 8)?;
        self.probes.borrow_mut().push(inspected.len());
        let Some(bytes) = memory.read(address, 1)? else {
            return Ok(None);
        };
        let Some(&byte) = bytes.first() else {
            return Ok(None);
        };
        let body = Arc::new(
            snapshot(byte, root, self.overwrite_code, self.tail_model)
                .map_err(|_| MemoryRefusal::BlockLayout)?,
        );
        self.snapshots
            .borrow_mut()
            .push((byte, Arc::downgrade(&body)));
        Ok(Some(body))
    }

    fn method_at_address(&self, target: Address, _memory: CodeMemory<'_>) -> Option<u32> {
        if target.offset() != CODE as i64 {
            return None;
        }
        match target.root() {
            analyssa::analysis::execution::Root::Number => Some(1),
            analyssa::analysis::execution::Root::Symbol => Some(3),
            _ => None,
        }
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match *method {
            99 => Some(self),
            88 => Some(&TailPatch),
            _ => None,
        }
    }

    fn admit_unevaluated_call(&self, _site: CallSite<'_, MockTarget>) -> bool {
        false
    }
}

struct TailPatch;

impl CallModel<MockTarget> for TailPatch {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.write(
            Address::new(Root::Symbol, CODE as i64, call.ptr_size()),
            &[0xff],
        );
        call.return_value(Value::Const(ConstValue::U8(42)));
    }
}

impl CallModel<MockTarget> for Host {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        self.prefix_runs
            .set(self.prefix_runs.get().saturating_add(1));
        let data = call.allocate_block(Some(1), false, None);
        call.return_value(Value::Addr(data));
    }
}

fn wrapper() -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::new(2, 0);
    let code = parameter(&mut builder, 0);
    let data = parameter(&mut builder, 1);
    builder.in_block(0, |block| {
        let result = block.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| {
            CallData::new(
                Some(dest),
                Callee::Indirect {
                    fptr: code,
                    signature: 0,
                },
                vec![data],
            )
            .made_tail()
            .into()
        })?;
        block.ret(Some(result))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

fn caller(known: bool, tail: bool, direct: bool) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let allocated = block.call(SsaDefSpec::tmp(MockType::Ptr), Callee::Direct(99), vec![])?;
        let data = if tail {
            allocated
        } else {
            block.const_value(
                SsaDefSpec::tmp(MockType::Ptr),
                ConstValue::NativeUInt(0x2100),
            )?
        };
        let code = block.const_value(
            SsaDefSpec::tmp(MockType::Ptr),
            if tail {
                ConstValue::Symbol(0x2000)
            } else {
                ConstValue::NativeUInt(CODE)
            },
        )?;
        let two = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
        block.store_indirect(data, two, MockType::U8)?;
        if known {
            let first = block.const_value(
                SsaDefSpec::tmp(MockType::U64),
                ConstValue::U64(0x0807060504030228),
            )?;
            block.store_indirect(code, first, MockType::U64)?;
        }
        let first = block.call(
            SsaDefSpec::tmp(MockType::U8),
            if tail {
                Callee::Direct(2)
            } else if direct {
                Callee::Direct(10)
            } else {
                Callee::Indirect {
                    fptr: code,
                    signature: 0,
                }
            },
            if tail { vec![code, data] } else { vec![data] },
        )?;
        let second = if tail {
            block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(4))?
        } else {
            block.store_indirect(code, two, MockType::U8)?;
            block.call(
                SsaDefSpec::tmp(MockType::U8),
                if direct {
                    Callee::Direct(10)
                } else {
                    Callee::Indirect {
                        fptr: code,
                        signature: 0,
                    }
                },
                vec![data],
            )?
        };
        let result = block.add(SsaDefSpec::tmp(MockType::U8), first, second)?;
        block.ret(Some(result))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

#[test]
fn owned_versions_keep_reaching_memory_and_original_return_continuation() -> Result<()> {
    let root = caller(true, false, false)?;
    let host = Host {
        wrapper: wrapper()?,
        prefix_runs: Cell::new(0),
        probes: RefCell::new(Vec::new()),
        retired: Cell::new(0),
        overwrite_code: false,
        tail_model: false,
        snapshots: RefCell::new(Vec::new()),
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_root_call_observations(4)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64),
        Some(46)
    );
    assert_eq!(host.prefix_runs.get(), 1);
    assert_eq!(
        host.retired.get(),
        2,
        "both source versions retire before caller writes"
    );
    let versions = host.snapshots.borrow();
    assert_eq!(
        versions.iter().map(|(byte, _)| *byte).collect::<Vec<_>>(),
        vec![40, 2]
    );
    assert!(
        versions.iter().all(|(_, weak)| weak.upgrade().is_none()),
        "owned execution bodies must be released after execution"
    );
    assert_eq!(facts.invocations().len(), 2);
    assert_ne!(
        facts.invocations().first().map(|call| call.body),
        facts.invocations().get(1).map(|call| call.body)
    );
    assert_eq!(
        facts
            .invocations()
            .iter()
            .map(|entry| entry
                .facts
                .returned()
                .and_then(Value::constant)
                .and_then(ConstValue::as_u64))
            .collect::<Vec<_>>(),
        vec![Some(42), Some(4)]
    );
    Ok(())
}

#[test]
fn an_owned_tail_body_returns_through_the_retired_borrowed_wrapper() -> Result<()> {
    let root = caller(true, true, false)?;
    let host = Host {
        wrapper: wrapper()?,
        prefix_runs: Cell::new(0),
        probes: RefCell::new(Vec::new()),
        retired: Cell::new(0),
        overwrite_code: false,
        tail_model: false,
        snapshots: RefCell::new(Vec::new()),
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_root_call_observations(4)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64),
        Some(46)
    );
    assert_eq!(host.prefix_runs.get(), 1);
    assert_eq!(host.snapshots.borrow().len(), 1);
    assert_eq!(host.retired.get(), 1, "the tail body retires exactly once");
    assert_eq!(
        host.probes.borrow().as_slice(),
        &[8],
        "decoding inspected the full known prefix"
    );
    let observations = facts.calls();
    let code = observations
        .iter()
        .flat_map(|(_, arguments)| arguments.iter())
        .find(|argument| argument.argument == 0 && argument.address.offset() == CODE as i64);
    assert_eq!(
        code.map(|argument| argument.bytes.as_slice()),
        Some([40].as_slice()),
        "the enclosing call observes exactly the code bytes the host fetched, not an empty or eight-byte use"
    );
    Ok(())
}

#[test]
fn an_unknown_code_byte_cannot_supply_an_owned_body() -> Result<()> {
    let root = caller(false, false, false)?;
    let host = Host {
        wrapper: wrapper()?,
        prefix_runs: Cell::new(0),
        probes: RefCell::new(Vec::new()),
        retired: Cell::new(0),
        overwrite_code: false,
        tail_model: false,
        snapshots: RefCell::new(Vec::new()),
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert_eq!(facts.completion(), Completion::UnevaluatedEffects);
    assert!(facts.returned().is_none());
    assert!(host.snapshots.borrow().is_empty());
    assert_eq!(host.probes.borrow().as_slice(), &[0]);
    assert_eq!(host.prefix_runs.get(), 1);
    Ok(())
}

#[test]
fn an_active_owned_source_cannot_be_overwritten_before_it_retires() -> Result<()> {
    let root = caller(true, false, false)?;
    let host = Host {
        wrapper: wrapper()?,
        prefix_runs: Cell::new(0),
        probes: RefCell::new(Vec::new()),
        retired: Cell::new(0),
        overwrite_code: true,
        tail_model: false,
        snapshots: RefCell::new(Vec::new()),
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(matches!(
        facts.completion(),
        Completion::Memory {
            reason: MemoryRefusal::InstructionSource,
            ..
        }
    ));
    assert!(facts.returned().is_none());
    assert_eq!(host.prefix_runs.get(), 1);
    assert_eq!(host.snapshots.borrow().len(), 1);
    assert_eq!(host.retired.get(), 0);
    Ok(())
}

#[test]
fn a_tail_model_runs_after_source_retirement_and_does_not_retire_it_twice() -> Result<()> {
    let root = caller(true, true, false)?;
    let host = Host {
        wrapper: wrapper()?,
        prefix_runs: Cell::new(0),
        probes: RefCell::new(Vec::new()),
        retired: Cell::new(0),
        overwrite_code: false,
        tail_model: true,
        snapshots: RefCell::new(Vec::new()),
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64),
        Some(46)
    );
    assert_eq!(host.prefix_runs.get(), 1);
    assert_eq!(host.retired.get(), 1);
    assert_eq!(host.snapshots.borrow().len(), 1);
    Ok(())
}

#[test]
fn direct_calls_select_current_versions_before_binding_and_keep_the_original_continuation()
-> Result<()> {
    for known in [true, false] {
        let root = caller(known, false, true)?;
        let host = Host {
            wrapper: wrapper()?,
            prefix_runs: Cell::new(0),
            probes: RefCell::new(Vec::new()),
            retired: Cell::new(0),
            overwrite_code: false,
            tail_model: false,
            snapshots: RefCell::new(Vec::new()),
        };
        let mut machine = MACHINE;
        machine.byte_order = Some(Endianness::Little);
        let context = AnalysisContext::new(machine);
        let mut facts = StaticExecution::new(&root, &context).with_host(&host).run();
        assert_eq!(
            host.prefix_runs.get(),
            1,
            "the suspended prefix is not rerun"
        );
        if known {
            assert!(facts.is_complete(), "{:?}", facts.completion());
            assert_eq!(
                facts
                    .returned()
                    .and_then(Value::constant)
                    .and_then(ConstValue::as_u64),
                Some(46)
            );
            assert_eq!(facts.take_body_methods(), vec![None, Some(1), Some(4)]);
            assert_eq!(host.retired.get(), 2);
        } else {
            assert_eq!(facts.completion(), Completion::UnevaluatedEffects);
            assert!(facts.returned().is_none());
            assert!(host.snapshots.borrow().is_empty());
        }
    }
    Ok(())
}
