//! A native body returns where it was called from: the address its caller
//! leaves it, on the stack or in a link register, is what each return goes
//! back through.

use std::result::Result as StdResult;

use analyssa::{
    Endianness, Result,
    analysis::{
        Machine, VerifyLevel,
        context::AnalysisContext,
        execution::{
            Address, CallModel, CallSite, CallStack, Completion, ExecutionHost, FindingKind,
            MemoryRefusal, ModelCall, ReturnAddressStorage, ReturnContract, Root, StaticExecution,
            Value,
        },
    },
    ir::{
        CallData, Callee, ConstValue, ReturnTransfer, SsaOp, SsaVarId,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::Target,
    testing::{MACHINE, MockTarget, MockType},
};

/// The stack-held contract every body of these hosts has.
fn stack_contract() -> StdResult<Option<ReturnContract>, MemoryRefusal> {
    let layout = MockTarget::type_layout(&MockType::Ptr, MACHINE.ptr_size)
        .ok_or(MemoryRefusal::ReturnControl)?;
    Ok(Some(ReturnContract {
        layout,
        storage: ReturnAddressStorage::Stack,
    }))
}

/// The machine the bodies run on: little-endian, so the words they store
/// are bytes the run knows.
fn machine() -> Machine {
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    machine
}

/// The address a call at `0x1000` continues at.
fn continuation() -> Value<MockTarget> {
    Value::Addr(Address::new(Root::Symbol, 0x1005, MACHINE.ptr_size))
}

/// An entry value standing for the body's stack pointer.
fn stack(builder: &mut SsaFunctionBuilder<MockTarget>) -> SsaVarId {
    let value = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = builder.function_mut().variable_mut(value) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    value
}

/// A caller calling method 2, and the callee, on the stack-held contract.
struct Stacked {
    caller: SsaFunction<MockTarget>,
    callee: SsaFunction<MockTarget>,
    caller_stack: SsaVarId,
    tail: bool,
}

/// What the callee writes over the word it returns through.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Overwrite {
    /// Nothing.
    Nothing,
    /// A word the run knows: zero.
    Known,
    /// A word the run does not know: a register it entered with.
    Unknown,
    /// The word its caller returns through, one word above its own.
    CallersReturn,
}

/// The caller leaves a stale 17 where the call puts the return address and
/// calls (or tail-calls) the callee. The callee returns the word at its
/// entry stack pointer, after overwriting it as `overwrite` says, or 42 for
/// the tail call.
fn stacked(overwrite: Overwrite, tail: bool) -> Result<Stacked> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let caller_stack = stack(&mut builder);
    builder.in_block(0, |block| {
        if !tail {
            let below = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: caller_stack,
                index: None,
                stride: 1,
                offset: -8,
                result_type: MockType::Ptr,
            })?;
            let stale = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(17))?;
            block.store_indirect(below, stale, MockType::U64)?;
        }
        let result = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| {
            let mut call = CallData::new(Some(dest), Callee::Direct(2), vec![]);
            call.tail = tail;
            call.into()
        })?;
        block.ret(Some(result))
    })?;
    let caller = builder.finish_verified(VerifyLevel::Full)?;
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let entered = stack(&mut builder);
    let register = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::U64,
    );
    builder.in_block(0, |block| {
        let first = block.load_indirect(SsaDefSpec::tmp(MockType::Ptr), entered, MockType::Ptr)?;
        match overwrite {
            Overwrite::Nothing => {}
            Overwrite::Known => {
                let wrong =
                    block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                block.store_indirect(entered, wrong, MockType::U64)?;
            }
            Overwrite::Unknown => block.store_indirect(entered, register, MockType::U64)?,
            Overwrite::CallersReturn => {
                let above =
                    block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: entered,
                        index: None,
                        stride: 1,
                        offset: 8,
                        result_type: MockType::Ptr,
                    })?;
                let callers =
                    block.load_indirect(SsaDefSpec::tmp(MockType::Ptr), above, MockType::Ptr)?;
                block.store_indirect(entered, callers, MockType::Ptr)?;
            }
        }
        let returned = if tail {
            block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(42))?
        } else {
            first
        };
        block.ret(Some(returned))
    })?;
    let callee = builder.finish_verified(VerifyLevel::Full)?;
    Ok(Stacked {
        caller,
        callee,
        caller_stack,
        tail,
    })
}

impl ExecutionHost<MockTarget> for Stacked {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 2).then_some(&self.callee)
    }
    fn body_return(
        &self,
        _: &SsaFunction<MockTarget>,
        _: Option<&u32>,
    ) -> StdResult<Option<ReturnContract>, MemoryRefusal> {
        stack_contract()
    }
    fn call_stack(&self, _: CallSite<'_, MockTarget>) -> Option<CallStack> {
        Some(CallStack {
            stack_pointer: Some(self.caller_stack),
            adjustment: if self.tail { 0 } else { -8 },
        })
    }
    fn call_return_address(
        &self,
        _: CallSite<'_, MockTarget>,
    ) -> StdResult<Option<Value<MockTarget>>, MemoryRefusal> {
        Ok(Some(continuation()))
    }
}

fn run(host: &Stacked) -> analyssa::analysis::execution::StaticFacts<MockTarget> {
    let context = AnalysisContext::new(machine());
    StaticExecution::new(&host.caller, &context)
        .with_host(host)
        .with_method(1)
        .run()
}

/// The call puts its continuation where the callee's return reads it, over
/// whatever the caller left there.
#[test]
fn a_call_overwrites_stale_memory_with_its_actual_return_target() -> Result<()> {
    let facts = run(&stacked(Overwrite::Nothing, false)?);
    assert_eq!(facts.completion(), Completion::Complete);
    assert_eq!(
        facts
            .returned()
            .and_then(Value::address)
            .and_then(Address::image_address),
        Some(0x1005)
    );
    Ok(())
}

/// A callee that overwrote the word it returns through with a value the run
/// does not know may still return to its caller, which is where the program
/// goes unless the write changed the address: the run goes on there and
/// records that it could not check the return.
#[test]
fn an_unknown_return_word_goes_on_at_the_caller_and_is_found() -> Result<()> {
    let facts = run(&stacked(Overwrite::Unknown, false)?);
    assert_eq!(facts.completion(), Completion::Complete);
    assert!(
        facts
            .findings()
            .iter()
            .any(|finding| finding.kind == FindingKind::UnknownReturn),
        "{:?}",
        facts.findings()
    );
    Ok(())
}

/// A callee that overwrote the word it returns through with another known
/// word cannot resume its caller.
#[test]
fn a_corrupted_return_word_cannot_resume_the_caller() -> Result<()> {
    let facts = run(&stacked(Overwrite::Known, false)?);
    assert_eq!(
        facts.completion(),
        Completion::Memory {
            block: 0,
            reason: MemoryRefusal::ReturnControl
        }
    );
    Ok(())
}

/// A callee that returns through the word its caller returns through goes
/// back past its caller: a return the run knows, though only as the caller's
/// own, and one the caller cannot resume from.
#[test]
fn a_return_through_the_callers_return_word_cannot_resume_the_caller() -> Result<()> {
    let facts = run(&stacked(Overwrite::CallersReturn, false)?);
    assert_eq!(
        facts.completion(),
        Completion::Memory {
            block: 0,
            reason: MemoryRefusal::ReturnControl
        }
    );
    Ok(())
}

/// A tail call returns where its caller would, through the word its caller
/// was entered with, and pushes none of its own.
#[test]
fn a_native_tail_inherits_the_formal_root_obligation_without_pushing_another_word() -> Result<()> {
    let facts = run(&stacked(Overwrite::Nothing, true)?);
    assert_eq!(facts.completion(), Completion::Complete);
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64),
        Some(42)
    );
    Ok(())
}

/// How a link-register callee returns.
#[derive(Clone, Copy)]
enum Link {
    /// Through the register's entry value.
    Entry,
    /// Through another value, as a body that clobbered its link register.
    Other,
    /// Naming no target.
    Unnamed,
}

/// A caller calling method 2, and the callee, on a link-register contract,
/// each with its entry value of the register.
struct Linked {
    caller: SsaFunction<MockTarget>,
    callee: SsaFunction<MockTarget>,
    caller_link: SsaVarId,
    link: SsaVarId,
}

/// A body's entry value of its link register.
fn link_register(builder: &mut SsaFunctionBuilder<MockTarget>) -> SsaVarId {
    builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    )
}

/// The caller keeps its own return address across the call and returns
/// through it; the callee returns as `link` says.
fn linked(link: Link) -> Result<Linked> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let caller_link = link_register(&mut builder);
    builder.in_block(0, |block| {
        let result = block.call(SsaDefSpec::tmp(MockType::U64), Callee::Direct(2), vec![])?;
        block.emit_no_defs(SsaOp::Return {
            value: Some(result),
            state: Box::default(),
            results: Default::default(),
            transfer: Some(ReturnTransfer::through(caller_link)),
        })
    })?;
    let caller = builder.finish_verified(VerifyLevel::Full)?;
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let entry = link_register(&mut builder);
    builder.in_block(0, |block| {
        let value = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(42))?;
        let target = match link {
            Link::Entry => Some(entry),
            Link::Other => Some(
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x2004))?,
            ),
            Link::Unnamed => None,
        };
        block.emit_no_defs(SsaOp::Return {
            value: Some(value),
            state: Box::default(),
            results: Default::default(),
            transfer: target.map(ReturnTransfer::through),
        })
    })?;
    let callee = builder.finish_verified(VerifyLevel::Full)?;
    Ok(Linked {
        caller,
        callee,
        caller_link,
        link: entry,
    })
}

impl ExecutionHost<MockTarget> for Linked {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 2).then_some(&self.callee)
    }
    fn body_return(
        &self,
        body: &SsaFunction<MockTarget>,
        _: Option<&u32>,
    ) -> StdResult<Option<ReturnContract>, MemoryRefusal> {
        let layout = MockTarget::type_layout(&MockType::Ptr, MACHINE.ptr_size)
            .ok_or(MemoryRefusal::ReturnControl)?;
        let entry = Some(if std::ptr::eq(body, &self.callee) {
            self.link
        } else {
            self.caller_link
        });
        Ok(Some(ReturnContract {
            layout,
            storage: ReturnAddressStorage::Register { entry, key: None },
        }))
    }
    fn call_return_address(
        &self,
        _: CallSite<'_, MockTarget>,
    ) -> StdResult<Option<Value<MockTarget>>, MemoryRefusal> {
        Ok(Some(continuation()))
    }
}

/// A link-register callee returning through the register's entry value
/// returns to its caller; one returning through another value, or naming
/// none, cannot.
#[test]
fn a_link_register_return_goes_back_through_its_entry_value() -> Result<()> {
    let context = AnalysisContext::new(machine());
    for (link, returns) in [
        (Link::Entry, true),
        (Link::Other, false),
        (Link::Unnamed, false),
    ] {
        let host = linked(link)?;
        let facts = StaticExecution::new(&host.caller, &context)
            .with_host(&host)
            .with_method(1)
            .run();
        if returns {
            assert_eq!(facts.completion(), Completion::Complete);
            assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U64(42))));
        } else {
            assert_eq!(
                facts.completion(),
                Completion::Memory {
                    block: 0,
                    reason: MemoryRefusal::ReturnControl
                }
            );
        }
    }
    Ok(())
}

/// A model that writes `[argument 0]` where `corrupt` says, and returns 42.
struct Writes {
    corrupt: bool,
}
impl CallModel<MockTarget> for Writes {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        if self.corrupt
            && let Some(address) = call.argument(0).and_then(Value::address)
        {
            call.write(address, &[0; 8]);
        }
        call.return_value(Value::Const(ConstValue::U64(42)));
    }
    fn tracks_all_memory_accesses(&self) -> bool {
        true
    }
    fn only_memory(&self) -> bool {
        true
    }
}

/// Models method 2 by [`Writes`]; every body holds its return address on the
/// stack.
struct Modelled(Writes);

impl ExecutionHost<MockTarget> for Modelled {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 2).then_some(&self.0)
    }
    fn body_return(
        &self,
        _: &SsaFunction<MockTarget>,
        _: Option<&u32>,
    ) -> StdResult<Option<ReturnContract>, MemoryRefusal> {
        stack_contract()
    }
}

/// A root handing a model the address of its own return word: what the
/// model writes there is what the root returns through.
#[test]
fn a_model_that_overwrites_its_callers_return_word_ends_the_callers_return() -> Result<()> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let entered = stack(&mut builder);
    builder.in_block(0, |block| {
        let result = block.call(
            SsaDefSpec::tmp(MockType::U64),
            Callee::Direct(2),
            vec![entered],
        )?;
        block.ret(Some(result))
    })?;
    let root = builder.finish_verified(VerifyLevel::Full)?;
    let context = AnalysisContext::new(machine());
    for corrupt in [false, true] {
        let host = Modelled(Writes { corrupt });
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_method(1)
            .run();
        if corrupt {
            assert_eq!(
                facts.completion(),
                Completion::Memory {
                    block: 0,
                    reason: MemoryRefusal::ReturnControl
                }
            );
            assert_eq!(facts.returned(), None);
        } else {
            assert_eq!(facts.completion(), Completion::Complete);
            assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U64(42))));
        }
    }
    Ok(())
}
