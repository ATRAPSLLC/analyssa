//! Keyed call inputs retain their values independently of language arguments.

use std::{cell::RefCell, process};

use analyssa::{
    analysis::{
        SsaVerifier, VerifyLevel,
        consts::Machine,
        context::AnalysisContext,
        execution::{
            CallModel, CallSite, Completion, ExecutionHost, LibraryModel, ModelCall, ModelControl,
            ParameterBindings, StaticExecution, Value,
        },
    },
    ir::{
        CallData, CallEffects, CallInputKey, Callee, ConstValue, SsaOp, StagedValue,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::{Endianness, Target},
    testing::{MACHINE, MockTarget, MockType},
};

/// The test host's semantic input slot; it is unrelated to argument indices.
const KEY: CallInputKey = CallInputKey::new(17);

/// Explicit byte order for private memory fixtures.
const LITTLE_MACHINE: Machine = Machine {
    byte_order: Some(Endianness::Little),
    ..MACHINE
};

/// A hermetic typed SSA body.
type Function = SsaFunction<MockTarget>;

/// Finishes and verifies fixtures before any execution result is examined.
fn verified(builder: SsaFunctionBuilder<MockTarget>) -> Function {
    let body = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&body)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    body
}

/// Creates an ordinary argument and a distinct hidden input, returning their sum.
fn sum_body() -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let ordinary = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I64,
    );
    let hidden = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::I64,
    );
    builder
        .in_block(0, |b| {
            let result = b.add(SsaDefSpec::tmp(MockType::I64), ordinary, hidden)?;
            b.ret(Some(result))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// Calls a body or model with ordinary value one and hidden value forty-one.
fn sum_caller(kind: u8, tail: bool, keys: &[CallInputKey]) -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let one = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
            let old = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(16))?;
            let hidden = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(41))?;
            let target =
                b.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x5000))?;
            let result = b.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| {
                let callee = match kind {
                    0 => Callee::Direct(5),
                    1 => Callee::Indirect {
                        fptr: target,
                        signature: 0,
                    },
                    _ => Callee::Virtual(5),
                };
                let mut call = CallData::new(Some(dest), callee, vec![one]);
                for (index, key) in keys.iter().enumerate() {
                    call = call.with_implicit_input(
                        *key,
                        if index.saturating_add(1) == keys.len() {
                            hidden
                        } else {
                            old
                        },
                    );
                }
                call.tail = tail;
                call.into()
            })?;
            b.ret(Some(result))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// Binds input slots using an explicit host contract, with no root assumptions.
struct BodyHost {
    body: Function,
    key: CallInputKey,
}

impl ExecutionHost<MockTarget> for BodyHost {
    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 5).then_some(&self.body)
    }

    fn parameters(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        body: &Function,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        for variable in body.variables() {
            match variable.origin() {
                VariableOrigin::Argument(0) => bindings.bind(variable.id(), 0),
                VariableOrigin::EntryLiveIn => bindings.bind_implicit(variable.id(), self.key),
                _ => {}
            }
        }
    }

    fn method_at(&self, address: u64) -> Option<u32> {
        (address == 0x5000).then_some(5)
    }

    fn dispatch(&self, method: &u32, receiver: &Value<MockTarget>) -> Option<u32> {
        (*method == 5 && receiver.integer() == Some(1)).then_some(5)
    }
}

/// Direct, indirect and virtual bodies receive the same values before normal
/// or tail transfer; repeated semantic keys take their last supplied value.
#[test]
fn body_binding_uses_separate_keyed_inputs_for_every_call_kind_and_tail_mode() {
    for kind in 0..3 {
        for tail in [false, true] {
            for keys in [
                vec![KEY],
                vec![KEY, KEY],
                vec![KEY, CallInputKey::new(0), CallInputKey::new(u32::MAX), KEY],
            ] {
                let caller = sum_caller(kind, tail, &keys);
                let host = BodyHost {
                    body: sum_body(),
                    key: KEY,
                };
                let context = AnalysisContext::new(MACHINE);
                let facts = StaticExecution::new(&caller, &context)
                    .with_host(&host)
                    .run();
                assert!(
                    facts.is_complete(),
                    "kind={kind} tail={tail} {:?}",
                    facts.completion()
                );
                assert_eq!(
                    facts.returned().and_then(Value::constant),
                    Some(&ConstValue::I64(42)),
                    "kind={kind} tail={tail} keys={keys:?}"
                );
            }
        }
    }
}

/// Missing or differently named inputs remain unknown; key zero cannot read
/// ordinary argument zero even though its numeric representation is the same.
#[test]
fn missing_implicit_keys_do_not_fall_back_to_an_argument_or_another_key() {
    for (keys, wanted) in [
        (vec![], KEY),
        (vec![KEY], CallInputKey::new(0)),
        (vec![CallInputKey::new(0)], KEY),
    ] {
        let caller = sum_caller(0, false, &keys);
        let host = BodyHost {
            body: sum_body(),
            key: wanted,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(facts.returned().and_then(Value::constant).is_none());
    }
    let caller = sum_caller(0, false, &[CallInputKey::new(0)]);
    let host = BodyHost {
        body: sum_body(),
        key: CallInputKey::new(0),
    };
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .run();
    assert_eq!(
        facts.returned().and_then(Value::constant),
        Some(&ConstValue::I64(42))
    );
}

/// A model observes each alternative's independently retained actual inputs.
struct SumModel {
    count: usize,
    seen: RefCell<Vec<(usize, i64, i64)>>,
    control: ModelControl,
}

impl ExecutionHost<MockTarget> for SumModel {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 5).then_some(self)
    }
}

impl CallModel<MockTarget> for SumModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        self.call_alternative(0, call);
    }
    fn alternatives(&self) -> usize {
        self.count
    }
    fn call_alternative(&self, alternative: usize, call: &mut ModelCall<'_, MockTarget>) {
        assert!(
            call.argument(1).is_none(),
            "hidden values do not change language arity"
        );
        assert!(call.implicit_input(CallInputKey::new(9)).is_none());
        let ordinary = call.integer(0).unwrap_or_else(|| process::abort());
        let hidden = call
            .implicit_input(KEY)
            .and_then(Value::integer)
            .unwrap_or_else(|| process::abort());
        self.seen.borrow_mut().push((alternative, ordinary, hidden));
        call.return_value(Value::Const(ConstValue::I64(
            ordinary
                .checked_add(hidden)
                .unwrap_or_else(|| process::abort()),
        )));
        call.set_control(self.control);
    }
}

/// Single and multiple model alternatives retain keyed values and ordinary arity.
#[test]
fn models_receive_keyed_inputs_in_all_alternatives_without_argument_leakage() {
    for count in [1, 2] {
        for tail in [false, true] {
            let caller = sum_caller(0, tail, &[KEY, CallInputKey::new(0), KEY]);
            let host = SumModel {
                count,
                seen: RefCell::new(Vec::new()),
                control: ModelControl::Return,
            };
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&caller, &context)
                .with_host(&host)
                .run();
            assert!(facts.is_complete(), "{:?}", facts.completion());
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&ConstValue::I64(42))
            );
            assert_eq!(
                *host.seen.borrow(),
                (0..count).map(|index| (index, 1, 41)).collect::<Vec<_>>()
            );
        }
    }
}

/// Hidden-input models obey control outcomes rather than inventing a normal return.
#[test]
fn implicit_input_models_preserve_trap_nonreturn_and_unsupported_control() {
    for control in [
        ModelControl::Trap,
        ModelControl::NoReturn,
        ModelControl::Unsupported,
    ] {
        let caller = sum_caller(0, false, &[KEY]);
        let host = SumModel {
            count: 1,
            seen: RefCell::new(Vec::new()),
            control,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert_eq!(*host.seen.borrow(), vec![(0, 1, 41)]);
        assert_eq!(facts.is_complete(), control != ModelControl::Unsupported);
        assert!(facts.returned().is_none());
        assert_eq!(
            facts.usage().traps,
            usize::from(control == ModelControl::Trap)
        );
        if control == ModelControl::Unsupported {
            assert!(matches!(facts.completion(), Completion::Stuck { .. }));
        }
    }
}

/// The independent domain carrying a possible private-stack pointer to a call.
#[derive(Clone, Copy)]
enum StackInput {
    /// A confirmed language argument.
    Ordinary,
    /// A value under a named host contract.
    Implicit,
    /// A stack candidate that no callee contract has promoted.
    Staged,
}

/// A private byte is passed as an ordinary or hidden pointer, then reloaded.
fn stack_caller(pure: bool, through_return: bool, hidden: bool) -> Function {
    stack_caller_with(
        pure,
        through_return,
        if hidden {
            StackInput::Implicit
        } else {
            StackInput::Ordinary
        },
    )
}

/// Stages a private byte in one input domain and optionally stores through the return.
fn stack_caller_with(pure: bool, through_return: bool, domain: StackInput) -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .unwrap_or_else(|| process::abort())
        .set_role(VariableRole::EntryStackPointer);
    builder
        .in_block(0, |b| {
            let slot = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            let byte = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x96))?;
            b.store_indirect(slot, byte, MockType::U8)?;
            let returned = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| {
                let args = if matches!(domain, StackInput::Ordinary) {
                    vec![slot]
                } else {
                    vec![]
                };
                let mut call = CallData::new(Some(dest), Callee::Direct(5), args);
                match domain {
                    StackInput::Ordinary => {}
                    StackInput::Implicit => call = call.with_implicit_input(KEY, slot),
                    StackInput::Staged => call.staged.push(StagedValue {
                        offset: 8,
                        value: slot,
                    }),
                }
                if pure {
                    call = call.with_effects(CallEffects::pure());
                }
                call.into()
            })?;
            if through_return {
                let other = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x27))?;
                b.store_indirect(returned, other, MockType::U8)?;
            }
            let loaded = b.load_indirect(SsaDefSpec::tmp(MockType::U8), slot, MockType::U8)?;
            b.ret(Some(loaded))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// Clobbers according to the model interface, or preserves memory and lifetimes.
struct MemoryModel(bool);

impl ExecutionHost<MockTarget> for MemoryModel {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 5).then_some(self)
    }
}
impl CallModel<MockTarget> for MemoryModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        if self.0 {
            call.clobber();
        } else {
            LibraryModel::NoWrites.call(call);
        }
    }
}

/// Both unknown and modeled writes invalidate private storage reachable through
/// hidden inputs, while a pure call preserves the exact independent byte oracle.
#[test]
fn hidden_stack_inputs_participate_in_unknown_call_and_model_memory_effects() {
    for model in [false, true] {
        for writes in [false, true] {
            let caller = stack_caller(!writes, false, true);
            let context = AnalysisContext::new(LITTLE_MACHINE);
            let host = MemoryModel(writes);
            let executor = StaticExecution::new(&caller, &context);
            let facts = if model {
                executor.with_host(&host).run()
            } else {
                executor.run()
            };
            assert!(facts.is_complete(), "{:?}", facts.completion());
            assert_eq!(
                facts.returned().and_then(Value::constant),
                if writes {
                    None
                } else {
                    Some(&ConstValue::U8(0x96))
                },
                "model={model} writes={writes}"
            );
            assert!(
                facts
                    .calls()
                    .iter()
                    .all(|(_, arguments)| arguments.is_empty()),
                "hidden pointers are not language-argument byte observations"
            );
        }
    }
}

/// Checks one input domain against the exact private byte and an unresolved-call control.
fn unknown_return_alias_case(hidden: bool) {
    for model in [false, true] {
        let caller = stack_caller(true, true, hidden);
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let host = MemoryModel(false);
        let executor = StaticExecution::new(&caller, &context);
        let facts = if model {
            executor.with_host(&host).run()
        } else {
            executor.run()
        };
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(
            facts.returned().and_then(Value::constant).is_none(),
            "model={model} hidden={hidden}"
        );
    }
}

/// An unknown model return can alias an ordinary private-pointer argument.
#[test]
fn unknown_returns_keep_stack_aliases_from_ordinary_arguments() {
    unknown_return_alias_case(false);
}

/// An unknown model return can alias a keyed implicit private-pointer input.
#[test]
fn unknown_returns_keep_stack_aliases_from_implicit_inputs() {
    unknown_return_alias_case(true);
}

/// Returns the byte addressed by the host's hidden entry input.
fn pointer_reader() -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .in_block(0, |b| {
            let byte = b.load_indirect(SsaDefSpec::tmp(MockType::U8), input, MockType::U8)?;
            b.ret(Some(byte))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// Supplies a hidden pointer into private caller storage, with matching byte result type.
fn pointer_caller(tail: bool) -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .unwrap_or_else(|| process::abort())
        .set_role(VariableRole::EntryStackPointer);
    builder
        .in_block(0, |b| {
            let slot = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            let byte = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x96))?;
            b.store_indirect(slot, byte, MockType::U8)?;
            let result = b.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| {
                let mut call = CallData::new(Some(dest), Callee::Direct(5), vec![])
                    .with_implicit_input(KEY, slot);
                call.tail = tail;
                call.into()
            })?;
            b.ret(Some(result))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// Reads the same hidden pointer through the model interface as a body would.
struct PointerReader;
impl ExecutionHost<MockTarget> for PointerReader {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 5).then_some(self)
    }
}
impl CallModel<MockTarget> for PointerReader {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let byte = call
            .implicit_input(KEY)
            .and_then(Value::address)
            .and_then(|address| call.read(address, 1))
            .and_then(|bytes| bytes.first().copied());
        if let Some(byte) = byte {
            call.return_value(Value::Const(ConstValue::U8(byte)));
        }
    }
}

/// Capturing a hidden pointer before tail transfer preserves its identity,
/// while retiring caller storage prevents either body or model from reviving bytes.
#[test]
fn implicit_pointer_capture_preserves_tail_storage_retirement_for_bodies_and_models() {
    for model in [false, true] {
        for tail in [false, true] {
            let caller = pointer_caller(tail);
            let body = BodyHost {
                body: pointer_reader(),
                key: KEY,
            };
            let context = AnalysisContext::new(LITTLE_MACHINE);
            let executor = StaticExecution::new(&caller, &context);
            let facts = if model {
                executor.with_host(&PointerReader).run()
            } else {
                executor.with_host(&body).run()
            };
            assert!(
                facts.is_complete(),
                "model={model} tail={tail} {:?}",
                facts.completion()
            );
            assert_eq!(
                facts.returned().and_then(Value::constant),
                if tail {
                    None
                } else {
                    Some(&ConstValue::U8(0x96))
                },
                "model={model} tail={tail}"
            );
        }
    }
}

/// Explicitly leaves a model return unknown, rather than omitting a result.
struct ExplicitUnknown;
impl ExecutionHost<MockTarget> for ExplicitUnknown {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 5).then_some(self)
    }
}
impl CallModel<MockTarget> for ExplicitUnknown {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Unknown);
    }
}

/// Explicit unknown model outputs retain possible aliases from both input domains.
#[test]
fn explicit_unknown_model_returns_keep_stack_aliases() {
    for hidden in [false, true] {
        let caller = stack_caller(true, true, hidden);
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&ExplicitUnknown)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(
            facts.returned().and_then(Value::constant).is_none(),
            "hidden={hidden}"
        );
    }
}

/// A body whose pointer input is either written through or returned for a later store.
fn pointer_alias_body(hidden: bool, write: bool) -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(usize::from(!hidden), 0);
    let input = builder.function_mut().create_variable(
        if hidden {
            VariableOrigin::EntryLiveIn
        } else {
            VariableOrigin::Argument(0)
        },
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .in_block(0, |b| {
            if write {
                let byte = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x27))?;
                b.store_indirect(input, byte, MockType::U8)?;
            }
            b.ret(Some(input))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// Either justifies an actual pointer binding or deliberately lacks a mapping.
struct AliasingBodyHost {
    body: Function,
    hidden: bool,
    bind: bool,
}
impl ExecutionHost<MockTarget> for AliasingBodyHost {
    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 5).then_some(&self.body)
    }
    fn parameters(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        body: &Function,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        if !self.bind {
            return;
        }
        for variable in body.variables() {
            if variable.version() != 0
                || !matches!(
                    variable.origin(),
                    VariableOrigin::Argument(0) | VariableOrigin::EntryLiveIn
                )
            {
                continue;
            }
            if self.hidden {
                bindings.bind_implicit(variable.id(), KEY);
            } else {
                bindings.bind(variable.id(), 0);
            }
        }
    }
}

/// A fully justified mapping has an exact independent byte result. Without a
/// mapping, either action may still reach storage whose pointer the caller supplied.
fn unbound_body_alias_case(hidden: bool, write: bool) {
    for bind in [true, false] {
        let caller = stack_caller(false, !write, hidden);
        let host = AliasingBodyHost {
            body: pointer_alias_body(hidden, write),
            hidden,
            bind,
        };
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert!(
            facts.is_complete(),
            "hidden={hidden} write={write} bind={bind} {:?}",
            facts.completion()
        );
        assert_eq!(
            facts.returned().and_then(Value::constant),
            if bind {
                Some(&ConstValue::U8(0x27))
            } else {
                None
            },
            "hidden={hidden} write={write} bind={bind}"
        );
    }
}

/// Declining an ordinary pointer mapping does not prove a returned alias private.
#[test]
fn unbound_ordinary_body_returns_may_alias_caller_storage() {
    unbound_body_alias_case(false, false);
}

/// Declining a hidden pointer mapping does not prove a returned alias private.
#[test]
fn unbound_implicit_body_returns_may_alias_caller_storage() {
    unbound_body_alias_case(true, false);
}

/// An unmapped ordinary pointer may still write the caller's private byte.
#[test]
fn unbound_ordinary_body_writes_may_alias_caller_storage() {
    unbound_body_alias_case(false, true);
}

/// An unmapped hidden pointer may still write the caller's private byte.
#[test]
fn unbound_implicit_body_writes_may_alias_caller_storage() {
    unbound_body_alias_case(true, true);
}

/// A model checks confirmed input boundaries before applying its declared effects.
struct StagedContractModel {
    /// Whether the model writes through possible caller aliases.
    writes: bool,
    /// Whether it explicitly supplies an unknown result.
    explicit_return: bool,
}

impl ExecutionHost<MockTarget> for StagedContractModel {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 5).then_some(self)
    }
}

impl CallModel<MockTarget> for StagedContractModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        assert!(
            call.argument(0).is_none(),
            "a staged candidate is not an argument"
        );
        assert!(
            call.implicit_input(KEY).is_none(),
            "a staged candidate has no semantic key"
        );
        if self.writes {
            call.clobber();
        } else {
            LibraryModel::NoWrites.call(call);
        }
        if self.explicit_return {
            call.return_value(Value::Unknown);
        }
    }
}

/// Opaque and modeled calls preserve candidate aliases without promoting their inputs.
#[test]
fn staged_pointers_retain_unknown_effect_and_return_aliases() {
    for modeled in [false, true] {
        for pure in [false, true] {
            for through_return in [false, true] {
                let caller = stack_caller_with(pure, through_return, StackInput::Staged);
                let context = AnalysisContext::new(LITTLE_MACHINE);
                let host = StagedContractModel {
                    writes: !pure,
                    explicit_return: false,
                };
                let executor = StaticExecution::new(&caller, &context);
                let facts = if modeled {
                    executor.with_host(&host).run()
                } else {
                    executor.run()
                };
                assert!(facts.is_complete(), "{:?}", facts.completion());
                assert_eq!(
                    facts.returned().and_then(Value::constant),
                    if pure && !through_return {
                        Some(&ConstValue::U8(0x96))
                    } else {
                        None
                    },
                    "modeled={modeled} pure={pure} through_return={through_return}",
                );
                assert!(
                    facts
                        .calls()
                        .iter()
                        .all(|(_, arguments)| arguments.is_empty())
                );
            }
        }
    }
}

/// An explicit unknown model result can carry a staged pointer as an alias.
#[test]
fn staged_pointers_retain_explicit_unknown_model_return_aliases() {
    let caller = stack_caller_with(true, true, StackInput::Staged);
    let context = AnalysisContext::new(LITTLE_MACHINE);
    let host = StagedContractModel {
        writes: false,
        explicit_return: true,
    };
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(facts.returned().and_then(Value::constant).is_none());
}

/// Unbound body inputs may still return or write storage reachable by staged pointers.
#[test]
fn staged_pointers_retain_unbound_body_aliases() {
    for write in [false, true] {
        let caller = stack_caller_with(false, !write, StackInput::Staged);
        let host = AliasingBodyHost {
            body: pointer_alias_body(true, write),
            hidden: true,
            bind: false,
        };
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert!(
            facts.is_complete(),
            "write={write}: {:?}",
            facts.completion()
        );
        assert!(
            facts.returned().and_then(Value::constant).is_none(),
            "write={write}"
        );
        assert!(
            facts
                .calls()
                .iter()
                .all(|(_, arguments)| arguments.is_empty())
        );
    }
}

/// A physically paired register whose callee view cannot represent its stack pointer.
struct UnsupportedRegisterHost {
    body: Function,
}

impl ExecutionHost<MockTarget> for UnsupportedRegisterHost {
    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 5).then_some(&self.body)
    }

    fn parameters(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        body: &Function,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        for variable in body.variables() {
            if variable.version() == 0 && variable.origin() == VariableOrigin::Argument(0) {
                bindings.bind_register_view(
                    variable.id(),
                    0,
                    MockTarget::type_layout(&MockType::Ptr, LITTLE_MACHINE.ptr_size),
                    variable.var_type(),
                    LITTLE_MACHINE,
                );
            }
        }
    }
}

/// An unsupported register read still exposes its possible stack alias to an unknown consumer.
#[test]
fn unsupported_register_views_preserve_stack_escape() {
    for writes in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let input = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::F64,
        );
        builder
            .in_block(0, |block| {
                if writes {
                    block.call_void(Callee::Direct(6), vec![input])?;
                }
                block.ret(None)
            })
            .unwrap_or_else(|_| process::abort());
        let host = UnsupportedRegisterHost {
            body: verified(builder),
        };
        let caller = stack_caller(false, false, false);
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            if writes {
                None
            } else {
                Some(&ConstValue::U8(0x96))
            },
            "writes={writes}"
        );
    }
}
