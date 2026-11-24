//! Constrained dispatch retains type selection, receiver shape and semantic inputs.

use std::{
    cell::{Cell, RefCell},
    process,
};

use analyssa::{
    analysis::{
        SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{CallModel, ExecutionHost, ModelCall, StaticExecution, Value},
    },
    ir::{CallData, CallInputKey, Callee, ConstValue, SsaDefSpec, SsaFunction, SsaFunctionBuilder},
    testing::{MACHINE, MockTarget, MockType},
};

/// The independent input domain used by the selected implementation.
const INPUT: CallInputKey = CallInputKey::new(17);

/// A typed call with one language argument and one semantic input.
fn caller(
    constrained: bool,
    instance: bool,
    tail: bool,
    constraint_type: u32,
) -> SsaFunction<MockTarget> {
    caller_with_args(constrained, instance, tail, constraint_type, true)
}

/// Builds a call while independently controlling whether a language argument exists.
fn caller_with_args(
    constrained: bool,
    instance: bool,
    tail: bool,
    constraint_type: u32,
    ordinary_present: bool,
) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |block| {
            let ordinary = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
            let hidden = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(41))?;
            let result = block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| {
                let callee = if constrained {
                    Callee::Constrained {
                        method: 5,
                        constraint_type,
                        instance,
                    }
                } else {
                    Callee::Direct(6)
                };
                let mut call = CallData::new(
                    Some(dest),
                    callee,
                    if ordinary_present {
                        vec![ordinary]
                    } else {
                        Vec::new()
                    },
                )
                .with_implicit_input(INPUT, hidden);
                call.tail = tail;
                call.into()
            })?;
            block.ret(Some(result))
        })
        .unwrap_or_else(|_| process::abort());
    let body = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&body)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    body
}

/// A host with an explicit implementation table, independent of the method declaration.
struct Host {
    /// Whether the host can prove the implementation selection.
    allowed: bool,
    /// Type and receiver observations at constrained dispatch.
    dispatches: RefCell<Vec<(u32, Option<i64>)>>,
    /// Number of actual implementation model invocations.
    calls: Cell<usize>,
    /// Ordinary virtual dispatch must not stand in for a type constraint.
    virtual_calls: Cell<usize>,
}

impl Host {
    /// Returns a fresh table allowing or declining the exact type selection.
    fn new(allowed: bool) -> Self {
        Self {
            allowed,
            dispatches: RefCell::new(Vec::new()),
            calls: Cell::new(0),
            virtual_calls: Cell::new(0),
        }
    }

    /// Resolves only method five under constraint three to implementation six.
    fn implementation(
        &self,
        method: &u32,
        constraint_type: &u32,
        receiver: Option<&Value<MockTarget>>,
    ) -> Option<u32> {
        self.dispatches
            .borrow_mut()
            .push((*constraint_type, receiver.and_then(Value::integer)));
        (self.allowed && *method == 5 && *constraint_type == 3).then_some(6)
    }
}

impl ExecutionHost<MockTarget> for Host {
    fn dispatch_constrained(
        &self,
        method: &u32,
        constraint_type: &u32,
        receiver: Option<&Value<MockTarget>>,
    ) -> Option<u32> {
        self.implementation(method, constraint_type, receiver)
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 6).then_some(self)
    }

    fn dispatch(&self, _: &u32, _: &Value<MockTarget>) -> Option<u32> {
        self.virtual_calls.set(
            self.virtual_calls
                .get()
                .checked_add(1)
                .unwrap_or_else(|| process::abort()),
        );
        None
    }
}

impl CallModel<MockTarget> for Host {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        self.calls.set(
            self.calls
                .get()
                .checked_add(1)
                .unwrap_or_else(|| process::abort()),
        );
        assert!(
            call.argument(1).is_none(),
            "the hidden value is not a language argument"
        );
        let ordinary = call.integer(0).unwrap_or(0);
        let hidden = call
            .implicit_input(INPUT)
            .and_then(Value::integer)
            .unwrap_or_else(|| process::abort());
        call.return_value(Value::Const(ConstValue::I64(
            ordinary
                .checked_add(hidden)
                .unwrap_or_else(|| process::abort()),
        )));
    }
}

/// Type selection reaches the exact model for static/instance and normal/tail calls.
#[test]
fn constrained_execution_uses_type_and_receiver_without_losing_semantic_inputs() {
    for instance in [false, true] {
        for tail in [false, true] {
            let host = Host::new(true);
            let receiver = Value::Const(ConstValue::I64(1));
            assert_eq!(
                host.implementation(&5, &3, instance.then_some(&receiver)),
                Some(6)
            );
            host.dispatches.borrow_mut().clear();
            let context = AnalysisContext::new(MACHINE);
            // Direct invocation is the independent exact implementation oracle.
            let direct = caller(false, instance, tail, 3);
            let control = StaticExecution::new(&direct, &context)
                .with_host(&host)
                .run();
            assert!(control.is_complete());
            assert_eq!(control.returned().and_then(Value::integer), Some(42));
            host.calls.set(0);
            let body = caller(true, instance, tail, 3);
            let facts = StaticExecution::new(&body, &context).with_host(&host).run();
            assert!(facts.is_complete(), "{:?}", facts.completion());
            assert_eq!(
                facts.returned().and_then(Value::integer),
                Some(42),
                "instance={instance} tail={tail}"
            );
            assert_eq!(host.calls.get(), 1);
            assert_eq!(*host.dispatches.borrow(), vec![(3, instance.then_some(1))]);
            assert_eq!(host.virtual_calls.get(), 0);
        }
    }
}

/// An unproven type selection cannot run the named declaration or another type's model.
#[test]
fn constrained_execution_declined_or_wrong_type_stays_unknown() {
    for instance in [false, true] {
        for (allowed, constraint_type) in [(false, 3), (true, 4)] {
            let host = Host::new(allowed);
            let context = AnalysisContext::new(MACHINE);
            let body = caller(true, instance, false, constraint_type);
            let facts = StaticExecution::new(&body, &context).with_host(&host).run();
            assert!(facts.is_complete());
            assert!(facts.returned().and_then(Value::constant).is_none());
            assert_eq!(host.calls.get(), 0);
            assert_eq!(
                *host.dispatches.borrow(),
                vec![(constraint_type, instance.then_some(1))]
            );
            assert_eq!(host.virtual_calls.get(), 0);
        }
    }
}

/// A static constrained call has no receiver even when its first captured value is hidden.
#[test]
fn constrained_execution_static_without_arguments_does_not_take_a_hidden_receiver() {
    let host = Host::new(true);
    let context = AnalysisContext::new(MACHINE);
    let body = caller_with_args(true, false, false, 3, false);
    let facts = StaticExecution::new(&body, &context).with_host(&host).run();
    assert!(facts.is_complete());
    assert_eq!(facts.returned().and_then(Value::integer), Some(41));
    assert_eq!(*host.dispatches.borrow(), vec![(3, None)]);
    assert_eq!(host.calls.get(), 1);
    assert_eq!(host.virtual_calls.get(), 0);
}
