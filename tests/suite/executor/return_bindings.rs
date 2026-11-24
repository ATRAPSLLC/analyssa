//! Result contracts apply before execution resumes and independently of observation caps.

use std::{cell::Cell, collections::BTreeMap, error::Error};

use analyssa::{
    analysis::{
        SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{
            CallModel, CallSite, ExecutionHost, ModelCall, ReturnBinding, StaticExecution, Value,
        },
    },
    ir::{
        CallData, Callee, ConstValue, FloatFormat, SsaOp,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::CmpKind,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    testing::{MACHINE, MockTarget, MockType},
};

/// The exact semantic floating result supplied by the final implementation.
fn answer() -> ConstValue<MockTarget> {
    ConstValue::Float(2.0f64.into())
}

/// Checks structural validity before every execution.
fn verified(
    builder: SsaFunctionBuilder<MockTarget>,
) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let body = builder.finish()?;
    assert!(SsaVerifier::new(&body).verify(VerifyLevel::Full).is_empty());
    Ok(body)
}

/// Returns a constant or forwards one actual call, with no observation-dependent behavior.
fn body(next: Option<u32>, tail: bool) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let value = if let Some(method) = next {
            block.emit_def(SsaDefSpec::tmp(MockType::F64), |dest| {
                let call = CallData::new(Some(dest), Callee::Direct(method), vec![]);
                SsaOp::Call(Box::new(if tail { call.made_tail() } else { call }))
            })?
        } else {
            block.const_value(SsaDefSpec::tmp(MockType::F64), answer())?
        };
        block.ret(Some(value))
    })?;
    verified(builder)
}

/// A selected model with an exact semantic result and a separately checked caller mapping.
struct Model;
impl CallModel<MockTarget> for Model {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Const(answer()));
    }
}

/// Immutable bodies and policies, plus lookup counters that never change answers.
struct Host {
    bodies: BTreeMap<u32, SsaFunction<MockTarget>>,
    reject: Option<u32>,
    model_reject: bool,
    model_lookups: Cell<usize>,
    model_bindings: Cell<usize>,
}
impl Host {
    fn new(bodies: BTreeMap<u32, SsaFunction<MockTarget>>, reject: Option<u32>) -> Self {
        Self {
            bodies,
            reject,
            model_reject: false,
            model_lookups: Cell::new(0),
            model_bindings: Cell::new(0),
        }
    }
}
impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        self.bodies.get(method)
    }
    fn body_return_binding(
        &self,
        site: CallSite<'_, MockTarget>,
        _: &u32,
        _: &SsaFunction<MockTarget>,
    ) -> ReturnBinding {
        if self.reject.as_ref() == site.caller_method {
            ReturnBinding::Word(64)
        } else {
            ReturnBinding::Float(FloatFormat::Double)
        }
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        self.model_lookups
            .set(self.model_lookups.get().saturating_add(1));
        (*method == 99).then_some(&Model)
    }
    fn model_return_binding(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        _: &dyn CallModel<MockTarget>,
    ) -> ReturnBinding {
        self.model_bindings
            .set(self.model_bindings.get().saturating_add(1));
        if self.model_reject {
            ReturnBinding::Word(64)
        } else {
            ReturnBinding::Float(FloatFormat::Double)
        }
    }
}

/// Retired observations receive segment results while the live caller receives the full composition.
#[test]
fn mixed_tail_contracts_preserve_precise_children() -> Result<(), Box<dyn Error>> {
    let expected_result = answer();
    let host = Host::new(
        BTreeMap::from([
            (1, body(Some(2), true)?),
            (2, body(Some(3), true)?),
            (3, body(None, false)?),
        ]),
        Some(1),
    );
    let root = body(Some(1), false)?;
    let context = AnalysisContext::new(MACHINE);
    for cap in [0, 1, 2, 3, usize::MAX] {
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_method(0)
            .with_transitive_call_observations(cap)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(facts.returned().and_then(Value::constant).is_none());
        let expected_count = cap.min(3);
        assert_eq!(facts.invocations().len(), expected_count, "cap={cap}");
        let methods: Vec<_> = facts
            .invocations()
            .iter()
            .map(|record| {
                facts
                    .body_methods()
                    .get(record.body)
                    .and_then(Option::as_ref)
                    .copied()
                    .ok_or("method")
            })
            .collect::<Result<_, _>>()?;
        assert_eq!(methods, (1u32..=3).take(expected_count).collect::<Vec<_>>());
        for (index, record) in facts.invocations().iter().enumerate() {
            assert_eq!(record.parent, index.checked_sub(1));
            let caller = index
                .checked_sub(1)
                .and_then(|parent| facts.invocations().get(parent))
                .map_or(0, |parent| parent.body);
            assert_eq!(record.caller_body, caller);
            assert_eq!((record.block, record.instruction), (0, 0));
        }
        for invocation in facts.invocations() {
            let method = facts
                .body_methods()
                .get(invocation.body)
                .and_then(Option::as_ref)
                .ok_or("method")?;
            assert_eq!(
                invocation.facts.returned().and_then(Value::constant),
                if *method == 1 {
                    None
                } else {
                    Some(&expected_result)
                },
                "method={method} cap={cap}"
            );
        }
    }
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_method(0)
        .with_root_call_observations(4)
        .run();
    assert!(facts.is_complete());
    assert!(facts.returned().and_then(Value::constant).is_none());
    assert_eq!(facts.invocations().len(), 1);
    assert!(
        facts
            .invocations()
            .first()
            .ok_or("root call")?
            .facts
            .returned()
            .and_then(Value::constant)
            .is_none()
    );
    Ok(())
}

/// Unobserved long tail spines retain rejection without consuming normal call depth.
#[test]
fn unobserved_tail_spines_and_root_tails_preserve_contracts() -> Result<(), Box<dyn Error>> {
    let expected_result = answer();
    let mut bodies = BTreeMap::new();
    for method in 1u32..=65 {
        bodies.insert(
            method,
            body((method < 65).then(|| method.saturating_add(1)), true)?,
        );
    }
    let context = AnalysisContext::new(MACHINE);
    for reject in [None, Some(32)] {
        let host = Host::new(bodies.clone(), reject);
        for tail in [false, true] {
            let root = body(Some(1), tail)?;
            for cap in [0, 1, 3, usize::MAX] {
                let facts = StaticExecution::new(&root, &context)
                    .with_host(&host)
                    .with_method(0)
                    .with_transitive_call_observations(cap)
                    .run();
                assert!(
                    facts.is_complete(),
                    "tail={tail} cap={cap}: {:?}",
                    facts.completion()
                );
                assert_eq!(
                    facts.returned().and_then(Value::constant),
                    if reject.is_some() {
                        None
                    } else {
                        Some(&expected_result)
                    }
                );
            }
        }
    }
    Ok(())
}

/// A rejected result reaches control flow as unknown rather than being filtered at publication.
#[test]
fn return_contracts_precede_caller_branches() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let result = block.call(SsaDefSpec::tmp(MockType::F64), Callee::Direct(1), vec![])?;
        let expected = block.const_value(SsaDefSpec::tmp(MockType::F64), answer())?;
        block.branch_cmp(result, expected, CmpKind::Eq, false, 1, 2)
    })?;
    for (block, value) in [(1, 7), (2, 9)] {
        builder.in_block(block, |block| {
            let value =
                block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(value))?;
            block.ret(Some(value))
        })?;
    }
    let root = verified(builder)?;
    let context = AnalysisContext::new(MACHINE);
    for reject in [None, Some(0)] {
        let host = Host::new(BTreeMap::from([(1, body(None, false)?)]), reject);
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_method(0)
            .run();
        assert!(facts.is_complete());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            if reject.is_some() {
                None
            } else {
                Some(&ConstValue::I32(7))
            }
        );
    }
    Ok(())
}

/// Selected model contracts apply to ordinary and tail calls, with body precedence and one lookup.
#[test]
fn model_results_and_body_precedence_use_selected_contract() -> Result<(), Box<dyn Error>> {
    let expected_result = answer();
    let context = AnalysisContext::new(MACHINE);
    for body_wins in [false, true] {
        for tail in [false, true] {
            let mut host = Host::new(
                if body_wins {
                    BTreeMap::from([(99, body(None, false)?)])
                } else {
                    BTreeMap::new()
                },
                None,
            );
            host.model_reject = true;
            let root = body(Some(99), tail)?;
            let facts = StaticExecution::new(&root, &context)
                .with_host(&host)
                .with_method(0)
                .run();
            assert!(facts.is_complete());
            assert_eq!(
                facts.returned().and_then(Value::constant),
                if body_wins {
                    Some(&expected_result)
                } else {
                    None
                }
            );
            assert_eq!(host.model_lookups.get(), usize::from(!body_wins));
            assert_eq!(host.model_bindings.get(), usize::from(!body_wins));
        }
    }
    let mut host = Host::new(BTreeMap::from([(1, body(Some(99), true)?)]), None);
    host.model_reject = true;
    let root = body(Some(1), false)?;
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_method(0)
        .with_transitive_call_observations(8)
        .run();
    assert!(facts.is_complete());
    assert!(facts.returned().and_then(Value::constant).is_none());
    assert!(
        facts
            .invocations()
            .first()
            .ok_or("retired model caller")?
            .facts
            .returned()
            .and_then(Value::constant)
            .is_none()
    );
    Ok(())
}

/// Rejected addresses preserve possible stack escape through a subsequent unknown call.
#[test]
fn rejected_results_keep_stack_provenance_without_inventing_effects() -> Result<(), Box<dyn Error>>
{
    let mut callee = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let input = callee.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    callee.in_block(0, |block| block.ret(Some(input)))?;
    let host = Host::new(BTreeMap::from([(1, verified(callee)?)]), None);
    let context = AnalysisContext::new(MACHINE);
    for unknown_consumer in [false, true] {
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
            .ok_or("SP")?
            .set_role(VariableRole::EntryStackPointer);
        builder.in_block(0, |block| {
            let slot = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x96))?;
            block.store_indirect(slot, byte, MockType::U8)?;
            let returned = block.call(
                SsaDefSpec::tmp(MockType::Ptr),
                Callee::Direct(1),
                vec![slot],
            )?;
            if unknown_consumer {
                block.call_void(Callee::Direct(66), vec![returned])?;
            }
            let loaded = block.load_indirect(SsaDefSpec::tmp(MockType::U8), slot, MockType::U8)?;
            block.ret(Some(loaded))
        })?;
        let root = verified(builder)?;
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_method(0)
            .run();
        assert!(facts.is_complete());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            if unknown_consumer {
                None
            } else {
                Some(&ConstValue::U8(0x96))
            }
        );
    }
    Ok(())
}
