//! Tail transfers release the caller's storage and inherit its return continuation.

use std::{cell::RefCell, process};

use analyssa::{
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        consts::Machine,
        context::AnalysisContext,
        execution::{CallModel, ExecutionHost, ModelCall, ModelControl, StaticExecution, Value},
    },
    ir::{
        CallData, Callee, ConstValue, SsaOp, SsaVarId,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::CmpKind,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// A declared little-endian machine for exact multi-byte stack contents.
const LITTLE_MACHINE: Machine = Machine {
    byte_order: Some(Endianness::Little),
    ..MACHINE
};

/// One hermetic execution body.
type Function = SsaFunction<MockTarget>;

/// Finishes a fixture and checks that its SSA is structurally valid.
fn verified(builder: SsaFunctionBuilder<MockTarget>) -> Function {
    let function = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&function)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    function
}

/// Declares the input stack pointer that execution initializes for this activation.
fn stack(builder: &mut SsaFunctionBuilder<MockTarget>) -> SsaVarId {
    let value = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(value)
        .unwrap_or_else(|| process::abort())
        .set_role(VariableRole::EntryStackPointer);
    value
}

/// Calls method one, optionally passing private storage or executing a later continuation.
fn caller(tail: bool, pointer: bool, continuation: bool) -> (Function, SsaVarId) {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = pointer.then(|| stack(&mut builder));
    let marker = builder
        .in_block(0, |block| {
            let marker = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(17))?;
            let args = if let Some(stack) = stack {
                let slot =
                    block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: stack,
                        index: None,
                        stride: 1,
                        offset: -8,
                        result_type: MockType::Ptr,
                    })?;
                block.store_indirect(slot, marker, MockType::I32)?;
                vec![slot]
            } else {
                vec![]
            };
            let result = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                SsaOp::Call(Box::new(CallData {
                    dest: Some(dest),
                    callee: Callee::Direct(1),
                    args,
                    effects: None,
                    tail,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                }))
            })?;
            let returned = if continuation {
                block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(99))?
            } else {
                result
            };
            block.ret(Some(returned))?;
            Ok(marker)
        })
        .unwrap_or_else(|_| process::abort());
    (verified(builder), marker)
}

/// Returns a constant, or tail-transfers to the next method.
fn chain_body(next: Option<u32>) -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |block| {
            let marker = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(42))?;
            let result = if let Some(method) = next {
                block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                    SsaOp::Call(Box::new(
                        CallData::new(Some(dest), Callee::Direct(method), vec![]).made_tail(),
                    ))
                })?
            } else {
                marker
            };
            block.ret(Some(result))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// Reads the storage the caller passed as argument zero.
fn reader() -> Function {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let argument = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .in_block(0, |block| {
            let value =
                block.load_indirect(SsaDefSpec::tmp(MockType::I32), argument, MockType::I32)?;
            block.ret(Some(value))
        })
        .unwrap_or_else(|_| process::abort());
    verified(builder)
}

/// A model that records any known argument bytes and chooses an explicit control outcome.
struct Model {
    /// Requested control outcome.
    control: ModelControl,
    /// What each actual invocation could read, without inventing an extent.
    reads: RefCell<Vec<Option<Vec<u8>>>>,
}

impl CallModel<MockTarget> for Model {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let bytes = call.address(0).and_then(|address| call.read(address, 4));
        let returned = if call.argument(0).is_some() {
            bytes
                .as_ref()
                .and_then(|bytes| bytes.first())
                .map_or(Value::Unknown, |byte| {
                    Value::Const(ConstValue::I32(i32::from(*byte)))
                })
        } else {
            Value::Const(ConstValue::I32(42))
        };
        self.reads.borrow_mut().push(bytes);
        call.return_value(returned);
        call.set_control(self.control);
    }

    fn only_memory(&self) -> bool {
        true
    }
}

/// Resolves numbered bodies or supplies one explicit model, with no stack-layout assumptions.
struct Host {
    /// Bodies numbered from method one.
    bodies: Vec<Function>,
    /// An optional model for method one.
    model: Option<Model>,
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&Function> {
        self.bodies
            .get(usize::try_from(method.checked_sub(1)?).ok()?)
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 1)
            .then_some(self.model.as_ref()?)
            .map(|model| model as &dyn CallModel<MockTarget>)
    }
}

/// Tail returns bypass the caller's later instructions, while ordinary calls resume them.
#[test]
fn tail_body_inherits_the_callers_return_continuation() {
    let host = Host {
        bodies: vec![chain_body(None)],
        model: None,
    };
    for tail in [false, true] {
        let (root, marker) = caller(tail, false, true);
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&root, &context).with_host(&host).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(if tail { 42 } else { 99 }))
        );
        assert_eq!(facts.constant(marker), Some(&ConstValue::I32(17)));
    }
}

/// A tail callee cannot read the retired caller's private stack contents.
#[test]
fn tail_body_releases_caller_storage_before_entry() {
    let host = Host {
        bodies: vec![reader()],
        model: None,
    };
    for tail in [false, true] {
        let (root, _) = caller(tail, true, false);
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&root, &context).with_host(&host).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let constant = facts.returned().and_then(Value::constant);
        if tail {
            assert_eq!(constant, None);
        } else {
            assert_eq!(
                constant,
                Some(&ConstValue::I32(17)),
                "root={root}\nfacts={facts:?}"
            );
        }
    }
}

/// Model execution sees the same released storage as a body entered by a tail transfer.
#[test]
fn tail_model_releases_caller_storage_before_execution() {
    for tail in [false, true] {
        let host = Host {
            bodies: vec![],
            model: Some(Model {
                control: ModelControl::Return,
                reads: RefCell::new(vec![]),
            }),
        };
        let (root, _) = caller(tail, true, false);
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&root, &context).with_host(&host).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let model = host.model.as_ref().unwrap_or_else(|| process::abort());
        assert_eq!(
            *model.reads.borrow(),
            vec![if tail { None } else { Some(vec![17, 0, 0, 0]) }]
        );
        assert_eq!(
            facts.returned().and_then(Value::constant),
            if tail {
                None
            } else {
                Some(&ConstValue::I32(17))
            }
        );
    }
}

/// Tail chains reuse one depth slot and never publish callee definitions as root SSA facts.
#[test]
fn tail_chain_reuses_depth_without_rebinding_root_facts() {
    let host = Host {
        bodies: (1..=32)
            .map(|method| chain_body((method < 32).then_some(method + 1)))
            .collect(),
        model: None,
    };
    let (root, marker) = caller(true, false, false);
    let context = AnalysisContext::new(LITTLE_MACHINE)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionDepth, 1));
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts.returned().and_then(Value::constant),
        Some(&ConstValue::I32(42))
    );
    assert_eq!(facts.constant(marker), Some(&ConstValue::I32(17)));
    assert_eq!(facts.calls().len(), 1);
}

/// An unanswered tail call returns an unknown value rather than executing a later constant.
#[test]
fn tail_unknown_call_does_not_resume_the_caller() {
    let host = Host {
        bodies: vec![],
        model: None,
    };
    let (root, marker) = caller(true, false, true);
    let context = AnalysisContext::new(LITTLE_MACHINE);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.returned().and_then(Value::constant), None);
    assert_eq!(facts.constant(marker), Some(&ConstValue::I32(17)));
}

/// Model return, trap, nonreturn and refusal remain distinct across a tail transfer.
#[test]
fn tail_models_preserve_control_outcomes() {
    for control in [
        ModelControl::Return,
        ModelControl::Trap,
        ModelControl::NoReturn,
        ModelControl::Unsupported,
    ] {
        let host = Host {
            bodies: vec![],
            model: Some(Model {
                control,
                reads: RefCell::new(vec![]),
            }),
        };
        let (root, _) = caller(true, false, true);
        let context = AnalysisContext::new(LITTLE_MACHINE);
        let facts = StaticExecution::new(&root, &context).with_host(&host).run();
        assert_eq!(facts.is_complete(), control != ModelControl::Unsupported);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            if control == ModelControl::Return {
                Some(&ConstValue::I32(42))
            } else {
                None
            }
        );
        assert_eq!(
            facts.usage().traps,
            usize::from(control == ModelControl::Trap)
        );
        if control == ModelControl::Unsupported {
            assert!(facts.calls().is_empty());
        }
    }
}

/// A nested tail transfer retires the middle frame while preserving grandcaller storage.
#[test]
fn tail_nested_transfer_preserves_enclosing_storage() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let argument = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    let entry = stack(&mut builder);
    builder
        .in_block(0, |block| {
            let own = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: entry,
                index: None,
                stride: 1,
                offset: -8,
                result_type: MockType::Ptr,
            })?;
            let value = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(42))?;
            block.store_indirect(own, value, MockType::I32)?;
            let returned = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                SsaOp::Call(Box::new(
                    CallData::new(Some(dest), Callee::Direct(2), vec![argument]).made_tail(),
                ))
            })?;
            block.ret(Some(returned))
        })
        .unwrap_or_else(|_| process::abort());
    let host = Host {
        bodies: vec![verified(builder), reader()],
        model: None,
    };
    let (root, marker) = caller(false, true, false);
    let context = AnalysisContext::new(LITTLE_MACHINE);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts.returned().and_then(Value::constant),
        Some(&ConstValue::I32(17))
    );
    assert_eq!(facts.constant(marker), Some(&ConstValue::I32(17)));
}

/// Recursive tail calls reuse one activation slot while preserving the original SSA facts.
#[test]
fn tail_recursion_reuses_depth_and_keeps_entry_fact_identity() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let count = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    builder
        .in_block(0, |block| {
            let zero = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
            block.branch_cmp(count, zero, CmpKind::Eq, false, 2, 1)
        })
        .unwrap_or_else(|_| process::abort());
    builder
        .in_block(1, |block| {
            let one = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(1))?;
            let next = block.sub(SsaDefSpec::tmp(MockType::I32), count, one)?;
            let returned = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                SsaOp::Call(Box::new(
                    CallData::new(Some(dest), Callee::Direct(1), vec![next]).made_tail(),
                ))
            })?;
            block.ret(Some(returned))
        })
        .unwrap_or_else(|_| process::abort());
    builder
        .in_block(2, |block| {
            let answer = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(42))?;
            block.ret(Some(answer))
        })
        .unwrap_or_else(|_| process::abort());
    let host = Host {
        bodies: vec![verified(builder)],
        model: None,
    };
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let marker = builder
        .in_block(0, |block| {
            let marker = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(17))?;
            let count = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(32))?;
            let returned = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                SsaOp::Call(Box::new(
                    CallData::new(Some(dest), Callee::Direct(1), vec![count]).made_tail(),
                ))
            })?;
            block.ret(Some(returned))?;
            Ok(marker)
        })
        .unwrap_or_else(|_| process::abort());
    let root = verified(builder);
    let context = AnalysisContext::new(LITTLE_MACHINE)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionDepth, 1));
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts.returned().and_then(Value::constant),
        Some(&ConstValue::I32(42))
    );
    assert_eq!(facts.constant(marker), Some(&ConstValue::I32(17)));
    assert_eq!(facts.calls().len(), 1);
    assert_eq!(facts.body_methods().len(), 2);
}

/// Supplies two model alternatives with distinct normal return values.
struct Alternatives;

impl CallModel<MockTarget> for Alternatives {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Const(ConstValue::I32(42)));
    }

    fn alternatives(&self) -> usize {
        2
    }

    fn call_alternative(&self, alternative: usize, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Const(ConstValue::I32(if alternative == 0 {
            42
        } else {
            43
        })));
    }

    fn only_memory(&self) -> bool {
        true
    }
}

impl ExecutionHost<MockTarget> for Alternatives {
    fn model(&self, _: &u32) -> Option<&dyn CallModel<MockTarget>> {
        Some(self)
    }
}

/// Every tail-model alternative reaches the inherited continuation without resuming its caller.
#[test]
fn tail_model_alternatives_meet_at_the_inherited_return() {
    let (root, marker) = caller(true, false, true);
    let context = AnalysisContext::new(LITTLE_MACHINE);
    let facts = StaticExecution::new(&root, &context)
        .with_host(&Alternatives)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.returned().and_then(Value::constant), None);
    assert_eq!(facts.constant(marker), Some(&ConstValue::I32(17)));
    assert_eq!(facts.calls().len(), 1);
}
