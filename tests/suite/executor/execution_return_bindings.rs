//! Return contracts compose through execution independently of retained call observations.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    ptr,
};

use analyssa::{
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            Address, CallModel, CallSite, ExecutionHost, ModelCall, ReturnBinding, Root,
            StaticExecution, StaticFacts, Value,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        CallData, Callee, ConstValue, SsaOp, SsaVarId,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

const WRITTEN_AT: u64 = 0x2000;
const WRITTEN_BYTE: u8 = 0x5a;
const MODEL: u32 = 99;

type Function = SsaFunction<MockTarget>;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Supplies one writable byte, with no assumed initial contents.
struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        (address == WRITTEN_AT).then_some(ImageRegion {
            start: WRITTEN_AT,
            end: WRITTEN_AT.saturating_add(1),
            mutability: Mutability::Writable,
        })
    }

    fn bytes(&self, _address: u64, _len: usize) -> Option<&[u8]> {
        None
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}

/// Uses the same builder and verifier boundary as the existing execution-call fixtures.
fn verified(builder: SsaFunctionBuilder<MockTarget>) -> TestResult<Function> {
    let body = builder.finish()?;
    assert!(
        SsaVerifier::new(&body)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    Ok(body)
}

fn tmp(ty: MockType) -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(ty)
}

/// Writes a byte and returns 42; the effect must survive rejected result knowledge.
fn leaf() -> TestResult<(Function, SsaVarId)> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let answer = builder.in_block(0, |block| {
        let pointer = block.const_value(tmp(MockType::U64), ConstValue::U64(WRITTEN_AT))?;
        let byte = block.const_value(tmp(MockType::U8), ConstValue::U8(WRITTEN_BYTE))?;
        block.store_indirect(pointer, byte, MockType::U8)?;
        let answer = block.const_value(tmp(MockType::I32), ConstValue::I32(42))?;
        block.ret(Some(answer))?;
        Ok(answer)
    })?;
    Ok((verified(builder)?, answer))
}

/// Returns the call result; a tail transfer must skip the later marker definition.
fn relay(next: u32, tail: bool) -> TestResult<(Function, SsaVarId, SsaVarId)> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let (result, marker) = builder.in_block(0, |block| {
        let result = block.emit_def(tmp(MockType::I32), |dest| {
            let mut call = CallData::new(Some(dest), Callee::Direct(next), vec![]);
            call.tail = tail;
            SsaOp::Call(Box::new(call))
        })?;
        let marker = block.const_value(tmp(MockType::I32), ConstValue::I32(99))?;
        block.ret(Some(result))?;
        Ok((result, marker))
    })?;
    Ok((verified(builder)?, result, marker))
}

/// Observation admission must affect retention alone, never result contracts or effects.
#[derive(Clone, Copy, Debug)]
enum Selection {
    None,
    Site,
    Roots,
    Descendants(usize),
}

fn execute(
    root: &Function,
    host: &dyn ExecutionHost<MockTarget>,
    selection: Selection,
    depth: usize,
) -> StaticFacts<MockTarget> {
    let image = Image;
    let context = AnalysisContext::new(MACHINE)
        .with_image(&image)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionDepth, depth));
    let execution = StaticExecution::new(root, &context)
        .with_host(host)
        .with_method(0);
    match selection {
        Selection::None => execution.run(),
        Selection::Site => execution.with_call_observations(&[(0, 0)]).run(),
        Selection::Roots => execution.with_root_call_observations(8).run(),
        Selection::Descendants(limit) => execution.with_transitive_call_observations(limit).run(),
    }
}

fn assert_write(facts: &StaticFacts<MockTarget>) {
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.image_writes(), vec![(WRITTEN_AT, vec![WRITTEN_BYTE])]);
}

/// The model's return is independent of its concrete memory effect.
struct AnswerModel {
    answer: i32,
}

impl CallModel<MockTarget> for AnswerModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let pointer = Address::new(Root::Symbol, WRITTEN_AT.cast_signed(), call.ptr_size());
        call.write(pointer, &[WRITTEN_BYTE]);
        call.return_value(Value::Const(ConstValue::I32(self.answer)));
    }

    fn only_memory(&self) -> bool {
        true
    }
}

/// Records actual binding requests while returning immutable per-boundary contracts.
struct Host {
    bodies: BTreeMap<u32, Function>,
    rejected_caller: Option<u32>,
    rejected_site: Option<(usize, usize)>,
    discard_caller: Option<u32>,
    reject_model: bool,
    body_bindings: RefCell<Vec<(Option<u32>, u32)>>,
    model_bindings: RefCell<Vec<Option<u32>>>,
    model: AnswerModel,
}

impl Host {
    fn new(bodies: BTreeMap<u32, Function>, rejected_caller: Option<u32>) -> Self {
        Self {
            bodies,
            rejected_caller,
            rejected_site: None,
            discard_caller: None,
            reject_model: false,
            body_bindings: RefCell::new(Vec::new()),
            model_bindings: RefCell::new(Vec::new()),
            model: AnswerModel { answer: 42 },
        }
    }
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&Function> {
        self.bodies.get(method)
    }

    fn body_return_binding(
        &self,
        site: CallSite<'_, MockTarget>,
        method: &u32,
        _body: &Function,
    ) -> ReturnBinding {
        self.body_bindings
            .borrow_mut()
            .push((site.caller_method.copied(), *method));
        if self
            .discard_caller
            .as_ref()
            .is_some_and(|caller| Some(caller) == site.caller_method)
        {
            ReturnBinding::Discard
        } else if self
            .rejected_caller
            .as_ref()
            .is_some_and(|caller| Some(caller) == site.caller_method)
            || (site.caller_method == Some(&0)
                && self.rejected_site == Some((site.block, site.instruction)))
        {
            ReturnBinding::Word(64)
        } else {
            ReturnBinding::Word(32)
        }
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == MODEL).then_some(&self.model)
    }

    fn model_return_binding(
        &self,
        site: CallSite<'_, MockTarget>,
        _method: &u32,
        model: &dyn CallModel<MockTarget>,
    ) -> ReturnBinding {
        assert!(ptr::addr_eq(
            model as *const dyn CallModel<MockTarget>,
            &self.model as &dyn CallModel<MockTarget> as *const dyn CallModel<MockTarget>
        ));
        self.model_bindings
            .borrow_mut()
            .push(site.caller_method.copied());
        if self.reject_model {
            ReturnBinding::Unknown
        } else {
            ReturnBinding::Word(32)
        }
    }
}

/// Uses the trait's actual default Identity contract, rather than an override returning Identity.
struct DefaultHost {
    body: Function,
}

impl ExecutionHost<MockTarget> for DefaultHost {
    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 1).then_some(&self.body)
    }
}

#[test]
fn direct_root_and_default_identity_calls_preserve42_and_memory_effects() -> TestResult {
    let (body, answer) = leaf()?;
    let facts = execute(&body, &(), Selection::None, 1);
    assert_write(&facts);
    assert_eq!(facts.constant(answer), Some(&ConstValue::I32(42)));
    assert_eq!(
        facts.returned().and_then(Value::constant),
        Some(&ConstValue::I32(42))
    );
    let host = DefaultHost { body };
    for tail in [false, true] {
        let (root, result, marker) = relay(1, tail)?;
        let facts = execute(
            &root,
            &host,
            Selection::Descendants(8),
            if tail { 1 } else { 2 },
        );
        assert_write(&facts);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(42))
        );
        assert_eq!(
            facts.constant(marker),
            if tail {
                None
            } else {
                Some(&ConstValue::I32(99))
            }
        );
        assert_eq!(
            facts.constant(result),
            if tail {
                None
            } else {
                Some(&ConstValue::I32(42))
            }
        );
        assert_eq!(facts.invocations().len(), 1);
        assert_eq!(
            facts
                .invocations()
                .first()
                .ok_or("missing observed call")?
                .facts
                .returned()
                .and_then(Value::constant),
            Some(&ConstValue::I32(42))
        );
    }
    Ok(())
}

#[test]
fn rejected_middle_tail_boundary_withholds_ancestors_but_preserves_child42_and_effects()
-> TestResult {
    for tail in [false, true] {
        for rejected_caller in [None, Some(1)] {
            let mut expected_usage = None;
            for selection in [
                Selection::None,
                Selection::Site,
                Selection::Roots,
                Selection::Descendants(0),
                Selection::Descendants(1),
                Selection::Descendants(2),
                Selection::Descendants(3),
            ] {
                let (root, result, marker) = relay(1, tail)?;
                let (answer, answer_id) = leaf()?;
                let host = Host::new(
                    BTreeMap::from([(1, relay(2, true)?.0), (2, relay(3, true)?.0), (3, answer)]),
                    rejected_caller,
                );
                let facts = execute(&root, &host, selection, if tail { 1 } else { 2 });
                assert_write(&facts);
                let expected = if rejected_caller.is_some() {
                    None
                } else {
                    Some(&ConstValue::I32(42))
                };
                assert_eq!(
                    facts.returned().and_then(Value::constant),
                    expected,
                    "tail={tail} selection={selection:?}"
                );
                assert_eq!(facts.constant(result), if tail { None } else { expected });
                assert_eq!(
                    facts.constant(marker),
                    if tail {
                        None
                    } else {
                        Some(&ConstValue::I32(99))
                    }
                );
                if let Some(usage) = expected_usage {
                    assert_eq!(facts.usage(), usage);
                } else {
                    expected_usage = Some(facts.usage());
                }
                assert_eq!(
                    *host.body_bindings.borrow(),
                    [(Some(0), 1), (Some(1), 2), (Some(2), 3)]
                );
                let count = match selection {
                    Selection::None => 0,
                    Selection::Site | Selection::Roots => 1,
                    Selection::Descendants(limit) => limit,
                };
                assert_eq!(facts.invocations().len(), count);
                assert_eq!(
                    facts.call_observations_truncated(),
                    matches!(selection, Selection::Descendants(limit) if limit < 3)
                );
                let methods: Vec<_> = facts
                    .invocations()
                    .iter()
                    .map(|invocation| facts.body_methods().get(invocation.body).copied().flatten())
                    .collect();
                assert_eq!(
                    methods,
                    [Some(1), Some(2), Some(3)]
                        .get(..count)
                        .ok_or("unexpected retained count")?
                );
                let parents: Vec<_> = facts
                    .invocations()
                    .iter()
                    .map(|invocation| invocation.parent)
                    .collect();
                assert_eq!(
                    parents,
                    (0..count)
                        .map(|index| index.checked_sub(1))
                        .collect::<Vec<_>>()
                );
                for invocation in facts.invocations() {
                    let method = facts
                        .body_methods()
                        .get(invocation.body)
                        .copied()
                        .flatten()
                        .ok_or("missing entered method")?;
                    let expected = if rejected_caller.is_some() && method == 1 {
                        None
                    } else {
                        Some(&ConstValue::I32(42))
                    };
                    assert_eq!(
                        invocation.facts.returned().and_then(Value::constant),
                        expected,
                        "method={method} selection={selection:?}"
                    );
                    assert_write(&invocation.facts);
                    if method == 3 {
                        assert_eq!(
                            invocation.facts.constant(answer_id),
                            Some(&ConstValue::I32(42))
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn model_tail_binding_and_retired_body_binding_both_apply_without_losing_effects() -> TestResult {
    for root_tail in [false, true] {
        for reject_body in [false, true] {
            for reject_model in [false, true] {
                for cap in [0, 1, 8] {
                    let (root, result, marker) = relay(1, root_tail)?;
                    let mut host = Host::new(
                        BTreeMap::from([(1, relay(MODEL, true)?.0)]),
                        reject_body.then_some(0),
                    );
                    host.reject_model = reject_model;
                    let facts = execute(
                        &root,
                        &host,
                        Selection::Descendants(cap),
                        if root_tail { 1 } else { 2 },
                    );
                    assert_write(&facts);
                    let expected = if reject_body || reject_model {
                        None
                    } else {
                        Some(&ConstValue::I32(42))
                    };
                    assert_eq!(facts.returned().and_then(Value::constant), expected);
                    assert_eq!(
                        facts.constant(result),
                        if root_tail { None } else { expected }
                    );
                    assert_eq!(
                        facts.constant(marker),
                        if root_tail {
                            None
                        } else {
                            Some(&ConstValue::I32(99))
                        }
                    );
                    assert_eq!(*host.body_bindings.borrow(), [(Some(0), 1)]);
                    assert_eq!(*host.model_bindings.borrow(), [Some(1)]);
                    assert_eq!(facts.invocations().len(), usize::from(cap != 0));
                    if let Some(invocation) = facts.invocations().first() {
                        // The enclosing caller's contract does not change this body's own return.
                        assert_eq!(
                            invocation.facts.returned().and_then(Value::constant),
                            if reject_model {
                                None
                            } else {
                                Some(&ConstValue::I32(42))
                            }
                        );
                        assert_write(&invocation.facts);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Two unknown branches tail-enter the same implementation from different contract sites.
fn converging_root() -> TestResult<Function> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    builder.in_block(0, |block| block.branch(condition, 1, 2))?;
    for index in [1, 2] {
        builder.in_block(index, |block| {
            let result = block.emit_def(tmp(MockType::I32), |dest| {
                SsaOp::Call(Box::new(
                    CallData::new(Some(dest), Callee::Direct(1), vec![]).made_tail(),
                ))
            })?;
            block.ret(Some(result))
        })?;
    }
    verified(builder)
}

/// Unobserved branch clones can reach identical body/activation/position with different bindings.
#[test]
fn converging_tail_arrivals_keep_accepting_and_rejecting_contracts_separate() -> TestResult {
    for model_tail in [false, true] {
        for rejected_block in [None, Some(1), Some(2)] {
            for cap in [0, 1, 2] {
                let root = converging_root()?;
                let body = if model_tail {
                    relay(MODEL, true)?.0
                } else {
                    leaf()?.0
                };
                let mut host = Host::new(BTreeMap::from([(1, body)]), None);
                host.rejected_site = rejected_block.map(|block| (block, 0));
                let facts = execute(&root, &host, Selection::Descendants(cap), 1);
                assert_write(&facts);
                assert_eq!(
                    facts.returned().and_then(Value::constant),
                    if rejected_block.is_some() {
                        None
                    } else {
                        Some(&ConstValue::I32(42))
                    },
                    "model={model_tail} rejected={rejected_block:?} cap={cap}"
                );
                assert!(
                    facts.returned().is_some(),
                    "withheld knowledge is not a void return"
                );
                assert_eq!(facts.invocations().len(), cap);
                assert_eq!(facts.call_observations_truncated(), cap < 2);
                let sites: BTreeSet<_> = facts
                    .invocations()
                    .iter()
                    .map(|invocation| (invocation.block, invocation.instruction))
                    .collect();
                assert_eq!(sites.len(), cap);
                if cap == 2 {
                    assert_eq!(sites, BTreeSet::from([(1, 0), (2, 0)]));
                }
                for invocation in facts.invocations() {
                    assert_eq!(invocation.parent, None);
                    assert_eq!(
                        facts.body_methods().get(invocation.body).copied().flatten(),
                        Some(1)
                    );
                    assert_eq!(
                        invocation.facts.returned().and_then(Value::constant),
                        Some(&ConstValue::I32(42))
                    );
                    assert_write(&invocation.facts);
                }
            }
        }
    }
    Ok(())
}

fn void_tail(next: u32) -> TestResult<Function> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        block.emit_no_defs(SsaOp::Call(Box::new(
            CallData::new(None, Callee::Direct(next), vec![]).made_tail(),
        )))?;
        block.ret(None)
    })?;
    verified(builder)
}

/// A retired void wrapper discards a payload, while its selected child still reports 42.
#[test]
fn discarded_tail_result_is_void_without_erasing_precise_child_effects() -> TestResult {
    for cap in [0, 1, 2] {
        let root = void_tail(1)?;
        let mut host = Host::new(BTreeMap::from([(1, void_tail(2)?), (2, leaf()?.0)]), None);
        host.discard_caller = Some(1);
        let facts = execute(&root, &host, Selection::Descendants(cap), 1);
        assert_write(&facts);
        assert_eq!(facts.returned(), None);
        assert_eq!(facts.invocations().len(), cap);
        for invocation in facts.invocations() {
            let method = facts
                .body_methods()
                .get(invocation.body)
                .copied()
                .flatten()
                .ok_or("missing entered method")?;
            match method {
                1 => assert_eq!(invocation.facts.returned(), None),
                2 => assert_eq!(
                    invocation.facts.returned(),
                    Some(&Value::Const(ConstValue::I32(42)))
                ),
                _ => return Err(format!("unexpected method {method}").into()),
            }
            assert_write(&invocation.facts);
        }
    }
    Ok(())
}

/// Returns its caller-supplied pointer without touching its storage.
fn pointer_identity() -> TestResult<Function> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder.in_block(0, |block| block.ret(Some(input)))?;
    verified(builder)
}

struct AliasHost {
    body: Function,
}

impl CallModel<MockTarget> for AliasHost {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        if let Some(value) = call.argument(0).cloned() {
            call.return_value(value);
        }
    }

    fn only_memory(&self) -> bool {
        true
    }
}

impl ExecutionHost<MockTarget> for AliasHost {
    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 1).then_some(&self.body)
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == MODEL).then_some(self)
    }
    fn body_return_binding(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        _: &Function,
    ) -> ReturnBinding {
        ReturnBinding::Unknown
    }
    fn model_return_binding(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        _: &dyn CallModel<MockTarget>,
    ) -> ReturnBinding {
        ReturnBinding::Unknown
    }
}

fn stack_caller(
    method: u32,
    tail: bool,
    unknown_consumer: bool,
) -> TestResult<(Function, SsaVarId)> {
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
        .ok_or("missing stack pointer")?
        .set_role(VariableRole::EntryStackPointer);
    let returned = builder.in_block(0, |block| {
        let slot = block.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 1,
            offset: -16,
            result_type: MockType::Ptr,
        })?;
        let byte = block.const_value(tmp(MockType::U8), ConstValue::U8(0x96))?;
        block.store_indirect(slot, byte, MockType::U8)?;
        let returned = block.emit_def(tmp(MockType::Ptr), |dest| {
            let mut call = CallData::new(Some(dest), Callee::Direct(method), vec![slot]);
            call.tail = tail;
            SsaOp::Call(Box::new(call))
        })?;
        if unknown_consumer {
            block.call_void(Callee::Direct(404), vec![returned])?;
        }
        if tail {
            block.ret(Some(returned))?;
        } else {
            let loaded = block.load_indirect(tmp(MockType::U8), slot, MockType::U8)?;
            block.ret(Some(loaded))?;
        }
        Ok(returned)
    })?;
    Ok((verified(builder)?, returned))
}

#[test]
fn rejected_body_and_model_addresses_keep_stack_derived_aliases() -> TestResult {
    let host = AliasHost {
        body: pointer_identity()?,
    };
    for method in [1, MODEL] {
        for consumer in [false, true] {
            let (root, returned) = stack_caller(method, false, consumer)?;
            let facts = execute(&root, &host, Selection::Descendants(8), 2);
            assert!(facts.is_complete(), "{:?}", facts.completion());
            assert!(
                facts
                    .value(returned)
                    .is_some_and(Value::may_reference_stack)
            );
            assert_eq!(facts.constant(returned), None);
            assert_eq!(
                facts.returned().and_then(Value::constant),
                if consumer {
                    None
                } else {
                    Some(&ConstValue::U8(0x96))
                }
            );
        }
        let (root, returned) = stack_caller(method, true, false)?;
        let facts = execute(&root, &host, Selection::Descendants(8), 1);
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(facts.returned().is_some_and(Value::may_reference_stack));
        assert_eq!(facts.constant(returned), None);
    }
    Ok(())
}
