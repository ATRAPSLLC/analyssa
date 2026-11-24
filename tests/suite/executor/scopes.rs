//! Actual call-site observations retain independent SSA and ordered-memory domains.

use std::{cell::Cell, error::Error};

use analyssa::{
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            CallSite, Completion, ExecutionHost, ParameterBindings, StaticExecution, Value,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        CallData, Callee, SsaOp, SsaVarId,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::CmpKind,
        value::ConstValue,
        variable::{DefSite, VariableOrigin},
    },
    passes::residual::{apply_static_facts, fold_constants},
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// One immutable hermetic execution body.
type Function = SsaFunction<MockTarget>;

/// Finishes a fixture only after checking its real SSA structure.
fn verified(builder: SsaFunctionBuilder<MockTarget>) -> Result<Function, Box<dyn Error>> {
    let function = builder.finish()?;
    assert!(
        SsaVerifier::new(&function)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    Ok(function)
}

/// Builds `return input + 1`, retaining the callee-local result identity.
fn increment() -> Result<(Function, SsaVarId), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let result = builder.in_block(0, |block| {
        let one = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(1))?;
        let result = block.add(SsaDefSpec::tmp(MockType::I32), input, one)?;
        block.ret(Some(result))?;
        Ok(result)
    })?;
    Ok((verified(builder)?, result))
}

/// Two calls supplying different constants to the same immutable callee.
fn distinct_caller() -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let five = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(5))?;
        block.call(
            SsaDefSpec::tmp(MockType::I32),
            Callee::Direct(1),
            vec![five],
        )?;
        let nine = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(9))?;
        let returned = block.call(
            SsaDefSpec::tmp(MockType::I32),
            Callee::Direct(1),
            vec![nine],
        )?;
        block.ret(Some(returned))
    })?;
    verified(builder)
}

/// The host counts actual entries independently of observation selection.
struct Host {
    /// Actual callee body.
    callee: Function,
    /// Calls for which execution requested this body.
    entries: Cell<usize>,
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&Function> {
        if *method != 1 {
            return None;
        }
        self.entries.set(self.entries.get().saturating_add(1));
        Some(&self.callee)
    }
}

/// Distinct call contexts keep their constants despite overlapping root/callee SSA ids.
#[test]
fn selected_calls_keep_distinct_values_and_source_domains() -> Result<(), Box<dyn Error>> {
    let (callee, result) = increment()?;
    let host = Host {
        callee,
        entries: Cell::new(0),
    };
    let caller = distinct_caller()?;
    let context = AnalysisContext::new(MACHINE);
    let baseline = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .run();
    assert!(baseline.invocations().is_empty());
    host.entries.set(0);
    let selected = [(0, 3), (0, 1), (0, 3), (99, 99)];
    let mut facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_call_observations(&selected)
        .run();
    assert!(facts.is_complete());
    assert!(facts.is_root_entry());
    assert_eq!(facts.usage().steps, baseline.usage().steps);
    assert_eq!(host.entries.get(), 2);
    assert_eq!(facts.constant(result), Some(&ConstValue::I32(9)));
    assert_eq!(facts.invocations().len(), 2);
    for (record, (instruction, expected)) in facts.invocations().iter().zip([(1, 6), (3, 10)]) {
        assert_eq!((record.block, record.instruction), (0, instruction));
        assert_eq!(facts.body_methods().get(record.body), Some(&Some(1)));
        assert!(!record.facts.is_root_entry());
        assert!(record.facts.body_methods().is_empty());
        assert_eq!(record.facts.usage().steps, 0);
        assert_eq!(
            record.facts.constant(result),
            Some(&ConstValue::I32(expected))
        );
        assert_eq!(
            record.facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(expected))
        );
        let mut generic = host.callee.clone();
        assert_eq!(fold_constants(&mut generic, &record.facts)?, 0);
        let before = format!("{generic:?}");
        assert_eq!(apply_static_facts(&mut generic, &record.facts)?, 0);
        assert_eq!(format!("{generic:?}"), before);
    }
    let records = facts.take_invocations();
    assert_eq!(records.len(), 2);
    assert!(facts.invocations().is_empty());
    Ok(())
}

/// Selected calls are withheld atomically after either later exhaustion or cancellation.
#[test]
fn incomplete_enclosing_runs_withhold_completed_call_records() -> Result<(), Box<dyn Error>> {
    let (callee, _) = increment()?;
    let host = Host {
        callee,
        entries: Cell::new(0),
    };
    let caller = distinct_caller()?;
    let context = AnalysisContext::new(MACHINE);
    let polls = Cell::new(0usize);
    let probe = || {
        polls.set(polls.get().saturating_add(1));
        false
    };
    let complete = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_cancellation(&probe)
        .with_call_observations(&[(0, 1), (0, 3)])
        .run();
    assert_eq!(complete.invocations().len(), 2);
    let stopped = AnalysisContext::new(MACHINE).with_budget(WorkBudget::DEFAULT.with_limit(
        Budget::ExecutionSteps,
        complete.usage().steps.saturating_sub(1),
    ));
    host.entries.set(0);
    let limited = StaticExecution::new(&caller, &stopped)
        .with_host(&host)
        .with_call_observations(&[(0, 1), (0, 3)])
        .run();
    assert!(!limited.is_complete());
    assert!(host.entries.get() >= 1);
    assert!(limited.invocations().is_empty());
    let stop = polls.get();
    polls.set(0);
    host.entries.set(0);
    let cancel = || {
        polls.set(polls.get().saturating_add(1));
        polls.get() >= stop
    };
    let cancelled = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_cancellation(&cancel)
        .with_call_observations(&[(0, 1), (0, 3)])
        .run();
    assert_eq!(cancelled.completion(), Completion::Cancelled);
    assert!(host.entries.get() >= 1);
    assert!(cancelled.invocations().is_empty());
    Ok(())
}

/// One writable byte with no initial contents attested.
struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
    fn region(&self, address: u64) -> Option<ImageRegion> {
        (address == 0x2000).then_some(ImageRegion {
            start: 0x2000,
            end: 0x2001,
            mutability: Mutability::Writable,
        })
    }
    fn bytes(&self, _address: u64, _len: usize) -> Option<&[u8]> {
        None
    }
}

/// Observed reads consume the actual caller's ordered state, including unknown clobbers.
#[test]
fn selected_calls_keep_ordered_memory_and_unknown_clobbers() -> Result<(), Box<dyn Error>> {
    for clobber in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let loaded = builder.in_block(0, |block| {
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x2000))?;
            let loaded =
                block.load_indirect(SsaDefSpec::tmp(MockType::U8), address, MockType::U8)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })?;
        let host = Host {
            callee: verified(builder)?,
            entries: Cell::new(0),
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x2000))?;
            let first = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            block.store_indirect(address, first, MockType::U8)?;
            if clobber {
                block.call_void(Callee::Direct(99), vec![])?;
            }
            block.call_void(Callee::Direct(1), vec![])?;
            let second = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(9))?;
            block.store_indirect(address, second, MockType::U8)?;
            block.call_void(Callee::Direct(1), vec![])?;
            block.ret(None)
        })?;
        let caller = verified(builder)?;
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let sites = [
            (0, 3usize.saturating_add(usize::from(clobber))),
            (0, 6usize.saturating_add(usize::from(clobber))),
        ];
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .with_call_observations(&sites)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(facts.invocations().len(), 2);
        assert_eq!(
            facts
                .invocations()
                .first()
                .ok_or("missing first call")?
                .facts
                .constant(loaded),
            (!clobber).then_some(&ConstValue::U8(7))
        );
        assert_eq!(
            facts
                .invocations()
                .get(1)
                .ok_or("missing second call")?
                .facts
                .constant(loaded),
            Some(&ConstValue::U8(9))
        );
        // The callee only reads 0x2000: the 9 there is its caller's write,
        // which is the root's exit fact and not the call's.
        assert_eq!(
            facts
                .invocations()
                .get(1)
                .ok_or("missing second call")?
                .facts
                .image_writes(),
            vec![]
        );
        assert_eq!(facts.image_writes(), vec![(0x2000, vec![9])]);
    }
    Ok(())
}

/// Builds two visits to one call site with constant or changing arguments.
fn repeated_caller(changing: bool) -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let (zero, one, count) = builder.in_block(0, |block| {
        let zero = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
        let one = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(1))?;
        let count = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(2))?;
        block.jump(1)?;
        Ok((zero, one, count))
    })?;
    let index = builder.in_block(1, |block| block.empty_phi(SsaDefSpec::tmp(MockType::I32)))?;
    let next = builder.in_block(1, |block| {
        block.call_void(Callee::Direct(1), vec![if changing { index } else { zero }])?;
        let next = block.add(SsaDefSpec::tmp(MockType::I32), index, one)?;
        block.branch_cmp(next, count, CmpKind::Lt, false, 1, 2)?;
        Ok(next)
    })?;
    builder.add_phi_operand(1, index, 0, zero)?;
    builder.add_phi_operand(1, index, 1, next)?;
    builder.in_block(2, |block| block.ret(None))?;
    verified(builder)
}

/// Two calls at the same source point intersect rather than keeping the first visit's value.
#[test]
fn repeated_selected_call_visits_intersect_values() -> Result<(), Box<dyn Error>> {
    let (callee, result) = increment()?;
    let host = Host {
        callee,
        entries: Cell::new(0),
    };
    let caller = repeated_caller(true)?;
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_call_observations(&[(1, 0)])
        .run();
    assert!(facts.is_complete());
    assert_eq!(host.entries.get(), 2);
    assert_eq!(facts.invocations().len(), 1);
    assert_eq!(
        facts
            .invocations()
            .first()
            .ok_or("missing first call")?
            .facts
            .constant(result),
        None
    );
    Ok(())
}

/// Tail descendants retain return continuation while their definitions remain in another domain.
#[test]
fn selected_tail_continuations_keep_the_original_callee_domain() -> Result<(), Box<dyn Error>> {
    struct TailHost {
        source: Function,
        target: Function,
    }
    impl ExecutionHost<MockTarget> for TailHost {
        fn body(&self, method: &u32) -> Option<&Function> {
            match method {
                1 => Some(&self.source),
                2 => Some(&self.target),
                _ => None,
            }
        }
    }
    for tail_root in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let value = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(99))?;
            block.ret(Some(value))
        })?;
        let target = verified(builder)?;
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let local = builder.in_block(0, |block| {
            let local = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(7))?;
            block.emit_no_defs(SsaOp::Call(Box::new(CallData {
                dest: None,
                callee: Callee::Direct(2),
                args: vec![],
                effects: None,
                tail: true,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: vec![],
            })))?;
            block.ret(None)?;
            Ok(local)
        })?;
        let host = TailHost {
            source: verified(builder)?,
            target,
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let input = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(5))?;
            let returned = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                SsaOp::Call(Box::new(CallData {
                    dest: Some(dest),
                    callee: Callee::Direct(1),
                    args: vec![input],
                    effects: None,
                    tail: tail_root,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: vec![],
                }))
            })?;
            block.ret(Some(returned))
        })?;
        let caller = verified(builder)?;
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .with_call_observations(&[(0, 1)])
            .run();
        assert!(facts.is_complete());
        assert_eq!(facts.invocations().len(), 1);
        let record = facts.invocations().first().ok_or("missing first call")?;
        assert_eq!(record.facts.constant(local), Some(&ConstValue::I32(7)));
        assert_eq!(
            record.facts.constant(SsaVarId::from_index(0)),
            Some(&ConstValue::I32(5))
        );
        assert_eq!(
            record.facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(99))
        );
        assert_eq!(
            facts.constant(SsaVarId::from_index(0)),
            Some(&ConstValue::I32(5))
        );
    }
    Ok(())
}

/// Multiple host bindings retain the executor's final admitted source, without observing stale entries.
#[test]
fn selected_entry_facts_follow_final_duplicate_bindings() -> Result<(), Box<dyn Error>> {
    struct Aliased {
        callee: Function,
    }
    impl ExecutionHost<MockTarget> for Aliased {
        fn body(&self, method: &u32) -> Option<&Function> {
            (*method == 1).then_some(&self.callee)
        }
        fn parameters(
            &self,
            _site: CallSite<'_, MockTarget>,
            _method: &u32,
            _body: &Function,
            bindings: &mut ParameterBindings<'_, MockTarget>,
        ) {
            bindings.bind(SsaVarId::from_index(0), 0);
            bindings.bind(SsaVarId::from_index(0), 1);
            // An unrelated host binding cannot manufacture an SSA identity.
            bindings.bind(SsaVarId::from_index(999), 0);
        }
    }
    let (callee, result) = increment()?;
    let host = Aliased { callee };
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let first = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(5))?;
        let last = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(9))?;
        let returned = block.call(
            SsaDefSpec::tmp(MockType::I32),
            Callee::Direct(1),
            vec![first, last],
        )?;
        block.ret(Some(returned))
    })?;
    let caller = verified(builder)?;
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_call_observations(&[(0, 2)])
        .run();
    assert!(facts.is_complete());
    let record = facts.invocations().first().ok_or("missing call")?;
    assert_eq!(
        record.facts.constant(SsaVarId::from_index(0)),
        Some(&ConstValue::I32(9))
    );
    assert_eq!(record.facts.constant(result), Some(&ConstValue::I32(10)));
    assert_eq!(record.facts.constant(SsaVarId::from_index(999)), None);
    Ok(())
}

/// A tail-self-call returns through the selected call without merging its new SSA activation.
#[test]
fn selected_tail_self_calls_keep_first_activation_definitions() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let one = builder.in_block(0, |block| {
        let one = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(1))?;
        let zero = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
        block.branch_cmp(input, zero, CmpKind::Eq, false, 1, 2)?;
        Ok(one)
    })?;
    let answer = builder.in_block(1, |block| {
        let answer = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(42))?;
        block.ret(Some(answer))?;
        Ok(answer)
    })?;
    let next = builder.in_block(2, |block| {
        let next = block.sub(SsaDefSpec::tmp(MockType::I32), input, one)?;
        block.emit_no_defs(SsaOp::Call(Box::new(CallData {
            dest: None,
            callee: Callee::Direct(1),
            args: vec![next],
            effects: None,
            tail: true,
            implicit_inputs: Default::default(),
            results: Default::default(),
            staged: vec![],
        })))?;
        block.ret(None)?;
        Ok(next)
    })?;
    let host = Host {
        callee: verified(builder)?,
        entries: Cell::new(0),
    };
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let input = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(2))?;
        let returned = block.call(
            SsaDefSpec::tmp(MockType::I32),
            Callee::Direct(1),
            vec![input],
        )?;
        block.ret(Some(returned))
    })?;
    let caller = verified(builder)?;
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_call_observations(&[(0, 1)])
        .run();
    assert!(facts.is_complete());
    assert_eq!(host.entries.get(), 3);
    let record = facts.invocations().first().ok_or("missing selected call")?;
    assert_eq!(record.facts.constant(input), Some(&ConstValue::I32(2)));
    assert_eq!(record.facts.constant(answer), None);
    assert_eq!(record.facts.constant(next), Some(&ConstValue::I32(1)));
    assert_eq!(
        record.facts.returned().and_then(Value::constant),
        Some(&ConstValue::I32(42))
    );
    Ok(())
}

/// Omitted bindings spoil entry constants in either order without hiding independent definitions.
#[test]
fn repeated_selected_call_bindings_include_missing_visits() -> Result<(), Box<dyn Error>> {
    struct PartialHost {
        callee: Function,
        supplied: [bool; 2],
        visits: Cell<usize>,
    }
    impl ExecutionHost<MockTarget> for PartialHost {
        fn body(&self, method: &u32) -> Option<&Function> {
            (*method == 1).then_some(&self.callee)
        }
        fn parameters(
            &self,
            _site: CallSite<'_, MockTarget>,
            _method: &u32,
            body: &Function,
            bindings: &mut ParameterBindings<'_, MockTarget>,
        ) {
            let visit = self.visits.get();
            self.visits.set(visit.saturating_add(1));
            if self.supplied.get(visit).copied().unwrap_or(false) {
                for variable in body.variables() {
                    if variable.version() == 0
                        && let VariableOrigin::Argument(index) = variable.origin()
                    {
                        bindings.bind(variable.id(), usize::from(index));
                    }
                }
            }
        }
    }
    let caller = repeated_caller(false)?;
    for supplied in [[true, false], [false, true], [true, true], [false, false]] {
        let (callee, result) = increment()?;
        let input = callee
            .variables()
            .iter()
            .find(|variable| variable.origin() == VariableOrigin::Argument(0))
            .ok_or("callee input")?
            .id();
        let independent = callee
            .blocks()
            .first()
            .and_then(|block| block.instructions().first())
            .and_then(|instruction| instruction.def())
            .ok_or("independent constant")?;
        let host = PartialHost {
            callee,
            supplied,
            visits: Cell::new(0),
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .with_call_observations(&[(1, 0)])
            .run();
        assert!(facts.is_complete());
        assert_eq!(host.visits.get(), 2);
        let observed = &facts.invocations().first().ok_or("call")?.facts;
        let complete = supplied.into_iter().all(|present| present);
        assert_eq!(
            observed.constant(input),
            complete.then_some(&ConstValue::I32(0))
        );
        assert_eq!(
            observed.constant(result),
            complete.then_some(&ConstValue::I32(1))
        );
        assert_eq!(observed.constant(independent), Some(&ConstValue::I32(1)));
    }
    Ok(())
}

/// Actual entry limits preserve execution and intersect repeated visits without new slots.
#[test]
fn root_call_observation_limits_count_actual_pairs() -> Result<(), Box<dyn Error>> {
    for repeated in [false, true] {
        let (callee, result) = increment()?;
        let host = Host {
            callee,
            entries: Cell::new(0),
        };
        let caller = if repeated {
            repeated_caller(true)?
        } else {
            distinct_caller()?
        };
        let context = AnalysisContext::new(MACHINE);
        let baseline = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        let pairs = if repeated { 1 } else { 2 };
        for maximum in [0, 1, 2] {
            host.entries.set(0);
            let facts = StaticExecution::new(&caller, &context)
                .with_host(&host)
                .with_root_call_observations(maximum)
                .run();
            assert!(facts.is_complete());
            assert_eq!(facts.usage(), baseline.usage());
            assert_eq!(host.entries.get(), 2);
            assert_eq!(facts.invocations().len(), maximum.min(pairs));
            assert_eq!(facts.call_observations_truncated(), maximum < pairs);
            if repeated && maximum > 0 {
                let record = facts.invocations().first().ok_or("repeated call")?;
                assert_eq!(record.facts.constant(result), None);
            }
        }
        let stopped = AnalysisContext::new(MACHINE)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionSteps, 1));
        let facts = StaticExecution::new(&caller, &stopped)
            .with_host(&host)
            .with_root_call_observations(0)
            .run();
        assert!(!facts.is_complete());
        assert!(!facts.call_observations_truncated());
        assert!(facts.invocations().is_empty());
    }
    Ok(())
}

/// **A call context's exit holds the bytes its own execution wrote, one run
/// per writing instruction.** The caller's byte before the call is not the
/// callee's. The callee's own byte and the byte the body it calls writes are
/// two writes, so two runs; a tail call leaves both to the retired callee,
/// whose own write before the tail call stays its own.
#[test]
fn selected_call_exits_keep_own_writes_split_by_writer() -> Result<(), Box<dyn Error>> {
    struct Bodies {
        source: Function,
        target: Function,
    }
    impl ExecutionHost<MockTarget> for Bodies {
        fn body(&self, method: &u32) -> Option<&Function> {
            match method {
                1 => Some(&self.source),
                2 => Some(&self.target),
                _ => None,
            }
        }
    }
    for tail in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x2002))?;
            let value = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(6))?;
            block.store_indirect(address, value, MockType::U8)?;
            block.ret(None)
        })?;
        let target = verified(builder)?;
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x2001))?;
            let value = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(5))?;
            block.store_indirect(address, value, MockType::U8)?;
            block.emit_no_defs(SsaOp::Call(Box::new(CallData {
                dest: None,
                callee: Callee::Direct(2),
                args: vec![],
                effects: None,
                tail,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: vec![],
            })))?;
            block.ret(None)
        })?;
        let host = Bodies {
            source: verified(builder)?,
            target,
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x2000))?;
            let value = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            block.store_indirect(address, value, MockType::U8)?;
            block.call_void(Callee::Direct(1), vec![])?;
            block.ret(None)
        })?;
        let caller = verified(builder)?;
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .with_call_observations(&[(0, 3)])
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let call = facts.invocations().first().ok_or("missing call record")?;
        assert_eq!(
            call.facts.image_writes(),
            vec![(0x2001, vec![5]), (0x2002, vec![6])],
            "tail {tail}"
        );
        assert_eq!(
            facts.image_writes(),
            vec![(0x2000, vec![7]), (0x2001, vec![5]), (0x2002, vec![6])],
            "tail {tail}"
        );
    }
    Ok(())
}
