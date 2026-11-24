//! A call every visit sends to one method becomes a direct call to it: an
//! indirect call, and a direct one that entered another method than it names.

use std::cell::RefCell;

use analyssa::{
    Error, Result,
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::WorkBudget,
        context::AnalysisContext,
        execution::{
            Address, CallModel, CallSite, CodeMemory, ExecutionHost, ModelCall, StaticExecution,
            Value,
        },
    },
    ir::{
        Callee, ConstValue, SsaOp,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        variable::{DefSite, VariableOrigin},
    },
    passes::residual::{ResidualRewrite, rewrite_static_facts},
    testing::{MACHINE, MockTarget, MockType},
};

/// Resolves code at 0x100 to method 7 and at 0x200 to method 8; nothing else.
/// A direct call naming method 5 enters method 7, as one naming code written
/// since enters the version recovered from it.
struct Host;

impl ExecutionHost<MockTarget> for Host {
    fn method_for_call(
        &self,
        _: CallSite<'_, MockTarget>,
        method: &u32,
        _: CodeMemory<'_>,
    ) -> Option<u32> {
        Some(if *method == 5 { 7 } else { *method })
    }

    fn method_at_call(
        &self,
        _: CallSite<'_, MockTarget>,
        address: Address,
        _: CodeMemory<'_>,
    ) -> Option<u32> {
        match address.image_address()? {
            0x100 => Some(7),
            0x200 => Some(8),
            _ => None,
        }
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        matches!(*method, 7 | 8).then_some(self)
    }
}

impl CallModel<MockTarget> for Host {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Const(ConstValue::I64(42)));
    }
}

/// A root that picks its call target on an unknown argument - `first` on one
/// path, `second` on the other - and calls it through one indirect site.
fn root(first: u32, second: u32) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I1,
    );
    builder.in_block(0, |block| block.branch(condition, 1, 2))?;
    let mut targets = Vec::new();
    for (index, address) in [(1, first), (2, second)] {
        let target = builder.in_block(index, |block| {
            let target =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(address))?;
            block.jump(3)?;
            Ok(target)
        })?;
        targets.push((index, target));
    }
    builder.in_block(3, |block| {
        let target = block.phi(SsaDefSpec::tmp(MockType::Ptr), targets)?;
        let result = block.call(
            SsaDefSpec::tmp(MockType::I64),
            Callee::Indirect {
                fptr: target,
                signature: 0,
            },
            Vec::new(),
        )?;
        block.ret(Some(result))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

/// A root that calls method `method` directly and returns what it returns.
fn direct_root(method: u32) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let result = block.call(
            SsaDefSpec::tmp(MockType::I64),
            Callee::Direct(method),
            Vec::new(),
        )?;
        block.ret(Some(result))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

/// The callee the rewritten root's one call names, if it was rewritten.
fn rewritten_callee(first: u32, second: u32) -> Result<Option<Callee<MockTarget>>> {
    rewritten(&root(first, second)?)
}

/// The callee `function`'s one call names once rewritten, if it was.
fn rewritten(function: &SsaFunction<MockTarget>) -> Result<Option<Callee<MockTarget>>> {
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(function, &context)
        .with_host(&Host)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    match rewrite_static_facts(function, &facts, &WorkBudget::DEFAULT, &|| false)? {
        ResidualRewrite::Rewritten { function, .. } => {
            assert!(
                SsaVerifier::new(&function)
                    .verify(VerifyLevel::Full)
                    .is_empty()
            );
            Ok(function
                .iter_instructions()
                .find_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::Call(call) => Some(call.callee.clone()),
                    _ => None,
                }))
        }
        ResidualRewrite::Unchanged => Ok(None),
        ResidualRewrite::Refused(reason) => Err(Error::new(format!("refused: {reason:?}"))),
    }
}

/// Every visit reaching one method makes the call direct to it.
#[test]
fn one_executed_target_becomes_a_direct_call() -> Result<()> {
    assert_eq!(rewritten_callee(0x100, 0x100)?, Some(Callee::Direct(7)));
    Ok(())
}

/// **A direct call that entered another method than it names is stated as a
/// call to that method.** The code at the address it names was written
/// since, and what ran is the version recovered from it; a direct call that
/// entered the method it names is left as it is.
#[test]
fn a_direct_call_entering_another_method_names_it() -> Result<()> {
    assert_eq!(rewritten(&direct_root(5)?)?, Some(Callee::Direct(7)));
    assert!(
        rewritten(&direct_root(8)?)?.is_none_or(|callee| callee == Callee::Direct(8)),
        "the call entered what it names"
    );
    Ok(())
}

/// Visits reaching two methods, or one reaching nothing, leave it indirect.
#[test]
fn differing_or_unresolved_targets_stay_indirect() -> Result<()> {
    for (first, second) in [(0x100, 0x200), (0x100, 0x300)] {
        let callee = rewritten_callee(first, second)?;
        assert!(
            callee
                .as_ref()
                .is_none_or(|callee| matches!(callee, Callee::Indirect { .. })),
            "{first:#x}/{second:#x}: {callee:?}"
        );
    }
    Ok(())
}

/// Bodies for methods 7 and 8, each returning 42, and every transfer the
/// execution offers it, with whether its target was computed.
struct Bodies {
    callee: SsaFunction<MockTarget>,
    offered: RefCell<Vec<(u32, bool)>>,
}

impl ExecutionHost<MockTarget> for Bodies {
    fn method_at_call(
        &self,
        _: CallSite<'_, MockTarget>,
        address: Address,
        _: CodeMemory<'_>,
    ) -> Option<u32> {
        (address.image_address()? == 0x100).then_some(7)
    }

    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        matches!(*method, 7 | 8).then_some(&self.callee)
    }

    fn observes_code_transfer(&self, _: Option<&u32>, to: &u32, computed: bool) -> bool {
        self.offered.borrow_mut().push((*to, computed));
        true
    }
}

/// **A transfer through a computed target is offered as computed; one the
/// call names is not.** The root calls method 8 directly, then method 7
/// through a pointer it computed.
#[test]
fn a_computed_call_target_is_offered_as_computed() -> Result<()> {
    let mut callee = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    callee.in_block(0, |block| {
        let value = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(42))?;
        block.ret(Some(value))
    })?;
    let mut root = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    root.in_block(0, |block| {
        block.call(
            SsaDefSpec::tmp(MockType::I64),
            Callee::Direct(8),
            Vec::new(),
        )?;
        let target =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x100))?;
        let result = block.call(
            SsaDefSpec::tmp(MockType::I64),
            Callee::Indirect {
                fptr: target,
                signature: 0,
            },
            Vec::new(),
        )?;
        block.ret(Some(result))
    })?;
    let root = root.finish_verified(VerifyLevel::Full)?;
    let host = Bodies {
        callee: callee.finish_verified(VerifyLevel::Full)?,
        offered: RefCell::new(Vec::new()),
    };
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_code_transfers(8)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(*host.offered.borrow(), vec![(8, false), (7, true)]);
    assert_eq!(facts.code_transfers().len(), 2);
    Ok(())
}
