//! A call takes what its callee's return hands back beside its result.

use std::error::Error;

use analyssa::{
    analysis::{
        context::AnalysisContext,
        execution::{Completion, ExecutionHost, StaticExecution, Value},
    },
    ir::{
        CallData, CallInput, CallInputKey, Callee, SsaOp,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
    },
    testing::{MACHINE, MockTarget, MockType},
};

/// The key both sides name the value returned beside the result by: NSIS's
/// `GetIntFromParm` leaves the string's first character in `edx`.
const BESIDE: CallInputKey = CallInputKey::new(2);

/// A host whose method 2 is `callee`, and which knows no other.
struct Host {
    callee: SsaFunction<MockTarget>,
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 2).then_some(&self.callee)
    }
}

/// A callee returning 17 and handing back 42 beside it, under `key`.
fn callee(key: CallInputKey) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let result = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(17))?;
        let beside = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
        block.emit_no_defs(SsaOp::returns_results(
            Some(result),
            vec![CallInput { key, value: beside }],
        ))
    })?;
    Ok(builder.finish()?)
}

/// A root calling `method` and returning what the call returns beside its
/// result under [`BESIDE`].
fn root(method: u32) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let (_, beside) = block.emit_two_defs(
            SsaDefSpec::tmp(MockType::U32),
            SsaDefSpec::tmp(MockType::U32),
            |result, beside| {
                SsaOp::from(
                    CallData::new(Some(result), Callee::Direct(method), vec![])
                        .with_result(BESIDE, beside),
                )
            },
        )?;
        block.ret(Some(beside))
    })?;
    Ok(builder.finish()?)
}

/// What the root returns: what its call received beside the result.
fn returned(method: u32, key: CallInputKey) -> Result<Option<Value<MockTarget>>, Box<dyn Error>> {
    let host = Host {
        callee: callee(key)?,
    };
    let root = root(method)?;
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_method(0)
        .run();
    assert_eq!(facts.completion(), Completion::Complete);
    Ok(facts.returned().cloned())
}

/// **A call takes what its callee hands back under the key it receives it
/// by.** The callee returns 17 and hands back 42 under the key the call
/// receives a second value by: the call's second value is 42. Handed back
/// under another key, or by a call no body answers, it is an unknown value,
/// no constant.
#[test]
fn a_call_takes_what_its_callee_hands_back_beside_its_result() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        returned(2, BESIDE)?,
        Some(Value::Const(ConstValue::U32(42))),
        "handed back under its key"
    );
    for (method, key, case) in [
        (2, CallInputKey::new(3), "handed back under another key"),
        (5, BESIDE, "a call no body answers"),
    ] {
        let value = returned(method, key)?;
        assert!(
            value.is_some() && value.as_ref().and_then(Value::constant).is_none(),
            "{case}: {value:?}"
        );
    }
    Ok(())
}
