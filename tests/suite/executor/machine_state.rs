//! The values a body's registers hold at its entry and after its calls.

use analyssa::{
    Result,
    analysis::{
        VerifyLevel,
        context::AnalysisContext,
        execution::{
            CallModel, Completion, ExecutionHost, MemoryRefusal, ModelCall, StaticExecution, Value,
        },
    },
    ir::{
        CallData, CallInputKey, CallOutput, Callee, ConstValue, SsaOp, SsaVarId,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        variable::{DefSite, VariableOrigin},
    },
    testing::{MACHINE, MockTarget, MockType},
};

type Function = SsaFunction<MockTarget>;
const CELL: CallInputKey = CallInputKey::new(101);

fn input(builder: &mut SsaFunctionBuilder<MockTarget>, ty: MockType) -> SsaVarId {
    builder
        .function_mut()
        .create_variable(VariableOrigin::EntryLiveIn, 0, DefSite::entry(), ty)
}

/// A model returning 42.
struct Returns;

impl CallModel<MockTarget> for Returns {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Const(ConstValue::I64(42)));
    }
}

/// Models method 2 by [`Returns`].
struct Modelled;

impl ExecutionHost<MockTarget> for Modelled {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 2).then_some(&Returns as &dyn CallModel<MockTarget>)
    }
}

/// A root calling method 2, whose outputs name [`CELL`] at `index`, and
/// returning the call's result.
fn keyed_call(index: u32) -> Result<Function> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let result = block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| {
            CallData::new(Some(dest), Callee::Direct(2), vec![]).into()
        })?;
        block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::CallClobber {
            outputs: vec![dest],
            machine: Some(vec![CallOutput { key: CELL, index }].into_boxed_slice()),
        })?;
        block.ret(Some(result))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

/// A call's output keys name the outputs it defines: one naming an output
/// the call does not define refuses the run rather than leaving a value
/// undefined.
#[test]
fn an_output_index_outside_the_fresh_definition_list_refuses_the_trial() -> Result<()> {
    let context = AnalysisContext::new(MACHINE);
    let named = keyed_call(0)?;
    let facts = StaticExecution::new(&named, &context)
        .with_host(&Modelled)
        .run();
    assert_eq!(facts.completion(), Completion::Complete);
    assert_eq!(
        facts.returned().and_then(Value::constant),
        Some(&ConstValue::I64(42))
    );
    let outside = keyed_call(1)?;
    let facts = StaticExecution::new(&outside, &context)
        .with_host(&Modelled)
        .run();
    assert!(matches!(
        facts.completion(),
        Completion::Memory {
            reason: MemoryRefusal::MachineState,
            ..
        }
    ));
    assert!(facts.returned().and_then(Value::constant).is_none());
    Ok(())
}

/// A host attesting what a root is entered with.
struct Entered(Vec<(SsaVarId, Value<MockTarget>)>);

impl ExecutionHost<MockTarget> for Entered {
    fn root_entry_values(
        &self,
        _: &SsaFunction<MockTarget>,
        _: Option<&u32>,
    ) -> Vec<(SsaVarId, Value<MockTarget>)> {
        self.0.clone()
    }
}

/// A root returning its entry value, and the definition inside it.
fn entered_root() -> Result<(Function, SsaVarId, SsaVarId)> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let entry = input(&mut builder, MockType::U64);
    let mut defined = None;
    builder.in_block(0, |block| {
        defined = Some(block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(1))?);
        block.ret(Some(entry))
    })?;
    let defined = defined.ok_or_else(|| analyssa::Error::new("definition"))?;
    Ok((builder.finish_verified(VerifyLevel::Full)?, entry, defined))
}

/// A root holds the entry values its host attests; nothing names a later
/// definition, and without the attestation the entry value stays unknown.
#[test]
fn root_entry_values_bind_only_entry_definitions() -> Result<()> {
    let (root, entry, defined) = entered_root()?;
    let context = AnalysisContext::new(MACHINE);
    let run = |host: &Entered| StaticExecution::new(&root, &context).with_host(host).run();
    let bound = run(&Entered(vec![(entry, Value::Const(ConstValue::U64(7)))]));
    assert_eq!(bound.completion(), Completion::Complete);
    assert_eq!(bound.returned(), Some(&Value::Const(ConstValue::U64(7))));
    let unbound = run(&Entered(Vec::new()));
    assert_eq!(unbound.completion(), Completion::Complete);
    assert!(unbound.returned().and_then(Value::constant).is_none());
    for refused in [
        vec![(defined, Value::Const(ConstValue::U64(7)))],
        vec![
            (entry, Value::Const(ConstValue::U64(7))),
            (entry, Value::Const(ConstValue::U64(8))),
        ],
    ] {
        assert_eq!(
            run(&Entered(refused)).completion(),
            Completion::Memory {
                block: 0,
                reason: MemoryRefusal::MachineState
            }
        );
    }
    Ok(())
}
