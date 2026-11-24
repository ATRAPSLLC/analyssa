//! A model's result reaches the registers its host states it leaves it in,
//! and no other output of its call.

use std::ptr;

use analyssa::{
    Error, Result,
    analysis::{
        VerifyLevel,
        context::AnalysisContext,
        execution::{
            CallModel, CallSite, Completion, ExecutionHost, MachineEffects, MachineOutput,
            MemoryRefusal, ModelCall, Placement, StaticExecution, Value,
        },
    },
    ir::{
        CallData, CallInputKey, CallOutput, Callee, ConstValue, SsaOp,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
    },
    target::Target,
    testing::{MACHINE, MockTarget, MockType},
};

const OTHER: CallInputKey = CallInputKey::new(100);
const RESULT: CallInputKey = CallInputKey::new(101);

struct Model {
    narrow: bool,
    alternatives: usize,
}
impl CallModel<MockTarget> for Model {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        self.call_alternative(0, call);
    }
    fn alternatives(&self) -> usize {
        self.alternatives
    }
    fn call_alternative(&self, alternative: usize, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Const(if self.narrow {
            ConstValue::U32(42)
        } else {
            ConstValue::U64(42u64.saturating_add(alternative as u64))
        }));
    }
    fn tracks_all_memory_accesses(&self) -> bool {
        true
    }
    fn only_memory(&self) -> bool {
        true
    }
}
struct Host {
    model: Model,
    attested: bool,
    duplicate: bool,
    placement: Placement,
}
impl ExecutionHost<MockTarget> for Host {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 2).then_some(&self.model)
    }
    fn model_machine_effects(
        &self,
        _: CallSite<'_, MockTarget>,
        method: &u32,
        model: &dyn CallModel<MockTarget>,
    ) -> MachineEffects {
        assert_eq!(*method, 2);
        assert!(ptr::addr_eq(model, &self.model));
        if !self.attested {
            return MachineEffects::default();
        }
        let Some(layout) = MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size) else {
            return MachineEffects::default();
        };
        MachineEffects {
            outputs: vec![
                MachineOutput {
                    key: RESULT,
                    layout,
                    placement: self.placement,
                };
                if self.duplicate { 2 } else { 1 }
            ],
        }
    }
}
/// Hands the model 1 in [`RESULT`] and 7 in [`OTHER`], reads the call's
/// [`RESULT`] output, and [`OTHER`] where `read_other` says, and returns
/// their sum with the 1.
fn program(read_other: bool) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let other = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(7))?;
        let before = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(1))?;
        block.emit_no_defs(
            CallData::new(None, Callee::Direct(2), vec![])
                .with_implicit_input(OTHER, other)
                .with_implicit_input(RESULT, before)
                .into(),
        )?;
        let actual = block.emit_def(SsaDefSpec::tmp(MockType::U64), |dest| SsaOp::CallClobber {
            outputs: vec![dest],
            machine: Some(
                vec![CallOutput {
                    key: RESULT,
                    index: 0,
                }]
                .into_boxed_slice(),
            ),
        })?;
        let mut sum = block.add(SsaDefSpec::tmp(MockType::U64), actual, before)?;
        if read_other {
            let other =
                block.emit_def(SsaDefSpec::tmp(MockType::U64), |dest| SsaOp::CallClobber {
                    outputs: vec![dest],
                    machine: Some(
                        vec![CallOutput {
                            key: OTHER,
                            index: 0,
                        }]
                        .into_boxed_slice(),
                    ),
                })?;
            sum = block.add(SsaDefSpec::tmp(MockType::U64), sum, other)?;
        }
        block.ret(Some(sum))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}
/// The call's output reads the model's 42, and the 1 handed to it in the
/// same register is still the value it was.
#[test]
fn model_actual_result_reaches_its_output_and_keeps_prior_ssa_unchanged() -> Result<()> {
    let function = program(false)?;
    let host = Host {
        model: Model {
            narrow: false,
            alternatives: 1,
        },
        attested: true,
        duplicate: false,
        placement: Placement::Whole,
    };
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&function, &context)
        .with_host(&host)
        .with_method(1)
        .run();
    assert_eq!(facts.completion(), Completion::Complete);
    assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U64(43))));
    Ok(())
}
/// A result the host states no register for, a narrow result in a wide
/// register, and a register the model leaves nothing in read unknown, never
/// what the caller handed the call.
#[test]
fn model_unsupported_effects_and_narrow_results_cannot_reuse_old_bits() -> Result<()> {
    for (narrow, attested, other) in [
        (true, true, false),
        (false, false, false),
        (false, true, true),
    ] {
        let function = program(other)?;
        let host = Host {
            model: Model {
                narrow,
                alternatives: 1,
            },
            attested,
            duplicate: false,
            placement: Placement::Whole,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .with_method(1)
            .run();
        assert_eq!(facts.completion(), Completion::Complete);
        assert!(facts.returned().and_then(Value::constant).is_none());
    }
    Ok(())
}
#[test]
fn narrow_model_results_reach_wide_cells_under_their_placement() -> Result<()> {
    for (placement, expected) in [
        (
            Placement::Extended {
                bits: 32,
                signed: false,
            },
            Some(43),
        ),
        (Placement::Low { bits: 32 }, None),
    ] {
        let function = program(false)?;
        let host = Host {
            model: Model {
                narrow: true,
                alternatives: 1,
            },
            attested: true,
            duplicate: false,
            placement,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .with_method(1)
            .run();
        assert_eq!(facts.completion(), Completion::Complete);
        let returned = facts.returned().ok_or_else(|| Error::new("returned"))?;
        // Unstated upper bits make no constant of the wide cell.
        assert_eq!(returned.constant().and_then(ConstValue::as_u64), expected);
    }
    Ok(())
}
#[test]
fn model_alternatives_join_physical_results_without_cross_path_updates() -> Result<()> {
    let function = program(false)?;
    let host = Host {
        model: Model {
            narrow: false,
            alternatives: 2,
        },
        attested: true,
        duplicate: false,
        placement: Placement::Whole,
    };
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&function, &context)
        .with_host(&host)
        .with_method(1)
        .run();
    assert_eq!(facts.completion(), Completion::Complete);
    assert!(facts.returned().and_then(Value::constant).is_none());
    Ok(())
}
#[test]
fn duplicate_model_outputs_refuse_instead_of_overwriting_a_location() -> Result<()> {
    let function = program(false)?;
    let host = Host {
        model: Model {
            narrow: false,
            alternatives: 1,
        },
        attested: true,
        duplicate: true,
        placement: Placement::Whole,
    };
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&function, &context)
        .with_host(&host)
        .with_method(1)
        .run();
    assert_eq!(
        facts.completion(),
        Completion::Memory {
            block: 0,
            reason: MemoryRefusal::MachineState
        }
    );
    Ok(())
}
