//! Host instruction contracts guard both roots and entered bodies.

use std::error::Error;

use analyssa::{
    PointerSize,
    analysis::{
        Machine,
        context::AnalysisContext,
        execution::{Completion, ExecutionHost, StaticExecution},
    },
    ir::{
        Callee,
        division::DivisionRules,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

struct Host {
    callee: SsaFunction<MockTarget>,
    root_stable: bool,
    callee_stable: bool,
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        matches!(*method, 1 | 2).then_some(&self.callee)
    }
    fn code_is_stable(&self, _body: &SsaFunction<MockTarget>, method: Option<&u32>) -> bool {
        if method == Some(&2) {
            self.callee_stable
        } else {
            method == Some(&1) || self.root_stable
        }
    }
}

#[test]
fn unstable_roots_and_callees_withhold_all_observations() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |b| {
        let value = b.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
        b.ret(Some(value))
    })?;
    let callee = builder.finish()?;
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |b| {
        b.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
        b.call(SsaDefSpec::tmp(MockType::U32), Callee::Direct(1), vec![])?;
        let value = b.call(SsaDefSpec::tmp(MockType::U32), Callee::Direct(2), vec![])?;
        b.ret(Some(value))
    })?;
    let root = builder.finish()?;
    for root_stable in [false, true] {
        for callee_stable in [false, true] {
            let host = Host {
                callee: callee.clone(),
                root_stable,
                callee_stable,
            };
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&root, &context)
                .with_host(&host)
                .with_method(0)
                .with_root_call_observations(2)
                .run();
            if root_stable && callee_stable {
                assert_eq!(facts.completion(), Completion::Complete);
                assert_eq!(
                    facts
                        .returned()
                        .and_then(|v| v.constant())
                        .and_then(ConstValue::as_u64_bits),
                    Some(42)
                );
                assert_eq!(facts.invocations().len(), 2);
            } else {
                assert_eq!(facts.completion(), Completion::UnstableCode);
                assert!(facts.constants().is_empty());
                assert!(facts.invocations().is_empty());
                assert!(facts.returned().is_none());
            }
        }
    }
    Ok(())
}

struct MachineHost {
    callee: SsaFunction<MockTarget>,
    root: Machine,
    callee_machine: Machine,
}

impl ExecutionHost<MockTarget> for MachineHost {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        matches!(*method, 1 | 2).then_some(&self.callee)
    }
    fn body_machine(&self, _: &SsaFunction<MockTarget>, method: Option<&u32>) -> Option<Machine> {
        Some(if method == Some(&2) {
            self.callee_machine
        } else {
            self.root
        })
    }
}

#[test]
fn incompatible_machine_roots_and_callees_withhold_all_observations() -> Result<(), Box<dyn Error>>
{
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let value = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
        block.ret(Some(value))
    })?;
    let callee = builder.finish()?;
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(13))?;
        block.call_void(Callee::Direct(1), vec![])?;
        let value = block.call(SsaDefSpec::tmp(MockType::U32), Callee::Direct(2), vec![])?;
        block.ret(Some(value))
    })?;
    let root = builder.finish()?;
    for difference in 0..4 {
        let mut changed = MACHINE;
        match difference {
            0 => changed.ptr_size = PointerSize::Bit32,
            1 => changed.byte_order = Some(Endianness::Big),
            2 => {
                changed.float.program_flushes_subnormals = !changed.float.program_flushes_subnormals
            }
            _ => changed.division = DivisionRules::UNSTATED,
        }
        // The fixture must actually vary this machine contract.
        assert_ne!(changed, MACHINE);
        for root_mismatch in [false, true] {
            let host = MachineHost {
                callee: callee.clone(),
                root: if root_mismatch { changed } else { MACHINE },
                callee_machine: changed,
            };
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&root, &context)
                .with_host(&host)
                .with_method(0)
                .with_root_call_observations(2)
                .run();
            assert_eq!(facts.completion(), Completion::IncompatibleMachine);
            assert!(facts.constants().is_empty());
            assert!(facts.invocations().is_empty());
            assert!(facts.returned().is_none());
        }
    }
    let context = AnalysisContext::new(MACHINE);
    let host = MachineHost {
        callee,
        root: MACHINE,
        callee_machine: MACHINE,
    };
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_method(0)
        .with_root_call_observations(2)
        .run();
    assert_eq!(facts.completion(), Completion::Complete);
    assert_eq!(facts.invocations().len(), 2);
    assert_eq!(
        facts
            .returned()
            .and_then(|value| value.constant())
            .and_then(ConstValue::as_u64_bits),
        Some(42)
    );
    Ok(())
}
