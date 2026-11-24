//! Call resolution consumes the pointer captured before a later memory write.

use std::cell::Cell;

use analyssa::{
    Result,
    analysis::{
        VerifyLevel,
        context::AnalysisContext,
        execution::{
            Address, CallSite, CallValues, CodeMemory, Completion, ExecutionHost, Root,
            StaticExecution, Value,
        },
    },
    ir::{
        CallData, CallInput, CallInputKey, Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
    },
    target::Endianness,
    testing::{self, MACHINE, MockTarget, MockType},
};

const TARGET: CallInputKey = CallInputKey::new(99);
const CELL: u32 = 0x3000;

struct Host {
    five: SsaFunction<MockTarget>,
    forty_two: SsaFunction<MockTarget>,
    resolutions: Cell<usize>,
}

impl ExecutionHost<MockTarget> for Host {
    fn method_for_call_with_inputs(
        &self,
        _: CallSite<'_, MockTarget>,
        declared: &u32,
        inputs: CallValues<'_, MockTarget>,
        memory: CodeMemory<'_>,
    ) -> Option<u32> {
        self.resolutions
            .set(self.resolutions.get().saturating_add(1));
        assert_eq!(*declared, 200);
        assert_eq!(
            memory.read_pointer(Address::new(
                Root::Symbol,
                i64::from(CELL),
                MACHINE.ptr_size
            )),
            Ok(Some(Address::new(Root::Symbol, 200, MACHINE.ptr_size)))
        );
        let target = inputs.implicit(TARGET)?.address()?.image_address()?;
        u32::try_from(target).ok()
    }

    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        match method {
            100 => Some(&self.five),
            200 => Some(&self.forty_two),
            _ => None,
        }
    }
}

fn caller(tail: bool) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let cell = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(CELL))?;
        let first = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(100))?;
        let second = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(200))?;
        block.store_indirect(cell, first, MockType::Ptr)?;
        let captured = block.load_indirect(SsaDefSpec::tmp(MockType::Ptr), cell, MockType::Ptr)?;
        block.store_indirect(cell, second, MockType::Ptr)?;
        let returned = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
            let mut call = CallData::new(Some(dest), Callee::Direct(200), vec![]);
            call.implicit_inputs.push(CallInput {
                key: TARGET,
                value: captured,
            });
            if tail {
                call.made_tail().into()
            } else {
                call.into()
            }
        })?;
        block.ret(Some(returned))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

#[test]
fn captured_call_pointer_survives_a_cell_write_before_normal_and_tail_entry() -> Result<()> {
    for tail in [false, true] {
        let host = Host {
            five: testing::const_i32_return(5),
            forty_two: testing::const_i32_return(42),
            resolutions: Cell::new(0),
        };
        let caller = caller(tail)?;
        let mut machine = MACHINE;
        machine.byte_order = Some(Endianness::Little);
        let context = AnalysisContext::new(machine);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert_eq!(facts.completion(), Completion::Complete, "tail={tail}");
        assert_eq!(
            facts
                .returned()
                .and_then(Value::constant)
                .and_then(ConstValue::as_i64),
            Some(5),
            "tail={tail}"
        );
        assert_eq!(
            host.resolutions.get(),
            1,
            "no repeated call resolution or prefix execution"
        );
    }
    Ok(())
}
