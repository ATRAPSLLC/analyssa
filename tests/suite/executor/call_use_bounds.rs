//! Same-execution consumer footprints bound pre-call byte observations.

use analyssa::{
    Error, Result,
    analysis::{
        VerifyLevel,
        context::AnalysisContext,
        execution::{CallSite, ExecutionHost, ParameterBindings, StaticExecution, Value},
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
        variable::{DefSite, VariableOrigin},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        (0x2000..0x2008).contains(&address).then_some(ImageRegion {
            start: 0x2000,
            end: 0x2008,
            mutability: Mutability::Writable,
        })
    }

    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}

struct Host {
    callee: SsaFunction<MockTarget>,
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 1).then_some(&self.callee)
    }

    fn parameters(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        body: &SsaFunction<MockTarget>,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        for variable in body.variables() {
            if variable.origin() == VariableOrigin::Argument(0) {
                bindings.bind(variable.id(), 0);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Consumer {
    Four,
    Alternatives,
    UnknownRead,
    UnknownCall,
    Overwrite,
}

fn consumer(kind: Consumer) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::new(2, 0);
    let pointer = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    let other = builder.function_mut().create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        if matches!(kind, Consumer::Alternatives) {
            MockType::Bool
        } else {
            MockType::Ptr
        },
    );
    if matches!(kind, Consumer::Alternatives) {
        builder.in_block(0, |block| block.branch(other, 1, 2))?;
        for (block, ty) in [(1, MockType::U32), (2, MockType::U64)] {
            builder.in_block(block, |block| {
                block.load_indirect(SsaDefSpec::tmp(ty), pointer, ty)?;
                let result =
                    block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
                block.ret(Some(result))
            })?;
        }
    } else {
        builder.in_block(0, |block| {
            block.load_indirect(SsaDefSpec::tmp(MockType::U32), pointer, MockType::U32)?;
            match kind {
                Consumer::UnknownRead => {
                    block.load_indirect(SsaDefSpec::tmp(MockType::U8), other, MockType::U8)?;
                }
                Consumer::UnknownCall => {
                    block.call_void(Callee::Direct(99), vec![pointer])?;
                }
                Consumer::Overwrite => {
                    let byte =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(9))?;
                    let length =
                        block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(4))?;
                    block.init_blk(pointer, byte, length, None)?;
                }
                Consumer::Four | Consumer::Alternatives => {}
            }
            let result = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
            block.ret(Some(result))
        })?;
    }
    builder.finish_verified(VerifyLevel::Full)
}

fn caller() -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let pointer = block.const_value(
            SsaDefSpec::tmp(MockType::Ptr),
            ConstValue::NativeUInt(0x2000),
        )?;
        let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
        let length = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(8))?;
        block.init_blk(pointer, byte, length, None)?;
        let result = block.call(
            SsaDefSpec::tmp(MockType::U32),
            Callee::Direct(1),
            vec![pointer],
        )?;
        block.ret(Some(result))
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

#[test]
fn bounded_consumers_keep_the_pre_call_snapshot_and_actual_return() -> Result<()> {
    for kind in [Consumer::Four, Consumer::Overwrite] {
        let root = caller()?;
        let host = Host {
            callee: consumer(kind)?,
        };
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_transitive_call_observations(4)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U32(42))));
        let calls = facts.calls();
        let observed = calls
            .first()
            .and_then(|(_, arguments)| arguments.first())
            .ok_or_else(|| Error::new("actual pointer argument observation"))?;
        assert_eq!(
            observed.bytes, [7; 4],
            "{kind:?}: later writes must not replace the pre-call bytes"
        );
    }
    Ok(())
}

#[test]
fn alternatives_union_their_accesses_and_unknown_effects_supply_no_cap() -> Result<()> {
    for kind in [
        Consumer::Alternatives,
        Consumer::UnknownRead,
        Consumer::UnknownCall,
    ] {
        let root = caller()?;
        let host = Host {
            callee: consumer(kind)?,
        };
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_transitive_call_observations(4)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U32(42))));
        let calls = facts.calls();
        let observed = calls
            .first()
            .and_then(|(_, arguments)| arguments.first())
            .ok_or_else(|| Error::new("uncapped pointer argument observation"))?;
        assert_eq!(
            observed.bytes, [7; 8],
            "a partial use cannot establish a shorter complete extent"
        );
    }
    Ok(())
}
