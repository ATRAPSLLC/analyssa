//! Store replay requires backing for every intermediate destination.

use std::{error::Error, ops::Range};

use analyssa::{
    PointerSize,
    analysis::{
        SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{CallModel, ExecutionHost, LibraryModel, StaticExecution},
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        Callee, SsaOp,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::CmpKind,
        value::ConstValue,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    passes::residual::residualize_loops,
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Writable regions with no initial contents attested.
struct Image {
    /// Whether the second destination is backed by a writable region.
    second: bool,
}

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, address: u64) -> Option<ImageRegion> {
        (address == 0x1000 || self.second && address == 0x2000).then_some(ImageRegion {
            start: address,
            end: address.saturating_add(1),
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

/// Repeats `A = 1; B = 1; A = 2`, exposing A's intermediate value if B faults.
fn stores(stack: bool, modeled: bool) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = if stack {
        let pointer = builder.function_mut().create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        builder
            .function_mut()
            .variable_mut(pointer)
            .ok_or("stack pointer")?
            .set_role(VariableRole::EntryStackPointer);
        Some(pointer)
    } else {
        None
    };
    let (first, second, one, two, zero, count) = builder.in_block(0, |block| {
        let (first, second) = if let Some(stack) = stack {
            let first = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -32,
                result_type: MockType::Ptr,
            })?;
            let second = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            (first, second)
        } else {
            let first =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
            let second =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x2000))?;
            (first, second)
        };
        let one = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
        let two = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
        let zero = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
        let count = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
        block.jump(1)?;
        Ok((first, second, one, two, zero, count))
    })?;
    let index = builder.in_block(1, |block| block.empty_phi(SsaDefSpec::tmp(MockType::U8)))?;
    let next = builder.in_block(1, |block| {
        for (address, value) in [(first, one), (second, one), (first, two)] {
            if modeled {
                block.call_void(Callee::Direct(7), vec![address, value, one])?;
            } else {
                block.store_indirect(address, value, MockType::U8)?;
            }
        }
        let next = block.add(SsaDefSpec::tmp(MockType::U8), index, one)?;
        block.branch_cmp(next, count, CmpKind::Lt, true, 1, 2)?;
        Ok(next)
    })?;
    builder.add_phi_operand(1, index, 0, zero)?;
    builder.add_phi_operand(1, index, 1, next)?;
    builder.in_block(2, |block| block.ret(None))?;
    let function = builder.finish()?;
    assert!(
        SsaVerifier::new(&function)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    Ok(function)
}

/// Absent backing withholds loop deletion while retaining conditional normal-exit facts.
#[test]
fn loop_replay_preserves_writes_before_a_possible_store_fault() -> Result<(), Box<dyn Error>> {
    for modeled in [false, true] {
        for second in [false, true] {
            let mut function = stores(false, modeled)?;
            let image = Image { second };
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            let facts = StaticExecution::new(&function, &context)
                .with_host(&Host(None))
                .run();
            assert!(facts.is_complete());
            assert_eq!(
                facts.image_writes(),
                vec![(0x1000, vec![2]), (0x2000, vec![1])]
            );
            assert_eq!(facts.loop_effect(1).is_some(), second);
            assert_eq!(
                residualize_loops(&mut function, &facts)?,
                usize::from(second)
            );
            assert!(
                SsaVerifier::new(&function)
                    .verify(VerifyLevel::Standard)
                    .is_empty()
            );
        }
    }
    Ok(())
}

/// Supplies catalog fills and an explicitly attested activation extent.
struct Host(Option<Range<i64>>);

impl ExecutionHost<MockTarget> for Host {
    fn stack_extent(
        &self,
        _body: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Range<i64>> {
        self.0.clone()
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 7).then_some(&LibraryModel::Fill as &dyn CallModel<MockTarget>)
    }
}

/// Stack replay needs the complete declared range at the target width, for direct and modeled writes.
#[test]
fn loop_replay_requires_an_attested_frame_extent() -> Result<(), Box<dyn Error>> {
    for width in [PointerSize::Bit32, PointerSize::Bit64] {
        for modeled in [false, true] {
            for (extent, expected) in [
                (None, false),
                (Some(-32..-15), true),
                (Some(-64..0), true),
                (Some(-32..-16), false),
                (Some(-31..0), false),
                (Some(-32..-32), false),
                (Some(Range { start: 0, end: -32 }), false),
                (Some(-2_147_483_649..0), width == PointerSize::Bit64),
            ] {
                let mut function = stores(true, modeled)?;
                let before = format!("{function:?}");
                let mut machine = MACHINE;
                machine.ptr_size = width;
                let context = AnalysisContext::new(machine);
                let facts = StaticExecution::new(&function, &context)
                    .with_host(&Host(extent.clone()))
                    .run();
                assert!(facts.is_complete(), "{width:?} {modeled} {extent:?}");
                assert_eq!(
                    facts.loop_effect(1).is_some(),
                    expected,
                    "{width:?} {modeled} {extent:?}"
                );
                assert_eq!(
                    residualize_loops(&mut function, &facts)?,
                    usize::from(expected)
                );
                if !expected {
                    assert_eq!(format!("{function:?}"), before);
                }
                assert!(
                    SsaVerifier::new(&function)
                        .verify(VerifyLevel::Standard)
                        .is_empty()
                );
            }
        }
    }
    Ok(())
}
