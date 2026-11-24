//! Vector write footprints retain permissions, selected lanes and source history.

use std::{error::Error, ops::Range};

use analyssa::{
    PointerSize,
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            Address, Completion, ExecutionHost, FindingKind, InstructionSourceId,
            InstructionSources, MemoryRefusal, MemoryWriteGuard, StaticExecution,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        lanes::VectorConst,
        ops::{AccessQualifiers, SsaOp, VectorPackKind, VectorSegmentLayout},
        value::ConstValue,
        variable::{DefSite, VariableOrigin},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

struct Image(Option<Range<u64>>);

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, at: u64) -> Option<ImageRegion> {
        if let Some(range) = self.0.as_ref().filter(|range| range.contains(&at)) {
            Some(ImageRegion {
                start: range.start,
                end: range.end,
                mutability: Mutability::ReadOnly,
            })
        } else {
            Some(ImageRegion {
                start: 0,
                end: u64::MAX,
                mutability: Mutability::Writable,
            })
        }
    }
    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }
}

#[derive(Debug)]
struct Guard(Range<u64>);

impl MemoryWriteGuard for Guard {
    fn check_write(
        &self,
        sources: &mut InstructionSources,
        address: Address,
        len: usize,
    ) -> Result<(), MemoryRefusal> {
        let Some(at) = address.image_address() else {
            return Ok(());
        };
        let prefix = address.linear_prefix_len(len);
        let end =
            at.saturating_add(u64::try_from(prefix).map_err(|_| MemoryRefusal::AddressRange)?);
        let suffix =
            u64::try_from(len.saturating_sub(prefix)).map_err(|_| MemoryRefusal::AddressRange)?;
        if at < self.0.end && self.0.start < end || self.0.start < suffix {
            sources.record_write(InstructionSourceId::new(1))?;
        }
        Ok(())
    }
}

struct Host(Option<Guard>);

impl ExecutionHost<MockTarget> for Host {
    fn admit_code_source(
        &self,
        _: &SsaFunction<MockTarget>,
        _: Option<&u32>,
        sources: &mut InstructionSources,
    ) -> bool {
        sources.enter(InstructionSourceId::new(1))
    }
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        self.0.as_ref().map(|guard| guard as &dyn MemoryWriteGuard)
    }
}

#[derive(Debug, Clone, Copy)]
enum Form {
    Plain,
    Masked,
    Scatter,
    Packed,
    Interleaved,
    Consecutive,
}

fn function(
    form: Form,
    mask: Option<u64>,
    indices: [u32; 4],
    base: Option<u64>,
    preserve: bool,
) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let base = base.map(u32::try_from).transpose()?;
    let mask = mask
        .map(|mask| {
            VectorConst::from_lanes(
                1,
                (0..4).map(|lane| u128::from(mask.checked_shr(lane).unwrap_or(0) & 1)),
            )
            .ok_or("predicate")
        })
        .transpose()?;
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
    let value = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::V4I32,
    );
    let unknown_mask = builder.function_mut().create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::Mask4,
    );
    let unknown_base = builder.function_mut().create_variable(
        VariableOrigin::Argument(2),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    let offsets = VectorConst::from_lanes(32, indices.map(u128::from)).ok_or("index vector")?;
    builder.in_block(0, |block| {
        let saved = if preserve {
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x8000))?;
            let seven = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            block.store_indirect(address, seven, MockType::U8)?;
            Some(address)
        } else {
            None
        };
        let addr = match base {
            Some(base) => {
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(base))?
            }
            None => unknown_base,
        };
        let mask = match mask {
            Some(mask) => {
                block.const_value(SsaDefSpec::tmp(MockType::Mask4), ConstValue::Vector(mask))?
            }
            None => unknown_mask,
        };
        let access = AccessQualifiers::default();
        let vector_type = MockType::V4I32;
        let op = match form {
            Form::Plain => SsaOp::VectorStore {
                addr,
                value,
                vector_type,
                access,
            },
            Form::Masked => SsaOp::VectorMaskedStore {
                addr,
                value,
                mask,
                vector_type,
                access,
            },
            Form::Scatter => {
                let indices = block.const_value(
                    SsaDefSpec::tmp(MockType::V4I32),
                    ConstValue::Vector(offsets),
                )?;
                SsaOp::VectorScatter {
                    base: addr,
                    indices,
                    value,
                    mask,
                    vector_type,
                    access,
                }
            }
            Form::Packed => SsaOp::VectorPackStore {
                addr,
                value,
                mask,
                vector_type,
                element_bits: 32,
                kind: VectorPackKind::Compress,
                access,
            },
            Form::Interleaved | Form::Consecutive => SsaOp::VectorSegmentStore {
                base: addr,
                values: vec![value, value],
                mask: Some(mask),
                vector_type,
                segments: 2,
                layout: if matches!(form, Form::Interleaved) {
                    VectorSegmentLayout::Interleaved
                } else {
                    VectorSegmentLayout::Consecutive
                },
                access,
            },
        };
        block.emit_no_defs(op)?;
        let result = if let Some(saved) = saved {
            block.load_indirect(SsaDefSpec::tmp(MockType::U8), saved, MockType::U8)?
        } else {
            block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?
        };
        block.ret(Some(result))
    })?;
    let function = builder.finish()?;
    let errors = SsaVerifier::new(&function).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{form:?}: {errors:?}");
    Ok(function)
}

fn check(
    form: Form,
    mask: Option<u64>,
    indices: [u32; 4],
    protected: Range<u64>,
    refused: bool,
) -> Result<(), Box<dyn Error>> {
    let function = function(form, mask, indices, Some(0x7000), false)?;
    for guarded in [false, true] {
        let image = Image((!guarded).then(|| protected.clone()));
        let host = Host(guarded.then(|| Guard(protected.clone())));
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .run();
        if refused && !guarded && mask.is_some() {
            // A lane the run knows is written is a store the machine makes,
            // and faults where the image may not be written: the path ends
            // there, found at the store.
            assert!(
                facts.is_complete(),
                "{form:?}, {mask:?}: {:?}",
                facts.completion()
            );
            assert!(facts.returned().is_none());
            assert!(
                facts
                    .findings()
                    .iter()
                    .any(|finding| finding.kind == FindingKind::ReadOnlyWrite),
                "{form:?}, {mask:?}"
            );
        } else if refused {
            // The guard refuses the write; a lane an unknown mask may leave
            // alone is no store the machine certainly makes, and refuses.
            let reason = if guarded {
                MemoryRefusal::InstructionSource
            } else {
                MemoryRefusal::ReadOnlyImage
            };
            assert!(
                matches!(facts.completion(), Completion::Memory { reason: actual, .. } if actual == reason),
                "{form:?}, {mask:?}, guard={guarded}: {:?}",
                facts.completion()
            );
            assert!(facts.returned().is_none());
            assert!(facts.invocations().is_empty());
        } else {
            assert!(facts.is_complete(), "{form:?}: {:?}", facts.completion());
            assert_eq!(
                facts
                    .returned()
                    .and_then(|value| value.constant())
                    .and_then(ConstValue::as_u64_bits),
                Some(42)
            );
        }
    }
    Ok(())
}

#[test]
fn full_vector_stores_enforce_source_and_readonly_contracts() -> Result<(), Box<dyn Error>> {
    check(Form::Plain, Some(15), [0, 4, 8, 12], 0x7000..0x7010, true)?;
    check(Form::Plain, Some(15), [0, 4, 8, 12], 0x7100..0x7104, false)
}

#[test]
fn predicated_vector_writes_check_only_selected_destinations() -> Result<(), Box<dyn Error>> {
    for (form, mask, indices, protected, refused) in [
        (Form::Masked, 1, [0, 4, 8, 12], 0x7000..0x7004, true),
        (Form::Masked, 8, [0, 4, 8, 12], 0x7000..0x7004, false),
        (Form::Packed, 8, [0, 4, 8, 12], 0x7000..0x7004, true),
        (Form::Scatter, 1, [12, 8, 4, 0], 0x7000..0x7004, false),
        (Form::Scatter, 8, [12, 8, 4, 0], 0x7000..0x7004, true),
        (Form::Interleaved, 1, [0, 4, 8, 12], 0x7004..0x7008, true),
        (Form::Consecutive, 1, [0, 4, 8, 12], 0x7004..0x7008, false),
        (Form::Consecutive, 2, [0, 4, 8, 12], 0x7004..0x7008, true),
    ] {
        check(form, Some(mask), indices, protected, refused)?;
    }
    Ok(())
}

#[test]
fn empty_predicates_preserve_memory_with_unknown_destinations() -> Result<(), Box<dyn Error>> {
    for form in [
        Form::Masked,
        Form::Scatter,
        Form::Packed,
        Form::Interleaved,
        Form::Consecutive,
    ] {
        for base in [Some(0x7000), None] {
            let function = function(form, Some(0), [0, 4, 8, 12], base, true)?;
            let image = Image(Some(0x7000..0x7100));
            let host = Host(Some(Guard(0x7000..0x7100)));
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            let facts = StaticExecution::new(&function, &context)
                .with_host(&host)
                .run();
            assert!(facts.is_complete(), "{form:?}: {:?}", facts.completion());
            assert_eq!(
                facts
                    .returned()
                    .and_then(|value| value.constant())
                    .and_then(ConstValue::as_u64_bits),
                Some(7)
            );
        }
    }
    Ok(())
}

#[test]
fn unknown_predicates_preserve_possible_known_write_obligations() -> Result<(), Box<dyn Error>> {
    for form in [
        Form::Masked,
        Form::Scatter,
        Form::Packed,
        Form::Interleaved,
        Form::Consecutive,
    ] {
        check(form, None, [0, 4, 8, 12], 0x7000..0x7004, true)?;
    }
    Ok(())
}

#[test]
fn vector_write_limits_precede_admission_and_havoc() -> Result<(), Box<dyn Error>> {
    for form in [
        Form::Plain,
        Form::Masked,
        Form::Scatter,
        Form::Packed,
        Form::Interleaved,
        Form::Consecutive,
    ] {
        let function = function(form, Some(1), [0, 4, 8, 12], Some(0x7000), false)?;
        let image = Image(Some(0x7000..0x7100));
        let host = Host(Some(Guard(0x7000..0x7100)));
        let context = AnalysisContext::new(MACHINE)
            .with_image(&image)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, 4));
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .run();
        assert!(
            matches!(facts.completion(), Completion::Exhausted(exhaustion) if exhaustion.budget == Budget::ExecutionBufferBytes),
            "{form:?}: {:?}",
            facts.completion()
        );
        assert!(facts.returned().is_none());
    }
    Ok(())
}

#[test]
fn scatter_offsets_are_signed_and_wrap_at_the_target_width() -> Result<(), Box<dyn Error>> {
    for (base, indices, pointer_size, protected) in [
        (
            0x7000,
            [0xffff_fffc, 4, 8, 12],
            PointerSize::Bit64,
            0x6ffc..0x7000,
        ),
        (0xffff_fffc, [8, 4, 8, 12], PointerSize::Bit32, 4..8),
    ] {
        let function = function(Form::Scatter, Some(1), indices, Some(base), false)?;
        let host = Host(Some(Guard(protected)));
        let mut machine = MACHINE;
        machine.ptr_size = pointer_size;
        let context = AnalysisContext::new(machine);
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .run();
        assert!(
            matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::InstructionSource,
                    ..
                }
            ),
            "{:?}",
            facts.completion()
        );
        assert!(facts.returned().is_none());
    }
    Ok(())
}
