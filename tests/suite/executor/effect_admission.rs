//! Missing write footprints are admitted only by the configured execution policy.

use std::{error::Error, num::NonZeroU16};

use analyssa::{
    PointerSize,
    analysis::{
        Machine, SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{
            Address, CallModel, Completion, ExecutionHost, InstructionSources, LibraryModel,
            MemoryRefusal, MemoryWriteGuard, ModelCall, Root, StaticExecution, Value,
        },
    },
    ir::{
        Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        lanes::VectorConst,
        ops::{AccessQualifiers, FenceKind, SsaOp, VectorPackKind, VectorSegmentLayout},
        value::ConstValue,
        variable::{DefSite, VariableOrigin},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

#[derive(Debug)]
struct Guard {
    reject: bool,
    reason: MemoryRefusal,
}

impl MemoryWriteGuard for Guard {
    fn check_write(
        &self,
        _: &mut InstructionSources,
        _: Address,
        _: usize,
    ) -> Result<(), MemoryRefusal> {
        Ok(())
    }

    fn check_unresolved_write(
        &self,
        _: Option<Address>,
        _: Option<usize>,
    ) -> Result<(), MemoryRefusal> {
        if self.reject {
            Err(self.reason)
        } else {
            Ok(())
        }
    }
}

struct OpaqueModel;

impl CallModel<MockTarget> for OpaqueModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.clobber();
    }
}

struct BoundedModel;

impl CallModel<MockTarget> for BoundedModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.write_unknown(Address::new(Root::Symbol, 0x5000, PointerSize::Bit64), 4);
    }
}

struct NumericModel;

impl CallModel<MockTarget> for NumericModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.write_unknown(Address::new(Root::Number, 0x5000, PointerSize::Bit64), 4);
    }
}

struct PrivateModel(bool);

impl CallModel<MockTarget> for PrivateModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let address = if self.0 {
            call.allocate_block(Some(4), false, None)
        } else {
            Address::new(Root::Stack, -4, PointerSize::Bit64)
        };
        call.write_unknown(address, 4);
    }
}

struct Host(Guard);

impl ExecutionHost<MockTarget> for Host {
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        Some(&self.0)
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match method {
            10 => Some(&LibraryModel::Copy),
            11 => Some(&LibraryModel::Fill),
            12 => Some(&OpaqueModel),
            13 => Some(&BoundedModel),
            14 => Some(&NumericModel),
            15 => Some(&PrivateModel(false)),
            16 => Some(&PrivateModel(true)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Trial {
    Store {
        unknown_destination: bool,
    },
    Copy {
        unknown_destination: bool,
        length: Option<u64>,
        native: bool,
    },
    Fill,
    OpaqueModel,
    BoundedModel,
    NumericModel,
    PrivateModel(bool),
    Fence,
    Vector {
        form: VectorForm,
        unknown_destination: bool,
        empty: bool,
    },
    QualifiedCopy {
        unknown_direction: bool,
    },
}

#[derive(Clone, Copy, Debug)]
enum VectorForm {
    Plain,
    Masked,
    Scatter,
    Packed,
    Segmented,
}

fn program(trial: Trial) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(6, 0);
    let mut input = |index, ty| {
        builder.function_mut().create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            ty,
        )
    };
    let source = input(0, MockType::Ptr);
    let unknown_destination = input(1, MockType::Ptr);
    let unknown_length = input(2, MockType::U64);
    let payload = input(3, MockType::U8);
    let vector = input(4, MockType::V4I32);
    let direction = input(5, MockType::I1);
    let mask_empty = matches!(trial, Trial::Vector { empty: true, .. });
    let predicate = VectorConst::from_lanes(1, [u128::from(!mask_empty); 4]).ok_or("predicate")?;
    builder.in_block(0, |block| {
        let destination =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x5000))?;
        let empty = matches!(
            trial,
            Trial::Copy {
                length: Some(0),
                ..
            } | Trial::Vector { empty: true, .. }
        );
        let saved = if empty {
            let address =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x9000))?;
            let value = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
            block.store_indirect(address, value, MockType::U32)?;
            Some(address)
        } else {
            None
        };

        match trial {
            Trial::Store {
                unknown_destination: unknown,
            } => {
                block.store_indirect(
                    if unknown {
                        unknown_destination
                    } else {
                        destination
                    },
                    payload,
                    MockType::U8,
                )?;
            }
            Trial::Copy {
                unknown_destination: unknown,
                length,
                native,
            } => {
                let destination = if unknown {
                    unknown_destination
                } else {
                    destination
                };
                let size = match length {
                    Some(length) => block
                        .const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(length))?,
                    None => unknown_length,
                };
                if native {
                    block.emit_no_defs(SsaOp::MoveBlk {
                        dest_addr: destination,
                        src_addr: source,
                        size,
                        src_access: AccessQualifiers::default(),
                        dest_access: AccessQualifiers::default(),
                    })?;
                } else {
                    block.call_void(Callee::Direct(10), vec![destination, source, size])?;
                }
            }
            Trial::Fill => {
                let size = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(4))?;
                block.call_void(Callee::Direct(11), vec![destination, payload, size])?;
            }
            Trial::OpaqueModel => {
                block.call_void(Callee::Direct(12), vec![])?;
            }
            Trial::BoundedModel => {
                block.call_void(Callee::Direct(13), vec![])?;
            }
            Trial::NumericModel => {
                block.call_void(Callee::Direct(14), vec![])?;
            }
            Trial::PrivateModel(heap) => {
                block.call_void(Callee::Direct(if heap { 16 } else { 15 }), vec![])?;
            }
            Trial::Fence => {
                block.emit_no_defs(SsaOp::Fence {
                    kind: FenceKind::Full,
                })?;
            }
            Trial::Vector {
                form,
                unknown_destination: unknown,
                empty: _,
            } => {
                let address = if unknown {
                    unknown_destination
                } else {
                    destination
                };
                let mask = block.const_value(
                    SsaDefSpec::tmp(MockType::Mask4),
                    ConstValue::Vector(predicate.clone()),
                )?;
                let vector_type = MockType::V4I32;
                let access = AccessQualifiers::default();
                let operation = match form {
                    VectorForm::Plain => SsaOp::VectorStore {
                        addr: address,
                        value: vector,
                        vector_type,
                        access,
                    },
                    VectorForm::Masked => SsaOp::VectorMaskedStore {
                        addr: address,
                        value: vector,
                        mask,
                        vector_type,
                        access,
                    },
                    VectorForm::Scatter => SsaOp::VectorScatter {
                        base: address,
                        indices: vector,
                        value: vector,
                        mask,
                        vector_type,
                        access,
                    },
                    VectorForm::Packed => SsaOp::VectorPackStore {
                        addr: address,
                        value: vector,
                        mask,
                        vector_type,
                        element_bits: 32,
                        kind: VectorPackKind::Compress,
                        access,
                    },
                    VectorForm::Segmented => SsaOp::VectorSegmentStore {
                        base: address,
                        values: vec![vector, vector],
                        vector_type,
                        mask: Some(mask),
                        segments: 2,
                        layout: VectorSegmentLayout::Interleaved,
                        access,
                    },
                };
                block.emit_no_defs(operation)?;
            }
            Trial::QualifiedCopy { unknown_direction } => {
                let size = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(4))?;
                block.emit_no_defs(SsaOp::CopyBlk {
                    dest_addr: destination,
                    src_addr: source,
                    size,
                    element_bytes: 1,
                    backward: unknown_direction.then_some(direction),
                    src_access: AccessQualifiers {
                        address_space: NonZeroU16::new(1),
                        ..AccessQualifiers::default()
                    },
                    dest_access: AccessQualifiers::default(),
                })?;
            }
        }
        let marker = match saved {
            Some(address) => {
                block.load_indirect(SsaDefSpec::tmp(MockType::U32), address, MockType::U32)?
            }
            None => block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?,
        };
        block.ret(Some(marker))
    })?;
    let function = builder.finish()?;
    let errors = SsaVerifier::new(&function).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{trial:?}: {errors:?}");
    Ok(function)
}

fn check(trial: Trial, must_refuse: bool) -> Result<(), Box<dyn Error>> {
    let function = program(trial)?;
    for (audit, order) in [false, true]
        .into_iter()
        .flat_map(|audit| [Endianness::Little, Endianness::Big].map(move |order| (audit, order)))
    {
        let context = AnalysisContext::new(Machine {
            byte_order: Some(order),
            ..MACHINE
        });
        let host = Host(Guard {
            reject: audit,
            reason: MemoryRefusal::UnresolvedWrite,
        });
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .run();
        if audit && must_refuse {
            assert!(
                matches!(
                    facts.completion(),
                    Completion::Memory {
                        reason: MemoryRefusal::UnresolvedWrite,
                        ..
                    }
                ),
                "{trial:?}: {:?}",
                facts.completion()
            );
            assert!(facts.returned().is_none());
            assert!(facts.constants().is_empty());
        } else {
            assert_eq!(
                facts.completion(),
                Completion::Complete,
                "{trial:?}/audit={audit}"
            );
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&ConstValue::U32(42)),
                "{trial:?}/audit={audit}/order={order:?}",
            );
        }
    }
    Ok(())
}

#[test]
fn unknown_scalar_destination_refuses() -> Result<(), Box<dyn Error>> {
    check(
        Trial::Store {
            unknown_destination: true,
        },
        true,
    )
}
#[test]
fn known_scalar_destination_unknown_payload_passes() -> Result<(), Box<dyn Error>> {
    check(
        Trial::Store {
            unknown_destination: false,
        },
        false,
    )
}
#[test]
fn unknown_copy_destination_refuses() -> Result<(), Box<dyn Error>> {
    for native in [false, true] {
        check(
            Trial::Copy {
                unknown_destination: true,
                length: Some(4),
                native,
            },
            true,
        )?;
    }
    Ok(())
}
#[test]
fn unknown_copy_extent_refuses() -> Result<(), Box<dyn Error>> {
    for native in [false, true] {
        check(
            Trial::Copy {
                unknown_destination: false,
                length: None,
                native,
            },
            true,
        )?;
    }
    Ok(())
}
#[test]
fn known_copy_destination_unknown_source_passes() -> Result<(), Box<dyn Error>> {
    for native in [false, true] {
        check(
            Trial::Copy {
                unknown_destination: false,
                length: Some(4),
                native,
            },
            false,
        )?;
    }
    Ok(())
}
#[test]
fn zero_copy_extent_with_unknown_addresses_passes() -> Result<(), Box<dyn Error>> {
    for native in [false, true] {
        check(
            Trial::Copy {
                unknown_destination: true,
                length: Some(0),
                native,
            },
            false,
        )?;
    }
    Ok(())
}
#[test]
fn known_fill_destination_unknown_byte_passes() -> Result<(), Box<dyn Error>> {
    check(Trial::Fill, false)
}
#[test]
fn model_clobber_refuses() -> Result<(), Box<dyn Error>> {
    check(Trial::OpaqueModel, true)
}
#[test]
fn model_known_unknown_write_passes() -> Result<(), Box<dyn Error>> {
    check(Trial::BoundedModel, false)
}
#[test]
fn fence_cache_clobber_is_not_unknown_write() -> Result<(), Box<dyn Error>> {
    check(Trial::Fence, false)
}

#[test]
fn unknown_vector_destinations_refuse_but_empty_masks_preserve_memory() -> Result<(), Box<dyn Error>>
{
    for form in [
        VectorForm::Plain,
        VectorForm::Masked,
        VectorForm::Scatter,
        VectorForm::Packed,
        VectorForm::Segmented,
    ] {
        check(
            Trial::Vector {
                form,
                unknown_destination: true,
                empty: false,
            },
            true,
        )?;
        if !matches!(form, VectorForm::Plain) {
            check(
                Trial::Vector {
                    form,
                    unknown_destination: true,
                    empty: true,
                },
                false,
            )?;
        }
    }
    Ok(())
}

#[test]
fn checked_vector_footprints_allow_unknown_payloads() -> Result<(), Box<dyn Error>> {
    for form in [
        VectorForm::Plain,
        VectorForm::Masked,
        VectorForm::Packed,
        VectorForm::Segmented,
    ] {
        check(
            Trial::Vector {
                form,
                unknown_destination: false,
                empty: false,
            },
            false,
        )?;
    }
    check(
        Trial::Vector {
            form: VectorForm::Scatter,
            unknown_destination: false,
            empty: false,
        },
        true,
    )?;
    Ok(())
}

#[test]
fn checked_block_destinations_survive_opaque_sources_and_directions() -> Result<(), Box<dyn Error>>
{
    for unknown_direction in [false, true] {
        check(Trial::QualifiedCopy { unknown_direction }, false)?;
    }
    Ok(())
}

#[test]
fn unplaced_numeric_writes_require_the_unresolved_policy() -> Result<(), Box<dyn Error>> {
    check(Trial::NumericModel, true)
}

#[test]
fn concrete_private_stack_and_fresh_heap_footprints_remain_covered() -> Result<(), Box<dyn Error>> {
    for heap in [false, true] {
        check(Trial::PrivateModel(heap), false)?;
    }
    Ok(())
}

#[test]
fn guard_refusals_keep_their_original_reason() -> Result<(), Box<dyn Error>> {
    for trial in [
        Trial::Store {
            unknown_destination: true,
        },
        Trial::Copy {
            unknown_destination: true,
            length: Some(4),
            native: true,
        },
        Trial::Vector {
            form: VectorForm::Scatter,
            unknown_destination: false,
            empty: false,
        },
    ] {
        let function = program(trial)?;
        let context = AnalysisContext::new(MACHINE);
        let host = Host(Guard {
            reject: true,
            reason: MemoryRefusal::ReadOnlyImage,
        });
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .run();
        assert!(
            matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::ReadOnlyImage,
                    ..
                }
            ),
            "{trial:?}: {:?}",
            facts.completion()
        );
        assert!(facts.returned().is_none());
    }
    Ok(())
}
