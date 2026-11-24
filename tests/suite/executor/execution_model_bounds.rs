//! Selected model bounds preserve binary bytes and exclude adjacent known data.

use std::{cell::Cell, error::Error};

use analyssa::{
    PointerSize,
    analysis::{
        context::AnalysisContext,
        execution::{
            ByteBound, CallModel, ExecutionHost, LibraryModel, StaticExecution, StaticFacts,
            UseBound, Value,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        CallData, Callee, SsaDefSpec, SsaFunction, SsaFunctionBuilder,
        ops::{AccessMode, CallEffects, CallLocation, PointeeExtent, SsaOp},
        value::ConstValue,
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Binary object and immediately adjacent known bytes.
const BYTES: &[u8] = b"A\0\xffNEIGHBOR";
/// Exact binary model input extent.
const PREFIX: &[u8] = b"A\0\xff";
/// Source image address.
const SOURCE: u64 = 0x1000;
/// Writable destination image address.
const DESTINATION: u64 = 0x2000;
/// Method whose body or model is selected.
const METHOD: u32 = 5;

/// Exposes the binary source and an initially unknown writable destination.
struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }

    fn region(&self, at: u64) -> Option<ImageRegion> {
        [
            (SOURCE, Mutability::ReadOnly),
            (DESTINATION, Mutability::Writable),
        ]
        .into_iter()
        .find_map(|(start, mutability)| {
            let end = start.checked_add(u64::try_from(BYTES.len()).ok()?)?;
            (start..end).contains(&at).then_some(ImageRegion {
                start,
                end,
                mutability,
            })
        })
    }

    fn bytes(&self, at: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(at.checked_sub(SOURCE)?).ok()?;
        BYTES.get(start..start.checked_add(len)?)
    }
}

/// Counts implementation selection queries and can offer both a body and model.
struct Host {
    body: Option<SsaFunction<MockTarget>>,
    model: LibraryModel,
    bodies: Cell<usize>,
    models: Cell<usize>,
    dispatches: Cell<usize>,
}

impl Host {
    /// Creates a host with one selected library model.
    fn new(model: LibraryModel) -> Self {
        Self {
            body: None,
            model,
            bodies: Cell::new(0),
            models: Cell::new(0),
            dispatches: Cell::new(0),
        }
    }
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        if *method != METHOD {
            return None;
        }
        self.bodies.set(self.bodies.get().saturating_add(1));
        self.body.as_ref()
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        if *method == 7 {
            return Some(&LibraryModel::Copy);
        }
        if *method != METHOD {
            return None;
        }
        self.models.set(self.models.get().saturating_add(1));
        Some(&self.model)
    }

    fn dispatch(&self, method: &u32, _: &Value<MockTarget>) -> Option<u32> {
        self.dispatches.set(self.dispatches.get().saturating_add(1));
        (*method == METHOD).then_some(METHOD)
    }

    fn method_at(&self, address: u64) -> Option<u32> {
        self.dispatches.set(self.dispatches.get().saturating_add(1));
        (address == 0x6000).then_some(METHOD)
    }
}

/// Describes one call without attaching an invented effect contract.
struct Case {
    count: Option<ConstValue<MockTarget>>,
    effects: Option<CallEffects<MockTarget>>,
    kind: u8,
    tail: bool,
    seed_destination: bool,
    missing_count: bool,
    pointer_size: PointerSize,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            count: Some(ConstValue::U64(3)),
            effects: None,
            kind: 0,
            tail: false,
            seed_destination: false,
            missing_count: false,
            pointer_size: PointerSize::Bit64,
        }
    }
}

/// Builds and runs a real model call, preserving the source instruction position.
fn execute(case: Case, host: &Host) -> Result<(StaticFacts<MockTarget>, usize), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let site = if case.seed_destination { 6 } else { 4 };
    builder.in_block(0, |block| {
        let dest =
            block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(DESTINATION))?;
        let source = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(SOURCE))?;
        let count = match case.count {
            Some(value) => {
                let ty = if matches!(value, ConstValue::Float(_)) {
                    MockType::F64
                } else {
                    MockType::U64
                };
                block.const_value(SsaDefSpec::tmp(ty), value)?
            }
            None => block.load_arg(SsaDefSpec::argument(0, MockType::U64), 0)?,
        };
        let pointer = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x6000))?;
        if case.seed_destination {
            let length = block.const_value(
                SsaDefSpec::tmp(MockType::U64),
                ConstValue::U64(BYTES.len() as u64),
            )?;
            block.call_void(Callee::Direct(7), vec![dest, source, length])?;
        }
        let callee = match case.kind {
            0 => Callee::Direct(METHOD),
            1 => Callee::Virtual(METHOD),
            _ => Callee::Indirect {
                fptr: pointer,
                signature: 0,
            },
        };
        let mut args = vec![dest, source];
        if !case.missing_count {
            args.push(count);
        }
        block.emit_no_defs(SsaOp::Call(Box::new(CallData {
            dest: None,
            callee,
            args,
            effects: case.effects,
            tail: case.tail,
            implicit_inputs: Default::default(),
            results: Default::default(),
            staged: Vec::new(),
        })))?;
        block.ret(None)
    })?;
    let body = builder.finish()?;
    let mut machine = MACHINE;
    machine.ptr_size = case.pointer_size;
    let context = AnalysisContext::new(machine).with_image(&Image);
    let facts = StaticExecution::new(&body, &context).with_host(host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    Ok((facts, site))
}

/// Reads one argument's recorded bytes without confusing an omitted observation with empty data.
fn bytes(facts: &StaticFacts<MockTarget>, site: usize, argument: usize) -> Option<&[u8]> {
    facts
        .call_arguments(0, site)
        .iter()
        .find(|seen| seen.argument == argument)
        .map(|seen| seen.bytes.as_slice())
}

/// Actual direct, virtual and indirect models bound source bytes before ordinary and tail effects.
#[test]
fn selected_copy_bounds_binary_source_and_resolves_once() -> Result<(), Box<dyn Error>> {
    for kind in 0..3 {
        for tail in [false, true] {
            let host = Host::new(LibraryModel::Copy);
            let (facts, site) = execute(
                Case {
                    kind,
                    tail,
                    ..Case::default()
                },
                &host,
            )?;
            assert_eq!(bytes(&facts, site, 1), Some(PREFIX));
            let source = facts
                .call_arguments(0, site)
                .iter()
                .find(|seen| seen.argument == 1)
                .ok_or("source observation")?;
            assert_eq!(source.address.image_address(), Some(SOURCE));
            assert_eq!(
                source.bound,
                ByteBound::Use {
                    bytes: u64::try_from(PREFIX.len())?,
                    by: UseBound::Model,
                },
                "the selected model states its use"
            );
            assert_eq!(facts.image_writes(), vec![(DESTINATION, PREFIX.to_vec())]);
            assert_eq!((host.bodies.get(), host.models.get()), (1, 1));
            assert_eq!(host.dispatches.get(), usize::from(kind != 0));
        }
    }
    Ok(())
}

/// Unsupported counts retain the honest available prefix.
#[test]
fn selected_copy_unknown_large_and_malformed_counts() -> Result<(), Box<dyn Error>> {
    for (count, missing_count, expected) in [
        (None, false, Some(BYTES)),
        (Some(ConstValue::U64(u64::MAX)), false, Some(BYTES)),
        (Some(ConstValue::Float(3.0_f64.into())), false, Some(BYTES)),
        (Some(ConstValue::U64(3)), true, Some(BYTES)),
    ] {
        let host = Host::new(LibraryModel::Copy);
        let (facts, site) = execute(
            Case {
                count,
                missing_count,
                ..Case::default()
            },
            &host,
        )?;
        assert_eq!(bytes(&facts, site, 1), expected);
    }
    Ok(())
}

/// Observation and execution interpret the same native count on a 32-bit target.
#[test]
fn selected_copy_masks_count_at_pointer_width() -> Result<(), Box<dyn Error>> {
    let host = Host::new(LibraryModel::Copy);
    let (facts, site) = execute(
        Case {
            count: Some(ConstValue::U64((1_u64 << 32) + 3)),
            pointer_size: PointerSize::Bit32,
            ..Case::default()
        },
        &host,
    )?;
    assert_eq!(bytes(&facts, site, 1), Some(PREFIX));
    assert_eq!(facts.image_writes(), vec![(DESTINATION, PREFIX.to_vec())]);
    Ok(())
}

/// Fill observes original binary bytes before changing them and excludes scalar address lookalikes.
#[test]
fn selected_fill_captures_pre_effect_bytes_for_normal_and_tail_calls() -> Result<(), Box<dyn Error>>
{
    for tail in [false, true] {
        let host = Host::new(LibraryModel::Fill);
        let (facts, site) = execute(
            Case {
                seed_destination: true,
                tail,
                ..Case::default()
            },
            &host,
        )?;
        assert_eq!(bytes(&facts, site, 0), Some(PREFIX));
        assert_eq!(
            bytes(&facts, site, 1),
            None,
            "fill byte is not another pointer"
        );
        // The seeding store and the fill both belong to the root's own
        // activation: one run.
        let mut expected = BYTES.to_vec();
        expected
            .get_mut(..3)
            .ok_or("known destination prefix")?
            .fill(0);
        assert_eq!(facts.image_writes(), vec![(DESTINATION, expected)]);
    }
    Ok(())
}

/// Complete instruction use unions remain intact before intersection with model bounds.
#[test]
fn selected_copy_intersects_instruction_use_union() -> Result<(), Box<dyn Error>> {
    for extents in [[2, 6], [6, 2]] {
        let mut effects = CallEffects::pure().with_access(
            CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::InArgument(2),
            },
            AccessMode::Write,
        );
        for extent in extents {
            effects = effects.with_access(
                CallLocation::Pointee {
                    arg: 1,
                    extent: PointeeExtent::Bytes(extent),
                },
                AccessMode::Read,
            );
        }
        let host = Host::new(LibraryModel::Copy);
        let (facts, site) = execute(
            Case {
                effects: Some(effects),
                ..Case::default()
            },
            &host,
        )?;
        assert_eq!(bytes(&facts, site, 1), Some(PREFIX));
    }
    Ok(())
}

/// The actual body's read at byte eight establishes its own nine-byte prefix.
#[test]
fn actual_body_precedes_catalog_observation_bounds() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let pointer =
            block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(SOURCE + 8))?;
        let value = block.load_indirect(SsaDefSpec::tmp(MockType::U8), pointer, MockType::U8)?;
        block.ret(Some(value))
    })?;
    let mut host = Host::new(LibraryModel::Copy);
    host.body = Some(builder.finish()?);
    let (facts, site) = execute(Case::default(), &host)?;
    assert_eq!(bytes(&facts, site, 1), BYTES.get(..9));
    assert_eq!((host.bodies.get(), host.models.get()), (1, 0));
    assert!(facts.image_writes().is_empty());
    Ok(())
}

/// A model without a bound keeps prefix reporting and acquires no new purity promise.
#[test]
fn unbounded_model_keeps_prefixes_and_existing_effect_contract() -> Result<(), Box<dyn Error>> {
    let host = Host::new(LibraryModel::NoWrites);
    let (facts, site) = execute(Case::default(), &host)?;
    assert_eq!(bytes(&facts, site, 1), Some(BYTES));
    assert!(!<LibraryModel as CallModel<MockTarget>>::only_memory(
        &LibraryModel::NoWrites
    ));
    Ok(())
}

/// A zero model count does not inspect even known adjacent bytes.
#[test]
fn selected_copy_zero_count_observes_no_bytes() -> Result<(), Box<dyn Error>> {
    let host = Host::new(LibraryModel::Copy);
    let (facts, site) = execute(
        Case {
            count: Some(ConstValue::U64(0)),
            ..Case::default()
        },
        &host,
    )?;
    assert!(facts.call_arguments(0, site).is_empty());
    assert!(facts.image_writes().is_empty());
    Ok(())
}
