//! Static execution's resource accounting and reentry: a library model costs
//! the run what the loop it stands for would, and a body entered again while
//! it is already on the stack runs as a fresh activation, as deep as the
//! configured depth allows.

use std::process;

use analyssa::{
    Result,
    analysis::{
        budget::{Budget, Exhaustion, WorkBudget},
        context::AnalysisContext,
        execution::{
            CallModel, Completion, ExecutionHost, LibraryModel, StaticExecution, StaticFacts,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        Callee,
        function::{SsaBlockBuilder, SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{CmpKind, SsaOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Read-only text: [`TEXT_LEN`] letters and a terminator.
const TEXT_AT: u64 = 0x1000;

/// The letters before the terminator at [`TEXT_AT`].
const TEXT_LEN: usize = 64;

/// The read-only text.
const TEXT: [u8; TEXT_LEN + 1] = {
    let mut text = [b'A'; TEXT_LEN + 1];
    text[TEXT_LEN] = 0;
    text
};

/// Writable data: a countdown counter, then the slots each activation fills.
const DATA_AT: u64 = 0x4000;

/// Where each activation of the countdown writes its own count.
const SLOTS_AT: u64 = DATA_AT + 0x100;

/// Where a copy lands.
const COPY_AT: u64 = DATA_AT + 0x180;

/// The count the countdown starts from.
const DEPTH: u8 = 5;

/// The writable data's length. Its bytes before the run are unknown: a root
/// may run after other code wrote them.
const DATA_LEN: u64 = 0x200;

/// The method number a model call names.
const MODELLED: u32 = 17;

/// The countdown's method number.
const COUNTDOWN: u32 = 9;

/// The second countdown's method number, for mutual recursion.
const PARTNER: u32 = 10;

/// [`TEXT`] read-only at [`TEXT_AT`], [`DATA_LEN`] writable bytes at [`DATA_AT`].
struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        [
            (TEXT_AT, TEXT.len() as u64, Mutability::ReadOnly),
            (DATA_AT, DATA_LEN, Mutability::Writable),
        ]
        .into_iter()
        .map(|(start, len, mutability)| (start, start.wrapping_add(len), mutability))
        .find(|(start, end, _)| (*start..*end).contains(&address))
        .map(|(start, end, mutability)| ImageRegion {
            start,
            end,
            mutability,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let from = usize::try_from(address.checked_sub(TEXT_AT)?).ok()?;
        TEXT.get(from..from.checked_add(len)?)
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}

/// Ends the test: a fixture the builder rejects is a broken test.
fn abort<E, R>(_: E) -> R {
    process::abort()
}

fn tmp(ty: MockType) -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(ty)
}

/// Runs `ssa` under `host` with `budget`.
fn run(
    ssa: &SsaFunction<MockTarget>,
    host: &dyn ExecutionHost<MockTarget>,
    budget: WorkBudget,
) -> StaticFacts<MockTarget> {
    let image = Image;
    let context = AnalysisContext::new(MACHINE)
        .with_image(&image)
        .with_budget(budget);
    StaticExecution::new(ssa, &context).with_host(host).run()
}

/// A host that models every call with one model.
struct ModelHost(LibraryModel);

impl ExecutionHost<MockTarget> for ModelHost {
    fn model(&self, _: &u32) -> Option<&dyn CallModel<MockTarget>> {
        Some(&self.0)
    }
}

/// Calls the modelled function on `arguments`, built in the entry block,
/// then returns.
fn calling(
    arguments: impl FnOnce(&mut SsaBlockBuilder<'_, MockTarget>) -> Result<Vec<SsaVarId>>,
) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let arguments = arguments(b)?;
            b.call(tmp(MockType::U64), Callee::Direct(MODELLED), arguments)?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// `memcpy(COPY_AT, TEXT_AT, len)`.
fn copy(len: i64) -> SsaFunction<MockTarget> {
    calling(|b| {
        let dest = b.const_value(tmp(MockType::U64), ConstValue::U64(COPY_AT))?;
        let source = b.const_value(tmp(MockType::U64), ConstValue::U64(TEXT_AT))?;
        let len = b.const_value(tmp(MockType::I64), ConstValue::I64(len))?;
        Ok(vec![dest, source, len])
    })
}

/// The empty string at the end of [`TEXT`].
const EMPTY_AT: u64 = TEXT_AT + TEXT_LEN as u64;

/// `strlen(string)`.
fn length(string: u64) -> SsaFunction<MockTarget> {
    calling(|b| {
        let string = b.const_value(tmp(MockType::U64), ConstValue::U64(string))?;
        Ok(vec![string])
    })
}

/// A model's copy costs a step per byte it copies, and a run that cannot pay
/// for it ends exhausted rather than copying for free.
#[test]
fn a_model_charges_the_run_for_each_byte_it_copies() {
    let host = ModelHost(LibraryModel::Copy);
    let short = run(&copy(1), &host, WorkBudget::DEFAULT);
    let long = run(&copy(TEXT_LEN as i64), &host, WorkBudget::DEFAULT);
    assert!(short.is_complete(), "{:?}", short.completion());
    assert!(long.is_complete(), "{:?}", long.completion());
    assert_eq!(
        long.image_writes(),
        vec![(COPY_AT, vec![b'A'; TEXT_LEN])],
        "the copy ran"
    );
    assert_eq!(
        long.usage().steps.saturating_sub(short.usage().steps),
        TEXT_LEN.saturating_sub(1)
    );

    let limit = short.usage().steps.saturating_add(TEXT_LEN / 2);
    let starved = run(
        &copy(TEXT_LEN as i64),
        &host,
        WorkBudget::DEFAULT.with_limit(Budget::ExecutionSteps, limit),
    );
    assert!(
        matches!(
            starved.completion(),
            Completion::Exhausted(Exhaustion {
                budget: Budget::ExecutionSteps,
                ..
            })
        ),
        "{:?}",
        starved.completion()
    );
    assert!(starved.image_writes().is_empty());
}

/// A string scan costs the bytes it inspected, whether it found the
/// terminator or stopped at the string limit.
#[test]
fn a_model_charges_the_run_for_each_byte_it_scans() {
    let host = ModelHost(LibraryModel::Length);
    let empty = run(&length(EMPTY_AT), &host, WorkBudget::DEFAULT);
    let whole = run(&length(TEXT_AT), &host, WorkBudget::DEFAULT);
    assert_eq!(
        whole.usage().steps.saturating_sub(empty.usage().steps),
        TEXT_LEN
    );

    let limit = 16;
    let bounded = WorkBudget::DEFAULT.with_limit(Budget::ExecutionStringBytes, limit);
    let empty = run(&length(EMPTY_AT), &host, bounded);
    let cut = run(&length(TEXT_AT), &host, bounded);
    assert!(cut.is_complete(), "{:?}", cut.completion());
    assert_eq!(
        cut.usage().steps.saturating_sub(empty.usage().steps),
        limit.saturating_sub(1)
    );
}

/// The counter's address, as a pointer.
fn counter(b: &mut SsaBlockBuilder<'_, MockTarget>) -> Result<SsaVarId> {
    let base = b.const_value(tmp(MockType::U64), ConstValue::U64(DATA_AT))?;
    b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
        dest,
        base,
        index: None,
        stride: 1,
        offset: 0,
        result_type: MockType::Ptr,
    })
}

/// Sets the counter to [`DEPTH`] and calls [`COUNTDOWN`].
fn start() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let counter = counter(b)?;
            let depth = b.const_value(tmp(MockType::U8), ConstValue::U8(DEPTH))?;
            b.store_indirect(counter, depth, MockType::U8)?;
            b.call_void(Callee::Direct(COUNTDOWN), vec![])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// `countdown`: while the counter is not zero, decrements it, calls `next`,
/// then writes the count it read to its own slot.
fn countdown(next: u32) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let (counter, count) = builder
        .in_block(0, |b| {
            let counter = counter(b)?;
            let count = b.load_indirect(tmp(MockType::U8), counter, MockType::U8)?;
            let zero = b.const_value(tmp(MockType::U8), ConstValue::U8(0))?;
            b.branch_cmp(count, zero, CmpKind::Eq, false, 2, 1)?;
            Ok((counter, count))
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let one = b.const_value(tmp(MockType::U8), ConstValue::U8(1))?;
            let left = b.sub(tmp(MockType::U8), count, one)?;
            b.store_indirect(counter, left, MockType::U8)?;
            b.call_void(Callee::Direct(next), vec![])?;
            let slots = b.const_value(tmp(MockType::U64), ConstValue::U64(SLOTS_AT))?;
            let slot = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: slots,
                index: Some(count),
                stride: 1,
                offset: 0,
                result_type: MockType::Ptr,
            })?;
            b.store_indirect(slot, count, MockType::U8)?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// The bodies of [`COUNTDOWN`] and [`PARTNER`].
struct Bodies {
    countdown: SsaFunction<MockTarget>,
    partner: SsaFunction<MockTarget>,
}

impl ExecutionHost<MockTarget> for Bodies {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        match *method {
            COUNTDOWN => Some(&self.countdown),
            PARTNER => Some(&self.partner),
            _ => None,
        }
    }
}

/// A body that calls itself, or calls a partner that calls it back, runs each
/// entry as its own activation: every one writes the count it read before
/// the inner call to its own slot after it. A depth too small for the
/// recursion ends the run exhausted at that depth, publishing nothing.
#[test]
fn a_reentered_body_runs_as_its_own_activation_within_the_depth() {
    // A run is one activation's writes, so each slot is a run of its own.
    let written: Vec<(u64, Vec<u8>)> = [(DATA_AT, 0)]
        .into_iter()
        .chain((1..=DEPTH).map(|count| (SLOTS_AT.saturating_add(u64::from(count)), count)))
        .map(|(address, byte)| (address, vec![byte]))
        .collect();
    for (name, host) in [
        (
            "self",
            Bodies {
                countdown: countdown(COUNTDOWN),
                partner: countdown(COUNTDOWN),
            },
        ),
        (
            "mutual",
            Bodies {
                countdown: countdown(PARTNER),
                partner: countdown(COUNTDOWN),
            },
        ),
    ] {
        // The root, one activation per count, and the one that reads zero.
        let depth = usize::from(DEPTH).saturating_add(2);
        let root = start();
        let facts = run(
            &root,
            &host,
            WorkBudget::DEFAULT.with_limit(Budget::ExecutionDepth, depth),
        );
        assert!(facts.is_complete(), "{name}: {:?}", facts.completion());
        assert_eq!(facts.image_writes(), written, "{name}");

        let shallow = run(
            &root,
            &host,
            WorkBudget::DEFAULT.with_limit(Budget::ExecutionDepth, depth.saturating_sub(1)),
        );
        assert_eq!(
            shallow.completion(),
            Completion::Exhausted(Exhaustion {
                budget: Budget::ExecutionDepth,
                limit: depth.saturating_sub(1),
                demand: depth,
            }),
            "{name}"
        );
        assert!(shallow.image_writes().is_empty(), "{name}");
    }
}
