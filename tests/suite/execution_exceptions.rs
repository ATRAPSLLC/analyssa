//! Static execution of functions with exception handlers: a raise inside a
//! protected region enters the handlers that guard it, with the values it
//! raises with, and goes on outward; code that cannot raise runs as it would
//! anywhere.

use std::process;

use analyssa::{
    analysis::{
        context::AnalysisContext,
        execution::{
            Allocation, AllocatorFamily, CallModel, CallSite, Completion, Conditions, Delivery,
            ExecutionHost, FindingKind, LibraryModel, ModelCall, ModelControl, Operand,
            StaticExecution, StaticFacts, Value,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        CallInputKey, CallOutput, Callee,
        exception::{BlockRange, SsaExceptionHandler},
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{AccessQualifiers, CallData, CallEffects, CallTermination, RuntimeValueKind, SsaOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Ends the test: a fixture the builder rejects is a broken test.
fn abort<E, R>(_: E) -> R {
    process::abort()
}

fn tmp(ty: MockType) -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(ty)
}

/// A clause of the mock's kind `flags` (`0` a catch, `1` a filter, `4` a
/// catch's continuation) protecting blocks `protected`, its handler at
/// `handler` where it is in this function, its filter at `filter`.
fn clause(
    flags: u32,
    protected: (usize, usize),
    handler: Option<(usize, usize)>,
    filter: Option<(usize, usize)>,
) -> SsaExceptionHandler<MockTarget> {
    let range = |(start, end)| BlockRange::new(start, end);
    SsaExceptionHandler {
        flags,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: range(protected),
        handler_range: handler.and_then(range),
        filter_range: filter.and_then(range),
        selector: None,
    }
}

/// Runs `ssa` alone against `host`.
fn run(
    ssa: &SsaFunction<MockTarget>,
    host: &dyn ExecutionHost<MockTarget>,
) -> StaticFacts<MockTarget> {
    let context = AnalysisContext::new(MACHINE);
    StaticExecution::new(ssa, &context).with_host(host).run()
}

/// The constant `facts` say the root returns.
fn returned(facts: &StaticFacts<MockTarget>) -> Option<ConstValue<MockTarget>> {
    facts.returned().and_then(Value::constant).cloned()
}

/// What a division divides by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Divisor {
    /// Zero: the division traps on every path.
    Zero,
    /// Five: it never traps.
    Five,
    /// The function's argument: it may.
    Argument,
}

/// `100 / divisor` in block 0, which then returns one; block 1 returns seven.
/// Where `protected`, block 0 is protected and block 1 is its handler, as an
/// SEH `__except` guards a division.
fn division(divisor: Divisor, protected: bool) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    builder
        .in_block(0, |b| {
            let hundred = b.const_value(tmp(MockType::I64), ConstValue::I64(100))?;
            let divisor = match divisor {
                Divisor::Zero => b.const_value(tmp(MockType::I64), ConstValue::I64(0))?,
                Divisor::Five => b.const_value(tmp(MockType::I64), ConstValue::I64(5))?,
                Divisor::Argument => b.load_arg(SsaDefSpec::argument(0, MockType::I64), 0)?,
            };
            b.div(tmp(MockType::I64), hundred, divisor, false)?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.ret(Some(one))
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let seven = b.const_value(tmp(MockType::I64), ConstValue::I64(7))?;
            b.ret(Some(seven))
        })
        .unwrap_or_else(abort);
    let mut ssa = builder.finish().unwrap_or_else(abort);
    if protected {
        ssa.set_exception_handlers(vec![clause(0, (0, 1), Some((1, 2)), None)]);
    }
    ssa
}

/// **A raise in a protected region enters its handler.** A division by zero
/// in an `__except`-guarded block raises on every path: the run goes on in
/// the handler, which returns seven, where unguarded the path ends at the
/// trap and nothing returns. A division by five never raises, and the
/// handler is never entered. A division by the argument may raise: both
/// the normal return and the handler's are paths, so the root returns no one
/// constant.
#[test]
fn a_raise_in_a_protected_region_enters_its_handler() {
    for (divisor, protected, expected) in [
        (Divisor::Zero, true, Some(ConstValue::I64(7))),
        (Divisor::Zero, false, None),
        (Divisor::Five, true, Some(ConstValue::I64(1))),
        (Divisor::Argument, true, None),
        (Divisor::Argument, false, Some(ConstValue::I64(1))),
    ] {
        let facts = run(&division(divisor, protected), &());
        assert!(
            facts.is_complete(),
            "{divisor:?} {protected}: {:?}",
            facts.completion()
        );
        assert_eq!(returned(&facts), expected, "{divisor:?} {protected}");
    }
}

/// Block 0 defines one, divides by zero where `early`, defines three,
/// divides by zero otherwise, and returns three; its handler, block 1,
/// returns the phi of the two values a raise in block 0 may bring.
fn phi_at_raise(early: bool) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let (first, second) = builder
        .in_block(0, |b| {
            let hundred = b.const_value(tmp(MockType::I64), ConstValue::I64(100))?;
            let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
            let first = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            if early {
                b.div(tmp(MockType::I64), hundred, zero, false)?;
            }
            let second = b.const_value(tmp(MockType::I64), ConstValue::I64(3))?;
            if !early {
                b.div(tmp(MockType::I64), hundred, zero, false)?;
            }
            b.ret(Some(second))?;
            Ok((first, second))
        })
        .unwrap_or_else(abort);
    let merged: SsaVarId = builder
        .in_block(1, |b| {
            let merged = b.empty_phi(tmp(MockType::I64))?;
            b.ret(Some(merged))?;
            Ok(merged)
        })
        .unwrap_or_else(abort);
    let mut ssa = builder.finish().unwrap_or_else(abort);
    if let Some(phi) = ssa
        .block_mut(1)
        .and_then(|block| block.phi_mut(0))
        .filter(|phi| phi.result() == merged)
    {
        phi.add_exceptional_operand(0, first);
        phi.add_exceptional_operand(0, second);
    }
    ssa.set_exception_handlers(vec![clause(0, (0, 1), Some((1, 2)), None)]);
    ssa
}

/// **A handler receives the values that stand where the raise happens.** A
/// handler phi merges every value a raise in the protected block may bring;
/// the one it takes is the one the storage holds at the raising
/// instruction: one where the block raises before redefining it, three
/// where it raises after.
#[test]
fn a_handler_receives_the_values_where_the_raise_happens() {
    for (early, expected) in [(true, ConstValue::I64(1)), (false, ConstValue::I64(3))] {
        let facts = run(&phi_at_raise(early), &());
        assert!(facts.is_complete(), "{early}: {:?}", facts.completion());
        assert_eq!(returned(&facts), Some(expected), "{early}");
    }
}

/// Block 0 calls method 9, which never returns, then returns one; block 1,
/// a landing pad, takes the exception the unwinder hands it and returns two.
fn landing_pad() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let mut effects = CallEffects::pure();
            effects.termination = CallTermination::NeverReturns;
            let mut call = CallData::new(None, Callee::Direct(9), vec![]);
            call.effects = Some(effects);
            b.emit_no_defs(SsaOp::Call(Box::new(call)))?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.ret(Some(one))
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::RuntimeValue {
                dest,
                kind: RuntimeValueKind::ExceptionObject,
            })?;
            let two = b.const_value(tmp(MockType::I64), ConstValue::I64(2))?;
            b.ret(Some(two))
        })
        .unwrap_or_else(abort);
    let mut ssa = builder.finish().unwrap_or_else(abort);
    ssa.set_exception_handlers(vec![clause(0, (0, 1), Some((1, 2)), None)]);
    ssa
}

/// **A call that never returns raises into its landing pad.** A C++ throw
/// never returns to the code after it; inside a call-site range it enters
/// the landing pad, which holds the exception the unwinder hands it, and the
/// root returns what the pad returns.
#[test]
fn a_call_that_never_returns_raises_into_its_landing_pad() {
    let facts = run(&landing_pad(), &());
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(returned(&facts), Some(ConstValue::I64(2)));
}

/// Refuses every effect the execution leaves unevaluated, as an audit does,
/// and knows the body of [`landing_pad`]'s callee: one that writes nothing
/// and returns, which the call's statement skips.
struct AuditKnowingCallee(SsaFunction<MockTarget>);

impl ExecutionHost<MockTarget> for AuditKnowingCallee {
    fn admit_unevaluated_call(&self, _: CallSite<'_, MockTarget>) -> bool {
        false
    }

    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 9).then_some(&self.0)
    }
}

/// **What a call that never returns did is seen where its raise is caught.**
/// Under an audit, the call [`landing_pad`] states never returns raises into
/// its own function's landing pad, which runs on whatever the skipped
/// callee left, so the audit refuses it even with the callee's body known.
#[test]
fn an_audit_refuses_a_never_returning_call_a_handler_catches() {
    let mut callee = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    callee.in_block(0, |b| b.ret(None)).unwrap_or_else(abort);
    let audit = AuditKnowingCallee(callee.finish().unwrap_or_else(abort));
    let facts = run(&landing_pad(), &audit);
    assert_eq!(facts.completion(), Completion::UnevaluatedEffects);
}

/// Answers every call with one model.
struct Modelled(&'static LibraryModel);

impl ExecutionHost<MockTarget> for Modelled {
    fn model(&self, _: &u32) -> Option<&dyn CallModel<MockTarget>> {
        Some(self.0)
    }
}

/// **A call that never returns runs its model, which states how it leaves.**
/// The call [`landing_pad`] states never returns, modelled as `exit` is,
/// ends the program: no handler runs, and nothing returns. Modelled as a
/// call that returns, it still resumes nothing after itself, as its
/// statement says. The control: unmodelled, the call may raise, and the
/// landing pad's two is returned.
#[test]
fn a_call_that_never_returns_leaves_as_its_model_states() {
    for model in [&LibraryModel::NoReturn, &LibraryModel::NoWrites] {
        let facts = run(&landing_pad(), &Modelled(model));
        assert!(facts.is_complete(), "{model:?}: {:?}", facts.completion());
        assert_eq!(returned(&facts), None, "{model:?}");
    }
    let unmodelled = run(&landing_pad(), &());
    assert_eq!(returned(&unmodelled), Some(ConstValue::I64(2)));
}

/// The method of [`thrower`].
const THROWER: u32 = 5;

/// Divides by zero and returns: a callee whose every path raises.
fn thrower() -> SsaFunction<MockTarget> {
    division(Divisor::Zero, false)
}

/// Block 0 calls the thrower, then returns one; block 1, its handler,
/// returns nine.
fn calls_thrower() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            b.call(tmp(MockType::I64), Callee::Direct(THROWER), vec![])?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.ret(Some(one))
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let nine = b.const_value(tmp(MockType::I64), ConstValue::I64(9))?;
            b.ret(Some(nine))
        })
        .unwrap_or_else(abort);
    let mut ssa = builder.finish().unwrap_or_else(abort);
    ssa.set_exception_handlers(vec![clause(0, (0, 1), Some((1, 2)), None)]);
    ssa
}

/// Knows the thrower's body.
struct Thrower(SsaFunction<MockTarget>);

impl ExecutionHost<MockTarget> for Thrower {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == THROWER).then_some(&self.0)
    }
}

/// **A raise unwinds the activations that do not handle it.** The callee
/// divides by zero with no handler of its own; the raise leaves it and is
/// its caller's, at the call, inside the caller's protected block, whose
/// handler returns nine.
#[test]
fn a_raise_unwinds_to_the_caller_that_handles_it() {
    let facts = run(&calls_thrower(), &Thrower(thrower()));
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(returned(&facts), Some(ConstValue::I64(9)));
}

/// Block 0 loads through the argument, a pointer the run does not know, and
/// returns one; where `protected`, block 1, its handler, returns two.
fn unknown_load(protected: bool) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    builder
        .in_block(0, |b| {
            let pointer = b.load_arg(SsaDefSpec::argument(0, MockType::Ptr), 0)?;
            b.load_indirect(tmp(MockType::U8), pointer, MockType::U8)?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.ret(Some(one))
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let two = b.const_value(tmp(MockType::I64), ConstValue::I64(2))?;
            b.ret(Some(two))
        })
        .unwrap_or_else(abort);
    let mut ssa = builder.finish().unwrap_or_else(abort);
    if protected {
        ssa.set_exception_handlers(vec![clause(0, (0, 1), Some((1, 2)), None)]);
    }
    ssa
}

/// **An access through an unknown address may fault into its handler.** A
/// load through a pointer the run does not know may fault, as a `__try`
/// probing memory expects: guarded, the handler's return is a path beside
/// the normal one, so the root returns no one constant; unguarded, a fault
/// is no path the program continues on, and the root returns one.
#[test]
fn an_access_through_an_unknown_address_may_fault_into_its_handler() {
    for (protected, expected) in [(true, None), (false, Some(ConstValue::I64(1)))] {
        let facts = run(&unknown_load(protected), &());
        assert!(facts.is_complete(), "{protected}: {:?}", facts.completion());
        assert_eq!(returned(&facts), expected, "{protected}");
    }
}

/// Image read-only from `0x1000` to `0x1010` and writable for the sixteen
/// bytes below it, its numbers its addresses.
struct Protected;

impl ImageData for Protected {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        [
            (0x0ff0, 0x1000, Mutability::Writable),
            (0x1000, 0x1010, Mutability::ReadOnly),
        ]
        .into_iter()
        .find(|(start, end, _)| (*start..*end).contains(&address))
        .map(|(start, end, mutability)| ImageRegion {
            start,
            end,
            mutability,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        const BYTES: [u8; 0x20] = [0; 0x20];
        let start = usize::try_from(address.checked_sub(0x0ff0)?).ok()?;
        BYTES.get(start..start.checked_add(len)?)
    }
}

/// How a store reaches the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Store {
    /// One byte at the address.
    Byte(u64),
    /// Sixteen bytes filled from the address.
    Fill(u64),
}

/// Block 1 stores as `store` says and returns one; block 2 returns three;
/// block 3 returns two, the handler of block 1 where `protected`. Block 0
/// goes to block 1, or where `branched`, to block 1 or 2 as the argument
/// says.
fn image_store(store: Store, branched: bool, protected: bool) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    builder
        .in_block(0, |b| {
            if branched {
                let taken = b.load_arg(SsaDefSpec::argument(0, MockType::Bool), 0)?;
                b.branch(taken, 1, 2)
            } else {
                b.jump(1)
            }
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let (Store::Byte(at) | Store::Fill(at)) = store;
            let pointer = b.const_value(tmp(MockType::U64), ConstValue::U64(at))?;
            let byte = b.const_value(tmp(MockType::U8), ConstValue::U8(7))?;
            match store {
                Store::Byte(_) => b.store_indirect(pointer, byte, MockType::U8)?,
                Store::Fill(_) => {
                    let size = b.const_value(tmp(MockType::U64), ConstValue::U64(16))?;
                    b.emit_no_defs(SsaOp::InitBlk {
                        dest_addr: pointer,
                        value: byte,
                        size,
                        backward: None,
                        access: AccessQualifiers::FLAT,
                    })?;
                }
            }
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.ret(Some(one))
        })
        .unwrap_or_else(abort);
    for (block, value) in [(2, 3), (3, 2)] {
        builder
            .in_block(block, |b| {
                let value = b.const_value(tmp(MockType::I64), ConstValue::I64(value))?;
                b.ret(Some(value))
            })
            .unwrap_or_else(abort);
    }
    let mut ssa = builder.finish().unwrap_or_else(abort);
    if protected {
        ssa.set_exception_handlers(vec![clause(0, (1, 2), Some((3, 4)), None)]);
    }
    ssa
}

/// **A store into read-only image faults its path, and the run goes on.**
/// The machine faults a store into storage the image states is never
/// written: the path ends at the store, a finding records it with the bytes
/// it reached, and every other path makes its facts, so a branch that stores
/// there leaves the root returning the other branch's three. A handler
/// guarding the store receives the raise and returns two. A fill that starts
/// in writable storage and runs into the read-only bytes faults as well,
/// raising from the path before it. The control: the same stores into
/// writable storage fault nowhere, find nothing and return one.
#[test]
fn a_store_into_read_only_image_faults_its_path() {
    let context = AnalysisContext::new(MACHINE).with_image(&Protected);
    for (store, branched, protected, expected, found) in [
        (Store::Byte(0x1000), false, false, None, Some((0x1000, 1))),
        (Store::Byte(0x1000), true, false, Some(3), Some((0x1000, 1))),
        (Store::Byte(0x1000), false, true, Some(2), Some((0x1000, 1))),
        (Store::Fill(0x0ff8), false, false, None, Some((0x0ff8, 16))),
        (
            Store::Fill(0x0ff8),
            false,
            true,
            Some(2),
            Some((0x0ff8, 16)),
        ),
        (Store::Byte(0x0ff8), false, false, Some(1), None),
        (Store::Byte(0x0ff8), false, true, Some(1), None),
        (Store::Fill(0x0ff0), false, true, Some(1), None),
    ] {
        let case = format!("{store:?} branched={branched} protected={protected}");
        let facts = StaticExecution::new(&image_store(store, branched, protected), &context)
            .with_host(&())
            .run();
        assert!(facts.is_complete(), "{case}: {:?}", facts.completion());
        assert_eq!(returned(&facts), expected.map(ConstValue::I64), "{case}");
        let faults: Vec<_> = facts
            .findings()
            .iter()
            .filter(|finding| finding.kind == FindingKind::ReadOnlyWrite)
            .map(|finding| {
                assert_eq!(finding.block, 1, "{case}");
                finding
                    .access
                    .and_then(|(address, len)| Some((address.image_address()?, len)))
            })
            .collect();
        assert_eq!(
            faults,
            found.map(Some).into_iter().collect::<Vec<_>>(),
            "{case}"
        );
        assert_eq!(facts.usage().traps, usize::from(found.is_some()), "{case}");
    }
}

/// A host that refuses effects the execution leaves unevaluated.
struct Audit;

impl ExecutionHost<MockTarget> for Audit {
    fn admit_unevaluated_call(&self, _: CallSite<'_, MockTarget>) -> bool {
        false
    }
}

/// **A handler that runs elsewhere is followed only where the function
/// resumes after it.** A catch funclet runs outside the function and the
/// function resumes at a continuation: the raise reaches the continuation
/// past effects the execution does not follow, so a host that admits them
/// sees the continuation's return, and one that refuses them ends the run.
/// A clause whose handler runs elsewhere and names no continuation may
/// resume the function where the execution cannot say, and ends the run.
#[test]
fn a_handler_that_runs_elsewhere_is_followed_where_the_function_resumes() {
    let mut resumed = division(Divisor::Zero, false);
    resumed.set_exception_handlers(vec![
        clause(0, (0, 1), None, None),
        clause(4, (0, 1), Some((1, 2)), None),
    ]);
    let admitted = run(&resumed, &());
    assert!(admitted.is_complete(), "{:?}", admitted.completion());
    assert_eq!(returned(&admitted), Some(ConstValue::I64(7)));
    let audited = run(&resumed, &Audit);
    assert_eq!(audited.completion(), Completion::UnevaluatedEffects);

    let mut nowhere = division(Divisor::Zero, false);
    nowhere.set_exception_handlers(vec![clause(0, (0, 1), None, None)]);
    let ended = run(&nowhere, &());
    assert_eq!(ended.completion(), Completion::Stuck { block: 0 });
}

/// **A raise enters a filter's handler, never its filter.** A filter
/// expression returns its verdict to the runtime, not to the function, so
/// its return is no return of the function: a division by zero guarded by a
/// filter at block 2, which returns ninety-nine, enters the handler at
/// block 1, which returns seven, and the root never returns ninety-nine.
#[test]
fn a_raise_enters_a_filter_s_handler_never_its_filter() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let hundred = b.const_value(tmp(MockType::I64), ConstValue::I64(100))?;
            let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
            b.div(tmp(MockType::I64), hundred, zero, false)?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.ret(Some(one))
        })
        .unwrap_or_else(abort);
    for (block, value) in [(1, 7), (2, 99)] {
        builder
            .in_block(block, |b| {
                let held = b.const_value(tmp(MockType::I64), ConstValue::I64(value))?;
                b.ret(Some(held))
            })
            .unwrap_or_else(abort);
    }
    let mut ssa = builder.finish().unwrap_or_else(abort);
    ssa.set_exception_handlers(vec![clause(1, (0, 1), Some((1, 2)), Some((2, 3)))]);
    let facts = run(&ssa, &());
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(returned(&facts), Some(ConstValue::I64(7)));
}

/// An image whose integers are module addresses as they stand, serving no
/// bytes: the coordinates a thrown `type_info` and a clause's caught type
/// are compared in.
struct Coordinates;

impl ImageData for Coordinates {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }

    fn region(&self, _: u64) -> Option<ImageRegion> {
        None
    }

    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }
}

/// The family of the exceptions [`Runtime`] allocates.
const EXCEPTIONS: AllocatorFamily = AllocatorFamily(30);

/// The module address of the `type_info` the throw names.
const THROWN_TYPE: u32 = 0x4028;

/// `__cxa_allocate_exception(size)`.
static ALLOCATE_EXCEPTION: LibraryModel = LibraryModel::Allocate(Allocation {
    family: EXCEPTIONS,
    size: Operand::Argument(0),
    count: None,
    alignment: None,
    zeroed: false,
    flags: None,
    delivery: Delivery::ReturnedOrEnds,
    requires: Conditions::NONE,
    pages: None,
    protection: None,
    heap: None,
});

/// `__cxa_throw(object, type_info, destructor)`.
static THROW: LibraryModel = LibraryModel::Throw {
    object: 0,
    type_info: 1,
    destructor: 2,
};

/// `__cxa_end_catch()`.
static END_CATCH: LibraryModel = LibraryModel::EndCatch { family: EXCEPTIONS };

/// The C++ runtime as [`throws`] calls it: methods 20 to 23.
struct Runtime;

impl ExecutionHost<MockTarget> for Runtime {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match method {
            20 => Some(&ALLOCATE_EXCEPTION),
            21 => Some(&THROW),
            22 => Some(&LibraryModel::BeginCatch),
            23 => Some(&END_CATCH),
            _ => None,
        }
    }
}

/// A clause protecting block 0 whose landing pad is block 1 and whose
/// runtime hands it `selector`: a catch of the type at `caught` (zero for
/// every type), or a cleanup (`flags` 3).
fn itanium(flags: u32, caught: u32, selector: Option<i64>) -> SsaExceptionHandler<MockTarget> {
    SsaExceptionHandler {
        class_token_or_filter: caught,
        selector,
        ..clause(flags, (0, 1), Some((1, 2)), None)
    }
}

/// `try { throw 1; }` with its landing pad: block 0 allocates a four-byte
/// exception, stores one in it, and throws it with the `type_info` at
/// [`THROWN_TYPE`] and `destructor`; block 1, the pad, compares the selector
/// it is handed with one, as a pad compares a catch's type filter; block 2,
/// that catch, takes the exception, finishes with it and returns -1; block 3,
/// any other selector, returns two.
fn throws(
    destructor: i64,
    clauses: Vec<SsaExceptionHandler<MockTarget>>,
) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let size = b.const_value(tmp(MockType::I64), ConstValue::I64(4))?;
            let object = b.call(tmp(MockType::Ptr), Callee::Direct(20), vec![size])?;
            let one = b.const_value(tmp(MockType::I32), ConstValue::I32(1))?;
            b.store_indirect(object, one, MockType::I32)?;
            let type_info =
                b.const_value(tmp(MockType::I64), ConstValue::I64(i64::from(THROWN_TYPE)))?;
            let destructor = b.const_value(tmp(MockType::I64), ConstValue::I64(destructor))?;
            // A catalog states `__cxa_throw` never returns, as its
            // declaration does.
            let mut effects = CallEffects::pure();
            effects.termination = CallTermination::NeverReturns;
            let mut call = CallData::new(
                None,
                Callee::Direct(21),
                vec![object, type_info, destructor],
            );
            call.effects = Some(effects);
            b.emit_no_defs(SsaOp::Call(Box::new(call)))?;
            let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
            b.ret(Some(zero))
        })
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let exception = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::RuntimeValue {
                dest,
                kind: RuntimeValueKind::ExceptionObject,
            })?;
            let selector = b.emit_def(tmp(MockType::I64), |dest| SsaOp::RuntimeValue {
                dest,
                kind: RuntimeValueKind::HandlerEntry,
            })?;
            let _ = exception;
            let filter = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            let chosen = b.ceq(tmp(MockType::Bool), selector, filter)?;
            b.branch(chosen, 2, 3)
        })
        .unwrap_or_else(abort);
    builder
        .in_block(2, |b| {
            let none = b.const_value(tmp(MockType::Ptr), ConstValue::I64(0))?;
            b.call(tmp(MockType::Ptr), Callee::Direct(22), vec![none])?;
            let call = CallData::new(None, Callee::Direct(23), vec![]);
            b.emit_no_defs(SsaOp::Call(Box::new(call)))?;
            let minus = b.const_value(tmp(MockType::I64), ConstValue::I64(-1))?;
            b.ret(Some(minus))
        })
        .unwrap_or_else(abort);
    builder
        .in_block(3, |b| {
            let two = b.const_value(tmp(MockType::I64), ConstValue::I64(2))?;
            b.ret(Some(two))
        })
        .unwrap_or_else(abort);
    let mut ssa = builder.finish().unwrap_or_else(abort);
    ssa.set_exception_handlers(clauses);
    ssa
}

/// Runs `ssa` against the C++ runtime's models, its integers module
/// addresses.
fn run_thrown(ssa: &SsaFunction<MockTarget>) -> StaticFacts<MockTarget> {
    let image = Coordinates;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    StaticExecution::new(ssa, &context)
        .with_host(&Runtime)
        .run()
}

/// **A thrown exception enters the clause the personality chooses for its
/// type.** `throw 1` at a pad whose one catch names the thrown type, handed
/// type filter one: the pad receives selector one, the catch takes and
/// finishes with the exception, and the root returns -1, never the other
/// arm's two. A catch-all chooses it as well. The control: a pad whose
/// runtime hands no selector (one stated nowhere) receives none, so both
/// arms are paths and the root returns no one constant.
#[test]
fn a_thrown_exception_enters_the_clause_its_type_chooses() {
    for caught in [THROWN_TYPE, 0] {
        let facts = run_thrown(&throws(0, vec![itanium(0, caught, Some(1))]));
        assert!(facts.is_complete(), "{caught:#x}: {:?}", facts.completion());
        assert_eq!(returned(&facts), Some(ConstValue::I64(-1)), "{caught:#x}");
    }
    let unstated = run_thrown(&throws(0, vec![itanium(0, THROWN_TYPE, None)]));
    assert!(unstated.is_complete(), "{:?}", unstated.completion());
    assert_eq!(returned(&unstated), None, "no selector, so both arms");
}

/// **A cleanup takes an exception no catch takes for certain.** A pad whose
/// only clause is a cleanup is entered with selector zero, so it takes the
/// other arm and returns two. Behind a catch of another type, which may
/// take a derived exception, both the catch's selector and the cleanup's
/// are entered: the root returns no one constant.
#[test]
fn a_cleanup_takes_an_exception_no_catch_takes_for_certain() {
    let cleanup = run_thrown(&throws(0, vec![itanium(3, 0, Some(0))]));
    assert!(cleanup.is_complete(), "{:?}", cleanup.completion());
    assert_eq!(returned(&cleanup), Some(ConstValue::I64(2)));
    let behind = run_thrown(&throws(
        0,
        vec![itanium(0, 0x5000, Some(1)), itanium(3, 0, Some(0))],
    ));
    assert!(behind.is_complete(), "{:?}", behind.completion());
    assert_eq!(
        returned(&behind),
        None,
        "the catch may take it, or the cleanup"
    );
}

/// **Finishing with an exception that has a destructor runs program code.**
/// `__cxa_end_catch` destroys the caught exception with the destructor its
/// throw named: a null one frees the object and the run completes; a known
/// non-null one is code the model cannot run, so the run does not complete.
#[test]
fn finishing_with_an_exception_runs_its_destructor() {
    let destroyed = run_thrown(&throws(0x9000, vec![itanium(0, THROWN_TYPE, Some(1))]));
    assert!(!destroyed.is_complete(), "{:?}", destroyed.completion());
}

/// **A throw's model accounts for the destructor it is handed.** The throw
/// calls no function itself: the destructor it takes runs when a handler
/// finishes with the exception, which its model declines. So the throw's
/// model states that callback parameter and no other, and no other model
/// states one.
#[test]
fn a_throw_accounts_for_its_destructor_alone() {
    assert!(THROW.accounts_for_callback(2));
    assert!(!THROW.accounts_for_callback(0));
    assert!(!END_CATCH.accounts_for_callback(0));
    assert!(!LibraryModel::NoReturn.accounts_for_callback(0));
}

/// The machine cell [`unwound_call`] hands its call.
const CELL: CallInputKey = CallInputKey::new(100);

/// Block 0 holds 40 in [`CELL`] and calls `method` with it, the call's
/// outputs stating the cell's value after it, then returns one; block 1,
/// the handler of a clause protecting block 0, returns that output.
fn unwound_call(method: u32) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let after = builder
        .in_block(0, |b| {
            let held = b.const_value(tmp(MockType::I64), ConstValue::I64(40))?;
            b.emit_def(tmp(MockType::I64), |dest| {
                CallData::new(Some(dest), Callee::Direct(method), vec![])
                    .with_implicit_input(CELL, held)
                    .into()
            })?;
            let after = b.emit_def(tmp(MockType::I64), |dest| SsaOp::CallClobber {
                outputs: vec![dest],
                machine: Some(
                    vec![CallOutput {
                        key: CELL,
                        index: 0,
                    }]
                    .into_boxed_slice(),
                ),
            })?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.ret(Some(one))?;
            Ok(after)
        })
        .unwrap_or_else(abort);
    // The handler reads the call's output: what the unwinder leaves in the
    // cell, as a return would.
    builder
        .in_block(1, |b| b.ret(Some(after)))
        .unwrap_or_else(abort);
    let mut ssa = builder.finish().unwrap_or_else(abort);
    ssa.set_exception_handlers(vec![clause(0, (0, 1), Some((1, 2)), None)]);
    ssa
}

/// A model that raises, as `__cxa_throw` does.
struct Raises;

impl CallModel<MockTarget> for Raises {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.set_control(ModelControl::Trap);
    }
}

/// The thrower's body for [`THROWER`], [`Raises`] for any other method.
struct Cells(SsaFunction<MockTarget>);

impl ExecutionHost<MockTarget> for Cells {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == THROWER).then_some(&self.0)
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method != THROWER).then_some(&Raises)
    }
}

/// **A call that raises leaves its handler what the unwinder restores.** The
/// handler reads the call's outputs as a return leaves them. An output names
/// a register the call clobbers, which an unwinder does not restore: what
/// the call was handed in it is gone, whether a callee body raised below it
/// or a model raised in its place. The handler runs, and the 40 the call took
/// is not what it reads.
#[test]
fn a_raising_call_leaves_its_handler_the_cells_the_unwinder_restores() {
    for method in [THROWER, 9] {
        let facts = run(&unwound_call(method), &Cells(thrower()));
        assert!(facts.is_complete(), "{method}: {:?}", facts.completion());
        assert_eq!(returned(&facts), None, "{method}");
    }
}
