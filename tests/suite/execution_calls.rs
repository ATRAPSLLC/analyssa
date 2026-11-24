//! Static execution across calls: a callee run in place leaves its writes in
//! the caller's memory, and a callee run alone reports them as its effect.

use std::{
    cell::{Cell, RefCell},
    ops::Range,
    process,
    sync::Arc,
};

use analyssa::{
    analysis::{
        Machine,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            Address, Allocation, AllocatorFamily, ByteBound, CallModel, CallSite, CallStack,
            Completion, Conditions, Deallocation, Delivery, Duplication, Ending, ExecutionHost,
            ExtentSource, FindingKind, FoundObject, FreeResult, LibraryModel, ModelCall,
            ModelControl, ObjectExtents, ObjectKind, Operand, ParameterBindings, Reallocation,
            ReleaseFault, ReplacementProof, Root, StaticExecution, StaticFacts, StorageObject,
            UseBound, Value, Witness,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        Callee,
        exception::{BlockRange, SsaExceptionHandler},
        function::{SsaBlockBuilder, SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        lanes::VectorConst,
        ops::{
            AccessMode, AccessQualifiers, AtomicAccessWidth, AtomicOrdering, CallEffects,
            CallLocation, Capture, CmpKind, CompareBlkData, PointeeExtent, SsaOp, VectorMaskMode,
        },
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Read-only source bytes.
const SOURCE_AT: u64 = 0x1000;

/// The source bytes.
const SOURCE: &[u8] = b"ABCD";

/// Writable data the callee fills.
const TARGET_AT: u64 = 0x2000;

/// The callee's method number.
const CALLEE: u32 = 5;

/// Read-only bytes at `SOURCE_AT`, writable ones at `TARGET_AT`.
struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        let len = SOURCE.len() as u64;
        [
            (SOURCE_AT, Mutability::ReadOnly),
            (TARGET_AT, Mutability::Writable),
        ]
        .into_iter()
        .find(|(start, _)| (*start..start.wrapping_add(len)).contains(&address))
        .map(|(start, mutability)| ImageRegion {
            start,
            end: start.wrapping_add(len),
            mutability,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(address.checked_sub(SOURCE_AT)?).ok()?;
        SOURCE.get(start..start.checked_add(len)?)
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

/// `for i in 0..4 { target[i] = source[i] ^ 0x20 }`, then return.
fn callee() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let (source, target, zero, one, count, mask) = builder
        .in_block(0, |b| {
            let source = b.const_value(tmp(MockType::U64), ConstValue::U64(SOURCE_AT))?;
            let target = b.const_value(tmp(MockType::U64), ConstValue::U64(TARGET_AT))?;
            let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            let count = b.const_value(tmp(MockType::I64), ConstValue::I64(4))?;
            let mask = b.const_value(tmp(MockType::U8), ConstValue::U8(0x20))?;
            b.jump(1)?;
            Ok((source, target, zero, one, count, mask))
        })
        .unwrap_or_else(abort);
    let index = builder
        .in_block(1, |b| b.empty_phi(tmp(MockType::I64)))
        .unwrap_or_else(abort);
    let next = builder
        .in_block(1, |b| {
            let from = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: source,
                index: Some(index),
                stride: 1,
                offset: 0,
                result_type: MockType::Ptr,
            })?;
            let byte = b.load_indirect(tmp(MockType::U8), from, MockType::U8)?;
            let flipped = b.xor(tmp(MockType::U8), byte, mask)?;
            let to = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: target,
                index: Some(index),
                stride: 1,
                offset: 0,
                result_type: MockType::Ptr,
            })?;
            b.store_indirect(to, flipped, MockType::U8)?;
            let next = b.add(tmp(MockType::I64), index, one)?;
            b.branch_cmp(next, count, CmpKind::Lt, false, 1, 2)?;
            Ok(next)
        })
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, index, 0, zero)
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, index, 1, next)
        .unwrap_or_else(abort);
    builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// Calls the callee, then hands the target to method 7.
fn caller() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            b.call_void(Callee::Direct(CALLEE), vec![])?;
            let target = b.const_value(tmp(MockType::U64), ConstValue::U64(TARGET_AT))?;
            b.call_void(Callee::Direct(7), vec![target])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// A host that knows the callee's body.
struct Host {
    callee: SsaFunction<MockTarget>,
}

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == CALLEE).then_some(&self.callee)
    }
}

fn run(
    ssa: &SsaFunction<MockTarget>,
    host: &dyn ExecutionHost<MockTarget>,
) -> StaticFacts<MockTarget> {
    let image = Image;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    StaticExecution::new(ssa, &context).with_host(host).run()
}

/// Run alone, the callee reports the bytes it leaves in writable data.
#[test]
fn a_callee_reports_what_it_leaves_written() {
    let facts = run(&callee(), &());
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.image_writes(), vec![(TARGET_AT, b"abcd".to_vec())]);
}

/// Run in place, the callee's writes are in the caller's memory at the next
/// call; with no body to run, nothing is known there.
#[test]
fn a_callee_run_in_place_leaves_its_writes_to_the_caller() {
    let caller = caller();
    let host = Host { callee: callee() };
    let facts = run(&caller, &host);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let handed = facts.call_arguments(0, 2);
    assert_eq!(
        handed.first().map(|argument| argument.bytes.as_slice()),
        Some(&b"abcd"[..])
    );
    let unknown = run(&caller, &());
    assert!(unknown.call_arguments(0, 2).is_empty());
}

/// The method number of [`second_of_two`].
const SECOND: u32 = 6;

/// `return b` for a callee of two parameters `(a, b)`.
fn second_of_two() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let second = builder.function_mut().create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::I64,
    );
    builder
        .in_block(0, |b| b.ret(Some(second)))
        .unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// Passes `9` as the one argument of a call to [`second_of_two`] - the call
/// names only the parameter the callee reads - and returns what it returns.
fn narrowed_caller() -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let returned = builder
        .in_block(0, |b| {
            let nine = b.const_value(tmp(MockType::I64), ConstValue::I64(9))?;
            let returned = b.call(tmp(MockType::I64), Callee::Direct(SECOND), vec![nine])?;
            b.ret(Some(returned))?;
            Ok(returned)
        })
        .unwrap_or_else(abort);
    (builder.finish().unwrap_or_else(abort), returned)
}

/// A host whose calls name the callee's second parameter alone, and which
/// binds by the call it is shown.
struct Narrowed {
    callee: SsaFunction<MockTarget>,
}

impl ExecutionHost<MockTarget> for Narrowed {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == SECOND).then_some(&self.callee)
    }

    fn parameters(
        &self,
        site: CallSite<'_, MockTarget>,
        _method: &u32,
        body: &SsaFunction<MockTarget>,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        // The call at `(0, 1)` passes the second parameter alone.
        if (site.block, site.instruction) != (0, 1) {
            return;
        }
        for variable in body.variables() {
            if variable.origin() == VariableOrigin::Argument(1) {
                bindings.bind(variable.id(), 0);
            }
        }
    }
}

/// Positional binding would hand the one argument to the callee's first
/// parameter; a host that binds by the call hands it to the one it fills.
#[test]
fn a_host_binds_parameters_by_the_call() {
    let (caller, returned) = narrowed_caller();
    let facts = run(
        &caller,
        &Narrowed {
            callee: second_of_two(),
        },
    );
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.constant(returned), Some(&ConstValue::I64(9)));
    let positional = run(
        &caller,
        &Positional {
            callee: second_of_two(),
        },
    );
    assert!(positional.is_complete(), "{:?}", positional.completion());
    assert_eq!(positional.constant(returned), None);
}

/// The same body, bound by the default: argument `i` to parameter `i`.
struct Positional {
    callee: SsaFunction<MockTarget>,
}

impl ExecutionHost<MockTarget> for Positional {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == SECOND).then_some(&self.callee)
    }
}

/// The receiver [`virtual_caller`] passes.
const RECEIVER: u64 = 0x3000;

/// Calls the callee virtually on a receiver, then hands the target to method 7.
fn virtual_caller() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let receiver = b.const_value(tmp(MockType::U64), ConstValue::U64(RECEIVER))?;
            b.call_void(Callee::Virtual(CALLEE), vec![receiver])?;
            let target = b.const_value(tmp(MockType::U64), ConstValue::U64(TARGET_AT))?;
            b.call_void(Callee::Direct(7), vec![target])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// A host that knows the callee's body and resolves a virtual call to it on
/// the one receiver it knows the exact type of.
struct Dispatching {
    callee: SsaFunction<MockTarget>,
}

impl ExecutionHost<MockTarget> for Dispatching {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == CALLEE).then_some(&self.callee)
    }

    fn dispatch(&self, method: &u32, receiver: &Value<MockTarget>) -> Option<u32> {
        let known = matches!(receiver, Value::Const(ConstValue::U64(RECEIVER)));
        (*method == CALLEE && known).then_some(CALLEE)
    }
}

/// A virtual call may run an override of the method it names, so the named
/// body is entered only when the host resolves the receiver to it; a host that
/// knows the body but cannot dispatch leaves the call opaque.
#[test]
fn a_virtual_call_enters_only_the_body_the_host_dispatches_to() {
    let caller = virtual_caller();
    let dispatched = run(&caller, &Dispatching { callee: callee() });
    assert!(dispatched.is_complete(), "{:?}", dispatched.completion());
    assert_eq!(
        dispatched
            .call_arguments(0, 3)
            .first()
            .map(|argument| argument.bytes.as_slice()),
        Some(&b"abcd"[..])
    );
    let undispatched = run(&caller, &Host { callee: callee() });
    assert!(undispatched.call_arguments(0, 3).is_empty());
}

/// **A callee's protected region runs as ordinary code where nothing in it
/// raises.** Every access of the copy loop is through an address the run
/// knows, so no instruction may raise and the handler is never entered: the
/// callee leaves the caller what it does without the region.
#[test]
fn exception_regions_in_a_callee_run_where_nothing_raises() {
    let caller = caller();
    let mut body = callee();
    body.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 2),
        handler_range: BlockRange::new(2, 3),
        filter_range: None,
        selector: None,
    }]);
    let handed = |facts: &StaticFacts<MockTarget>| {
        facts
            .call_arguments(0, 2)
            .first()
            .map(|argument| argument.bytes.clone())
    };
    let facts = run(&caller, &Host { callee: body });
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let control = run(&caller, &Host { callee: callee() });
    assert!(control.is_complete());
    assert_eq!(handed(&facts), Some(b"abcd".to_vec()));
    assert_eq!(handed(&facts), handed(&control));
}

/// A callee without an ABI mapping must not release any caller-owned address.
#[test]
fn detached_callee_preserves_deep_caller_storage_passed_by_pointer() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .unwrap_or_else(|| process::abort())
        .set_role(VariableRole::EntryStackPointer);
    builder
        .in_block(0, |b| {
            let slot = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -(1i64 << 33),
                result_type: MockType::Ptr,
            })?;
            let initial = b.const_value(tmp(MockType::I32), ConstValue::I32(17))?;
            b.store_indirect(slot, initial, MockType::I32)?;
            b.call_void(analyssa::ir::Callee::Direct(CALLEE), vec![slot])?;
            let loaded = b.load_indirect(tmp(MockType::I32), slot, MockType::I32)?;
            b.ret(Some(loaded))
        })
        .unwrap_or_else(abort);
    let caller = builder.finish().unwrap_or_else(abort);

    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let pointer = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .in_block(0, |b| {
            let value = b.const_value(tmp(MockType::I32), ConstValue::I32(42))?;
            b.store_indirect(pointer, value, MockType::I32)?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let body = builder.finish().unwrap_or_else(abort);
    let facts = run(&caller, &Host { callee: body });
    assert!(facts.is_complete());
    assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::I32(42))));
}

/// A host exposing only the supplied call model.
struct ModelHost<'a>(&'a dyn CallModel<MockTarget>);

impl ExecutionHost<MockTarget> for ModelHost<'_> {
    fn model(&self, _: &u32) -> Option<&dyn CallModel<MockTarget>> {
        Some(self.0)
    }
}

/// Allocation success and failure must both contribute to the return value.
#[test]
fn fallible_allocation_explores_the_null_result() {
    for zeroed in [false, true] {
        let model = allocation(zeroed);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |b| {
                let size = b.const_value(tmp(MockType::I64), ConstValue::I64(4))?;
                let pointer = b.call(
                    tmp(MockType::Ptr),
                    analyssa::ir::Callee::Direct(17),
                    vec![size],
                )?;
                b.branch(pointer, 1, 2)
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let result = b.const_value(tmp(MockType::I32), ConstValue::I32(1))?;
                b.ret(Some(result))
            })
            .unwrap_or_else(abort);
        builder
            .in_block(2, |b| {
                let result = b.const_value(tmp(MockType::I32), ConstValue::I32(0))?;
                b.ret(Some(result))
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let facts = run(&ssa, &ModelHost(&model));
        assert!(facts.is_complete());
        assert!(facts.returned().and_then(Value::constant).is_none());
        assert!(facts.usage().paths > 0);
    }
}

/// A model's abnormal control outcome prevents observing the following return.
#[test]
fn models_do_not_invent_returns_after_exit_or_trap() {
    struct Control(ModelControl);
    impl CallModel<MockTarget> for Control {
        fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
            call.set_control(self.0);
        }
    }
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            b.call_void(analyssa::ir::Callee::Direct(17), vec![])?;
            let result = b.const_value(tmp(MockType::I32), ConstValue::I32(99))?;
            b.ret(Some(result))
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    for control in [
        ModelControl::NoReturn,
        ModelControl::Trap,
        ModelControl::Unsupported,
    ] {
        let facts = run(&ssa, &ModelHost(&Control(control)));
        assert_eq!(
            facts.usage().traps,
            usize::from(control == ModelControl::Trap)
        );
        assert!(facts.returned().is_none());
        assert!(facts.constants().is_empty());
        if control == ModelControl::Unsupported {
            assert_eq!(facts.completion(), Completion::Stuck { block: 0 });
        } else {
            assert!(facts.is_complete());
        }
    }
    let control = run(&ssa, &ModelHost(&LibraryModel::NoWrites));
    assert_eq!(control.returned(), Some(&Value::Const(ConstValue::I32(99))));
    let no_return = run(&ssa, &ModelHost(&LibraryModel::NoReturn));
    assert!(no_return.returned().is_none());
}

/// `malloc(n)`, or `calloc` of a size alone where `zeroed`.
const fn allocation(zeroed: bool) -> LibraryModel {
    LibraryModel::Allocate(Allocation {
        family: AllocatorFamily(1),
        size: Operand::Argument(0),
        count: None,
        alignment: None,
        zeroed,
        flags: None,
        delivery: Delivery::Returned,
        requires: Conditions::NONE,
        pages: None,
        protection: None,
        heap: None,
    })
}

/// The zeroing allocator [`AllocatorHost`] models.
const ZEROED: LibraryModel = allocation(true);

/// The deallocator [`AllocatorHost`] models.
const FREE: LibraryModel = LibraryModel::Free(Deallocation {
    family: AllocatorFamily(1),
    pointer: Operand::Argument(0),
    size: None,
    result: FreeResult::Nothing,
    requires: Conditions::NONE,
    whole: None,
    released: None,
    heap: None,
});

/// Models an allocator, a deallocator, and a memory-preserving external call.
struct AllocatorHost;

impl ExecutionHost<MockTarget> for AllocatorHost {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match method {
            17 => Some(&ZEROED),
            18 => Some(&FREE),
            19 => Some(&LibraryModel::NoWrites),
            _ => None,
        }
    }
}

/// **Heap facts need a live allocation on the executing path; a read of an
/// ended block or past a block's end is a finding, and the run goes on.** A
/// consumer handed the block sees its bytes while it is live, and nothing
/// once it is freed. A read of the freed block, or one byte past its end, is
/// found at the reading instruction, and the consumer after it still sees
/// what the path holds.
#[test]
fn heap_consumers_observe_lifetimes_and_extent_proofs() {
    for (helper, invalid_access, finding, expected) in [
        (None, false, None, b"*\0\0\0".as_slice()),
        (Some(19), false, None, b"*\0\0\0".as_slice()),
        (Some(18), false, None, b"".as_slice()),
        (Some(20), false, None, b"".as_slice()),
        (
            Some(18),
            true,
            Some(FindingKind::Ended(Ending::Every)),
            b"".as_slice(),
        ),
        (
            None,
            true,
            Some(FindingKind::PastExtent(ExtentSource::Allocator)),
            b"*\0\0\0".as_slice(),
        ),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let pointer = builder
            .in_block(0, |b| {
                let size = b.const_value(tmp(MockType::U64), ConstValue::U64(4))?;
                let pointer = b.call(
                    tmp(MockType::Ptr),
                    analyssa::ir::Callee::Direct(17),
                    vec![size],
                )?;
                b.branch(pointer, 1, 2)?;
                Ok(pointer)
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let contents = b.const_value(tmp(MockType::U8), ConstValue::U8(42))?;
                b.store_indirect(pointer, contents, MockType::U8)?;
                if let Some(helper) = helper {
                    b.call_void(analyssa::ir::Callee::Direct(helper), vec![pointer])?;
                }
                if invalid_access {
                    let offset = if helper == Some(18) { 0 } else { 4 };
                    let target = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: pointer,
                        index: None,
                        stride: 1,
                        offset,
                        result_type: MockType::Ptr,
                    })?;
                    b.load_indirect(tmp(MockType::U8), target, MockType::U8)?;
                }
                b.call_void(analyssa::ir::Callee::Direct(99), vec![pointer])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let host = AllocatorHost;
        let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let found: Vec<FindingKind> = facts
            .findings()
            .iter()
            .filter(|found| found.block == 1)
            .map(|found| found.kind)
            .collect();
        assert_eq!(found, finding.into_iter().collect::<Vec<_>>());
        let consumer = facts
            .calls()
            .into_iter()
            .filter(|(site, _)| site.0 == 1)
            .max_by_key(|(site, _)| site.1)
            .map(|(_, arguments)| arguments)
            .unwrap_or_default();
        if expected.is_empty() {
            assert!(consumer.is_empty());
        } else {
            assert_eq!(
                consumer.first().map(|bytes| bytes.bytes.as_slice()),
                Some(expected)
            );
        }
    }
}

/// The reallocator [`UnstatedHost`] models: `realloc(pointer, n)`.
const REALLOC: LibraryModel = LibraryModel::Reallocate(Reallocation {
    family: AllocatorFamily(1),
    pointer: 0,
    size: 1,
    count: None,
    flags: None,
    null_allocates: true,
    heap: None,
});

/// The duplicator [`UnstatedHost`] models: `strdup(source)`.
const STRDUP: LibraryModel = LibraryModel::Duplicate(Duplication {
    family: AllocatorFamily(1),
    source: 0,
    limit: None,
});

/// Models `malloc` (17), `realloc` (21) and `strdup` (22).
struct UnstatedHost;

impl ExecutionHost<MockTarget> for UnstatedHost {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match method {
            17 => Some(&ZEROED),
            21 => Some(&REALLOC),
            22 => Some(&STRDUP),
            _ => None,
        }
    }
}

/// **A block whose size its call does not state has no extent to be past.**
/// `malloc` of a parameter, `realloc` of a null pointer to a parameter's
/// size and `strdup` of a string the run cannot read each make a block no
/// byte of which is known to belong to it, so a store at its ninth byte is
/// no finding and the run completes. The control: `malloc(8)` and
/// `realloc(NULL, 8)` state eight bytes, and the same store is past them.
#[test]
fn a_block_of_unstated_size_has_no_extent_to_be_past() {
    for (method, stated, found) in [
        (17, false, false),
        (21, false, false),
        (22, false, false),
        (17, true, true),
        (21, true, true),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let parameter = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::U64,
        );
        let pointer = builder
            .in_block(0, |b| {
                let size = if stated {
                    b.const_value(tmp(MockType::U64), ConstValue::U64(8))?
                } else {
                    parameter
                };
                let null = b.const_value(tmp(MockType::Ptr), ConstValue::U64(0))?;
                let arguments = match method {
                    21 => vec![null, size],
                    22 => vec![parameter],
                    _ => vec![size],
                };
                let pointer = b.call(
                    tmp(MockType::Ptr),
                    analyssa::ir::Callee::Direct(method),
                    arguments,
                )?;
                b.branch(pointer, 1, 2)?;
                Ok(pointer)
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let ninth = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: pointer,
                    index: None,
                    stride: 1,
                    offset: 8,
                    result_type: MockType::Ptr,
                })?;
                let contents = b.const_value(tmp(MockType::U8), ConstValue::U8(42))?;
                b.store_indirect(ninth, contents, MockType::U8)?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&UnstatedHost)
            .run();
        assert!(
            facts.is_complete(),
            "{method} {stated}: {:?}",
            facts.completion()
        );
        let past: Vec<FindingKind> = facts
            .findings()
            .iter()
            .map(|found| found.kind)
            .filter(|kind| *kind == FindingKind::PastExtent(ExtentSource::Allocator))
            .collect();
        assert_eq!(!past.is_empty(), found, "{method} {stated}: {past:?}");
    }
}

/// What a deallocation passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Freed {
    /// A null pointer.
    Null,
    /// A parameter, an address the execution cannot name.
    Unknown,
    /// The allocation, a second time.
    Twice,
    /// One byte past the allocation's start.
    Interior,
}

/// **The free model's effect is what the pointer names.** Freeing null
/// changes nothing, and the live allocation's bytes reach a later consumer;
/// freeing an address the execution cannot name may end any object, so the
/// allocation's bytes no longer reach it, and the run continues; freeing the
/// allocation twice, or from its interior, is a deallocation no allocator
/// defines: a finding at the call, which ends nothing, and the run continues,
/// the twice-freed block ended and the interior-freed one live.
#[test]
fn a_free_model_acts_on_what_its_pointer_names() {
    for freed in [Freed::Null, Freed::Unknown, Freed::Twice, Freed::Interior] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let parameter = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let pointer = builder
            .in_block(0, |b| {
                let size = b.const_value(tmp(MockType::U64), ConstValue::U64(4))?;
                let pointer = b.call(tmp(MockType::Ptr), Callee::Direct(17), vec![size])?;
                b.branch(pointer, 1, 2)?;
                Ok(pointer)
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let contents = b.const_value(tmp(MockType::U8), ConstValue::U8(42))?;
                b.store_indirect(pointer, contents, MockType::U8)?;
                let freed_pointer = match freed {
                    Freed::Null => b.const_value(tmp(MockType::Ptr), ConstValue::U64(0))?,
                    Freed::Unknown => parameter,
                    Freed::Twice => {
                        b.call_void(Callee::Direct(18), vec![pointer])?;
                        pointer
                    }
                    Freed::Interior => b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: pointer,
                        index: None,
                        stride: 1,
                        offset: 1,
                        result_type: MockType::Ptr,
                    })?,
                };
                b.call_void(Callee::Direct(18), vec![freed_pointer])?;
                b.call_void(Callee::Direct(99), vec![pointer])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&AllocatorHost)
            .run();
        assert!(facts.is_complete(), "{freed:?}: {:?}", facts.completion());
        let consumer = facts
            .calls()
            .into_iter()
            .filter(|(site, _)| site.0 == 1)
            .max_by_key(|(site, _)| site.1)
            .map(|(_, arguments)| arguments)
            .unwrap_or_default();
        let reached = consumer.first().map(|bytes| bytes.bytes.clone());
        let (live, fault) = match freed {
            Freed::Null => (true, None),
            Freed::Unknown => (false, None),
            Freed::Twice => (false, Some(ReleaseFault::Repeated)),
            Freed::Interior => (true, Some(ReleaseFault::Interior)),
        };
        assert_eq!(reached, live.then(|| b"*\0\0\0".to_vec()), "{freed:?}");
        let found: Vec<FindingKind> = facts.findings().iter().map(|found| found.kind).collect();
        assert_eq!(
            found,
            fault
                .map(FindingKind::Release)
                .into_iter()
                .collect::<Vec<_>>(),
            "{freed:?}"
        );
    }
}

/// A heap load is proven only inside a live allocation, at an alignment that
/// cannot fault: a byte of it is, and a wider read whose alignment the
/// allocation does not state is not, though both read known bytes.
#[test]
fn heap_loads_are_proven_inside_a_live_allocation() {
    for (ty, expected, proven) in [
        (MockType::U8, ConstValue::U8(42), true),
        (MockType::U16, ConstValue::U16(42), false),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let pointer = builder
            .in_block(0, |b| {
                let size = b.const_value(tmp(MockType::U64), ConstValue::U64(4))?;
                let pointer = b.call(tmp(MockType::Ptr), Callee::Direct(17), vec![size])?;
                b.branch(pointer, 1, 2)?;
                Ok(pointer)
            })
            .unwrap_or_else(abort);
        let loaded = builder
            .in_block(1, |b| {
                let contents = b.const_value(tmp(MockType::U8), ConstValue::U8(42))?;
                b.store_indirect(pointer, contents, MockType::U8)?;
                let loaded = b.load_indirect(tmp(ty), pointer, ty)?;
                b.call_void(Callee::Direct(99), vec![loaded])?;
                b.ret(None)?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        });
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&AllocatorHost)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(facts.constant(loaded), Some(&expected));
        assert_eq!(
            facts.replacement_proof(loaded),
            proven.then_some(ReplacementProof::Accessible)
        );
    }
}

/// A callee trap terminates its caller path without confusing overlapping SSA value IDs.
#[test]
fn callee_overflow_does_not_return_or_poison_caller_replacement_proofs() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let left = b.const_value(tmp(MockType::U8), ConstValue::U8(255))?;
            let right = b.const_value(tmp(MockType::U8), ConstValue::U8(1))?;
            b.emit_def(tmp(MockType::U8), |dest| SsaOp::AddOvf {
                dest,
                left,
                right,
                unsigned: true,
                flags: None,
            })?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let host = Host {
        callee: builder.finish().unwrap_or_else(abort),
    };
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let checked = builder
        .in_block(0, |b| {
            let left = b.const_value(tmp(MockType::U8), ConstValue::U8(2))?;
            let right = b.const_value(tmp(MockType::U8), ConstValue::U8(3))?;
            let checked = b.emit_def(tmp(MockType::U8), |dest| SsaOp::AddOvf {
                dest,
                left,
                right,
                unsigned: true,
                flags: None,
            })?;
            b.call_void(analyssa::ir::Callee::Direct(CALLEE), vec![])?;
            b.call_void(analyssa::ir::Callee::Direct(99), vec![checked])?;
            b.ret(None)?;
            Ok(checked)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
    assert!(facts.is_complete());
    assert_eq!(facts.calls().len(), 1);
    assert_eq!(facts.usage().traps, 1);
    assert_eq!(facts.constant(checked), Some(&ConstValue::U8(5)));
    assert_eq!(
        facts.replacement_proof(checked),
        Some(ReplacementProof::Completed)
    );
}

/// The same SSA body may be entered under distinct method identities.
struct AliasedBodies<'a> {
    shared: &'a SsaFunction<MockTarget>,
    leaf: &'a SsaFunction<MockTarget>,
    callers: RefCell<Vec<u32>>,
}
impl ExecutionHost<MockTarget> for AliasedBodies<'_> {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        match method {
            1 | 2 => Some(self.shared),
            4 => Some(self.leaf),
            _ => None,
        }
    }
    fn call_stack(&self, site: CallSite<'_, MockTarget>) -> Option<CallStack> {
        if let Some(method) = site.caller_method
            && [1, 2].contains(method)
        {
            self.callers.borrow_mut().push(*method);
        }
        None
    }
}

/// A geometry cache must preserve the method identity supplied to nested host bindings.
#[test]
fn body_aliases_keep_distinct_caller_method_contexts() {
    let mut root = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    root.in_block(0, |b| {
        b.call_void(analyssa::ir::Callee::Direct(1), vec![])?;
        b.call_void(analyssa::ir::Callee::Direct(2), vec![])?;
        b.ret(None)
    })
    .unwrap_or_else(|_| process::abort());
    let root = root.finish().unwrap_or_else(|_| process::abort());
    let mut shared = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    shared
        .in_block(0, |b| {
            b.call_void(analyssa::ir::Callee::Direct(4), vec![])?;
            b.ret(None)
        })
        .unwrap_or_else(|_| process::abort());
    let shared = shared.finish().unwrap_or_else(|_| process::abort());
    let mut leaf = SsaFunctionBuilder::<MockTarget>::new(4, 0);
    leaf.in_block(0, |b| b.ret(None))
        .unwrap_or_else(|_| process::abort());
    let leaf = leaf.finish().unwrap_or_else(|_| process::abort());
    let host = AliasedBodies {
        shared: &shared,
        leaf: &leaf,
        callers: RefCell::new(Vec::new()),
    };
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_method(0)
        .run();
    assert!(facts.is_complete());
    assert_eq!(*host.callers.borrow(), [1, 2]);
}

/// Maps two aliases to one middle body and one identity to its leaf.
struct BodyProvenanceHost {
    middle: SsaFunction<MockTarget>,
    leaf: SsaFunction<MockTarget>,
    entered: Cell<usize>,
}

impl ExecutionHost<MockTarget> for BodyProvenanceHost {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        let body = match *method {
            101 | 102 => Some(&self.middle),
            203 => Some(&self.leaf),
            _ => None,
        };
        if body.is_some() {
            self.entered.set(self.entered.get().saturating_add(1));
        }
        body
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 666).then_some(&LibraryModel::NoWrites as &dyn CallModel<MockTarget>)
    }
}

/// Builds repeated, aliased, transitive, modelled and opaque calls with known results.
fn body_provenance_fixture() -> (SsaFunction<MockTarget>, BodyProvenanceHost) {
    let mut leaf = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    leaf.in_block(0, |b| {
        let nine = b.const_value(tmp(MockType::I32), ConstValue::I32(9))?;
        b.ret(Some(nine))
    })
    .unwrap_or_else(abort);
    let mut middle = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    middle
        .in_block(0, |b| {
            let nine = b.call(
                tmp(MockType::I32),
                analyssa::ir::Callee::Direct(203),
                vec![],
            )?;
            b.ret(Some(nine))
        })
        .unwrap_or_else(abort);
    let mut root = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    root.in_block(0, |b| {
        for method in [101, 101, 102, 666, 999] {
            b.call_void(analyssa::ir::Callee::Direct(method), vec![])?;
        }
        let nine = b.const_value(tmp(MockType::I32), ConstValue::I32(9))?;
        b.ret(Some(nine))
    })
    .unwrap_or_else(abort);
    (
        root.finish().unwrap_or_else(abort),
        BodyProvenanceHost {
            middle: middle.finish().unwrap_or_else(abort),
            leaf: leaf.finish().unwrap_or_else(abort),
            entered: Cell::new(0),
        },
    )
}

/// Provenance follows actual transitive bodies and keeps aliases without inventing opaque bodies.
#[test]
fn entered_body_provenance_distinguishes_aliases_and_moves_its_allocation() {
    let (root, host) = body_provenance_fixture();
    let context = AnalysisContext::new(MACHINE);
    let mut facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_method(777)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::I32(9))));
    let expected = [Some(777), Some(101), Some(203), Some(102)];
    assert_eq!(facts.body_methods(), expected);
    assert!(host.entered.get() > expected.len());
    let allocation = facts.body_methods().as_ptr();
    let methods = facts.take_body_methods();
    assert_eq!(methods, expected);
    assert_eq!(methods.as_ptr(), allocation);
    assert!(facts.body_methods().is_empty());
}

/// Exhaustion after body resolution withholds the entire provenance table.
#[test]
fn entered_body_provenance_is_withheld_on_execution_refusal() {
    let (root, host) = body_provenance_fixture();
    let context = AnalysisContext::new(MACHINE)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionSteps, 2));
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_method(777)
        .run();
    assert!(matches!(facts.completion(), Completion::Exhausted(_)));
    assert!(host.entered.get() > 0);
    assert!(facts.body_methods().is_empty());
}

/// A read-only region containing a binary object immediately followed by known neighbors.
struct ObservationImage;

const OBSERVATION_BYTES: &[u8] = b"A\0\xffBneighboring known bytes";

impl ImageData for ObservationImage {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
    fn region(&self, address: u64) -> Option<ImageRegion> {
        (SOURCE_AT..SOURCE_AT.wrapping_add(OBSERVATION_BYTES.len() as u64))
            .contains(&address)
            .then_some(ImageRegion {
                start: SOURCE_AT,
                end: SOURCE_AT.wrapping_add(OBSERVATION_BYTES.len() as u64),
                mutability: Mutability::ReadOnly,
            })
    }
    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let offset = usize::try_from(address.checked_sub(SOURCE_AT)?).ok()?;
        OBSERVATION_BYTES.get(offset..offset.checked_add(len)?)
    }
}

/// Builds one unmodelled call whose pointer use is described independently of its bytes.
fn bounded_observation_fixture(
    kind: u8,
    effects: Option<CallEffects<MockTarget>>,
    length: Option<u64>,
) -> (SsaFunction<MockTarget>, usize) {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let mut site = 0;
    builder
        .in_block(0, |block| {
            let pointer = block.const_value(tmp(MockType::U64), ConstValue::U64(SOURCE_AT))?;
            let length = match length {
                Some(length) => block.const_value(tmp(MockType::U64), ConstValue::U64(length))?,
                None => block.load_arg(SsaDefSpec::argument(0, MockType::U64), 0)?,
            };
            let function = block.const_value(tmp(MockType::U64), ConstValue::U64(0x6000))?;
            site = 3;
            let args = vec![pointer, length];
            let effects = effects.map(Box::new);
            block.emit_no_defs(match kind {
                0 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: None,
                    callee: analyssa::ir::Callee::Direct(CALLEE),
                    args,
                    effects: (effects).map(|effects| *effects),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })),
                1 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: None,
                    callee: analyssa::ir::Callee::Virtual(CALLEE),
                    args,
                    effects: (effects).map(|effects| *effects),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })),
                _ => SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: None,
                    callee: analyssa::ir::Callee::Indirect {
                        fptr: function,
                        signature: 0,
                    },
                    args,
                    effects: (effects).map(|effects| *effects),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })),
            })?;
            block.ret(None)
        })
        .unwrap_or_else(abort);
    (builder.finish().unwrap_or_else(abort), site)
}

/// Checks exact binary bytes rather than a successful completion or payload count alone.
fn bounded_observation_case(
    kind: u8,
    extent: PointeeExtent,
    length: Option<u64>,
    expected: &[u8],
    bound: ByteBound,
) {
    let effects =
        CallEffects::pure().with_access(CallLocation::Pointee { arg: 0, extent }, AccessMode::Read);
    observation_effects_case(kind, Some(effects), length, expected, bound);
}

/// A use the call's contract bounds to `bytes`.
const fn contract(bytes: u64) -> ByteBound {
    ByteBound::Use {
        bytes,
        by: UseBound::Contract,
    }
}

/// Checks a complete use contract without relying on the projection algorithm.
fn observation_effects_case(
    kind: u8,
    effects: Option<CallEffects<MockTarget>>,
    length: Option<u64>,
    expected: &[u8],
    bound: ByteBound,
) {
    let (ssa, site) = bounded_observation_fixture(kind, effects, length);
    let image = ObservationImage;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let seen = facts.call_arguments(0, site);
    if expected.is_empty() {
        assert!(seen.is_empty());
    } else {
        assert_eq!(seen.len(), 1);
        let argument = seen.first().unwrap_or_else(|| process::abort());
        assert_eq!(argument.argument, 0);
        assert_eq!(argument.address.image_address(), Some(SOURCE_AT));
        assert_eq!(argument.bytes, expected);
        assert_eq!(argument.bound, bound);
    }
}

/// A direct consumer's fixed byte use excludes neighboring objects.
#[test]
fn observation_bound_direct_fixed_extent() {
    bounded_observation_case(
        0,
        PointeeExtent::Bytes(4),
        Some(4),
        b"A\0\xffB",
        contract(4),
    );
}

/// A virtual call retains its call-site use contract without guessing its implementation.
#[test]
fn observation_bound_virtual_fixed_extent() {
    bounded_observation_case(
        1,
        PointeeExtent::Bytes(4),
        Some(4),
        b"A\0\xffB",
        contract(4),
    );
}

/// An indirect call uses the same stated pointer extent as a named call.
#[test]
fn observation_bound_indirect_fixed_extent() {
    bounded_observation_case(
        2,
        PointeeExtent::Bytes(4),
        Some(4),
        b"A\0\xffB",
        contract(4),
    );
}

/// A direct call's resolved length argument bounds its binary input.
#[test]
fn observation_bound_direct_length_argument() {
    bounded_observation_case(
        0,
        PointeeExtent::InArgument(1),
        Some(3),
        b"A\0\xff",
        contract(3),
    );
}

/// A virtual call's resolved length argument excludes adjacent known bytes.
#[test]
fn observation_bound_virtual_length_argument() {
    bounded_observation_case(
        1,
        PointeeExtent::InArgument(1),
        Some(3),
        b"A\0\xff",
        contract(3),
    );
}

/// An indirect call's length argument has the same byte-count interpretation.
#[test]
fn observation_bound_indirect_length_argument() {
    bounded_observation_case(
        2,
        PointeeExtent::InArgument(1),
        Some(3),
        b"A\0\xff",
        contract(3),
    );
}

/// Empty uses read no pointed-to bytes.
#[test]
fn observation_bound_zero_extent_reads_no_bytes() {
    for kind in 0..3 {
        bounded_observation_case(
            kind,
            PointeeExtent::InArgument(1),
            Some(0),
            &[],
            contract(0),
        );
    }
}

/// Unknown and unbounded uses retain prefix semantics without guessing a complete buffer.
#[test]
fn observation_bound_unknown_lengths_keep_known_prefixes() {
    for kind in 0..3 {
        bounded_observation_case(
            kind,
            PointeeExtent::InArgument(1),
            None,
            OBSERVATION_BYTES,
            ByteBound::Prefix,
        );
        bounded_observation_case(
            kind,
            PointeeExtent::Unbounded,
            Some(4),
            OBSERVATION_BYTES,
            ByteBound::Prefix,
        );
    }
}

/// Multiple bounded uses cover their union rather than the shortest declared use.
#[test]
fn observation_bound_multiple_extents_take_the_union() {
    for reverse in [false, true] {
        let extents = if reverse { [6, 2] } else { [2, 6] };
        let mut effects = CallEffects::pure();
        for bytes in extents {
            effects = effects.with_access(
                CallLocation::Pointee {
                    arg: 0,
                    extent: PointeeExtent::Bytes(bytes),
                },
                AccessMode::Read,
            );
        }
        observation_effects_case(0, Some(effects), Some(4), b"A\0\xffBne", contract(6));
    }
}

/// An unbounded use cannot be hidden by another, narrower use of the same argument.
#[test]
fn observation_bound_unbounded_use_dominates_in_either_order() {
    for reverse in [false, true] {
        let extents = if reverse {
            [PointeeExtent::Unbounded, PointeeExtent::Bytes(2)]
        } else {
            [PointeeExtent::Bytes(2), PointeeExtent::Unbounded]
        };
        let mut effects = CallEffects::pure();
        for extent in extents {
            effects =
                effects.with_access(CallLocation::Pointee { arg: 0, extent }, AccessMode::Read);
        }
        observation_effects_case(
            0,
            Some(effects),
            Some(4),
            OBSERVATION_BYTES,
            ByteBound::Prefix,
        );
    }
}

/// Callbacks, escaped memory and malformed argument indices establish no use bound.
#[test]
fn observation_bound_incomplete_contracts_keep_prefixes() {
    for case in 0..6 {
        let mut effects = CallEffects::pure().with_access(
            CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::Bytes(2),
            },
            AccessMode::Read,
        );
        match case {
            0 => effects.calls_back = true,
            1 => effects = effects.with_access(CallLocation::Escaped, AccessMode::Update),
            2 => {
                effects = effects.with_access(
                    CallLocation::Pointee {
                        arg: 99,
                        extent: PointeeExtent::Bytes(2),
                    },
                    AccessMode::Read,
                )
            }
            3 => {
                effects = CallEffects::pure().with_access(
                    CallLocation::Pointee {
                        arg: 0,
                        extent: PointeeExtent::InArgument(99),
                    },
                    AccessMode::Read,
                )
            }
            4 => effects = CallEffects::unknown(),
            _ => effects = effects.with_access(CallLocation::Field(0), AccessMode::Read),
        }
        observation_effects_case(
            0,
            Some(effects),
            Some(4),
            OBSERVATION_BYTES,
            ByteBound::Prefix,
        );
    }
    observation_effects_case(0, None, Some(4), OBSERVATION_BYTES, ByteBound::Prefix);
}

/// Oversized bounds do not synthesize bytes beyond the independently available data.
#[test]
fn observation_bound_large_use_keeps_only_known_bytes() {
    for kind in 0..3 {
        bounded_observation_case(
            kind,
            PointeeExtent::Bytes(u64::MAX),
            Some(4),
            OBSERVATION_BYTES,
            contract(u64::MAX),
        );
    }
}

/// Supplies a zeroing allocation while leaving the consumer unmodelled.
struct ObservationAllocator;

impl CallModel<MockTarget> for ObservationAllocator {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let pointer = call.allocate_block(Some(3), true, None);
        call.return_value(Value::Addr(pointer));
    }
    fn only_memory(&self) -> bool {
        true
    }
}

impl ExecutionHost<MockTarget> for ObservationAllocator {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        if *method == 9 { Some(self) } else { None }
    }
}

/// A known allocation extent remains a bound when a consumer declares a larger use.
#[test]
fn observation_bound_heap_extent_stops_the_known_prefix() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |block| {
            let size = block.const_value(tmp(MockType::U64), ConstValue::U64(3))?;
            let pointer = block.call(
                tmp(MockType::Ptr),
                analyssa::ir::Callee::Direct(9),
                vec![size],
            )?;
            let effects = CallEffects::pure().with_access(
                CallLocation::Pointee {
                    arg: 0,
                    extent: PointeeExtent::Bytes(4096),
                },
                AccessMode::Read,
            );
            block.emit_no_defs(SsaOp::Call(Box::new(analyssa::ir::CallData {
                dest: None,
                callee: analyssa::ir::Callee::Direct(CALLEE),
                args: vec![pointer],
                effects: Some(effects),
                tail: false,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: Vec::new(),
            })))?;
            block.ret(None)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context)
        .with_host(&ObservationAllocator)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let seen = facts.call_arguments(0, 2);
    assert_eq!(seen.len(), 1);
    let argument = seen.first().unwrap_or_else(|| process::abort());
    assert_eq!(argument.argument, 0);
    assert_eq!(argument.bytes, [0, 0, 0]);
    assert_eq!(
        argument.bound,
        ByteBound::Object {
            start: 0,
            bytes: 3,
            kind: ObjectKind::Heap,
        }
    );
    assert!(argument.complete(), "the whole allocation is known");
    assert_eq!(
        argument.allocated_at,
        Some((0, 1)),
        "the call that allocated it is the root's second instruction"
    );
}

/// A table of one object outside the heap.
#[derive(Debug)]
struct OneObject(StorageObject);

impl ObjectExtents for OneObject {
    fn object(&self, address: Address) -> Option<StorageObject> {
        self.0.reach(address).map(|_| self.0)
    }
}

/// States the image object the observed pointer lies in.
struct ObservationObject(StorageObject);

impl ExecutionHost<MockTarget> for ObservationObject {
    fn object_extents(
        &self,
        _root: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Arc<dyn ObjectExtents>> {
        Some(Arc::new(OneObject(self.0)))
    }
}

/// **An object the host proves stops the run at its end, and a use within
/// it is tighter still.** The image holds `A\0\xffB` and known neighbours
/// past it; the host states a two-byte object at the pointer. Without a
/// contract the run is the object's two bytes, whole; a contract of four
/// bytes reaches past the object, which still bounds it; a contract of one
/// byte is tighter than the object.
#[test]
fn an_object_the_host_proves_bounds_the_observed_run() {
    let object = StorageObject {
        start: i64::try_from(SOURCE_AT).unwrap_or_else(|_| process::abort()),
        bytes: 2,
        kind: ObjectKind::Global,
        source: ExtentSource::Symbol,
    };
    let global = ByteBound::Object {
        start: object.start,
        bytes: 2,
        kind: ObjectKind::Global,
    };
    let read = |bytes| {
        Some(CallEffects::pure().with_access(
            CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::Bytes(bytes),
            },
            AccessMode::Read,
        ))
    };
    for (effects, expected, bound) in [
        (None, &b"A\0"[..], global),
        (read(4), &b"A\0"[..], global),
        (read(1), &b"A"[..], contract(1)),
    ] {
        let (ssa, site) = bounded_observation_fixture(0, effects, Some(4));
        let image = ObservationImage;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let host = ObservationObject(object);
        let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let seen = facts.call_arguments(0, site);
        assert_eq!(seen.len(), 1, "{bound:?}");
        let argument = seen.first().unwrap_or_else(|| process::abort());
        assert_eq!(argument.bytes, expected, "{bound:?}");
        assert_eq!(argument.bound, bound);
        assert!(argument.complete(), "{bound:?}: the whole of its bound");
    }
}

/// **An access past an object's end is a finding of its extent's source.**
/// The host states a two-byte image object at the image's first byte. A
/// four-byte load there runs past it: with a symbol's extent the run finds an
/// access past an exact extent, with a recovered one past an inferred extent,
/// and either way it goes on and completes. A two-byte load inside the
/// object finds nothing, whatever its extent's source.
#[test]
fn an_access_past_an_extent_is_found_by_its_source() {
    for (source, ty, past) in [
        (ExtentSource::Symbol, MockType::U32, true),
        (ExtentSource::Recovered, MockType::U32, true),
        (ExtentSource::Symbol, MockType::U16, false),
        (ExtentSource::Recovered, MockType::U16, false),
    ] {
        let object = StorageObject {
            start: i64::try_from(SOURCE_AT).unwrap_or_else(|_| process::abort()),
            bytes: 2,
            kind: ObjectKind::Global,
            source,
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |b| {
                let pointer = b.const_value(tmp(MockType::U64), ConstValue::U64(SOURCE_AT))?;
                let loaded = b.load_indirect(tmp(ty), pointer, ty)?;
                b.ret(Some(loaded))
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let image = ObservationImage;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let host = ObservationObject(object);
        let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
        assert!(
            facts.is_complete(),
            "{source:?} {ty:?}: {:?}",
            facts.completion()
        );
        // The finding carries the four bytes the load reached and the object
        // it ran past, on the one path the function has.
        let found: Vec<_> = facts
            .findings()
            .iter()
            .map(|found| {
                (
                    (found.body, found.block, found.instruction, found.kind),
                    found.access.map(|(at, len)| (at.root(), at.offset(), len)),
                    found.object.clone(),
                    found.witness.clone(),
                )
            })
            .collect();
        let expected: Vec<_> = past
            .then_some((
                (0, 0, 1, FindingKind::PastExtent(source)),
                Some((Root::Symbol, object.start, 4)),
                Some(FoundObject::Stated {
                    root: Root::Symbol,
                    object,
                }),
                Witness::default(),
            ))
            .into_iter()
            .collect();
        assert_eq!(found, expected, "{source:?} {ty:?}");
    }
}

/// **A recovered extent bounds no observation.** The host states the same
/// two-byte object, recovered rather than read from a symbol: the observed
/// run behind the pointer is not stopped at its end, since nothing states
/// the object ends there.
#[test]
fn a_recovered_extent_bounds_no_observed_run() {
    let object = StorageObject {
        start: i64::try_from(SOURCE_AT).unwrap_or_else(|_| process::abort()),
        bytes: 2,
        kind: ObjectKind::Global,
        source: ExtentSource::Recovered,
    };
    let (ssa, site) = bounded_observation_fixture(0, None, Some(4));
    let image = ObservationImage;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    let host = ObservationObject(object);
    let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let seen = facts.call_arguments(0, site);
    let argument = seen.first().unwrap_or_else(|| process::abort());
    assert!(
        !matches!(argument.bound, ByteBound::Object { .. }),
        "{:?}",
        argument.bound
    );
}

/// A complete no-memory contract provides no pointed-to input use.
#[test]
fn observation_bound_no_memory_contract_has_no_input_bytes() {
    for kind in 0..3 {
        observation_effects_case(
            kind,
            Some(CallEffects::pure()),
            Some(4),
            &[],
            ByteBound::Prefix,
        );
    }
}

/// A numeric length equal to an image address is not another pointer use.
#[test]
fn observation_bound_unlisted_scalar_arguments_have_no_input_bytes() {
    for kind in 0..3 {
        bounded_observation_case(
            kind,
            PointeeExtent::Bytes(4),
            Some(SOURCE_AT),
            b"A\0\xffB",
            contract(4),
        );
    }
}

/// An invalid count index invalidates no-use assumptions for the whole contract.
#[test]
fn observation_bound_invalid_count_index_discards_the_contract() {
    for kind in 0..3 {
        let effects = CallEffects::pure().with_access(
            CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::InArgument(99),
            },
            AccessMode::Read,
        );
        let (ssa, site) = bounded_observation_fixture(kind, Some(effects), Some(SOURCE_AT));
        let image = ObservationImage;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let seen = facts.call_arguments(0, site);
        assert_eq!(seen.len(), 2);
        for (index, argument) in seen.iter().enumerate() {
            assert_eq!(argument.argument, index);
            assert_eq!(argument.bytes, OBSERVATION_BYTES);
        }
    }
}

/// The contract or alias path exercised by the unanswered-call fixture.
#[derive(Clone, Copy)]
enum UnansweredCase {
    Pure,
    ReturnStore,
    CaptureThenUnknown,
    StackTarget,
}

/// Builds an unanswered call between one byte store and its independent reload.
fn pure_unanswered_fixture(
    kind: u8,
    storage: u8,
    case: UnansweredCase,
) -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .unwrap_or_else(|| process::abort())
        .set_role(VariableRole::EntryStackPointer);
    let returned = builder
        .in_block(0, |block| {
            let pointer = match storage {
                0 => block.const_value(tmp(MockType::Ptr), ConstValue::U64(TARGET_AT))?,
                1 => block.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -16,
                    result_type: MockType::Ptr,
                })?,
                _ => {
                    let size = block.const_value(tmp(MockType::U64), ConstValue::U64(3))?;
                    block.call(
                        tmp(MockType::Ptr),
                        analyssa::ir::Callee::Direct(9),
                        vec![size],
                    )?
                }
            };
            let byte = block.const_value(tmp(MockType::U8), ConstValue::U8(0x96))?;
            block.store_indirect(pointer, byte, MockType::U8)?;
            let function = block.const_value(tmp(MockType::Ptr), ConstValue::U64(0x6000))?;
            let mut contract = CallEffects::pure();
            if matches!(case, UnansweredCase::CaptureThenUnknown) {
                contract.captures = Capture::Any;
            }
            let effects = Some(Box::new(contract));
            let args = if matches!(case, UnansweredCase::StackTarget) {
                Vec::new()
            } else {
                vec![pointer]
            };
            let function = if matches!(case, UnansweredCase::StackTarget) {
                pointer
            } else {
                function
            };
            let result = block.emit_def(tmp(MockType::Ptr), |dest| match kind {
                0 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: Some(dest),
                    callee: analyssa::ir::Callee::Direct(CALLEE),
                    args,
                    effects: (effects).map(|effects| *effects),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })),
                1 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: Some(dest),
                    callee: analyssa::ir::Callee::Virtual(CALLEE),
                    args,
                    effects: (effects).map(|effects| *effects),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })),
                _ => SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: Some(dest),
                    callee: analyssa::ir::Callee::Indirect {
                        fptr: function,
                        signature: 0,
                    },
                    args,
                    effects: (effects).map(|effects| *effects),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })),
            })?;
            if matches!(
                case,
                UnansweredCase::ReturnStore | UnansweredCase::StackTarget
            ) {
                let other = block.const_value(tmp(MockType::U8), ConstValue::U8(0x27))?;
                block.store_indirect(result, other, MockType::U8)?;
            }
            if matches!(case, UnansweredCase::CaptureThenUnknown) {
                block.call_void(analyssa::ir::Callee::Direct(SECOND), vec![])?;
            }
            let loaded = block.load_indirect(tmp(MockType::U8), pointer, MockType::U8)?;
            block.ret(Some(loaded))?;
            Ok(result)
        })
        .unwrap_or_else(abort);
    (builder.finish().unwrap_or_else(abort), returned)
}

/// Pure direct calls preserve globals, private stack bytes and allocation lifetimes.
#[test]
fn pure_unanswered_direct_call_preserves_state() {
    pure_unanswered_state_case(0);
}

/// A call-site purity promise applies to an unanswered virtual call without choosing an override.
#[test]
fn pure_unanswered_virtual_call_preserves_state() {
    pure_unanswered_state_case(1);
}

/// A pure indirect call does not havoc memory just because its target is unresolved.
#[test]
fn pure_unanswered_indirect_call_preserves_state() {
    pure_unanswered_state_case(2);
}

/// Checks every storage domain against an independent byte value.
fn pure_unanswered_state_case(kind: u8) {
    for storage in 0..3 {
        let (ssa, _) = pure_unanswered_fixture(kind, storage, UnansweredCase::Pure);
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&ObservationAllocator)
            .run();
        assert!(
            facts.is_complete(),
            "kind={kind} storage={storage} {:?}",
            facts.completion()
        );
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::U8(0x96)),
            "kind={kind} storage={storage}"
        );
    }
}

/// A pure call can return an argument alias without publishing it elsewhere.
#[test]
fn pure_unanswered_return_keeps_stack_alias_provenance() {
    for kind in 0..3 {
        let (ssa, returned) = pure_unanswered_fixture(kind, 1, UnansweredCase::ReturnStore);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(
            facts
                .value(returned)
                .is_some_and(Value::may_reference_stack)
        );
        assert!(facts.returned().and_then(Value::constant).is_none());
    }
}

/// A capture cannot hide a private buffer from a later unknown call.
#[test]
fn pure_unanswered_capture_then_unknown_call_forgets_private_data() {
    for kind in 0..3 {
        let (ssa, _) = pure_unanswered_fixture(kind, 1, UnansweredCase::CaptureThenUnknown);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(facts.returned().and_then(Value::constant).is_none());
    }
}

/// A stack-derived target can return stack bits even when it receives no pointer argument.
#[test]
fn pure_unanswered_stack_target_keeps_return_provenance() {
    let (ssa, returned) = pure_unanswered_fixture(2, 1, UnansweredCase::StackTarget);
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(
        facts
            .value(returned)
            .is_some_and(Value::may_reference_stack)
    );
    assert!(facts.returned().and_then(Value::constant).is_none());
}

/// A known model computes a result before any unanswered-call fallback is considered.
struct PureKnownModel;

impl CallModel<MockTarget> for PureKnownModel {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        call.return_value(Value::Const(ConstValue::U64(0x45)));
    }
    fn only_memory(&self) -> bool {
        true
    }
}

impl ExecutionHost<MockTarget> for PureKnownModel {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == CALLEE).then_some(self)
    }
    fn dispatch(&self, method: &u32, _: &Value<MockTarget>) -> Option<u32> {
        Some(*method)
    }
    fn method_at(&self, address: u64) -> Option<u32> {
        (address == 0x6000).then_some(CALLEE)
    }
}

/// Named, virtual and indirect call models retain precedence over a purity promise.
#[test]
fn pure_unanswered_known_models_take_precedence() {
    for kind in 0..3 {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |block| {
                let receiver = block.const_value(tmp(MockType::Ptr), ConstValue::U64(0x3000))?;
                let function = block.const_value(tmp(MockType::Ptr), ConstValue::U64(0x6000))?;
                let args = vec![receiver];
                let effects = Some(Box::new(CallEffects::pure()));
                let result = block.emit_def(tmp(MockType::U64), |dest| match kind {
                    0 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                        dest: Some(dest),
                        callee: analyssa::ir::Callee::Direct(CALLEE),
                        args,
                        effects: (effects).map(|effects| *effects),
                        tail: false,
                        implicit_inputs: Default::default(),
                        results: Default::default(),
                        staged: Vec::new(),
                    })),
                    1 => SsaOp::Call(Box::new(analyssa::ir::CallData {
                        dest: Some(dest),
                        callee: analyssa::ir::Callee::Virtual(CALLEE),
                        args,
                        effects: (effects).map(|effects| *effects),
                        tail: false,
                        implicit_inputs: Default::default(),
                        results: Default::default(),
                        staged: Vec::new(),
                    })),
                    _ => SsaOp::Call(Box::new(analyssa::ir::CallData {
                        dest: Some(dest),
                        callee: analyssa::ir::Callee::Indirect {
                            fptr: function,
                            signature: 0,
                        },
                        args,
                        effects: (effects).map(|effects| *effects),
                        tail: false,
                        implicit_inputs: Default::default(),
                        results: Default::default(),
                        staged: Vec::new(),
                    })),
                })?;
                block.ret(Some(result))
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&PureKnownModel)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::U64(0x45))
        );
    }
}

/// States two adjacent two-byte image objects: `A` at the image's first byte,
/// `B` just past it, with their extents from `0`.
#[derive(Debug, Clone, Copy)]
struct AdjacentObjects(ExtentSource);

impl ObjectExtents for AdjacentObjects {
    fn object(&self, address: Address) -> Option<StorageObject> {
        [0u64, 2]
            .into_iter()
            .map(|at| StorageObject {
                start: i64::try_from(SOURCE_AT.wrapping_add(at))
                    .unwrap_or_else(|_| process::abort()),
                bytes: 2,
                kind: ObjectKind::Global,
                source: self.0,
            })
            .find(|object| object.reach(address).is_some())
    }
}

impl ExecutionHost<MockTarget> for AdjacentObjects {
    fn stack_extent(
        &self,
        _body: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Range<i64>> {
        Some(-32..0)
    }

    fn object_extents(
        &self,
        _root: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Arc<dyn ObjectExtents>> {
        Some(Arc::new(*self))
    }
}

/// **A pointer is judged against the object it was formed into, not the
/// one it lands in.** `A` and `B` are adjacent two-byte objects. A pointer
/// written as `A`'s address and indexed by two reaches `B`'s first byte: a
/// read past `A`, found though the byte is `B`'s. Indexed by one it stays in
/// `A`, and `B`'s address written as a literal reads `B`. A pointer written
/// into the middle of `A`, which forms it into nothing, is formed into `A`
/// once indexed, and indexed past `A`'s end it reads past `A` too.
#[test]
fn a_pointer_is_judged_against_the_object_it_was_formed_into() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder
        .in_block(0, |b| {
            let literal = |b: &mut SsaBlockBuilder<'_, MockTarget>, at: u64| {
                b.const_value(
                    tmp(MockType::Ptr),
                    ConstValue::U64(SOURCE_AT.wrapping_add(at)),
                )
            };
            let index = |b: &mut SsaBlockBuilder<'_, MockTarget>, count: u64| {
                let one = b.const_value(tmp(MockType::U64), ConstValue::U64(count))?;
                let zero = b.const_value(tmp(MockType::U64), ConstValue::U64(0))?;
                b.add(tmp(MockType::U64), one, zero)
            };
            let indexed = |b: &mut SsaBlockBuilder<'_, MockTarget>, base, index| {
                b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base,
                    index: Some(index),
                    stride: 1,
                    offset: 0,
                    result_type: MockType::Ptr,
                })
            };
            let a = literal(b, 0)?;
            let two = index(b, 2)?;
            let one = index(b, 1)?;
            let past = indexed(b, a, two)?;
            let within = indexed(b, a, one)?;
            let b_start = literal(b, 2)?;
            let middle = literal(b, 1)?;
            let from_middle = indexed(b, middle, one)?;
            // An anchor: `A`'s address plus a constant reaching `B`, as a section
            // anchor reaches its objects.
            let anchored = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: a,
                index: None,
                stride: 1,
                offset: 2,
                result_type: MockType::Ptr,
            })?;
            // An index folded before the constant that reaches the object, as
            // `anchor + i + 2`: indexed by zero it reads `B`, by two past `B`.
            let zero = index(b, 0)?;
            let folded = |b: &mut SsaBlockBuilder<'_, MockTarget>, index| {
                let indexed = indexed(b, a, index)?;
                b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: indexed,
                    index: None,
                    stride: 1,
                    offset: 2,
                    result_type: MockType::Ptr,
                })
            };
            let folded_in = folded(b, zero)?;
            let folded_past = folded(b, two)?;
            for pointer in [
                past,
                within,
                b_start,
                from_middle,
                anchored,
                folded_in,
                folded_past,
            ] {
                b.load_indirect(tmp(MockType::U8), pointer, MockType::U8)?;
            }
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let loads: Vec<usize> = ssa
        .block(0)
        .map(|block| block.instructions())
        .unwrap_or_default()
        .iter()
        .enumerate()
        .filter(|(_, instruction)| matches!(instruction.op(), SsaOp::LoadIndirect { .. }))
        .map(|(at, _)| at)
        .collect();
    let image = ObservationImage;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    let at = |load: usize| loads.get(load).copied().unwrap_or_else(|| process::abort());
    // Only an exact extent is one a pointer is formed into: against
    // recovered extents every read is judged where it lands, and finds nothing.
    for (source, expected) in [
        (
            ExtentSource::Symbol,
            vec![
                (at(0), FindingKind::PastExtent(ExtentSource::Symbol)),
                (at(6), FindingKind::PastExtent(ExtentSource::Symbol)),
            ],
        ),
        (ExtentSource::Recovered, Vec::new()),
    ] {
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&AdjacentObjects(source))
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let found: Vec<(usize, FindingKind)> = facts
            .findings()
            .iter()
            .map(|found| (found.instruction, found.kind))
            .collect();
        assert_eq!(found, expected, "{source:?} {loads:?}");
    }
}

/// A function that spills a pointer formed into `A` to its frame and loads
/// it back, by plain accesses or by atomic ones where `atomic`, then reads
/// two bytes past it.
fn spilled_pointer(atomic: bool) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = builder.function_mut().variable_mut(stack) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    builder
        .in_block(0, |b| {
            let slot = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            let a = b.const_value(tmp(MockType::U64), ConstValue::U64(SOURCE_AT))?;
            let zero = b.const_value(tmp(MockType::U64), ConstValue::U64(0))?;
            let nothing = b.add(tmp(MockType::U64), zero, zero)?;
            let formed = b.emit_def(tmp(MockType::U64), |dest| SsaOp::PtrAdd {
                dest,
                base: a,
                index: Some(nothing),
                stride: 1,
                offset: 0,
                result_type: MockType::U64,
            })?;
            let reloaded = if atomic {
                b.emit_no_defs(SsaOp::AtomicStore {
                    addr: slot,
                    value: formed,
                    value_type: MockType::U64,
                    ordering: AtomicOrdering::Release,
                    width: AtomicAccessWidth::Bits64,
                    access: AccessQualifiers::FLAT,
                })?;
                b.emit_def(tmp(MockType::U64), |dest| SsaOp::AtomicLoad {
                    dest,
                    addr: slot,
                    value_type: MockType::U64,
                    ordering: AtomicOrdering::Acquire,
                    width: AtomicAccessWidth::Bits64,
                    reserves: false,
                    access: AccessQualifiers::FLAT,
                })?
            } else {
                b.store_indirect(slot, formed, MockType::U64)?;
                b.load_indirect(tmp(MockType::U64), slot, MockType::U64)?
            };
            let past = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: reloaded,
                index: None,
                stride: 1,
                offset: 2,
                result_type: MockType::Ptr,
            })?;
            b.load_indirect(tmp(MockType::U8), past, MockType::U8)?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    builder.finish().unwrap_or_else(abort)
}

/// **A pointer stored and loaded back is still formed into its object.**
/// `A`'s address, indexed by zero, is a pointer formed into `A`; stored to
/// the frame and loaded back, by plain or by atomic accesses, then moved two
/// bytes by a constant, which forms nothing new, it reads `B`'s first byte:
/// a read past `A`. Loaded back with no object, the same read would land at
/// `B`'s start and find nothing.
#[test]
fn a_spilled_pointer_keeps_its_object() {
    let image = ObservationImage;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    for atomic in [false, true] {
        let ssa = spilled_pointer(atomic);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&AdjacentObjects(ExtentSource::Symbol))
            .run();
        assert!(facts.is_complete(), "{atomic}: {:?}", facts.completion());
        let found: Vec<FindingKind> = facts.findings().iter().map(|found| found.kind).collect();
        assert_eq!(
            found,
            [FindingKind::PastExtent(ExtentSource::Symbol)],
            "{atomic}"
        );
    }
}

/// Every object of [`FrameObjects`] is this long.
const OBJECT_BYTES: i64 = 16;

/// Where `A`, the first object of [`FrameObjects`], starts in the frame.
const OBJECT_A: i64 = -32;

/// Where `B`, just past `A`, starts.
const OBJECT_B: i64 = OBJECT_A + OBJECT_BYTES;

/// The `memset` model's method number.
const FILL: u32 = 60;

/// The `memcpy` model's method number.
const COPY_BYTES: u32 = 61;

/// The `strlen` model's method number.
const LENGTH_OF: u32 = 62;

/// States two adjacent sixteen-byte frame objects, `A` at `OBJECT_A` and `B`
/// just past it, with their extents from `0`, in a 64-byte frame, and
/// models `memset`, `memcpy` and `strlen`.
#[derive(Debug, Clone, Copy)]
struct FrameObjects(ExtentSource);

impl ObjectExtents for FrameObjects {
    fn object(&self, address: Address) -> Option<StorageObject> {
        if !matches!(address.root(), Root::Stack) {
            return None;
        }
        [OBJECT_A, OBJECT_B]
            .into_iter()
            .map(|start| StorageObject {
                start,
                bytes: OBJECT_BYTES.unsigned_abs(),
                kind: ObjectKind::Frame,
                source: self.0,
            })
            .find(|object| object.reach(address).is_some())
    }
}

impl ExecutionHost<MockTarget> for FrameObjects {
    fn stack_extent(
        &self,
        _body: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Range<i64>> {
        Some(-64..0)
    }

    fn object_extents(
        &self,
        _root: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Arc<dyn ObjectExtents>> {
        Some(Arc::new(*self))
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match *method {
            FILL => Some(&LibraryModel::Fill),
            COPY_BYTES => Some(&LibraryModel::Copy),
            LENGTH_OF => Some(&LibraryModel::Length),
            _ => None,
        }
    }
}

/// One function accessing memory through `pointer` in every way an access
/// is made: a scalar store, a vector load and store, a gather and a scatter
/// at offsets inside `B`, an atomic load and store, a block copy into it and out of it, a block fill, a block compare
/// with it on either side, and a `memset`, a `memcpy` into it and out of it
/// and a `strlen` given it. Both objects are zeroed first. `pointer` reaches
/// `B`'s start: as the frame address moved by a constant, which forms
/// nothing, where `formed` is false, or as `A`'s address indexed by sixteen,
/// which forms it into `A`, where it is true. The other side of a copy or a
/// compare is `B`'s own address. Returns the function and the instructions
/// that use `pointer`.
fn every_access_through(formed: bool) -> (SsaFunction<MockTarget>, Vec<usize>) {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = builder.function_mut().variable_mut(stack) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    let mut through = None;
    builder
        .in_block(0, |b| {
            let frame = |b: &mut SsaBlockBuilder<'_, MockTarget>, offset| {
                b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset,
                    result_type: MockType::Ptr,
                })
            };
            let a_start = frame(b, OBJECT_A)?;
            let b_start = frame(b, OBJECT_B)?;
            let sixteen = b.const_value(tmp(MockType::U64), ConstValue::U64(16))?;
            let zero = b.const_value(tmp(MockType::U64), ConstValue::U64(0))?;
            let byte = b.const_value(tmp(MockType::U8), ConstValue::U8(0))?;
            for object in [a_start, b_start] {
                b.emit_no_defs(SsaOp::InitBlk {
                    dest_addr: object,
                    value: byte,
                    size: sixteen,
                    backward: None,
                    access: AccessQualifiers::FLAT,
                })?;
            }
            let pointer = if formed {
                let index = b.add(tmp(MockType::U64), sixteen, zero)?;
                b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: a_start,
                    index: Some(index),
                    stride: 1,
                    offset: 0,
                    result_type: MockType::Ptr,
                })?
            } else {
                frame(b, OBJECT_B)?
            };
            through = Some(pointer);
            let four = b.const_value(tmp(MockType::U64), ConstValue::U64(4))?;
            let word = b.const_value(tmp(MockType::U32), ConstValue::U32(7))?;
            let fill = b.const_value(tmp(MockType::U32), ConstValue::U32(0x61))?;
            let flags_in = b.const_value(tmp(MockType::U32), ConstValue::U32(0))?;
            let forward = b.const_value(tmp(MockType::Bool), ConstValue::False)?;
            b.store_indirect(pointer, word, MockType::U32)?;
            let vector = b.emit_def(tmp(MockType::V4I32), |dest| SsaOp::VectorLoad {
                dest,
                addr: pointer,
                vector_type: MockType::V4I32,
                access: AccessQualifiers::FLAT,
            })?;
            b.emit_no_defs(SsaOp::VectorStore {
                addr: pointer,
                value: vector,
                vector_type: MockType::V4I32,
                access: AccessQualifiers::FLAT,
            })?;
            let every = VectorConst::from_lanes(1, [1; 4]).unwrap_or_else(|| process::abort());
            let active = b.const_value(tmp(MockType::Mask4), ConstValue::Vector(every))?;
            let offsets =
                VectorConst::from_lanes(32, [0, 4, 8, 12]).unwrap_or_else(|| process::abort());
            let indices = b.const_value(tmp(MockType::V4I32), ConstValue::Vector(offsets))?;
            b.emit_def(tmp(MockType::V4I32), |dest| SsaOp::VectorGather {
                dest,
                base: pointer,
                indices,
                mask: active,
                passthrough: None,
                vector_type: MockType::V4I32,
                mode: VectorMaskMode::Zero,
                access: AccessQualifiers::FLAT,
            })?;
            b.emit_def(tmp(MockType::U32), |dest| SsaOp::AtomicLoad {
                dest,
                addr: pointer,
                value_type: MockType::U32,
                ordering: AtomicOrdering::Acquire,
                width: AtomicAccessWidth::Bits32,
                reserves: false,
                access: AccessQualifiers::FLAT,
            })?;
            b.emit_no_defs(SsaOp::AtomicStore {
                addr: pointer,
                value: word,
                value_type: MockType::U32,
                ordering: AtomicOrdering::Release,
                width: AtomicAccessWidth::Bits32,
                access: AccessQualifiers::FLAT,
            })?;
            for (dest_addr, src_addr) in [(pointer, b_start), (b_start, pointer)] {
                b.emit_no_defs(SsaOp::CopyBlk {
                    dest_addr,
                    src_addr,
                    size: four,
                    element_bytes: 1,
                    backward: None,
                    src_access: AccessQualifiers::FLAT,
                    dest_access: AccessQualifiers::FLAT,
                })?;
            }
            b.emit_no_defs(SsaOp::InitBlk {
                dest_addr: pointer,
                value: byte,
                size: four,
                backward: None,
                access: AccessQualifiers::FLAT,
            })?;
            for (left, right) in [(pointer, b_start), (b_start, pointer)] {
                let defs = [
                    tmp(MockType::U64),
                    tmp(MockType::Ptr),
                    tmp(MockType::Ptr),
                    tmp(MockType::U32),
                ];
                b.emit_many_defs(&defs, |ids| {
                    let &[remaining, left_next, right_next, flags] = ids else {
                        return SsaOp::Nop;
                    };
                    SsaOp::CompareBlk(Box::new(CompareBlkData {
                        remaining,
                        left_next,
                        right_next,
                        flags,
                        left,
                        right,
                        count: four,
                        flags_in: Some(flags_in),
                        backward: Some(forward),
                        element_bytes: 1,
                        while_equal: true,
                        left_access: AccessQualifiers::FLAT,
                        right_access: AccessQualifiers::FLAT,
                    }))
                })?;
            }
            b.call(
                tmp(MockType::Ptr),
                Callee::Direct(FILL),
                vec![pointer, fill, four],
            )?;
            for (to, from) in [(pointer, b_start), (b_start, pointer)] {
                b.call(
                    tmp(MockType::Ptr),
                    Callee::Direct(COPY_BYTES),
                    vec![to, from, four],
                )?;
            }
            b.call(tmp(MockType::U64), Callee::Direct(LENGTH_OF), vec![pointer])?;
            // Last: the run models no scatter, which forgets the memory it
            // may reach once its lanes are judged.
            b.emit_no_defs(SsaOp::VectorScatter {
                base: pointer,
                indices,
                value: vector,
                mask: active,
                vector_type: MockType::V4I32,
                access: AccessQualifiers::FLAT,
            })?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let through = through.unwrap_or_else(|| process::abort());
    let sites = ssa
        .block(0)
        .map(|block| block.instructions())
        .unwrap_or_default()
        .iter()
        .enumerate()
        .filter(|(_, instruction)| instruction.op().uses().contains(&through))
        .map(|(at, _)| at)
        .collect();
    (ssa, sites)
}

/// **Every access through a pointer is judged against the object it was
/// formed into.** A pointer formed into the frame object `A` and indexed
/// onto `B`'s start reaches past `A` whatever the access: a scalar, vector
/// or atomic one, a gather's or a scatter's lanes, a block copy on either
/// side, a block fill, a block
/// compare on either side, and a library model's read or write through the
/// argument it is handed. Each is a finding past `A`'s extent, though every
/// byte lies in `B`. The same accesses through `B`'s own address land in `B`
/// and find nothing, and against recovered extents no pointer is formed
/// into an object, so nothing is found either.
#[test]
fn every_access_through_a_pointer_is_judged_against_its_object() {
    let context = AnalysisContext::new(MACHINE);
    for (formed, source) in [
        (true, ExtentSource::Symbol),
        (false, ExtentSource::Symbol),
        (true, ExtentSource::Recovered),
    ] {
        let (ssa, sites) = every_access_through(formed);
        // Sixteen accesses: scalar store, vector load and store, gather,
        // scatter, atomic load and store, two copies, a fill, two compares
        // and four calls.
        assert_eq!(sites.len(), 16, "{formed}");
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameObjects(source))
            .run();
        assert!(
            facts.is_complete(),
            "{formed} {source:?}: {:?}",
            facts.completion()
        );
        let found: Vec<(usize, FindingKind)> = facts
            .findings()
            .iter()
            .map(|found| (found.instruction, found.kind))
            .collect();
        let expected: Vec<(usize, FindingKind)> = if formed && source == ExtentSource::Symbol {
            sites
                .iter()
                .map(|site| (*site, FindingKind::PastExtent(ExtentSource::Symbol)))
                .collect()
        } else {
            Vec::new()
        };
        assert_eq!(found, expected, "{formed} {source:?} {sites:?}");
    }
}
