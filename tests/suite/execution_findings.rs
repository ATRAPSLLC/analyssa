//! What a finding carries: the bytes it reached, the object it was judged
//! against, and the decisions of the path that found it.

use std::{collections::BTreeSet, process};

use analyssa::{
    analysis::{
        context::AnalysisContext,
        execution::{
            Allocation, AllocatorFamily, CallModel, Conditions, Deallocation, Decision, Delivery,
            Ending, ExecutionHost, Finding, FindingKind, FoundObject, FreeResult, LibraryModel,
            Operand, Root, StaticExecution,
        },
    },
    ir::{
        Callee,
        exception::{BlockRange, SsaExceptionHandler},
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::CmpKind,
        value::ConstValue,
    },
    testing::{MACHINE, MockTarget, MockType},
};

/// Ends the test: a fixture the builder rejects is a broken test.
fn abort<E, R>(_: E) -> R {
    process::abort()
}

fn tmp(ty: MockType) -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(ty)
}

/// The allocator's method.
const MALLOC: u32 = 17;

/// The deallocator's method.
const FREE: u32 = 18;

/// The family both belong to.
const FAMILY: AllocatorFamily = AllocatorFamily(1);

/// `malloc(n)`: succeeds or returns null.
const ALLOCATE: LibraryModel = LibraryModel::Allocate(Allocation {
    family: FAMILY,
    size: Operand::Argument(0),
    count: None,
    alignment: None,
    zeroed: true,
    flags: None,
    delivery: Delivery::Returned,
    requires: Conditions::NONE,
    pages: None,
    protection: None,
    heap: None,
});

/// `free(p)`.
const RELEASE: LibraryModel = LibraryModel::Free(Deallocation {
    family: FAMILY,
    pointer: Operand::Argument(0),
    size: None,
    result: FreeResult::Nothing,
    requires: Conditions::NONE,
    whole: None,
    released: None,
    heap: None,
});

/// `posix_memalign(out, n)`'s method.
const MEMALIGN: u32 = 21;

/// `posix_memalign(out, n)`: writes a new block through `out`, or fails.
const ALLOCATE_OUT: LibraryModel = LibraryModel::Allocate(Allocation {
    family: FAMILY,
    size: Operand::Argument(1),
    count: None,
    alignment: None,
    zeroed: true,
    flags: None,
    delivery: Delivery::Out {
        out: 0,
        failure: 12,
    },
    requires: Conditions::NONE,
    pages: None,
    protection: None,
    heap: None,
});

/// Models the allocators and the deallocator.
struct Allocator;

impl ExecutionHost<MockTarget> for Allocator {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match *method {
            MALLOC => Some(&ALLOCATE),
            FREE => Some(&RELEASE),
            MEMALIGN => Some(&ALLOCATE_OUT),
            _ => None,
        }
    }
}

/// Where a function built by [`freed_then`] allocates and frees, and where
/// it reads the freed block.
struct Sites {
    /// The allocating call, `(block, instruction)`.
    malloc: (usize, usize),
    /// The freeing call.
    free: (usize, usize),
    /// The read of the freed block.
    read: (usize, usize),
}

/// The instruction index the next emitted instruction of `block` takes.
fn next(builder: &SsaFunctionBuilder<MockTarget>, block: usize) -> usize {
    builder
        .function()
        .block(block)
        .map_or(0, |block| block.instructions().len())
}

/// Allocates four bytes and frees them in block 0, then hands the block and
/// the argument to `rest`, which builds the rest and returns where it reads
/// the freed block.
fn freed_then(
    rest: impl FnOnce(
        &mut SsaFunctionBuilder<MockTarget>,
        analyssa::ir::variable::SsaVarId,
        analyssa::ir::variable::SsaVarId,
    ) -> (usize, usize),
) -> (SsaFunction<MockTarget>, Sites) {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let mut malloc = (0, 0);
    let mut free = (0, 0);
    let (block, argument) = builder
        .in_block(0, |b| {
            let four = b.const_value(tmp(MockType::U64), ConstValue::U64(4))?;
            let argument = b.load_arg(SsaDefSpec::argument(0, MockType::I64), 0)?;
            Ok((four, argument))
        })
        .unwrap_or_else(abort);
    malloc.1 = next(&builder, 0);
    let pointer = builder
        .in_block(0, |b| {
            b.call(tmp(MockType::Ptr), Callee::Direct(MALLOC), vec![block])
        })
        .unwrap_or_else(abort);
    free.1 = next(&builder, 0);
    builder
        .in_block(0, |b| b.call_void(Callee::Direct(FREE), vec![pointer]))
        .unwrap_or_else(abort);
    let read = rest(&mut builder, pointer, argument);
    let ssa = builder.finish().unwrap_or_else(abort);
    (ssa, Sites { malloc, free, read })
}

/// The one finding of `kind` the run of `ssa` makes.
fn found(ssa: &SsaFunction<MockTarget>, kind: FindingKind) -> Finding {
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(ssa, &context)
        .with_host(&Allocator)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let mut matching = facts.findings().iter().filter(|found| found.kind == kind);
    let first = matching.next().cloned();
    assert!(matching.next().is_none(), "{:?}", facts.findings());
    assert!(first.is_some(), "no {kind:?} in {:?}", facts.findings());
    first.unwrap_or_else(|| process::abort())
}

/// The block an allocation made, as a finding names it.
fn block(sites: &Sites, root: Root) -> FoundObject {
    FoundObject::Block {
        root,
        size: 4,
        extent: Some(4),
        family: Some(FAMILY),
        allocated_at: Some(sites.malloc),
        ended_at: BTreeSet::from([sites.free]),
    }
}

/// **A read of a freed block carries the block and the path that read it.**
/// The read is past the block's lifetime: the finding names the bytes it
/// reached, the block with the call that made it and the one that ended
/// it, and its witness is the allocator's alternative the path took,
/// success; the failure's null reaches no block. Read behind a branch on the
/// argument, the witness also holds the branch's side.
#[test]
fn a_read_of_a_freed_block_carries_the_block_and_its_path() {
    for behind_branch in [false, true] {
        let (ssa, sites) = freed_then(|builder, pointer, argument| {
            if behind_branch {
                builder
                    .in_block(0, |b| {
                        let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
                        b.branch_cmp(argument, zero, CmpKind::Eq, false, 1, 2)
                    })
                    .unwrap_or_else(abort);
                let read = (1, next(builder, 1));
                builder
                    .in_block(1, |b| {
                        b.load_indirect(tmp(MockType::U8), pointer, MockType::U8)?;
                        b.ret(None)
                    })
                    .unwrap_or_else(abort);
                builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
                read
            } else {
                let read = (0, next(builder, 0));
                builder
                    .in_block(0, |b| {
                        b.load_indirect(tmp(MockType::U8), pointer, MockType::U8)?;
                        b.ret(None)
                    })
                    .unwrap_or_else(abort);
                read
            }
        });
        let finding = found(&ssa, FindingKind::Ended(Ending::Every));
        assert_eq!((finding.block, finding.instruction), sites.read);
        let (at, len) = finding.access.unwrap_or_else(|| process::abort());
        assert_eq!((at.offset(), len), (0, 1));
        assert_eq!(finding.object, Some(block(&sites, at.root())));
        let mut decisions = vec![Decision::Alternative {
            body: 0,
            site: sites.malloc,
            alternative: 0,
        }];
        if behind_branch {
            decisions.push(Decision::Branch {
                body: 0,
                block: 0,
                successor: 1,
            });
        }
        assert_eq!(finding.witness.decisions, decisions, "{behind_branch}");
        assert!(!finding.witness.joined);
    }
}

/// **A finding in a handler carries the raise that entered it.** The freed
/// block is read in the handler guarding a division by the argument, which
/// may raise: the witness holds the allocator's success and the raise into
/// the handler, at the division.
#[test]
fn a_finding_in_a_handler_carries_the_raise() {
    let mut division = (0, 0);
    let (mut ssa, sites) = freed_then(|builder, pointer, argument| {
        let hundred = builder
            .in_block(0, |b| {
                b.const_value(tmp(MockType::I64), ConstValue::I64(100))
            })
            .unwrap_or_else(abort);
        division = (0, next(builder, 0));
        builder
            .in_block(0, |b| {
                b.div(tmp(MockType::I64), hundred, argument, false)?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let read = (1, next(builder, 1));
        builder
            .in_block(1, |b| {
                b.load_indirect(tmp(MockType::U8), pointer, MockType::U8)?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        read
    });
    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 2),
        filter_range: None,
        selector: None,
    }]);
    let finding = found(&ssa, FindingKind::Ended(Ending::Every));
    assert_eq!((finding.block, finding.instruction), sites.read);
    assert_eq!(
        finding.witness.decisions,
        [
            Decision::Alternative {
                body: 0,
                site: sites.malloc,
                alternative: 0,
            },
            Decision::Raise {
                body: 0,
                site: division,
                handler: 1,
            },
        ]
    );
}

/// **A release found inside a model carries the path into it.** A second
/// `free` of the block is a repeated release, found where the deallocator
/// runs: the finding names the released address, reaching no byte, the
/// block as the first `free` left it, and the allocator's success the path
/// took.
#[test]
fn a_repeated_release_carries_the_block_and_its_path() {
    let (ssa, sites) = freed_then(|builder, pointer, _| {
        let again = (0, next(builder, 0));
        builder
            .in_block(0, |b| {
                b.call_void(Callee::Direct(FREE), vec![pointer])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        again
    });
    let finding = found(
        &ssa,
        FindingKind::Release(analyssa::analysis::execution::ReleaseFault::Repeated),
    );
    assert_eq!((finding.block, finding.instruction), sites.read);
    let (at, len) = finding.access.unwrap_or_else(|| process::abort());
    assert_eq!((at.offset(), len), (0, 0));
    assert_eq!(finding.object, Some(block(&sites, at.root())));
    assert_eq!(
        finding.witness.decisions,
        [Decision::Alternative {
            body: 0,
            site: sites.malloc,
            alternative: 0,
        }]
    );
}

/// **A finding a model makes carries the alternative it makes it in.** An
/// out-parameter allocator handed the freed block as its cell writes the new
/// block's address there on success: a write to ended storage, found inside
/// that alternative, whose witness holds both allocators' successes.
#[test]
fn a_finding_inside_a_model_carries_its_alternative() {
    let (ssa, sites) = freed_then(|builder, pointer, _| {
        let four = builder
            .in_block(0, |b| b.const_value(tmp(MockType::U64), ConstValue::U64(4)))
            .unwrap_or_else(abort);
        let call = (0, next(builder, 0));
        builder
            .in_block(0, |b| {
                b.call(
                    tmp(MockType::I32),
                    Callee::Direct(MEMALIGN),
                    vec![pointer, four],
                )?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        call
    });
    let finding = found(&ssa, FindingKind::Ended(Ending::Every));
    assert_eq!(
        finding.witness.decisions,
        [
            Decision::Alternative {
                body: 0,
                site: sites.malloc,
                alternative: 0,
            },
            Decision::Alternative {
                body: 0,
                site: sites.read,
                alternative: 0,
            },
        ]
    );
}
