//! Static execution over whole functions: what it establishes about a buffer
//! decoded from read-only data, and what it must not establish when the
//! program's input decides the result.

use std::{ops::Range, process, sync::mpsc, thread, time::Duration};

use analyssa::{
    PointerSize,
    analysis::{
        Machine,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        execution::{
            Address, ArgumentBytes, ByteBound, CallModel, Completion, ExecutionHost, FindingKind,
            InstructionSources, LibraryModel, MemoryRefusal, MemoryWriteGuard, ObjectKind, Root,
            StaticExecution, StaticFacts, UnwrittenUseKind,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        Callee,
        division::DivisionRules,
        function::{SsaBlockBuilder, SsaDefSpec, SsaEditOptions, SsaFunction, SsaFunctionBuilder},
        instruction::SsaInstruction,
        ops::{
            AccessQualifiers, AtomicAccessWidth, AtomicOrdering, CmpKind, CompareBlkData,
            FenceKind, NativeKindedData, SsaOp, SystemOpKind, TrapOp, VectorSegmentLayout,
        },
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
    },
    passes::residual::residualize_loops,
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Where the encoded bytes live.
const ENCODED_AT: u64 = 0x1000;

/// `"hello, world!!"` XOR `0x5a`.
const ENCODED: [u8; 14] = [
    0x32, 0x3f, 0x36, 0x36, 0x35, 0x76, 0x7a, 0x2d, 0x35, 0x28, 0x36, 0x3e, 0x7b, 0x7b,
];

/// The plain text.
const PLAIN: &[u8] = b"hello, world!!";

/// Attests the decoder fixture's allocated 32 bytes below its entry stack pointer.
struct FrameHost;

impl ExecutionHost<MockTarget> for FrameHost {
    fn stack_extent(
        &self,
        _body: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Range<i64>> {
        Some(-32..0)
    }
}

/// The encoded bytes as read-only data.
struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        let end = ENCODED_AT.wrapping_add(ENCODED.len() as u64);
        (ENCODED_AT..end).contains(&address).then_some(ImageRegion {
            start: ENCODED_AT,
            end,
            mutability: Mutability::ReadOnly,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(address.checked_sub(ENCODED_AT)?).ok()?;
        ENCODED.get(start..start.checked_add(len)?)
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

/// The decode loop, and the value of its call argument.
///
/// ```text
/// 0: buf = sp - 32; src = 0x1000; jump 1
/// 1: i = phi(0, i'); byte = src[i] ^ key; buf[i] = byte; i' = i + 1
///    if i' < 14 goto 1 else 2
/// 2: emit(buf, 14); return
/// ```
///
/// `key` is `0x5a`, or the function's parameter when `keyed_by_caller`; the
/// bound is 14, or the parameter when `bounded_by_caller`.
struct Decoder {
    ssa: SsaFunction<MockTarget>,
    /// The call's block and index.
    call: (usize, usize),
}

fn decoder(keyed_by_caller: bool, bounded_by_caller: bool) -> Decoder {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let (stack, parameter) = {
        let ssa = builder.function_mut();
        let stack = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        if let Some(variable) = ssa.variable_mut(stack) {
            variable.set_role(VariableRole::EntryStackPointer);
        }
        let parameter = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::U8,
        );
        (stack, parameter)
    };
    let (buffer, source, zero, one, count, fixed_key) = builder
        .in_block(0, |b| {
            let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -32,
                result_type: MockType::Ptr,
            })?;
            let source = b.const_value(tmp(MockType::U64), ConstValue::U64(ENCODED_AT))?;
            let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            let count = b.const_value(tmp(MockType::I64), ConstValue::I64(ENCODED.len() as i64))?;
            let fixed_key = b.const_value(tmp(MockType::U8), ConstValue::U8(0x5a))?;
            b.jump(1)?;
            Ok((buffer, source, zero, one, count, fixed_key))
        })
        .unwrap_or_else(abort);
    let key = if keyed_by_caller {
        parameter
    } else {
        fixed_key
    };
    let bound = if bounded_by_caller { parameter } else { count };
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
            let encoded = b.load_indirect(tmp(MockType::U8), from, MockType::U8)?;
            let plain = b.xor(tmp(MockType::U8), encoded, key)?;
            let to = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: buffer,
                index: Some(index),
                stride: 1,
                offset: 0,
                result_type: MockType::Ptr,
            })?;
            b.store_indirect(to, plain, MockType::U8)?;
            let next = b.add(tmp(MockType::I64), index, one)?;
            b.branch_cmp(next, bound, CmpKind::Lt, false, 1, 2)?;
            Ok(next)
        })
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, index, 0, zero)
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, index, 1, next)
        .unwrap_or_else(abort);
    builder
        .in_block(2, |b| {
            b.call_void(Callee::Direct(7), vec![buffer, count])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    Decoder {
        ssa: builder.finish().unwrap_or_else(abort),
        call: (2, 0),
    }
}

fn run(ssa: &SsaFunction<MockTarget>) -> StaticFacts<MockTarget> {
    let image = Image;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    StaticExecution::new(ssa, &context)
        .with_host(&FrameHost)
        .run()
}

fn stack(offset: i64) -> Address {
    Address::new(Root::Stack, offset, PointerSize::Bit64)
}

/// A fixed key over read-only bytes decodes: the loop's effect is the plain
/// text, and the call is handed it.
#[test]
fn a_fixed_key_decodes() {
    let decoder = decoder(false, false);
    let facts = run(&decoder.ssa);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts.usage().state_snapshots,
        0,
        "an exact path must not retain growing per-iteration snapshots"
    );
    let effect = facts.loop_effect(1).unwrap_or_else(|| abort(()));
    assert_eq!(effect.writes, vec![(stack(-32), PLAIN.to_vec())]);
    assert_eq!(effect.exit, (1, 2));
    let arguments = facts.call_arguments(decoder.call.0, decoder.call.1);
    assert_eq!(
        arguments.first().map(|argument| argument.bytes.as_slice()),
        Some(PLAIN)
    );
}

/// A key the caller supplies decodes nothing: the loop has no effect the
/// execution can state, and the call is handed no known byte.
#[test]
fn a_callers_key_decodes_nothing() {
    let decoder = decoder(true, false);
    let facts = run(&decoder.ssa);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(facts.loop_effect(1).is_none());
    assert!(
        facts
            .call_arguments(decoder.call.0, decoder.call.1)
            .is_empty()
    );
}

/// A loop the caller bounds closes: the joins at its header generalise the
/// counter, the exploration finishes, and the loop has no effect that holds
/// on every run.
#[test]
fn a_callers_bound_closes_the_loop() {
    let decoder = decoder(false, true);
    let facts = run(&decoder.ssa);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(facts.loop_effect(1).is_none());
}

/// Ordered effects remain observable even when all decoded bytes are known.
#[test]
fn fences_prevent_loop_replacement() {
    for kind in [
        FenceKind::Full,
        FenceKind::Acquire,
        FenceKind::Release,
        FenceKind::AcqRel,
        FenceKind::SeqCst,
        FenceKind::Loads,
        FenceKind::Stores,
    ] {
        let mut decoder = decoder(false, false);
        decoder
            .ssa
            .edit(SsaEditOptions::new(), |editor| {
                editor.insert_before_terminator(
                    1,
                    SsaInstruction::synthetic(SsaOp::Fence { kind }),
                )?;
                Ok(())
            })
            .unwrap_or_else(abort);
        let facts = run(&decoder.ssa);
        assert!(facts.is_complete());
        assert!(facts.loop_effect(1).is_none(), "{kind:?}");
        assert_eq!(
            residualize_loops(&mut decoder.ssa, &facts).unwrap_or_else(abort),
            0
        );
    }
}

/// What the decoder loop also does on each iteration, for
/// [`intermediate_effects_decide_loop_replacement`].
#[derive(Debug, Clone, Copy)]
enum Intermediate {
    /// `1 / index`: zero on the first iteration.
    DivideByIndex,
    /// `index / 1`.
    DivideByOne,
    /// A volatile read of the byte the iteration decodes.
    VolatileLoad,
    /// A volatile write of the byte the iteration stores.
    VolatileStore,
}

/// **What a replaced loop does on the way is part of the claim.** A division
/// that traps on one iteration, or a volatile access, keeps the decoder
/// loop; a division that completes on every iteration, by its divisor or by
/// the machine's own outcome for a zero divisor, does not.
#[test]
fn intermediate_effects_decide_loop_replacement() {
    for (case, rules, replaced) in [
        (Intermediate::DivideByIndex, DivisionRules::CHECKED, false),
        (Intermediate::DivideByIndex, DivisionRules::ARM, true),
        (Intermediate::DivideByOne, DivisionRules::CHECKED, true),
        (Intermediate::VolatileLoad, DivisionRules::CHECKED, false),
        (Intermediate::VolatileStore, DivisionRules::CHECKED, false),
    ] {
        let mut decoder = decoder(false, false);
        let body = decoder.ssa.block(1).unwrap_or_else(|| abort(()));
        let index = body
            .phi_nodes()
            .first()
            .unwrap_or_else(|| abort(()))
            .result();
        let def = |at: usize| {
            body.instruction(at)
                .and_then(|instruction| instruction.op().dest())
                .unwrap_or_else(|| abort(()))
        };
        let (from, encoded, to) = (def(0), def(1), def(3));
        decoder
            .ssa
            .edit(SsaEditOptions::new(), |editor| {
                let one = editor.create_fresh_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(1, 0),
                    MockType::I64,
                );
                editor.insert_before_terminator(
                    1,
                    SsaInstruction::synthetic(SsaOp::Const {
                        dest: one,
                        value: ConstValue::I64(1),
                    }),
                )?;
                let defines = !matches!(case, Intermediate::VolatileStore);
                let dest = defines.then(|| {
                    editor.create_fresh_variable(
                        VariableOrigin::Phi,
                        0,
                        DefSite::instruction(1, 0),
                        if matches!(case, Intermediate::VolatileLoad) {
                            MockType::U8
                        } else {
                            MockType::I64
                        },
                    )
                });
                let dest = dest.unwrap_or(one);
                let volatile = AccessQualifiers {
                    volatile: true,
                    ..AccessQualifiers::default()
                };
                let op = match case {
                    Intermediate::DivideByIndex => SsaOp::Div {
                        dest,
                        left: one,
                        right: index,
                        unsigned: true,
                        flags: None,
                    },
                    Intermediate::DivideByOne => SsaOp::Div {
                        dest,
                        left: index,
                        right: one,
                        unsigned: true,
                        flags: None,
                    },
                    Intermediate::VolatileLoad => SsaOp::LoadIndirect {
                        dest,
                        addr: from,
                        value_type: MockType::U8,
                        access: volatile,
                    },
                    Intermediate::VolatileStore => SsaOp::StoreIndirect {
                        addr: to,
                        value: encoded,
                        value_type: MockType::U8,
                        access: volatile,
                    },
                };
                editor.insert_before_terminator(1, SsaInstruction::synthetic(op))?;
                Ok(())
            })
            .unwrap_or_else(abort);
        let image = Image;
        let context = AnalysisContext::new(MACHINE.with_division_rules(rules)).with_image(&image);
        let facts = StaticExecution::new(&decoder.ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert_eq!(facts.loop_effect(1).is_some(), replaced, "{case:?}");
        assert_eq!(
            residualize_loops(&mut decoder.ssa, &facts).unwrap_or_else(abort),
            usize::from(replaced),
            "{case:?}"
        );
    }
}

/// **Skipping a decoder preserves the initialized bytes the consumer
/// observes.** The decoder is entered under an unknown condition, and its
/// bytes are written back on that arm alone: the entry still tests, the
/// residual lies on a block split onto the edge into the loop, and the arm
/// that skips the loop reaches the consumer as it did.
#[test]
fn conditional_loop_entry_writes_its_residual_on_its_own_arm() {
    let mut decoder = decoder(false, false);
    let condition = decoder
        .ssa
        .variables()
        .iter()
        .find(|v| matches!(v.origin(), VariableOrigin::Argument(0)))
        .map(|v| v.id())
        .unwrap_or_else(|| abort(()));
    let buffer = decoder
        .ssa
        .block(0)
        .and_then(|block| block.instruction(0))
        .and_then(|i| i.op().dest())
        .unwrap_or_else(|| abort(()));
    decoder
        .ssa
        .edit(SsaEditOptions::new(), |editor| {
            let byte = editor.create_fresh_variable(
                VariableOrigin::Phi,
                0,
                DefSite::entry(),
                MockType::U8,
            );
            let count = editor.create_fresh_variable(
                VariableOrigin::Phi,
                0,
                DefSite::entry(),
                MockType::U64,
            );
            editor.insert_before_terminator(
                0,
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: byte,
                    value: ConstValue::U8(b'!'),
                }),
            )?;
            editor.insert_before_terminator(
                0,
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: count,
                    value: ConstValue::U64(14),
                }),
            )?;
            editor.insert_before_terminator(
                0,
                SsaInstruction::synthetic(SsaOp::InitBlk {
                    dest_addr: buffer,
                    value: byte,
                    size: count,
                    backward: None,
                    access: AccessQualifiers::FLAT,
                }),
            )?;
            editor.replace_terminator_op(
                0,
                SsaOp::Branch {
                    condition,
                    true_target: 1,
                    false_target: 2,
                },
            )?;
            // The new edge reaches block 2, which has no phi and reads only
            // values block 0 defines: nothing it carries needs restating.
            editor.declare_repaired();
            Ok(())
        })
        .unwrap_or_else(abort);
    let facts = run(&decoder.ssa);
    assert!(facts.is_complete());
    assert!(
        facts.loop_effect(1).is_some(),
        "the taken edge still decodes"
    );
    assert!(
        facts
            .call_arguments(decoder.call.0, decoder.call.1)
            .is_empty(),
        "the two paths emit different bytes"
    );
    let decoded: Vec<u64> = facts
        .loop_effect(1)
        .and_then(|effect| effect.writes.first())
        .map(|(_, bytes)| bytes.iter().map(|byte| u64::from(*byte)).collect())
        .unwrap_or_else(|| abort(()));
    assert_eq!(
        residualize_loops(&mut decoder.ssa, &facts).unwrap_or_else(abort),
        1
    );
    decoder.ssa.validate().unwrap_or_else(abort);
    let entry = decoder.ssa.block(0).unwrap_or_else(|| abort(()));
    let Some(SsaOp::Branch {
        condition: tested,
        true_target,
        false_target,
    }) = entry.control_terminator()
    else {
        return abort(());
    };
    assert_eq!((*tested, *false_target), (condition, 2));
    // Nothing the residual writes is hoisted onto the entry.
    assert!(
        !entry
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.op(), SsaOp::StoreIndirect { .. }))
    );
    let residual = decoder.ssa.block(*true_target).unwrap_or_else(|| abort(()));
    assert_eq!(residual.successors(), vec![2]);
    let stored: Vec<u64> = residual
        .instructions()
        .iter()
        .filter_map(|instruction| match instruction.op() {
            SsaOp::StoreIndirect { value, .. } => decoder.ssa.get_var_constant(*value)?.as_u64(),
            _ => None,
        })
        .collect();
    assert_eq!(stored, decoded);
}

/// A possibly faulting unused read is not erased with a deterministic decoder.
#[test]
fn an_unknown_read_prevents_loop_replacement() {
    let mut decoder = decoder(false, false);
    decoder
        .ssa
        .edit(SsaEditOptions::new(), |editor| {
            let address = editor.create_fresh_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            let loaded = editor.create_fresh_variable(
                VariableOrigin::Phi,
                0,
                DefSite::entry(),
                MockType::U8,
            );
            editor.insert_before_terminator(
                1,
                SsaInstruction::synthetic(SsaOp::LoadIndirect {
                    dest: loaded,
                    addr: address,
                    value_type: MockType::U8,
                    access: AccessQualifiers::FLAT,
                }),
            )?;
            Ok(())
        })
        .unwrap_or_else(abort);
    let facts = run(&decoder.ssa);
    assert!(facts.is_complete());
    assert!(facts.loop_effect(1).is_none());
    assert_eq!(
        residualize_loops(&mut decoder.ssa, &facts).unwrap_or_else(abort),
        0
    );
}

/// Fills repeat the low element in target byte order, including partial tails.
#[test]
fn fills_follow_target_byte_order() {
    for (order, width, expected) in [
        (Endianness::Little, 1, vec![0x44; 5]),
        (Endianness::Big, 1, vec![0x44; 5]),
        (Endianness::Little, 2, vec![0x44, 0x33, 0x44, 0x33, 0x44]),
        (Endianness::Big, 2, vec![0x33, 0x44, 0x33, 0x44, 0x33]),
        (Endianness::Little, 4, vec![0x44, 0x33, 0x22, 0x11, 0x44]),
        (Endianness::Big, 4, vec![0x11, 0x22, 0x33, 0x44, 0x11]),
        (Endianness::Little, 8, vec![0x44, 0x33, 0x22, 0x11, 0x88]),
        (Endianness::Big, 8, vec![0xff, 0xee, 0xdd, 0x88, 0x11]),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let stack = {
            let ssa = builder.function_mut();
            let value = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            if let Some(v) = ssa.variable_mut(value) {
                v.set_role(VariableRole::EntryStackPointer);
            }
            value
        };
        builder
            .in_block(0, |b| {
                let target = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -16,
                    result_type: MockType::Ptr,
                })?;
                let (ty, value) = match width {
                    1 => (MockType::U8, ConstValue::U8(0x44)),
                    2 => (MockType::U16, ConstValue::U16(0x3344)),
                    4 => (MockType::U32, ConstValue::U32(0x11223344)),
                    _ => (MockType::U64, ConstValue::U64(0xffeedd8811223344)),
                };
                let value = b.const_value(tmp(ty), value)?;
                let size = b.const_value(tmp(MockType::U64), ConstValue::U64(5))?;
                b.init_blk(target, value, size, None)?;
                b.call_void(Callee::Direct(7), vec![target, size])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(Machine {
            byte_order: Some(order),
            ..MACHINE
        });
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete());
        assert_eq!(
            facts.call_arguments(0, 4).first().map(|a| &a.bytes),
            Some(&expected),
            "{order:?} element {width}"
        );
    }
}

/// Runs a budget regression with a bounded wait even if the interpreter loops.
fn completion_with_timeout(build: fn() -> SsaFunction<MockTarget>) -> Completion {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let ssa = build();
        let context = AnalysisContext::new(MACHINE)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionSteps, 16));
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.constants().is_empty());
        let _ = sender.send(facts.completion());
    });
    let completion = receiver.recv_timeout(Duration::from_secs(10));
    assert!(
        completion.is_ok(),
        "static execution did not finish within ten seconds"
    );
    completion.unwrap_or_else(abort)
}

/// Every control-only iteration consumes work, so finite budgets always stop it.
#[test]
fn a_control_only_loop_exhausts_its_budget() {
    let completion = completion_with_timeout(|| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |b| b.jump(0)).unwrap_or_else(abort);
        builder.finish().unwrap_or_else(abort)
    });
    assert!(
        matches!(completion, Completion::Exhausted(exhaustion) if exhaustion.budget == Budget::ExecutionSteps && exhaustion.limit == 16)
    );
}

/// Calls without intervening arithmetic consume work even when opaque.
#[test]
fn a_call_only_loop_exhausts_its_budget() {
    let completion = completion_with_timeout(|| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |b| {
                b.call_void(Callee::Direct(7), vec![])?;
                b.jump(0)
            })
            .unwrap_or_else(abort);
        builder.finish().unwrap_or_else(abort)
    });
    assert!(
        matches!(completion, Completion::Exhausted(exhaustion) if exhaustion.budget == Budget::ExecutionSteps)
    );
}

/// Losing a stack-derived index must invalidate every potentially addressed byte.
#[test]
fn unknown_stack_offsets_do_not_preserve_stale_plaintext() {
    for unknown in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let (stack, input) = {
            let ssa = builder.function_mut();
            let stack = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            ssa.variable_mut(stack)
                .unwrap_or_else(|| abort(()))
                .set_role(VariableRole::EntryStackPointer);
            let input = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::I64,
            );
            (stack, input)
        };
        builder
            .in_block(0, |b| {
                let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -16,
                    result_type: MockType::Ptr,
                })?;
                let contents =
                    b.const_value(tmp(MockType::U64), ConstValue::U64(0x0807_0605_0403_0201))?;
                b.store_indirect(buffer, contents, MockType::U64)?;
                let index = if unknown {
                    input
                } else {
                    b.const_value(tmp(MockType::I64), ConstValue::I64(3))?
                };
                let target = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: buffer,
                    index: Some(index),
                    stride: 1,
                    offset: 0,
                    result_type: MockType::Ptr,
                })?;
                let zero = b.const_value(tmp(MockType::U8), ConstValue::U8(0))?;
                b.store_indirect(target, zero, MockType::U8)?;
                b.call_void(analyssa::ir::Callee::Direct(99), vec![buffer])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let facts = run(&ssa);
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let calls = facts.calls();
        let (_, arguments) = calls.first().unwrap_or_else(|| abort(()));
        if unknown {
            assert!(arguments.is_empty());
        } else {
            assert_eq!(
                arguments.first().map(|bytes| bytes.bytes.as_slice()),
                Some([1, 2, 3, 0, 5, 6, 7, 8].as_slice())
            );
        }
    }
}

/// An unknown loop bound must converge with a separate header and body as well.
#[test]
fn unknown_bounds_keep_arrival_history_after_an_exit_path_returns() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let bound = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I64,
    );
    let (zero, one) = builder
        .in_block(0, |b| {
            let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            b.jump(1)?;
            Ok((zero, one))
        })
        .unwrap_or_else(abort);
    let index = builder
        .in_block(1, |b| b.empty_phi(tmp(MockType::I64)))
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| b.branch_cmp(index, bound, CmpKind::Lt, false, 2, 3))
        .unwrap_or_else(abort);
    let next = builder
        .in_block(2, |b| {
            let next = b.add(tmp(MockType::I64), index, one)?;
            b.jump(1)?;
            Ok(next)
        })
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, index, 0, zero)
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, index, 2, next)
        .unwrap_or_else(abort);
    builder.in_block(3, |b| b.ret(None)).unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let context = AnalysisContext::new(MACHINE)
        .with_budget(WorkBudget::default().with_limit(Budget::ExecutionSteps, 1000));
    let facts = StaticExecution::new(&ssa, &context)
        .with_host(&FrameHost)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert!(facts.usage().steps < 1000);
}

/// Loading partial pointer bits must not erase their ability to address the stack.
#[test]
fn partially_overwritten_stack_pointers_invalidate_their_pointees() {
    for partial in [false, true] {
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
            .unwrap_or_else(|| abort(()))
            .set_role(VariableRole::EntryStackPointer);
        builder
            .in_block(0, |b| {
                let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -32,
                    result_type: MockType::Ptr,
                })?;
                let slot = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -16,
                    result_type: MockType::Ptr,
                })?;
                let contents =
                    b.const_value(tmp(MockType::U64), ConstValue::U64(0x0807_0605_0403_0201))?;
                let zero = b.const_value(tmp(MockType::U8), ConstValue::U8(0))?;
                b.store_indirect(buffer, contents, MockType::U64)?;
                b.store_indirect(slot, buffer, MockType::Ptr)?;
                if partial {
                    b.store_indirect(slot, zero, MockType::U8)?;
                }
                let reloaded = b.load_indirect(tmp(MockType::Ptr), slot, MockType::Ptr)?;
                b.store_indirect(reloaded, zero, MockType::U8)?;
                b.call_void(analyssa::ir::Callee::Direct(99), vec![buffer])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let facts = run(&ssa);
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let calls = facts.calls();
        let (_, arguments) = calls.first().unwrap_or_else(|| abort(()));
        if partial {
            assert!(arguments.is_empty());
        } else {
            assert_eq!(
                arguments.first().map(|bytes| bytes.bytes.as_slice()),
                Some([0, 2, 3, 4, 5, 6, 7, 8].as_slice())
            );
        }
    }
}

/// Stack alignment retains one symbolic base without assuming an entry SP value.
#[test]
fn aligned_stack_locals_retain_identity_and_conservative_aliasing() {
    for alias in [false, true] {
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
            .unwrap_or_else(|| abort(()))
            .set_role(VariableRole::EntryStackPointer);
        builder
            .in_block(0, |b| {
                let mask = b.const_value(tmp(MockType::I64), ConstValue::I64(-16))?;
                let aligned = b.and(tmp(MockType::U64), stack, mask)?;
                let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: aligned,
                    index: None,
                    stride: 1,
                    offset: -64,
                    result_type: MockType::Ptr,
                })?;
                let contents =
                    b.const_value(tmp(MockType::U64), ConstValue::U64(0x0807_0605_0403_0201))?;
                b.store_indirect(buffer, contents, MockType::U64)?;
                if alias {
                    let other = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: stack,
                        index: None,
                        stride: 1,
                        offset: -64,
                        result_type: MockType::Ptr,
                    })?;
                    b.store_indirect(other, contents, MockType::U64)?;
                }
                b.call_void(analyssa::ir::Callee::Direct(99), vec![buffer])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let facts = run(&ssa);
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let calls = facts.calls();
        let (_, arguments) = calls.first().unwrap_or_else(|| abort(()));
        if alias {
            assert!(arguments.is_empty());
        } else {
            assert_eq!(
                arguments.first().map(|bytes| bytes.bytes.as_slice()),
                Some([1, 2, 3, 4, 5, 6, 7, 8].as_slice())
            );
        }
    }
}

/// A proven checked overflow ends its path before observing a later consumer.
#[test]
fn checked_overflow_stops_before_unreachable_observations() {
    for kind in 0..3 {
        for overflow in [false, true] {
            for unknown in [false, true] {
                let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
                let input = builder.function_mut().create_variable(
                    VariableOrigin::Argument(0),
                    0,
                    DefSite::entry(),
                    MockType::U8,
                );
                builder
                    .in_block(0, |b| {
                        let left = b.const_value(
                            tmp(MockType::U8),
                            ConstValue::U8(if kind == 1 { 0 } else { 255 }),
                        )?;
                        let right = if unknown {
                            input
                        } else {
                            b.const_value(
                                tmp(MockType::U8),
                                ConstValue::U8(if overflow && kind == 2 {
                                    2
                                } else {
                                    u8::from(overflow)
                                }),
                            )?
                        };
                        b.emit_def(tmp(MockType::U8), |dest| match kind {
                            0 => SsaOp::AddOvf {
                                dest,
                                left,
                                right,
                                unsigned: true,
                                flags: None,
                            },
                            1 => SsaOp::SubOvf {
                                dest,
                                left,
                                right,
                                unsigned: true,
                                flags: None,
                            },
                            _ => SsaOp::MulOvf {
                                dest,
                                left,
                                right,
                                unsigned: true,
                                flags: None,
                            },
                        })?;
                        b.call_void(analyssa::ir::Callee::Direct(99), vec![])?;
                        b.ret(None)
                    })
                    .unwrap_or_else(abort);
                let ssa = builder.finish().unwrap_or_else(abort);
                let facts = run(&ssa);
                assert!(facts.is_complete(), "{:?}", facts.completion());
                let trapped = overflow && !unknown;
                assert_eq!(facts.calls().len(), usize::from(!trapped));
                assert_eq!(facts.usage().traps, usize::from(trapped));
            }
        }
    }
}

/// A proven trap discards only its path; another admitted path can still reach its consumer.
#[test]
fn checked_overflow_keeps_other_branch_observations() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U8,
    );
    builder
        .in_block(0, |b| b.branch(input, 1, 2))
        .unwrap_or_else(abort);
    builder
        .in_block(1, |b| {
            let max = b.const_value(tmp(MockType::U8), ConstValue::U8(255))?;
            let one = b.const_value(tmp(MockType::U8), ConstValue::U8(1))?;
            b.emit_def(tmp(MockType::U8), |dest| SsaOp::AddOvf {
                dest,
                left: max,
                right: one,
                unsigned: true,
                flags: None,
            })?;
            b.call_void(analyssa::ir::Callee::Direct(91), vec![])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    builder
        .in_block(2, |b| {
            b.call_void(analyssa::ir::Callee::Direct(92), vec![])?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let facts = run(&ssa);
    assert!(facts.is_complete());
    assert_eq!(facts.calls().len(), 1);
    assert_eq!(facts.usage().traps, 1);
    assert_eq!(facts.calls().first().map(|(site, _)| *site), Some((2, 0)));
}

/// A pointer observation is a linear prefix even when the underlying machine wraps.
#[test]
fn call_observations_stop_before_image_address_wrap() {
    for width in [
        PointerSize::Bit8,
        PointerSize::Bit16,
        PointerSize::Bit32,
        PointerSize::Bit64,
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |b| {
                let first = b.const_value(tmp(MockType::Ptr), ConstValue::Symbol(0))?;
                let last = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: first,
                    index: None,
                    stride: 1,
                    offset: -1,
                    result_type: MockType::Ptr,
                })?;
                let x = b.const_value(tmp(MockType::U8), ConstValue::U8(b'x'))?;
                let y = b.const_value(tmp(MockType::U8), ConstValue::U8(b'y'))?;
                b.store_indirect(last, x, MockType::U8)?;
                b.store_indirect(first, y, MockType::U8)?;
                b.call_void(analyssa::ir::Callee::Direct(99), vec![last])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(Machine {
            ptr_size: width,
            ..MACHINE
        });
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let calls = facts.calls();
        let (_, arguments) = calls.first().unwrap_or_else(|| abort(()));
        assert_eq!(
            arguments.first().map(|argument| argument.bytes.as_slice()),
            Some(b"x".as_slice())
        );
        assert_eq!(
            arguments
                .first()
                .map(|argument| argument.address.pointer_size()),
            Some(width)
        );
    }
}

/// **The bytes a run wrote are observed at every use of a pointer to them.**
/// A root builds `"hi"` in its frame, reads its first byte, stores the
/// pointer to it in another slot and returns it: the load, the store and the
/// return each show the bytes behind the pointer, as the run left them there.
/// A load of the image's own bytes shows nothing, since the image holds them
/// unchanged.
#[test]
fn bytes_a_run_wrote_are_observed_at_each_use_of_a_pointer_to_them() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let entry = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = builder.function_mut().variable_mut(entry) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    builder
        .in_block(0, |b| {
            let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: entry,
                index: None,
                stride: 0,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            for (offset, byte) in [(0, b'h'), (1, b'i'), (2, 0)] {
                let at = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: buffer,
                    index: None,
                    stride: 0,
                    offset,
                    result_type: MockType::Ptr,
                })?;
                let value = b.const_value(tmp(MockType::U8), ConstValue::U8(byte))?;
                b.store_indirect(at, value, MockType::U8)?;
            }
            // Instruction 10: a load through the pointer.
            b.load_indirect(tmp(MockType::U8), buffer, MockType::U8)?;
            // Instruction 12: the pointer stored in another slot.
            let slot = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: entry,
                index: None,
                stride: 0,
                offset: -24,
                result_type: MockType::Ptr,
            })?;
            b.store_indirect(slot, buffer, MockType::Ptr)?;
            // Instruction 14: a load of the image's own bytes.
            let image = b.const_value(tmp(MockType::Ptr), ConstValue::U64(ENCODED_AT))?;
            b.load_indirect(tmp(MockType::U8), image, MockType::U8)?;
            b.ret(Some(buffer))
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let facts = run(&ssa);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let shown = |bytes: &ArgumentBytes| (bytes.argument, bytes.address, bytes.bytes.clone());
    let uses: Vec<_> = facts
        .uses()
        .into_iter()
        .map(|(site, bytes)| (site, bytes.into_iter().map(shown).collect::<Vec<_>>()))
        .collect();
    let hi = || b"hi\0".to_vec();
    assert_eq!(
        uses,
        [
            ((0, 10), vec![(0, stack(-16), hi())]),
            ((0, 12), vec![(1, stack(-16), hi())]),
        ]
    );
    assert_eq!(
        facts.returned_bytes().map(shown),
        Some((0, stack(-16), hi()))
    );
    let frame = ByteBound::Object {
        start: -32,
        bytes: 32,
        kind: ObjectKind::Frame,
    };
    for (_, bytes) in facts.uses() {
        assert!(bytes.iter().all(|bytes| bytes.bound == frame), "{bytes:?}");
    }
    assert_eq!(facts.returned_bytes().map(|bytes| bytes.bound), Some(frame));
}

/// **The frame a pointer lies in ends the run behind it.** A root writes
/// eight bytes from four below its entry stack pointer, the last four past
/// the 32-byte frame the host promises, and loads through a pointer to the
/// first: the load shows the frame's last four bytes, the whole of what is
/// left of the frame there, not the four written past it.
#[test]
fn the_frame_a_pointer_lies_in_ends_its_run() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let entry = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = builder.function_mut().variable_mut(entry) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    builder
        .in_block(0, |b| {
            let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: entry,
                index: None,
                stride: 0,
                offset: -4,
                result_type: MockType::Ptr,
            })?;
            for offset in 0..8 {
                let at = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: buffer,
                    index: None,
                    stride: 0,
                    offset,
                    result_type: MockType::Ptr,
                })?;
                let value = b.const_value(
                    tmp(MockType::U8),
                    ConstValue::U8(u8::try_from(offset).unwrap_or_else(abort)),
                )?;
                b.store_indirect(at, value, MockType::U8)?;
            }
            b.load_indirect(tmp(MockType::U8), buffer, MockType::U8)?;
            b.ret(None)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let facts = run(&ssa);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let uses = facts.uses();
    assert_eq!(uses.len(), 1, "one use: {uses:?}");
    let seen = uses
        .first()
        .map(|(_, seen)| seen)
        .unwrap_or_else(|| abort(()));
    assert_eq!(seen.len(), 1, "one operand: {seen:?}");
    let bytes = seen.first().unwrap_or_else(|| abort(()));
    assert_eq!(bytes.address, stack(-4));
    assert_eq!(bytes.bytes, [0, 1, 2, 3]);
    assert_eq!(
        bytes.bound,
        ByteBound::Object {
            start: -32,
            bytes: 32,
            kind: ObjectKind::Frame,
        }
    );
    assert!(bytes.complete(), "the rest of the frame is known");
}

/// **A block compare shows the bytes the run wrote on the side that holds
/// them.** A root builds `"hi"` in its frame and compares it with the
/// image's bytes, as an inlined `memcmp` against a constant does: the
/// compare shows `"hi"` behind its left operand, and nothing behind its
/// right, the image's own unchanged bytes.
#[test]
fn a_block_compare_shows_the_written_side() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let entry = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = builder.function_mut().variable_mut(entry) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    let site = builder
        .in_block(0, |b| {
            let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: entry,
                index: None,
                stride: 0,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            for (offset, byte) in [(0, b'h'), (1, b'i')] {
                let at = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: buffer,
                    index: None,
                    stride: 0,
                    offset,
                    result_type: MockType::Ptr,
                })?;
                let value = b.const_value(tmp(MockType::U8), ConstValue::U8(byte))?;
                b.store_indirect(at, value, MockType::U8)?;
            }
            let image = b.const_value(tmp(MockType::Ptr), ConstValue::U64(ENCODED_AT))?;
            let count = b.const_value(tmp(MockType::U64), ConstValue::U64(2))?;
            let flags_in = b.const_value(tmp(MockType::U32), ConstValue::U32(0))?;
            let forward = b.const_value(tmp(MockType::Bool), ConstValue::False)?;
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
                    left: buffer,
                    right: image,
                    count,
                    flags_in: Some(flags_in),
                    backward: Some(forward),
                    element_bytes: 1,
                    while_equal: true,
                    left_access: AccessQualifiers::FLAT,
                    right_access: AccessQualifiers::FLAT,
                }))
            })?;
            b.ret(None)?;
            Ok(11)
        })
        .unwrap_or_else(abort);
    let ssa = builder.finish().unwrap_or_else(abort);
    let facts = run(&ssa);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let uses: Vec<_> = facts
        .uses()
        .into_iter()
        .map(|(at, bytes)| {
            (
                at,
                bytes
                    .iter()
                    .map(|bytes| (bytes.argument, bytes.address, bytes.bytes.clone()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert_eq!(uses, [((0, site), vec![(0, stack(-16), b"hi".to_vec())])]);
}

/// **A use that walks a buffer keeps the buffer from where the walk began.**
/// A root builds `"hi"` in its frame and reads it a byte per iteration, as a
/// loop comparing it with input does: each visit reads one byte further
/// along the same bytes, and the use shows them from the first.
#[test]
fn a_use_walking_a_buffer_shows_it_from_where_the_walk_began() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let entry = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = builder.function_mut().variable_mut(entry) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    let (buffer, zero, one, count) = builder
        .in_block(0, |b| {
            let buffer = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: entry,
                index: None,
                stride: 0,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            for (offset, byte) in [(0, b'h'), (1, b'i'), (2, 0)] {
                let at = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: buffer,
                    index: None,
                    stride: 0,
                    offset,
                    result_type: MockType::Ptr,
                })?;
                let value = b.const_value(tmp(MockType::U8), ConstValue::U8(byte))?;
                b.store_indirect(at, value, MockType::U8)?;
            }
            let zero = b.const_value(tmp(MockType::I64), ConstValue::I64(0))?;
            let one = b.const_value(tmp(MockType::I64), ConstValue::I64(1))?;
            let count = b.const_value(tmp(MockType::I64), ConstValue::I64(2))?;
            b.jump(1)?;
            Ok((buffer, zero, one, count))
        })
        .unwrap_or_else(abort);
    let index = builder
        .in_block(1, |b| b.empty_phi(tmp(MockType::I64)))
        .unwrap_or_else(abort);
    let next = builder
        .in_block(1, |b| {
            let at = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: buffer,
                index: Some(index),
                stride: 1,
                offset: 0,
                result_type: MockType::Ptr,
            })?;
            b.load_indirect(tmp(MockType::U8), at, MockType::U8)?;
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
    let ssa = builder.finish().unwrap_or_else(abort);
    let facts = run(&ssa);
    assert!(facts.is_complete(), "{:?}", facts.completion());
    let uses: Vec<_> = facts
        .uses()
        .into_iter()
        .map(|(at, bytes)| {
            (
                at,
                bytes
                    .iter()
                    .map(|bytes| (bytes.argument, bytes.address, bytes.bytes.clone()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert_eq!(uses, [((1, 1), vec![(0, stack(-16), b"hi\0".to_vec())])]);
}

/// What a function does with the bytes it loads from its own frame, below
/// its entry stack pointer, after storing four bytes at `sp-16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tail {
    /// Branches on the never-written `u32` at `sp-8`.
    Branch,
    /// Branches on that `u32` masked with zero.
    MaskedZero,
    /// Branches on that `u32` masked with `0xff`.
    MaskedByte,
    /// Copies the eight bytes at `sp-16`, four stored and four of padding,
    /// to `sp-32`, and returns nothing.
    Copy,
    /// Copies them, then returns the copied padding.
    CopyReturned,
    /// Loads through the never-written `u64` at `sp-8`.
    Address,
    /// Hands the never-written `u32` to a function it cannot follow.
    Argument,
    /// Stores the stored field's `u32` back and returns it.
    Written,
    /// Hands the never-written bytes at `sp-8` to `strlen`, which reads them.
    ModelRead,
    /// Hands them to `memcpy`, which copies them to `sp-32`.
    ModelCopy,
}

/// The method `strlen` is called through.
const LENGTH: u32 = 50;

/// The method `memcpy` is called through.
const COPY: u32 = 51;

/// Models `strlen` and `memcpy`.
struct StringHost;

impl ExecutionHost<MockTarget> for StringHost {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match *method {
            LENGTH => Some(&LibraryModel::Length),
            COPY => Some(&LibraryModel::Copy),
            _ => None,
        }
    }
}

/// The uses of never-written bits a run of `tail` records.
fn unwritten_uses(tail: Tail) -> Vec<UnwrittenUseKind> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = {
        let ssa = builder.function_mut();
        let stack = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        if let Some(variable) = ssa.variable_mut(stack) {
            variable.set_role(VariableRole::EntryStackPointer);
        }
        stack
    };
    let at = |b: &mut SsaBlockBuilder<'_, MockTarget>, offset| {
        b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 1,
            offset,
            result_type: MockType::Ptr,
        })
    };
    builder
        .in_block(0, |b| {
            let field = at(b, -16)?;
            let stored = b.const_value(tmp(MockType::U32), ConstValue::U32(7))?;
            b.store_indirect(field, stored, MockType::U32)?;
            let local = at(b, -8)?;
            match tail {
                Tail::Branch | Tail::MaskedZero | Tail::MaskedByte => {
                    let loaded = b.load_indirect(tmp(MockType::U32), local, MockType::U32)?;
                    let decided = match tail {
                        Tail::Branch => loaded,
                        _ => {
                            let mask = b.const_value(
                                tmp(MockType::U32),
                                ConstValue::U32(if tail == Tail::MaskedZero { 0 } else { 0xff }),
                            )?;
                            b.emit_def(tmp(MockType::U32), |dest| SsaOp::And {
                                dest,
                                left: loaded,
                                right: mask,
                                flags: None,
                            })?
                        }
                    };
                    b.branch(decided, 1, 1)
                }
                Tail::Copy | Tail::CopyReturned => {
                    let record = b.load_indirect(tmp(MockType::U64), field, MockType::U64)?;
                    let copy = at(b, -32)?;
                    b.store_indirect(copy, record, MockType::U64)?;
                    if tail == Tail::Copy {
                        return b.ret(None);
                    }
                    let padding = at(b, -28)?;
                    let copied = b.load_indirect(tmp(MockType::U32), padding, MockType::U32)?;
                    b.ret(Some(copied))
                }
                Tail::Address => {
                    let pointer = b.load_indirect(tmp(MockType::Ptr), local, MockType::Ptr)?;
                    let through = b.load_indirect(tmp(MockType::U8), pointer, MockType::U8)?;
                    b.ret(Some(through))
                }
                Tail::Argument => {
                    let loaded = b.load_indirect(tmp(MockType::U32), local, MockType::U32)?;
                    b.call_void(Callee::Direct(99), vec![loaded])?;
                    b.ret(None)
                }
                Tail::ModelRead => {
                    b.call(tmp(MockType::U64), Callee::Direct(LENGTH), vec![local])?;
                    b.ret(None)
                }
                Tail::ModelCopy => {
                    let copy = at(b, -32)?;
                    let bytes = b.const_value(tmp(MockType::U64), ConstValue::U64(8))?;
                    b.call(
                        tmp(MockType::Ptr),
                        Callee::Direct(COPY),
                        vec![copy, local, bytes],
                    )?;
                    b.ret(None)
                }
                Tail::Written => {
                    let loaded = b.load_indirect(tmp(MockType::U32), field, MockType::U32)?;
                    b.ret(Some(loaded))
                }
            }
        })
        .unwrap_or_else(abort);
    if matches!(tail, Tail::Branch | Tail::MaskedZero | Tail::MaskedByte) {
        builder.in_block(1, |b| b.ret(None)).unwrap_or_else(abort);
    }
    let ssa = builder.finish().unwrap_or_else(abort);
    let context = AnalysisContext::new(Machine {
        byte_order: Some(Endianness::Little),
        ..MACHINE
    });
    let facts = StaticExecution::new(&ssa, &context)
        .with_host(&StringHost)
        .run();
    facts
        .findings()
        .iter()
        .filter_map(|found| match found.kind {
            FindingKind::Unwritten(kind) => Some(kind),
            _ => None,
        })
        .collect()
}

/// **A use of bits never written is recorded; copying them is not a use.**
/// A branch on a local read before its store records one, and so does an
/// address formed from one, a call handed one and the root returning
/// copied padding; masking the local with zero writes every bit and records
/// nothing, masking it with `0xff` keeps its low byte never written; copying
/// a record with its padding, or returning a stored field, records nothing.
/// A model reading never-written bytes through a pointer it was handed
/// uses them; one copying them does not.
#[test]
fn a_use_of_bits_never_written_is_recorded_and_a_copy_is_not() {
    for (tail, expected) in [
        (Tail::Branch, vec![UnwrittenUseKind::Branch]),
        (Tail::MaskedZero, vec![]),
        (Tail::MaskedByte, vec![UnwrittenUseKind::Branch]),
        (Tail::Copy, vec![]),
        (Tail::CopyReturned, vec![UnwrittenUseKind::Return]),
        (Tail::Address, vec![UnwrittenUseKind::Address]),
        (Tail::Argument, vec![UnwrittenUseKind::Argument]),
        (Tail::Written, vec![]),
        (Tail::ModelRead, vec![UnwrittenUseKind::Argument]),
        (Tail::ModelCopy, vec![]),
    ] {
        assert_eq!(unwritten_uses(tail), expected, "{tail:?}");
    }
}

/// Refuses every write whose destination the execution cannot resolve.
#[derive(Debug)]
struct ResolvedWritesOnly;

impl MemoryWriteGuard for ResolvedWritesOnly {
    fn check_unresolved_write(
        &self,
        _pointer: Option<Address>,
        _len: Option<usize>,
    ) -> Result<(), MemoryRefusal> {
        Err(MemoryRefusal::UnresolvedWrite)
    }

    fn check_write(
        &self,
        _sources: &mut InstructionSources,
        _address: Address,
        _len: usize,
    ) -> Result<(), MemoryRefusal> {
        Ok(())
    }
}

/// Attests a 64-byte frame below the entry stack pointer and admits only
/// resolved writes.
struct IndexedStoreHost;

impl ExecutionHost<MockTarget> for IndexedStoreHost {
    fn stack_extent(
        &self,
        _body: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Range<i64>> {
        Some(-64..0)
    }

    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        Some(&ResolvedWritesOnly)
    }
}

/// **A store at a bounded offset inside one object writes some element of
/// it.** The frame holds known bytes at `sp-40` and `sp-12`; a byte is
/// stored at `sp-16` plus the parameter masked with `mask`, or the parameter
/// itself where there is none. Masked with 7 the store can reach only
/// `sp-16..sp-8`, inside the frame: the root runs on, the byte at `sp-12`
/// becomes unknown and the one at `sp-40` stays. Masked with `0xff` it can
/// reach past the frame, and unmasked anywhere: both stores are still
/// refused as unresolved.
#[test]
fn a_bounded_indexed_store_writes_some_element_of_its_object() {
    for mask in [Some(7u64), Some(0xff), None] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let (stack, index) = {
            let ssa = builder.function_mut();
            let stack = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            if let Some(variable) = ssa.variable_mut(stack) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            let index = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::U64,
            );
            (stack, index)
        };
        let at = |b: &mut SsaBlockBuilder<'_, MockTarget>, index, offset| {
            b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index,
                stride: 1,
                offset,
                result_type: MockType::Ptr,
            })
        };
        let (kept, forgotten) = builder
            .in_block(0, |b| {
                let far = at(b, None, -40)?;
                let near = at(b, None, -12)?;
                let first = b.const_value(tmp(MockType::U8), ConstValue::U8(0x11))?;
                let second = b.const_value(tmp(MockType::U8), ConstValue::U8(0x22))?;
                b.store_indirect(far, first, MockType::U8)?;
                b.store_indirect(near, second, MockType::U8)?;
                let offset = match mask {
                    Some(mask) => {
                        let mask = b.const_value(tmp(MockType::U64), ConstValue::U64(mask))?;
                        b.emit_def(tmp(MockType::U64), |dest| SsaOp::And {
                            dest,
                            left: index,
                            right: mask,
                            flags: None,
                        })?
                    }
                    None => index,
                };
                let element = at(b, Some(offset), -16)?;
                let byte = b.const_value(tmp(MockType::U8), ConstValue::U8(0x41))?;
                b.store_indirect(element, byte, MockType::U8)?;
                let kept = b.load_indirect(tmp(MockType::U8), far, MockType::U8)?;
                let forgotten = b.load_indirect(tmp(MockType::U8), near, MockType::U8)?;
                b.ret(Some(kept))?;
                Ok((kept, forgotten))
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        });
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&IndexedStoreHost)
            .run();
        if mask == Some(7) {
            assert!(facts.is_complete(), "{mask:?}: {:?}", facts.completion());
            assert_eq!(facts.constant(kept), Some(&ConstValue::U8(0x11)));
            assert_eq!(facts.constant(forgotten), None);
        } else {
            assert!(
                matches!(
                    facts.completion(),
                    Completion::Memory {
                        reason: MemoryRefusal::UnresolvedWrite,
                        ..
                    }
                ),
                "{mask:?}: {:?}",
                facts.completion()
            );
        }
    }
}

/// A store shape [`every_store_shape_at_a_bounded_offset_writes_some_element`]
/// makes through its element pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundedShape {
    /// A store of a truth value: one bit, stored in a byte.
    Truth,
    /// A masked vector store of sixteen bytes.
    MaskedVector,
    /// A structure store of two sixteen-byte segments.
    Segments,
    /// A four-byte block fill walking up.
    FillUp,
    /// A four-byte block fill walking down from its pointer.
    FillDown,
    /// A four-byte block copy into it.
    Copy,
    /// A four-byte block move into it.
    Move,
    /// A block fill at the exact address the element pointer starts from,
    /// as long as the masked argument.
    FillSized,
    /// A block copy there, as long as the masked argument.
    CopySized,
    /// A four-byte atomic store.
    AtomicStore,
    /// A four-byte atomic exchange.
    Exchange,
    /// A four-byte atomic load.
    AtomicLoad,
}

impl BoundedShape {
    /// Every shape.
    const ALL: [Self; 12] = [
        Self::Truth,
        Self::MaskedVector,
        Self::Segments,
        Self::FillUp,
        Self::FillDown,
        Self::Copy,
        Self::Move,
        Self::FillSized,
        Self::CopySized,
        Self::AtomicStore,
        Self::Exchange,
        Self::AtomicLoad,
    ];

    /// Whether it is atomic, which a run without an exact address refuses
    /// by the atomic contract rather than as an unresolved write.
    const fn atomic(self) -> bool {
        matches!(self, Self::AtomicStore | Self::Exchange | Self::AtomicLoad)
    }

    /// Whether it writes.
    const fn writes(self) -> bool {
        !matches!(self, Self::AtomicLoad)
    }
}

/// **Every store shape at a bounded offset inside one object writes some
/// element of it.** The frame holds known bytes at `sp-60` and `sp-46`; each
/// shape reaches memory at `sp-48` plus the first argument masked with
/// `mask`: a store of a truth value, one bit its type's byte holds, a masked
/// vector store, a two-segment structure store, a block
/// fill walking up and one walking down, a block copy and a block move into
/// it, an atomic store, an exchange and an atomic load; and a block fill and
/// a block copy at `sp-48` itself, as long as the masked argument. Masked with 7 every
/// byte it may reach lies in the frame: the root runs on, a writing shape
/// forgets the byte at `sp-46`, which it may reach, and every shape keeps the
/// one at `sp-60`, which none reaches. Masked with `0xff` it may reach past
/// the frame: a plain shape is refused as an unresolved write, an atomic one
/// by the atomic contract, as before.
#[test]
fn every_store_shape_at_a_bounded_offset_writes_some_element() {
    for shape in BoundedShape::ALL {
        for mask in [7u64, 0xff] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
            let (stack, index, lanes, vector) = {
                let ssa = builder.function_mut();
                let stack = ssa.create_variable(
                    VariableOrigin::EntryLiveIn,
                    0,
                    DefSite::entry(),
                    MockType::Ptr,
                );
                if let Some(variable) = ssa.variable_mut(stack) {
                    variable.set_role(VariableRole::EntryStackPointer);
                }
                let mut argument = |number, ty| {
                    ssa.create_variable(VariableOrigin::Argument(number), 0, DefSite::entry(), ty)
                };
                (
                    stack,
                    argument(0, MockType::U64),
                    argument(1, MockType::Mask4),
                    argument(2, MockType::V4I32),
                )
            };
            let at = |b: &mut SsaBlockBuilder<'_, MockTarget>, index, offset| {
                b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index,
                    stride: 1,
                    offset,
                    result_type: MockType::Ptr,
                })
            };
            let (kept, reached) = builder
                .in_block(0, |b| {
                    let far = at(b, None, -60)?;
                    let near = at(b, None, -46)?;
                    let source = at(b, None, -8)?;
                    let first = b.const_value(tmp(MockType::U8), ConstValue::U8(0x11))?;
                    let second = b.const_value(tmp(MockType::U8), ConstValue::U8(0x22))?;
                    b.store_indirect(far, first, MockType::U8)?;
                    b.store_indirect(near, second, MockType::U8)?;
                    let mask = b.const_value(tmp(MockType::U64), ConstValue::U64(mask))?;
                    let offset = b.emit_def(tmp(MockType::U64), |dest| SsaOp::And {
                        dest,
                        left: index,
                        right: mask,
                        flags: None,
                    })?;
                    let element = at(b, Some(offset), -48)?;
                    let start = at(b, None, -48)?;
                    let four = b.const_value(tmp(MockType::U64), ConstValue::U64(4))?;
                    let byte = b.const_value(tmp(MockType::U8), ConstValue::U8(0x41))?;
                    let word = b.const_value(tmp(MockType::U32), ConstValue::U32(0x41))?;
                    let down = b.const_value(tmp(MockType::Bool), ConstValue::True)?;
                    let access = AccessQualifiers::FLAT;
                    match shape {
                        BoundedShape::Truth => {
                            let truth = b.const_value(tmp(MockType::Bool), ConstValue::True)?;
                            b.store_indirect(element, truth, MockType::Bool)?;
                        }
                        BoundedShape::MaskedVector => b.emit_no_defs(SsaOp::VectorMaskedStore {
                            addr: element,
                            value: vector,
                            mask: lanes,
                            vector_type: MockType::V4I32,
                            access,
                        })?,
                        BoundedShape::Segments => b.emit_no_defs(SsaOp::VectorSegmentStore {
                            base: element,
                            values: vec![vector, vector],
                            mask: None,
                            vector_type: MockType::V4I32,
                            segments: 2,
                            layout: VectorSegmentLayout::Interleaved,
                            access,
                        })?,
                        BoundedShape::FillUp | BoundedShape::FillDown => {
                            b.emit_no_defs(SsaOp::InitBlk {
                                dest_addr: element,
                                value: byte,
                                size: four,
                                backward: (shape == BoundedShape::FillDown).then_some(down),
                                access,
                            })?;
                        }
                        BoundedShape::Copy => b.emit_no_defs(SsaOp::CopyBlk {
                            dest_addr: element,
                            src_addr: source,
                            size: four,
                            element_bytes: 1,
                            backward: None,
                            src_access: access,
                            dest_access: access,
                        })?,
                        BoundedShape::Move => b.emit_no_defs(SsaOp::MoveBlk {
                            dest_addr: element,
                            src_addr: source,
                            size: four,
                            src_access: access,
                            dest_access: access,
                        })?,
                        BoundedShape::FillSized => b.emit_no_defs(SsaOp::InitBlk {
                            dest_addr: start,
                            value: byte,
                            size: offset,
                            backward: None,
                            access,
                        })?,
                        BoundedShape::CopySized => b.emit_no_defs(SsaOp::CopyBlk {
                            dest_addr: start,
                            src_addr: source,
                            size: offset,
                            element_bytes: 1,
                            backward: None,
                            src_access: access,
                            dest_access: access,
                        })?,
                        BoundedShape::AtomicStore => b.emit_no_defs(SsaOp::AtomicStore {
                            addr: element,
                            value: word,
                            value_type: MockType::U32,
                            ordering: AtomicOrdering::SeqCst,
                            width: AtomicAccessWidth::Bits32,
                            access,
                        })?,
                        BoundedShape::Exchange => {
                            b.emit_def(tmp(MockType::U32), |dest| SsaOp::AtomicExchange {
                                dest,
                                addr: element,
                                value: word,
                                ordering: AtomicOrdering::SeqCst,
                                width: AtomicAccessWidth::Bits32,
                                access,
                            })?;
                        }
                        BoundedShape::AtomicLoad => {
                            b.emit_def(tmp(MockType::U32), |dest| SsaOp::AtomicLoad {
                                dest,
                                addr: element,
                                value_type: MockType::U32,
                                ordering: AtomicOrdering::Acquire,
                                width: AtomicAccessWidth::Bits32,
                                reserves: false,
                                access,
                            })?;
                        }
                    }
                    let kept = b.load_indirect(tmp(MockType::U8), far, MockType::U8)?;
                    let reached = b.load_indirect(tmp(MockType::U8), near, MockType::U8)?;
                    b.ret(Some(kept))?;
                    Ok((kept, reached))
                })
                .unwrap_or_else(abort);
            let ssa = builder.finish().unwrap_or_else(abort);
            let context = AnalysisContext::new(Machine {
                byte_order: Some(Endianness::Little),
                ..MACHINE
            });
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&IndexedStoreHost)
                .run();
            if mask == 7 {
                assert!(
                    facts.is_complete(),
                    "{shape:?} {mask}: {:?}",
                    facts.completion()
                );
                assert_eq!(
                    facts.constant(kept),
                    Some(&ConstValue::U8(0x11)),
                    "{shape:?}"
                );
                let left = (!shape.writes()).then_some(&ConstValue::U8(0x22));
                assert_eq!(facts.constant(reached), left, "{shape:?}");
            } else {
                let refused = if shape.atomic() {
                    MemoryRefusal::AtomicContract
                } else {
                    MemoryRefusal::UnresolvedWrite
                };
                assert!(
                    matches!(
                        facts.completion(),
                        Completion::Memory { reason, .. } if reason == refused
                    ),
                    "{shape:?} {mask}: {:?}",
                    facts.completion()
                );
            }
        }
    }
}

/// **A conditional trap writes nothing, and ends a path only where it
/// fires.** The frame holds a known byte at `sp-12`; a `teq` compares the
/// argument with zero, as MIPS checks a divisor, and the byte is read back.
/// On an unknown argument the trap may or may not fire: the root runs on
/// under a guard that refuses every unresolved write, and the byte stays
/// known. Comparing zero with zero it fires on every path, and no path
/// returns.
#[test]
fn a_conditional_trap_writes_nothing() {
    for fires in [false, true] {
        let mut kept = None;
        let ssa = of_argument(|builder, stack, argument| {
            kept = Some(
                builder
                    .in_block(0, |b| {
                        let near = b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: stack,
                            index: None,
                            stride: 1,
                            offset: -12,
                            result_type: MockType::Ptr,
                        })?;
                        let byte = b.const_value(tmp(MockType::U8), ConstValue::U8(0x22))?;
                        b.store_indirect(near, byte, MockType::U8)?;
                        let zero = b.const_value(tmp(MockType::U64), ConstValue::U64(0))?;
                        let compared = if fires { zero } else { argument };
                        b.emit_no_defs(SsaOp::SystemOp(Box::new(NativeKindedData {
                            kind: SystemOpKind::Trap {
                                op: TrapOp::TrapIfEqual,
                                vector: Some(7),
                            },
                            mnemonic: String::new(),
                            metadata: None,
                            outputs: Vec::new(),
                            inputs: vec![compared, zero],
                        })))?;
                        let kept = b.load_indirect(tmp(MockType::U8), near, MockType::U8)?;
                        b.ret(Some(kept))?;
                        Ok(kept)
                    })
                    .unwrap_or_else(abort),
            );
        });
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        });
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&IndexedStoreHost)
            .run();
        assert!(facts.is_complete(), "{fires}: {:?}", facts.completion());
        let kept = kept.unwrap_or_else(|| abort(()));
        if fires {
            assert_eq!(facts.returned(), None);
        } else {
            assert_eq!(facts.constant(kept), Some(&ConstValue::U8(0x22)));
        }
    }
}

/// A function of one 64-bit argument, built by `build` over its blocks.
fn of_argument(
    build: impl FnOnce(&mut SsaFunctionBuilder<MockTarget>, SsaVarId, SsaVarId),
) -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let (stack, argument) = {
        let ssa = builder.function_mut();
        let stack = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        if let Some(variable) = ssa.variable_mut(stack) {
            variable.set_role(VariableRole::EntryStackPointer);
        }
        let argument = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::U64,
        );
        (stack, argument)
    };
    build(&mut builder, stack, argument);
    builder.finish().unwrap_or_else(abort)
}

/// **A store indexed by a value its branch bounded writes some element of
/// its object.** The frame holds known bytes at `sp-40` and `sp-12`; where
/// the argument is below eight, unsigned, a byte is stored at `sp-16` plus
/// the argument. The side that took the branch knows the argument's high
/// bits clear, so the store can reach only `sp-16..sp-8`: the root runs on,
/// the byte at `sp-12` becomes unknown on that side and the one at `sp-40`
/// stays on both. Bounded at 64 instead, it can reach past the frame and is
/// still refused as unresolved.
#[test]
fn a_store_indexed_by_a_branch_bounded_value_writes_some_element() {
    for bound in [8u64, 64] {
        let mut kept = None;
        let ssa = of_argument(|builder, stack, index| {
            let at = |b: &mut SsaBlockBuilder<'_, MockTarget>, index, offset| {
                b.emit_def(tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index,
                    stride: 1,
                    offset,
                    result_type: MockType::Ptr,
                })
            };
            let far = builder
                .in_block(0, |b| {
                    let far = at(b, None, -40)?;
                    let near = at(b, None, -12)?;
                    let first = b.const_value(tmp(MockType::U8), ConstValue::U8(0x11))?;
                    let second = b.const_value(tmp(MockType::U8), ConstValue::U8(0x22))?;
                    b.store_indirect(far, first, MockType::U8)?;
                    b.store_indirect(near, second, MockType::U8)?;
                    let limit = b.const_value(tmp(MockType::U64), ConstValue::U64(bound))?;
                    b.branch_cmp(index, limit, CmpKind::Lt, true, 1, 2)?;
                    Ok(far)
                })
                .unwrap_or_else(abort);
            builder
                .in_block(1, |b| {
                    let element = at(b, Some(index), -16)?;
                    let byte = b.const_value(tmp(MockType::U8), ConstValue::U8(0x41))?;
                    b.store_indirect(element, byte, MockType::U8)?;
                    b.jump(2)
                })
                .unwrap_or_else(abort);
            kept = Some(
                builder
                    .in_block(2, |b| {
                        let kept = b.load_indirect(tmp(MockType::U8), far, MockType::U8)?;
                        b.ret(Some(kept))?;
                        Ok(kept)
                    })
                    .unwrap_or_else(abort),
            );
        });
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        });
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&IndexedStoreHost)
            .run();
        if bound == 8 {
            assert!(facts.is_complete(), "{bound}: {:?}", facts.completion());
            let kept = kept.unwrap_or_else(|| abort(()));
            assert_eq!(facts.constant(kept), Some(&ConstValue::U8(0x11)));
        } else {
            assert!(
                matches!(
                    facts.completion(),
                    Completion::Memory {
                        reason: MemoryRefusal::UnresolvedWrite,
                        ..
                    }
                ),
                "{bound}: {:?}",
                facts.completion()
            );
        }
    }
}

/// **A branch on a value a taken branch fixed is decided.** Where the
/// argument is 5, a branch on the argument plus one being 6 goes one way
/// only: the run forks once, at the first branch. A branch on a difference
/// computed before the first (`argument - 5`, asked whether zero) on the side
/// that fixed the argument at 7 has one side no input satisfies: it forks no
/// further, and that side is counted infeasible.
#[test]
fn a_branch_on_a_value_its_path_fixed_is_decided() {
    let ssa = of_argument(|builder, _, argument| {
        builder
            .in_block(0, |b| {
                let five = b.const_value(tmp(MockType::U64), ConstValue::U64(5))?;
                b.branch_cmp(argument, five, CmpKind::Eq, false, 1, 3)
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let one = b.const_value(tmp(MockType::U64), ConstValue::U64(1))?;
                let next = b.add(tmp(MockType::U64), argument, one)?;
                let six = b.const_value(tmp(MockType::U64), ConstValue::U64(6))?;
                b.branch_cmp(next, six, CmpKind::Eq, false, 2, 3)
            })
            .unwrap_or_else(abort);
        for (block, value) in [(2usize, 1u64), (3, 0)] {
            builder
                .in_block(block, |b| {
                    let returned = b.const_value(tmp(MockType::U64), ConstValue::U64(value))?;
                    b.ret(Some(returned))
                })
                .unwrap_or_else(abort);
        }
    });
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.usage().paths, 1, "{:?}", facts.usage());
    assert_eq!(facts.usage().infeasible_paths, 0, "{:?}", facts.usage());

    let ssa = of_argument(|builder, _, argument| {
        builder
            .in_block(0, |b| {
                let five = b.const_value(tmp(MockType::U64), ConstValue::U64(5))?;
                let difference = b.emit_def(tmp(MockType::U64), |dest| SsaOp::Sub {
                    dest,
                    left: argument,
                    right: five,
                    flags: None,
                })?;
                let seven = b.const_value(tmp(MockType::U64), ConstValue::U64(7))?;
                b.branch_cmp(argument, seven, CmpKind::Eq, false, 1, 3)?;
                Ok(difference)
            })
            .and_then(|difference| {
                builder.in_block(1, |b| {
                    let zero = b.const_value(tmp(MockType::U64), ConstValue::U64(0))?;
                    b.branch_cmp(difference, zero, CmpKind::Eq, false, 2, 3)
                })
            })
            .unwrap_or_else(abort);
        for (block, value) in [(2usize, 1u64), (3, 0)] {
            builder
                .in_block(block, |b| {
                    let returned = b.const_value(tmp(MockType::U64), ConstValue::U64(value))?;
                    b.ret(Some(returned))
                })
                .unwrap_or_else(abort);
        }
    });
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.usage().paths, 1, "{:?}", facts.usage());
    assert_eq!(facts.usage().infeasible_paths, 1, "{:?}", facts.usage());
}

/// **Two branches on one flag, a merge between them, run on no common
/// path.** A branch on the argument as a flag sends each side through its
/// own block to one merge, and on by one edge to a block that branches on the
/// argument again. The side that took it as set knows it is not zero, and the
/// one that took it clear knows it is zero: arriving by the same edge they
/// stand apart, as the argument is read there, and the second branch goes
/// one way on each. The run forks once, and the second branch's other side
/// on the set path is no path.
#[test]
fn a_flag_branched_on_twice_across_a_merge_forks_once() {
    let ssa = of_argument(|builder, _, flag| {
        builder
            .in_block(0, |b| b.branch(flag, 1, 2))
            .unwrap_or_else(abort);
        for (block, next) in [(1usize, 3usize), (2, 3), (3, 4)] {
            builder
                .in_block(block, |b| b.jump(next))
                .unwrap_or_else(abort);
        }
        builder
            .in_block(4, |b| b.branch(flag, 5, 6))
            .unwrap_or_else(abort);
        for (block, value) in [(5usize, 1u64), (6, 0)] {
            builder
                .in_block(block, |b| {
                    let returned = b.const_value(tmp(MockType::U64), ConstValue::U64(value))?;
                    b.ret(Some(returned))
                })
                .unwrap_or_else(abort);
        }
    });
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.usage().paths, 1, "{:?}", facts.usage());
    assert_eq!(facts.usage().infeasible_paths, 1, "{:?}", facts.usage());
}
