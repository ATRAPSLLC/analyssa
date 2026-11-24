//! Explicit stack low bits preserve physical aliases across masking and calls.

use std::error::Error;

use analyssa::{
    analysis::{
        SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{
            Address, CallSite, CallStack, Completion, ExecutionHost, InstructionSources,
            MemoryRefusal, MemoryWriteGuard, StackCongruence, StaticExecution, Value,
        },
    },
    ir::{
        CallData, Callee, ConstValue, SsaDefSpec, SsaFunction, SsaFunctionBuilder, SsaOp, SsaVarId,
        function::SsaBlockBuilder,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    testing::{MACHINE, MockTarget, MockType},
};

type Function = SsaFunction<MockTarget>;

/// Declares the actual entry-stack value independently of ordinary parameters.
fn stack(builder: &mut SsaFunctionBuilder<MockTarget>) -> Result<SsaVarId, Box<dyn Error>> {
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .ok_or("entry variable")?
        .set_role(VariableRole::EntryStackPointer);
    Ok(stack)
}

/// Moves an existing address without changing its root identity.
fn pointer(
    block: &mut SsaBlockBuilder<'_, MockTarget>,
    base: SsaVarId,
    offset: i64,
) -> analyssa::Result<SsaVarId> {
    block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
        dest,
        base,
        index: None,
        stride: 1,
        offset,
        result_type: MockType::Ptr,
    })
}

/// Checks every supplied SSA contract before interpreting the fixture.
fn verified(builder: SsaFunctionBuilder<MockTarget>) -> Result<Function, Box<dyn Error>> {
    let function = builder.finish()?;
    let errors = SsaVerifier::new(&function).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    Ok(function)
}

/// Stores through one view and returns the exact byte loaded through its alias.
fn aliased(plain_offset: i64, mask: i64, aligned_write: bool) -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let stack = stack(&mut builder)?;
    builder.in_block(0, |block| {
        let plain = pointer(block, stack, plain_offset)?;
        let integer = block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::PtrToInt {
            dest,
            operand: stack,
            target: MockType::I64,
        })?;
        let mask = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(mask))?;
        let aligned = block.and(SsaDefSpec::tmp(MockType::I64), integer, mask)?;
        let aligned = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::IntToPtr {
            dest,
            operand: aligned,
            target: MockType::Ptr,
        })?;
        let plain = pointer(block, plain, -1)?;
        let aligned = pointer(block, aligned, -1)?;
        let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(42))?;
        let (write, read) = if aligned_write {
            (aligned, plain)
        } else {
            (plain, aligned)
        };
        block.store_indirect(write, byte, MockType::U8)?;
        let loaded = block.load_indirect(SsaDefSpec::tmp(MockType::U8), read, MockType::U8)?;
        block.ret(Some(loaded))
    })?;
    verified(builder)
}

/// Keeps unresolved destinations explicit and supplies only declared entry contracts.
#[derive(Debug)]
struct Host {
    /// Optional body entered by the caller fixture.
    callee: Option<Function>,
    /// Caller value that actually names the callee entry stack.
    call_stack: Option<SsaVarId>,
    /// Whether the root receives an explicit low-bit assumption.
    root_fact: bool,
    /// Whether the callee declares the same physical entry requirement.
    callee_requirement: bool,
}

impl MemoryWriteGuard for Host {
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
        Err(MemoryRefusal::UnresolvedWrite)
    }
}

impl ExecutionHost<MockTarget> for Host {
    fn entry_stack_congruence(
        &self,
        _: &Function,
        method: Option<&u32>,
    ) -> Option<StackCongruence> {
        let supplied = if method == Some(&0) {
            self.root_fact
        } else {
            self.callee_requirement
        };
        if supplied {
            StackCongruence::from_low_bits(4, 8)
        } else {
            None
        }
    }
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        Some(self)
    }
    fn body(&self, method: &u32) -> Option<&Function> {
        (*method == 1).then_some(self.callee.as_ref()).flatten()
    }
    fn call_stack(&self, _: CallSite<'_, MockTarget>) -> Option<CallStack> {
        self.call_stack.map(|stack_pointer| CallStack {
            stack_pointer: Some(stack_pointer),
            adjustment: 0,
        })
    }
}

/// A declared residue preserves both directions of the same physical alias.
#[test]
fn explicit_root_low_bits_preserve_stack_aliases_in_both_directions() -> Result<(), Box<dyn Error>>
{
    for aligned_write in [false, true] {
        let function = aliased(-8, -16, aligned_write)?;
        let host = Host {
            callee: None,
            call_stack: None,
            root_fact: true,
            callee_requirement: false,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&function, &context)
            .with_method(0)
            .with_host(&host)
            .run();
        assert_eq!(facts.completion(), Completion::Complete);
        assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U8(42))));
    }
    Ok(())
}

/// A mask that is no low-bit clear (`-15`) or clears everything (`0`) never
/// turns an unknown destination into a private stack object.
#[test]
fn arbitrary_and_zero_masks_remain_conservative() -> Result<(), Box<dyn Error>> {
    for (root_fact, mask) in [(true, -15), (true, 0), (false, -15), (false, 0)] {
        let function = aliased(-8, mask, true)?;
        let host = Host {
            callee: None,
            call_stack: None,
            root_fact,
            callee_requirement: false,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&function, &context)
            .with_method(0)
            .with_host(&host)
            .run();
        assert!(
            matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::UnresolvedWrite,
                    ..
                }
            ),
            "{root_fact} {mask}: {:?}",
            facts.completion()
        );
        assert!(facts.returned().is_none());
    }
    Ok(())
}

/// Which pointer an access goes through: the entry stack's, or the frame
/// `and sp, -16` realigned from it.
#[derive(Clone, Copy, Debug)]
enum View {
    Entry,
    Realigned,
}

/// One realigned-frame case: the stores made in order, the byte loaded, and
/// what it holds (`None` for unknown).
type Case = (&'static [(View, i64, u8)], (View, i64), Option<u8>);

/// Stores each byte of `stores` at its offset through its view, then returns
/// the byte loaded at `load`. Nothing states the stack pointer's low bits.
fn realigned_frame(
    stores: &[(View, i64, u8)],
    load: (View, i64),
) -> Result<Function, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let stack = stack(&mut builder)?;
    builder.in_block(0, |block| {
        let integer = block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::PtrToInt {
            dest,
            operand: stack,
            target: MockType::I64,
        })?;
        let mask = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(-16))?;
        let aligned = block.and(SsaDefSpec::tmp(MockType::I64), integer, mask)?;
        let aligned = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::IntToPtr {
            dest,
            operand: aligned,
            target: MockType::Ptr,
        })?;
        let base = |view| match view {
            View::Entry => stack,
            View::Realigned => aligned,
        };
        for &(view, offset, byte) in stores {
            let at = pointer(block, base(view), offset)?;
            let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(byte))?;
            block.store_indirect(at, byte, MockType::U8)?;
        }
        let at = pointer(block, base(load.0), load.1)?;
        let loaded = block.load_indirect(SsaDefSpec::tmp(MockType::U8), at, MockType::U8)?;
        block.ret(Some(loaded))
    })?;
    verified(builder)
}

/// **A realigned frame is exact among itself, with no low bits stated, and
/// aliases only its window of the frame it was realigned from.** `and sp,
/// -16` lowers the entry pointer by less than 16, so a byte the realigned
/// frame stores at -1 lies somewhere in the entry frame's -16 through -1.
/// Read back through the realigned frame it is the byte stored. The entry
/// frame's bytes in that window, and the realigned byte after the entry frame
/// stores there, are unknown, not refused; a store above the window leaves the
/// realigned byte as it was. With no evidence of the low bits, the write
/// guard admits every one of these stores: the realigned frame is stack.
#[test]
fn a_realigned_frame_aliases_only_its_window() -> Result<(), Box<dyn Error>> {
    let cases: &[Case] = &[
        (
            &[(View::Realigned, -1, 42)],
            (View::Realigned, -1),
            Some(42),
        ),
        (
            &[(View::Entry, -16, 7), (View::Realigned, -1, 42)],
            (View::Entry, -16),
            None,
        ),
        (
            &[(View::Entry, -1, 7), (View::Realigned, -1, 42)],
            (View::Entry, -1),
            None,
        ),
        (
            &[(View::Realigned, -1, 42), (View::Entry, -16, 7)],
            (View::Realigned, -1),
            None,
        ),
        (
            &[(View::Realigned, -1, 42), (View::Entry, 0, 7)],
            (View::Realigned, -1),
            Some(42),
        ),
        (
            &[(View::Entry, -17, 7), (View::Realigned, -1, 42)],
            (View::Entry, -17),
            Some(7),
        ),
    ];
    for (stores, load, expected) in cases {
        let function = realigned_frame(stores, *load)?;
        let host = Host {
            callee: None,
            call_stack: None,
            root_fact: false,
            callee_requirement: false,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&function, &context)
            .with_method(0)
            .with_host(&host)
            .run();
        assert_eq!(
            facts.completion(),
            Completion::Complete,
            "{stores:?} {load:?}"
        );
        let returned = facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64);
        assert_eq!(returned, expected.map(u64::from), "{stores:?} {load:?}");
    }
    Ok(())
}

/// Stronger masks than the stated low bits, and no stated bits at all,
/// realign: the entry frame's byte in the window is unknown, and nothing is
/// refused.
#[test]
fn absent_or_weaker_low_bits_realign_the_frame() -> Result<(), Box<dyn Error>> {
    for (root_fact, mask) in [(false, -16), (true, -32)] {
        let function = aliased(-8, mask, true)?;
        let host = Host {
            callee: None,
            call_stack: None,
            root_fact,
            callee_requirement: false,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&function, &context)
            .with_method(0)
            .with_host(&host)
            .run();
        assert_eq!(
            facts.completion(),
            Completion::Complete,
            "{root_fact} {mask}"
        );
        assert_ne!(
            facts.returned(),
            Some(&Value::Const(ConstValue::U8(42))),
            "{root_fact} {mask}"
        );
    }
    Ok(())
}

/// Builds an actual mapped call whose entry coordinate can differ from the root's residue.
fn caller(offset: i64, tail: bool) -> Result<(Function, SsaVarId), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let stack = stack(&mut builder)?;
    let call_stack = builder.in_block(0, |block| {
        let call_stack = pointer(block, stack, offset)?;
        let result = block.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| {
            let call = CallData::new(Some(dest), Callee::Direct(1), vec![]);
            SsaOp::Call(Box::new(if tail { call.made_tail() } else { call }))
        })?;
        block.ret(Some(result))?;
        Ok(call_stack)
    })?;
    Ok((verified(builder)?, call_stack))
}

/// Normal and tail calls inherit the existing root base before the caller retires.
#[test]
fn mapped_call_and_tail_keep_root_base_distinct_from_callee_entry() -> Result<(), Box<dyn Error>> {
    for tail in [false, true] {
        for (offset, plain, requirement) in [(-40, 0, false), (-48, -8, true)] {
            for aligned_write in [false, true] {
                let (function, call_stack) = caller(offset, tail)?;
                let host = Host {
                    callee: Some(aliased(plain, -16, aligned_write)?),
                    call_stack: Some(call_stack),
                    root_fact: true,
                    callee_requirement: requirement,
                };
                let context = AnalysisContext::new(MACHINE);
                let facts = StaticExecution::new(&function, &context)
                    .with_method(0)
                    .with_host(&host)
                    .run();
                assert_eq!(
                    facts.completion(),
                    Completion::Complete,
                    "{tail} {offset} {aligned_write}"
                );
                assert_eq!(
                    facts.returned(),
                    Some(&Value::Const(ConstValue::U8(42))),
                    "{tail} {offset} {aligned_write}"
                );
            }
        }
    }
    Ok(())
}

/// A callee label neither overrides nor strengthens the path's stack evidence.
/// The root's low bits put the call at 0 mod 16, where the callee's label
/// claims 8: the callee runs on the stack it is passed, so its `and sp, -16`
/// is the pointer itself and the realigned byte is the plain one (42), as a
/// compiler that enters a local function below its convention's alignment
/// arranges. Without evidence (no root bits, or no mapped call stack) the
/// label adds none: the callee's frame is realigned, so the byte read through
/// the entry frame stays unknown.
#[test]
fn a_callee_label_neither_overrides_nor_strengthens_stack_evidence() -> Result<(), Box<dyn Error>> {
    for (root_fact, mapped, proved) in [
        (true, true, true),
        (false, true, false),
        (true, false, false),
    ] {
        let (function, call_stack) = caller(-40, false)?;
        let host = Host {
            callee: Some(aliased(0, -16, true)?),
            call_stack: mapped.then_some(call_stack),
            root_fact,
            callee_requirement: true,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&function, &context)
            .with_method(0)
            .with_host(&host)
            .run();
        assert_aliased(&facts, proved, &format!("{root_fact} {mapped}"));
    }
    Ok(())
}

/// Checks that `facts` completed and returned the aliased byte (42) exactly
/// when `proved`.
fn assert_aliased(
    facts: &analyssa::analysis::execution::StaticFacts<MockTarget>,
    proved: bool,
    label: &str,
) {
    assert_eq!(facts.completion(), Completion::Complete, "{label}");
    assert_eq!(
        facts.returned() == Some(&Value::Const(ConstValue::U8(42))),
        proved,
        "{label}: {:?}",
        facts.returned()
    );
}

/// Supplies an actual entry-relative measurement without requiring an SSA pointer.
struct EntryRelative {
    /// Underlying code and declared contracts.
    host: Host,
    /// Measured callee entry relative to the current activation entry.
    offset: Option<i64>,
}

impl ExecutionHost<MockTarget> for EntryRelative {
    fn entry_stack_congruence(
        &self,
        body: &Function,
        method: Option<&u32>,
    ) -> Option<StackCongruence> {
        self.host.entry_stack_congruence(body, method)
    }
    fn body(&self, method: &u32) -> Option<&Function> {
        self.host.body(method)
    }
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        Some(&self.host)
    }
    fn call_stack(&self, site: CallSite<'_, MockTarget>) -> Option<CallStack> {
        self.host.call_stack(site)
    }
    fn call_stack_from_entry(&self, _: CallSite<'_, MockTarget>) -> Option<i64> {
        self.offset
    }
}

/// Builds a call with an unrelated unknown input and no entry-stack SSA role.
fn caller_without_stack(tail: bool) -> Result<(Function, SsaVarId), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let unknown = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder.in_block(0, |block| {
        let result = block.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| {
            let call = CallData::new(Some(dest), Callee::Direct(1), vec![]);
            SsaOp::Call(Box::new(if tail { call.made_tail() } else { call }))
        })?;
        block.ret(Some(result))
    })?;
    let function = verified(builder)?;
    assert!(function.entry_stack_pointer().is_none());
    Ok((function, unknown))
}

/// Removing an unused SSA stack input preserves measured normal and tail entry coordinates.
#[test]
fn entry_relative_measurement_preserves_root_without_an_ssa_stack_input()
-> Result<(), Box<dyn Error>> {
    for tail in [false, true] {
        for (offset, plain, requirement) in [(-40, 0, false), (-48, -8, true)] {
            for aligned_write in [false, true] {
                let (function, _) = caller_without_stack(tail)?;
                let host = EntryRelative {
                    host: Host {
                        callee: Some(aliased(plain, -16, aligned_write)?),
                        call_stack: None,
                        root_fact: true,
                        callee_requirement: requirement,
                    },
                    offset: Some(offset),
                };
                let context = AnalysisContext::new(MACHINE);
                let facts = StaticExecution::new(&function, &context)
                    .with_method(0)
                    .with_host(&host)
                    .run();
                assert_eq!(
                    facts.completion(),
                    Completion::Complete,
                    "{tail} {offset} {aligned_write}"
                );
                assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U8(42))));
            }
        }
    }
    Ok(())
}

/// A measured entry carries the root's low bits into the callee, and missing
/// measurements and explicit unknown SSA mappings acquire none. Measured at
/// -40 the callee is entered at 0 mod 16, below its label's 8: its `and sp,
/// -16` is the pointer itself and the realigned byte is the plain one (42).
/// Unmeasured, or with the mapping explicitly unknown, the callee's frame is
/// realigned and the aliased byte unknown.
#[test]
fn entry_relative_mapping_carries_evidence_and_never_invents_it() -> Result<(), Box<dyn Error>> {
    for tail in [false, true] {
        for (offset, explicit_unknown, proved) in [
            (Some(-40), false, true),
            (None, false, false),
            (Some(-48), true, false),
        ] {
            let (function, unknown) = caller_without_stack(tail)?;
            let host = EntryRelative {
                host: Host {
                    callee: Some(aliased(0, -16, true)?),
                    call_stack: explicit_unknown.then_some(unknown),
                    root_fact: true,
                    callee_requirement: true,
                },
                offset,
            };
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&function, &context)
                .with_method(0)
                .with_host(&host)
                .run();
            assert_aliased(
                &facts,
                proved,
                &format!("{tail} {offset:?} {explicit_unknown}"),
            );
        }
    }
    Ok(())
}

/// The low bits of a stack address its root's residue states: an unaligned
/// access splits at `(sp - 6) & 3`, which is 2 whenever entry SP is eight
/// modulo sixteen, whether the frame was formed as an address or through its
/// integer. The control has no residue, and the bits stay unknown.
#[test]
fn a_root_residue_states_the_low_bits_of_a_stack_address() -> Result<(), Box<dyn Error>> {
    for round_trip in [false, true] {
        {
            let mut builder = SsaFunctionBuilder::new(0, 0);
            let stack = stack(&mut builder)?;
            builder.in_block(0, |block| {
                let frame = if round_trip {
                    let integer =
                        block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::PtrToInt {
                            dest,
                            operand: stack,
                            target: MockType::I64,
                        })?;
                    let delta =
                        block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(-16))?;
                    let frame = block.add(SsaDefSpec::tmp(MockType::I64), integer, delta)?;
                    block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::IntToPtr {
                        dest,
                        operand: frame,
                        target: MockType::Ptr,
                    })?
                } else {
                    pointer(block, stack, -16)?
                };
                let field = pointer(block, frame, 10)?;
                let field =
                    block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::PtrToInt {
                        dest,
                        operand: field,
                        target: MockType::I64,
                    })?;
                let three =
                    block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(3))?;
                let low = block.and(SsaDefSpec::tmp(MockType::I64), field, three)?;
                block.ret(Some(low))
            })?;
            let function = verified(builder)?;
            for root_fact in [true, false] {
                let host = Host {
                    callee: None,
                    call_stack: None,
                    root_fact,
                    callee_requirement: false,
                };
                let context = AnalysisContext::new(MACHINE);
                let facts = StaticExecution::new(&function, &context)
                    .with_method(0)
                    .with_host(&host)
                    .run();
                assert_eq!(facts.completion(), Completion::Complete);
                let low = facts
                    .returned()
                    .and_then(Value::constant)
                    .and_then(|constant| constant.as_u64_bits());
                assert_eq!(
                    low,
                    root_fact.then_some(2),
                    "round_trip={round_trip}: {:?}",
                    facts.returned()
                );
            }
        }
    }
    Ok(())
}
