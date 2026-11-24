//! Transitive actual accesses bound pre-call bytes without rerunning a prefix.

use analyssa::{
    Error, Result,
    analysis::{
        VerifyLevel,
        context::AnalysisContext,
        execution::{
            ArgumentBindings, CallModel, CallSite, ExecutionHost, LibraryModel, ModelCall,
            ParameterBindings, StaticExecution, Value,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        CallData, Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::{Endianness, Target},
    testing::{MACHINE, MockTarget, MockType},
};

const BEFORE: [u8; 8] = [1, 2, 3, 4, 9, 10, 11, 12];
const WORD: u32 = 0x04030201;

struct Image;
impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, at: u64) -> Option<ImageRegion> {
        (0x2000..0x2008).contains(&at).then_some(ImageRegion {
            start: 0x2000,
            end: 0x2008,
            mutability: Mutability::Writable,
        })
    }
    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }
    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Body,
    Tail,
    Read,
    Copy,
    String,
    Unknown,
    Alternatives,
    Partial,
    NoWrites,
}

struct Host {
    wrapper: SsaFunction<MockTarget>,
    leaf: SsaFunction<MockTarget>,
    kind: Kind,
}
impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        match *method {
            1 => Some(&self.wrapper),
            2 if matches!(self.kind, Kind::Body | Kind::Tail | Kind::Alternatives) => {
                Some(&self.leaf)
            }
            _ => None,
        }
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        match (*method, self.kind) {
            (2, Kind::Copy) => Some(&LibraryModel::Copy),
            (2, Kind::String) => Some(&LibraryModel::Length),
            (2, Kind::NoWrites) => Some(&LibraryModel::NoWrites),
            (2, Kind::Read | Kind::Unknown | Kind::Partial) | (3, _) => Some(self),
            _ => None,
        }
    }
}
impl CallModel<MockTarget> for Host {
    fn tracks_all_memory_accesses(&self) -> bool {
        !matches!(self.kind, Kind::Partial)
    }

    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        if call.argument(0).is_none() {
            let block = call.allocate_block(Some(4), false, None);
            call.return_value(Value::Addr(block));
            return;
        }
        let Some(pointer) = call.address(0) else {
            call.clobber();
            return;
        };
        let Some(bytes) = call.read(pointer, 4) else {
            call.clobber();
            return;
        };
        let Ok(bytes) = <[u8; 4]>::try_from(bytes.as_slice()) else {
            call.clobber();
            return;
        };
        if matches!(self.kind, Kind::Unknown) {
            call.clobber();
        }
        call.return_value(Value::Const(ConstValue::U32(u32::from_le_bytes(bytes))));
    }
}

fn argument(
    builder: &mut SsaFunctionBuilder<MockTarget>,
    index: u16,
    ty: MockType,
) -> analyssa::ir::variable::SsaVarId {
    builder
        .function_mut()
        .create_variable(VariableOrigin::Argument(index), 0, DefSite::entry(), ty)
}

fn wrapper(kind: Kind) -> Result<SsaFunction<MockTarget>> {
    let mut b = SsaFunctionBuilder::new(1, 0);
    let pointer = argument(&mut b, 0, MockType::Ptr);
    b.in_block(0, |b| {
        if matches!(kind, Kind::NoWrites) {
            let word = b.load_indirect(SsaDefSpec::tmp(MockType::U32), pointer, MockType::U32)?;
            b.call_void(Callee::Direct(2), vec![pointer])?;
            return b.ret(Some(word));
        }
        if matches!(kind, Kind::Copy) {
            let dest = b.call(SsaDefSpec::tmp(MockType::Ptr), Callee::Direct(3), vec![])?;
            let count = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(4))?;
            b.call_void(Callee::Direct(2), vec![dest, pointer, count])?;
            let result = b.load_indirect(SsaDefSpec::tmp(MockType::U32), dest, MockType::U32)?;
            return b.ret(Some(result));
        }
        let ty = if matches!(kind, Kind::String) {
            MockType::U64
        } else {
            MockType::U32
        };
        let result = b.emit_def(SsaDefSpec::tmp(ty), |dest| {
            let call = CallData::new(Some(dest), Callee::Direct(2), vec![pointer]);
            if matches!(kind, Kind::Tail) {
                call.made_tail().into()
            } else {
                call.into()
            }
        })?;
        b.ret(Some(result))
    })?;
    b.finish_verified(VerifyLevel::Full)
}

fn leaf(alternatives: bool) -> Result<SsaFunction<MockTarget>> {
    let mut b = SsaFunctionBuilder::new(2, 0);
    let pointer = argument(&mut b, 0, MockType::Ptr);
    if alternatives {
        let condition = argument(&mut b, 1, MockType::Bool);
        b.in_block(0, |b| b.branch(condition, 1, 2))?;
        for (block, width) in [(1, MockType::U32), (2, MockType::U64)] {
            b.in_block(block, |b| {
                b.load_indirect(SsaDefSpec::tmp(width), pointer, width)?;
                let word =
                    b.load_indirect(SsaDefSpec::tmp(MockType::U32), pointer, MockType::U32)?;
                b.ret(Some(word))
            })?;
        }
    } else {
        b.in_block(0, |b| {
            let word = b.load_indirect(SsaDefSpec::tmp(MockType::U32), pointer, MockType::U32)?;
            b.ret(Some(word))
        })?;
    }
    b.finish_verified(VerifyLevel::Full)
}

fn caller(string: bool) -> Result<SsaFunction<MockTarget>> {
    let mut b = SsaFunctionBuilder::new(0, 0);
    b.in_block(0, |b| {
        let pointer = b.const_value(
            SsaDefSpec::tmp(MockType::Ptr),
            ConstValue::NativeUInt(0x2000),
        )?;
        let bytes = if string {
            [b'a', b'b', b'c', 0, 9, 10, 11, 12]
        } else {
            BEFORE
        };
        let value = b.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(u64::from_le_bytes(bytes)),
        )?;
        b.store_indirect(pointer, value, MockType::U64)?;
        let ty = if string { MockType::U64 } else { MockType::U32 };
        let result = b.call(SsaDefSpec::tmp(ty), Callee::Direct(1), vec![pointer])?;
        b.ret(Some(result))
    })?;
    b.finish_verified(VerifyLevel::Full)
}

fn check(kind: Kind, bytes: usize, value: u64) -> Result<()> {
    let root = caller(matches!(kind, Kind::String))?;
    let host = Host {
        wrapper: wrapper(kind)?,
        leaf: leaf(matches!(kind, Kind::Alternatives))?,
        kind,
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine).with_image(&Image);
    // Inner-body observation records are deliberately absent; actual accesses
    // must still cover the outer call, including nested tail transfers.
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_root_call_observations(1)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64),
        Some(value)
    );
    let observations = facts.calls();
    let observed = observations
        .first()
        .and_then(|(_, args)| args.first())
        .ok_or_else(|| Error::new("outer pre-call observation"))?;
    let expected = if matches!(kind, Kind::String) {
        b"abc\0".as_slice()
    } else {
        BEFORE.get(..bytes).ok_or_else(|| Error::new("extent"))?
    };
    assert_eq!(observed.bytes, expected);
    Ok(())
}

#[test]
fn nested_body_accesses_bound_the_outer_snapshot() -> Result<()> {
    check(Kind::Body, 4, u64::from(WORD))
}
#[test]
fn nested_tail_accesses_preserve_outer_obligations() -> Result<()> {
    check(Kind::Tail, 4, u64::from(WORD))
}
#[test]
fn nested_custom_model_reads_bound_the_outer_snapshot() -> Result<()> {
    check(Kind::Read, 4, u64::from(WORD))
}
#[test]
fn library_copy_source_reads_and_actual_destination_value_are_covered() -> Result<()> {
    check(Kind::Copy, 4, u64::from(WORD))
}
#[test]
fn known_string_model_uses_include_the_terminator_only() -> Result<()> {
    check(Kind::String, 4, 3)
}
#[test]
fn nested_unknown_effects_keep_the_uncapped_pre_call_snapshot() -> Result<()> {
    check(Kind::Unknown, 8, u64::from(WORD))
}
#[test]
fn nested_alternatives_union_all_accesses() -> Result<()> {
    check(Kind::Alternatives, 8, u64::from(WORD))
}

struct StackHost {
    callee: SsaFunction<MockTarget>,
}
impl ExecutionHost<MockTarget> for StackHost {
    fn body(&self, _: &u32) -> Option<&SsaFunction<MockTarget>> {
        Some(&self.callee)
    }
    fn call_stack_from_entry(&self, _: CallSite<'_, MockTarget>) -> Option<i64> {
        Some(0)
    }
    fn parameters(
        &self,
        _: CallSite<'_, MockTarget>,
        _: &u32,
        body: &SsaFunction<MockTarget>,
        bindings: &mut ParameterBindings<'_, MockTarget>,
    ) {
        let Some(layout) = MockTarget::type_layout(&MockType::U32, MACHINE.ptr_size) else {
            return;
        };
        if !bindings.stage_stack(1, 0, layout) {
            return;
        }
        for variable in body.variables() {
            if variable.origin() == VariableOrigin::Argument(0) {
                let _ = bindings.bind_stack(variable.id(), 0, &MockType::U32);
            }
        }
    }
}

#[test]
fn physical_entry_reads_and_promoted_writes_are_inside_the_call_use_scope() -> Result<()> {
    let mut b = SsaFunctionBuilder::new(0, 0);
    let stack = b.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    b.function_mut()
        .variable_mut(stack)
        .ok_or_else(|| Error::new("stack"))?
        .set_role(VariableRole::EntryStackPointer);
    b.in_block(0, |b| {
        let before = b.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(u64::from_le_bytes(BEFORE)),
        )?;
        b.store_indirect(stack, before, MockType::U64)?;
        let word = b.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(WORD))?;
        let result = b.call(
            SsaDefSpec::tmp(MockType::U32),
            Callee::Direct(1),
            vec![stack, word],
        )?;
        b.ret(Some(result))
    })?;
    let root = b.finish_verified(VerifyLevel::Full)?;
    let mut b = SsaFunctionBuilder::new(1, 0);
    let input = argument(&mut b, 0, MockType::U32);
    b.in_block(0, |b| b.ret(Some(input)))?;
    let host = StackHost {
        callee: b.finish_verified(VerifyLevel::Full)?,
    };
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64),
        Some(u64::from(WORD))
    );
    let calls = facts.calls();
    let observed = calls
        .first()
        .and_then(|(_, args)| args.first())
        .ok_or_else(|| Error::new("physical pre-call observation"))?;
    assert_eq!(observed.bytes, BEFORE[..4]);
    Ok(())
}

#[test]
fn partial_models_do_not_certify_complete_caller_extents() -> Result<()> {
    check(Kind::Partial, 8, u64::from(WORD))
}

#[test]
fn no_writes_is_not_a_complete_read_contract() -> Result<()> {
    check(Kind::NoWrites, 8, u64::from(WORD))
}

struct OutgoingStoreHost {
    wrapper: SsaFunction<MockTarget>,
}
impl ExecutionHost<MockTarget> for OutgoingStoreHost {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 1).then_some(&self.wrapper)
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 2).then_some(self)
    }
    fn call_stack_from_entry(&self, _: CallSite<'_, MockTarget>) -> Option<i64> {
        Some(0)
    }
    fn arguments(
        &self,
        _: CallSite<'_, MockTarget>,
        method: Option<&u32>,
        _: Option<&SsaFunction<MockTarget>>,
        _: Option<&dyn CallModel<MockTarget>>,
        bindings: &mut ArgumentBindings<'_, MockTarget>,
    ) {
        if method == Some(&2) {
            let layout = MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size);
            let _ = bindings.stage_ordinary_stack(1, None, Some(0), layout);
        }
    }
}
impl CallModel<MockTarget> for OutgoingStoreHost {
    fn tracks_all_memory_accesses(&self) -> bool {
        true
    }
    fn call(&self, _: &mut ModelCall<'_, MockTarget>) {}
}

#[test]
fn source_argument_stores_cover_outer_bytes_before_the_nested_model() -> Result<()> {
    let after = 0x8877665544332211;
    let mut b = SsaFunctionBuilder::new(1, 0);
    let pointer = argument(&mut b, 0, MockType::Ptr);
    b.in_block(0, |b| {
        let value = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(after))?;
        b.call_void(Callee::Direct(2), vec![pointer, value])?;
        b.ret(None)
    })?;
    let host = OutgoingStoreHost {
        wrapper: b.finish_verified(VerifyLevel::Full)?,
    };
    let mut b = SsaFunctionBuilder::new(0, 0);
    let stack = b.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    b.function_mut()
        .variable_mut(stack)
        .ok_or_else(|| Error::new("stack"))?
        .set_role(VariableRole::EntryStackPointer);
    b.in_block(0, |b| {
        let before = b.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(u64::from_le_bytes(BEFORE)),
        )?;
        b.store_indirect(stack, before, MockType::U64)?;
        b.call_void(Callee::Direct(1), vec![stack])?;
        let result = b.load_indirect(SsaDefSpec::tmp(MockType::U64), stack, MockType::U64)?;
        b.ret(Some(result))
    })?;
    let root = b.finish_verified(VerifyLevel::Full)?;
    let mut machine = MACHINE;
    machine.byte_order = Some(Endianness::Little);
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&root, &context).with_host(&host).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64),
        Some(after)
    );
    let calls = facts.calls();
    let observed = calls
        .first()
        .and_then(|(_, args)| args.first())
        .ok_or_else(|| Error::new("pre-store outer snapshot"))?;
    assert_eq!(observed.bytes, BEFORE);
    Ok(())
}
