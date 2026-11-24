//! Structural call ancestry and normal returns remain distinct from physical frame lifetime.

use std::{collections::BTreeMap, error::Error};

use analyssa::{
    Result as BuildResult,
    analysis::{
        SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{ExecutionHost, StaticExecution, Value},
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        CallData, Callee, SsaOp,
        function::{SsaBlockBuilder, SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{CallEffects, CallTermination, CmpKind},
        value::ConstValue,
        variable::{DefSite, VariableOrigin},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Shared immutable code bodies, with no observation-dependent host behavior.
struct Host(BTreeMap<u32, SsaFunction<MockTarget>>);

impl ExecutionHost<MockTarget> for Host {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        self.0.get(method)
    }
}

/// One writable byte with a defined image lifetime.
struct Image;

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, at: u64) -> Option<ImageRegion> {
        (at == 0x9000).then_some(ImageRegion {
            start: 0x9000,
            end: 0x9001,
            mutability: Mutability::Writable,
        })
    }
    fn bytes(&self, at: u64, len: usize) -> Option<&[u8]> {
        (at == 0x9000 && len == 1).then_some(&[0])
    }
}

/// Finishes each fixture only after full structural verification.
fn verified(
    builder: SsaFunctionBuilder<MockTarget>,
) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let ssa = builder.finish()?;
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
    Ok(ssa)
}

/// Writes an observable image byte from the active body.
fn store(block: &mut SsaBlockBuilder<'_, MockTarget>, byte: u8) -> BuildResult<()> {
    let pointer = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x9000))?;
    let value = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(byte))?;
    block.store_indirect(pointer, value, MockType::U8)
}

/// Builds an argument-preserving call, with a later write only for an ordinary caller.
fn wrapper(target: u32, tail: bool) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    builder.in_block(0, |block| {
        let value = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
            let call = CallData::new(Some(dest), Callee::Direct(target), vec![input]);
            SsaOp::Call(Box::new(if tail { call.made_tail() } else { call }))
        })?;
        if !tail {
            store(block, 22)?;
        }
        block.ret(Some(value))
    })?;
    verified(builder)
}

/// Returns its actual supplied input while exposing an earlier write.
fn leaf() -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let input = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    builder.in_block(0, |block| {
        store(block, 11)?;
        block.ret(Some(input))
    })?;
    verified(builder)
}

/// Identical local sites in distinct wrappers retain separate entered domains and correct exits.
#[test]
fn structural_parents_separate_siblings_and_tail_return_chains() -> Result<(), Box<dyn Error>> {
    for tail in [false, true] {
        let host = Host(BTreeMap::from([
            (1, wrapper(3, tail)?),
            (2, wrapper(3, tail)?),
            (3, leaf()?),
        ]));
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let first = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(7))?;
            block.call_void(Callee::Direct(1), vec![first])?;
            let second = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(9))?;
            let returned = block.call(
                SsaDefSpec::tmp(MockType::I32),
                Callee::Direct(2),
                vec![second],
            )?;
            block.ret(Some(returned))
        })?;
        let root = verified(builder)?;
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let baseline = StaticExecution::new(&root, &context).with_host(&host).run();
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_transitive_call_observations(4)
            .run();
        assert!(facts.is_complete());
        assert_eq!(facts.usage(), baseline.usage());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(9))
        );
        assert_eq!(facts.invocations().len(), 4);
        for (parent_slot, child_slot, expected) in [(0, 1, 7), (2, 3, 9)] {
            let parent = facts.invocations().get(parent_slot).ok_or("parent")?;
            let child = facts.invocations().get(child_slot).ok_or("child")?;
            assert_eq!(parent.parent, None);
            assert_eq!(child.parent, Some(parent_slot));
            assert_eq!(child.caller_body, parent.body);
            assert_eq!((child.block, child.instruction), (0, 0));
            assert_eq!(
                parent.facts.returned().and_then(Value::constant),
                Some(&ConstValue::I32(expected))
            );
            assert_eq!(
                child.facts.returned().and_then(Value::constant),
                Some(&ConstValue::I32(expected))
            );
            assert_eq!(child.facts.image_writes(), vec![(0x9000, vec![11])]);
            assert_eq!(
                parent.facts.image_writes(),
                vec![(0x9000, vec![if tail { 11 } else { 22 }])]
            );
        }
        for maximum in 0..4 {
            let limited = StaticExecution::new(&root, &context)
                .with_host(&host)
                .with_transitive_call_observations(maximum)
                .run();
            assert!(limited.is_complete());
            assert!(limited.call_observations_truncated());
            assert_eq!(limited.usage(), baseline.usage());
            assert_eq!(limited.invocations().len(), maximum);
            for (index, record) in limited.invocations().iter().enumerate() {
                assert!(record.parent.is_none_or(|parent| parent < index));
            }
        }
        let root_only = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_root_call_observations(4)
            .run();
        assert_eq!(root_only.invocations().len(), 2);
        assert!(
            root_only
                .invocations()
                .iter()
                .all(|record| record.parent.is_none())
        );
    }
    Ok(())
}

/// Observation truncation in a tail chain leaves admitted return obligations intact.
#[test]
fn bounded_tail_chains_keep_existing_returns_without_skipping_execution()
-> Result<(), Box<dyn Error>> {
    let mut bodies = BTreeMap::new();
    for method in 1..17u32 {
        bodies.insert(method, wrapper(method.saturating_add(1), true)?);
    }
    bodies.insert(17, leaf()?);
    let host = Host(bodies);
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let value = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(42))?;
        let returned = block.call(
            SsaDefSpec::tmp(MockType::I32),
            Callee::Direct(1),
            vec![value],
        )?;
        block.ret(Some(returned))
    })?;
    let root = verified(builder)?;
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let baseline = StaticExecution::new(&root, &context).with_host(&host).run();
    for maximum in [0, 1, 3, 16, 17] {
        let facts = StaticExecution::new(&root, &context)
            .with_host(&host)
            .with_transitive_call_observations(maximum)
            .run();
        assert!(facts.is_complete());
        assert_eq!(facts.usage(), baseline.usage());
        assert_eq!(facts.call_observations_truncated(), maximum < 17);
        assert_eq!(facts.invocations().len(), maximum);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(42))
        );
        for (index, record) in facts.invocations().iter().enumerate() {
            assert_eq!(record.parent, index.checked_sub(1));
            assert_eq!(
                record.facts.returned().and_then(Value::constant),
                Some(&ConstValue::I32(42))
            );
            assert_eq!(record.facts.image_writes(), vec![(0x9000, vec![11])]);
        }
    }
    Ok(())
}

/// Repeat visits meet both wrapper and nested input domains in either value order.
#[test]
fn repeated_nested_visits_do_not_publish_the_first_input() -> Result<(), Box<dyn Error>> {
    for tail in [false, true] {
        for descending in [false, true] {
            let host = Host(BTreeMap::from([(1, wrapper(3, tail)?), (3, leaf()?)]));
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            let (start, one, end) = builder.in_block(0, |block| {
                let start = block.const_value(
                    SsaDefSpec::tmp(MockType::I32),
                    ConstValue::I32(i32::from(descending)),
                )?;
                let one = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(1))?;
                let end = block.const_value(
                    SsaDefSpec::tmp(MockType::I32),
                    ConstValue::I32(if descending { 0 } else { 2 }),
                )?;
                block.jump(1)?;
                Ok((start, one, end))
            })?;
            let index =
                builder.in_block(1, |block| block.empty_phi(SsaDefSpec::tmp(MockType::I32)))?;
            let next = builder.in_block(1, |block| {
                block.call_void(Callee::Direct(1), vec![index])?;
                let next = if descending {
                    block.sub(SsaDefSpec::tmp(MockType::I32), index, one)?
                } else {
                    block.add(SsaDefSpec::tmp(MockType::I32), index, one)?
                };
                block.branch_cmp(
                    next,
                    end,
                    if descending { CmpKind::Ge } else { CmpKind::Lt },
                    false,
                    1,
                    2,
                )?;
                Ok(next)
            })?;
            builder.add_phi_operand(1, index, 0, start)?;
            builder.add_phi_operand(1, index, 1, next)?;
            builder.in_block(2, |block| block.ret(None))?;
            let root = verified(builder)?;
            let context = AnalysisContext::new(MACHINE).with_image(&Image);
            let facts = StaticExecution::new(&root, &context)
                .with_host(&host)
                .with_transitive_call_observations(2)
                .run();
            assert!(facts.is_complete());
            assert_eq!(facts.invocations().len(), 2);
            assert!(!facts.call_observations_truncated());
            for record in facts.invocations() {
                assert_eq!(record.facts.returned().and_then(Value::constant), None);
            }
        }
    }
    Ok(())
}

/// A tail descendant that cannot return supplies no ancestor exit observations.
#[test]
fn nonreturning_tail_descendants_do_not_invent_exits() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        store(block, 11)?;
        let mut effects = CallEffects::pure();
        effects.termination = CallTermination::NeverReturns;
        let mut call = CallData::new(None, Callee::Direct(99), vec![]);
        call.effects = Some(effects);
        block.emit_no_defs(SsaOp::Call(Box::new(call)))?;
        block.ret(None)
    })?;
    let host = Host(BTreeMap::from([
        (1, wrapper(3, true)?),
        (3, verified(builder)?),
    ]));
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        block.call_void(Callee::Direct(1), vec![])?;
        block.ret(None)
    })?;
    let root = verified(builder)?;
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let facts = StaticExecution::new(&root, &context)
        .with_host(&host)
        .with_transitive_call_observations(2)
        .run();
    assert!(facts.is_complete());
    assert_eq!(facts.invocations().len(), 2);
    for record in facts.invocations() {
        assert_eq!(record.facts.returned(), None);
        assert!(record.facts.image_writes().is_empty());
    }
    Ok(())
}
