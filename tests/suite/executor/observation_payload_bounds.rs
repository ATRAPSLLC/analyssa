//! Borrowed output admission counts actual observations without copying their payloads.

use std::cell::Cell;

use analyssa::{
    Error, Result,
    analysis::{
        VerifyLevel,
        context::AnalysisContext,
        execution::{
            CallModel, EdgeByteRequest, EdgeObservationPlan, EdgeRequestLimits, EdgeSite,
            ExecutionHost, ModelCall, ObservationSizeRefusal, StaticExecution,
        },
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

struct Image;
impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, address: u64) -> Option<ImageRegion> {
        (0x1000..0x1003).contains(&address).then_some(ImageRegion {
            start: 0x1000,
            end: 0x1003,
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

struct Host<'a> {
    body: &'a SsaFunction<MockTarget>,
}
impl ExecutionHost<MockTarget> for Host<'_> {
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 1).then_some(self.body)
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 9).then_some(self)
    }
}
impl CallModel<MockTarget> for Host<'_> {
    fn call(&self, _: &mut ModelCall<'_, MockTarget>) {}
}

fn body() -> Result<(SsaFunction<MockTarget>, SsaVarId)> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    let pointer = builder.in_block(0, |block| {
        let pointer = block.const_value(
            SsaDefSpec::tmp(MockType::Ptr),
            ConstValue::NativeUInt(0x1000),
        )?;
        let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x61))?;
        let count = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(3))?;
        block.const_value(
            SsaDefSpec::tmp(MockType::Unknown),
            ConstValue::DecryptedString("decoded".into()),
        )?;
        block.init_blk(pointer, byte, count, None)?;
        block.jump(1)?;
        Ok(pointer)
    })?;
    builder.in_block(1, |block| {
        block.call_void(Callee::Direct(9), vec![pointer])?;
        block.ret(None)
    })?;
    Ok((builder.finish_verified(VerifyLevel::Full)?, pointer))
}

fn plan(
    body: &SsaFunction<MockTarget>,
    pointer: SsaVarId,
) -> Result<EdgeObservationPlan<'_, MockTarget>> {
    EdgeObservationPlan::new(
        &[EdgeByteRequest {
            body,
            site: EdgeSite::of(body, 0, 1)?,
            pointer,
            maximum_bytes: 3,
        }],
        MACHINE.ptr_size,
        EdgeRequestLimits {
            sites: 1,
            requests: 1,
        },
    )
}

#[test]
fn exact_mixed_count_matches_moved_output_and_preserves_buffers() -> Result<()> {
    let (body, pointer) = body()?;
    let plan = plan(&body, pointer)?;
    let host = Host { body: &body };
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut facts = StaticExecution::new(&body, &context)
        .with_host(&host)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(facts.is_complete());
    assert!(!facts.edge_output_exhausted());
    let before = format!("{facts:?}");
    let string_storage = facts
        .constants()
        .into_iter()
        .find_map(|(_, constant)| match constant {
            ConstValue::DecryptedString(text) => Some(text.as_ptr()),
            _ => None,
        })
        .ok_or_else(|| Error::new("string storage"))?;
    let call_storage = facts
        .calls()
        .into_iter()
        .flat_map(|(_, arguments)| arguments)
        .find(|argument| !argument.bytes.is_empty())
        .map(|argument| argument.bytes.as_ptr())
        .ok_or_else(|| Error::new("call storage"))?;
    // Seven string bytes plus three bytes each at edge, call and exit.
    assert_eq!(
        facts.observation_payload_bytes(15, &|| false),
        Err(ObservationSizeRefusal::OutputLimit)
    );
    assert_eq!(facts.observation_payload_bytes(16, &|| false), Ok(16));
    assert_eq!(format!("{facts:?}"), before);
    let polls = Cell::new(0usize);
    let count = || {
        polls.set(polls.get().saturating_add(1));
        false
    };
    assert_eq!(facts.observation_payload_bytes(16, &count), Ok(16));
    let total = polls.get();
    for stop in 1..=total {
        polls.set(0);
        let cancelled = || {
            polls.set(polls.get().saturating_add(1));
            polls.get() == stop
        };
        assert_eq!(
            facts.observation_payload_bytes(16, &cancelled),
            Err(ObservationSizeRefusal::Cancelled)
        );
        assert_eq!(polls.get(), stop);
        assert_eq!(format!("{facts:?}"), before);
    }
    let calls = facts.take_calls();
    let edges = facts.take_edges();
    let writes = facts
        .image_writes_within(16)
        .ok_or_else(|| Error::new("exit writes"))?;
    let constants = facts.take_constants();
    let actual = calls
        .iter()
        .flat_map(|call| &call.arguments)
        .map(|arg| arg.bytes.len())
        .sum::<usize>()
        .saturating_add(
            edges
                .iter()
                .flat_map(|edge| &edge.prefixes)
                .map(|prefix| prefix.bytes.len())
                .sum(),
        )
        .saturating_add(writes.iter().map(|write| write.bytes.len()).sum())
        .saturating_add(
            constants
                .iter()
                .map(|(_, value)| value.payload_bytes())
                .sum(),
        );
    assert_eq!(actual, 16);
    assert_eq!(
        calls
            .iter()
            .flat_map(|call| &call.arguments)
            .find(|arg| !arg.bytes.is_empty())
            .map(|arg| arg.bytes.as_ptr()),
        Some(call_storage)
    );
    assert_eq!(
        constants.iter().find_map(|(_, constant)| match constant {
            ConstValue::DecryptedString(text) => Some(text.as_ptr()),
            _ => None,
        }),
        Some(string_storage)
    );
    Ok(())
}

#[test]
fn child_payload_is_admitted_separately_and_edge_overflow_stays_sticky() -> Result<()> {
    let (body, pointer) = body()?;
    let plan = plan(&body, pointer)?;
    let host = Host { body: &body };
    let context = AnalysisContext::new(MACHINE).with_image(&Image);
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        block.call_void(Callee::Direct(1), vec![])?;
        block.ret(None)
    })?;
    let caller = builder.finish_verified(VerifyLevel::Full)?;
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .with_transitive_call_observations(1)
        .with_edge_observations(&plan, 3)
        .run();
    assert!(facts.is_complete());
    assert_eq!(facts.observation_payload_bytes(3, &|| false), Ok(3));
    let child = facts
        .invocations()
        .first()
        .ok_or_else(|| Error::new("child scope"))?;
    assert_eq!(
        child.facts.observation_payload_bytes(15, &|| false),
        Err(ObservationSizeRefusal::OutputLimit)
    );
    assert_eq!(child.facts.observation_payload_bytes(16, &|| false), Ok(16));
    assert_eq!(facts.observation_payload_bytes(3, &|| false), Ok(3));
    let facts = StaticExecution::new(&body, &context)
        .with_host(&host)
        .with_edge_observations(&plan, 2)
        .run();
    assert!(facts.is_complete());
    assert!(facts.edge_output_exhausted());
    assert_eq!(
        facts.observation_payload_bytes(usize::MAX, &|| false),
        Err(ObservationSizeRefusal::OutputLimit)
    );
    assert_eq!(
        facts.observation_payload_bytes(usize::MAX, &|| true),
        Err(ObservationSizeRefusal::Cancelled)
    );
    Ok(())
}
