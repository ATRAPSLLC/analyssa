//! Code bounds stop their own resolution boundary without poisoning a later run.

use std::{cell::Cell, result::Result as StdResult, sync::Arc};

use analyssa::{
    Result,
    analysis::{
        VerifyLevel,
        budget::{Budget, Exhaustion},
        context::AnalysisContext,
        execution::{
            Address, BranchContinuation, BranchSite, CallModel, CallSite, CodeMemory, Completion,
            ExecutionHost, InstructionSources, MemoryRefusal, ModelCall, StaticExecution, Value,
        },
        image::{ImageData, ImageRegion},
    },
    ir::{
        Callee, ConstValue, SsaOp,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

type Function = SsaFunction<MockTarget>;
const LIMIT: Exhaustion = Exhaustion {
    budget: Budget::CodeRequests,
    limit: 0,
    demand: 1,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    Direct,
    Indirect,
    CallBody,
    Branch,
    BranchBody,
}

struct EmptyImage;
impl ImageData for EmptyImage {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn region(&self, _: u64) -> Option<ImageRegion> {
        None
    }
    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }
}

fn source(boundary: Boundary) -> Result<Function> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| match boundary {
        Boundary::Direct | Boundary::CallBody => {
            let result = block.call(SsaDefSpec::tmp(MockType::I64), Callee::Direct(7), vec![])?;
            block.ret(Some(result))
        }
        Boundary::Indirect => {
            let target =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(200))?;
            let result = block.call(
                SsaDefSpec::tmp(MockType::I64),
                Callee::Indirect {
                    fptr: target,
                    signature: 0,
                },
                vec![],
            )?;
            block.ret(Some(result))
        }
        Boundary::Branch | Boundary::BranchBody => {
            let target =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(200))?;
            block.emit_no_defs(SsaOp::IndirectBranch {
                target,
                resolved_targets: vec![],
                state: Box::default(),
            })?;
            Ok(())
        }
    })?;
    builder.finish_verified(VerifyLevel::Full)
}

fn next() -> Result<Arc<Function>> {
    let mut builder = SsaFunctionBuilder::new(0, 0);
    builder.in_block(0, |block| {
        let value = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(42))?;
        block.ret(Some(value))
    })?;
    Ok(Arc::new(builder.finish_verified(VerifyLevel::Full)?))
}

struct Host {
    boundary: Boundary,
    armed: Cell<bool>,
    model_runs: Cell<usize>,
    retirements: Cell<usize>,
    next: Arc<Function>,
    image: EmptyImage,
}

impl Host {
    fn signal(&self, at: Boundary, memory: CodeMemory<'_>) {
        if self.armed.get() && self.boundary == at {
            let memory = memory.with_initial_instructions(&self.image);
            assert!(!memory.refuse_code_limit(Exhaustion {
                limit: 1,
                demand: 1,
                ..LIMIT
            }));
            assert!(memory.refuse_code_limit(LIMIT));
            assert!(memory.refuse_code_limit(Exhaustion {
                budget: Budget::CodeVersionBytes,
                ..LIMIT
            }));
        }
    }
}

impl CallModel<MockTarget> for Host {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        self.model_runs.set(self.model_runs.get().saturating_add(1));
        call.return_value(Value::Const(ConstValue::I64(42)));
    }
}

impl ExecutionHost<MockTarget> for Host {
    fn method_for_call(
        &self,
        _: CallSite<'_, MockTarget>,
        method: &u32,
        memory: CodeMemory<'_>,
    ) -> Option<u32> {
        self.signal(Boundary::Direct, memory);
        Some(*method)
    }

    fn method_at_call(
        &self,
        _: CallSite<'_, MockTarget>,
        _: Address,
        memory: CodeMemory<'_>,
    ) -> Option<u32> {
        self.signal(Boundary::Indirect, memory);
        Some(7)
    }

    fn body_snapshot(
        &self,
        method: &u32,
        memory: CodeMemory<'_>,
    ) -> StdResult<Option<Arc<Function>>, MemoryRefusal> {
        self.signal(
            if *method == 7 {
                Boundary::CallBody
            } else {
                Boundary::BranchBody
            },
            memory,
        );
        Ok((*method == 200).then(|| Arc::clone(&self.next)))
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 7).then_some(self as &dyn CallModel<MockTarget>)
    }

    fn branch_continuation(
        &self,
        _: BranchSite<'_, MockTarget>,
        _: &Value<MockTarget>,
        memory: CodeMemory<'_>,
    ) -> StdResult<Option<BranchContinuation<MockTarget>>, MemoryRefusal> {
        self.signal(Boundary::Branch, memory);
        Ok(Some(BranchContinuation {
            method: 200,
            bindings: vec![],
            stack: None,
            frame: Vec::new(),
        }))
    }

    fn retire_code_source(
        &self,
        _: &Function,
        _: Option<&u32>,
        _: &mut InstructionSources,
    ) -> bool {
        self.retirements
            .set(self.retirements.get().saturating_add(1));
        true
    }
}

#[test]
fn every_resolution_boundary_honors_its_first_limit_before_fallback() -> Result<()> {
    for boundary in [
        Boundary::Direct,
        Boundary::Indirect,
        Boundary::CallBody,
        Boundary::Branch,
        Boundary::BranchBody,
    ] {
        let source = source(boundary)?;
        let host = Host {
            boundary,
            armed: Cell::new(true),
            model_runs: Cell::new(0),
            retirements: Cell::new(0),
            next: next()?,
            image: EmptyImage,
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&source, &context)
            .with_host(&host)
            .run();
        assert_eq!(
            facts.completion(),
            Completion::Exhausted(LIMIT),
            "{boundary:?}"
        );
        assert!(facts.returned().is_none());
        assert_eq!(host.model_runs.get(), 0, "{boundary:?}");
        assert_eq!(host.retirements.get(), 0, "{boundary:?}");
        host.armed.set(false);
        let facts = StaticExecution::new(&source, &context)
            .with_host(&host)
            .run();
        assert_eq!(facts.completion(), Completion::Complete, "{boundary:?}");
        assert_eq!(
            facts
                .returned()
                .and_then(Value::constant)
                .and_then(ConstValue::as_i64),
            Some(42),
            "{boundary:?}"
        );
    }
    Ok(())
}
