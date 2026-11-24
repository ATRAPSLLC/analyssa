//! The pass orders this crate recommends, as values.
//!
//! "Run the normalization pipeline" should mean the same thing in every
//! consumer, and it did not: each host listed the passes itself, in an order it
//! arrived at separately, and the lists drifted. These are the lists, once, so
//! a host takes one and a reader can see what it contains.
//!
//! # Relationship to the scheduler
//!
//! [`PassScheduler`](crate::scheduling::PassScheduler) does not consume these,
//! and deliberately: it derives its order from the capabilities passes declare,
//! so a host that registers its own passes gets them ordered against these
//! without saying where they go. A [`PipelineSpec`] is the other thing - an
//! order stated outright - for a caller running a fixed list over one function.
//! Two answers to two different questions; collapsing them would cost the
//! scheduler the property that makes it useful.

use crate::{
    passes::{
        AlgebraicSimplificationPass, ArmMergingPass, BlockMergingPass,
        ControlFlowSimplificationPass, CopyPropagationPass, DeadCodeEliminationPass,
        DuplicateMergingPass, GlobalValueNumberingPass, JumpThreadingPass, LicmPass,
        LoopCanonicalizationPass, LoopRejoiningPass, MemoryOptimizationPass, MootTestPass,
        OpaquePredicatePass, ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
    },
    scheduling::{
        env::FunctionPass,
        pipeline::{PassGroup, PipelineSpec},
    },
    target::Target,
};

/// The passes that make a function simpler without knowing anything about how
/// it was made complicated.
///
/// What a front end's output needs before anything else reads it, and what any
/// consumer wants between its own rewrites. Four groups, in the order their
/// results feed each other: the shape of the control flow first, then the
/// values within it, then the loops that shape leaves visible, then memory,
/// whose rewrites expose new values for the next iteration.
///
/// # Returns
///
/// A spec ready for [`FunctionPipeline`](crate::scheduling::FunctionPipeline).
#[must_use]
pub fn normalization_spec<T>() -> PipelineSpec<T>
where
    T: Target + Send + Sync,
    T::MethodRef: Send + Sync,
{
    PipelineSpec::new(vec![
        PassGroup::new(
            "control-flow",
            vec![
                Box::new(ControlFlowSimplificationPass),
                Box::new(BlockMergingPass),
                Box::new(DuplicateMergingPass::new()),
                Box::new(LoopRejoiningPass::new()),
                Box::new(ArmMergingPass::new()),
                Box::new(MootTestPass::new()),
            ],
        ),
        PassGroup::new(
            "values",
            vec![
                Box::new(AlgebraicSimplificationPass::new()),
                Box::new(ReassociationPass::new()),
                Box::new(StrengthReductionPass::new()),
                Box::new(GlobalValueNumberingPass::new()),
                Box::new(CopyPropagationPass),
                Box::new(ValueRangePropagationPass::new()),
            ],
        ),
        PassGroup::new(
            "loops",
            vec![
                Box::new(LoopCanonicalizationPass::new()),
                Box::new(LicmPass::new()),
            ],
        ),
        PassGroup::new(
            "memory",
            vec![
                Box::new(MemoryOptimizationPass),
                Box::new(DeadCodeEliminationPass),
            ],
        ),
    ])
}

/// Normalization, plus the passes that undo what an obfuscator did on purpose.
///
/// The difference is jump threading and opaque-predicate removal: both are
/// about control flow that exists to be confusing rather than to compute
/// something, and both are wasted work on code no one obfuscated. They go
/// first, because everything they remove is work the rest of the list would
/// otherwise do on code that is about to disappear.
///
/// A host with its own deobfuscation passes - a decryptor, a virtual-machine
/// lifter - appends groups of its own rather than replacing this: what is here
/// is target-independent, and what a host adds is not.
///
/// # Returns
///
/// A spec ready for [`FunctionPipeline`](crate::scheduling::FunctionPipeline).
#[must_use]
pub fn deobfuscation_spec<T>() -> PipelineSpec<T>
where
    T: Target + Send + Sync,
    T::MethodRef: Send + Sync,
{
    let mut spec = normalization_spec::<T>();
    spec.groups.insert(
        0,
        PassGroup::new(
            "unobfuscate",
            vec![
                Box::new(OpaquePredicatePass::<T>::new()),
                Box::new(JumpThreadingPass::new()),
            ],
        ),
    );
    spec
}

/// Every pass either preset lists, for a caller assembling its own.
///
/// # Returns
///
/// One fresh instance of each, in no particular order.
#[must_use]
pub fn all_passes<T>() -> Vec<Box<dyn FunctionPass<T>>>
where
    T: Target + Send + Sync,
    T::MethodRef: Send + Sync,
{
    vec![
        Box::new(ControlFlowSimplificationPass),
        Box::new(BlockMergingPass),
        Box::new(DuplicateMergingPass::new()),
        Box::new(LoopRejoiningPass::new()),
        Box::new(ArmMergingPass::new()),
        Box::new(MootTestPass::new()),
        Box::new(AlgebraicSimplificationPass::new()),
        Box::new(ReassociationPass::new()),
        Box::new(StrengthReductionPass::new()),
        Box::new(GlobalValueNumberingPass::new()),
        Box::new(CopyPropagationPass),
        Box::new(ValueRangePropagationPass::new()),
        Box::new(LoopCanonicalizationPass::new()),
        Box::new(LicmPass::new()),
        Box::new(MemoryOptimizationPass),
        Box::new(DeadCodeEliminationPass),
        Box::new(OpaquePredicatePass::<T>::new()),
        Box::new(JumpThreadingPass::new()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::NullListener,
        scheduling::{FunctionPipeline, PassEnv},
        testing::{self, MACHINE, MockTarget},
    };

    #[test]
    fn the_presets_name_every_pass_once() {
        for spec in [
            normalization_spec::<MockTarget>(),
            deobfuscation_spec::<MockTarget>(),
        ] {
            let mut names: Vec<&str> = spec
                .groups
                .iter()
                .flat_map(|group| group.passes.iter().map(|pass| pass.name()))
                .collect();
            let total = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), total, "a pass appears twice: {names:?}");
        }
    }

    #[test]
    fn deobfuscation_is_normalization_plus_the_obfuscation_passes() {
        let normalization = normalization_spec::<MockTarget>();
        let deobfuscation = deobfuscation_spec::<MockTarget>();
        assert_eq!(
            deobfuscation.groups.len(),
            normalization.groups.len().saturating_add(1)
        );
        assert_eq!(
            deobfuscation.groups.first().map(|group| group.name),
            Some("unobfuscate"),
            "it runs before the work it makes unnecessary"
        );
    }

    #[test]
    fn normalization_runs_clean_over_the_fixtures() {
        for (label, mut ssa) in [
            ("diamond", testing::diamond_phi_fixture()),
            ("loop", testing::loop_counter_fixture()),
            ("try/catch", testing::try_catch_fixture()),
            ("scalar", testing::scalar_rewrite_fixture()),
        ] {
            let mut pipeline = FunctionPipeline::new(normalization_spec::<MockTarget>());
            let env =
                PassEnv::<MockTarget>::new(&0u32, &NullListener, AnalysisContext::new(MACHINE));
            let report = pipeline.run(&mut ssa, &env);
            assert!(
                report.rollbacks.is_empty(),
                "{label}: a pass produced IR the verifier rejected: {:?}",
                report.rollbacks
            );
            assert!(report.converged, "{label}: {report:?}");
            assert!(ssa.validate().is_ok(), "{label}: {ssa}");
        }
    }
}
