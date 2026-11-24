//! Fuzzes the normalization passes over malformed IR.
//!
//! Passes run on whatever a frontend produced, and a frontend fed a hostile
//! binary produces IR no compiler would emit. A pass must not panic on it; the
//! checked-edit boundary is allowed to reject the edit, but the process must
//! survive.
//!
//! Each pass runs on its own copy, so one pass's rewrite cannot mask another's
//! crash.

#![no_main]

use std::collections::BTreeMap;

use analyssa::{
    analysis::{budget::WorkBudget, context::AnalysisContext},
    events::NullListener,
    ir::function::SsaConstructOptions,
    passes,
    passes::{
        AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
        CopyPropagationPass, DeadCodeEliminationPass, GlobalValueNumberingPass, JumpThreadingPass,
        LicmPass, LoopCanonicalizationPass, MemoryOptimizationPass, OpaquePredicatePass,
        ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
    },
    scheduling::{FunctionPass, PassEnv},
    testing::MACHINE,
};
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let Some(ssa) = common::from_bytes(data) else {
        return;
    };
    let method = 0u32;
    let context = AnalysisContext::new(MACHINE);

    let mut gvn = ssa.clone();
    let _ = GlobalValueNumberingPass.run(&mut gvn, &PassEnv::new(&method, &NullListener, context));

    let mut dce = ssa.clone();
    let _ = DeadCodeEliminationPass.run(&mut dce, &PassEnv::new(&method, &NullListener, context));

    let mut algebraic = ssa.clone();
    let _ = AlgebraicSimplificationPass.run(
        &mut algebraic,
        &PassEnv::new(&method, &NullListener, context),
    );

    let mut copying = ssa.clone();
    let _ = CopyPropagationPass.propagate(
        &mut copying,
        &PassEnv::new(&method, &NullListener, context),
        |_, _| {},
    );

    let mut memory = ssa.clone();
    let _ = MemoryOptimizationPass.run(&mut memory, &PassEnv::new(&method, &NullListener, context));

    let mut blockmerge = ssa.clone();
    let _ = BlockMergingPass.run(
        &mut blockmerge,
        &PassEnv::new(&method, &NullListener, context),
    );

    let mut controlflow = ssa.clone();
    let _ = ControlFlowSimplificationPass.run(
        &mut controlflow,
        &PassEnv::new(&method, &NullListener, context),
    );

    let mut licm = ssa.clone();
    let _ = LicmPass.run(&mut licm, &PassEnv::new(&method, &NullListener, context));

    let mut reassociate = ssa.clone();
    let _ = ReassociationPass.run(
        &mut reassociate,
        &PassEnv::new(&method, &NullListener, context),
    );

    let mut predicates = ssa.clone();
    let _ = OpaquePredicatePass::new().run(
        &mut predicates,
        &PassEnv::new(&method, &NullListener, context),
    );

    let mut ranges = ssa.clone();
    let _ =
        ValueRangePropagationPass.run(&mut ranges, &PassEnv::new(&method, &NullListener, context));
    let _ = passes::ranges::analyze(&ssa, &WorkBudget::DEFAULT, &BTreeMap::new());

    let mut threading = ssa.clone();
    let _ = JumpThreadingPass.run(
        &mut threading,
        &PassEnv::new(&method, &NullListener, context),
    );

    // With the proved ranges, so the signed division and modulo arms - which a
    // predicate that always answered `false` could never reach - are fuzzed too.
    let mut strength = ssa.clone();
    let _ = StrengthReductionPass.run(
        &mut strength,
        &PassEnv::new(&method, &NullListener, context),
    );

    let mut loopcanon = ssa.clone();
    let _ = LoopCanonicalizationPass.run(
        &mut loopcanon,
        &PassEnv::new(&method, &NullListener, context),
    );

    // The repairs a scheduler applies after a pass, over the same shapes.
    let mut repaired = ssa.clone();
    repaired.repair_ssa();

    let mut rebuilt = ssa.clone();
    let _ = rebuilt.construct_ssa(&SsaConstructOptions::default());

    let mut canonical = ssa.clone();
    canonical.canonicalize();

    let mut entry = ssa.clone();
    let _ = entry.ensure_entry_has_no_predecessors();
});
