//! Cost of the SSA maintenance operations the optimization pipeline pays per
//! pass.
//!
//! Motivation: visus-core's `guard_pass` deep-clones the whole `SsaFunction`
//! before every pass so a verifier rejection can roll back, and every edit
//! session closes with a boundary repair - `recompute_uses` or `repair_ssa` -
//! and a session that changed the graph has reconstructed the values it
//! disturbed. A renaming `construct_ssa` is the whole-function bound on that
//! reconstruction. Which of these dominates decides whether the pipeline's
//! effort belongs on eliminating snapshots or on avoiding repairs.
//!
//! These benchmarks measure it. Run with:
//!
//! ```text
//! cargo bench -p analyssa --bench ssarepair
//! ```
//!
//! # Groups
//!
//! | Group | Question it answers |
//! |-------|---------------------|
//! | `ssa_maintenance` | On fixed small functions, how do snapshot and repair compare? |
//! | `ssa_maintenance_scaling` | How does each term grow with function size? |
//! | `analysis_construction` | What do the per-pass analysis rebuilds cost? |
//! | `pass_transaction` | What does group-vs-per-pass snapshotting actually save? |
//! | `exception_model` | What do a handler's exceptional phi operands cost? |
//! | `instruction_identity` | What does stamping every instruction cost at scale? |
//! | `facts_overhead` | Is a host that carries no facts charged for the ones that do? |
//!
//! `ssa_maintenance_scaling` is the group that settles the roadmap question.
//! Snapshot cost is linear in total IR size, while construction is driven by
//! dominance frontiers and phi placement - so a ratio measured on a
//! ten-instruction fixture does not extrapolate to the pathological functions
//! (a 258k-instruction, 20k-block data-misread-as-code) that the roadmap's
//! copy-on-write item is actually about. Two fixture families are swept
//! independently: `straight_line` grows instructions at a fixed block count,
//! `diamond_chain` grows blocks and phis together.

#![allow(missing_docs)]

use std::hint::black_box;

use analyssa::{
    analysis::{
        AnalysisContext, DefUseIndex, LoopAnalyzer, SsaCfg,
        budget::{Budget, WorkBudget},
        memory::MemorySsa,
    },
    events::NullListener,
    graph::{RootedGraph, algorithms::compute_dominators},
    ir::function::{SsaConstructOptions, SsaFunction},
    passes::{
        AlgebraicSimplificationPass, CopyPropagationPass, DeadCodeEliminationPass,
        GlobalValueNumberingPass, LicmPass,
    },
    scheduling::{FunctionPass, PassEnv, PassTransaction},
    target::Target,
    testing::{self, FactfulTarget, MACHINE, MockTarget},
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

/// The fixtures span the shapes that make repair costs differ: straight-line
/// code, a phi merge, and a loop with a back edge.
fn fixtures() -> Vec<(&'static str, SsaFunction<MockTarget>)> {
    vec![
        ("scalar", testing::scalar_rewrite_fixture()),
        ("diamond_phi", testing::diamond_phi_fixture()),
        ("loop_counter", testing::loop_counter_fixture()),
    ]
}

/// Copy-propagation iteration budget used by the benchmarked pass group.
const COPY_PROP_ITERATIONS: usize = 4;

/// Sizes swept by the scaling groups. The top end is kept modest so the suite
/// stays runnable in CI; the curve's *shape* is what matters, and that is
/// already unambiguous across two decades of size.
const SCALES: [usize; 4] = [16, 64, 256, 1024];

/// Builds both fixture families at one scale, labelled for the report.
fn scaled(scale: usize) -> [(String, SsaFunction<MockTarget>); 2] {
    [
        (
            format!("straight_line/{scale}"),
            testing::straight_line_fixture(scale),
        ),
        (
            format!("diamond_chain/{scale}"),
            testing::diamond_chain_fixture(scale),
        ),
    ]
}

/// Total instruction count, the natural per-unit denominator for the timings.
fn instruction_count<T: Target>(ssa: &SsaFunction<T>) -> usize {
    ssa.blocks()
        .iter()
        .map(|block| block.instructions().len())
        .sum()
}

/// Reports each fixture's size, so the timings below can be read per-unit
/// rather than as bare numbers.
fn report_sizes<T: Target>(name: &str, ssa: &SsaFunction<T>) {
    println!(
        "fixture {name}: {} blocks, {} variables, {} instructions",
        ssa.block_count(),
        ssa.variable_count(),
        instruction_count(ssa)
    );
}

fn bench_maintenance(c: &mut Criterion) {
    for (name, ssa) in fixtures() {
        report_sizes(name, &ssa);
    }

    let mut group = c.benchmark_group("ssa_maintenance");

    for (name, ssa) in fixtures() {
        // The snapshot `guard_pass` takes before every pass.
        group.bench_with_input(BenchmarkId::new("clone", name), &ssa, |b, ssa| {
            b.iter(|| black_box(ssa.clone()));
        });

        // Snapshotting into a reused buffer - what the pass-group rollback
        // point does. Isolates the allocator saving from the memcpy.
        group.bench_with_input(BenchmarkId::new("clone_from", name), &ssa, |b, ssa| {
            let mut buffer = ssa.clone();
            b.iter(|| {
                buffer.clone_from(black_box(ssa));
                black_box(&buffer);
            });
        });

        // The cheapest boundary repair (`SsaEditScope::UsesOnly`).
        group.bench_with_input(BenchmarkId::new("recompute_uses", name), &ssa, |b, ssa| {
            b.iter_batched(
                || ssa.clone(),
                |mut working| {
                    working.recompute_uses();
                    black_box(working)
                },
                BatchSize::SmallInput,
            );
        });

        // The middle tier (`SsaEditScope::InstructionsOnly`).
        group.bench_with_input(BenchmarkId::new("repair_ssa", name), &ssa, |b, ssa| {
            b.iter_batched(
                || ssa.clone(),
                |mut working| {
                    working.repair_ssa();
                    black_box(working)
                },
                BatchSize::SmallInput,
            );
        });

        // The whole-function Cytron reconstruction, the bound on what a
        // CFG-modifying edit's scoped repair can cost.
        group.bench_with_input(BenchmarkId::new("construct_ssa", name), &ssa, |b, ssa| {
            b.iter_batched(
                || ssa.clone(),
                |mut working| {
                    let _ = working.construct_ssa(&SsaConstructOptions::renaming_only());
                    black_box(working)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Sweeps the same operations across function sizes so the snapshot and repair
/// terms can be compared by *growth rate* rather than at one arbitrary point.
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssa_maintenance_scaling");
    // Reconstruction on the largest fixtures is slow enough that the default
    // sample size turns a bench run into minutes of wall clock for no extra
    // signal.
    group.sample_size(20);

    for scale in SCALES {
        for (id, ssa) in scaled(scale) {
            report_sizes(&id, &ssa);

            // Snapshot cost: expected linear in total IR size.
            group.bench_with_input(BenchmarkId::new("clone", &id), &ssa, |b, ssa| {
                b.iter(|| black_box(ssa.clone()));
            });
            group.bench_with_input(BenchmarkId::new("clone_from", &id), &ssa, |b, ssa| {
                let mut buffer = ssa.clone();
                b.iter(|| {
                    buffer.clone_from(black_box(ssa));
                    black_box(&buffer);
                });
            });

            // Repair cost: expected to grow faster than the snapshot on the
            // block/phi-heavy family and comparably on the flat one.
            group.bench_with_input(BenchmarkId::new("recompute_uses", &id), &ssa, |b, ssa| {
                b.iter_batched(
                    || ssa.clone(),
                    |mut working| {
                        working.recompute_uses();
                        black_box(working)
                    },
                    BatchSize::SmallInput,
                );
            });
            group.bench_with_input(BenchmarkId::new("construct_ssa", &id), &ssa, |b, ssa| {
                b.iter_batched(
                    || ssa.clone(),
                    |mut working| {
                        let _ = working.construct_ssa(&SsaConstructOptions::renaming_only());
                        black_box(working)
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

/// Cost of the analyses passes rebuild from scratch on every run.
///
/// Within one fixpoint iteration each scheduled pass reconstructs what the
/// previous pass discarded. `src/passes/` has 149 `count_uses` /
/// `recompute_uses` call sites, 12 `SsaCfg::from_ssa`, and 4
/// `DefUseIndex::build*` - so the per-variable use lists, not the heavier
/// indexes, are what the pipeline rebuilds most.
///
/// These numbers are the input to the roadmap's analysis-caching item, but they
/// are only half of it: caching is worth its invalidation complexity only if
/// construction is a meaningful *share* of a pass's runtime, which is what
/// `pass_cost` measures alongside these.
fn bench_analysis_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_construction");
    group.sample_size(30);

    for scale in SCALES {
        for (id, ssa) in scaled(scale) {
            group.bench_with_input(BenchmarkId::new("SsaCfg::from_ssa", &id), &ssa, |b, ssa| {
                b.iter(|| black_box(SsaCfg::from_ssa(black_box(ssa))));
            });

            group.bench_with_input(BenchmarkId::new("DefUseIndex", &id), &ssa, |b, ssa| {
                b.iter(|| black_box(DefUseIndex::build_with_ops(black_box(ssa))));
            });

            group.bench_with_input(BenchmarkId::new("MemorySsa", &id), &ssa, |b, ssa| {
                b.iter(|| {
                    black_box(MemorySsa::build(
                        black_box(ssa),
                        &AnalysisContext::new(MACHINE),
                    ))
                });
            });

            // The pipeline's most-rebuilt analysis by call-site count.
            group.bench_with_input(BenchmarkId::new("count_uses", &id), &ssa, |b, ssa| {
                b.iter(|| black_box(black_box(ssa).count_uses()));
            });

            // CFG + dominator tree, as GVN and range propagation build it.
            group.bench_with_input(BenchmarkId::new("cfg+dominators", &id), &ssa, |b, ssa| {
                b.iter(|| {
                    let cfg = SsaCfg::from_ssa(black_box(ssa));
                    let entry = cfg.entry();
                    black_box(compute_dominators(&cfg, entry))
                });
            });

            // Loop forest, as LICM builds it.
            group.bench_with_input(BenchmarkId::new("loop_forest", &id), &ssa, |b, ssa| {
                b.iter(|| black_box(LoopAnalyzer::new(black_box(ssa)).analyze()));
            });
        }
    }

    group.finish();
}

/// Cost of each scheduled pass in isolation, so the `analysis_construction`
/// numbers can be read as a *share* rather than in the abstract.
///
/// This is the measurement that decides whether an analysis cache is worth
/// building: if constructing what a pass consumes is a small slice of that
/// pass's runtime, caching buys little no matter how many call sites there are.
///
/// Every pass runs against a fresh clone, so each measurement includes the
/// analyses that pass builds for itself.
fn bench_pass_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("pass_cost");
    group.sample_size(30);

    for scale in SCALES {
        for (id, ssa) in scaled(scale) {
            macro_rules! bench_pass {
                ($name:literal, $run:expr) => {
                    group.bench_with_input(BenchmarkId::new($name, &id), &ssa, |b, ssa| {
                        b.iter_batched(
                            || ssa.clone(),
                            |mut working| {
                                let method = 0u32;
                                let events = NullListener;
                                let run: fn(&mut SsaFunction<MockTarget>, &u32, &NullListener) =
                                    $run;
                                run(&mut working, &method, &events);
                                black_box(working)
                            },
                            BatchSize::SmallInput,
                        );
                    });
                };
            }

            bench_pass!("algebraic", |s, m, e| {
                let _ = AlgebraicSimplificationPass
                    .run(s, &PassEnv::new(m, e, AnalysisContext::new(MACHINE)));
            });
            bench_pass!("copying", |s, m, e| {
                let _ = CopyPropagationPass.propagate(
                    s,
                    &PassEnv::new(m, e, AnalysisContext::new(MACHINE)).with_budget(
                        WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, COPY_PROP_ITERATIONS),
                    ),
                    |_, _| {},
                );
            });
            bench_pass!("deadcode", |s, m, e| {
                let _ = DeadCodeEliminationPass
                    .run_iteration(s, &PassEnv::new(m, e, AnalysisContext::new(MACHINE)));
            });
            bench_pass!("gvn", |s, m, e| {
                let _ = GlobalValueNumberingPass
                    .run(s, &PassEnv::new(m, e, AnalysisContext::new(MACHINE)));
            });
            bench_pass!("licm", |s, m, e| {
                let _ =
                    LicmPass.run(s, &PassEnv::new(m, e, AnalysisContext::new(MACHINE))) == Ok(true);
            });
        }
    }

    group.finish();
}

/// Runs a representative group of cheap rewrite passes, reporting whether any
/// of them changed the IR - the shape `PassTransaction::run_group` expects.
fn rewrite_group(ssa: &mut SsaFunction<MockTarget>) -> bool {
    let method = 0u32;
    let events = NullListener;
    let mut changed = AlgebraicSimplificationPass.run(
        ssa,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    changed |= CopyPropagationPass.propagate(
        ssa,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, COPY_PROP_ITERATIONS)),
        |_, _| {},
    );
    changed |= DeadCodeEliminationPass.run_iteration(
        ssa,
        &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)),
    ) > 0;
    changed
}

/// What transactional pass grouping actually saves.
///
/// `PassTransaction::run_group` exists to amortize one snapshot across a whole
/// pass group instead of paying one per pass, and to skip verification when a
/// group reports no rewrite - the case that describes every terminating
/// fixpoint iteration. Both effects are measured against the per-pass baseline
/// the group replaces.
fn bench_pass_transaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("pass_transaction");
    group.sample_size(30);

    for scale in SCALES {
        for (id, ssa) in scaled(scale) {
            // Baseline: snapshot before every pass, as `guard_pass` does.
            group.bench_with_input(
                BenchmarkId::new("per_pass_snapshot", &id),
                &ssa,
                |b, ssa| {
                    b.iter_batched(
                        || ssa.clone(),
                        |mut working| {
                            let method = 0u32;
                            let events = NullListener;
                            let snapshot = working.clone();
                            let _ = AlgebraicSimplificationPass.run(
                                &mut working,
                                &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)),
                            );
                            black_box(&snapshot);
                            let snapshot = working.clone();
                            let _ = CopyPropagationPass.propagate(
                                &mut working,
                                &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE))
                                    .with_budget(
                                        WorkBudget::DEFAULT
                                            .with_limit(Budget::CopyRounds, COPY_PROP_ITERATIONS),
                                    ),
                                |_, _| {},
                            );
                            black_box(&snapshot);
                            let snapshot = working.clone();
                            let _ = DeadCodeEliminationPass.run_iteration(
                                &mut working,
                                &PassEnv::new(&method, &events, AnalysisContext::new(MACHINE)),
                            );
                            black_box(&snapshot);
                            black_box(working)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            // One snapshot for the whole group, verified once.
            group.bench_with_input(BenchmarkId::new("group_snapshot", &id), &ssa, |b, ssa| {
                let mut transaction = PassTransaction::<MockTarget>::new();
                b.iter_batched_ref(
                    || ssa.clone(),
                    |working| {
                        black_box(transaction.run_group(working, rewrite_group));
                    },
                    BatchSize::SmallInput,
                );
            });

            // The terminating iteration: nothing changes, so the group skips
            // verification entirely. This case repeats once per function per
            // fixpoint, so its cost is paid on every run of the pipeline.
            group.bench_with_input(BenchmarkId::new("group_unchanged", &id), &ssa, |b, ssa| {
                let mut transaction = PassTransaction::<MockTarget>::new();
                b.iter_batched_ref(
                    || ssa.clone(),
                    |working| {
                        black_box(transaction.run_group(working, |_| false));
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

/// What the exception value model costs to maintain.
///
/// A handler's phi carries an operand per program point a throw could have
/// left from, so a protected region's phis are wider than a merge's - and
/// reconstruction has to derive them. The question is whether that width is a
/// cost a host pays per repair, or one it pays once at construction: the two
/// answers point at different places to optimise.
fn bench_exception_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("exception_model");
    let fixture = testing::try_catch_fixture();

    group.bench_function("construct_ssa", |b| {
        b.iter_batched_ref(
            || fixture.clone(),
            |ssa| {
                let _ = black_box(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("exceptional_webs", |b| {
        b.iter(|| black_box(fixture.exceptional_webs().len()));
    });

    group.finish();
}

/// What giving every instruction an identity costs.
///
/// Every repair boundary stamps the instructions a pass added, so this is paid
/// per pass rather than once. It is a linear walk, and the point of measuring
/// it is to know it stays one.
fn bench_instruction_identity(c: &mut Criterion) {
    let mut group = c.benchmark_group("instruction_identity");

    for scale in SCALES {
        let fixture = testing::straight_line_fixture(scale);
        group.bench_with_input(
            BenchmarkId::new("assign_instr_ids", scale),
            &scale,
            |b, _| {
                b.iter_batched_ref(
                    || fixture.clone(),
                    |ssa| black_box(ssa.assign_instr_ids()),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Whether a host that carries no facts pays for the mechanism.
///
/// `()` facts are zero-sized and every funnel is supposed to compile away, so
/// the two sides of this group should sit on top of each other. A gap means
/// the "costs nothing" claim in the documentation has stopped being true.
fn bench_facts_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("facts_overhead");

    let without: SsaFunction<MockTarget> = testing::diamond_phi_fixture();
    group.bench_function("unit_facts/construct_ssa", |b| {
        b.iter_batched_ref(
            || without.clone(),
            |ssa| {
                let _ = black_box(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
            },
            BatchSize::SmallInput,
        );
    });

    let with: SsaFunction<FactfulTarget> = testing::factful_diamond_phi_fixture();
    group.bench_function("host_facts/construct_ssa", |b| {
        b.iter_batched_ref(
            || with.clone(),
            |ssa| {
                let _ = black_box(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_maintenance,
    bench_scaling,
    bench_analysis_construction,
    bench_pass_cost,
    bench_pass_transaction,
    bench_exception_model,
    bench_instruction_identity,
    bench_facts_overhead
);
criterion_main!(benches);
