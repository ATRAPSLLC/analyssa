# analyssa

Target-agnostic SSA IR, analyses, and optimization pipeline for lifters,
deobfuscators, and binary-analysis tooling.

[![Crates.io](https://img.shields.io/crates/v/analyssa.svg)](https://crates.io/crates/analyssa)
[![Docs.rs](https://img.shields.io/docsrs/analyssa)](https://docs.rs/analyssa)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Status

`analyssa` is a standalone crate for target-agnostic SSA construction, analysis,
optimization passes, and scheduler integration.

## What's here

- `bitset` / `graph` - generic data-structure utilities.
- `ir` - SSA IR core: blocks, instructions, phi nodes, variables,
  ops, values, exception handlers, functions.
- `analysis` - SSA-level analyses: CFG, constants, evaluator, liveness,
  memory SSA, points-to and address analysis, patterns, phi placement,
  resolver, verifier, symbolic execution, and dataflow.
- `passes` - reusable optimization and deobfuscation pass bodies.
- `interproc` - call graph plus an SCC-ordered summary solver for
  interprocedural analyses built on caller-supplied transfer functions.
- `scheduling` / `host` / `world` - traits and scheduler glue for host crates
  that own method storage, dirty tracking, and interprocedural traversal,
  including transactional pass-group execution with rollback.
- `target` - `Target` trait + `MockTarget` for unit testing without a
  concrete instruction-set host.
- `pointer` - `PointerSize` for hosts to communicate target pointer
  width to generic passes.

## What a host works with

Five things carry most of the weight, and a host that knows them can read the
rest of the API from them.

- **Construction.** A front end emits blocks, instructions and an exception
  table, and says which values are versions of one storage location by putting
  them in a rename group. `SsaFunction::construct_ssa` does the rest: it places
  the phis over the iterated dominance frontier and resolves every read to the
  definition that reaches it. `finish_construction` is the same finishing
  sequence alone, for a front end that placed its own phis. The order those
  steps run in is the contract, and getting it wrong is quiet - prune before
  placing phis and a handler loses its operands.
- **Instruction identity.** `InstrId` is what an instruction keeps while passes
  move it. Anything a host wants to remember about a particular instruction is
  keyed by that rather than by a position, which the next pass invalidates
  without telling anyone. `locate_instruction` turns an id back into a place.
- **Facts.** What a host knows about a value, an instruction or a function
  lives on it - `Target::VariableFacts` and its two siblings - instead of in a
  table beside it. Values are created through funnels that say where they came
  from (`create_variable_derived`, `create_variable_joined`), so a fact follows
  a value through renaming, substitution and rollback. `()` is the type for a
  host that carries none, and costs nothing.
- **The exception value model.** A throw leaves a protected block from wherever
  it happened, so what a handler finds in a slot is the join over every program
  point the throw could have come from. A phi at a handler entry says that in
  its operands: `PhiEdge::Exceptional` marks the ones a throw brings, beside the
  normal ones a terminator names. The object a handler is entered with has a
  definition, `SsaOp::RuntimeValue`, rather than an exemption from the rules.
- **The pipeline.** `PipelineSpec` states the groups and what protects each one;
  `FunctionPipeline::run` drives the fixpoint and returns a `PipelineReport`.
  The report is the point: a group that fails verification is rolled back *and
  reports no change*, so a driver that returns only "changed" cannot tell a
  function with nothing to do from one whose every rewrite was discarded.
  `rollbacks` is empty on a healthy run; when it is not, re-run with
  `PipelineSpec::diagnose` to find out which pass.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | Derives `Serialize`/`Deserialize` across the IR data model: the operation-kind taxonomy, the native descriptors, and the SSA graph itself (`SsaFunction` and everything reachable from it). Generic types use `#[serde(bound(...))]`, so `SsaFunction<T>` serializes exactly when your `Target`'s associated types do - hosts that don't serialize the IR are unaffected, and nothing is forced onto `Target`. Excluded: borrowed views (`SsaDefs`, `MemoryEffect`), transient pass machinery (builders/editors), and `SsaFeatureToken` (its `&'static str` opcode cannot deserialize). |

## Usage

A consuming crate implements the `Target` trait for its host (mapping
the trait's associated types - `TypeRef`, `MethodRef`, etc. - to its
metadata model), then constructs `SsaFunction<T>` and runs analyses
or transformation passes against it.

```rust
use analyssa::{analysis::DefUseIndex, testing};

let ssa = testing::const_i32_return(42);
let index = DefUseIndex::build_with_ops(&ssa);

assert_eq!(ssa.variables().len(), 1);
assert!(index.full_definition(analyssa::ir::SsaVarId::from_index(0)).is_some());
```

`MockTarget` (`analyssa::testing::MockTarget`) is the minimal `Target`
implementation for tests and examples that do not need host-specific metadata.

## Development

```bash
# Nightly: rustfmt.toml uses `group_imports` / `imports_granularity`, which
# stable rustfmt silently ignores. Formatting only - the crate builds on stable.
cargo +nightly fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --doc --all-features
cargo llvm-cov --all-targets --all-features --summary-only
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --no-deps --all-features --document-private-items
cargo bench --bench ssarepair   # SSA maintenance costs; see benches/
```

The CI workflow runs these checks on pull requests and pushes to `main`.

## Minimum Rust Version

1.88. Pinned by `rust-version` in `Cargo.toml`; CI exercises both `1.88` and
`stable`.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md).

## License

Copyright 2026 ATRAPS LLC. Licensed under the Apache License,
Version 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
