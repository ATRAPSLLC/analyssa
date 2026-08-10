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

- `bitset` / `graph` — generic data-structure utilities.
- `ir` — SSA IR core: blocks, instructions, phi nodes, variables,
  ops, values, exception handlers, functions.
- `analysis` — SSA-level analyses: CFG, constants, evaluator, liveness,
  memory SSA, points-to and address analysis, patterns, phi placement,
  resolver, verifier, symbolic execution, and dataflow.
- `passes` — reusable optimization and deobfuscation pass bodies.
- `interproc` — call graph plus an SCC-ordered summary solver for
  interprocedural analyses built on caller-supplied transfer functions.
- `scheduling` / `host` / `world` — traits and scheduler glue for host crates
  that own method storage, dirty tracking, and interprocedural traversal,
  including transactional pass-group execution with rollback.
- `target` — `Target` trait + `MockTarget` for unit testing without a
  concrete instruction-set host.
- `pointer` — `PointerSize` for hosts to communicate target pointer
  width to generic passes.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | Derives `Serialize`/`Deserialize` across the IR data model: the operation-kind taxonomy, the native descriptors, and the SSA graph itself (`SsaFunction` and everything reachable from it). Generic types use `#[serde(bound(...))]`, so `SsaFunction<T>` serializes exactly when your `Target`'s associated types do — hosts that don't serialize the IR are unaffected, and nothing is forced onto `Target`. Excluded: borrowed views (`SsaDefs`, `MemoryEffect`), transient pass machinery (builders/editors), and `SsaFeatureToken` (its `&'static str` opcode cannot deserialize). |

## Usage

A consuming crate implements the `Target` trait for its host (mapping
the trait's associated types — `TypeRef`, `MethodRef`, etc. — to its
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
# stable rustfmt silently ignores. Formatting only — the crate builds on stable.
cargo +nightly fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --doc --all-features
cargo llvm-cov --all-targets --all-features --summary-only
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --no-deps --all-features --document-private-items
cargo bench --bench ssa_repair   # SSA maintenance costs; see benches/
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
