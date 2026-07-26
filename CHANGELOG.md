# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-07-26

Speculative evaluation for `SsaEvaluator`. Consumers that explore alternative
executions — a symbolic executor taking both arms of a branch, a control-flow
deobfuscator tracing every path through a dispatcher — previously had to clone
the evaluator per fork, paying for everything learned so far on every fork. A
checkpoint costs nothing to take and one journal entry per mutation to undo, so
the cost tracks the speculation rather than the accumulated state.

### Added

- **`SsaEvaluator::checkpoint` and `SsaEvaluator::rollback`**, with the
  `EvaluatorMark` token they exchange (re-exported from `analysis`) — mark a
  state, evaluate speculatively, and restore exactly what was marked. Marks
  nest, and rolling back to an outer mark subsumes any inner ones, so an
  abandoned branch is discarded in one step rather than unwound level by level.
  Rolling back a stale mark leaves the state unchanged rather than corrupting
  it. Journaling is latched on by the first checkpoint, so evaluators that never
  speculate are unaffected; when `track_memory` or `track_path` is on, the mark
  additionally snapshots those two structures outright instead of journaling
  them.

### Performance

- **Trivial-phi elimination in rebuild mode is no longer quadratic.** The
  rebuild path substituted each trivial phi one at a time, scanning the whole
  function per substitution, and a rebuild produces trivial phis in proportion
  to the function — so every fixpoint round scaled with the square of the
  function. The substitutions are now composed into a single map (back-to-front,
  which reproduces the order-sensitive sequential result exactly) and applied in
  one pass. The repair path had already been corrected for this in 0.4.0; this
  brings the rebuild path in line.

## [0.4.0] - 2026-07-25

Memory optimization arrives: a Memory SSA-backed pass doing store-to-load
forwarding, redundant load elimination, and dead store elimination, together
with the alias machinery it needs — points-to and address analyses, address
spaces on indirect accesses, and a rewritten `MemoryLocation::Indirect` that
keys on the decoded address rather than the SSA value holding it. Alongside it:
an interprocedural summary framework, a transactional pass-group primitive
(`scheduling::PassTransaction`), and a widening hook that gives the generic
dataflow solver a termination argument for infinite-height domains.

Normalization is substantially faster on large functions. Profiling a real
17,284-function binary found nearly all of its lift time in one 27,384-block
function, spent on work that scaled with the function rather than with the
change being made. Fixing that takes the binary's lift from **1014.5 s to
34.6 s**. Separately, trivial-phi elimination went from cubic to quadratic —
**26 s to 48 ms** on a 1,600-phi chain.

### Added

- **Memory optimization pass** (`passes::memory`, `MemoryOptimizationPass`) —
  store-to-load forwarding, redundant load elimination, and block-local dead
  store elimination, every rewrite gated on an alias proof rather than a
  syntactic shape. Calls, fences, atomics, volatile prefixes, and opaque native
  operations are barriers.
- **Points-to and address analyses** (`analysis::pointsto`, `analysis::address`)
  — an inclusion-based (Andersen) constraint solver with a field-sensitive
  intraprocedural extractor, and a normalized `base + index*stride + offset`
  address model with a cell-identity `AliasKey` projection.
- **`Target::field_member_index`** — optional host accessor giving field-sensitive
  points-to a stable per-field cell identity.
- **Address spaces on indirect accesses.** `SsaOp::{LoadIndirect, StoreIndirect}`
  carry `address_space: Option<u16>`, and alias analysis treats distinct spaces
  as disjoint — covering x86 `FS:`/`GS:`.
- **Interprocedural summary framework** (`interproc`) — `CallGraph` over an
  opaque method key plus `solve`, driving a caller-supplied `SummaryTransfer`
  over the call graph's SCCs, callees before callers, with a bounded fixpoint on
  recursive components.
- **Transactional pass execution** (`scheduling::PassTransaction`) — runs a pass
  group against one reusable snapshot, verifies once, and rolls back on verifier
  rejection or panic, re-running the group pass-by-pass to isolate the culprit.
  Offered to hosts driving their own pass loop; `PassScheduler` snapshots per
  pass under `verify_hard` and does not use it.
- **`EventKind::LoadForwarded` and `EventKind::DeadStoreRemoved`**, with matching
  event-builder support.
- **`DataFlowAnalysis::widen`** — a widening hook the solver applies between
  iterations, giving infinite-height domains a termination argument.
- **`ValueRange` implements `MeetSemiLattice` and `JoinSemiLattice`**, so range
  domains compose with the generic solver.
- **`VariableOrigin::EntryLiveIn`** — a value the caller supplies whose position
  in the signature is not known. The other four variants describe a CIL method,
  which has a declared signature and local table; a machine-code front end has
  neither, so a register read before the body ever writes it is neither an
  argument at a known index nor a declared local. Front ends previously had to
  stamp such a value `Argument(0)`, which asserted a signature position the
  lifter has not recovered — and, at `rebuild`'s origin-keyed grouping, collapsed
  every live-in of a function into argument group 0. `is_entry_defined` accepts
  the new variant, so a read of one needs no definition in the body; `Phi` stays
  rejected, so a variable whose definition was destroyed still cannot masquerade
  as an entry value. ABI recovery may later reclassify one as a real
  `Argument(n)`.
- **`VariableOrigin::is_entry_live_in` and `is_caller_supplied`** — the latter
  covers `Argument` and `EntryLiveIn` together, which is the predicate consumers
  actually want when asking "does the body have to define this?".
- **The entry-defined origin set is now a tested public contract**
  (`tests/verifier.rs`), not a verifier implementation detail. Tightening it
  previously broke a downstream lifter with no failure in this suite.
- **Fuzz targets** (`fuzz/`) over `SsaFunction::validate` and the pass pipeline,
  fed deliberately malformed IR. Wired into CI as a smoke gate.

### Changed (breaking)

- **`VariableOrigin` gained a fifth variant** (`EntryLiveIn`, see **Added**). The
  enum is not `#[non_exhaustive]`, so any exhaustive `match` on it downstream
  needs a new arm; a `_` arm compiles unchanged, but check whether it should
  instead join the `Argument` arm — the question to ask is whether the site cares
  about *caller-supplied* (use `is_caller_supplied`) or about a *signature
  index*. All in-crate matches were audited: `rebuild`'s group derivation and
  version-stack seeding treat `EntryLiveIn` as naming no argument/local slot, and
  `deadcode`'s `LoadArg`/`LoadLocal` liveness bridge excludes it, since a
  machine-code front end emits neither op. Serde encoding is additive but not
  backward compatible in the reading direction: a 0.3.0 deserializer rejects a
  payload containing the new variant.
- **Removed the block-duplication methods on `SsaFunction<T>`** —
  `allocate_fresh_variables_for_block`, `clone_block_with_remap`,
  `duplicate_block`, `remap_block_targets`. Shipped in 0.2.0 with no caller in
  this crate or any consumer, and never executed by a test. Adding tests showed
  why that mattered: they draw ids from the variable allocator without appending
  the matching rows, breaking the `variables[i].id().index() == i` invariant that
  `SsaFunction::variable` indexes on. Reimplement against the invariant when a
  pass needs it; the code remains in history at `2aaa81b`.
- **The address model is now target-pointer-width aware.** `normalize_address`,
  `alias_keys_for_function`, `MemorySsa::build`, and `passes::memory::run` take a
  `PointerSize`, and `IndirectLocation` carries one. Address arithmetic wraps at
  the pointer width, and without it a sign-extended `-8` and a zero-extended
  `0xFFFF_FFF8` — the two lowerings a 32-bit frontend emits for one displacement
  — decoded 4 GiB apart, so `may_alias` proved two accesses to *one cell*
  disjoint. Pass `SsaPassHost::ptr_size()`; `PointerSize::Bit64` reproduces the
  previous behaviour exactly.
- **`SsaPassHost::events()` now returns `&dyn EventListener<T>`** instead of
  `&EventLog<T>`, so a host can supply `NullListener` rather than retaining every
  event for the process lifetime. `EventListener` gained `recorded_count()` and
  `count_by_kind_since()`, both defaulting to "no history". `&self.events` still
  coerces.
- **`ConstValue::to_bytes` takes a `PointerSize`** and emits pointer-width bytes
  for `NativeInt`/`NativeUInt` rather than always eight.
- **Removed `SsaEditScope::StructuredCfg`.** It ran `rebuild_ssa` exactly as
  `CfgModifying` does while documenting a cheaper boundary it never had. Use
  `CfgModifying`.
- **`SsaOp::LoadIndirect` and `SsaOp::StoreIndirect` gained an `address_space`
  field.** Exhaustive constructions must supply it; `None` preserves previous
  behaviour. Patterns using `..` are unaffected.
- **`MemoryLocation::Indirect` now carries an `IndirectLocation`** (decoded
  base/index/stride/offset plus access width and address space) instead of a bare
  `SsaVarId`. See **Fixed**.
- **`World::methods_reverse_topological` now returns `Vec<Vec<T::MethodRef>>`** —
  the call graph's SCCs in reverse topological order rather than a flat list. The
  grouping is load-bearing: a non-recursive component converges in one visit
  while a recursive one needs a bounded fixpoint, and a flat list cannot express
  that. The default now derives the order from `callees` instead of falling back
  to `all_methods` with no ordering guarantee.
- **The crate is now edition 2024** (from 2021). `rust-version` is unchanged at
  1.88.
- **`ir::ops` is now a directory module**, split into `def`, `kinds`, `vector`,
  `native`, `effects`, `visit`, `control`, `classify`, and `display`. Additive
  only: every type and method is re-exported from `ir::ops`, verified by diffing
  the public surface (108 types, 80 methods) before and after.

### Fixed

**Miscompiles.** Passes could produce wrong optimized IR, silently, in release
builds, on shapes that are ordinary for lifted native code:

- **Secondary (`flags`) definitions were dropped.** `BinaryOpInfo` presents a
  two-definition op through a single-`dest` view and `value_key()` omits `flags`,
  so GVN deleted an instruction whose flags a later `BranchFlags` still read.
  Strength reduction and reassociation built replacements with `flags: None`.
  All three now check *liveness* of the flags definition — a dead one still
  optimizes, a live one declines the rewrite.
- **Memory SSA versioned only the location a definition named**, so a barrier or
  an overlapping store left a may-aliasing cell's version untouched and the
  memory pass forwarded a value that had already been destroyed. Versioning now
  covers the may-alias closure, in both phi placement and renaming.
- **Range propagation proved `x & negative_mask` was the constant 0**, folding
  live branches away; `And` now delegates to `ValueRange::and_constant`.
- **A phi folded to a constant when its first operand was phi-defined**, deleting
  the opposite arm. Self-referential loop phis now fold correctly too.
- **Reassociation overwrote a shared `Const` in place**, corrupting every other
  reader; the use-count guard checked the intermediate result, not the constant.
- **Chained shifts combined past the operand width**, where masking semantics
  make two shifts unequal to one.
- **SCCP treated every second incoming edge of a merge as a back edge**,
  collapsing all merge precision, and never pruned a constant branch in the entry
  block. Back edges are now detected structurally.
- **Alias analysis treated distinct SSA ids as distinct objects.** Two ids
  routinely name one object, so `NoAlias` was unsound; `must_alias` is unchanged,
  so forwarding precision is preserved.
- **Self-cancelling identities fired on floats** (`x - x` is NaN, `x == x` is
  false for NaN) and emitted `I32` constants regardless of operand width.
- **Constraint reasoning read "cannot compare" as "proved"** (`is_none_or` where
  `is_some_and` was meant), and recorded signed constraints on the false edge of
  unsigned comparisons.
- **LICM hoisted uses above their definitions** — the trampoline filter ran after
  the operand-availability fixpoint; both now run to a joint fixpoint.
- **Block merging destroyed exception-region boundaries** that its own coalescing
  path protects, and treated `Leave` as a plain forwarding jump.
- **`Rcl`/`Rcr` folded as plain rotates.** They rotate *through the carry flag*,
  which is not an SSA operand; they no longer fold.
- **Shifts folded with the distance masked by the value's own type**, which
  matches no ISA. Out-of-range distances no longer fold.

**Robustness on malformed input.** The crate's inputs are attacker-controlled
binaries, and these shapes abort the process rather than being reported:

- **The verifier and several passes panicked on IR naming a non-existent block.**
  Raw `BitSet::contains`/`insert` assert, and block and variable indices taken
  straight from the IR reached them in `check_dominance`, `place_pruned_phis`,
  `SsaRebuilder::pre_clean_unreachable`, `DefUseIndex::build`, loop analysis, and
  LICM. All now use the bounds-tolerant accessors. The last four were found by
  the new fuzz targets, not by inspection.
- **`VarSet` now grows rather than dropping an out-of-range variable id**, which
  previously made the variable read back as absent — "dead", for a liveness set.
- **Symbolic expressions are bounded by node count, not only depth.** `x op x`
  doubles the node count while adding one to the depth, so the depth cap could
  not fire before memory was exhausted.
- **Memory SSA is bounded by a `locations × blocks` budget**, degrading to
  "optimize nothing" instead of exhausting memory; the points-to budget now
  counts work rather than worklist pops.
- **A panicking pass no longer destroys a method.** The scheduler *takes* the SSA
  before running a pass, so an unwind skipped every reinsertion.

**Verifier gaps.** Each of these let a class of corruption ship undetected:

- **A read of a variable nothing defines** is now reported as `UndefinedUse`.
  Two things hid it: `refresh_def_sites`, `strip_nops`, and block compaction all
  rewrote a destroyed definition to `DefSite::entry()`, making it look like an
  argument; and the orphan check exempted exactly that shape. This is what let
  three passes drop a `flags` definition undetected. One variable legitimately
  has no definition and is exempt: the exception object a catch or filter
  handler receives from the runtime, identified by the `Pop` at the handler's
  entry. `SsaRebuilder` already carved out that shape, and without the matching
  carve-out here the verifier rejected the IR the rebuilder is built to produce
  — failing every method with a handler as soon as a CFG-modifying pass
  triggered a rebuild.
- **A non-dense variables table** is now reported as `VariableTableNotDense`.
  `SsaFunction::variable(id)` is a raw index, so a dropped row silently redirects
  every higher id to a different variable's def site, type, and uses.
- **Duplicate phi operands on one edge** are now reported. The check compared
  operand predecessors as a *set*, so duplicates collapsed to one bit.
- **Boundary verification defaults to `VerifyLevel::Full`**, which includes the
  dominance check a mis-scoped edit breaks. Opt down with
  `SsaEditOptions::with_verify_level`.

**Determinism.** The downstream similarity pipeline is content-addressed and
requires byte-identical output: Memory SSA location iteration and
`IndexedGraph::find_any_cycle` both walked hash containers.

**Pre-existing soundness defects**, surfaced while building the memory and
numeric analyses:

- **Value-range propagation could prove a loop guard that is not true.** The
  analysis is optimistic, so its conclusions are sound only *at* the fixpoint,
  but it terminated on an iteration cap — and a truncated run leaves ranges too
  narrow, which proves comparisons that are not provable. On `i = 0; while i < 10
  { i += 1 }` the guard folded to constant `true` below a budget of 32, and the
  next pipeline iteration deleted the loop exit. Convergence is now load-bearing
  (`RangeResult` answers `None` to every query unless the fixpoint was reached)
  and widening provides the termination argument in place of the cap.
- **Range arithmetic ignored the destination width**, so a wrapping 32-bit `add`
  read as positive in `i64` and folded `sum < 0` to `false` — the opposite of the
  truth. Results escaping the declared width now yield no information. `Neg`,
  `Shl`, `Or`, and `Xor` transfer functions were added at the same time.
- **Memory SSA renaming leaked versions across sibling dominator subtrees.** The
  walk pushed versions but never popped on subtree exit, so a block's entry
  version depended on traversal order. Renaming is now an explicit-stack DFS that
  restores scope.
- **`MemoryLocation::Indirect` keyed on the address SSA value, not the address.**
  Accesses at different offsets off one base were unrelated rather than provably
  disjoint, and — the unsound direction — two `Indirect` locations with different
  address values were reported `NoAlias`, though two SSA pointers can hold one
  address. Identity was also unstable under GVN and LICM.
- **Array-element locations never used a constant index.**
  `ArrayIndex::Constant` was unreachable, so every array element may-aliased
  every other.

### Performance

**Large-function normalization.** A profile of a real 17,284-function PE (ANGLE
`libGLESv2.dll`) found 96 % of its lift time in a single 27,384-block function,
spent on work that scaled with the function rather than with the change being
made. Six fixes, in descending order of impact:

- LICM opened an edit scope — a whole-function snapshot, repair, and verification
  — *per loop*, and that function has 3,633 loops. It now processes the whole
  loop forest in one scope.
- `replace_uses_checked_with` scanned every instruction to find one variable's
  uses, once per redundant pair in GVN. The editor now maintains the use index
  across the session.
- `var_id_capacity` walks the function for the largest variable id, and eleven
  pass sites called it purely to size a bitset. The new O(1) `var_id_bound`
  replaces it for sizing.
- `SsaVarId::PLACEHOLDER` encodes `u32::MAX - 1`, and `var_id_capacity` folded it
  in like a real id — so a function holding an unrenamed phi reported ~4.29
  billion and callers allocated 512 MB bitsets. Now excluded.
- The verifier keyed four `HashMap`s on `SsaVarId` (SipHash on a dense integer)
  and allocated one per block.
- `get_definition` fell through to a whole-function scan for any phi-defined
  variable, which can only return `None`.

Net effect: whole-binary lift **1014.5 s → 34.6 s (29×)**, worst function
**978.1 s → 7.5 s (129×)**. Functions over 100 ms fell from 26 to 11. The one
behavioural change is LICM's failure granularity: a verifier rejection now
discards that invocation's hoists rather than one loop's.

**Algorithmic fixes.**

- **Trivial-phi elimination was cubic.** `replace_uses_checked_with` rewrites
  instruction uses only, so a phi feeding another phi survived its round and the
  fixpoint ran once per phi. Phi operands are now collapsed in the same round:
  **200 phis 42 ms → 0.66 ms, 800 phis 3.0 s → 10 ms, 1600 phis 26 s → 48 ms**.
- `SsaCfg::from_ssa` deduped synthetic handler edges with a scan of the whole
  edge list per handler — O(H·E) on an attacker-controlled handler table, on a
  structure most passes rebuild.
- The scheduler built the call graph twice per pass batch, inside its own
  fixpoint.
- Dominance frontiers allocated a dense n×n bitmatrix per call; rows are now
  allocated on first write (`BitSet::lazy`).
- `trace_to_phi` recursed into both operands of every binary op with no
  memoization — 2²⁰ traversals under its depth cap.
- `IndexedGraph::find_any_cycle` restarted a DFS from every node, O(V·(V+E)); it
  is now one Tarjan pass.
- Per-iteration rebuilds removed from block merging, control-flow simplification,
  loop canonicalization, copy propagation, and DCE; `compute_nesting` no longer
  recomputes a `BitSet` popcount per sort comparison; the dataflow solver folds
  the meet in place via `MeetSemiLattice::meet_into`.

**Allocation.** `DefUseIndex` and `SsaCfg::from_ssa` now use flattened
offsets-plus-values arrays instead of `Vec<Vec<_>>`; `SsaBlock`, `SsaVariable`,
and `PhiNode` implement `clone_from` so snapshot capacity reuse reaches inside
per-block and per-variable buffers; the verifier and rebuild repair loops reuse
scratch buffers instead of allocating per instruction; the points-to solver no
longer clones on every worklist pop. Measured on a native-lift consumer across
six per-architecture real-binary fixtures: **−37…−43 % allocations**,
**−13…−23 % allocated bytes**, **−13…−32 % wall-clock** on lift-and-verify
(p = 0.00).

Two supporting API additions, usable by hosts: **`ir::VarSet` and
`ir::VarMap<T>`**, dense containers keyed by `SsaVarId` that hold the rule that a
placeholder is not a variable so it cannot be forgotten at a call site; and
**`SsaFunction::var_id_bound`**, the O(1) sizing bound above.

`benches/ssa_repair.rs` measures these costs directly, including a size sweep
putting `rebuild_ssa` at 20–30× a whole-function snapshot — stable across two
decades of function size — and `PassTransaction` group execution at 1.6–1.8×
faster than per-pass snapshotting, rising to 20–28× on the terminating fixpoint
iteration.

## [0.3.0] - 2026-07-16

Native operation coverage for modern lifters — x87/FPU, SVE/SME, AMX tiles,
vector crypto, system/compute intrinsics, PAC — plus a domain-typed conversion
family replacing the single `Conv` op. Several effect-classification and
value-numbering correctness fixes ship alongside; see **Fixed**.

### Changed (breaking)

- **`SsaOp::Conv` is removed**, replaced by a domain-typed conversion family so
  each conversion carries only the fields that are meaningful for it:

  | Old | New | Notes |
  |-----|-----|-------|
  | `Conv` (int → int) | `IntConv` | Identical field set — mechanical rename. |
  | `Conv` (int → ptr) | `IntToPtr` | No `overflow_check`/`unsigned`. |
  | `Conv` (ptr → int) | `PtrToInt` | No `overflow_check`/`unsigned`. |
  | `Conv` (int → float) | `IntToFloat` | Keeps `overflow_check`/`unsigned`. |
  | `Conv` (float → int) | `FloatToInt` | Keeps `overflow_check`/`unsigned`. |
  | `Conv` (float → float) | `FloatConv` | Width change only; drops both flags. |

  `PtrAdd` is a new address-computation op, not a `Conv` replacement.
- **`SsaOp` gained 55 variants** (146 → 200), and `VectorUnaryKind` (+20),
  `VectorBinaryKind` (+23), `VectorCompareKind` (+8), `VectorTernaryKind` (+3),
  `AtomicRmwOp` (+3), and `SsaOpClass` (+`NativeIntrinsic`) also grew. No enum in
  this crate is `#[non_exhaustive]`, so downstream exhaustive matches must handle
  the new variants. This is deliberate: a lifter that silently ignores an
  unhandled op is a bug, so op additions are intended to break the build.
- **`ConstValue` equality is now structural, and the type is `Eq + Hash`.**
  Float arms compare and hash **bitwise**, which makes `Eq` reflexive and lets
  constants be hash-map keys. Two visible consequences: `F64(NAN) == F64(NAN)` is
  now `true`, and `F64(0.0) == F64(-0.0)` is now `false` (they are
  distinguishable at runtime via `1.0 / x`). IEEE-754 *semantic* comparison is
  unchanged and still lives in `ceq` and the `c*_un` family.
- **`SsaOp` and its operand payloads are now `Eq + Hash`.** No `Target` change is
  required — the trait already demanded `Eq + Hash` on its associated types.

### Fixed

- **`VectorStructLoadReplicate` (AArch64 `ld2r`/`ld3r`/`ld4r`) was classified
  `Pure`** despite reading memory through its address base. GVN could CSE two of
  them across an intervening store, and DCE could delete them outright. It now
  classifies as `Read` + `TrapClass::MemoryFault` and reports `may_throw()`, like
  every other vector load.
- **GVN could merge distinct `ComputeFlags` computations.** `ComputeFlags` models
  the flags of `bsf`/`bsr`/`popcnt`/`bt` alike but carries no opcode
  discriminator, so its result is not a function of its SSA operands; two
  different native flag computations over the same operands produced one key and
  the wrong flags value. It is no longer value-numbered.
- **GVN could merge `CallClobber` markers**, aliasing the fresh undefined values
  of two different calls' caller-saved registers (all of its operands are
  definitions, which the key normalizes to a sentinel). It is no longer
  value-numbered. `Phi` is likewise excluded — it is block-relative, and the key
  does not encode the defining block.
- **`SsaEditor::nop_instruction` left the removed value's `result_type` stamped
  on the `Nop`.** `set_op_preserving_type` now clears the type when the new op
  has no destination.
- **`SystemOpKind::Barrier` (`dsb`/`dmb`/`isb`/`mfence`) was classified `Write`.**
  It is an ordering construct, not a clobber: it now classifies as `Fence`, so
  Memory SSA emits a `MemoryOp::Barrier` and the verifier's fence invariant
  applies.
- **`setffr`/`wrffr` were deletable by DCE.** They write the SVE first-fault
  register, which is not an SSA operand, and their `outputs` may be empty — pure
  plus zero definitions meant DCE removed them, dropping the FFR initialization a
  following first-faulting load depends on. They now report `Opaque` effects.
- **Pointer conversions sign-extended when folded.** `IntToPtr`/`PtrToInt`
  hardcoded a signed source, so a 32-bit `0x8000_0000` would widen to
  `0xFFFF_FFFF_8000_0000` on a 64-bit target. Pointers are unsigned and now
  zero-extend. (Latent: no in-tree `Target` implements `convert_const`.)

- **LICM did not converge on functions with deep loop-invariant dependency
  chains.** The hoist phase moved only one dependency "wave" per invocation —
  leaving dependent invariants for a later call — so deeply-chained invariants in
  large loops needed O(chain-depth) expensive invocations and routinely exhausted
  the driving fixpoint's iteration cap with work still pending. It now hoists a
  loop's entire invariant chain in one pass, inserting in topological order so a
  hoisted definition always precedes its hoisted uses.
- **Loop canonicalization oscillated against CFG simplification.** `controlflow`
  (jump threading) and `blockmerge` (trampoline elimination) now preserve
  canonical loop preheaders and unified latches instead of removing them as empty
  forwarding blocks — which `loopcanon` immediately re-inserted, so the
  normalization fixpoint never settled. Loop-simplify form is now stable (the
  same trade-off LLVM's `simplifycfg` makes).

### Performance

- **GVN's generic value key is now structured rather than a formatted string.**
  It stores the operand-normalized `SsaOp` and probes via derived `Eq`/`Hash`,
  removing a per-candidate deep clone, a `Debug` render, and a `String`
  allocation, and turning every map probe from a string compare into a
  structural one.
- LICM `can_hoist` is O(1) per candidate instead of O(loop). The
  "result feeds a phi on a loop back-edge" test is precomputed once per loop as a
  single backward taint propagation seeded from the back-edge phi operands,
  replacing a fresh forward def-use traversal per candidate
  (O(candidates × loop) → O(loop)).
- LICM invariant detection and hoist-availability no longer scan every variable
  in the function once per loop; the loop-body-defined variable set is built once
  per loop in O(loop-body) (O(loops × variables) → O(loops × loop-body)).
- Loop canonicalization re-analyzes the loop forest once per pass instead of once
  per transformation: each pass now canonicalizes every loop the forest reports
  (still one transformation per loop per pass, preserving phi-management
  simplicity), turning O(transformations × loop-analysis) into
  O(passes × loop-analysis).
- GVN builds the CFG and dominator tree once per run rather than rebuilding it
  (and rescanning every block) for each redundant value pair.

Together these make the normalization pipeline converge in a couple of iterations
on control-flow-flattened inputs (many small nested loops with long invariant
chains), where it previously ran to the iteration cap without converging and
spent super-linear time per call. Output is unchanged.

### Added

- Native intrinsic modeling: `NativeIntrinsic`, `SystemOp`, `ComputeOp`,
  `VectorCrypto`, `TileOp`, `BcdAdjust`, x87/FPU (`FpTranscendental`,
  `FpuControl`), and pointer authentication (`PacKind`).
- SVE/SME and AMX operations: predicate/first-fault ops, SVE compute, SME
  outer-product and ZA-tile ops, matrix multiply-accumulate, tile operations.
- **`serde` feature** (off by default): serializes the IR data model — the
  operation-kind taxonomy, the native descriptors, and the SSA graph itself
  (`SsaFunction` and everything reachable from it: blocks, instructions, ops,
  phis, variables, constants, exception handlers).

  Generic IR types carry `#[serde(bound(...))]`, so `SsaFunction<T>` is
  serializable exactly when the host's `Target` associated types are. Hosts that
  don't serialize the IR are unaffected: the impl does not apply to them, and no
  serde bound is forced onto `Target`.

  Two encoding decisions are part of the wire contract:
  - `SsaVarId` serializes as its **logical index**, not its internal
    complement encoding (an `Option`-niche layout optimization that must not leak
    into a persisted format).
  - Maps keyed by `VariableOrigin` encode as `(key, value)` sequences. A derived
    map would compile but fail at runtime with "key must be a string" in every
    format that requires string keys, JSON included.

  Not covered: borrowed views (`SsaDefs<'a>`, `MemoryEffect<'a, T>`), transient
  pass machinery (builders, editors, edit reports), and `SsaFeatureToken` — its
  `&'static str` opcode can serialize but cannot deserialize, so a derive there
  would be a one-way trap.
- `num_enum::{IntoPrimitive, TryFromPrimitive}` on seven kind enums. **Note:**
  this pins those enums' numeric discriminants as public API — inserting a
  variant mid-enum silently remaps any value a host persisted by discriminant.
  The `serde` derives encode by variant *name* and are not affected.
- New public helpers: `SsaVarId::as_u32`, `SsaInstruction::set_op_preserving_type`,
  `SsaOp::{visit_operands, visit_operands_mut, replace_uses_with,
  arith_signedness, compare_kind, memory_effect}`, and a `fp_classify` builder.
- `ConstValue` scalar folds recurse lane-wise into `Vector` operands (previously
  returned `None`).
- `SsaEditor::replace_uses_checked_with` — replace instruction uses against a
  caller-supplied dominator tree, letting a pass that rewrites many variables
  (e.g. GVN) build the tree once and reuse it across the whole batch instead of
  per replacement.
- `BitSet::{contains_checked, insert_checked}` — bounds-tolerant accessors that
  treat an out-of-range index as unset rather than panicking, for callers whose
  index comes from data that may legitimately name a position outside the set
  (e.g. a terminator referencing a block that was never recovered).
- The lattice traits are re-exported from `analysis::dataflow`
  (`MeetSemiLattice`, `JoinSemiLattice`, `Lattice`), matching every other trait
  the module documents. They were previously reachable only through
  `analysis::dataflow::lattice::`, so the path the docs told you to use did not
  compile.

## [0.2.0] - 2026-06-03

Major release: a target-agnostic **native SSA substrate** for modern lifters,
new construction/editing infrastructure, and a broad performance/memory pass.
A few IR enum shapes changed (see **Changed**).

### Added

- Native SSA substrate: target-independent pointer sizes & endianness;
  first-class SIMD/vector operations with target-independent lane/vector-type
  semantics; native atomics (exchange, lock-RMW, compare-exchange); native
  opaque operations with machine-state clobbers; multi-output SSA definitions;
  boolean ops and native condition helpers; native flag semantics; and implicit
  wide (low/high, quotient/remainder) arithmetic.
- Memory effect summaries, exception/interrupt support, and native operation
  classes plus target-generic feature tokens.
- Fluent SSA builder (`ir::function::builder`) and a checked, verifier-preserving
  SSA editor (`ir::function::editor`); all built-in passes migrated onto the
  checked mutation APIs.
- Recommended normalization-pipeline API, pass bisection/debug hooks, and
  structured verifier diagnostics.
- Expanded `Target` trait (vector descriptors, pointer sizes, endianness),
  `MockTarget`, and a much larger test suite (builder, scheduling, verifier,
  pipeline, and canonicalization coverage).
- Allocation-free helpers: `SsaOp::{uses_var, for_each_successor, has_successor}`,
  `SsaInstruction::{uses_var, for_each_variable}`, `SsaBlock::for_each_successor`,
  `SymbolicExpr::for_each_variable`, `SsaFunction::compute_predecessors`,
  `BitSet::is_full`, `EventLog::into_events`, `EventListener::is_enabled`.

### Changed (breaking)

- `SsaOp::NativeOpaque` is now a tuple variant wrapping a boxed payload
  (`NativeOpaque(Box<NativeOpaqueData>)`) instead of an inline struct variant.
- `ConstValue` heap-bearing arms are now boxed: `Vector(Box<[ConstValue<T>]>)`,
  `DecryptedString(Box<str>)`, `DecryptedArray(Box<DecryptedArrayData<T>>)`.
- `SsaOp` and `ConstValue` gained many new variants for the native substrate;
  exhaustive downstream matches must handle them.

### Performance & memory

- Shrunk core IR types ~3–4×: `SsaOp` 168→40 B, `SsaInstruction` 176→48 B,
  `Option<SsaVarId>` 16→4 B (niche-encoded `SsaVarId`), `ConstValue` 40→24 B,
  `PhiOperand` 16→8 B; guarded by a `size_of` regression test.
- Removed hot-loop allocations (`uses()`/`successors()` purge, word-skipping
  `BitSet` iterator, DFS/postorder scratch reuse, cached solver exit set).
- `DefUseIndex` uses dense `Vec`-indexed storage instead of `BTreeMap`.
- De-quadratized DCE, GVN removal, jump-threading safety check, LICM invariant
  detection, predicate branch evaluation, trivial-phi predecessor build, and
  block-merge redirection; dataflow solver uses an RPO-priority worklist.
- `DirectedGraph` stores neighbor ids inline, `IndexedGraph` dedups edges in
  O(1), and cycle detection is iterative (stack-safe on deep graphs).
- Scheduler dirty-set membership is O(1); logging allocates nothing under
  `NullListener`.

## [0.1.0] - 2026-05-09

Initial standalone release.

### Added

- Target-agnostic SSA IR for blocks, instructions, phi nodes, variables, values,
  exception handlers, and functions.
- SSA analyses for CFGs, constants, liveness, memory, phi placement, symbolic
  expressions, dataflow, def-use, loop structure, and verification.
- Optimization and deobfuscation passes including algebraic simplification,
  block merging, control-flow cleanup, copy propagation, dead-code elimination,
  GVN, LICM, loop canonicalization, reassociation, scheduling, strength
  reduction, and jump threading.
- Generic graph and bitset utilities used by the IR and analyses.
- Host adapter traits and a pass scheduler for integrating analyssa into target
  lifters without tying the crate to one instruction set or metadata model.
