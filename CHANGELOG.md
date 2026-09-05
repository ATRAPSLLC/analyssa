# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [0.6.0] - 2026-09-04

Structured control-flow recovery arrives: any control-flow graph becomes a tree
of statements, total by construction, with the exception model rebuilt
underneath it — a clause is now three typed block ranges instead of five loose
indices, and one exception-aware flow view replaces five hand-written copies of
the same loop. An address becomes a value the IR can carry, so a function,
global or import stays distinguishable from the integer it lowers to. And the
native taxonomy gains an identity per instruction where it had one per class,
with an operation table that keeps those identities injective.

The defects fixed alongside share a shape with the 0.5.0 set, one step further
out: a fact about the function *asserted from metadata* rather than established
against the IR as it stands. A leader assumed to be available everywhere it was
found. A phi identified by an origin that does not identify it. A block index
taken from a def site, a phi operand, or an exception table and used to index a
set sized for the current CFG.

### Added

- **`analysis::structure`** — structured control-flow recovery: loops classified
  by where they test, multi-way branches, protected regions with their handlers,
  and conditions spanning several blocks (`Predicate`), which is what turns two
  nested branches back into `a && b`. Total by construction, with `Region::Goto`
  as the fallback for edges no structured form expresses, so quality is a count
  (`StructureMetrics::resized_inputs` and `depth_limited`) rather than a
  precondition.

- **`analysis::structure::BlockSet` and `StructureOptions::max_depth`** — the two
  types that make the recovery total. `BlockSet` carries the index domain it was
  built for, so `contains` is total over every `NodeId`; `max_depth` (with
  `DEFAULT_MAX_DEPTH`) bounds the recovery's stack *and* every later traversal of
  the result, the derived `Drop` and `Clone` included, and caps how many blocks
  one condition may fold. Reaching either costs quality, never correctness.

- **`ProtectedHandlerKind` and `HandlerFilter`** — the input side of a handler,
  split from `HandlerKind`, which now describes only a recovered one. A filter
  expression rides `ProtectedHandlerKind::Filter`, so a filter without an
  expression and an expression on a `finally` are unrepresentable.

- **`analysis::recovery`** — the bridge from an `SsaFunction` to that recovery.
  `structure_ssa` reads the exception table into `ProtectedRegions`, derives
  condition-only blocks from instructions, and returns `Recovered`: the tree
  *and* the clauses it could not convert (`RejectedClause`, `ClauseRejection`).
  Clauses sharing a try range become one region with several handlers, so
  `try/catch(A)/catch(B)` does not lose a catch.

- **`compute_post_dominators`, `control_dependences` and `ReverseGraph`** in
  `graph::algorithms`, with `PostDominatorTree` and `ControlDependences` — the
  results are indexed by the original nodes, over a reversed graph rooted at a
  virtual exit so a function with several returns, or none, still has one.
  `immediate_post_dominator` is the single bounds guard, and `strict_controllers`
  is the answer a doc sentence could not give: under Ferrante-Ottenstein-Warren a
  loop header lies in its own post-dominance frontier, so walking `controllers`
  as a parent chain does not terminate. `DominatorTree::is_reachable` exposes
  reachability the tree already knew, instead of a second BFS.

- **`analysis::exceptions`** — one exception-aware flow view, replacing five
  hand-written copies of one loop over the exception table: `FunctionRoots`
  (where control can enter), `EhCfg` (the graph in which those are reachable) and
  `EhDominance`. The last keeps apart three questions that were one answer used
  for all of them — `definition_reaches`, which knows a throw can interrupt a
  guard part-way, `dominates_block`, and the verifier's floor
  `definition_is_well_formed`.

- **`ExceptionBlocks` and `SsaFunction::exception_blocks`** — the block roles an
  exception table assigns (`is_runtime_entry`, `is_region_start`,
  `is_region_end`), indexed once instead of re-derived from five loose
  `Option<usize>` fields at eleven reader sites. It borrows the function, because
  an owned snapshot threaded through a mutating fixpoint goes stale unnoticed.

- **`ir::exception::BlockRange`, and one exception clause as three of them.** A
  clause part is a non-empty half-open block range, and the pairing is the type:
  a start without its end is not a value `BlockRange` can hold.
  `SsaExceptionHandler` carries `protected_range`, `handler_range` and
  `filter_range`, each an `Option`, which is the whole of a clause's legitimate
  partiality. `parts` and `entry_blocks` are the total views the barriers use;
  `layout` is the checked one, yielding a `ClauseLayout` and `LaidOutHandler` or
  an `ExceptionTableError`. `ClausePart` names which part a range belongs to,
  and `BlockRange::to_bitset` is the crate's only interval-to-set expansion.

- **`ir::exception::HandlerKind` and `Target::handler_kind`** — one handler
  taxonomy and one classifier, re-exported from `analysis::structure` so the enum
  the recovery reports and the enum the IR classifies are the same enum.

- **`ExceptionTableError` and `SsaVerifier::check_exception_table`, at
  `VerifyLevel::Quick`** — reports `VerifierError::MalformedExceptionClause` for
  a clause whose parts run past the last block, claim each other's blocks, or
  disagree about being a filter. `EmptyBlockRange` is the deserialization-side
  refusal of a pair covering no block, the one path no constructor guards.

- **`ConstValue::Symbol(T::SymbolRef)`, with `Target::SymbolRef` and
  `Target::symbol_address`** — a located or named entity used as a value. It
  folds exactly as the other reference constants do, which is to say not at all:
  address arithmetic stays the address model's job. `SymbolRef` is distinct from
  `MethodRef`, which names a *callable*, and `symbol_address` defaults to `None`,
  so hosts whose symbols are metadata tokens need write nothing.

- **`passes::algebraic` collapses redundant integer conversion chains.** A lift
  emits one conversion per width change, so a value narrowed, widened and
  narrowed again arrives as `(uint16_t)(uint64_t)(uint32_t)x` — 40.8% of `x86_64`
  conversions took another conversion as their operand. An intermediate is
  unobservable when the result is no wider than it *and* no wider than the
  source; where the result is wider, the two are interchangeable exactly when
  they agree on sign or zero. A conversion that checks overflow is never removed.
  Over the committed fixtures, `x86_64` integer conversions fell 26.7% and
  `arm_64` 15.8%; three-deep casts in rendered source fell 64%.

- **`analysis::convert::collapse_conversion_chain`** — that rule as a documented,
  testable analysis instead of a fast path buried in a pass, with an oracle
  matrix over all 2048 three-link shapes the mock target can express.
  `testing::MockConvLink` and `MockConversionChain` build a chain of any depth
  over any mock integer width, laid out so a test can damage one def site
  deliberately.

- **`SsaFunction::recorded_definition` and `RecordedDefinition`** — the crate's
  one guarded, scan-free def-site dereference. A def site is variable *metadata*:
  an edit that does not restate it leaves it naming whatever instruction now sits
  at that index, so dereferencing it unchecked answers with a foreign operation
  rather than failing. `get_definition`, `get_definition_instruction` and
  `try_constant_value` are built on it.

- **`analysis::cache::FunctionAnalyses`, `scheduling::AnalysisCache` and
  `CacheStats`** — the derived analyses of one function, each computed at most
  once and only if asked for, and a per-method cache of those that outlives the
  visit which built it. The handle borrows the function and hands that borrow
  back through `ir()`, so pairing two functions' results by mistake fails to
  compile; the map lock is released before a caller touches an entry, so an
  analysis never runs under it.

- **Smaller additions.** `SsaOp::first_use` (the first operand without the `Vec`
  that `uses()` allocates); `PointsTo::empty`, a `const` empty relation a
  consumer can borrow; `SsaEditor::replace_phi_predecessor_group_for_result`, the
  origin-keyed rewrite keyed by the phi's result instead;
  `SsaFunction::demote_runtime_entry_phis`, which removes phis from blocks no
  terminator transfers to; `SsaCfg::predecessor_blocks` and
  `to_predecessor_sets`; `SsaBlock::control_terminator`, the crate's single
  definition of "the terminator"; and `MemorySsa::classify_memory_operation`,
  now public, since "does this touch memory, and where" is a question a consumer
  can have without wanting the versioning built on top of it.

- **A payload on every many-to-one system kind** — `CacheMaintenanceOp` (13
  instructions), `TlbMaintenanceOp` (13), `BarrierOp` (3), `HypervisorOp` (32),
  `HardwareEngineOp` (12), `InterruptReturnOp` (13), `BreakpointOp` (17),
  `TrapOp` (16) and `SysRegOp` (72), carried by the matching `SystemOpKind`
  variants, by the new `SystemOpKind::ControlRegister`, and by `SsaOp::Break`.
  Five of these classes had been rendering a *representative* member's mnemonic
  for the whole class, so `clflush` read as `invd` and `xend` as `serialize` —
  worse than a generic label, because it looks correct. `SsaOp::Break` and
  `SystemOpKind::Trap` now distinguish `ud2` from a breakpoint trap and
  `int 0x80` from `int 0x2d`, and `SsaBlockBuilder::break_` takes the
  `BreakpointOp` as a required argument.

- **`MachineStateOp` and `SystemOpKind::MachineState`**, replacing
  `SystemOpKind::Privileged` — one variant per operation across 138 instructions,
  from descriptor tables and port I/O through the CET shadow stack and key locker
  to the ARM, AArch64 and MIPS machine-state operations. `mnemonic`, `kind_str`,
  `effects` and `writes_destination_operand` are all per-operation, and there is
  no catch-all arm to fall into.

- **Eleven `FlagAdjustKind` variants for the x86 flag writers** — the clear/set
  pairs for carry, direction, alignment check and interrupt, plus
  `LoadStatusFromReg` (`sahf`), `StoreStatusToReg` (`lahf`) and
  `SetRegFromCarry` (`salc`) — with `FlagAdjustResult` and
  `FlagAdjustKind::{result, output_arity, effects, effects_for_outputs,
  kind_str, mnemonic}`. `flags_written` names the bits each writes and
  `defines_register` marks the two whose result is a register, so a caller can
  tell whether a pending comparison survives without knowing the architecture.
  `SsaBlockBuilder::flag_adjust` and `flag_adjust_state` are the first way to
  construct one inside this crate, which is why every defect in it was latent.

- **`FlagsMask::DIRECTION`, `INTERRUPT` and `ALIGN_CHECK`, with `NAMED`, `ALL`,
  `is_valid`, `undefined_bits`, `from_bits_checked`, `intersects` and the
  matching `NativeFlagBit` variants.** The control flags sit deliberately outside
  `FlagsMask::x86_status`: no comparison produces them and no condition code
  reads them. `NAMED` is the one table in bit order, `ALL` a `const` fold of it,
  `is_valid` its complement, and `Display` a loop over it, so a bit added to the
  table becomes valid, printable and checkable in one edit.

- **`SsaVerifier::check_flag_masks`, at `VerifyLevel::Standard`** — reports
  `InvalidNativeOperation` for an `SsaOp::ReadFlags` naming a bit no flag
  defines, and for an `SsaOp::FlagAdjust` whose output count differs from
  `FlagAdjustKind::output_arity`. `SsaBlockBuilder::read_flags` refuses the same
  mask at construction; the verifier's copy is for persisted blobs.

- **Identity for the remaining native kinds** — eight `SysRegOp` variants with
  `effects` (model-specific, extended-control, control and debug registers, read
  and written); `SystemOpKind::Hint(HintOp)`, the carrier that makes `endbr64`
  distinguishable from padding, whose effects keep a lifted `nop` non-removable
  without making it a memory barrier; `SystemTransactionKind::ResumeLoadTracking`
  and `SuspendLoadTracking` with `{mnemonic, kind_str, effects}`;
  `PacKind::kind_str` and `mnemonic`, keeping the sign/authenticate pair apart;
  and eight `FpuControlKind` variants for the SSE control word and the `xsave`
  family.

- **`ir::ops::table`, `OpKindTable` and `OpKindIter`** — the contract every
  operation-kind enum shares: a pinned `COUNT`, an index-to-variant `from_index`,
  an injective `kind_str`, an optional `mnemonic`, and an `all` iterator. Sixteen
  enums implement it, and one registry drives the count, spelling and cross-table
  injectivity checks for all of them at once. `SystemOpKind::identities` and
  `ComputeKind::identities` extend that to the two payload-carrying taxonomies,
  built family by family from an exhaustive `family()` match because Rust cannot
  count the variants of an enum with payloads.

### Changed

- **Breaking.** **`Target` gains a required `SymbolRef` associated type and
  `ConstValue` gains a `Symbol` variant**, so implementors and exhaustive matches
  both need updating; a host with no notion of a symbol can point `SymbolRef` at
  any `Clone + Eq + Hash + Debug` type. `const_value_i64` reads a symbol's
  address, so it does not vanish from address normalisation, `AliasKey` and the
  range analyses, and Display and `serde` bounds across the IR data model follow
  the new variant.

- **Breaking.** **`SsaOp::defs` returns an owning `SsaDefs`, which loses its
  lifetime parameter**; `SsaDefs::new` goes with the change. The iterator is the
  `Def`- and `FlagsDef`-role operands of `SsaOp::visit_operands`, so it cannot
  disagree with `SsaOp::dest`, `SsaOp::flags_dest` or `SsaOp::replace_def`, and
  it holds two definitions without allocating.

- **The operand policy of the forty-five boxed-payload operations is written
  once** — `outputs` are definitions, `inputs` are uses, definitions come first,
  and the stack effect is `(inputs.len(), outputs.len())`, in one blanket
  implementation no payload type can override. A payload struct that grows a
  field no longer compiles until someone decides whether it is an operand.

- **Breaking.** **`Display` for seventeen operations separates operands with `, `
  instead of a space** — the vector operations with boxed payloads plus
  `FpTranscendental` and `FpuControl`. Two spellings of one list were maintained
  across forty-five render arms; the rendering is now one helper.
  `VectorSegmentLoad` renders its destinations the way every other operation
  does rather than `Debug`-formatting a `Vec`.

- **Breaking.** **`analysis::structure`'s caller-supplied block sets are
  `BlockSet` rather than `BitSet`, and `StructureOptions` gained a required
  `max_depth`** — every struct literal must be rewritten. `StructureOptions` also
  gains `Debug`, `Clone`, `PartialEq` and `Eq`.

- **Two shapes recover one level flatter.** `Structured::labels` is now exactly
  the set of `Region::Goto` targets, since a block placed at top level because no
  region owned it is reached by falling into it rather than by jumping to it. And
  a conditional whose first arm cannot fall through returns the other arm as the
  resume point instead of nesting it as an `else`, so `if (c) return; ..` stays
  one level deep however often it repeats — which keeps the early-return spine,
  ordinary in decompiled input, off the depth bound.

- **The conversion-chain collapse moved out of `passes::algebraic` into
  `analysis::convert`**, whose contract permits the def-use walk that
  `analysis::algebraic`'s O(1)-per-operation contract does not. The pass now
  decides only how to spell the answer, its module header states the bound as
  O(n + C·d), and it no longer allocates per instruction: the operand-type
  binding it used to build and discard is gone, the remaining lookup going
  through `SsaOp::first_use`.

- **Breaking.** **`MockType` gains `MockType::I8` and `MockType::I16`**, which
  make the conversion rule's (width, signedness) space a space rather than a
  pair, and give `analysis::algebraic`'s 8- and 16-bit constant widths their
  first test coverage. Exhaustive matches on this fixture enum need two arms.

- **Breaking.** **`VerifierError::PhiInEntryBlock` becomes
  `PhiWithoutPredecessors`.** A phi operand names the terminator edge it arrived
  along, so the rule is about having no incoming edges rather than about being
  block 0 — a handler entry the runtime dispatches to has none either, and a
  zero-operand phi there, previously reported nowhere, is now an error. The
  rebuild runs `demote_runtime_entry_phis` before each trivial-phi elimination,
  so the crate cannot emit IR its own verifier rejects.

  `SsaRebuilder` also roots dominance at the exception-aware view, replacing the
  per-handler dominator computation merged in afterwards: N root-local trees give
  contradictory `dominates` answers for a block reachable from two of them, and
  each cost a whole-CFG pass. Entering a handler now kills the groups the
  protected region may have reassigned, since the runtime dispatches from
  part-way through the region and those groups have no single reaching value.

- **Breaking.** **`SsaFunction::block_predecessors`, `block_successors`,
  `compute_predecessors` and `block_has_successor` are removed**;
  `analysis::SsaCfg` is the only place the crate derives a CFG relation. Two
  relations answered "which blocks flow into this one" under near-identical names
  on two types, and disagreed in three ways: the removed ones spliced in an
  exception edge, deduplicated inconsistently, and dropped self-loops.
  `SsaCfg::block_predecessors` is the exact inverse of `block_successors`,
  multiplicity included. `LoopInfo::preheader` therefore counts distinct
  predecessor blocks rather than edges, so a header entered twice from one block
  by `Branch c, header, header` still has a preheader, and `passes::blockmerge`
  refuses to inline an entry trampoline across an exception-region boundary
  explicitly, where that refusal was an accident of the spliced edge.

- **Breaking.** **`SsaBlock::terminator` and `terminator_op` are renamed to
  `last_instruction` and `last_op`.** Both were the block's last instruction
  unconditionally, while all 35 call sites were asking a control-flow question,
  which `SsaCfg::from_ssa` answered a third way. `control_terminator` is now that
  question, and every successor, target and editor path routes through it. Two
  behaviours follow, both on IR the verifier already reports as
  `TerminatorNotLast`: a block whose last instruction is not a terminator
  contributes no CFG edges, and `set_target` **appends** a `Jump` rather than
  writing over the last instruction, which previously deleted a computation the
  block still needed.

- **Breaking.** **`analysis::SsaCfg` no longer draws an edge from a protected
  region to its handler**, so a block's successor count is its terminator's
  arity. The synthetic edge was the wrong *edge* — control reaches a handler from
  wherever it threw, not from the region's first block — it fired for 1 of 29
  clauses on the corpus's one exception-carrying fixture, and it broke arity for
  every other consumer. A consumer needing handler reachability reads
  `SsaFunction::exception_handlers`.

- **Breaking.** **An exception clause is three `Option<BlockRange>` parts, not
  five loose block indices.** Every struct literal and field read of
  `SsaExceptionHandler` changes, and its `Debug`, `Clone` and serde forms change
  shape, so a host persisting the serde form must migrate. `filter_offset` is
  gated on `SsaExceptionHandler::kind`, so `class_token_or_filter` is read as an
  IL offset only where it is one, and `remap_block_indices` maps each part by its
  members — exact for the order-preserving remap compaction performs, and a
  superset otherwise, which is the conservative direction.

- **Breaking.** **`Target::handler_kind` replaces `Target::is_filter_handler`.**
  Every host `impl Target` fails to compile until it supplies the new one; in
  this crate only `MockTarget`, which reads its `u32` flags as 0 = catch,
  1 = filter, 2 = finally, 3 = fault. `filter_offset` therefore answers `Some`
  for flags 1, where it previously answered `None` for everything.

- **Everything that reads a clause now reads it through the same accessors.**
  `passes::blockmerge` fences every part of every clause via
  `SsaExceptionHandler::parts`, so a merge at a filter's exclusive end no longer
  happens, and `ExceptionBlocks` and `FunctionRoots` go through `BlockRange` too.
  `SsaVerifier::verify` can consequently report errors for functions it
  previously accepted, at every level including `VerifyLevel::Quick`; passes are
  unaffected, and `rebuild_ssa` filters the new error out, since reconstruction
  never rewrites a clause's ranges.

- **The edit-scope boundary now leaves both the def sites and the use index
  exact, for every scope.** `repair_ssa` and `rebuild_ssa` rewrite instructions
  and variables without restoring the index, so an edit above `UsesOnly` finished
  with an index that no longer described the function — and that index is the
  candidate list `replace_uses_checked_indexed` iterates, so the next pass's
  rewrites were being chosen from it.

- **Breaking.** **Post-dominance seeds one virtual-exit edge per exit-less
  region, not per stranded node**, and `compute_post_dominators` /
  `control_dependences` return the two new result types. The old seeding
  established a *forward* property — "this node reaches no sink" — with a walk
  that only moved backwards. On `0 -> 1 -> 2 -> 1` the virtual exit gained edges
  to both 0 and 1, so node 1 stopped post-dominating node 0 although it is node
  0's only successor, and node 1 was reported control dependent on a block with
  one unconditional successor.

  Seeding is now one edge per terminal strongly connected component, which
  unifies the two cases: a terminal singleton with no successors is exactly a
  real sink. Graphs where every node reaches a sink are bit-identical, seed order
  included, and the O(V²) rescan becomes O(V + E). `control_dependences` returns
  `node_count` rows rather than `node_count + 1`; the extra row was provably
  always empty.

- **Breaking.** **`MemorySsa::build` drops its `cfg` parameter, and Memory SSA is
  rooted at handler and filter entries.** Handler blocks were unreachable in the
  terminator-only CFG, so phi placement and renaming skipped them: a store inside
  a handler reached the rejoin block with no merge point, and `passes::memory`
  forwarded the try-path store into a load there — the wrong value on every path
  that actually threw. A depth-1 handler still got its phi by accident, because
  `compute_dominance_frontiers` inserts before it notices the unreachable-node
  sentinel, which is why the suite was green.

  `MemoryDefSite::ExceptionEntry` is the other half of the rule: the runtime
  dispatches into a handler from part-way through the protected region, so a
  store that region makes is not known to have run. Its locations get a version
  standing for "unknown", which `passes::memory` declines by construction. A
  store that completed *before* the region is unaffected.

- **Breaking.** **`SsaOp::FlagAdjust` effects and identity are derived from
  `FlagAdjustKind`.** The operation sat in a shared "pure value-producing
  compute" arm that never inspected the kind — right while the enum held only
  NZCV producers, but `cld`, `std`, `clac` and `stac` write control flags no SSA
  operand models, so `removable_when_unused()` said yes and dead-code elimination
  silently removed the SMAP window around a user-pointer access. Purity now
  follows `FlagAdjustKind::result`, and `opcode_name` and `Display` read the kind
  instead of emitting a constant `"flags.adjust"`; the spellings come from the
  private table `coarse_token` already used, so coarse tokens are byte-identical.

- **`FlagAdjustKind::flags_written` reports OVERFLOW for `setf8`/`setf16`.**
  AArch64 `setf8`/`setf16` write N, Z **and** V and leave C alone; V is the point
  of the instruction. Reporting only N and Z told a caller a stale V was still
  live, which folds a `b.vs` against a flag `setf8` had overwritten.

- **Breaking.** **The system and compute taxonomies have one representation per
  instruction, so several `kind_str` fingerprints move** — `rdmsr` is
  `"system.sysreg.msr.read"`, `rdtscp` is `"system.timestamp.aux"`, every
  `PacKind` reads `"compute.pac.*"`, and `HintOp` is spelled `"system.hint.*"`.
  These keys are checked injective over the union of every table plus both
  composites' `identities()`. `SsaFeatureToken` and `similarity_class`
  fingerprints move with them, so a host caching them should invalidate;
  `coarse_token` is unaffected.

- **Breaking.** **Two effect and mnemonic corrections in the native tables.**
  RISC-V `csrrw`, `csrrs` and `csrrc` are `ReadWrite`, each reading a CSR and
  writing it back in one instruction, while ARM `ldc` is `Read`; and
  `MachineStateOp::mnemonic` for `InterruptDisable` and `InterruptEnable` is
  `"di"` and `"ei"`, the MIPS spellings those variants name.

- **`FlagsMask::Display` renders undefined bits as `?0x…`.** `from_bits` stays
  unmasked so a persisted mask round-trips exactly; what makes that safe is that
  a bit outside `FlagsMask::NAMED` is now visible everywhere, rather than
  printing `none` or a truncated but plausible-looking list.

- **`SsaOp::coarse_token` no longer normalizes its `kind_str` inputs.** Five arms
  lowercased and trimmed a crate literal into a fresh `String`, which cannot
  change the answer: every registered table's keys are pinned equal to their own
  normalized form. The host-supplied mnemonic arm still normalizes, because that
  string comes from a decoder.

### Removed

- **`SsaOp::set_dest`.** Nothing called it — not this crate, its tests, its
  benches or its fuzz targets — and SSA renaming rewrites definitions through
  `SsaOp::replace_def`, which handles secondary outputs too. It also disagreed
  with the operand walk about `Call`, `CallVirt` and `CallIndirect`, where it
  *created* a destination the walk reports nothing for.

- **`ProtectedHandler::filter_entry` and `ProtectedHandler::filter_blocks`.** Two
  fields independent of each other and of the handler's kind, so a filter handler
  with no expression recovered silently as one without a filter and a `finally`
  could carry one. `ProtectedHandlerKind::Filter` carries a `HandlerFilter`
  instead.

- **`SsaOp::NativeIntrinsic`, `NativeIntrinsicData`, `NativeIntrinsicId` and
  `SsaOpClass::NativeIntrinsic`.** Nothing constructed the operation, in this
  crate or any front-end; `SystemOp` and `ComputeOp` are its documented
  structured replacement and had already taken every case it covered.

- **`SysRegNamespace`, `SystemOpKind::ReadSysReg` and
  `SystemOpKind::WriteSysReg`.** A namespace names the register *file* an operand
  reaches while the variant beside it names the instruction, so `rdmsr` was
  representable twice and both spellings answered `"system.sysreg.read"`.
  `SystemOpKind::ControlRegister` is the single home, and the architecture a
  variant came from rides `NativeKindedData::metadata`.

- **`BarrierOp::TransactionEnd`, `TransactionTest`, `TransactionAbort`,
  `ResumeLoadTracking` and `SuspendLoadTracking`.** `xend`, `xtest`, `xabort` and
  the load-tracking pair were representable as barriers *and* as
  `SystemTransactionKind`, and the two homes disagreed about whether a commit
  fences, so whether a store could move across `xend` depended on which spelling
  a front-end picked. `BarrierOp` now names `serialize`, `mcommit` and `pcommit`.

- **`NativeClobber`, `NativeStateAccess`, `NativeStateAccessKind`,
  `NativeStateLocation`, `NativeRegister`, and the `clobbers` field on every
  operation payload that carried one.** A clobber is a claim that an operation
  changed state it cannot name, and nothing acted on one: memory SSA, the passes,
  type propagation and the block helpers all read `SsaEffects`, leaving a
  parallel, unconsulted restatement where a wrong claim could sit unchallenged.
  Registers written are `outputs`, flags defined are the flags value, the
  register file a system operation touches is named by its `SystemOpKind`, and
  genuinely unknown semantics declare `SsaEffectKind::Opaque`.
  `SsaBlockBuilder::native_opaque` loses its `clobbers` parameter.

- **`NativeExceptionKind`, `native_is_filter_handler` and
  `Target::is_filter_handler`.** A second copy of the handler taxonomy, variant
  for variant, plus a boolean where the question has four answers — so a catch, a
  finally and a fault could not be told apart and structured recovery could not
  name what a handler does. `HandlerKind` is the one home and
  `Target::handler_kind` the one classifier, required and total.

- **The five loose block-index fields of an exception clause**
  (`try_start_block`, `try_end_block`, `handler_start_block`,
  `handler_end_block`, `filter_start_block`). Five options that only meant
  anything in pairs, with nothing typing the pairing: a start could exist without
  its end, an exclusive end could swallow a neighbouring part, and the filter had
  no end at all.

### Fixed

- **The conversion-chain collapse dereferenced def sites without the guard.** It
  carried its own copy of `get_definition`'s fast path, minus the check that the
  instruction found actually defines the variable asked for, so a def site stale
  by one index handed the walk an unrelated conversion whose widths then decided
  whether to rewrite — silent wrong code, where a refusal would have cost only
  the optimization. Every link now resolves through `recorded_definition`.

  Five other sites had made the same unguarded copy. `try_constant_value` yields
  `None` rather than an unrelated `Const`'s value, so an induction stride becomes
  unknown instead of wrong, and `SsaEvaluator`'s three scan-free lookups answer
  `false`/`None` instead of deriving from a foreign instruction. LICM's back-edge
  taint set and `analysis::loops`' induction update take the *recovering* lookup
  instead, since losing a link there only costs an optimization.

- **A cyclic def-use chain made the collapse walk spin forever.** Malformed IR
  can present exact def sites that lead in a circle, and the walk had no bound
  and no visited set. It now refuses once it has taken more steps than
  `SsaFunction::var_id_bound` — exact by pigeonhole, with no allocation and no
  arbitrary constant.

- **GVN forwarded a redundant computation to a leader that did not reach it.**
  Blocks are visited in index order, which says nothing about dominance: two
  sibling branches both computing `x + 1` made the lower-numbered one the leader
  and rewrote the other's uses to a definition that does not dominate them. A
  leader is now only a leader where it is available, and availability is
  dominance. On the MIPS fixture this was `DominanceViolation { def_block: 1,
  use_block: 7 }` — the last remaining reason a debug and a release build lifted
  the same bytes differently.

- **Loop canonicalization rewrote every header phi to one value.**
  `insert_preheader` and `unify_latches` keyed their phi maps on
  `VariableOrigin`, but `VariableOrigin::Phi` is a unit variant, so every phi
  merging earlier phis — the ordinary shape in lifted native code — shares it.
  One inserted phi stood for all of them, a dominance violation for all but one.
  Both now key on the header phi's result.

- **`LoopInfo::contains` panicked on a block index the loop's set cannot
  represent.** Callers reach it with indices from def sites, phi operands and CFG
  metadata, all of which describe the function as it was. A `BitSet` bounds
  assert is a panic: over 21,600 binaries it fired 8 times from LICM alone, each
  unwinding into `guard_pass`, which rolled the pass back and left the function
  unoptimised with nothing upstream the wiser. Such an index now answers "not in
  this loop". Dead-code elimination had the same defect from the other metadata
  source — `handler_start_block` and `filter_start_block` can name a block the
  current function does not have — and now reads them with the checked accessors
  the verifier already used.

- **A load could resolve to a different value on consecutive runs.** More than
  one stored location can must-alias a query and they can hold different values;
  returning whichever the map yielded first made the answer a function of the
  hasher's per-instance seed, so the same bytes lifted to different SSA. The most
  recently stored version now wins, with the lower variable id breaking a tie.

- **`analysis::structure` read caller-supplied block sets with the panicking
  `BitSet::contains`.** `structure_with` is documented "Always succeeds", yet a
  set narrower than the graph aborted on the first two-armed branch and a
  protected region sized to itself aborted in `pick_follow`. Sets are now
  renormalised once at the boundary and counted in
  `StructureMetrics::resized_inputs`; an `entry` outside the graph recovers as
  `Region::Empty`, and an out-of-range successor id contributes no edge.

- **Structuring recursed once per chained conditional, on an unbounded stack.**
  The depth was Θ(conditionals), not Θ(source nesting), and the tree was as deep
  as the walk, so the caller's derived `Drop`, `Clone` and `PartialEq` recursed
  just as far. `StructureOptions::max_depth` now bounds both.

- **Protected regions were lost or printed unprotected.** Two regions sharing a
  try start kept only one, because the entry-to-region map held a single index
  per block — so the ordinary `try { try {} catch {} } finally {}` encoding
  dropped a clause; the map now holds every region starting at a block, outermost
  last. And a region whose entry is unreachable — nothing says a `try` is
  reached, since the runtime takes the edge into the handler — left its handler
  blocks to the orphan pass, which emitted them outside any `Region::Try`; such
  regions are now drained and opened at top level.

- **Three defects in how a conditional's blocks are placed.** A join folded into
  the predicate stayed the conditional's resume point, becoming a `Region::Goto`
  to a block with no statement position — on exactly the shape merging exists to
  remove. A handler's joins were bounded by a loop the handler is not inside,
  since the runtime enters a handler where the enclosing loop's back edge does
  not, so a handler-internal diamond degraded to an over-wide arm plus a goto;
  entering a handler or filter now raises a loop-frame barrier, while `break` and
  `continue` still name the enclosing loop. And a construction site that dropped
  a conditional dropped the blocks its condition spans, which nothing else
  places; both such sites now emit them through one shared helper.

- **An exception clause could be mapped half-way, by specification.**
  `remap_block_indices` was documented to advance a removed exclusive end to the
  next surviving block anywhere in the function, and to clear a start without
  touching its end — so a region that began somewhere and ended nowhere, and a
  handler whose end had crossed into the filter block, were states the IR could
  hold and no check could refuse. The range type deletes both.

- **Nothing converted an exception table into `ProtectedRegion`s.** The recovery
  advertised protected regions with their handlers while a host holding an
  `SsaFunction` had no in-crate path to one: the IR carried block ranges where
  the recovery wanted sets, the filter had no stored extent, and `Target` could
  not tell a finally from a catch. `analysis::recovery::structure_ssa` is that
  path, and it reports the clauses it could not convert instead of dropping them.

- **`structure` and `structure_protected` declared every block condition-only
  without saying so.** Both are generic over the graph traits `SsaCfg`
  implements, so the obvious call on the crate's own CFG type asserted that every
  SSA block — stores, calls, everything — computes nothing but its own branch
  condition. The claim is now in both `///` blocks, and `condition_only_blocks`
  answers it from block contents. A tree recovered through `structure_ssa`
  therefore has fewer merged conditions and more gotos, which is the sound
  direction.

### Security

- **`arrayref` 0.3.10 and later are banned in `deny.toml`.** The release was
  published as part of an ongoing attack on crates.io. This crate's tree does not
  contain it; the ban is prophylactic, so a future dependency cannot pull it in
  silently. The range is open-ended because the publishing account itself is
  compromised, which makes every later version equally untrusted until ownership
  is confirmed recovered.

## [0.5.0] - 2026-08-04

Correctness of the SSA rebuild and the phi transforms. On a 125 MB x86-64
reference binary (50,000 functions), pass rollbacks went from 6,094 to **0** and
verifier-reported undefined uses from ~28,960 to **0**; no function is now
floored by normalization. Output is 6% leaner (10,089,320 to 9,490,820 lowered
rows) while landing 4% more rewrites (94,257 to 98,467), because rejected work is
no longer discarded wholesale.

Every defect below shares a shape: a value substitution or a definition record
that was *stated* rather than *established* — a chain composed without resolving
to a surviving target, a label asserting a definition that did not exist, or a
repair applied on one path and not its sibling.

### Changed

- **A pass group no longer trusts a pass's own "changed" return.**
  `SsaFunction` now tracks whether a checked edit mutated it, and
  `PassTransaction::run_group` treats a mutated function as changed regardless of
  what the pass reported. The precondition that made this necessary — a pass
  editing under `SsaRollbackPolicy::Never` must report the change when the edit
  fails, or the group skips both verification and rollback — was unstated,
  unenforced, and had been got wrong by eight passes. It is now structural
  rather than a convention: a pass that mutates and then claims otherwise is
  still verified and still rolled back.

### Added

- **`SsaFunction::take_edit_dirty`** — reports whether a checked edit mutated the
  function, clearing the flag. For consumers driving their own pass groups, who
  need the same guarantee `PassTransaction` now provides.

- **`SsaFunction::refresh_def_sites` is public.** It recomputes every variable's
  definition site from the IR as it actually stands. Front ends that build SSA
  incrementally cannot always know an instruction's final index while lowering,
  and a stale index that runs past the end of its block fails index-bounds
  verification. This is the routine `repair_ssa` already used internally.

### Fixed

- **Trivial-phi substitution chains in rebuild mode resolved to a deleted
  value.** The back-to-front composition introduced in 0.4.1 resolves each source
  only through entries already inserted, so for `[(p2, x), (p1, p2)]` it records
  `p1 -> p2` while `p2` is retired in the same round. Uses of `p1` were rewritten
  to a phi deleted moments later. Each chain is now walked to a target that is
  not itself being replaced, with a cycle guard — the repair path had always done
  this. The 0.4.1 change remains correct as a performance fix; only its
  composition order was wrong.

- **Self-referential phis were removed without rewriting their uses.** A phi
  recorded as `(result, result)` has no other value to substitute and is
  deliberately absent from the substitution map, but rebuild mode retired it
  anyway, stranding every use on a variable nothing defined. It is now retired
  only once nothing reads it, which is the condition the repair path already
  applied.

- **`pre_clean_unreachable` inlined trivial phis without resolving chains.** Its
  replacement map was applied entry by entry, so a phi inlined to a value that
  was itself another phi being inlined in the same pass left the intermediate in
  place — a use naming a phi the same loop had already removed. Because the map
  is a `BTreeMap`, whether the chain resolved depended on key order, which made
  the failure rare and order-dependent rather than reliably reproducible.

- **`repair_ssa` did not repair same-block future uses.** An instruction-scope
  edit can leave a use naming a definition later in its own block. The rebuild
  path repairs exactly this; the repair path did not, so the transactional guard
  rejected the result as `IntraBlockCycle` and discarded the pass's work.

- **Entry replacements were labelled as phi-defined.** The stand-in
  `repair_same_block_future_uses` fabricates for a use with no prior definition
  is undefined by construction — it represents the value incoming to the
  function. Copying the source variable's origin labelled it `Phi`, asserting
  that a phi defined it. It now carries `EntryLiveIn`, whose contract is exactly
  that the caller supplies it.

- **`clear_all_phis` discarded phis the rebuild could not reconstruct.** A phi
  whose result belongs to no rename group, or to a group with no recorded
  definition, cannot be re-placed by `place_phis`; clearing it destroyed its
  definition outright. Such phis are now retained.

- **Phi results and phi-operand uses were invisible to the rebuild's def/use
  collection.** `collect_defs` did not record a phi result as a definition of its
  group, so a group whose only definition reaching a region was a phi contributed
  no block there and no phi was re-placed where one was still needed.
  `collect_uses_and_liveness` did not attribute a phi operand to the predecessor
  it flows from, understating liveness on that edge.

- **`expand_phi_predecessor` could leave two operands on one edge.** A
  predecessor reaching a block both directly and through the block being bypassed
  had an operand added for an edge it already named. Duplicate operands have no
  defined meaning and consumers disagree — `PhiNode::operand_from` returns the
  first, SCCP meets them all and yields Bottom. The replaced edge is now dropped
  and operands are added only for predecessors the phi does not already name.

- **Seven passes reported "unchanged" after applying edits.** `SsaEditOptions::new()`
  defaults to `SsaRollbackPolicy::Never`, so a failed edit or boundary repair
  leaves the edits applied. Returning `false`/`0` then tells the pass-group
  transaction nothing changed, and its `Unchanged` arm returns *without verifying
  and without rolling back* — so damaged IR was kept, and kept unchecked. On a
  125 MB reference binary seven edit sessions failed this way and produced zero
  rollbacks, meaning seven functions carried mutated, unverified IR.
  `algebraic`, `ranges`, `reassociate`, `strength` and `threading` now report the
  change so the transaction verifies and rolls back, as do `controlflow` and
  `blockmerge` (below).

  The mixed policy across passes is deliberate and unchanged: passes that verify
  inside their own edit session (`copying`, `predicates`, `licm`) need
  `OnFailure` for that verification to mean anything, while passes that delegate
  to the transaction use `Never` to avoid a second snapshot — the transaction
  already clones once per pass, where `OnFailure` clones per edit session. What
  was missing was the unstated precondition that a `Never` pass must report the
  change on failure.

- **`gvn` reported "unchanged" only correctly in debug builds.** Its rollback
  policy is `OnFailure` under `debug_assertions` and `Never` otherwise, so
  `return 0` on failure was true under test and false in release — the one
  configuration where the damaged IR would ship. The return now depends on which
  policy actually ran.

- **`controlflow` and `blockmerge` reported "unchanged" after applying edits.**
  Under `SsaRollbackPolicy::Never` a failed boundary repair leaves the edits in
  place. Reporting zero told the caller nothing had changed, and a pass-group
  transaction treats "unchanged" as "nothing to verify" — so damaged IR was kept,
  and kept unchecked. Both now report the applied edits, which lets the
  transaction verify the function and roll it back.

### Ownership

- Recorded ATRAPS LLC as copyright holder and added a `NOTICE` file. The Apache-2.0
  appendix was never filled in — it still carried the literal
  `[yyyy] [name of copyright owner]` placeholder, so nothing in this repo stated
  who owned it.
- Added a `repository` field. The manifest declared `documentation` but no
  repository, so crates.io showed no source link for any published version.
- Dropped the deprecated `authors` field.
- Publishing now uses crates.io trusted publishing instead of a stored registry token.

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
