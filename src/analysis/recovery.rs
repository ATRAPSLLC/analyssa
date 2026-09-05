//! The bridge from an [`SsaFunction`] to structured control flow.
//!
//! [`mod@crate::analysis::structure`] recovers a statement tree from a graph and is
//! deliberately ignorant of everything else: it is generic over
//! `GraphBase + Successors + Predecessors`, knows nothing of [`Target`], and is
//! testable against a bare `DirectedGraph`. That is what keeps it honest, and it
//! is also why it cannot be handed a function. Two things stand between the two:
//!
//! - **Protected regions.** The IR stores a clause as half-open block ranges,
//!   because that is what regenerates an IL exception table. The structurer
//!   consumes membership as sets. [`ProtectedRegions`] is the conversion, and
//!   the only one.
//! - **Which blocks are condition-only.** The structurer must be told which
//!   blocks compute nothing but their own branch condition, because folding a
//!   condition *moves* a block and rendering `while (c)` runs its header once
//!   per iteration. A caller holding a bare graph has no instruction-level view
//!   and so declares every block; a caller holding an `SsaFunction` does have
//!   one, and [`condition_only_blocks`] answers from the blocks' contents.
//!
//! # Losses are returned, not dropped
//!
//! A clause can fail to become a region — it names blocks the function does not
//! have, or its protected range half-overlaps another's. [`structure_ssa`]
//! returns [`Recovered`], which carries the tree *and*
//! [`ProtectedRegions::rejected`]. A bare `Structured` would have no channel for
//! that, and the one path a host is told to use would be the one that loses
//! clauses silently — the exact degradation this module exists to make visible.
//!
//! # What a recovered tree is not
//!
//! The graph is built by [`SsaCfg::from_ssa`], which draws only the edges
//! terminators take. Handler blocks are therefore unreachable from the entry:
//! the edge into a handler is taken by the runtime and appears in no
//! instruction. They are placed inside their `Region::Try` by the structurer, so
//! the tree still accounts for them. A graph that also carried exception edges
//! would make a block ending in an unconditional jump look like a conditional
//! branch, and it would be recovered as one.

use crate::{
    BitSet,
    analysis::{
        cfg::SsaCfg,
        defuse::DefUseIndex,
        structure::{
            BlockSet, DEFAULT_MAX_DEPTH, HandlerFilter, ProtectedHandler, ProtectedHandlerKind,
            ProtectedRegion, StructureOptions, Structured, structure_with,
        },
    },
    graph::NodeId,
    ir::{
        SsaFunction,
        exception::{BlockRange, ExceptionTableError, LaidOutHandler},
        ops::SsaOp,
    },
    target::Target,
};

/// Why a clause of the exception table did not become a protected region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum ClauseRejection {
    /// The clause cannot be laid out over the function's blocks at all.
    #[error("{0}")]
    Malformed(ExceptionTableError),

    /// The clause's protected range overlaps another accepted clause's without
    /// either containing the other.
    ///
    /// Nesting is accepted — `try { try {} catch {} } finally {}` is the
    /// ordinary ECMA-335 and JVM encoding, and both regions open. A *partial*
    /// overlap is a shape no structured form has: neither region can open
    /// inside the other, so one of them would have to be torn.
    #[error("its protected range partially overlaps that of clause {other}")]
    PartiallyOverlapping {
        /// The already-accepted clause it collides with.
        other: usize,
    },
}

/// One clause of the exception table that did not become a protected region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RejectedClause {
    /// Index of the clause in
    /// [`SsaFunction::exception_handlers`](crate::ir::SsaFunction::exception_handlers).
    pub clause: usize,

    /// Why it was rejected.
    pub reason: ClauseRejection,
}

/// The protected regions of a function, and the clauses that did not become
/// one.
///
/// Built by [`ProtectedRegions::from_function`], which is the crate's only
/// conversion from an exception table into the structurer's input.
///
/// # Grouping
///
/// Clauses are grouped by **identical protected range**: `try/catch(A)/catch(B)`
/// is two table rows sharing one try range, and it becomes one region with two
/// handlers. One region per row would give the two an identical `entry`, and
/// `structure_with` keys its protected map by entry — so the second catch would
/// be dropped and swept up as an orphan.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProtectedRegions {
    /// The regions, in the declaration order of the clause that opened each.
    regions: Vec<ProtectedRegion>,

    /// The clauses that did not become one, in table order.
    rejected: Vec<RejectedClause>,
}

impl ProtectedRegions {
    /// Reads `ssa`'s exception table into protected regions.
    ///
    /// Every clause takes one of four routes:
    ///
    /// - it lays out and opens a new region;
    /// - it lays out and joins an existing region with the same protected
    ///   range;
    /// - it maps no block of this function — a funclet handler — and is skipped
    ///   silently, which is 28 of the 29 clauses on the corpus's one
    ///   exception-carrying fixture;
    /// - it is rejected, and appears in [`rejected`](Self::rejected).
    #[must_use]
    pub fn from_function<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let block_count = ssa.block_count();
        let mut regions: Vec<ProtectedRegion> = Vec::new();
        // The protected range behind each accepted region and the clause that
        // opened it, parallel to `regions`.
        let mut opened: Vec<(BlockRange, usize)> = Vec::new();
        let mut rejected: Vec<RejectedClause> = Vec::new();

        for (clause, handler) in ssa.exception_handlers().iter().enumerate() {
            let layout = match handler.layout(block_count) {
                Ok(Some(layout)) => layout,
                Ok(None) => continue,
                Err(reason) => {
                    rejected.push(RejectedClause {
                        clause,
                        reason: ClauseRejection::Malformed(reason),
                    });
                    continue;
                }
            };

            let collision = opened.iter().find(|(range, _)| {
                range.overlaps(&layout.protected)
                    && !range.contains_range(&layout.protected)
                    && !layout.protected.contains_range(range)
            });
            if let Some((_, other)) = collision {
                rejected.push(RejectedClause {
                    clause,
                    reason: ClauseRejection::PartiallyOverlapping { other: *other },
                });
                continue;
            }

            let recovered = ProtectedHandler {
                kind: match layout.kind {
                    LaidOutHandler::Catch => ProtectedHandlerKind::Catch,
                    LaidOutHandler::Filter(filter) => ProtectedHandlerKind::Filter(HandlerFilter {
                        entry: NodeId::new(filter.start()),
                        blocks: blocks_of(filter, block_count),
                    }),
                    LaidOutHandler::Finally => ProtectedHandlerKind::Finally,
                    LaidOutHandler::Fault => ProtectedHandlerKind::Fault,
                },
                entry: NodeId::new(layout.handler.start()),
                blocks: blocks_of(layout.handler, block_count),
            };

            match opened
                .iter()
                .position(|(range, _)| *range == layout.protected)
            {
                Some(index) => {
                    if let Some(region) = regions.get_mut(index) {
                        region.handlers.push(recovered);
                    }
                }
                None => {
                    regions.push(ProtectedRegion {
                        entry: NodeId::new(layout.protected.start()),
                        blocks: blocks_of(layout.protected, block_count),
                        handlers: vec![recovered],
                    });
                    opened.push((layout.protected, clause));
                }
            }
        }

        Self { regions, rejected }
    }

    /// The recovered regions, ready for
    /// [`StructureOptions::regions`](crate::analysis::StructureOptions::regions).
    #[must_use]
    pub fn regions(&self) -> &[ProtectedRegion] {
        &self.regions
    }

    /// The clauses that did not become a region, and why.
    ///
    /// Empty for a well-formed table. A non-empty answer is the exception table
    /// saying something the recovered tree does not contain, which a host has to
    /// see: a `catch` that is missing from the tree and missing from this list
    /// is indistinguishable from one that never existed.
    #[must_use]
    pub fn rejected(&self) -> &[RejectedClause] {
        &self.rejected
    }

    /// Whether the function has no protected region at all.
    ///
    /// Says nothing about [`rejected`](Self::rejected): a table whose every
    /// clause was rejected is empty *and* lossy.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

/// Expands one clause range into a block set over `0..block_count`.
///
/// The single expansion point: [`BlockRange::to_bitset`] is the crate's only
/// interval-to-set conversion, and this is its only caller, so no set narrower
/// than the graph can reach `structure_with`.
///
/// `to_bitset` declines a range that does not fit, which
/// [`layout`](crate::ir::SsaExceptionHandler::layout) has already ruled out for
/// every range reaching here. The empty set stands in for that unreachable case
/// because it is the reading that claims least: a region covering no block.
fn blocks_of(range: BlockRange, block_count: usize) -> BlockSet {
    BlockSet::from_bits(
        block_count,
        range
            .to_bitset(block_count)
            .unwrap_or_else(|| BitSet::new(block_count)),
    )
}

/// Returns the blocks of `ssa` that compute nothing but their own branch
/// condition.
///
/// This is the set [`StructureOptions::condition_only`] asks for, answered from
/// the instructions a block holds rather than assumed. A block qualifies when
/// all four hold:
///
/// 1. **it has no phi nodes** — a phi is a statement about where control came
///    from, and moving the block moves that question;
/// 2. **it ends in a branching terminator** — a jump, a two-way branch or a
///    multi-way dispatch. A `return`, a `throw` and a region exit are not
///    conditions and are never folded into one, and `endfinally` / `endfilter`
///    additionally carry unwind semantics a move would lose;
/// 3. **every other instruction is pure** — no store, no call, nothing that can
///    trap. Folding a block's test into another's condition relocates its work,
///    and a block that writes memory cannot be relocated;
/// 4. **nothing it defines is read elsewhere** — a value the block computes for
///    another block is work that must happen whether the condition short-circuits
///    or not.
///
/// The result is a subset of the "every block" set the graph-only entry points
/// declare, so a caller can expect fewer merged conditions, more `Endless` loops
/// with an explicit `break`, and more gotos. That is the sound direction, and it
/// is the first time the question is answered from block contents.
#[must_use]
pub fn condition_only_blocks<T: Target>(ssa: &SsaFunction<T>) -> BlockSet {
    let block_count = ssa.block_count();
    let mut set = BlockSet::new(block_count);
    let defs = DefUseIndex::build(ssa);

    for (index, block) in ssa.iter_blocks() {
        if !block.has_no_phis() {
            continue;
        }
        let instructions = block.instructions();
        let Some(last) = instructions.last() else {
            continue;
        };
        if !is_branching_terminator(last.op()) {
            continue;
        }
        let body_is_pure = instructions
            .iter()
            .all(|instr| instr.op().is_pure() || is_branching_terminator(instr.op()));
        if !body_is_pure {
            continue;
        }
        let escapes = defs.defs_in_block(index).iter().any(|var| {
            defs.uses_of(*var)
                .is_some_and(|uses| uses.iter().any(|use_site| use_site.block != index))
        });
        if escapes {
            continue;
        }
        set.insert(NodeId::new(index));
    }

    set
}

/// Whether `op` transfers control within the function without leaving it.
fn is_branching_terminator<T: Target>(op: &SsaOp<T>) -> bool {
    matches!(
        op,
        SsaOp::Jump { .. }
            | SsaOp::Branch { .. }
            | SsaOp::BranchCmp { .. }
            | SsaOp::BranchFlags { .. }
            | SsaOp::IndirectBranch { .. }
            | SsaOp::Switch { .. }
    )
}

/// A recovered function: the statement tree, and what the exception table said
/// that the tree could not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovered {
    /// The statement tree.
    pub structured: Structured,

    /// The protected regions that went into it, and the clauses that did not.
    pub regions: ProtectedRegions,
}

/// Recovers structured control flow for an SSA function, with its protected
/// regions.
///
/// The end-to-end entry point: one call turns an [`SsaFunction`] into a
/// statement tree in which every clause the table records that *can* be
/// represented appears as a `Region::Try` with its handlers, and every clause
/// that cannot is named in [`ProtectedRegions::rejected`].
///
/// Takes the function rather than a [`SsaCfg`], deliberately. The recovery is
/// only correct on a graph carrying terminator edges alone, and taking a graph
/// would let a caller hand in one built any other way — a precondition with
/// nothing to enforce it.
///
/// Always succeeds. Recovery quality is a count on
/// [`Structured::metrics`](crate::analysis::Structured), and clause loss is a
/// list on the regions; neither is an error.
///
/// # Examples
///
/// ```rust
/// use analyssa::{analysis::recovery::structure_ssa, testing::try_catch_fixture};
///
/// // B0 -> B1 (the protected body) -> B3; B2 is the handler, which no
/// // terminator names.
/// let recovered = structure_ssa(&try_catch_fixture());
///
/// assert_eq!(recovered.regions.regions().len(), 1, "one try region");
/// assert!(recovered.regions.rejected().is_empty(), "and nothing was lost");
/// assert_eq!(
///     recovered.structured.metrics.ifs, 0,
///     "the runtime enters the handler; no terminator branches to it"
/// );
/// ```
#[must_use]
pub fn structure_ssa<T: Target>(ssa: &SsaFunction<T>) -> Recovered {
    let regions = ProtectedRegions::from_function(ssa);
    let condition_only = condition_only_blocks(ssa);
    let cfg = SsaCfg::from_ssa(ssa);

    let structured = structure_with(
        &cfg,
        NodeId::new(0),
        &StructureOptions {
            regions: regions.regions(),
            // A block that computes only its condition can also be *said*
            // inside one, so the weaker claim is admitted for exactly the same
            // blocks. Nothing else is: a block carrying a store is no more
            // expressible in a condition than it is movable into one.
            condition_expressible: condition_only.clone(),
            condition_only,
            max_depth: DEFAULT_MAX_DEPTH,
        },
    );

    Recovered {
        structured,
        regions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            SsaExceptionHandler,
            block::SsaBlock,
            instruction::SsaInstruction,
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType, try_catch_fixture},
    };

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    /// Builds a function whose block `index` holds `ops[index]`.
    fn function(ops: Vec<Vec<SsaOp<MockTarget>>>) -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        for (index, block_ops) in ops.into_iter().enumerate() {
            let mut block = SsaBlock::new(index);
            for op in block_ops {
                block.add_instruction(instr(op));
            }
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        ssa
    }

    fn clause(
        flags: u32,
        protected: Option<BlockRange>,
        handler: Option<BlockRange>,
        filter: Option<BlockRange>,
    ) -> SsaExceptionHandler<MockTarget> {
        SsaExceptionHandler {
            flags,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: protected,
            handler_range: handler,
            filter_range: filter,
        }
    }

    /// `try/catch(A)/catch(B)` is two table rows sharing one protected range.
    ///
    /// One region per row would give both the same `entry`, and
    /// `structure_with` keys its protected map by entry — so the second catch
    /// would be dropped and swept up as an orphan. Grouping by protected range
    /// is what keeps both.
    #[test]
    fn sibling_catches_become_one_region_with_two_handlers() {
        let mut ssa = function(vec![
            vec![SsaOp::Jump { target: 1 }],
            vec![SsaOp::Jump { target: 4 }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
        ]);
        ssa.set_exception_handlers(vec![
            clause(0, BlockRange::new(1, 2), BlockRange::new(2, 3), None),
            clause(0, BlockRange::new(1, 2), BlockRange::new(3, 4), None),
        ]);

        let regions = ProtectedRegions::from_function(&ssa);
        assert!(regions.rejected().is_empty());
        assert_eq!(regions.regions().len(), 1, "one try range, one region");

        let region = regions.regions().first().expect("one region");
        assert_eq!(region.entry, NodeId::new(1));
        assert_eq!(region.handlers.len(), 2, "both catches survive");
        assert_eq!(
            region
                .handlers
                .iter()
                .map(|handler| handler.entry)
                .collect::<Vec<_>>(),
            vec![NodeId::new(2), NodeId::new(3)]
        );
    }

    /// `try { try {} catch {} } finally {}` is the ordinary nested encoding, and
    /// both regions open. The inner region's blocks are a subset of the outer's,
    /// and both sets are stated over the function's whole index domain.
    #[test]
    fn a_nested_clause_expands_to_a_contained_block_set() {
        let mut ssa = function(vec![
            vec![SsaOp::Jump { target: 1 }],
            vec![SsaOp::Jump { target: 2 }],
            vec![SsaOp::Jump { target: 5 }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
        ]);
        ssa.set_exception_handlers(vec![
            clause(0, BlockRange::new(1, 2), BlockRange::new(3, 4), None),
            clause(2, BlockRange::new(1, 3), BlockRange::new(4, 5), None),
        ]);

        let regions = ProtectedRegions::from_function(&ssa);
        assert!(regions.rejected().is_empty(), "nesting is not a collision");
        assert_eq!(regions.regions().len(), 2);

        let inner = regions.regions().first().expect("the inner region");
        let outer = regions.regions().get(1).expect("the outer region");
        assert_eq!(inner.blocks.node_count(), 6, "sized to the whole function");
        assert_eq!(outer.blocks.node_count(), 6);
        assert_eq!(
            inner.blocks.iter().collect::<Vec<_>>(),
            vec![NodeId::new(1)]
        );
        assert_eq!(
            outer.blocks.iter().collect::<Vec<_>>(),
            vec![NodeId::new(1), NodeId::new(2)]
        );
        assert!(
            inner
                .blocks
                .iter()
                .all(|block| outer.blocks.contains(block)),
            "the inner region lies inside the outer one"
        );
    }

    /// A filter can be laid out after the handler it belongs to, so its extent
    /// is stored rather than derived. Deriving it as
    /// `[filter_start, handler_start)` reverses exactly this range.
    #[test]
    fn a_filter_laid_out_after_its_handler_still_expands() {
        let mut ssa = function(vec![
            vec![SsaOp::Jump { target: 1 }],
            vec![SsaOp::Jump { target: 4 }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
        ]);
        ssa.set_exception_handlers(vec![clause(
            1, // a filter handler, in `MockTarget`'s flag convention
            BlockRange::new(1, 2),
            BlockRange::new(2, 3),
            BlockRange::new(3, 4),
        )]);

        let regions = ProtectedRegions::from_function(&ssa);
        assert!(regions.rejected().is_empty());

        let handler = regions
            .regions()
            .first()
            .and_then(|region| region.handlers.first())
            .expect("one handler");
        assert_eq!(handler.entry, NodeId::new(2));
        let filter = match &handler.kind {
            ProtectedHandlerKind::Filter(filter) => filter,
            other => panic!("expected a filter handler, got {other:?}"),
        };
        assert_eq!(filter.entry, NodeId::new(3), "the filter follows the body");
        assert_eq!(
            filter.blocks.iter().collect::<Vec<_>>(),
            vec![NodeId::new(3)]
        );
        assert_eq!(filter.blocks.node_count(), 5);
    }

    /// A clause the layout refuses is *named*, not quietly skipped -- and it is
    /// named on the path a host actually calls.
    ///
    /// A `Structured` on its own has no channel for this, so shipping the
    /// advertised entry point as the lossy one would put the silent degradation
    /// back exactly where the design removed it.
    #[test]
    fn a_malformed_clause_is_rejected_not_dropped() {
        let mut ssa = function(vec![
            vec![SsaOp::Jump { target: 1 }],
            vec![SsaOp::Jump { target: 2 }],
            vec![SsaOp::Return { value: None }],
        ]);
        ssa.set_exception_handlers(vec![clause(
            0,
            BlockRange::new(0, 1),
            BlockRange::new(1, 9),
            None,
        )]);

        let recovered = structure_ssa(&ssa);

        assert!(
            recovered.regions.regions().is_empty(),
            "the clause could not be laid out, so it opened no region"
        );
        assert_eq!(
            recovered.regions.rejected(),
            &[RejectedClause {
                clause: 0,
                reason: ClauseRejection::Malformed(ExceptionTableError::OutOfRange {
                    part: crate::ir::exception::ClausePart::Handler,
                    end: 9,
                    block_count: 3,
                }),
            }],
            "and `structure_ssa` says so"
        );
        assert_eq!(
            recovered.structured.metrics.unreached, 0,
            "the tree still accounts for every block"
        );
    }

    /// Two protected ranges that overlap without either containing the other
    /// are not a region shape any structured form has, so the later one is
    /// rejected rather than torn.
    #[test]
    fn a_partially_overlapping_clause_is_rejected() {
        let mut ssa = function(vec![
            vec![SsaOp::Jump { target: 1 }],
            vec![SsaOp::Jump { target: 2 }],
            vec![SsaOp::Jump { target: 3 }],
            vec![SsaOp::Jump { target: 6 }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
            vec![SsaOp::Return { value: None }],
        ]);
        ssa.set_exception_handlers(vec![
            clause(0, BlockRange::new(1, 3), BlockRange::new(4, 5), None),
            clause(0, BlockRange::new(2, 4), BlockRange::new(5, 6), None),
        ]);

        let regions = ProtectedRegions::from_function(&ssa);
        assert_eq!(regions.regions().len(), 1, "the first clause is kept");
        assert_eq!(
            regions.rejected(),
            &[RejectedClause {
                clause: 1,
                reason: ClauseRejection::PartiallyOverlapping { other: 0 },
            }]
        );
    }

    /// The derived set answers from block contents, and says no to the two
    /// shapes that cannot move: a block that writes memory, and a block whose
    /// value another block reads.
    #[test]
    fn condition_only_admits_a_pure_test_and_excludes_a_store() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 4);
        let local = |ssa: &mut SsaFunction<MockTarget>, idx: u16, block, instr| {
            ssa.create_variable(
                VariableOrigin::Local(idx),
                0,
                DefSite::instruction(block, instr),
                MockType::I32,
            )
        };
        let condition = local(&mut ssa, 0, 0, 0);
        let address = local(&mut ssa, 1, 1, 0);
        let stored = local(&mut ssa, 2, 1, 1);
        let escaping = local(&mut ssa, 3, 2, 0);

        // B0: a pure test, read only by its own branch.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: condition,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        // B1: a store, which cannot be relocated into a condition. Everything
        // it reads it also defines, so the only reason it is excluded is the
        // write.
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: address,
            value: ConstValue::I32(0x1000),
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: stored,
            value: ConstValue::I32(7),
        }));
        b1.add_instruction(instr(SsaOp::StoreIndirect {
            addr: address,
            value: stored,
            value_type: MockType::I32,
            address_space: None,
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b1);

        // B2: pure, but the value it computes is read in B3.
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: escaping,
            value: ConstValue::I32(2),
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);

        // B3: leaves the function; not a condition at all.
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return {
            value: Some(escaping),
        }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let set = condition_only_blocks(&ssa);

        assert_eq!(
            set.node_count(),
            ssa.block_count(),
            "the set states the domain it was built for"
        );
        assert!(set.contains(NodeId::new(0)), "a pure test can be folded");
        assert!(!set.contains(NodeId::new(1)), "a store cannot be moved");
        assert!(
            !set.contains(NodeId::new(2)),
            "nor can a block another block reads a value from"
        );
        assert!(!set.contains(NodeId::new(3)), "a return is not a condition");
    }

    /// The graph handed to the structurer carries terminator edges only.
    ///
    /// A handler is entered by the runtime, so nothing branches to it. If this
    /// ever reports an `if`, `SsaCfg` has grown an exception edge again and
    /// every unconditional jump out of a protected block now looks like a
    /// two-way branch.
    #[test]
    fn structure_ssa_keeps_the_handler_out_of_the_control_flow() {
        let recovered = structure_ssa(&try_catch_fixture());

        assert_eq!(recovered.regions.regions().len(), 1);
        assert!(recovered.regions.rejected().is_empty());
        assert_eq!(
            recovered.structured.metrics.ifs, 0,
            "no terminator branches into a handler"
        );
        assert_eq!(
            recovered.structured.metrics.unreached, 0,
            "the handler is still placed, inside its `Try`"
        );
    }

    /// With no table, the recovery is exactly the graph recovery rooted at the
    /// entry block -- same tree, no regions, nothing rejected.
    #[test]
    fn an_empty_exception_table_matches_the_graph_entry_point() {
        let ssa = function(vec![
            vec![SsaOp::Jump { target: 1 }],
            vec![SsaOp::Jump { target: 2 }],
            vec![SsaOp::Return { value: None }],
        ]);

        let recovered = structure_ssa(&ssa);
        assert!(recovered.regions.is_empty());
        assert!(recovered.regions.rejected().is_empty());

        let condition_only = condition_only_blocks(&ssa);
        let cfg = SsaCfg::from_ssa(&ssa);
        let direct = structure_with(
            &cfg,
            NodeId::new(0),
            &StructureOptions {
                regions: &[],
                condition_expressible: condition_only.clone(),
                condition_only,
                max_depth: DEFAULT_MAX_DEPTH,
            },
        );

        assert_eq!(recovered.structured, direct);
    }
}
