//! One exception-aware flow view of a function.
//!
//! # Why this exists
//!
//! A lifted function has more than one entry. Control reaches a handler or a
//! filter because the runtime dispatched to it, not because any terminator
//! named it, so a graph built from terminators alone says those blocks are
//! unreachable - and every analysis rooted at the entry block then answers
//! "unreachable, therefore no phi / no check / no loop / no dominance" for all
//! of them.
//!
//! This module is where "control can enter this block without a terminator
//! naming it" is represented, once. Every analysis that needs the notion -
//! reachability, dominance, dominance frontiers, dataflow order - asks here
//! rather than deriving it privately from the exception table, so the answers
//! cannot disagree with each other.
//!
//! # The two types here, and the graph
//!
//! - [`FunctionRoots`] answers *where control can start*: the entry block, plus
//!   one [`ExceptionRoot`] per handler and filter entry.
//! - [`SsaCfg`] is the flow graph in which those roots are reachable, built by
//!   attaching each root to every block of the region it covers.
//! - [`EhDominance`] is dominance over that graph. It keeps three distinct
//!   questions apart: [`EhDominance::definition_reaches`] (may a pass rewrite
//!   through here), [`EhDominance::dominates_block`] (plain block dominance),
//!   and [`EhDominance::definition_is_well_formed`] (is the IR legal at all).
//!   They have different answers, and using one where another is meant is
//!   either a miscompile or a false rejection.
//!
//! # What the edge model claims, and what it does not
//!
//! Every block of a protected region has an edge to the roots that cover it, so
//! the graph says what is true: a throw anywhere in the region reaches the
//! handler. What an edge cannot say is *where in the block* the throw happened,
//! and that matters, because the block may have redefined a value before it.
//! The values are therefore carried by the phi at the root rather than by the
//! edge: one exceptional operand per value a throw could arrive with, which is
//! the value the block was entered holding plus every version it defined. A
//! merge over program points, stated in the operands, where an edge can only
//! merge over edges.
//!
//! Dominance keeps the matching rule: a definition positioned *inside* an
//! edge-source block does not reach the handler
//! ([`EhDominance::definition_reaches`]), because the throw need not have got
//! that far.
//!
//! # The normal path through a finally
//!
//! A `finally` is not only an exception construct: it runs on every exit from
//! its region, and the ordinary exit is a `leave`. A leave does not go where it
//! says - control runs the finally first and resumes at the leave's target
//! afterwards - so a graph built from the terminator is wrong twice over, and
//! [`FinallyFlow`] is the correction. [`SsaCfg`] applies it, so the leave's
//! edge goes to the finally and `endfinally` gets the edges it resumes on.
//!
//! There is no virtual super-root. A node `R` with edges to the entry and to
//! every root would make `dominates(entry, x)` false for every block reachable
//! from both - which is the join after every try/catch and everything after it.
//! Every root is instead reachable *from the entry block itself*, so the node
//! count equals the block count, `immediate_dominator(entry)` stays `None`, and
//! dominance outside the exception-reached set matches the terminator-only
//! answer exactly.

use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
    ops::Range,
};

use crate::{
    BitSet,
    analysis::cfg::SsaCfg,
    graph::{
        GraphBase, NodeId,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        SsaFunction,
        exception::{ClausePart, HandlerKind},
        ops::SsaOp,
        variable::DefSite,
    },
    target::Target,
};

/// What kind of entry an [`ExceptionRoot`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootKind {
    /// A handler entry: the runtime dispatches here when the region throws.
    Handler,
    /// A filter entry: the runtime evaluates this before choosing a handler.
    Filter,
    /// A block reached only by the runtime unwinding into it, recognised by its
    /// leading `EndFinally` or `Rethrow`.
    ///
    /// This is a **preservation** root, not a flow root. It gets no edge and no
    /// dominance answer: promoting it would stop unreachable-block cleanup from
    /// clearing such a block, and would remove the filter that currently
    /// suppresses the errors it would otherwise raise, so construction would
    /// begin returning `Err` on input it accepts.
    Terminator,
}

/// One place control can enter a function other than its entry block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionRoot {
    /// The block control enters.
    pub block: usize,
    /// Whether this is a handler, a filter, or a preservation root.
    pub kind: RootKind,
    /// The block whose execution can dispatch here, when the function has one
    /// and it is reachable. `None` means the edge falls back to the entry.
    pub guard: Option<usize>,
    /// The protected region whose throw reaches this root, when known.
    pub region: Option<Range<usize>>,
    /// Index into [`SsaFunction::exception_handlers`], for a root that came
    /// from a clause.
    pub handler_index: Option<usize>,
}

/// Every place control can enter a function.
///
/// This is the crate's single reading of the exception table's entry points.
/// Reachability, dominance, SSA reconstruction, verification and dead-code
/// elimination all take their root set from here, so they cannot disagree about
/// which blocks the runtime can dispatch to.
#[derive(Debug, Clone)]
pub struct FunctionRoots {
    /// Roots that get a graph edge: handler and filter entries.
    flow_roots: Vec<ExceptionRoot>,
    /// Roots that get no edge but must not be deleted.
    preservation_roots: Vec<usize>,
    /// The handler entry of the clause each preservation root unwinds, by
    /// the same index; `None` for one in no handler.
    preservation_owners: Vec<Option<usize>>,
    /// Membership test for [`Self::is_flow_root`].
    flow_root_blocks: BitSet,
    /// Blocks that guard some flow root.
    guards: BitSet,
    /// Blocks inside some protected region.
    protected: BitSet,
    /// The protected region of each flow root, indexed as `flow_roots`.
    regions: Vec<Option<Range<usize>>>,
    /// Per block, the flow roots a throw from inside it can reach.
    covering: Vec<Vec<usize>>,

    /// The inverse of [`Self::covering`]: for each flow root, the blocks a
    /// throw in which may dispatch to it, in block order.
    covered_by: Vec<Vec<usize>>,
}

impl FunctionRoots {
    /// Collects every entry point of `ssa`.
    ///
    /// A clause naming a block the function does not have contributes nothing,
    /// matching the rule the CFG already applies to an out-of-range successor.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function whose exceptional and ordinary entry points to collect.
    ///
    /// # Returns
    ///
    /// The complete root set: flow roots (handler and filter entries) with their
    /// guards and protected regions, plus the preservation roots.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let block_count = ssa.block_count();
        let mut flow_roots: Vec<ExceptionRoot> = Vec::new();
        let mut flow_root_blocks = BitSet::new(block_count);
        let mut guards = BitSet::new(block_count);
        let mut protected = BitSet::new(block_count);
        for (handler_index, handler) in ssa.exception_handlers().iter().enumerate() {
            let region = handler
                .protected_range
                .filter(|range| range.start() < block_count)
                .map(|range| range.start()..range.end().min(block_count));
            if let Some(region) = region.clone() {
                for block in region {
                    protected.insert_checked(block);
                }
            }
            let guard = handler
                .protected_range
                .map(|range| range.start())
                .filter(|block| *block < block_count);
            if let Some(block) = guard {
                guards.insert_checked(block);
            }
            for (part, range) in handler.parts() {
                let kind = match part {
                    ClausePart::Protected => continue,
                    ClausePart::Handler => RootKind::Handler,
                    ClausePart::Filter => RootKind::Filter,
                };
                let block = range.start();
                if block >= block_count {
                    continue;
                }
                if !flow_root_blocks.insert_checked(block) {
                    for root in &mut flow_roots {
                        if root.block == block {
                            if root.region != region {
                                root.region = None;
                            }
                            break;
                        }
                    }
                    continue;
                }
                flow_roots.push(ExceptionRoot {
                    block,
                    kind,
                    guard,
                    region: region.clone(),
                    handler_index: Some(handler_index),
                });
            }
        }
        let regions = (0..flow_roots.len())
            .map(|index| flow_roots.get(index).and_then(|root| root.region.clone()))
            .collect::<Vec<_>>();
        let mut covering = vec![Vec::new(); block_count];
        for handler in ssa.exception_handlers() {
            let blocks = handler
                .protected_range
                .filter(|range| range.start() < block_count)
                .map_or(0..block_count, |range| {
                    range.start()..range.end().min(block_count)
                });
            for root in handler.entry_blocks().filter(|root| *root < block_count) {
                for block in blocks.clone() {
                    if let Some(slot) = covering.get_mut(block)
                        && !slot.contains(&root)
                    {
                        slot.push(root);
                    }
                }
            }
        }
        for handler in ssa.exception_handlers() {
            let Some(filter) = handler
                .filter_range
                .filter(|range| range.start() < block_count)
                .map(|range| range.start()..range.end().min(block_count))
            else {
                continue;
            };
            let Some(root) = handler
                .handler_range
                .map(|range| range.start())
                .filter(|root| *root < block_count)
            else {
                continue;
            };
            for block in filter {
                if let Some(slot) = covering.get_mut(block)
                    && !slot.contains(&root)
                {
                    slot.push(root);
                }
            }
        }
        let mut covered_by = vec![Vec::new(); block_count];
        for (block, roots) in covering.iter().enumerate() {
            for root in roots {
                if let Some(slot) = covered_by.get_mut(*root) {
                    slot.push(block);
                }
            }
        }
        let mut preservation_roots = Vec::new();
        for (index, block) in ssa.iter_blocks() {
            if !flow_root_blocks.contains_checked(index)
                && block.instructions().first().is_some_and(|instruction| {
                    matches!(instruction.op(), SsaOp::EndFinally | SsaOp::Rethrow)
                })
            {
                preservation_roots.push(index);
            }
        }

        let mut preservation_owners = Vec::with_capacity(preservation_roots.len());
        for block in &preservation_roots {
            let mut best: Option<(usize, usize)> = None;
            for handler in ssa.exception_handlers() {
                if let Some(range) = handler.handler_range
                    && range.contains(*block)
                {
                    let width = range.end().saturating_sub(range.start());
                    if best.is_none_or(|(known, _)| width < known) {
                        best = Some((width, range.start()));
                    }
                }
            }
            preservation_owners.push(best.map(|(_, start)| start));
        }
        Self {
            flow_roots,
            preservation_roots,
            preservation_owners,
            flow_root_blocks,
            guards,
            protected,
            regions,
            covering,
            covered_by,
        }
    }

    /// The function's ordinary entry block.
    #[must_use]
    pub fn entry(&self) -> NodeId {
        NodeId::new(0)
    }

    /// Every root that gets a graph edge, in clause order.
    #[must_use]
    pub fn flow_roots(&self) -> &[ExceptionRoot] {
        &self.flow_roots
    }

    /// Blocks that must survive elimination although no edge reaches them.
    pub fn preservation_roots(&self) -> impl Iterator<Item = usize> + '_ {
        self.preservation_roots.iter().copied()
    }

    /// Every block control can enter the function at, the entry first: the
    /// entry block, each flow root in clause order, then each preservation
    /// root. A root may appear more than once, and may name a block the
    /// function does not have.
    pub fn entries(&self) -> impl Iterator<Item = usize> + '_ {
        iter::once(0)
            .chain(self.flow_roots.iter().map(|root| root.block))
            .chain(self.preservation_roots())
    }

    /// Every block control can reach.
    ///
    /// Everything [`SsaCfg`]'s complete relation reaches from the entry. That
    /// relation draws an edge from every block of a protected region to the
    /// handler or filter a throw there reaches, so a handler is live exactly
    /// when some block it protects is: the runtime dispatches to it only from
    /// there. A handler whose protected code nothing reaches is dead with it,
    /// and treating it as a root of its own kept code alive that reads values
    /// only the dead region defines. A flow root whose region is unknown is a
    /// root of its own, as nothing shows what can dispatch to it. A
    /// preservation root gets no edge from anywhere, so it is a root too - but
    /// one inside a handler only once that handler is live, since it unwinds
    /// that clause; rooted unconditionally, it revived code whose values the
    /// dead clause was the only one to define.
    ///
    /// The one answer to "which blocks are live": dead-code elimination,
    /// unreachable-block pruning and the SSA rebuild each derived it privately
    /// from the exception table, and the rebuild's copy omitted the
    /// preservation roots the other two kept.
    ///
    /// # Arguments
    ///
    /// * `cfg` - The flow view of the function these roots were read from.
    #[must_use]
    pub fn live_blocks<T: Target>(&self, cfg: &SsaCfg<'_, T>) -> BitSet {
        let block_count = cfg.block_count();
        let mut live = BitSet::new(block_count);
        let mut worklist: Vec<usize> = Vec::new();
        let unknown_regions = self
            .flow_roots
            .iter()
            .zip(&self.regions)
            .filter(|(_, region)| region.is_none())
            .map(|(root, _)| root.block);
        // A preservation root in a handler is live once that handler is: it
        // unwinds that clause and no other, and rooted on its own it brought
        // back code that reads values only the dead clause defined.
        let unowned = self
            .preservation_roots
            .iter()
            .zip(&self.preservation_owners)
            .filter(|(_, owner)| owner.is_none())
            .map(|(root, _)| *root);
        let walk = |root: usize, live: &mut BitSet, worklist: &mut Vec<usize>| {
            if root >= block_count || live.contains(root) {
                return false;
            }
            worklist.push(root);
            while let Some(block) = worklist.pop() {
                if block < block_count && live.insert(block) {
                    worklist.extend(cfg.block_successors(block).iter().copied());
                }
            }
            true
        };
        for root in iter::once(0).chain(unknown_regions).chain(unowned) {
            walk(root, &mut live, &mut worklist);
        }
        loop {
            let mut grew = false;
            for (&root, owner) in self
                .preservation_roots
                .iter()
                .zip(&self.preservation_owners)
            {
                if owner.is_some_and(|owner| live.contains_checked(owner)) {
                    grew |= walk(root, &mut live, &mut worklist);
                }
            }
            if !grew {
                break;
            }
        }
        live
    }

    /// Whether control can enter `block` without a terminator naming it.
    #[must_use]
    pub fn is_flow_root(&self, block: usize) -> bool {
        self.flow_root_blocks.contains_checked(block)
    }

    /// Whether `block` guards some flow root.
    #[must_use]
    pub fn is_guard(&self, block: usize) -> bool {
        self.guards.contains_checked(block)
    }

    /// Whether `block` lies inside some protected region.
    #[must_use]
    pub fn is_protected(&self, block: usize) -> bool {
        self.protected.contains_checked(block)
    }

    /// The protected region whose throw reaches `block`, when `block` is a flow
    /// root of a clause with a known region.
    #[must_use]
    pub fn region_of(&self, block: usize) -> Option<Range<usize>> {
        let index = self.flow_roots.iter().position(|r| r.block == block)?;
        self.regions.get(index).cloned().flatten()
    }

    /// The handler and filter entries a throw from inside `block` can reach.
    ///
    /// Every block of a protected region, not only its first: an instruction
    /// anywhere in the region can throw, and control then continues at the
    /// region's filter or handler rather than at the block's terminator
    /// successors. [`SsaCfg`] draws one edge per root, from its guard, which is
    /// what dominance needs; a question about where control can *go* from a
    /// given block - is a value observed before the function exits, is a block
    /// executable - needs this instead. A root whose region is unknown is
    /// listed for every block.
    ///
    /// # Arguments
    ///
    /// * `block` - The block a throw would leave.
    ///
    /// # Returns
    ///
    /// The root blocks, in clause order; empty outside every protected region
    /// and for a block the function does not have.
    #[must_use]
    pub fn exceptional_successors(&self, block: usize) -> &[usize] {
        self.covering.get(block).map_or(&[], Vec::as_slice)
    }

    /// Whether an exception raised inside `block` can reach `target` without
    /// passing through `block` again.
    ///
    /// A throw leaves a block from any of its instructions, so it can leave
    /// before a definition in the block has run. Where it then reaches, that
    /// definition does not hold, even when `block` dominates the place: a
    /// `finally` the protected code also enters by `leave` is dominated by the
    /// region's first block, and a value that block computes is still missing
    /// when the first instruction of the region throws.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function both blocks are in.
    /// * `block` - The block the exception is raised in.
    /// * `target` - The block it may reach.
    ///
    /// # Returns
    ///
    /// `true` if a path from one of `block`'s exceptional successors reaches
    /// `target` without entering `block`.
    #[must_use]
    pub fn raise_reaches<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        block: usize,
        target: usize,
    ) -> bool {
        let starts = self.exceptional_successors(block);
        if starts.is_empty() {
            return false;
        }
        let cfg = SsaCfg::from_ssa(ssa);
        let mut seen = BitSet::new(ssa.block_count().max(1));
        let mut work: Vec<usize> = Vec::new();
        for &start in starts {
            if start != block && seen.insert_checked(start) {
                work.push(start);
            }
        }
        while let Some(current) = work.pop() {
            if current == target {
                return true;
            }
            let normal = cfg.block_successors(current).iter().copied();
            let raised = self.exceptional_successors(current).iter().copied();
            for next in normal.chain(raised) {
                if next != block && seen.insert_checked(next) {
                    work.push(next);
                }
            }
        }
        false
    }

    /// The blocks a throw in which may dispatch to `root`.
    ///
    /// The inverse of [`Self::exceptional_successors`]: the protected blocks a
    /// handler or filter entry can be reached from, plus - for a handler that
    /// has one - the blocks of its filter, which runs in between and can write
    /// the same storage.
    ///
    /// # Arguments
    ///
    /// * `root` - The handler or filter entry block.
    ///
    /// # Returns
    ///
    /// The source blocks in ascending order; empty for a block that is not a
    /// flow root, or whose region no block belongs to.
    #[must_use]
    pub fn exceptional_predecessors(&self, root: usize) -> &[usize] {
        self.covered_by.get(root).map_or(&[], Vec::as_slice)
    }

    /// Whether the function has any exceptional entry at all.
    ///
    /// When this is `false`, every answer in this module is identical to the
    /// entry-rooted one, and callers may take the cheaper path.
    #[must_use]
    pub fn has_flow_roots(&self) -> bool {
        !self.flow_roots.is_empty()
    }
}

/// Where the normal path through a `finally` goes.
///
/// A `finally` runs on *every* exit from its region, not only on a throw. The
/// ordinary exit is a `leave`, and it does not go where it says: control enters
/// the finally first, and resumes at the leave's target only once `endfinally`
/// runs. A leave that exits several nested regions runs each of their finallys
/// in turn, innermost first.
///
/// Reading the terminator alone therefore gets the graph wrong in both
/// directions: it gives the leave an edge to a block control does not reach
/// yet, and it gives `endfinally` no edge at all - so a value the finally
/// assigns looks dead, a value live across the leave looks like it bypasses the
/// finally, and any block only a finally reaches looks unreachable. This is the
/// correction, and [`SsaCfg`] applies it, so no caller has to know the rule.
///
/// # What it approximates
///
/// `endfinally` returns to whichever leave entered the finally, which is a
/// return edge rather than a branch. Modelled here as an edge to *every*
/// possible resume point, exactly as a subroutine analysed without call-return
/// matching. That over-approximates paths, never values: the merge at a resume
/// point names the storage the values already share, so a read there still
/// reads what the taken path put in it. Only a path-sensitive answer - which
/// constant arrived - is coarsened.
///
/// An `endfinally` reached by unwinding does not resume in this function at
/// all; the edges below are the normal path, and the unwind path continues
/// outward as it always did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FinallyFlow {
    /// Block ending in a `leave` that must run a finally, to the entry of the
    /// first one it runs.
    diverted: BTreeMap<usize, usize>,
    /// Block ending in `endfinally`, to where control resumes when it does.
    resumes: BTreeMap<usize, Vec<usize>>,
}

impl FinallyFlow {
    /// Reads the exception table and the `leave` instructions of `ssa`.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function whose finallys and leaves are read.
    ///
    /// # Returns
    ///
    /// The correction, empty when the function has no finally or fault clause.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let mut clauses = Vec::new();
        for handler in ssa.exception_handlers() {
            if !matches!(handler.kind(), HandlerKind::Finally) {
                continue;
            }
            if let (Some(protected), Some(handler)) =
                (handler.protected_range, handler.handler_range)
            {
                clauses.push(FinallyClause {
                    protected: protected.start()..protected.end(),
                    handler: handler.start()..handler.end(),
                });
            }
        }
        if clauses.is_empty() {
            return Self::default();
        }
        let mut flow = Self::default();
        let mut resumes: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        let n = ssa.block_count();
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            let Some(SsaOp::Leave { target }) = block.control_terminator() else {
                continue;
            };
            let target = *target;
            if target >= n {
                continue;
            }
            // Width plus source position preserves stable ordering for equal-width clauses.
            let mut ordered = BTreeMap::new();
            for (index, clause) in clauses.iter().enumerate() {
                if clause.covers(block_idx) && !clause.covers(target) {
                    ordered.insert(
                        (
                            clause.protected.end.saturating_sub(clause.protected.start),
                            index,
                        ),
                        clause,
                    );
                }
            }
            if ordered.is_empty() {
                continue;
            }

            let mut exited = Vec::with_capacity(ordered.len());
            for clause in ordered.into_values() {
                exited.push(clause);
            }
            let mut entries = Vec::new();
            for clause in &exited {
                let entry = clause.handler.start;
                if entry < n && entries.last().copied() != Some(entry) {
                    entries.push(entry);
                }
            }
            let Some(first) = entries.first().copied() else {
                continue;
            };

            flow.diverted.insert(block_idx, first);
            for (position, clause) in exited.iter().enumerate() {
                let resume = entries
                    .get(position.saturating_add(1))
                    .copied()
                    .unwrap_or(target);
                for end in clause.handler.start.min(n)..clause.handler.end.min(n) {
                    if ssa.block(end).is_none_or(|block| {
                        !matches!(block.control_terminator(), Some(SsaOp::EndFinally))
                    }) {
                        continue;
                    }
                    let targets = resumes.entry(end).or_default();
                    targets.insert(resume);
                }
            }
        }
        for (end, targets) in resumes {
            let mut ordered = Vec::with_capacity(targets.len());
            for target in targets {
                ordered.push(target);
            }

            flow.resumes.insert(end, ordered);
        }
        flow
    }

    /// Whether the function has any finally to route through.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.diverted.is_empty() && self.resumes.is_empty()
    }

    /// The finally entry a `leave` in `block` reaches first, if any.
    ///
    /// # Arguments
    ///
    /// * `block` - The block whose terminator is being resolved.
    #[must_use]
    pub fn diverted_target(&self, block: usize) -> Option<usize> {
        self.diverted.get(&block).copied()
    }

    /// Where control resumes when the `endfinally` in `block` runs, ascending.
    ///
    /// Empty for a block that is not the end of a finally any leave enters, and
    /// for one reached only by unwinding.
    ///
    /// # Arguments
    ///
    /// * `block` - The block whose `endfinally` is being resolved.
    #[must_use]
    pub fn resume_targets(&self, block: usize) -> &[usize] {
        self.resumes.get(&block).map_or(&[], Vec::as_slice)
    }
}

/// One finally clause, as [`FinallyFlow`] reads it.
#[derive(Debug, Clone)]
struct FinallyClause {
    /// The protected region, as a half-open block range.
    protected: Range<usize>,
    /// The handler body, as a half-open block range.
    handler: Range<usize>,
}

impl FinallyClause {
    /// Whether `block` is inside this clause at all - guarded or handling.
    ///
    /// A `leave` out of the handler leaves the clause exactly as one out of the
    /// region does.
    fn covers(&self, block: usize) -> bool {
        self.protected.contains(&block) || self.handler.contains(&block)
    }
}

/// Dominance over the exception-aware view, with the three questions callers
/// actually ask kept apart.
///
/// They are ordered by strictness, and the order is the contract:
///
/// ```text
/// definition_reaches  ⊆  dominates_block  ⊆  definition_is_well_formed
/// ```
///
/// - [`Self::definition_reaches`] is the **rewrite** question, and the
///   strictest. A pass may replace a use with a value only if that value is
///   guaranteed to hold at the use. It is EH dominance *minus* a definition
///   positioned at an instruction inside an edge-source block, because the
///   synthesized edge claims the guard ran to completion and a throw need not
///   have let it.
/// - [`Self::dominates_block`] is plain block dominance in the EH graph.
/// - [`Self::definition_is_well_formed`] is the **verifier's floor**, and the
///   loosest: EH dominance *or* normal-path dominance. It encodes an explicit
///   contract - a definition must dominate its use on the normal path, and this
///   crate does not model partial execution of a protected region, so a value
///   defined inside a region and read at the region's join stays legal. Without
///   that, every ungrouped temporary defined in a try body and used after the
///   join would be a violation, which is ordinary IR the crate itself emits.
#[derive(Debug, Clone)]
pub struct EhDominance {
    /// Dominance over the exception-aware graph.
    tree: DominatorTree,
    /// Dominance over the terminator relation alone, for the normal-path half
    /// of [`Self::definition_is_well_formed`].
    ///
    /// `None` when the function has no exceptional entry: the exception-aware
    /// graph then has no edge the terminators lack, so the two relations are
    /// one and `tree` answers for both.
    normal_tree: Option<DominatorTree>,
    /// Blocks reachable only because an exceptional edge exists.
    exception_reachable: BitSet,
    /// Tails of the synthesized edges.
    edge_sources: BitSet,
}

impl EhDominance {
    /// Computes dominance over `eh`.
    ///
    /// Computes dominance over both the exception-aware graph and the
    /// terminator-only graph, then records the blocks reachable in the former
    /// but not the latter - the ones the synthesized edges added.
    ///
    /// The exception-aware graph's dominance frontiers are
    /// [`compute_dominance_frontiers`](crate::graph::algorithms::compute_dominance_frontiers)
    /// over it and [`Self::tree`]. A frontier row is keyed on the block holding
    /// a definition, and it is the row - not the dominance relation - that
    /// decides where a phi goes. A handler that rejoins the normal path has the
    /// join in its frontier, so a definition inside the handler demands a phi
    /// at that join, though under the terminator relation alone the handler
    /// has no row at all.
    ///
    /// # Arguments
    ///
    /// * `eh` - The exception-aware view to compute dominance over.
    #[must_use]
    pub fn of<T: Target>(eh: &SsaCfg<'_, T>) -> Self {
        let entry = NodeId::new(0);
        let tree = compute_dominators(eh, entry);
        let normal_tree = eh
            .roots()
            .has_flow_roots()
            .then(|| compute_dominators(&eh.terminator_view(), entry));

        let block_count = eh.node_count();
        let mut exception_reachable = BitSet::new(block_count);
        if let Some(normal_tree) = &normal_tree {
            for block in 0..block_count {
                let node = NodeId::new(block);
                // Reachable now, but not without the synthesized edges.
                if tree.is_reachable(node) && !normal_tree.is_reachable(node) {
                    exception_reachable.insert_checked(block);
                }
            }
        }

        Self {
            tree,
            normal_tree,
            exception_reachable,
            edge_sources: eh.edge_sources(),
        }
    }

    /// The exception-aware dominator tree.
    #[must_use]
    pub const fn tree(&self) -> &DominatorTree {
        &self.tree
    }

    /// Blocks reachable only via an exceptional edge.
    #[must_use]
    pub const fn exception_reachable(&self) -> &BitSet {
        &self.exception_reachable
    }

    /// Tails of the synthesized exceptional edges.
    #[must_use]
    pub const fn edge_sources(&self) -> &BitSet {
        &self.edge_sources
    }

    /// Whether `a` dominates `b` in the exception-aware graph.
    #[must_use]
    pub fn dominates_block(&self, a: NodeId, b: NodeId) -> bool {
        self.tree.dominates(a, b)
    }

    /// Whether a value defined at `def` is guaranteed to hold at the top of
    /// `use_block` - the question a rewrite must ask.
    ///
    /// Stricter than [`Self::dominates_block`] by exactly one rule: a
    /// definition positioned at an *instruction* does not reach a block an
    /// exception raised in its own block can reach without passing through
    /// that block again ([`FunctionRoots::raise_reaches`]), since the throw
    /// need not have reached the instruction. A definition at the *top* of a
    /// block (a phi) is unaffected, and so is every function with no exception
    /// table.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function this dominance was built for.
    /// * `roots` - The function's exception roots.
    /// * `def` - The definition site being tested, including whether it sits at
    ///   a phi or at a specific instruction.
    /// * `use_block` - The block at whose top the value must hold.
    ///
    /// # Returns
    ///
    /// `true` if the definition is guaranteed to hold at the top of `use_block`.
    #[must_use]
    pub fn definition_reaches<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        roots: &FunctionRoots,
        def: DefSite,
        use_block: usize,
    ) -> bool {
        if !self.dominates_block(NodeId::new(def.block()), NodeId::new(use_block)) {
            return false;
        }
        !(def.index().is_some()
            && def.block() != use_block
            && roots.raise_reaches(ssa, def.block(), use_block))
    }

    /// Whether a definition in `def_block` may legally be used in `use_block` -
    /// the verifier's floor.
    ///
    /// Looser than [`Self::dominates_block`]: normal-path dominance also
    /// suffices. A value defined inside a protected region and read at the
    /// region's join dominates that use on every path the IR models, and this
    /// crate does not model a partially executed region, so rejecting it would
    /// reject IR the crate itself produces.
    ///
    /// # Arguments
    ///
    /// * `def_block` - The block holding the definition.
    /// * `use_block` - The block holding the use.
    ///
    /// # Returns
    ///
    /// `true` if the definition dominates the use on the exception-aware graph
    /// or on the normal path.
    #[must_use]
    pub fn definition_is_well_formed(&self, def_block: usize, use_block: usize) -> bool {
        let (def, used) = (NodeId::new(def_block), NodeId::new(use_block));
        self.tree.dominates(def, used)
            || self
                .normal_tree
                .as_ref()
                .is_some_and(|normal| normal.dominates(def, used))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::algorithms::compute_dominance_frontiers,
        ir::{
            SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            instruction::SsaInstruction,
            variable::VariableOrigin,
        },
        testing::{
            MockTarget, MockType, diamond_chain_fixture, diamond_phi_fixture, loop_counter_fixture,
            try_catch_fixture,
        },
    };

    /// An `endfinally` block of a finally nothing reaches is not live, though
    /// no edge has to reach a preservation root for it to count; with the
    /// `try` reached, it is.
    ///
    /// `B0` jumps to `exit` (`B5`) or into the try `B1`; the finally `B2..B5`
    /// ends in `B4`, a leading `endfinally` reached through `B3`.
    #[test]
    fn a_preservation_root_lives_and_dies_with_its_handler() {
        let build = |enter_try: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
            let condition = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let ops = [
                SsaOp::Jump {
                    target: if enter_try { 1 } else { 5 },
                },
                SsaOp::Leave { target: 5 },
                SsaOp::Branch {
                    condition,
                    true_target: 3,
                    false_target: 4,
                },
                SsaOp::Jump { target: 4 },
                SsaOp::EndFinally,
                SsaOp::returns(None),
            ];
            for (index, op) in ops.into_iter().enumerate() {
                let mut block = SsaBlock::new(index);
                block.add_instruction(SsaInstruction::synthetic(op));
                ssa.add_block(block);
            }
            ssa.set_exception_handlers(vec![SsaExceptionHandler {
                flags: 2,
                try_offset: 0,
                try_length: 0,
                handler_offset: 0,
                handler_length: 0,
                class_token_or_filter: 0,
                protected_range: BlockRange::new(1, 2),
                handler_range: BlockRange::new(2, 5),
                filter_range: None,
                selector: None,
            }]);
            ssa.recompute_uses();
            ssa
        };
        let ssa = build(false);
        let roots = FunctionRoots::of(&ssa);
        assert_eq!(roots.preservation_roots().collect::<Vec<_>>(), [4]);
        let live = roots.live_blocks(&SsaCfg::from_ssa(&ssa));
        assert!(
            !live.contains(4),
            "the finally is dead, and its endfinally with it"
        );
        assert!(live.contains(5));

        let ssa = build(true);
        let live = FunctionRoots::of(&ssa).live_blocks(&SsaCfg::from_ssa(&ssa));
        assert!((1..=5).all(|block| live.contains(block)));
    }

    /// A handler is live exactly when code it protects is: with the protected
    /// block reached its handler is, and with the entry jumping past the
    /// protected block both are dead.
    ///
    /// Every handler entry used to be a root of its own, so a handler whose
    /// `try` became unreachable stayed live while the code it reads was
    /// removed with the `try`, and dead-code elimination left it reading
    /// values nothing defined.
    #[test]
    fn a_handler_is_live_only_while_its_protected_code_is() {
        let ssa = try_catch_fixture();
        let live = FunctionRoots::of(&ssa).live_blocks(&SsaCfg::from_ssa(&ssa));
        assert!(live.contains(1) && live.contains(2));

        let mut ssa = try_catch_fixture();
        let entry = ssa.block_mut(0).expect("the entry block");
        entry.instructions_mut().pop();
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        let live = FunctionRoots::of(&ssa).live_blocks(&SsaCfg::from_ssa(&ssa));
        assert!(!live.contains(1), "nothing reaches the protected block");
        assert!(!live.contains(2), "so nothing dispatches to its handler");
        assert!(live.contains(0) && live.contains(3));
    }

    /// Every block of a protected region can throw to its handler, not only
    /// the region's first block, which is the only one `SsaCfg` draws an edge
    /// from.
    #[test]
    fn every_block_of_a_region_reaches_its_handler() {
        let mut ssa = try_catch_fixture();
        let handlers: Vec<_> = ssa
            .exception_handlers()
            .iter()
            .cloned()
            .map(|mut handler| {
                handler.protected_range = BlockRange::new(0, 2);
                handler
            })
            .collect();
        ssa.set_exception_handlers(handlers);
        let roots = FunctionRoots::of(&ssa);

        assert_eq!(roots.exceptional_successors(0), &[2]);
        assert_eq!(roots.exceptional_successors(1), &[2]);
        assert!(roots.exceptional_successors(2).is_empty());
        assert!(roots.exceptional_successors(3).is_empty());
        assert!(roots.exceptional_successors(99).is_empty());
    }

    /// Two clauses sharing a handler each protect their own blocks: the
    /// handler is one root, reached from the blocks of both regions, and its
    /// region is unknown rather than either clause's alone.
    #[test]
    fn a_shared_handler_is_reached_from_every_clause_region() {
        let mut ssa = try_catch_fixture();
        let mut handlers = ssa.exception_handlers().to_vec();
        let mut second = handlers.first().cloned().expect("the fixture has a clause");
        if let Some(first) = handlers.first_mut() {
            first.protected_range = BlockRange::new(0, 1);
        }
        second.protected_range = BlockRange::new(1, 2);
        handlers.push(second);
        ssa.set_exception_handlers(handlers);
        let roots = FunctionRoots::of(&ssa);

        assert_eq!(roots.flow_roots().len(), 1, "one handler entry, one root");
        assert_eq!(roots.exceptional_successors(0), &[2]);
        assert_eq!(
            roots.exceptional_successors(1),
            &[2],
            "the second clause's block"
        );
        assert_eq!(roots.exceptional_predecessors(2), &[0, 1]);
        assert!(roots.is_protected(1));
        assert_eq!(
            roots.region_of(2),
            None,
            "two regions reach the root, which no one range states"
        );
        let eh = SsaCfg::from_ssa(&ssa);
        assert!(eh.is_exception_edge(1, 2));
    }

    /// A handler whose protected region is unknown may be reached from any
    /// block.
    #[test]
    fn an_unscoped_handler_is_reachable_from_every_block() {
        let mut ssa = try_catch_fixture();
        let handlers: Vec<_> = ssa
            .exception_handlers()
            .iter()
            .cloned()
            .map(|mut handler| {
                handler.protected_range = None;
                handler
            })
            .collect();
        ssa.set_exception_handlers(handlers);
        let roots = FunctionRoots::of(&ssa);

        for block in 0..ssa.block_count() {
            assert_eq!(roots.exceptional_successors(block), &[2], "B{block}");
        }
    }

    #[test]
    fn every_flow_root_is_reachable_from_the_entry() {
        let ssa = try_catch_fixture();
        let eh = SsaCfg::from_ssa(&ssa);
        let dominance = EhDominance::of(&eh);

        assert!(eh.roots().has_flow_roots());
        for root in eh.roots().flow_roots() {
            assert!(
                dominance.tree().is_reachable(NodeId::new(root.block)),
                "root B{} must be reachable by construction",
                root.block
            );
        }
    }

    /// The whole point of attaching roots to their guard rather than to a
    /// virtual super-root: everything outside the exception-reached set keeps
    /// the answers it had.
    #[test]
    fn entry_dominance_is_unchanged_outside_exception_reach() {
        let ssa = try_catch_fixture();
        let eh = SsaCfg::from_ssa(&ssa);
        let dominance = EhDominance::of(&eh);
        let plain = compute_dominators(&eh.terminator_view(), NodeId::new(0));

        for block in 0..eh.node_count() {
            if dominance.exception_reachable().contains_checked(block) {
                continue;
            }
            for other in 0..eh.node_count() {
                if dominance.exception_reachable().contains_checked(other) {
                    continue;
                }
                let (a, b) = (NodeId::new(block), NodeId::new(other));
                assert_eq!(
                    dominance.dominates_block(a, b),
                    plain.dominates(a, b),
                    "dominance moved for B{block} -> B{other}, outside exception reach"
                );
            }
        }

        // ... while frontier *contents* do move, which is not the same
        // statement and is how the new phis at try/catch joins arise. The join
        // In this graph B3 has two predecessors, one of them the handler, so it enters
        // the handler's frontier -- and a definition inside the handler
        // therefore demands a phi at the join. Under the terminator relation
        // alone the handler was unreachable and had no frontier row at all.
        //
        // Note the row is the *handler's*, not the guard's: the guard dominates
        // the join here, so the join cannot be in the guard's frontier. What
        // matters for phi placement is the frontier of the block holding the
        // definition, which for a store in a handler is the handler.
        let frontiers = compute_dominance_frontiers(&eh, dominance.tree());
        let handler_frontier = frontiers.get(2).expect("the handler has a frontier row");
        assert!(
            handler_frontier.contains_checked(3),
            "the try/catch join must enter the handler's frontier"
        );
        assert!(
            compute_dominators(&eh.terminator_view(), NodeId::new(0))
                .is_reachable(NodeId::new(2))
                .eq(&false),
            "and the handler had no frontier at all without the exception edge"
        );
    }

    #[test]
    fn a_terminator_root_gets_no_edge() {
        let ssa = try_catch_fixture();
        let roots = FunctionRoots::of(&ssa);
        assert!(
            roots
                .flow_roots()
                .iter()
                .all(|root| root.kind != RootKind::Terminator),
            "a preservation root must never appear among the flow roots"
        );
    }

    #[test]
    fn an_exception_edge_is_never_duplicated() {
        let ssa = try_catch_fixture();
        let eh = SsaCfg::from_ssa(&ssa);

        for block in 0..eh.node_count() {
            let successors: Vec<usize> = eh.block_successors(block).to_vec();
            let mut deduped = successors.clone();
            deduped.sort_unstable();
            deduped.dedup();
            assert_eq!(
                successors.len(),
                deduped.len(),
                "B{block} has a duplicate successor: {successors:?}"
            );
        }
    }

    /// A clause whose protected block already transfers to the handler gains no
    /// second, parallel edge in the graph - though the one edge does carry both
    /// kinds of transfer, and a throw on it may not have run the block's later
    /// definitions.
    #[test]
    fn a_self_guarding_region_adds_no_parallel_edge() {
        let mut ssa = try_catch_fixture();
        let mut handlers = ssa.exception_handlers().to_vec();
        if let Some(handler) = handlers.first_mut() {
            // Guard B0, which already jumps to B1; make B1 the handler entry.
            handler.protected_range = BlockRange::new(0, 1);
            handler.handler_range = BlockRange::new(1, 2);
        }
        ssa.set_exception_handlers(handlers);

        let eh = SsaCfg::from_ssa(&ssa);
        assert!(
            eh.is_exception_edge(0, 1),
            "a throw in B0 reaches the handler, whatever its terminator also does"
        );
        assert!(
            eh.terminator_successors(0).contains(&1),
            "and the terminator goes there too, so one edge carries both"
        );
        assert_eq!(
            eh.block_successors(0),
            eh.terminator_successors(0),
            "listed once: the complete relation gains no parallel edge"
        );
        assert!(
            eh.edge_sources().contains_checked(0),
            "B0 can still throw part-way through, so its definitions are not \
             guaranteed to have run"
        );
    }

    /// A handler the protected code also enters normally - a `finally` its
    /// `leave` runs - is dominated by the region's first block. A value that
    /// block computes still does not reach it: the first instruction of the
    /// region may throw before the value exists.
    #[test]
    fn a_definition_does_not_reach_a_handler_its_block_also_enters_normally() {
        let mut ssa = try_catch_fixture();
        // B1 throws to B3 and jumps there too; nothing enters B2 any more.
        let handlers: Vec<_> = ssa
            .exception_handlers()
            .iter()
            .cloned()
            .map(|mut handler| {
                handler.handler_range = BlockRange::new(3, 4);
                handler
            })
            .collect();
        ssa.set_exception_handlers(handlers);
        let eh = SsaCfg::from_ssa(&ssa);
        let dominance = EhDominance::of(&eh);
        let roots = FunctionRoots::of(&ssa);

        assert!(
            dominance.dominates_block(NodeId::new(1), NodeId::new(3)),
            "B1 dominates the handler: both ways in pass through it"
        );
        assert!(roots.raise_reaches(&ssa, 1, 3));
        assert!(
            !dominance.definition_reaches(&ssa, &roots, DefSite::instruction(1, 0), 3),
            "a throw from B1 reaches the handler before B1's definition ran"
        );
        assert!(dominance.definition_reaches(&ssa, &roots, DefSite::phi(1), 3));
        assert!(
            !roots.raise_reaches(&ssa, 0, 3),
            "B0 is protected by nothing and raises nowhere"
        );
    }

    /// The strictness order the three predicates promise.
    #[test]
    fn a_definition_in_an_edge_source_does_not_reach_the_handler() {
        let ssa = try_catch_fixture();
        let eh = SsaCfg::from_ssa(&ssa);
        let dominance = EhDominance::of(&eh);
        let roots = FunctionRoots::of(&ssa);

        // B1 guards the handler, so it is the tail of the synthesized edge.
        assert!(eh.edge_sources().contains_checked(1));
        assert!(dominance.exception_reachable().contains_checked(2));

        let mid_guard = DefSite::instruction(1, 0);
        assert!(
            dominance.dominates_block(NodeId::new(1), NodeId::new(2)),
            "block dominance holds -- the edge says so"
        );
        assert!(
            !dominance.definition_reaches(&ssa, &roots, mid_guard, 2),
            "but a definition part-way through the guard need not have executed"
        );

        // The top of the same block is unaffected: control reached the block.
        assert!(dominance.definition_reaches(&ssa, &roots, DefSite::phi(1), 2));
    }

    /// A clause with no usable guard still gets its root attached, from the
    /// entry.
    ///
    /// Its region is unknown, so a throw may be anywhere: every reachable block
    /// can dispatch to the handler, and each is an edge source. That is the
    /// conservative reading, and the only sound one when the table does not say
    /// where the region is.
    #[test]
    fn a_root_without_a_usable_guard_falls_back_to_the_entry() {
        let mut ssa = try_catch_fixture();
        let mut handlers = ssa.exception_handlers().to_vec();
        if let Some(handler) = handlers.first_mut() {
            handler.protected_range = None;
        }
        ssa.set_exception_handlers(handlers);

        let eh = SsaCfg::from_ssa(&ssa);
        assert!(eh.is_exception_edge(0, 2), "the fallback edge leaves B0");
        assert!(eh.edge_sources().contains_checked(0));
        assert!(
            eh.edge_sources().contains_checked(1),
            "an unknown region means a throw may be anywhere, B1 included"
        );
        assert!(EhDominance::of(&eh).tree().is_reachable(NodeId::new(2)));
    }

    #[test]
    fn a_function_without_a_table_is_the_plain_cfg() {
        let ssa = diamond_phi_fixture();
        let eh = SsaCfg::from_ssa(&ssa);
        let dominance = EhDominance::of(&eh);

        assert!(!eh.roots().has_flow_roots());
        assert!(eh.edge_sources().is_empty());
        assert!(dominance.exception_reachable().is_empty());
        for block in 0..eh.node_count() {
            let mut through_eh: Vec<usize> = eh.block_successors(block).to_vec();
            through_eh.sort_unstable();
            let mut plain = eh.terminator_successors(block).to_vec();
            plain.sort_unstable();
            assert_eq!(through_eh, plain);
        }
    }

    /// The verifier's floor is exception-aware dominance or normal-path
    /// dominance, and a function with no exceptional entry has one relation:
    /// the terminator graph is the exception-aware graph, so asking either
    /// gives the same answer for every pair of blocks.
    #[test]
    fn without_an_exceptional_entry_both_dominance_relations_are_one() {
        for ssa in [
            diamond_phi_fixture(),
            loop_counter_fixture(),
            diamond_chain_fixture(3),
        ] {
            let eh = SsaCfg::from_ssa(&ssa);
            assert!(!eh.roots().has_flow_roots());
            let dominance = EhDominance::of(&eh);
            let plain = compute_dominators(&eh.terminator_view(), NodeId::new(0));
            for def in 0..eh.node_count() {
                for used in 0..eh.node_count() {
                    let (def_node, used_node) = (NodeId::new(def), NodeId::new(used));
                    assert_eq!(
                        dominance.definition_is_well_formed(def, used),
                        dominance.dominates_block(def_node, used_node)
                            || plain.dominates(def_node, used_node),
                        "B{def} against B{used}"
                    );
                    assert_eq!(
                        dominance.dominates_block(def_node, used_node),
                        plain.dominates(def_node, used_node),
                        "B{def} against B{used}"
                    );
                }
            }
        }
    }

    /// The control for the above: with an exceptional entry the two relations
    /// part, and the floor still reads the normal path. The guard-less clause
    /// is entered from B0, so B1 no longer dominates the join B3 in the
    /// exception-aware graph, though it does on the normal path.
    #[test]
    fn with_an_exceptional_entry_the_floor_still_reads_the_normal_path() {
        let mut ssa = try_catch_fixture();
        let mut handlers = ssa.exception_handlers().to_vec();
        if let Some(handler) = handlers.first_mut() {
            handler.protected_range = None;
        }
        ssa.set_exception_handlers(handlers);

        let eh = SsaCfg::from_ssa(&ssa);
        assert!(eh.roots().has_flow_roots());
        let dominance = EhDominance::of(&eh);
        assert!(!dominance.dominates_block(NodeId::new(1), NodeId::new(3)));
        assert!(dominance.definition_is_well_formed(1, 3));
    }

    #[test]
    fn a_clause_naming_a_block_the_function_lacks_contributes_no_root() {
        let mut ssa = try_catch_fixture();
        let mut handlers = ssa.exception_handlers().to_vec();
        if let Some(handler) = handlers.first_mut() {
            handler.handler_range = BlockRange::new(99, 100);
        }
        ssa.set_exception_handlers(handlers);

        let roots = FunctionRoots::of(&ssa);
        assert!(roots.flow_roots().is_empty());
        assert!(!roots.has_flow_roots());
    }

    #[test]
    fn a_preservation_root_is_found_by_its_leading_op() {
        let ssa = try_catch_fixture();
        let roots = FunctionRoots::of(&ssa);
        // Nothing in this fixture begins with EndFinally or Rethrow.
        assert_eq!(roots.preservation_roots().count(), 0);
    }
}
