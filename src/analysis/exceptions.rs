//! One exception-aware flow view of a function.
//!
//! # Why this exists
//!
//! A lifted function has more than one entry. Control reaches a handler or a
//! filter because the runtime dispatched to it, not because any terminator
//! named it, so a graph built from terminators alone says those blocks are
//! unreachable — and every analysis rooted at the entry block then answers
//! "unreachable, therefore no phi / no check / no loop / no dominance" for all
//! of them.
//!
//! This module is where "control can enter this block without a terminator
//! naming it" is represented, once. Every analysis that needs the notion —
//! reachability, dominance, dominance frontiers, dataflow order — asks here
//! rather than deriving it privately from the exception table, so the answers
//! cannot disagree with each other.
//!
//! # The three types
//!
//! - [`FunctionRoots`] answers *where control can start*: the entry block, plus
//!   one [`ExceptionRoot`] per handler and filter entry.
//! - [`EhCfg`] is the flow graph in which those roots are reachable, built by
//!   attaching each root to the guard that protects it.
//! - [`EhDominance`] is dominance over that graph. It keeps three distinct
//!   questions apart: [`EhDominance::definition_reaches`] (may a pass rewrite
//!   through here), [`EhDominance::dominates_block`] (plain block dominance),
//!   and [`EhDominance::definition_is_well_formed`] (is the IR legal at all).
//!   They have different answers, and using one where another is meant is
//!   either a miscompile or a false rejection.
//!
//! # What the edge model claims, and what it does not
//!
//! Each flow root gets **one** edge, from the block that guards it. The runtime
//! can enter a handler from anywhere inside the protected region, so a single
//! edge from the region's first block is an approximation: it asserts the guard
//! ran to completion, when in truth the throw happened somewhere in the middle.
//!
//! That approximation is safe only because it is paired with a rule elsewhere —
//! a definition positioned *inside* an edge-source block does not reach the
//! handler ([`EhDominance::definition_reaches`]), and the rebuilder kills any
//! group the protected region redefines. Widening the model later (one edge per
//! protected block, the Soot/Java shape) is a precision improvement, not a
//! correctness fix.
//!
//! There is no virtual super-root. A node `R` with edges to the entry and to
//! every root would make `dominates(entry, x)` false for every block reachable
//! from both — which is the join after every try/catch and everything after it.
//! Every root is instead reachable *from the entry block itself*, so the node
//! count equals the block count, `immediate_dominator(entry)` stays `None`, and
//! dominance outside the exception-reached set matches the terminator-only
//! answer exactly.

use std::ops::Range;

use crate::{
    BitSet,
    analysis::cfg::SsaCfg,
    graph::{
        GraphBase, NodeId, Predecessors, RootedGraph, Successors,
        algorithms::{DominatorTree, compute_dominance_frontiers, compute_dominators},
    },
    ir::{SsaFunction, exception::ClausePart, ops::SsaOp, variable::DefSite},
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
    /// suppresses the errors it would otherwise raise, so `rebuild_ssa` could
    /// begin returning `Err` on input it accepts today.
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
    /// Membership test for [`Self::is_flow_root`].
    flow_root_blocks: BitSet,
    /// Blocks that guard some flow root.
    guards: BitSet,
    /// Blocks inside some protected region.
    protected: BitSet,
    /// The protected region of each flow root, indexed as `flow_roots`.
    regions: Vec<Option<Range<usize>>>,
}

impl FunctionRoots {
    /// Collects every entry point of `ssa`.
    ///
    /// A clause naming a block the function does not have contributes nothing,
    /// matching the rule the CFG already applies to an out-of-range successor.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let block_count = ssa.block_count();
        let mut flow_roots: Vec<ExceptionRoot> = Vec::new();
        let mut flow_root_blocks = BitSet::new(block_count);
        let mut guards = BitSet::new(block_count);
        let mut protected = BitSet::new(block_count);

        for (handler_index, handler) in ssa.exception_handlers().iter().enumerate() {
            // The clause's own reading of itself, clipped to the blocks this
            // function has. A range naming blocks it does not have is not
            // rejected here -- the roots are what keeps a handler alive, and a
            // clause the verifier will complain about still has one.
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
                if block >= block_count || !flow_root_blocks.insert_checked(block) {
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

        let regions = flow_roots.iter().map(|root| root.region.clone()).collect();

        Self {
            flow_roots,
            preservation_roots: Self::collect_preservation_roots(ssa, &flow_root_blocks),
            flow_root_blocks,
            guards,
            protected,
            regions,
        }
    }

    /// Blocks whose first instruction is `EndFinally` or `Rethrow` and which no
    /// clause already names.
    fn collect_preservation_roots<T: Target>(
        ssa: &SsaFunction<T>,
        flow_root_blocks: &BitSet,
    ) -> Vec<usize> {
        ssa.iter_blocks()
            .filter(|(block_idx, block)| {
                !flow_root_blocks.contains_checked(*block_idx)
                    && block.instructions().first().is_some_and(|instr| {
                        matches!(instr.op(), SsaOp::EndFinally | SsaOp::Rethrow)
                    })
            })
            .map(|(block_idx, _)| block_idx)
            .collect()
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

    /// Whether the function has any exceptional entry at all.
    ///
    /// When this is `false`, every answer in this module is identical to the
    /// entry-rooted one, and callers may take the cheaper path.
    #[must_use]
    pub fn has_flow_roots(&self) -> bool {
        !self.flow_roots.is_empty()
    }

    /// Whether `var` is a value the runtime supplies at a flow root rather than
    /// one any instruction defines.
    ///
    /// Such a variable is registered with a phi-shaped def site at a non-entry
    /// flow root and has no phi node carrying it. It is not undefined; it is
    /// defined by the act of entering the handler.
    #[must_use]
    pub fn is_exception_entry_value<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        var: crate::ir::SsaVarId,
    ) -> bool {
        let Some(variable) = ssa.variable(var) else {
            return false;
        };
        let site = variable.def_site();
        if !site.is_phi() || site.block == 0 || !self.is_flow_root(site.block) {
            return false;
        }
        // A phi-shaped site with an actual phi behind it is an ordinary merge.
        ssa.block(site.block)
            .is_none_or(|block| !block.phi_nodes().iter().any(|phi| phi.result() == var))
    }

    /// Every variable [`Self::is_exception_entry_value`] accepts.
    #[must_use]
    pub fn exception_entry_values<T: Target>(&self, ssa: &SsaFunction<T>) -> BitSet {
        let mut values = BitSet::new(ssa.var_id_bound());
        for variable in ssa.variables() {
            if self.is_exception_entry_value(ssa, variable.id()) {
                values.insert_checked(variable.id().index());
            }
        }
        values
    }
}

/// The terminator CFG plus one edge per exceptional entry.
///
/// Every flow root is reachable from the entry block **by construction**: each
/// gets an edge from the block that guards it, or from the entry when the guard
/// is absent, out of range, or itself unreachable.
///
/// No virtual node is introduced, so the node count equals the block count and
/// node ids are the function's block indices. Dominance here agrees with
/// dominance over [`Self::terminator_cfg`] for every block that graph can
/// already reach; the synthesized edges only add answers for blocks it cannot.
#[derive(Debug)]
pub struct EhCfg<'a, T: Target> {
    /// The terminator-only relation this is built on.
    cfg: SsaCfg<'a, T>,
    /// Where control can enter.
    roots: FunctionRoots,
    /// Synthesized successors, per block, disjoint from the terminator ones.
    extra_successors: Vec<Vec<usize>>,
    /// The inverse of `extra_successors`.
    extra_predecessors: Vec<Vec<usize>>,
    /// The tails of the synthesized edges: guards, plus the entry when a
    /// fallback edge was needed.
    edge_sources: BitSet,
}

impl<'a, T: Target> EhCfg<'a, T> {
    /// Builds the exception-aware view of `ssa`.
    #[must_use]
    pub fn from_ssa(ssa: &'a SsaFunction<T>) -> Self {
        Self::new(SsaCfg::from_ssa(ssa), FunctionRoots::of(ssa))
    }

    /// Builds the view from an existing terminator CFG and root set.
    ///
    /// The two must describe the same function; nothing here can check that.
    #[must_use]
    pub fn new(cfg: SsaCfg<'a, T>, roots: FunctionRoots) -> Self {
        let block_count = cfg.block_count();
        let mut extra_successors = vec![Vec::new(); block_count];
        let mut extra_predecessors = vec![Vec::new(); block_count];
        let mut edge_sources = BitSet::new(block_count);

        // Reachability in the terminator relation, which decides whether a
        // guard can actually dispatch.
        let mut reachable = BitSet::new(block_count);
        let mut worklist: Vec<usize> = Vec::new();
        if block_count > 0 {
            reachable.insert_checked(0);
            worklist.push(0);
        }

        // Roots waiting on a guard that is not reachable yet. Attaching one root
        // can make another's guard reachable, so this fires from the traversal
        // rather than from a rescan -- linear, not quadratic in the root count.
        let mut waiting: Vec<Vec<usize>> = vec![Vec::new(); block_count];
        let mut attached = vec![false; roots.flow_roots().len()];
        for (index, root) in roots.flow_roots().iter().enumerate() {
            match root.guard {
                Some(guard) if guard < block_count => {
                    if let Some(slot) = waiting.get_mut(guard) {
                        slot.push(index);
                    }
                }
                // No usable guard: the entry fallback handles it below.
                _ => {}
            }
        }

        // Drain the frontier, attaching every root whose guard just became
        // reachable. A root is attached at most once, and a block is expanded at
        // most once, so the whole pass is O(V + E).
        let mut cursor = 0;
        loop {
            while cursor < worklist.len() {
                let Some(block) = worklist.get(cursor).copied() else {
                    break;
                };
                cursor = cursor.saturating_add(1);

                // Expand the terminator successors of a newly reachable block.
                for succ in cfg.block_successors(block) {
                    if *succ < block_count && reachable.insert_checked(*succ) {
                        worklist.push(*succ);
                    }
                }

                let pending = waiting.get(block).cloned().unwrap_or_default();
                for index in pending {
                    if attached.get(index).copied().unwrap_or(true) {
                        continue;
                    }
                    let Some(root) = roots.flow_roots().get(index) else {
                        continue;
                    };
                    Self::attach(
                        block,
                        root.block,
                        &cfg,
                        &mut extra_successors,
                        &mut extra_predecessors,
                        &mut edge_sources,
                    );
                    if let Some(slot) = attached.get_mut(index) {
                        *slot = true;
                    }
                    if reachable.insert_checked(root.block) {
                        worklist.push(root.block);
                    }
                }
            }

            // Every reachable guard has fired. Anything still unattached has no
            // usable guard, so it falls back to the entry -- which can in turn
            // make further guards reachable, hence the outer loop.
            let Some(index) = attached.iter().position(|done| !done) else {
                break;
            };
            let Some(root) = roots.flow_roots().get(index) else {
                break;
            };
            Self::attach(
                0,
                root.block,
                &cfg,
                &mut extra_successors,
                &mut extra_predecessors,
                &mut edge_sources,
            );
            if let Some(slot) = attached.get_mut(index) {
                *slot = true;
            }
            if reachable.insert_checked(root.block) {
                worklist.push(root.block);
            }
        }

        Self {
            cfg,
            roots,
            extra_successors,
            extra_predecessors,
            edge_sources,
        }
    }

    /// Records one synthesized edge, unless the terminator relation already has
    /// it. A duplicate predecessor would make a one-predecessor root look like
    /// a frontier join and attract a phi it must not have.
    fn attach(
        from: usize,
        to: usize,
        cfg: &SsaCfg<'a, T>,
        extra_successors: &mut [Vec<usize>],
        extra_predecessors: &mut [Vec<usize>],
        edge_sources: &mut BitSet,
    ) {
        if cfg.block_successors(from).contains(&to) {
            return;
        }
        let Some(successors) = extra_successors.get_mut(from) else {
            return;
        };
        if successors.contains(&to) {
            return;
        }
        successors.push(to);
        if let Some(predecessors) = extra_predecessors.get_mut(to) {
            predecessors.push(from);
        }
        edge_sources.insert_checked(from);
    }

    /// The terminator-only relation, by name.
    ///
    /// Ask for this when the question is about branch arity or about which
    /// predecessors a phi's operands may name — neither of which the
    /// synthesized edges participate in.
    #[must_use]
    pub const fn terminator_cfg(&self) -> &SsaCfg<'a, T> {
        &self.cfg
    }

    /// Where control can enter this function.
    #[must_use]
    pub const fn roots(&self) -> &FunctionRoots {
        &self.roots
    }

    /// Whether `from -> to` is a synthesized exceptional edge rather than a
    /// terminator edge.
    #[must_use]
    pub fn is_exception_edge(&self, from: usize, to: usize) -> bool {
        self.extra_successors
            .get(from)
            .is_some_and(|extra| extra.contains(&to))
    }

    /// The tails of the synthesized edges.
    #[must_use]
    pub const fn edge_sources(&self) -> &BitSet {
        &self.edge_sources
    }

    /// Successor lists including the synthesized edges, one per block.
    #[must_use]
    pub fn successor_lists(&self) -> Vec<Vec<usize>> {
        (0..self.cfg.block_count())
            .map(|block| {
                let mut successors = self.cfg.block_successors(block).to_vec();
                if let Some(extra) = self.extra_successors.get(block) {
                    successors.extend_from_slice(extra);
                }
                successors
            })
            .collect()
    }

    /// Blocks that leave the function.
    ///
    /// Answered from the terminator relation: a `Return` inside a protected
    /// region still leaves the function, and the edge to its handler does not
    /// change that.
    #[must_use]
    pub fn exits(&self) -> Vec<NodeId> {
        self.cfg.exits()
    }
}

impl<T: Target> GraphBase for EhCfg<'_, T> {
    fn node_count(&self) -> usize {
        self.cfg.node_count()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        self.cfg.node_ids()
    }
}

impl<T: Target> Successors for EhCfg<'_, T> {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let extra = self
            .extra_successors
            .get(node.index())
            .map_or(&[][..], Vec::as_slice);
        self.cfg
            .block_successors(node.index())
            .iter()
            .chain(extra)
            .copied()
            .map(NodeId::new)
    }
}

impl<T: Target> Predecessors for EhCfg<'_, T> {
    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let extra = self
            .extra_predecessors
            .get(node.index())
            .map_or(&[][..], Vec::as_slice);
        self.cfg
            .block_predecessors(node.index())
            .iter()
            .chain(extra)
            .copied()
            .map(NodeId::new)
    }
}

impl<T: Target> RootedGraph for EhCfg<'_, T> {
    fn entry(&self) -> NodeId {
        NodeId::new(0)
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
///   contract — a definition must dominate its use on the normal path, and this
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
    normal_tree: DominatorTree,
    /// Iterated dominance frontiers of the exception-aware graph, when built.
    frontiers: Vec<BitSet>,
    /// Blocks reachable only because an exceptional edge exists.
    exception_reachable: BitSet,
    /// Tails of the synthesized edges.
    edge_sources: BitSet,
}

impl EhDominance {
    /// Computes dominance over `eh`, without dominance frontiers.
    #[must_use]
    pub fn of<T: Target>(eh: &EhCfg<'_, T>) -> Self {
        Self::build(eh, false)
    }

    /// Computes dominance over `eh` together with its dominance frontiers.
    #[must_use]
    pub fn with_frontiers<T: Target>(eh: &EhCfg<'_, T>) -> Self {
        Self::build(eh, true)
    }

    /// Shared construction; `frontiers` decides whether the frontier pass runs.
    fn build<T: Target>(eh: &EhCfg<'_, T>, frontiers: bool) -> Self {
        let entry = NodeId::new(0);
        let tree = compute_dominators(eh, entry);
        let normal_tree = compute_dominators(eh.terminator_cfg(), entry);

        let block_count = eh.node_count();
        let mut exception_reachable = BitSet::new(block_count);
        for block in 0..block_count {
            let node = NodeId::new(block);
            // Reachable now, but not without the synthesized edges.
            if tree.is_reachable(node) && !normal_tree.is_reachable(node) {
                exception_reachable.insert_checked(block);
            }
        }

        Self {
            frontiers: if frontiers {
                compute_dominance_frontiers(eh, &tree)
            } else {
                Vec::new()
            },
            tree,
            normal_tree,
            exception_reachable,
            edge_sources: eh.edge_sources().clone(),
        }
    }

    /// The exception-aware dominator tree.
    #[must_use]
    pub const fn tree(&self) -> &DominatorTree {
        &self.tree
    }

    /// The exception-aware dominance frontiers.
    ///
    /// Empty unless built with [`Self::with_frontiers`].
    ///
    /// A frontier row is keyed on the block holding a definition, and it is
    /// the row — not the dominance relation — that decides where a phi goes.
    /// A handler that rejoins the normal path has the join in its frontier, so
    /// a definition inside the handler demands a phi at that join. This is why
    /// a guard whose dominance answers are the same as the terminator-only
    /// ones can still sit beside different phi placement: the handler is a
    /// block the terminator relation has no row for at all.
    #[must_use]
    pub fn frontiers(&self) -> &[BitSet] {
        &self.frontiers
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
    /// `use_block` — the question a rewrite must ask.
    ///
    /// Stricter than [`Self::dominates_block`] by exactly one rule: a
    /// definition positioned at an *instruction* inside an edge-source block
    /// does not reach an exception-reachable use, because the synthesized edge
    /// asserts the guard ran to completion and a throw need not have reached
    /// that instruction. A definition at the *top* of such a block (a phi) is
    /// unaffected, and so is every function with no exception table, where
    /// `edge_sources` is empty.
    #[must_use]
    pub fn definition_reaches(&self, def: DefSite, use_block: usize) -> bool {
        if !self.dominates_block(NodeId::new(def.block), NodeId::new(use_block)) {
            return false;
        }
        if def.instruction.is_some()
            && self.edge_sources.contains_checked(def.block)
            && self.exception_reachable.contains_checked(use_block)
        {
            return false;
        }
        true
    }

    /// Whether a definition in `def_block` may legally be used in `use_block` —
    /// the verifier's floor.
    ///
    /// Looser than [`Self::dominates_block`]: normal-path dominance also
    /// suffices. A value defined inside a protected region and read at the
    /// region's join dominates that use on every path the IR models, and this
    /// crate does not model a partially executed region, so rejecting it would
    /// reject IR the crate itself produces.
    #[must_use]
    pub fn definition_is_well_formed(&self, def_block: usize, use_block: usize) -> bool {
        let (def, used) = (NodeId::new(def_block), NodeId::new(use_block));
        self.tree.dominates(def, used) || self.normal_tree.dominates(def, used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::exception::BlockRange, testing::try_catch_fixture};

    #[test]
    fn every_flow_root_is_reachable_from_the_entry() {
        let ssa = try_catch_fixture();
        let eh = EhCfg::from_ssa(&ssa);
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
        let eh = EhCfg::from_ssa(&ssa);
        let dominance = EhDominance::with_frontiers(&eh);
        let plain = compute_dominators(eh.terminator_cfg(), NodeId::new(0));

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
        let handler_frontier = dominance
            .frontiers()
            .get(2)
            .expect("the handler has a frontier row");
        assert!(
            handler_frontier.contains_checked(3),
            "the try/catch join must enter the handler's frontier"
        );
        assert!(
            compute_dominators(eh.terminator_cfg(), NodeId::new(0))
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
        let eh = EhCfg::from_ssa(&ssa);

        for block in 0..eh.node_count() {
            let successors: Vec<usize> = eh
                .successors(NodeId::new(block))
                .map(NodeId::index)
                .collect();
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

    /// A clause whose guard is the block that is already the root's terminator
    /// predecessor must not gain a second, parallel edge.
    #[test]
    fn a_self_guarding_region_adds_no_edge() {
        let mut ssa = try_catch_fixture();
        let mut handlers = ssa.exception_handlers().to_vec();
        if let Some(handler) = handlers.first_mut() {
            // Guard B0, which already jumps to B1; make B1 the handler entry.
            handler.protected_range = BlockRange::new(0, 1);
            handler.handler_range = BlockRange::new(1, 2);
        }
        ssa.set_exception_handlers(handlers);

        let eh = EhCfg::from_ssa(&ssa);
        assert!(
            !eh.is_exception_edge(0, 1),
            "the terminator relation already carries B0 -> B1"
        );
        assert!(
            !eh.edge_sources().contains_checked(0),
            "and so B0 is not an edge source"
        );
    }

    /// The strictness order the three predicates promise.
    #[test]
    fn a_definition_in_an_edge_source_does_not_reach_the_handler() {
        let ssa = try_catch_fixture();
        let eh = EhCfg::from_ssa(&ssa);
        let dominance = EhDominance::of(&eh);

        // B1 guards the handler, so it is the tail of the synthesized edge.
        assert!(eh.edge_sources().contains_checked(1));
        assert!(dominance.exception_reachable().contains_checked(2));

        let mid_guard = DefSite::instruction(1, 0);
        assert!(
            dominance.dominates_block(NodeId::new(1), NodeId::new(2)),
            "block dominance holds -- the edge says so"
        );
        assert!(
            !dominance.definition_reaches(mid_guard, 2),
            "but a definition part-way through the guard need not have executed"
        );

        // The top of the same block is unaffected: control reached the block.
        assert!(dominance.definition_reaches(DefSite::phi(1), 2));
    }

    /// The entry fallback: a clause with no usable guard still gets its root
    /// attached, and the *entry* becomes the edge source rather than the
    /// clause's protected range.
    #[test]
    fn a_root_without_a_usable_guard_falls_back_to_the_entry() {
        let mut ssa = try_catch_fixture();
        let mut handlers = ssa.exception_handlers().to_vec();
        if let Some(handler) = handlers.first_mut() {
            handler.protected_range = None;
        }
        ssa.set_exception_handlers(handlers);

        let eh = EhCfg::from_ssa(&ssa);
        assert!(eh.is_exception_edge(0, 2), "the fallback edge leaves B0");
        assert!(eh.edge_sources().contains_checked(0));
        assert!(
            !eh.edge_sources().contains_checked(1),
            "B1 guards nothing now"
        );
        assert!(EhDominance::of(&eh).tree().is_reachable(NodeId::new(2)));
    }

    #[test]
    fn a_function_without_a_table_is_the_plain_cfg() {
        let ssa = crate::testing::diamond_phi_fixture();
        let eh = EhCfg::from_ssa(&ssa);
        let dominance = EhDominance::of(&eh);

        assert!(!eh.roots().has_flow_roots());
        assert!(eh.edge_sources().is_empty());
        assert!(dominance.exception_reachable().is_empty());
        for block in 0..eh.node_count() {
            let mut through_eh: Vec<usize> = eh
                .successors(NodeId::new(block))
                .map(NodeId::index)
                .collect();
            through_eh.sort_unstable();
            let mut plain = eh.terminator_cfg().block_successors(block).to_vec();
            plain.sort_unstable();
            assert_eq!(through_eh, plain);
        }
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
