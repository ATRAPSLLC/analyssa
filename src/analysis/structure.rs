//! Structured control-flow recovery: an arbitrary CFG to a region tree.
//!
//! A decompiler's back end cannot print a graph. It needs a tree of statements —
//! sequences, `if`/`else`, loops, `switch` — and this module recovers one from
//! any control-flow graph the front end produces.
//!
//! # Totality
//!
//! The recovery is **total**: every graph yields a region tree that reproduces
//! its control flow exactly. There is no input for which structuring fails, and
//! no input for which it silently drops an edge. That is not a claim about
//! compiler output being well behaved — hand-written assembly, obfuscators,
//! `setjmp`, computed gotos and self-modifying dispatch all produce graphs no
//! `while` loop describes. Totality comes instead from [`Region::Goto`]: any
//! edge that no structured form can express becomes an explicit jump to a
//! labelled block, which is always available and always correct.
//!
//! Totality is quantified over three things, and each is established rather
//! than assumed:
//!
//! - **The index domain.** Every caller-supplied block set is a [`BlockSet`],
//!   which carries the node count it was built for and answers `contains` for
//!   any node at all. [`structure_with`] renormalises a set built for a
//!   different width once, at the boundary, and counts the renormalisation in
//!   [`StructureMetrics::resized_inputs`]; it also rejects an entry outside the
//!   graph, and drops successor ids the graph reports past its own node count.
//! - **The depth of the output.** No tree this module returns nests deeper than
//!   [`StructureOptions::max_depth`]; see [`Structured`].
//! - **The scope a decision belongs to.** Every walk that enters a region —
//!   an arm, a loop body, a protected body, a handler, a filter — enters it
//!   through one scope operation, so the confinement, the enclosing follow and
//!   the loop-frame barrier are decided together and released together.
//!
//! Quality is therefore a *measurement*, not a precondition. The output is
//! better when it uses fewer gotos, and [`StructureMetrics::gotos`] counts them,
//! but a high count is ugly rather than wrong.
//!
//! # How it works
//!
//! The recovery walks the graph forward from the entry, keeping three facts:
//!
//! - **First arrival**, to know which block owns which region. A block is
//!   placed where the walk first reaches it, and every later arrival is a
//!   transfer to that placement.
//! - **Post-dominance**, to find the *join* of a conditional — the first block
//!   both arms reach. That join bounds the arms and resumes the parent
//!   sequence, which is what turns a diamond into `if`/`else` instead of two
//!   gotos. See [`compute_post_dominators`].
//! - **Natural loops**, to know which back edges are `continue` and which exit
//!   edges are `break`.
//!
//! Any block reached a second time has already been emitted, so it becomes a
//! goto. This single rule is what makes irreducible control flow terminate:
//! a cycle with two entries has no back edge dominating it, so no loop is
//! detected, the second entry is simply a revisit, and the cycle prints as a
//! labelled block with a goto.
//!
//! # Loops
//!
//! A natural loop is classified by *where it tests*, which is a structural
//! question and not the [`LoopType`](super::loops::LoopType) the loop analysis
//! records for canonicalisation:
//!
//! - the header branches out of the body — [`StructuredLoop::While`]
//! - a single latch branches out — [`StructuredLoop::DoWhile`]
//! - neither — [`StructuredLoop::Endless`], exited by `break` (or not at all)
//!
//! A loop with several latches is still one loop: every latch becomes a
//! `continue`. A loop leaving to several different blocks keeps one of them as
//! the follow — the block control resumes at — and reaches the others by goto.
//!
//! # Labelled transfers
//!
//! [`Region::Break`] and [`Region::Continue`] both name the loop header they
//! act on, rather than implying the innermost loop. A break out of two nested
//! loops is expressible in the tree; whether the printer renders it as a
//! labelled break or falls back to a goto is the printer's decision, made with
//! its target language in view, and is not lost here.

use std::collections::{BTreeMap, BTreeSet};

/// The crate's handler taxonomy, defined beside the exception table it comes
/// from and re-exported here so a caller reading a recovered [`HandlerRegion`]
/// finds it alongside.
///
/// One definition, not a recovery-side copy: two enums with the same four
/// variants can disagree about a clause, and nothing would catch it.
pub use crate::ir::exception::HandlerKind;
use crate::{
    analysis::loops::{LoopForest, detect_loops},
    bitset::BitSet,
    graph::{
        GraphBase, NodeId, Predecessors, Successors,
        algorithms::{PostDominatorTree, compute_dominators, compute_post_dominators},
    },
};

/// Default nesting bound for a recovered tree; see [`StructureOptions::max_depth`].
pub const DEFAULT_MAX_DEPTH: usize = 256;

/// A set of blocks over a stated index domain.
///
/// The domain is `0..node_count`, and it is part of the value rather than a
/// precondition on the reader: [`BlockSet::contains`] answers for any
/// [`NodeId`] at all, and a node outside the domain is simply not a member.
/// That is what lets a caller hand a set to [`structure_with`] without knowing
/// which of the ten places inside the walk will read it.
///
/// A set whose `node_count` disagrees with the graph's is renormalised once, at
/// the [`structure_with`] boundary, and the renormalisation is counted in
/// [`StructureMetrics::resized_inputs`]. A non-zero count means a caller built
/// a set for a different function, which is a defect in the caller — the
/// recovery still answers, conservatively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSet {
    /// Size of the index domain: members lie in `0..node_count`.
    node_count: usize,

    /// Membership, exactly `node_count` bits wide.
    bits: BitSet,
}

impl BlockSet {
    /// Returns the empty set over `0..node_count`.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            bits: BitSet::new(node_count),
        }
    }

    /// Returns the set of every block in `0..node_count`.
    #[must_use]
    pub fn full(node_count: usize) -> Self {
        Self {
            node_count,
            bits: BitSet::full(node_count),
        }
    }

    /// Returns the set of `nodes` over `0..node_count`.
    ///
    /// A node outside the domain is dropped, which is the same answer
    /// [`BlockSet::insert`] gives it.
    #[must_use]
    pub fn from_nodes(node_count: usize, nodes: impl IntoIterator<Item = NodeId>) -> Self {
        let mut set = Self::new(node_count);
        for node in nodes {
            set.insert(node);
        }
        set
    }

    /// Returns `bits` read as a block set over `0..node_count`.
    ///
    /// Zero-extending when `bits` is narrower and truncating when it is wider,
    /// so the result always has exactly the stated domain.
    #[must_use]
    pub fn from_bits(node_count: usize, bits: BitSet) -> Self {
        if bits.len() == node_count {
            return Self { node_count, bits };
        }
        Self::from_nodes(node_count, bits.iter().map(NodeId::new))
    }

    /// Returns the size of the index domain.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns whether `node` is a member.
    ///
    /// Total: a node outside the domain is not a member.
    #[must_use]
    pub fn contains(&self, node: NodeId) -> bool {
        self.bits.contains_checked(node.index())
    }

    /// Adds `node` and returns whether it was absent.
    ///
    /// A node outside the domain is not added, and the answer is `false`.
    pub fn insert(&mut self, node: NodeId) -> bool {
        self.bits.insert_checked(node.index())
    }

    /// Returns the members in ascending index order.
    pub fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.bits.iter().map(NodeId::new)
    }

    /// Returns the number of members.
    #[must_use]
    pub fn count(&self) -> usize {
        self.bits.count()
    }
}

/// How a recovered loop tests its continuation condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructuredLoop {
    /// Tested before the body: the header itself branches out of the loop.
    While,

    /// Tested after the body: a single latch branches out of the loop.
    DoWhile,

    /// Not tested at a header or latch. Left by `break`, or never left.
    Endless,
}

/// A condition, possibly spanning several blocks.
///
/// A machine has no `&&`. Source that reads `if (a && b)` compiles to two
/// branches, and recovering it as two nested conditionals is not merely uglier:
/// the block both false arms lead to then has two predecessors and only one
/// place in the tree, so the second reaches it by `goto`. Measured over the
/// committed fixtures, that shape is the **largest single source of gotos** —
/// 63% to 72% of them target a block with exactly two predecessors.
///
/// Recovering the condition as a tree removes the cause rather than the
/// symptom.
///
/// Nesting is bounded. Each fold alternating between [`Predicate::All`] and
/// [`Predicate::Any`] adds one level, and the recovery performs at most
/// [`StructureOptions::max_depth`] folds per condition, so every recursive
/// traversal of a predicate this module produces — including the derived
/// `Drop`, `Clone` and `PartialEq` — is bounded by that number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Predicate {
    /// The branch at `block`, negated when its first successor is the arm
    /// *not* taken when the predicate holds.
    Test {
        /// Block whose terminator performs the test.
        block: NodeId,

        /// Whether the test reads inverted.
        negated: bool,
    },

    /// Every part holds. Parts are evaluated in order and later ones only when
    /// the earlier ones held, which is what the control flow does.
    All(Vec<Predicate>),

    /// Any part holds, with the same short-circuit order.
    Any(Vec<Predicate>),
}

impl Predicate {
    /// Calls `visit` for every block that takes part in the condition.
    pub fn for_each_block(&self, visit: &mut impl FnMut(NodeId)) {
        match self {
            Predicate::Test { block, .. } => visit(*block),
            Predicate::All(parts) | Predicate::Any(parts) => {
                for part in parts {
                    part.for_each_block(visit);
                }
            }
        }
    }

    /// Returns the condition that holds exactly when this one does not.
    ///
    /// De Morgan rather than a wrapper, so a negated `a && b` reads as
    /// `!a || !b` and keeps its short-circuit order.
    #[must_use]
    pub fn negate(&self) -> Self {
        match self {
            Predicate::Test { block, negated } => Predicate::Test {
                block: *block,
                negated: !negated,
            },
            Predicate::All(parts) => Predicate::Any(parts.iter().map(Predicate::negate).collect()),
            Predicate::Any(parts) => Predicate::All(parts.iter().map(Predicate::negate).collect()),
        }
    }

    /// Returns the block the condition starts at.
    ///
    /// Evaluation order is the order the parts appear, so this is the block
    /// control reaches first — the one whose statements run whatever the
    /// condition turns out to be.
    #[must_use]
    pub fn head(&self) -> Option<NodeId> {
        match self {
            Predicate::Test { block, .. } => Some(*block),
            Predicate::All(parts) | Predicate::Any(parts) => {
                parts.first().and_then(Predicate::head)
            }
        }
    }

    /// Returns the number of tests the condition performs.
    #[must_use]
    pub fn tests(&self) -> usize {
        match self {
            Predicate::Test { .. } => 1,
            Predicate::All(parts) | Predicate::Any(parts) => {
                parts.iter().map(Predicate::tests).sum()
            }
        }
    }
}

/// A recovered `if`/`else`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfRegion {
    /// The condition, which may span several blocks.
    pub predicate: Predicate,

    /// Statements executed when the condition holds.
    pub then_branch: Region,

    /// Statements executed otherwise, if that arm is not empty.
    pub else_branch: Option<Region>,
}

/// A recovered loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopRegion {
    /// Header block: the single entry, which dominates the whole body.
    pub header: NodeId,

    /// Where the continuation condition is tested.
    pub kind: StructuredLoop,

    /// The condition under which the loop continues.
    ///
    /// Tested at the header for [`StructuredLoop::While`] and at the latch for
    /// [`StructuredLoop::DoWhile`]. `None` for [`StructuredLoop::Endless`],
    /// which tests nowhere a loop form can express.
    pub predicate: Option<Predicate>,

    /// The loop body.
    pub body: Region,

    /// Block control resumes at after the loop, if the loop is left at all.
    pub follow: Option<NodeId>,
}

/// One arm of a recovered multi-way branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchCase {
    /// Successor indices of the switch head that select this arm.
    ///
    /// Several indices share one arm when several case values branch to the
    /// same block, which is how `case 1: case 2:` reaches the front end.
    pub selectors: Vec<usize>,

    /// First block of the arm.
    pub target: NodeId,

    /// Statements of the arm.
    pub body: Region,
}

/// A recovered multi-way branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchRegion {
    /// Block whose terminator performs the dispatch.
    pub head: NodeId,

    /// The arms, ordered by their lowest selector index.
    pub cases: Vec<SwitchCase>,

    /// Block control resumes at after the switch, if the arms rejoin.
    pub follow: Option<NodeId>,
}

/// One handler attached to a protected region, as recovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandlerRegion {
    /// What the handler does.
    pub kind: HandlerKind,

    /// First block of the handler body.
    pub entry: NodeId,

    /// The filter expression, for [`HandlerKind::Filter`].
    pub filter: Option<Region>,

    /// The handler body.
    pub body: Region,
}

/// A recovered protected region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TryRegion {
    /// First block of the protected body.
    pub entry: NodeId,

    /// The protected body.
    pub body: Region,

    /// Handlers attached to the region, in declaration order.
    pub handlers: Vec<HandlerRegion>,

    /// Block control resumes at when the body completes normally.
    pub follow: Option<NodeId>,
}

/// The filter expression of a filter handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandlerFilter {
    /// First block of the filter expression.
    pub entry: NodeId,

    /// Blocks belonging to the filter expression.
    pub blocks: BlockSet,
}

/// What a handler to recover does, carrying the filter where it has one.
///
/// Split from [`HandlerKind`], which describes a *recovered* handler, because
/// the input has one more thing to say and one fewer way to say it wrongly: a
/// filter expression exists exactly for [`ProtectedHandlerKind::Filter`], so a
/// filter without an expression and an expression on a `finally` are both
/// unrepresentable rather than reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtectedHandlerKind {
    /// Runs for a matching exception type.
    Catch,

    /// Runs the given filter expression first, and the body only if it accepts.
    Filter(HandlerFilter),

    /// Runs on every exit from the region, raising or not.
    Finally,

    /// Runs only when the region raises, without catching.
    Fault,
}

impl ProtectedHandlerKind {
    /// Returns what the recovered handler does.
    #[must_use]
    pub const fn kind(&self) -> HandlerKind {
        match self {
            ProtectedHandlerKind::Catch => HandlerKind::Catch,
            ProtectedHandlerKind::Filter(_) => HandlerKind::Filter,
            ProtectedHandlerKind::Finally => HandlerKind::Finally,
            ProtectedHandlerKind::Fault => HandlerKind::Fault,
        }
    }

    /// Returns the filter expression, for a filter handler.
    #[must_use]
    pub const fn filter(&self) -> Option<&HandlerFilter> {
        match self {
            ProtectedHandlerKind::Filter(filter) => Some(filter),
            ProtectedHandlerKind::Catch
            | ProtectedHandlerKind::Finally
            | ProtectedHandlerKind::Fault => None,
        }
    }
}

/// A handler to recover, described by the blocks it occupies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedHandler {
    /// What the handler does, and its filter expression where it has one.
    pub kind: ProtectedHandlerKind,

    /// First block of the handler body.
    pub entry: NodeId,

    /// Blocks belonging to the handler body.
    pub blocks: BlockSet,
}

/// A protected region to recover, described by the blocks it occupies.
///
/// Ranges come from the binary's own exception tables, which is why they are an
/// input rather than something structuring infers: the blocks a `try` covers are
/// recorded by the compiler and cannot be recovered from control flow, because
/// the edge into a handler is taken by the runtime and appears nowhere in the
/// code.
///
/// Two regions may share an entry block — `try { try {} catch {} } finally {}`
/// is the ordinary ECMA-335 and JVM encoding of nested clauses — and both open.
/// The one covering more blocks opens outside the other; ties resolve so that
/// the later-declared region is the outer one, matching the innermost-first
/// clause order those tables are written in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedRegion {
    /// First block of the protected body.
    pub entry: NodeId,

    /// Blocks belonging to the protected body.
    pub blocks: BlockSet,

    /// Handlers attached to this region, in declaration order.
    pub handlers: Vec<ProtectedHandler>,
}

/// A node of the recovered statement tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Region {
    /// No statements. Produced by an arm that falls straight to the join.
    Empty,

    /// The statements of one basic block.
    Block(NodeId),

    /// Statements executed in order.
    Seq(Vec<Region>),

    /// A two-way branch.
    If(Box<IfRegion>),

    /// A loop.
    Loop(Box<LoopRegion>),

    /// A multi-way branch.
    Switch(Box<SwitchRegion>),

    /// A protected region and its handlers.
    Try(Box<TryRegion>),

    /// Leave the loop with the named header.
    Break(NodeId),

    /// Begin the next iteration of the loop with the named header.
    Continue(NodeId),

    /// Transfer to a block that structuring could not reach in tree order.
    ///
    /// The target is recorded in [`Structured::labels`], so a printer knows to
    /// give it a label.
    Goto(NodeId),
}

impl Region {
    /// Calls `visit` for every block placed in this region, in printed order.
    ///
    /// A [`Region::Goto`] target is not a placement: the block is placed
    /// wherever structuring reached it, and the goto only transfers there.
    ///
    /// Recursive, and safely so: a tree this module produces nests no deeper
    /// than [`StructureOptions::max_depth`]. See [`Structured`].
    pub fn for_each_block(&self, visit: &mut impl FnMut(NodeId)) {
        match self {
            Region::Empty | Region::Break(_) | Region::Continue(_) | Region::Goto(_) => {}
            Region::Block(node) => visit(*node),
            Region::Seq(regions) => {
                for region in regions {
                    region.for_each_block(visit);
                }
            }
            Region::If(inner) => {
                inner.predicate.for_each_block(visit);
                inner.then_branch.for_each_block(visit);
                if let Some(other) = &inner.else_branch {
                    other.for_each_block(visit);
                }
            }
            Region::Loop(inner) => {
                if inner.kind == StructuredLoop::While
                    && let Some(predicate) = &inner.predicate
                {
                    predicate.for_each_block(visit);
                }
                inner.body.for_each_block(visit);
            }
            Region::Switch(inner) => {
                visit(inner.head);
                for case in &inner.cases {
                    case.body.for_each_block(visit);
                }
            }
            Region::Try(inner) => {
                inner.body.for_each_block(visit);
                for handler in &inner.handlers {
                    if let Some(filter) = &handler.filter {
                        filter.for_each_block(visit);
                    }
                    handler.body.for_each_block(visit);
                }
            }
        }
    }
}

/// Counts describing how well a graph structured.
///
/// These exist to be measured over a corpus. [`StructureMetrics::gotos`] is the
/// quality signal: it is zero for control flow that a structured language
/// expresses directly, and rises with irreducibility, multi-exit loops and
/// case fall-through.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructureMetrics {
    /// Basic blocks placed in the tree.
    pub blocks: usize,

    /// Blocks the walk never reached, and which are therefore absent from the
    /// tree. Unreachable code, not a structuring failure.
    pub unreached: usize,

    /// Gotos emitted.
    pub gotos: usize,

    /// Loops recovered.
    pub loops: usize,

    /// Loops recovered as [`StructuredLoop::Endless`].
    pub endless_loops: usize,

    /// Two-way branches recovered.
    pub ifs: usize,

    /// Two-way branches recovered with both arms non-empty.
    pub if_elses: usize,

    /// Multi-way branches recovered.
    pub switches: usize,

    /// Conditions recovered as spanning more than one block.
    pub merged_conditions: usize,

    /// Protected regions recovered.
    pub tries: usize,

    /// Handlers recovered across all protected regions.
    pub handlers: usize,

    /// `break` transfers emitted.
    pub breaks: usize,

    /// `continue` transfers emitted.
    pub continues: usize,

    /// Function-ending blocks placed more than once.
    ///
    /// A tail with no successors is repeated rather than jumped to, so this is
    /// the number of gotos that became a duplicated `return` instead. See
    /// [`Structured::blocks`] for what that means for placement.
    pub replicated_tails: usize,

    /// Caller-supplied [`BlockSet`]s renormalised at the [`structure_with`]
    /// boundary because their `node_count` disagreed with the graph's.
    ///
    /// Zero for every well-formed input. A non-zero count is the loud signal
    /// that a caller built a set for a different function: the recovery still
    /// answers, but a block the caller meant to include may have been dropped.
    pub resized_inputs: usize,

    /// Recoveries cut short by [`StructureOptions::max_depth`].
    ///
    /// Counts both the nesting bound — a walk abandoned at the limit, which
    /// leaves its blocks reachable by goto or flattened inside their protected
    /// region — and a condition whose folding stopped at the limit.
    pub depth_limited: usize,
}

/// A structured control-flow recovery.
///
/// # Depth
///
/// The recovery descends at most [`StructureOptions::max_depth`] nested
/// regions, and each descent contributes at most two levels to the tree — the
/// construct and the sequence below it — so no tree this module returns nests
/// deeper than `2 * max_depth + 4`. No [`Predicate`] it returns nests deeper
/// than `max_depth` folds either.
///
/// That bound is what makes every recursive traversal of the result safe on a
/// bounded machine stack — including the ones the caller does not write, the
/// derived `Drop`, `Clone`, `PartialEq` and `Debug`, as well as
/// [`Region::for_each_block`].
///
/// A recovery that would have exceeded the bound degrades instead: see
/// [`StructureMetrics::depth_limited`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Structured {
    /// The statement tree.
    pub root: Region,

    /// Blocks a [`Region::Goto`] targets, which need a label when printed.
    ///
    /// Exactly the set of [`Region::Goto`] targets in [`Structured::root`],
    /// and nothing else: a block the walk placed at top level because no
    /// region owned it is reached by falling into it, not by jumping, so it
    /// needs no label.
    pub labels: BTreeSet<NodeId>,

    /// Quality counts for the recovery.
    pub metrics: StructureMetrics,
}

impl Structured {
    /// Returns every block placed in the tree, in printed order.
    ///
    /// Every block reachable from the entry appears at least once, which is the
    /// invariant a caller checks to know the tree accounts for the whole graph.
    /// A block appears more than once only when it ends the function and was
    /// replicated rather than jumped to; [`StructureMetrics::replicated_tails`]
    /// counts those.
    #[must_use]
    pub fn blocks(&self) -> Vec<NodeId> {
        let mut blocks = Vec::new();
        self.root.for_each_block(&mut |node| blocks.push(node));
        blocks
    }
}

/// Recovers structured control flow for `graph` entered at `entry`.
///
/// Always succeeds, for every graph and every entry — an entry outside the
/// graph recovers as [`Region::Empty`]. See the module documentation for what
/// totality costs and how to measure it.
///
/// Every block is declared [`condition-only`](StructureOptions::condition_only):
/// a caller holding a bare graph has no instruction-level view of it, so its
/// nodes carry no statements that folding a condition or hoisting a `while`
/// header could misplace. A caller whose nodes *do* carry statements must go
/// through [`structure_with`] with the set it can justify, or the recovery may
/// move work that cannot move. A caller holding an `SsaFunction` has
/// [`structure_ssa`](crate::analysis::recovery::structure_ssa), which derives
/// that set from the instructions each block holds.
///
/// # Examples
///
/// ```rust
/// use analyssa::{analysis::structure::structure, graph::DirectedGraph};
///
/// let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
/// let head = graph.add_node(());
/// let then_arm = graph.add_node(());
/// let join = graph.add_node(());
/// graph.add_edge(head, then_arm, ()).unwrap();
/// graph.add_edge(head, join, ()).unwrap();
/// graph.add_edge(then_arm, join, ()).unwrap();
///
/// let recovered = structure(&graph, head);
/// assert_eq!(recovered.metrics.ifs, 1);
/// assert_eq!(recovered.metrics.gotos, 0);
/// ```
#[must_use]
pub fn structure<G>(graph: &G, entry: NodeId) -> Structured
where
    G: GraphBase + Successors + Predecessors,
{
    structure_protected(graph, entry, &[])
}

/// Recovers structured control flow for a graph carrying protected regions.
///
/// `graph` must contain only the edges its terminators take, which is what
/// [`SsaCfg::from_ssa`](crate::analysis::SsaCfg::from_ssa) builds. A graph that
/// also carries exception edges makes a block ending in an unconditional jump
/// look like a conditional branch, and it would be recovered as one — so a
/// caller building the graph from stored rows must drop them there.
///
/// Handler blocks are therefore unreachable from the entry, since the edge into
/// a handler is taken by the runtime. They are placed inside their
/// [`Region::Try`], so the tree still accounts for them — including when the
/// region's own entry is unreachable too.
///
/// Always succeeds, on the same terms as [`structure`], and declares every
/// block condition-only on the same grounds — so a caller whose nodes carry
/// statements wants [`structure_with`], or, holding an `SsaFunction`,
/// [`structure_ssa`](crate::analysis::recovery::structure_ssa), which builds the
/// regions from the exception table and the condition-only set from the blocks.
#[must_use]
pub fn structure_protected<G>(graph: &G, entry: NodeId, regions: &[ProtectedRegion]) -> Structured
where
    G: GraphBase + Successors + Predecessors,
{
    let node_count = graph.node_count();
    structure_with(
        graph,
        entry,
        &StructureOptions {
            regions,
            condition_only: BlockSet::full(node_count),
            // Nothing here can render a statement, so nothing can be expressed
            // inside a condition either; `condition_only` above already admits
            // every block on the grounds that none carries statements at all.
            condition_expressible: BlockSet::new(node_count),
            max_depth: DEFAULT_MAX_DEPTH,
        },
    )
}

/// What control-flow recovery needs beyond the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureOptions<'a> {
    /// Protected regions to recover; see [`structure_protected`].
    pub regions: &'a [ProtectedRegion],

    /// Blocks the caller has established compute nothing but their own branch
    /// condition.
    ///
    /// Two recoveries depend on it, and both are unsound without it.
    ///
    /// Folding one block's test into another's condition to recover
    /// `if (a && b)` *moves* that block, which a block that stores to memory
    /// cannot survive. And rendering a loop as `while (c)` puts its header
    /// outside the body, where a header that loads the value it tests would run
    /// once instead of every iteration — code that does not loop.
    ///
    /// Whether a block is such a block is a question about the instructions it
    /// holds, which this module does not see, so the caller answers it. An
    /// empty set is always safe: it costs merged conditions and turns pre-tested
    /// loops into `Endless` ones with an explicit `break`, both of which are
    /// longer to read and correct.
    pub condition_only: BlockSet,

    /// Blocks whose statements the caller can render *inside* a condition.
    ///
    /// A weaker claim than [`Self::condition_only`], and it answers a different
    /// question. That set asks whether a block can be **moved**; this one asks
    /// whether it can be **said** — because a `while` does not move its header
    /// anywhere. C evaluates a condition before every iteration and lets it
    /// carry the work: `while ((c = *p++) != 0)` runs the assignment exactly
    /// where the header ran it, and the body reads what it assigned.
    ///
    /// Short-circuit merging consults it too, and for the same reason. Joining
    /// a block's test onto a condition that already leads to its own target
    /// puts the block behind an `&&` or an `||`, and C evaluates the right
    /// operand of either exactly when control took the edge that reached the
    /// block — so the work runs on the paths it always ran on, and on no
    /// others.
    ///
    /// An empty set is always safe. It costs merged conditions and leaves
    /// pre-tested loops `Endless`, both of which are longer to read and
    /// correct.
    pub condition_expressible: BlockSet,

    /// How deep the recovered tree may nest.
    ///
    /// The walk descends one level per region it enters, and the tree it builds
    /// is as deep as the walk — so this is at once a bound on the machine stack
    /// the recovery uses and a bound on the stack every later traversal of the
    /// result uses, the derived `Drop` and `Clone` included. It also caps the
    /// number of folds one condition may take.
    ///
    /// [`DEFAULT_MAX_DEPTH`] is the value the entry points use. Clamped to at
    /// least 1, so depth 1 is never limited and the recovery always terminates.
    /// Reaching the bound costs recovery quality, never correctness: see
    /// [`StructureMetrics::depth_limited`].
    pub max_depth: usize,
}

/// Returns `set` sized to `node_count`, counting any resize into `resized`.
///
/// The single normalisation point. Every [`BlockSet`] the walk reads comes
/// through here, so the walk's own reads need no width check and cannot get a
/// silently wrong answer from a set built for another function.
fn normalised(node_count: usize, set: &BlockSet, resized: &mut usize) -> BlockSet {
    if set.node_count() == node_count {
        return set.clone();
    }
    *resized = resized.saturating_add(1);
    BlockSet::from_nodes(node_count, set.iter())
}

/// Returns `region` with every block set sized to `node_count`.
///
/// Written as an exhaustive destructure at each level on purpose: a block set
/// added to any of these types fails to compile here rather than reaching the
/// walk at the caller's width.
fn normalised_region(
    node_count: usize,
    region: &ProtectedRegion,
    resized: &mut usize,
) -> ProtectedRegion {
    let ProtectedRegion {
        entry,
        blocks,
        handlers,
    } = region;
    ProtectedRegion {
        entry: *entry,
        blocks: normalised(node_count, blocks, resized),
        handlers: handlers
            .iter()
            .map(|handler| {
                let ProtectedHandler {
                    kind,
                    entry,
                    blocks,
                } = handler;
                ProtectedHandler {
                    kind: match kind {
                        ProtectedHandlerKind::Filter(filter) => {
                            let HandlerFilter { entry, blocks } = filter;
                            ProtectedHandlerKind::Filter(HandlerFilter {
                                entry: *entry,
                                blocks: normalised(node_count, blocks, resized),
                            })
                        }
                        ProtectedHandlerKind::Catch => ProtectedHandlerKind::Catch,
                        ProtectedHandlerKind::Finally => ProtectedHandlerKind::Finally,
                        ProtectedHandlerKind::Fault => ProtectedHandlerKind::Fault,
                    },
                    entry: *entry,
                    blocks: normalised(node_count, blocks, resized),
                }
            })
            .collect(),
    }
}

/// Recovers structured control flow with full control over the recovery.
///
/// Always succeeds, on the same terms as [`structure`]. This is the one
/// boundary at which the caller's index domain is closed: block sets are
/// renormalised to the graph's node count, an `entry` outside the graph
/// recovers as [`Region::Empty`], and successor ids the graph reports past its
/// own node count contribute no edge.
#[must_use]
pub fn structure_with<G>(graph: &G, entry: NodeId, options: &StructureOptions<'_>) -> Structured
where
    G: GraphBase + Successors + Predecessors,
{
    let node_count = graph.node_count();
    let mut resized = 0usize;
    let condition_only = normalised(node_count, &options.condition_only, &mut resized);
    let condition_expressible =
        normalised(node_count, &options.condition_expressible, &mut resized);
    let regions: Vec<ProtectedRegion> = options
        .regions
        .iter()
        .map(|region| normalised_region(node_count, region, &mut resized))
        .collect();

    let mut labels = BTreeSet::new();
    let mut metrics = StructureMetrics {
        resized_inputs: resized,
        ..StructureMetrics::default()
    };

    let root = if node_count == 0 || entry.index() >= node_count {
        Region::Empty
    } else {
        let dominators = compute_dominators(graph, entry);
        let post_dominators = compute_post_dominators(graph);
        let loops = detect_loops(graph, &dominators);

        let mut headers = BTreeMap::new();
        for (index, info) in loops.loops().iter().enumerate() {
            headers.insert(info.header, index);
        }

        // Ascending by size, then by declaration index, and pushed in that
        // order: `Vec::pop` therefore yields the *largest* region first, which
        // is the one that must open first so the smaller one opens inside its
        // body. See `Structurer::protected`.
        let mut order: Vec<usize> = (0..regions.len()).collect();
        order.sort_by_cached_key(|index| {
            (regions.get(*index).map_or(0, ProtectedRegion::size), *index)
        });
        let mut protected: BTreeMap<NodeId, Vec<usize>> = BTreeMap::new();
        for index in order {
            if let Some(region) = regions.get(index) {
                protected.entry(region.entry).or_default().push(index);
            }
        }

        let mut structurer = Structurer {
            graph,
            node_count,
            post_dominators,
            loops: &loops,
            headers,
            regions: &regions,
            protected,
            condition_only,
            condition_expressible,
            visited: BlockSet::new(node_count),
            labels: BTreeSet::new(),
            frames: Vec::new(),
            frame_barrier: 0,
            enclosing: Vec::new(),
            confined: Vec::new(),
            depth: 0,
            max_depth: options.max_depth.max(1),
            metrics,
        };

        let root = structurer.emit_function(entry);
        labels = structurer.labels;
        metrics = structurer.metrics;
        root
    };

    metrics.unreached = node_count.saturating_sub(metrics.blocks);

    Structured {
        root,
        labels,
        metrics,
    }
}

impl ProtectedRegion {
    /// Returns how many blocks the protected body covers.
    fn size(&self) -> usize {
        self.blocks.count()
    }
}

/// A condition grown across several blocks, and what it selects between.
struct MergedCondition {
    /// The recovered condition.
    predicate: Predicate,

    /// Block reached when the condition holds.
    taken: NodeId,

    /// Block reached when it does not.
    not_taken: NodeId,

    /// Blocks folded into the condition beyond the one it started at.
    consumed: Vec<NodeId>,

    /// Whether folding stopped at the fold bound rather than at a block that
    /// could not join the condition.
    truncated: bool,
}

/// One test folded into a growing condition.
struct ConditionStep {
    /// The test, holding exactly when control reaches [`ConditionStep::kept`].
    test: Predicate,

    /// Block the grown condition now selects on this side.
    kept: NodeId,
}

/// Conjoins two conditions, flattening rather than nesting.
fn all(left: Predicate, right: Predicate) -> Predicate {
    match left {
        Predicate::All(mut parts) => {
            parts.push(right);
            Predicate::All(parts)
        }
        other => Predicate::All(vec![other, right]),
    }
}

/// Disjoins two conditions, flattening rather than nesting.
fn any(left: Predicate, right: Predicate) -> Predicate {
    match left {
        Predicate::Any(mut parts) => {
            parts.push(right);
            Predicate::Any(parts)
        }
        other => Predicate::Any(vec![other, right]),
    }
}

/// Where a loop tests its continuation condition, and what that implies for
/// how its body is walked.
struct LoopShape {
    /// The recovered form.
    kind: StructuredLoop,

    /// The condition under which the loop continues, if it has one.
    predicate: Option<Predicate>,

    /// Latch consumed as the test, whose successors must not be walked.
    tail: Option<NodeId>,

    /// Block the body begins at.
    body_start: Option<NodeId>,
}

/// One enclosing loop during the walk.
struct LoopFrame<'a> {
    /// The loop's header block.
    header: NodeId,

    /// Blocks belonging to the loop, as [`crate::analysis::loops::detect_loops`]
    /// recorded them — one bit per node of the graph.
    body: &'a BitSet,

    /// Block control resumes at on `break`, if the loop is left.
    follow: Option<NodeId>,

    /// Latch whose terminator became the loop condition, and whose outgoing
    /// edges must therefore not be walked as ordinary control flow.
    tail: Option<NodeId>,
}

impl LoopFrame<'_> {
    /// Returns whether `node` belongs to the loop.
    ///
    /// Total over every [`NodeId`]: a loop body is sized to the graph's node
    /// count, so an index past it is simply not a member.
    fn holds(&self, node: NodeId) -> bool {
        self.body.contains_checked(node.index())
    }
}

/// Whether a walk may see the loop frames its caller was inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Barrier {
    /// The walk continues inside the same loops.
    Inherit,

    /// The walk is entered by the runtime rather than by an edge of the loop,
    /// so no enclosing loop bounds it.
    Raise,
}

/// One region the walk enters, and everything entering it decides.
///
/// Confinement, enclosing follow and loop-frame barrier are three answers to
/// one question — which scope's state governs the walk about to happen — and
/// keeping them in three hand-managed stacks is what let them disagree. A
/// `Scope` has no default and no builder, so entering a region without
/// deciding all three does not compile.
#[derive(Debug, Clone, Copy)]
struct Scope<'a> {
    /// Blocks the walk may place, or `None` to keep the enclosing confinement.
    confinement: Option<&'a BlockSet>,

    /// Block the enclosing region resumes at, which the walk must leave for
    /// rather than place, or `None` when the region has no follow.
    follow: Option<NodeId>,

    /// Whether the enclosing loops still apply.
    barrier: Barrier,
}

/// Walk state for one function.
struct Structurer<'a, G> {
    /// The graph being structured.
    graph: &'a G,

    /// Node count of [`Structurer::graph`], and so the index domain every
    /// [`BlockSet`] here shares.
    node_count: usize,

    /// Post-dominance, used to find conditional joins.
    post_dominators: PostDominatorTree,

    /// Natural loops of the graph.
    loops: &'a LoopForest,

    /// Loop header to its index in [`LoopForest::loops`].
    headers: BTreeMap<NodeId, usize>,

    /// Protected regions to recover, normalised to [`Structurer::node_count`].
    regions: &'a [ProtectedRegion],

    /// Try entry block to the regions starting there that the walk has not
    /// entered yet, **outermost last**.
    ///
    /// `Vec::pop` therefore returns the region that must open first, and
    /// `emit_try` re-enters the walk at the same block, so the next region
    /// down opens inside the body of the one above it. Reversing the order
    /// would nest the clauses inside out.
    ///
    /// An index is removed when the walk enters its region, so the map means
    /// "regions not yet entered" — which is what `chain` reads while it is
    /// still live, and what lets `emit_function` drain the regions whose entry
    /// the walk never reached.
    protected: BTreeMap<NodeId, Vec<usize>>,

    /// Blocks that compute only their own branch condition.
    condition_only: BlockSet,

    /// Blocks whose statements the caller renders inside a condition; see
    /// [`StructureOptions::condition_expressible`].
    condition_expressible: BlockSet,

    /// Blocks already placed in the tree.
    visited: BlockSet,

    /// Blocks a goto targets.
    labels: BTreeSet<NodeId>,

    /// Enclosing loops, innermost last.
    frames: Vec<LoopFrame<'a>>,

    /// Index into [`Structurer::frames`] below which the walk sees no loop.
    ///
    /// A handler or filter is entered by the runtime, not by an edge of the
    /// loop the protected region sits in, so the loop's body does not bound
    /// the handler's joins. `transfer` deliberately still scans every frame:
    /// the `Region::Try` is emitted lexically inside the loop body, so
    /// `break`/`continue` out of a handler is legal and must not become a goto.
    frame_barrier: usize,

    /// Follow blocks of enclosing regions, innermost last.
    enclosing: Vec<NodeId>,

    /// Block sets the walk is confined to, innermost last.
    ///
    /// A protected body and a handler each own a fixed set of blocks, and the
    /// walk must not place a block belonging to neither inside them. Escapes
    /// leave by goto and are placed at the end of the function instead.
    confined: Vec<&'a BlockSet>,

    /// Regions entered and not yet left; see [`StructureOptions::max_depth`].
    depth: usize,

    /// The nesting bound, clamped to at least 1.
    max_depth: usize,

    /// Counts collected during the walk.
    metrics: StructureMetrics,
}

impl<'a, G> Structurer<'a, G>
where
    G: GraphBase + Successors + Predecessors,
{
    /// Returns the successors of `node` that the graph can actually hold.
    ///
    /// A successor id at or past the node count names no block, so it
    /// contributes no edge — the same rule
    /// [`SsaCfg::from_ssa`](crate::analysis::SsaCfg::from_ssa) states for
    /// itself. Routing every read through here is what keeps an out-of-range id
    /// from reaching a set index or a follow choice.
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let limit = self.node_count;
        self.graph
            .successors(node)
            .filter(move |successor| successor.index() < limit)
    }

    /// Returns how many predecessors of `node` the graph can actually hold.
    fn predecessor_count(&self, node: NodeId) -> usize {
        let limit = self.node_count;
        self.graph
            .predecessors(node)
            .filter(|predecessor| predecessor.index() < limit)
            .count()
    }

    /// Runs `walk` inside `scope`, releasing everything it entered.
    ///
    /// The one place a region is entered, so confinement, enclosing follow and
    /// loop-frame barrier are pushed together and popped together whatever the
    /// walk does.
    fn in_scope<R>(&mut self, scope: Scope<'a>, walk: impl FnOnce(&mut Self) -> R) -> R {
        let saved_barrier = self.frame_barrier;
        if scope.barrier == Barrier::Raise {
            self.frame_barrier = self.frames.len();
        }
        if let Some(blocks) = scope.confinement {
            self.confined.push(blocks);
        }
        if let Some(node) = scope.follow {
            self.enclosing.push(node);
        }

        let result = walk(self);

        if scope.follow.is_some() {
            self.enclosing.pop();
        }
        if scope.confinement.is_some() {
            self.confined.pop();
        }
        self.frame_barrier = saved_barrier;
        result
    }

    /// Returns the innermost loop frame the current walk is inside.
    ///
    /// `None` inside a handler or filter, whose walk the enclosing loop does
    /// not bound; see [`Structurer::frame_barrier`].
    fn enclosing_frame(&self) -> Option<&LoopFrame<'a>> {
        self.frames
            .get(self.frame_barrier..)
            .and_then(<[LoopFrame<'a>]>::last)
    }

    /// Emits the whole function, including any block the tree walk did not
    /// place.
    ///
    /// The main walk follows control flow from the entry and places everything
    /// it reaches, so ordinarily it leaves nothing behind. Two things can
    /// remain. A protected region whose entry is unreachable — the runtime
    /// takes the edge into a handler, and nothing says the try itself is
    /// reached — is drained first, through `emit_try`, so its handlers are
    /// still inside a [`Region::Try`] rather than printed as unprotected code
    /// at top level. Then any block a bounded walk had to leave behind is
    /// placed, reached by the goto that already names it.
    ///
    /// Running both unconditionally is what makes the placement invariant hold
    /// by construction rather than by argument: after them, every block
    /// reachable from the entry, and every block a region owns, appears in the
    /// tree.
    fn emit_function(&mut self, entry: NodeId) -> Region {
        let mut parts = vec![self.emit(entry, None, false)];

        while let Some(start) = self.protected.keys().next().copied() {
            let Some(index) = self.take_protected(start) else {
                break;
            };
            let (region, _) = self.emit_try(index, None);
            parts.push(region);
        }

        let mut reachable = self.reachable_from(entry);
        for region in self.regions {
            reachable.insert(region.entry);
            for block in region.blocks.iter() {
                reachable.insert(block);
            }
            for handler in &region.handlers {
                reachable.insert(handler.entry);
                for block in handler.blocks.iter() {
                    reachable.insert(block);
                }
                if let Some(filter) = handler.kind.filter() {
                    reachable.insert(filter.entry);
                    for block in filter.blocks.iter() {
                        reachable.insert(block);
                    }
                }
            }
        }

        // One ascending pass rather than a rescan per orphan: emitting one
        // orphan can place later ones, and the `visited` test skips those.
        let orphans: Vec<NodeId> = reachable.iter().collect();
        for orphan in orphans {
            if self.visited.contains(orphan) {
                continue;
            }
            parts.push(self.emit(orphan, None, false));
        }

        Self::collapse(parts)
    }

    /// Returns the blocks reachable from `entry`.
    fn reachable_from(&self, entry: NodeId) -> BlockSet {
        let mut seen = BlockSet::new(self.node_count);
        let mut stack = vec![entry];
        while let Some(node) = stack.pop() {
            if seen.insert(node) {
                stack.extend(self.successors(node));
            }
        }
        seen
    }

    /// Takes the region that must open next at `entry`, if one is left.
    fn take_protected(&mut self, entry: NodeId) -> Option<usize> {
        let remaining = self.protected.get_mut(&entry)?;
        let index = remaining.pop();
        if remaining.is_empty() {
            self.protected.remove(&entry);
        }
        index
    }

    /// Emits the region beginning at `start` and ending before `follow`, one
    /// nesting level deeper.
    ///
    /// Every descent goes through here, so this is the single place nesting is
    /// counted. At the bound the walk is abandoned rather than deepened; see
    /// [`StructureOptions::max_depth`].
    fn emit(&mut self, start: NodeId, follow: Option<NodeId>, skip_transfer: bool) -> Region {
        let depth = self.depth.saturating_add(1);
        if depth > self.max_depth {
            return self.abandon(start);
        }
        self.depth = depth;
        let region = self.emit_sequence(start, follow, skip_transfer);
        self.depth = self.depth.saturating_sub(1);
        region
    }

    /// Gives up on a walk that would exceed the nesting bound.
    ///
    /// Unconfined, the block becomes a goto and the orphan pass places it at
    /// top level — the module's existing escape hatch, at depth 0.
    ///
    /// Confined, a goto would defer a block the enclosing region owns to top
    /// level *outside* its [`Region::Try`], printing protected code as
    /// unprotected. The rest of the confinement is flattened in place instead:
    /// coarse protection rather than wrong protection.
    fn abandon(&mut self, start: NodeId) -> Region {
        self.metrics.depth_limited = self.metrics.depth_limited.saturating_add(1);
        let Some(owned) = self.confined.last().copied() else {
            return self.goto(start);
        };
        if !owned.contains(start) {
            return self.goto(start);
        }
        self.emit_flat(owned)
    }

    /// Emits every unplaced block of `owned` with each of its edges explicit.
    ///
    /// Two levels deep at most, whatever the shape of the code: one statement
    /// per block, and every outgoing edge a goto. It says the same thing the
    /// structured form would have said, in the longest way there is.
    fn emit_flat(&mut self, owned: &BlockSet) -> Region {
        let mut parts = Vec::new();
        for node in owned.iter() {
            if self.visited.contains(node) {
                continue;
            }
            self.visited.insert(node);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);

            let arms = self.arms(node);
            let region = match (arms.first(), arms.get(1)) {
                (None, _) => Region::Block(node),
                (Some((only, _)), None) => {
                    let jump = self.goto(*only);
                    Region::Seq(vec![Region::Block(node), jump])
                }
                (Some((taken, _)), Some((not_taken, _))) if arms.len() == 2 => {
                    let then_branch = self.goto(*taken);
                    let else_branch = self.goto(*not_taken);
                    self.metrics.ifs = self.metrics.ifs.saturating_add(1);
                    self.metrics.if_elses = self.metrics.if_elses.saturating_add(1);
                    Region::If(Box::new(IfRegion {
                        predicate: Predicate::Test {
                            block: node,
                            negated: false,
                        },
                        then_branch,
                        else_branch: Some(else_branch),
                    }))
                }
                _ => {
                    let mut cases = Vec::with_capacity(arms.len());
                    for (target, selectors) in &arms {
                        let body = self.goto(*target);
                        cases.push(SwitchCase {
                            selectors: selectors.clone(),
                            target: *target,
                            body,
                        });
                    }
                    self.metrics.switches = self.metrics.switches.saturating_add(1);
                    Region::Switch(Box::new(SwitchRegion {
                        head: node,
                        cases,
                        follow: None,
                    }))
                }
            };
            parts.push(region);
        }
        Self::collapse(parts)
    }

    /// Emits the region beginning at `start` and ending before `follow`.
    ///
    /// `skip_transfer` suppresses the `break`/`continue` test for the first
    /// block only, which is how a loop body may begin at its own header without
    /// immediately reading as `continue`.
    fn emit_sequence(
        &mut self,
        start: NodeId,
        follow: Option<NodeId>,
        skip_transfer: bool,
    ) -> Region {
        let mut sequence: Vec<Region> = Vec::new();
        let mut cursor = start;
        let mut first = true;

        loop {
            if Some(cursor) == follow {
                break;
            }

            if !(first && skip_transfer) {
                if let Some(transfer) = self.transfer(cursor) {
                    sequence.push(transfer);
                    break;
                }
                if self.enclosing.contains(&cursor) {
                    sequence.push(self.goto(cursor));
                    break;
                }
            }
            first = false;

            if self
                .confined
                .last()
                .is_some_and(|owned| !owned.contains(cursor))
            {
                sequence.push(self.goto(cursor));
                break;
            }

            if self.visited.contains(cursor) {
                sequence.push(self.revisit(cursor));
                break;
            }

            if let Some(index) = self.take_protected(cursor) {
                let (region, resume) = self.emit_try(index, follow);
                sequence.push(region);
                match resume {
                    Some(next) => {
                        cursor = next;
                        continue;
                    }
                    None => break,
                }
            }

            // A header already on the frame stack is the loop we are inside,
            // reached as the first block of its own body. Re-entering it here
            // would not terminate.
            let entering_loop = self.headers.contains_key(&cursor)
                && !self.frames.iter().any(|frame| frame.header == cursor);
            if entering_loop {
                let (region, resume) = self.emit_loop(cursor, follow);
                sequence.push(region);
                match resume {
                    Some(next) => {
                        cursor = next;
                        continue;
                    }
                    None => break,
                }
            }

            let arms = self.arms(cursor);
            // A latch consumed as a loop condition keeps its statements as a
            // block; its branch is the loop's test, not a conditional.
            let is_tail = self
                .enclosing_frame()
                .is_some_and(|frame| frame.tail == Some(cursor));
            // A conditional or a dispatch owns the block it branches from, so
            // that a condition spanning several blocks has one home in the
            // tree. Its statements still print before the test.
            let owns_head = !is_tail && arms.len() >= 2;

            self.visited.insert(cursor);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);
            if !owns_head {
                sequence.push(Region::Block(cursor));
            }

            if is_tail {
                break;
            }

            match arms.len() {
                0 => break,
                1 => match arms.first() {
                    Some((target, _)) => {
                        cursor = *target;
                        continue;
                    }
                    None => break,
                },
                2 => {
                    let (region, resume) = self.emit_if(cursor, &arms, follow);
                    sequence.push(region);
                    match resume {
                        Some(next) => {
                            cursor = next;
                            continue;
                        }
                        None => break,
                    }
                }
                _ => {
                    let (region, resume) = self.emit_switch(cursor, &arms, follow);
                    sequence.push(region);
                    match resume {
                        Some(next) => {
                            cursor = next;
                            continue;
                        }
                        None => break,
                    }
                }
            }
        }

        Self::collapse(sequence)
    }

    /// Reduces an emitted sequence to its simplest equivalent region.
    ///
    /// Nested sequences are flattened. A `Seq` inside a `Seq` is never a real
    /// nesting level — it only records that one part of the walk returned
    /// several statements — and leaving it in would make the tree's shape
    /// depend on how the walk was decomposed rather than on the code.
    fn collapse(sequence: Vec<Region>) -> Region {
        let mut flattened: Vec<Region> = Vec::with_capacity(sequence.len());
        for region in sequence {
            match region {
                Region::Empty => {}
                Region::Seq(inner) => {
                    flattened.extend(inner.into_iter().filter(|part| *part != Region::Empty));
                }
                other => flattened.push(other),
            }
        }
        match flattened.len() {
            0 => Region::Empty,
            1 => flattened.pop().unwrap_or(Region::Empty),
            _ => Region::Seq(flattened),
        }
    }

    /// Emits a second arrival at an already-placed block.
    ///
    /// Ordinarily this is a goto: the block has a home in the tree and this
    /// path jumps to it. A block with no successors is the exception. It ends
    /// the function, so placing it twice repeats a straight-line tail and
    /// changes nothing about what runs — and a `goto` to a bare `return` is
    /// strictly worse to read than the `return` itself, which is the single
    /// most common goto in compiled code.
    ///
    /// The trade is size: a sink is duplicated however long it is, once per
    /// path that reaches it. There is no length at which that becomes wrong,
    /// only one at which it stops being an improvement, and picking that length
    /// would be picking a number the code cannot justify. Duplicating every
    /// sink is the rule that needs no number.
    fn revisit(&mut self, target: NodeId) -> Region {
        let sink = self.successors(target).next().is_none();
        if sink && !self.headers.contains_key(&target) && !self.protected.contains_key(&target) {
            self.metrics.replicated_tails = self.metrics.replicated_tails.saturating_add(1);
            return Region::Block(target);
        }
        self.goto(target)
    }

    /// Records `target` as needing a label and counts the transfer.
    fn goto(&mut self, target: NodeId) -> Region {
        self.labels.insert(target);
        self.metrics.gotos = self.metrics.gotos.saturating_add(1);
        Region::Goto(target)
    }

    /// Returns the loop transfer `node` represents, if it is one.
    ///
    /// Enclosing loops are tested innermost first, so a block that is both the
    /// inner header and an outer follow continues the inner loop. Every frame
    /// is scanned, barrier or not: a handler is emitted lexically inside the
    /// loop body its protected region sits in, so a `break` or `continue` from
    /// it is legal.
    fn transfer(&mut self, node: NodeId) -> Option<Region> {
        let matched = self.frames.iter().rev().find_map(|frame| {
            if node == frame.header {
                Some(Region::Continue(frame.header))
            } else if Some(node) == frame.follow {
                Some(Region::Break(frame.header))
            } else {
                None
            }
        })?;

        match matched {
            Region::Continue(_) => {
                self.metrics.continues = self.metrics.continues.saturating_add(1)
            }
            _ => self.metrics.breaks = self.metrics.breaks.saturating_add(1),
        }
        Some(matched)
    }

    /// Returns whether `node` is a loop header or follow on the frame stack.
    fn is_transfer(&self, node: NodeId) -> bool {
        self.frames
            .iter()
            .any(|frame| node == frame.header || Some(node) == frame.follow)
    }

    /// Returns the distinct successors of `node` with the selector indices that
    /// reach each, in first-selector order.
    ///
    /// Collapsing duplicates is what keeps a branch whose arms both target one
    /// block from reading as a conditional with two identical arms.
    fn arms(&self, node: NodeId) -> Vec<(NodeId, Vec<usize>)> {
        let mut arms: Vec<(NodeId, Vec<usize>)> = Vec::new();
        for (index, successor) in self.successors(node).enumerate() {
            match arms.iter_mut().find(|(target, _)| *target == successor) {
                Some((_, selectors)) => selectors.push(index),
                None => arms.push((successor, vec![index])),
            }
        }
        arms
    }

    /// Returns the block both arms of a branch at `node` rejoin at, if that
    /// block can bound the arms.
    ///
    /// The join is the immediate post-dominator, rejected when it is the
    /// virtual exit (the arms never rejoin), when it has already been placed
    /// (the arms will reach it by goto), or when it lies outside the enclosing
    /// loop other than as that loop's own follow (using it would drag the code
    /// after the loop inside it).
    fn join(&self, node: NodeId, follow: Option<NodeId>) -> Option<NodeId> {
        // `immediate_post_dominator` answers `None` for the virtual exit, so a
        // real block index is all that reaches the guards below.
        let join = self.post_dominators.immediate_post_dominator(node)?;
        if join == node {
            return None;
        }
        if Some(join) == follow {
            return Some(join);
        }
        if self.visited.contains(join) {
            return None;
        }
        if self
            .confined
            .last()
            .is_some_and(|owned| !owned.contains(join))
        {
            return None;
        }
        if let Some(frame) = self.enclosing_frame()
            && !frame.holds(join)
            && Some(join) != frame.follow
        {
            return None;
        }
        Some(join)
    }

    /// Returns the blocks of `predicate` as a sequence of statements.
    ///
    /// The blocks a condition spans are placed by the [`Region::If`] that
    /// holds it, so anything that drops the conditional has to place them
    /// itself or lose them from the tree.
    fn predicate_region(predicate: &Predicate) -> Region {
        let mut parts = Vec::new();
        predicate.for_each_block(&mut |block| parts.push(Region::Block(block)));
        Self::collapse(parts)
    }

    /// Returns whether control can reach the statement after `region`.
    ///
    /// A `continue` is deliberately not an exit here. It is implicit at a loop
    /// body's tail, so treating it as one would hoist the other arm out of the
    /// `else` and leave behind a `continue` that is no longer in tail position
    /// and can no longer be dropped — longer output, for nothing.
    fn leaves_the_sequence(&self, region: &Region) -> bool {
        match region {
            Region::Break(_) | Region::Goto(_) => true,
            Region::Block(node) => self.successors(*node).next().is_none(),
            Region::Empty
            | Region::Continue(_)
            | Region::Loop(_)
            | Region::Switch(_)
            | Region::Try(_) => false,
            Region::Seq(parts) => parts
                .last()
                .is_some_and(|last| self.leaves_the_sequence(last)),
            Region::If(inner) => match &inner.else_branch {
                Some(other) => {
                    self.leaves_the_sequence(&inner.then_branch) && self.leaves_the_sequence(other)
                }
                None => false,
            },
        }
    }

    /// Returns whether walking `node` next would place exactly one
    /// function-ending block.
    ///
    /// Every reason `emit_sequence` would do something else is ruled out here,
    /// so the answer is what the walk would produce and not a guess at it.
    fn ends_the_function(&self, node: NodeId, follow: Option<NodeId>) -> bool {
        Some(node) != follow
            && !self.visited.contains(node)
            && !self.is_transfer(node)
            && !self.enclosing.contains(&node)
            && !self.headers.contains_key(&node)
            && !self.protected.contains_key(&node)
            && self
                .confined
                .last()
                .is_none_or(|owned| owned.contains(node))
            && self.successors(node).next().is_none()
    }

    /// Emits a two-way branch at `head` and returns the block to resume at.
    ///
    /// An arm that cannot fall through ends the conditional: the other arm is
    /// returned as the resume point rather than walked as an `else`, so
    /// `if (c) return; ..` stays one statement deep however many times it
    /// repeats. That is what keeps the early-return spine — a few hundred
    /// sequential `if (x) return;` tests, ordinary in decompiled code — off the
    /// nesting bound, which is then a backstop rather than a limit real input
    /// reaches.
    fn emit_if(
        &mut self,
        head: NodeId,
        arms: &[(NodeId, Vec<usize>)],
        follow: Option<NodeId>,
    ) -> (Region, Option<NodeId>) {
        let join = self.join(head, follow);
        let Some(merged) = self.merge_condition(head, arms) else {
            return (Region::Empty, None);
        };

        for block in &merged.consumed {
            self.visited.insert(*block);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);
        }
        if !merged.consumed.is_empty() {
            self.metrics.merged_conditions = self.metrics.merged_conditions.saturating_add(1);
        }
        if merged.truncated {
            self.metrics.depth_limited = self.metrics.depth_limited.saturating_add(1);
        }

        // The join `merge_condition` folded into the condition has no statement
        // position left to resume at; the arms still stop there, which is what
        // makes an arm looping back to it end cleanly instead of jumping.
        let resume = join.filter(|node| !merged.consumed.contains(node));
        let scope = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };

        // The failing arm ends the function. Emitting it first inverts the
        // condition and leaves the taken arm as the parent's continuation,
        // which is both the shorter output and the one that does not nest.
        if join.is_none() && self.ends_the_function(merged.not_taken, follow) {
            let otherwise = self.in_scope(scope, |walk| walk.emit(merged.not_taken, join, false));
            self.metrics.ifs = self.metrics.ifs.saturating_add(1);
            let region = Region::If(Box::new(IfRegion {
                predicate: merged.predicate.negate(),
                then_branch: otherwise,
                else_branch: None,
            }));
            return (region, Some(merged.taken));
        }

        let taken = self.in_scope(scope, |walk| walk.emit(merged.taken, join, false));
        if self.leaves_the_sequence(&taken) {
            self.metrics.ifs = self.metrics.ifs.saturating_add(1);
            let region = Region::If(Box::new(IfRegion {
                predicate: merged.predicate,
                then_branch: taken,
                else_branch: None,
            }));
            return (region, Some(merged.not_taken));
        }
        let not_taken = self.in_scope(scope, |walk| walk.emit(merged.not_taken, join, false));

        let region = match (taken, not_taken) {
            // Unreachable for any input the walk can produce, and lossy if it
            // were not: the conditional owns the blocks its condition spans.
            (Region::Empty, Region::Empty) => Self::predicate_region(&merged.predicate),
            // An empty `then` reads as a negated condition with one arm, which
            // is always better than `if (c) {} else { .. }`.
            (Region::Empty, other) => {
                self.metrics.ifs = self.metrics.ifs.saturating_add(1);
                Region::If(Box::new(IfRegion {
                    predicate: merged.predicate.negate(),
                    then_branch: other,
                    else_branch: None,
                }))
            }
            (then_branch, otherwise) => {
                self.metrics.ifs = self.metrics.ifs.saturating_add(1);
                let else_branch = (otherwise != Region::Empty).then_some(otherwise);
                if else_branch.is_some() {
                    self.metrics.if_elses = self.metrics.if_elses.saturating_add(1);
                }
                Region::If(Box::new(IfRegion {
                    predicate: merged.predicate,
                    then_branch,
                    else_branch,
                }))
            }
        };

        (region, resume)
    }

    /// Emits a multi-way branch at `head` and returns the block to resume at.
    fn emit_switch(
        &mut self,
        head: NodeId,
        arms: &[(NodeId, Vec<usize>)],
        follow: Option<NodeId>,
    ) -> (Region, Option<NodeId>) {
        let join = self.join(head, follow);
        let scope = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };

        let cases = self.in_scope(scope, |walk| {
            let mut cases = Vec::with_capacity(arms.len());
            for (target, selectors) in arms {
                let body = walk.emit(*target, join, false);
                cases.push(SwitchCase {
                    selectors: selectors.clone(),
                    target: *target,
                    body,
                });
            }
            cases
        });

        self.metrics.switches = self.metrics.switches.saturating_add(1);

        let region = Region::Switch(Box::new(SwitchRegion {
            head,
            cases,
            follow: join,
        }));
        (region, join)
    }

    /// Grows the condition at `head` across every block that only tests, and
    /// returns the two blocks the whole condition selects between.
    ///
    /// A block qualifies as part of the condition when reaching it means the
    /// condition so far has already decided one way, it does nothing but test,
    /// and one of its own arms rejoins the block the condition already leads
    /// to. That is exactly the shape `&&` and `||` compile to, and the walk
    /// repeats so a chain of any length collapses into one condition — up to
    /// [`StructureOptions::max_depth`] folds, past which the rest of the chain
    /// recovers as further conditionals rather than as one predicate too deep
    /// to traverse.
    fn merge_condition(
        &self,
        head: NodeId,
        arms: &[(NodeId, Vec<usize>)],
    ) -> Option<MergedCondition> {
        let (taken, _) = arms.first()?;
        let (not_taken, _) = arms.get(1)?;

        let mut merged = MergedCondition {
            predicate: Predicate::Test {
                block: head,
                negated: false,
            },
            taken: *taken,
            not_taken: *not_taken,
            consumed: Vec::new(),
            truncated: false,
        };

        loop {
            if merged.consumed.len() >= self.max_depth {
                merged.truncated = true;
                break;
            }
            // Reaching `taken` means the condition so far held; a test there
            // whose other arm rejoins `not_taken` conjoins with it.
            if let Some(step) = self.chain(merged.taken, merged.not_taken) {
                merged.consumed.push(merged.taken);
                merged.predicate = all(merged.predicate, step.test);
                merged.taken = step.kept;
                continue;
            }
            // Symmetrically, a test on the failing side that rejoins `taken`
            // disjoins with it — but with the opposite polarity. `chain` reports
            // the test that holds when control reaches the block it keeps, and
            // here that block is the one the *whole* condition rejects, so what
            // joins the disjunction is its negation.
            if let Some(step) = self.chain(merged.not_taken, merged.taken) {
                merged.consumed.push(merged.not_taken);
                merged.predicate = any(merged.predicate, step.test.negate());
                merged.not_taken = step.kept;
                continue;
            }
            break;
        }

        Some(merged)
    }

    /// Returns the test `candidate` contributes to a condition that already
    /// leads to `shared`, when it contributes one.
    ///
    /// The returned test holds exactly when control at `candidate` reaches
    /// [`ConditionStep::kept`] rather than `shared`.
    fn chain(&self, candidate: NodeId, shared: NodeId) -> Option<ConditionStep> {
        if candidate == shared
            || !(self.condition_only.contains(candidate)
                || self.condition_expressible.contains(candidate))
        {
            return None;
        }
        // Folding a block into a condition moves its text, not the paths it
        // runs on: short-circuit evaluation reaches the operand exactly when
        // control reached the block. That leaves the structural conditions —
        // nothing else reaches it, it is not itself a region a walk must enter
        // (a loop header or a protected region), and it belongs to whatever
        // region the walk is currently confined to.
        if self.predecessor_count(candidate) != 1
            || self.headers.contains_key(&candidate)
            || self.protected.contains_key(&candidate)
            || self.visited.contains(candidate)
        {
            return None;
        }
        if self
            .confined
            .last()
            .is_some_and(|owned| !owned.contains(candidate))
        {
            return None;
        }
        if let Some(frame) = self.enclosing_frame()
            && !frame.holds(candidate)
        {
            return None;
        }

        let arms = self.arms(candidate);
        let (first, _) = arms.first()?;
        let (second, _) = arms.get(1)?;
        if arms.len() != 2 {
            return None;
        }

        // The arm that rejoins is the one the condition already accounts for;
        // the other is what the grown condition now selects.
        let (kept, negated) = if *second == shared {
            (*first, false)
        } else if *first == shared {
            (*second, true)
        } else {
            return None;
        };

        Some(ConditionStep {
            test: Predicate::Test {
                block: candidate,
                negated,
            },
            kept,
        })
    }

    /// Emits the protected region at `index` and returns the block to resume at.
    ///
    /// The body and each handler are walked confined to their own blocks, so a
    /// `leave` out of the region does not drag the code after it inside. A
    /// handler and a filter are entered by the runtime rather than by an edge,
    /// so they also raise the loop-frame barrier: the loop the region sits in
    /// does not bound their joins, although a `break` or `continue` out of them
    /// still names it.
    fn emit_try(&mut self, index: usize, follow: Option<NodeId>) -> (Region, Option<NodeId>) {
        let regions: &'a [ProtectedRegion] = self.regions;
        let Some(region) = regions.get(index) else {
            return (Region::Empty, None);
        };
        let entry = region.entry;
        let blocks = &region.blocks;

        let try_follow = self.pick_follow(entry, blocks.iter(), |node| blocks.contains(node));
        self.metrics.tries = self.metrics.tries.saturating_add(1);

        let outer = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };
        let recovered = self.in_scope(outer, |walk| {
            let body = walk.in_scope(
                Scope {
                    confinement: Some(blocks),
                    follow: None,
                    barrier: Barrier::Inherit,
                },
                |walk| walk.emit(entry, try_follow, false),
            );

            let mut handlers = Vec::with_capacity(region.handlers.len());
            for handler in &region.handlers {
                walk.metrics.handlers = walk.metrics.handlers.saturating_add(1);

                let filter = handler.kind.filter().map(|filter| {
                    walk.in_scope(
                        Scope {
                            confinement: Some(&filter.blocks),
                            follow: None,
                            barrier: Barrier::Raise,
                        },
                        |walk| walk.emit(filter.entry, None, false),
                    )
                });

                // A handler that completes normally resumes after the region,
                // exactly as the protected body does, so it ends there rather
                // than jumping there.
                let body = walk.in_scope(
                    Scope {
                        confinement: Some(&handler.blocks),
                        follow: None,
                        barrier: Barrier::Raise,
                    },
                    |walk| walk.emit(handler.entry, try_follow, false),
                );

                handlers.push(HandlerRegion {
                    kind: handler.kind.kind(),
                    entry: handler.entry,
                    filter,
                    body,
                });
            }

            TryRegion {
                entry,
                body,
                handlers,
                follow: try_follow,
            }
        });

        (Region::Try(Box::new(recovered)), try_follow)
    }

    /// Emits the loop headed at `header` and returns the block to resume at.
    fn emit_loop(&mut self, header: NodeId, follow: Option<NodeId>) -> (Region, Option<NodeId>) {
        let loops: &'a LoopForest = self.loops;
        let Some(info) = self
            .headers
            .get(&header)
            .copied()
            .and_then(|index| loops.loops().get(index))
        else {
            return (Region::Empty, None);
        };
        let body_set: &'a BitSet = &info.body;

        let loop_follow = self.pick_follow(header, body_set.iter().map(NodeId::new), |node| {
            body_set.contains_checked(node.index())
        });
        let LoopShape {
            kind,
            predicate,
            tail,
            body_start,
        } = self.classify(header, body_set, &info.latches, loop_follow);

        self.metrics.loops = self.metrics.loops.saturating_add(1);
        if kind == StructuredLoop::Endless {
            self.metrics.endless_loops = self.metrics.endless_loops.saturating_add(1);
        }

        // A `while` consumes its header as the condition, so the header is
        // placed here rather than walked as an ordinary block of the body.
        if kind == StructuredLoop::While {
            self.visited.insert(header);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);
        }

        let scope = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };
        let body = self.in_scope(scope, |walk| {
            walk.frames.push(LoopFrame {
                header,
                body: body_set,
                follow: loop_follow,
                tail,
            });
            let body = match body_start {
                Some(start) => walk.emit(start, None, start == header),
                None => Region::Empty,
            };
            let body = Self::hoist_loop_exit(body, header);
            let body = walk.drop_trailing_continue(body, header);
            walk.frames.pop();
            body
        });

        let region = Region::Loop(Box::new(LoopRegion {
            header,
            kind,
            predicate,
            body,
            follow: loop_follow,
        }));
        (region, loop_follow)
    }

    /// Turns a loop whose body ends `if (c) { .. continue; } break;` into one
    /// that ends `if (!c) break; ..`.
    ///
    /// The two run identically, and the second is the shape a reader expects: a
    /// loop states its exit condition and then does its work, rather than
    /// wrapping the work in the negation of it. The form arises whenever a
    /// pre-tested loop's header does more than test — the header stays inside
    /// the body, its branch becomes a conditional, and the exit lands after it.
    ///
    /// Only an `if` with no `else` qualifies. With one, the `break` after it is
    /// reached from neither arm and the rewrite would not be the same loop.
    fn hoist_loop_exit(body: Region, header: NodeId) -> Region {
        let Region::Seq(mut parts) = body else {
            return body;
        };
        if !matches!(parts.last(), Some(Region::Break(target)) if *target == header) {
            return Region::Seq(parts);
        }
        let Some(index) = parts.len().checked_sub(2) else {
            return Region::Seq(parts);
        };
        let Some(Region::If(branch)) = parts.get(index) else {
            return Region::Seq(parts);
        };
        if branch.else_branch.is_some() || !Self::ends_with_continue(&branch.then_branch, header) {
            return Region::Seq(parts);
        }

        parts.pop();
        let Some(Region::If(branch)) = parts.pop() else {
            return Region::Seq(parts);
        };
        let IfRegion {
            predicate,
            then_branch,
            ..
        } = *branch;

        parts.push(Region::If(Box::new(IfRegion {
            predicate: predicate.negate(),
            then_branch: Region::Break(header),
            else_branch: None,
        })));
        parts.push(then_branch);
        Self::collapse(parts)
    }

    /// Returns `true` when `region` ends by continuing the loop at `header`.
    fn ends_with_continue(region: &Region, header: NodeId) -> bool {
        match region {
            Region::Continue(target) => *target == header,
            Region::Seq(parts) => parts
                .last()
                .is_some_and(|last| Self::ends_with_continue(last, header)),
            _ => false,
        }
    }

    /// Removes every `continue` that falls in the loop body's tail position,
    /// where the loop repeats anyway and the statement says nothing.
    ///
    /// Tail position is not just the body's last statement. Both arms of a
    /// trailing `if` are in tail position, as is the single arm of one without
    /// an `else`, since the missing arm also falls off the end — so
    /// `if (c) { ..; continue; }` at the end of a body loses its `continue`
    /// too. A nested loop's tail belongs to that loop, so the walk stops there.
    fn drop_trailing_continue(&mut self, body: Region, header: NodeId) -> Region {
        let mut removed = 0usize;
        let trimmed = Self::trim_tail(body, header, &mut removed);
        self.metrics.continues = self.metrics.continues.saturating_sub(removed);
        trimmed
    }

    /// Rewrites `region` with tail-position `continue`s to `header` removed,
    /// counting the removals into `removed`.
    fn trim_tail(region: Region, header: NodeId, removed: &mut usize) -> Region {
        match region {
            Region::Continue(target) if target == header => {
                *removed = removed.saturating_add(1);
                Region::Empty
            }
            Region::Seq(mut regions) => match regions.pop() {
                Some(last) => {
                    regions.push(Self::trim_tail(last, header, removed));
                    Self::collapse(regions)
                }
                None => Region::Empty,
            },
            Region::If(mut inner) => {
                inner.then_branch = Self::trim_tail(inner.then_branch, header, removed);
                inner.else_branch = inner
                    .else_branch
                    .map(|arm| Self::trim_tail(arm, header, removed))
                    .filter(|arm| *arm != Region::Empty);
                if inner.then_branch == Region::Empty && inner.else_branch.is_none() {
                    // The whole conditional was one redundant `continue`, so
                    // the branch itself says nothing about control flow — but
                    // the blocks its condition spans are placed by this `If`
                    // and nowhere else, so they stay.
                    return Self::predicate_region(&inner.predicate);
                }
                Region::If(inner)
            }
            Region::Switch(mut inner) => {
                for case in &mut inner.cases {
                    let body = std::mem::replace(&mut case.body, Region::Empty);
                    case.body = Self::trim_tail(body, header, removed);
                }
                Region::Switch(inner)
            }
            // A `continue` at the end of a protected body still runs the
            // region's `finally` on the way out, so it is not implicit.
            Region::Try(inner) => Region::Try(inner),
            other => other,
        }
    }

    /// Returns the block a header leaves the region for when its own test
    /// fails, for a header the recovery may render as the region's condition.
    ///
    /// `None` for a header that cannot be a condition, that is not a two-way
    /// branch, or whose arms do not split into one inside the region and one
    /// outside it — in each case the header names no exit and has no claim on
    /// the choice.
    fn tested_exit(&self, header: NodeId, inside: &impl Fn(NodeId) -> bool) -> Option<NodeId> {
        if !self.condition_only.contains(header) && !self.condition_expressible.contains(header) {
            return None;
        }
        let arms = self.arms(header);
        if arms.len() != 2 {
            return None;
        }
        arms.iter()
            .find(|(target, _)| !inside(*target))
            .map(|(target, _)| *target)
            .filter(|_| arms.iter().any(|(target, _)| inside(*target)))
    }

    /// Chooses the block control resumes at after a region.
    ///
    /// A loop or a protected body may leave to several different blocks. One of
    /// them becomes the follow and the rest are reached by goto, so the choice
    /// decides how much of the function stays structured: the immediate
    /// post-dominator of the header is the block every exit path reaches, and
    /// is preferred. Failing that the most frequently targeted exit wins,
    /// breaking ties on the lowest id so the recovery does not depend on edge
    /// order.
    fn pick_follow(
        &self,
        header: NodeId,
        members: impl Iterator<Item = NodeId>,
        inside: impl Fn(NodeId) -> bool,
    ) -> Option<NodeId> {
        let mut counts: BTreeMap<NodeId, usize> = BTreeMap::new();
        for node in members {
            for successor in self.successors(node) {
                if !inside(successor) {
                    let slot = counts.entry(successor).or_insert(0);
                    *slot = slot.saturating_add(1);
                }
            }
        }
        if counts.is_empty() {
            return None;
        }

        if let Some(candidate) = self.post_dominators.immediate_post_dominator(header)
            && counts.contains_key(&candidate)
        {
            return Some(candidate);
        }

        // No exit catches every path out. A header that can be the region's
        // condition still names one: the block it leaves to when its test
        // fails, which is what `while` means. Preferring it over the exit with
        // the most edges is what lets the condition be written as the loop's
        // own, instead of as an `if ... break` inside an endless body — the
        // same control flow spelled longer.
        if let Some(tested) = self.tested_exit(header, &inside)
            && counts.contains_key(&tested)
        {
            return Some(tested);
        }

        counts
            .iter()
            .max_by_key(|(node, count)| (**count, std::cmp::Reverse(**node)))
            .map(|(node, _)| *node)
    }

    /// Classifies a loop by where it tests its continuation condition.
    ///
    /// A loop that tests nowhere structured is still a loop; it becomes
    /// [`StructuredLoop::Endless`] and its exits become `break`.
    fn classify(
        &self,
        header: NodeId,
        body: &BitSet,
        latches: &[NodeId],
        follow: Option<NodeId>,
    ) -> LoopShape {
        self.pre_tested(header, body, follow)
            .or_else(|| self.post_tested(header, body, latches, follow))
            .unwrap_or(LoopShape {
                kind: StructuredLoop::Endless,
                predicate: None,
                tail: None,
                body_start: Some(header),
            })
    }

    /// Recognises a loop whose header branches out of the body.
    ///
    /// Two conditions, and the second is the one that is easy to miss.
    ///
    /// The exit must be the block the loop resumes at. A header that branches
    /// somewhere else leaves by a path the `while` form cannot express, so the
    /// loop is not pre-tested even though its header does branch.
    ///
    /// And the header must compute **nothing but its own condition**. A `while`
    /// evaluates its condition before every iteration, but there is nowhere in
    /// the form to put a header's other statements except before the loop,
    /// where they run once. So a header that loads the value it tests, or
    /// computes one the body also reads, is not a `while` header: rendering it
    /// as one hoists that work out of the loop and produces code that does not
    /// loop. Such a loop is [`StructuredLoop::Endless`] instead, which puts the
    /// whole header inside the body and leaves by `break` — longer to read, and
    /// right.
    fn pre_tested(
        &self,
        header: NodeId,
        body: &BitSet,
        follow: Option<NodeId>,
    ) -> Option<LoopShape> {
        // Either the header carries nothing that must be shown, or the caller
        // can show it inside the condition. A `while` puts its header nowhere
        // else, so those are the two ways one can exist.
        if !self.condition_only.contains(header) && !self.condition_expressible.contains(header) {
            return None;
        }
        let arms = self.arms(header);
        if arms.len() != 2 {
            return None;
        }
        let (entry, selectors) = arms
            .iter()
            .find(|(target, _)| body.contains_checked(target.index()))?;
        let (exit, _) = arms
            .iter()
            .find(|(target, _)| !body.contains_checked(target.index()))?;

        (Some(*exit) == follow).then(|| LoopShape {
            kind: StructuredLoop::While,
            predicate: Some(Predicate::Test {
                block: header,
                negated: selectors.first().copied() != Some(0),
            }),
            tail: None,
            // A header that re-enters itself is the whole loop: it is both the
            // condition and the work, and the condition already carries it.
            // Emitting the header as the body too would re-place a block that
            // has already been placed, which is a jump back into it — the
            // `for (;;) { goto header; }` shape.
            body_start: (*entry != header).then_some(*entry),
        })
    }

    /// Recognises a loop whose single latch branches out of the body.
    ///
    /// Several latches cannot be post-tested: there is no one place the
    /// condition lives, so the loop reads as endless with a `continue` per
    /// latch instead.
    fn post_tested(
        &self,
        header: NodeId,
        body: &BitSet,
        latches: &[NodeId],
        follow: Option<NodeId>,
    ) -> Option<LoopShape> {
        let [latch] = latches else { return None };
        if *latch == header {
            return None;
        }
        let arms = self.arms(*latch);
        if arms.len() != 2 {
            return None;
        }
        let (_, selectors) = arms.iter().find(|(target, _)| *target == header)?;
        let (exit, _) = arms
            .iter()
            .find(|(target, _)| !body.contains_checked(target.index()))?;

        (Some(*exit) == follow).then(|| LoopShape {
            kind: StructuredLoop::DoWhile,
            predicate: Some(Predicate::Test {
                block: *latch,
                negated: selectors.first().copied() != Some(0),
            }),
            tail: Some(*latch),
            body_start: Some(header),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::graph::DirectedGraph;

    /// The walk over the graph type every fixture here builds.
    ///
    /// Names the type parameter for the handful of associated functions that
    /// are tested directly and use neither `self` nor the graph.
    type Walk<'a> = Structurer<'a, DirectedGraph<'static, (), ()>>;

    /// A graph whose adjacency the test writes out directly.
    ///
    /// [`DirectedGraph`] cannot hold an edge to a node it does not have, so it
    /// cannot produce the one input the successor guard exists for: a host
    /// building a CFG from stale rows, whose successor ids outlive the blocks
    /// they named.
    struct RawGraph {
        /// Successor ids per node, unchecked.
        edges: Vec<Vec<NodeId>>,
    }

    impl GraphBase for RawGraph {
        fn node_count(&self) -> usize {
            self.edges.len()
        }

        fn node_ids(&self) -> impl Iterator<Item = NodeId> {
            (0..self.edges.len()).map(NodeId::new)
        }
    }

    impl Successors for RawGraph {
        fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
            self.edges
                .get(node.index())
                .cloned()
                .unwrap_or_default()
                .into_iter()
        }
    }

    impl Predecessors for RawGraph {
        fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
            let mut found = Vec::new();
            for (index, targets) in self.edges.iter().enumerate() {
                if targets.contains(&node) {
                    found.push(NodeId::new(index));
                }
            }
            found.into_iter()
        }
    }

    /// Builds a graph of `count` nodes with the given edges, in edge order.
    fn graph_of(count: usize, edges: &[(usize, usize)]) -> DirectedGraph<'static, (), ()> {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        for _ in 0..count {
            graph.add_node(());
        }
        for (from, to) in edges {
            graph
                .add_edge(NodeId::new(*from), NodeId::new(*to), ())
                .expect("edge endpoints exist");
        }
        graph
    }

    /// Nodes reachable from `entry`.
    fn reachable(graph: &DirectedGraph<'_, (), ()>, entry: NodeId) -> BTreeSet<NodeId> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![entry];
        while let Some(node) = stack.pop() {
            if seen.insert(node) {
                stack.extend(graph.successors(node));
            }
        }
        seen
    }

    /// Every block a [`Region::Goto`] in the tree targets.
    ///
    /// Written with an explicit stack: a caller must be able to run it on a
    /// tree at the nesting bound.
    fn goto_targets(root: &Region) -> BTreeSet<NodeId> {
        let mut targets = BTreeSet::new();
        let mut stack = vec![root];
        while let Some(region) = stack.pop() {
            match region {
                Region::Goto(target) => {
                    targets.insert(*target);
                }
                Region::Seq(parts) => stack.extend(parts.iter()),
                Region::If(inner) => {
                    stack.push(&inner.then_branch);
                    stack.extend(inner.else_branch.iter());
                }
                Region::Loop(inner) => stack.push(&inner.body),
                Region::Switch(inner) => stack.extend(inner.cases.iter().map(|case| &case.body)),
                Region::Try(inner) => {
                    stack.push(&inner.body);
                    for handler in &inner.handlers {
                        stack.extend(handler.filter.iter());
                        stack.push(&handler.body);
                    }
                }
                Region::Empty | Region::Block(_) | Region::Break(_) | Region::Continue(_) => {}
            }
        }
        targets
    }

    /// How deeply the tree nests, counted with an explicit stack so the
    /// measurement itself survives a tree at the bound.
    fn region_depth(root: &Region) -> usize {
        let mut deepest = 0usize;
        let mut stack = vec![(root, 1usize)];
        while let Some((region, depth)) = stack.pop() {
            deepest = deepest.max(depth);
            let next = depth.saturating_add(1);
            match region {
                Region::Seq(parts) => stack.extend(parts.iter().map(|part| (part, next))),
                Region::If(inner) => {
                    stack.push((&inner.then_branch, next));
                    stack.extend(inner.else_branch.iter().map(|arm| (arm, next)));
                }
                Region::Loop(inner) => stack.push((&inner.body, next)),
                Region::Switch(inner) => {
                    stack.extend(inner.cases.iter().map(|case| (&case.body, next)));
                }
                Region::Try(inner) => {
                    stack.push((&inner.body, next));
                    for handler in &inner.handlers {
                        stack.extend(handler.filter.iter().map(|filter| (filter, next)));
                        stack.push((&handler.body, next));
                    }
                }
                Region::Empty
                | Region::Block(_)
                | Region::Break(_)
                | Region::Continue(_)
                | Region::Goto(_) => {}
            }
        }
        deepest
    }

    /// Asserts the invariants that hold of every recovery, whatever the graph.
    ///
    /// `resized_inputs` is checked here rather than by an assertion inside the
    /// module: these tests run with `debug_assertions` on, so a `debug_assert!`
    /// at the normalisation boundary would abort the very fixtures written to
    /// prove a short set no longer panics. The counter is the loud signal, and
    /// this is where it is read.
    fn assert_recovery_invariants(recovered: &Structured, max_depth: usize) {
        assert_eq!(
            recovered.metrics.resized_inputs, 0,
            "a fixture handed the recovery a set built for another node count"
        );
        assert_eq!(
            recovered.labels,
            goto_targets(&recovered.root),
            "labels must be exactly the goto target set"
        );
        // The bound is on the walk's descent, and one descent buys at most two
        // tree levels: the construct plus the sequence under it.
        let depth = region_depth(&recovered.root);
        let bound = max_depth.saturating_mul(2).saturating_add(4);
        assert!(
            depth <= bound,
            "tree nests {depth} deep, past the bound of {bound} for max_depth {max_depth}"
        );
    }

    /// Asserts the recovery placed every reachable block exactly once.
    ///
    /// This is the totality property. It must hold for every graph, including
    /// the irreducible and multi-latch ones a compiler would not emit, because
    /// a tree that drops or duplicates a block does not describe the function.
    fn assert_total(graph: &DirectedGraph<'_, (), ()>, entry: NodeId, recovered: &Structured) {
        assert_bounded_total(graph, entry, recovered, DEFAULT_MAX_DEPTH);
    }

    /// [`assert_total`] for a recovery run with a non-default depth bound.
    fn assert_bounded_total(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        recovered: &Structured,
        max_depth: usize,
    ) {
        assert_recovery_invariants(recovered, max_depth);

        let placed = recovered.blocks();
        let unique: BTreeSet<NodeId> = placed.iter().copied().collect();
        assert_eq!(
            unique,
            reachable(graph, entry),
            "placement != reachable set"
        );

        // Only a block that ends the function may be placed more than once,
        // and every repeat must be accounted for as a replicated tail.
        assert_eq!(
            placed.len().saturating_sub(unique.len()),
            recovered.metrics.replicated_tails,
            "unaccounted duplicate placement: {placed:?}"
        );
        for node in &unique {
            let count = placed.iter().filter(|entry| *entry == node).count();
            assert!(
                count == 1 || graph.successors(*node).next().is_none(),
                "block {node:?} placed {count} times but does not end the function"
            );
        }
    }

    /// Builds a block set over `count` nodes.
    fn set_of(count: usize, members: &[usize]) -> BlockSet {
        BlockSet::from_nodes(count, members.iter().map(|member| NodeId::new(*member)))
    }

    /// Every block placed anywhere in `region`.
    fn placed_in(region: &Region) -> BTreeSet<NodeId> {
        let mut placed = BTreeSet::new();
        region.for_each_block(&mut |node| {
            placed.insert(node);
        });
        placed
    }

    /// Asserts placement totality where a protected region contributes blocks
    /// the entry cannot reach, because the runtime takes the edge into them.
    fn assert_total_protected(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        regions: &[ProtectedRegion],
        recovered: &Structured,
    ) {
        assert_recovery_invariants(recovered, DEFAULT_MAX_DEPTH);

        let placed = recovered.blocks();
        let unique: BTreeSet<NodeId> = placed.iter().copied().collect();
        assert_eq!(
            placed.len().saturating_sub(unique.len()),
            recovered.metrics.replicated_tails,
            "unaccounted duplicate placement: {placed:?}"
        );

        let mut expected = reachable(graph, entry);
        for region in regions {
            expected.insert(region.entry);
            expected.extend(region.blocks.iter());
            for handler in &region.handlers {
                expected.insert(handler.entry);
                expected.extend(handler.blocks.iter());
                if let Some(filter) = handler.kind.filter() {
                    expected.insert(filter.entry);
                    expected.extend(filter.blocks.iter());
                }
            }
        }
        assert_eq!(unique, expected, "placement != reachable set");
    }

    /// Structures with every block declared free of side effects, which is the
    /// condition under which conditions may be merged.
    fn structure_merging(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        condition_only: &[usize],
    ) -> Structured {
        let options = StructureOptions {
            regions: &[],
            condition_only: set_of(graph.node_count(), condition_only),
            condition_expressible: BlockSet::new(graph.node_count()),
            max_depth: DEFAULT_MAX_DEPTH,
        };
        structure_with(graph, entry, &options)
    }

    /// Builds a recovery in which the merged blocks compute as well as test,
    /// which is what [`StructureOptions::condition_expressible`] admits.
    fn structure_expressing(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        condition_expressible: &[usize],
    ) -> Structured {
        let options = StructureOptions {
            regions: &[],
            condition_only: BlockSet::new(graph.node_count()),
            condition_expressible: set_of(graph.node_count(), condition_expressible),
            max_depth: DEFAULT_MAX_DEPTH,
        };
        structure_with(graph, entry, &options)
    }

    /// Structures every block as condition-only under an explicit depth bound.
    fn structure_bounded(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        regions: &[ProtectedRegion],
        max_depth: usize,
    ) -> Structured {
        let options = StructureOptions {
            regions,
            condition_only: BlockSet::full(graph.node_count()),
            condition_expressible: BlockSet::new(graph.node_count()),
            max_depth,
        };
        structure_with(graph, entry, &options)
    }

    /// Collects the loop header named by every break in the tree.
    fn collect_breaks(region: &Region, into: &mut Vec<NodeId>) {
        match region {
            Region::Break(header) => into.push(*header),
            Region::Seq(regions) => {
                for inner in regions {
                    collect_breaks(inner, into);
                }
            }
            Region::If(inner) => {
                collect_breaks(&inner.then_branch, into);
                if let Some(other) = &inner.else_branch {
                    collect_breaks(other, into);
                }
            }
            Region::Loop(inner) => collect_breaks(&inner.body, into),
            Region::Switch(inner) => {
                for case in &inner.cases {
                    collect_breaks(&case.body, into);
                }
            }
            Region::Try(inner) => {
                collect_breaks(&inner.body, into);
                for handler in &inner.handlers {
                    collect_breaks(&handler.body, into);
                }
            }
            _ => {}
        }
    }

    /// Counts protected regions in a subtree.
    fn count_tries(region: &Region, into: &mut usize) {
        match region {
            Region::Try(inner) => {
                *into = into.saturating_add(1);
                count_tries(&inner.body, into);
                for handler in &inner.handlers {
                    count_tries(&handler.body, into);
                }
            }
            Region::Seq(regions) => {
                for inner in regions {
                    count_tries(inner, into);
                }
            }
            Region::If(inner) => {
                count_tries(&inner.then_branch, into);
                if let Some(other) = &inner.else_branch {
                    count_tries(other, into);
                }
            }
            Region::Loop(inner) => count_tries(&inner.body, into),
            Region::Switch(inner) => {
                for case in &inner.cases {
                    count_tries(&case.body, into);
                }
            }
            _ => {}
        }
    }

    /// Returns the first [`Region::Try`] found anywhere in `region`.
    fn first_try(region: &Region) -> Option<&TryRegion> {
        match region {
            Region::Try(inner) => Some(inner),
            Region::Seq(parts) => parts.iter().find_map(first_try),
            Region::If(inner) => first_try(&inner.then_branch)
                .or_else(|| inner.else_branch.as_ref().and_then(first_try)),
            Region::Loop(inner) => first_try(&inner.body),
            Region::Switch(inner) => inner.cases.iter().find_map(|case| first_try(&case.body)),
            _ => None,
        }
    }

    // -- The index domain ---------------------------------------------------

    #[test]
    fn a_short_condition_only_set_does_not_panic() {
        // The set is sized for no graph at all while the graph has four nodes.
        // Every read of it inside the walk used to be `assert!(index < len)`,
        // under a doc that says the recovery always succeeds.
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let options = StructureOptions {
            regions: &[],
            condition_only: BlockSet::new(0),
            condition_expressible: BlockSet::new(0),
            max_depth: DEFAULT_MAX_DEPTH,
        };
        let recovered = structure_with(&graph, NodeId::new(0), &options);

        assert_eq!(
            recovered.metrics.resized_inputs, 2,
            "both sets were resized"
        );
        assert_eq!(recovered.blocks().len(), 4, "and every block still placed");
    }

    #[test]
    fn a_region_sized_to_itself_does_not_panic() {
        // The region's set is sized to the region rather than to the function,
        // which `pick_follow` used to read past the end of.
        let graph = graph_of(7, &[(0, 1), (1, 2), (2, 5), (3, 4), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(3, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(5, &[3, 4]),
            }],
        }];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_eq!(recovered.metrics.resized_inputs, 2);
        assert_eq!(recovered.metrics.tries, 1);
        let placed = placed_in(&recovered.root);
        assert!(
            placed.contains(&NodeId::new(3)) && placed.contains(&NodeId::new(4)),
            "the handler's blocks are still placed: {placed:?}"
        );
    }

    #[test]
    fn an_entry_outside_the_graph_is_an_empty_recovery() {
        let graph = graph_of(3, &[(0, 1), (1, 2)]);
        let recovered = structure(&graph, NodeId::new(9));
        assert_eq!(recovered.root, Region::Empty);
        assert_eq!(recovered.metrics.blocks, 0);
        assert_eq!(recovered.metrics.unreached, 3);
        assert!(recovered.labels.is_empty());
    }

    #[test]
    fn an_out_of_range_successor_contributes_no_edge() {
        // Block 0 branches to block 1 and to an id the graph does not hold, so
        // it is a one-armed block and not a conditional.
        let graph = RawGraph {
            edges: vec![vec![NodeId::new(1), NodeId::new(7)], Vec::new()],
        };
        let recovered = structure(&graph, NodeId::new(0));

        assert_eq!(recovered.metrics.ifs, 0, "an absent id is not a second arm");
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(
            recovered.blocks(),
            vec![NodeId::new(0), NodeId::new(1)],
            "and both real blocks are still placed"
        );
    }

    #[test]
    fn a_recovery_within_its_bounds_reports_no_degradation() {
        // The counts that say the recovery gave something up, read out by
        // name. A plain diamond gives nothing up, and naming them here is what
        // keeps a count from being added and then never looked at.
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let recovered = structure(&graph, NodeId::new(0));
        let StructureMetrics {
            blocks,
            unreached,
            resized_inputs,
            depth_limited,
            replicated_tails,
            ..
        } = recovered.metrics;
        assert_eq!(blocks, 4);
        assert_eq!(unreached, 0);
        assert_eq!(resized_inputs, 0);
        assert_eq!(depth_limited, 0);
        assert_eq!(replicated_tails, 0);
    }

    #[test]
    fn a_block_set_is_total_over_every_node_id() {
        let set = BlockSet::from_nodes(4, [NodeId::new(1), NodeId::new(9)]);
        assert_eq!(set.node_count(), 4);
        assert!(set.contains(NodeId::new(1)));
        assert!(
            !set.contains(NodeId::new(9)),
            "out of domain is not a member"
        );
        assert!(!set.contains(NodeId::new(usize::MAX)));
        assert_eq!(set.count(), 1);

        let widened = BlockSet::from_bits(6, BitSet::new(2));
        assert_eq!(widened.node_count(), 6);
        assert_eq!(BlockSet::full(3).count(), 3);
    }

    // -- The depth of the output --------------------------------------------

    #[test]
    fn nested_diamonds_are_bounded_by_max_depth() {
        // `if (c0) { if (c1) { .. } }` nested 2000 deep: test i branches into
        // test i+1 or out to exit i, and exit i+1 falls into exit i, so exit i
        // post-dominates test i and every level is a real nesting level.
        let depth: usize = 2000;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for index in 0..depth {
            let exit = depth.saturating_add(index);
            if index.saturating_add(1) < depth {
                edges.push((index, index.saturating_add(1)));
                edges.push((index, exit));
                edges.push((exit.saturating_add(1), exit));
            } else {
                edges.push((index, depth.saturating_mul(2)));
                edges.push((index, exit));
                edges.push((depth.saturating_mul(2), exit));
            }
        }
        let graph = graph_of(depth.saturating_mul(2).saturating_add(1), &edges);
        let entry = NodeId::new(0);
        let recovered = structure(&graph, entry);

        assert_total(&graph, entry, &recovered);
        assert!(
            recovered.metrics.depth_limited > 0,
            "2000 levels must reach the bound of {DEFAULT_MAX_DEPTH}"
        );

        // The derived traversals recurse to the tree's depth in the caller, so
        // the bound has to hold for them too.
        let copy = recovered.clone();
        assert_eq!(copy, recovered);
        drop(copy);
        drop(recovered);
    }

    #[test]
    fn a_chain_of_conditionals_does_not_recurse_per_conditional() {
        // The early-return spine: `if (x) return; if (y) return; ..`, in both
        // edge orders, because which arm the front end puts first is not the
        // recovery's choice. Neither order may nest.
        let tests: usize = 10_000;
        let last = tests.saturating_mul(2);
        for exit_first in [false, true] {
            let mut edges: Vec<(usize, usize)> = Vec::new();
            for index in 0..tests {
                let sink = tests.saturating_add(index);
                let next = if index.saturating_add(1) < tests {
                    index.saturating_add(1)
                } else {
                    last
                };
                if exit_first {
                    edges.push((index, sink));
                    edges.push((index, next));
                } else {
                    edges.push((index, next));
                    edges.push((index, sink));
                }
            }
            let graph = graph_of(last.saturating_add(1), &edges);
            let entry = NodeId::new(0);
            let recovered = structure(&graph, entry);

            assert_total(&graph, entry, &recovered);
            assert_eq!(
                recovered.metrics.depth_limited, 0,
                "an early-return spine must not nest (exit arm first: {exit_first})"
            );
            assert_eq!(recovered.metrics.ifs, tests);
        }
    }

    #[test]
    fn a_long_merged_condition_is_bounded() {
        // `if (a0 && a1 && ..)` a thousand terms long. Every fold nests the
        // predicate one level when it alternates, so the fold count is bounded
        // by the same number as the tree's depth.
        let tests: usize = 1000;
        let success = tests;
        let failure = tests.saturating_add(1);
        let end = tests.saturating_add(2);
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for index in 0..tests {
            let next = if index.saturating_add(1) < tests {
                index.saturating_add(1)
            } else {
                success
            };
            edges.push((index, next));
            edges.push((index, failure));
        }
        edges.push((success, end));
        edges.push((failure, end));

        let graph = graph_of(end.saturating_add(1), &edges);
        let entry = NodeId::new(0);
        let recovered = structure_bounded(&graph, entry, &[], 8);

        assert_bounded_total(&graph, entry, &recovered, 8);
        assert!(
            recovered.metrics.depth_limited > 0,
            "folding must stop at the bound"
        );
        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert!(
            branch.predicate.tests() <= 9,
            "at most eight folds beyond the head test, got {}",
            branch.predicate.tests()
        );
    }

    #[test]
    fn max_depth_of_zero_still_terminates() {
        // Clamped to one, so the outermost walk is never the limited one and
        // the orphan pass always makes progress.
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let entry = NodeId::new(0);
        let recovered = structure_bounded(&graph, entry, &[], 0);

        assert_bounded_total(&graph, entry, &recovered, 1);
        assert!(recovered.metrics.depth_limited > 0);
    }

    #[test]
    fn a_depth_limited_protected_body_keeps_its_blocks_inside_the_try() {
        // A goto out of a confined walk would place a block the region owns at
        // top level, outside the `Region::Try` — protected code printed as
        // unprotected. The flat emission keeps it in.
        let graph = graph_of(7, &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(7, &[1, 2, 3, 4]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Finally,
                entry: NodeId::new(6),
                blocks: set_of(7, &[6]),
            }],
        }];
        let entry = NodeId::new(0);
        let recovered = structure_bounded(&graph, entry, &regions, 2);

        assert!(recovered.metrics.depth_limited > 0);
        let protected = first_try(&recovered.root).expect("a try region");
        let inside = placed_in(&protected.body);
        for block in [1, 2, 3, 4] {
            assert!(
                inside.contains(&NodeId::new(block)),
                "block {block} belongs to the region and must stay inside it: {inside:?}"
            );
        }
    }

    // -- Protected regions --------------------------------------------------

    #[test]
    fn two_regions_sharing_an_entry_both_open() {
        // `try { try { .. } catch { .. } } finally { .. }` is the ordinary
        // encoding: two clauses, one try start. Declared innermost first, as
        // an exception table records them.
        let graph = graph_of(6, &[(0, 1), (1, 2), (2, 5), (3, 5), (4, 5)]);
        let regions = vec![
            ProtectedRegion {
                entry: NodeId::new(1),
                blocks: set_of(6, &[1]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(3),
                    blocks: set_of(6, &[3]),
                }],
            },
            ProtectedRegion {
                entry: NodeId::new(1),
                blocks: set_of(6, &[1, 2]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Finally,
                    entry: NodeId::new(4),
                    blocks: set_of(6, &[4]),
                }],
            },
        ];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 2, "both clauses open");

        let outer = first_try(&recovered.root).expect("the outer try");
        assert_eq!(
            outer.handlers.first().map(|handler| handler.kind),
            Some(HandlerKind::Finally),
            "the region covering more blocks is the containing one"
        );
        let inner = first_try(&outer.body).expect("the inner try inside the outer body");
        assert_eq!(
            inner.handlers.first().map(|handler| handler.kind),
            Some(HandlerKind::Catch),
            "and the smaller region opens inside it"
        );
    }

    #[test]
    fn an_unreached_protected_region_still_opens_its_try() {
        // Nothing reaches block 1, so the walk never enters the region. Its
        // handler must still be emitted inside a `Region::Try`: a bare
        // labelled block at top level prints protected code as unprotected.
        let graph = graph_of(5, &[(0, 4), (1, 2)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(5, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(5, &[3]),
            }],
        }];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 1);

        let protected = first_try(&recovered.root).expect("a try region");
        assert_eq!(protected.entry, NodeId::new(1));
        assert_eq!(
            protected.handlers.first().map(|handler| &handler.body),
            Some(&Region::Block(NodeId::new(3))),
            "the handler is inside the try, not beside it"
        );
    }

    // -- One scope rule -----------------------------------------------------

    #[test]
    fn a_folded_join_is_not_a_goto_target() {
        // `for (;;) { A; if (c) continue; if (d) continue; break; }`. Block 3
        // is the join of the branch at 2 and is also folded into its
        // condition, so it has no statement position left to resume at.
        let graph = graph_of(5, &[(0, 1), (1, 2), (2, 1), (2, 3), (3, 1), (3, 4)]);
        let entry = NodeId::new(0);
        let recovered = structure(&graph, entry);

        assert_total(&graph, entry, &recovered);
        assert!(
            !recovered.labels.contains(&NodeId::new(3)),
            "a folded block has no label position: {:?}",
            recovered.labels
        );
        assert_eq!(recovered.metrics.merged_conditions, 1);
    }

    #[test]
    fn a_self_looping_while_has_an_empty_body() {
        // The header is the whole loop: it is both the condition and the work,
        // and a `while` already runs its condition every iteration. Emitting
        // the header as the body too would place it twice.
        let graph = graph_of(3, &[(0, 1), (1, 1), (1, 2)]);
        let entry = NodeId::new(0);
        let recovered = structure(&graph, entry);

        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(looped.body, Region::Empty);
    }

    #[test]
    fn a_handler_conditional_inside_a_loop_recovers_as_an_if() {
        // The handler holds a diamond rejoining at 7, which lies outside the
        // loop the protected region sits in. The runtime enters the handler,
        // not the loop's back edge, so the loop's body does not bound the
        // handler's joins — and without that barrier the diamond degrades to
        // an over-wide arm plus a goto.
        let graph = graph_of(
            8,
            &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 1),
                (3, 4),
                (3, 5),
                (4, 7),
                (5, 7),
            ],
        );
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(2),
            blocks: set_of(8, &[2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(8, &[3, 4, 5, 7]),
            }],
        }];
        let entry = NodeId::new(0);
        let recovered = structure_protected(&graph, entry, &regions);

        assert_total_protected(&graph, entry, &regions, &recovered);
        assert_eq!(recovered.metrics.loops, 1);
        assert_eq!(
            recovered.metrics.if_elses, 1,
            "the handler's diamond is an if/else"
        );
        assert!(
            !recovered.labels.contains(&NodeId::new(7)),
            "the diamond rejoins in tree order: {:?}",
            recovered.labels
        );
    }

    #[test]
    fn a_handler_that_continues_the_enclosing_loop_still_continues() {
        // The barrier stops the enclosing loop from *bounding* the handler's
        // walk. It must not stop the handler from naming that loop: the try is
        // emitted lexically inside the loop body, so a jump from the handler
        // back to the header is a `continue` and not a goto.
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 5), (2, 4), (4, 1), (3, 1)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(2),
            blocks: set_of(6, &[2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(6, &[3]),
            }],
        }];
        let entry = NodeId::new(0);
        let recovered = structure_protected(&graph, entry, &regions);

        assert_total_protected(&graph, entry, &regions, &recovered);
        let protected = first_try(&recovered.root).expect("a try region");
        assert_eq!(
            protected.handlers.first().map(|handler| &handler.body),
            Some(&Region::Seq(vec![
                Region::Block(NodeId::new(3)),
                Region::Continue(NodeId::new(1)),
            ])),
            "a handler still continues the loop it sits in"
        );
    }

    // -- No construction site drops a block ---------------------------------

    #[test]
    fn predicate_region_places_every_test_block() {
        let predicate = Predicate::All(vec![
            Predicate::Test {
                block: NodeId::new(0),
                negated: false,
            },
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: true,
                },
                Predicate::Test {
                    block: NodeId::new(2),
                    negated: false,
                },
            ]),
        ]);

        let region = Walk::predicate_region(&predicate);
        let mut placed = Vec::new();
        region.for_each_block(&mut |node| placed.push(node));
        assert_eq!(
            placed,
            vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
            "every block the condition spans keeps a statement position"
        );
    }

    #[test]
    fn trim_tail_keeps_the_predicate_blocks() {
        // `if (c) continue;` at a loop body's tail says nothing about control
        // flow, but the block the condition tests in is placed by this `if`
        // and nowhere else.
        let header = NodeId::new(0);
        let region = Region::If(Box::new(IfRegion {
            predicate: Predicate::Test {
                block: NodeId::new(1),
                negated: false,
            },
            then_branch: Region::Continue(header),
            else_branch: None,
        }));

        let mut removed = 0usize;
        let trimmed = Walk::trim_tail(region, header, &mut removed);
        assert_eq!(removed, 1);
        assert_eq!(trimmed, Region::Block(NodeId::new(1)));
    }

    // -- The pre-existing recovery ------------------------------------------

    #[test]
    fn straight_line_is_a_sequence() {
        let graph = graph_of(3, &[(0, 1), (1, 2)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.ifs, 0);
        assert_eq!(
            recovered.root,
            Region::Seq(vec![
                Region::Block(NodeId::new(0)),
                Region::Block(NodeId::new(1)),
                Region::Block(NodeId::new(2)),
            ])
        );
    }

    #[test]
    fn diamond_is_an_if_else() {
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.if_elses, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        // The conditional owns the block it branches from, so it is the first
        // element and the join follows it.
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if first, got {top:?}");
        };
        assert_eq!(branch.then_branch, Region::Block(NodeId::new(1)));
        assert_eq!(branch.else_branch, Some(Region::Block(NodeId::new(2))));
        assert_eq!(top.get(1), Some(&Region::Block(NodeId::new(3))));
    }

    #[test]
    fn empty_then_arm_inverts_rather_than_printing_an_empty_block() {
        // 0 branches to 2 directly on its first edge; only the second edge has
        // a body. The `then` arm must become that body, with then_index naming
        // the second successor so the printer negates.
        let graph = graph_of(3, &[(0, 2), (0, 1), (1, 2)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Test {
                block: NodeId::new(0),
                negated: true
            },
            "condition must read as negated"
        );
        assert_eq!(branch.then_branch, Region::Block(NodeId::new(1)));
        assert_eq!(branch.else_branch, None);
    }

    #[test]
    fn pre_tested_loop_is_a_while() {
        // 0 -> 1 header; 1 -> 2 body -> 1; 1 -> 3 exit.
        let graph = graph_of(4, &[(0, 1), (1, 2), (1, 3), (2, 1)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.loops, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(
            looped.predicate,
            Some(Predicate::Test {
                block: NodeId::new(1),
                negated: false
            })
        );
        assert_eq!(looped.follow, Some(NodeId::new(3)));
        assert_eq!(looped.body, Region::Block(NodeId::new(2)));
    }

    /// A header that does work the condition depends on is not a `while`
    /// header, because `while` runs the header once and the condition always.
    #[test]
    fn a_header_that_computes_more_than_its_condition_is_not_a_while() {
        // 0 -> 1 header; 1 -> 2 body -> 1; 1 -> 3 exit. The header loads the
        // value it tests, so hoisting it above the loop would load once and
        // then spin.
        let graph = graph_of(4, &[(0, 1), (1, 2), (1, 3), (2, 1)]);
        let entry = NodeId::new(0);

        // Cleared: the header is exactly a test, so `while` is exact.
        let cleared = structure_merging(&graph, entry, &[0, 1, 2, 3]);
        let Region::Seq(top) = &cleared.root else {
            panic!("expected a sequence, got {:?}", cleared.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);

        // Not cleared: the loop keeps its header inside the body and leaves by
        // `break`, which runs the header every iteration.
        let uncleared = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &uncleared);
        let Region::Seq(top) = &uncleared.root else {
            panic!("expected a sequence, got {:?}", uncleared.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(
            looped.kind,
            StructuredLoop::Endless,
            "an unclear header must stay inside the loop"
        );
        assert_eq!(uncleared.metrics.breaks, 1, "and leave by break");

        let mut placed = Vec::new();
        looped.body.for_each_block(&mut |node| placed.push(node));
        assert!(
            placed.contains(&NodeId::new(1)),
            "the header runs each iteration, so it is part of the body: {placed:?}"
        );
    }

    #[test]
    fn post_tested_loop_is_a_do_while() {
        // 0 -> 1 header -> 2 latch; latch branches back to 1 or out to 3. The
        // latch terminator is the condition, so it must not also appear as a
        // conditional inside the body.
        let graph = graph_of(4, &[(0, 1), (1, 2), (2, 1), (2, 3)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.ifs, 0, "the latch test is the loop test");

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::DoWhile);
        assert_eq!(
            looped.predicate,
            Some(Predicate::Test {
                block: NodeId::new(2),
                negated: false
            })
        );
        assert_eq!(
            looped.body,
            Region::Seq(vec![
                Region::Block(NodeId::new(1)),
                Region::Block(NodeId::new(2)),
            ])
        );
    }

    #[test]
    fn loop_tested_in_the_middle_is_endless_with_a_break() {
        // 0 -> 1 -> 2; 2 branches to 3 (exit) or 4; 4 -> 1. Neither the header
        // nor the latch tests, so this is `for (;;) { ..; if (c) break; .. }`.
        let graph = graph_of(5, &[(0, 1), (1, 2), (2, 3), (2, 4), (4, 1)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1);
        assert_eq!(recovered.metrics.endless_loops, 1);
        assert_eq!(recovered.metrics.breaks, 1);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    #[test]
    fn infinite_loop_has_no_follow() {
        let graph = graph_of(3, &[(0, 1), (1, 2), (2, 1)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.endless_loops, 1);
        assert_eq!(recovered.metrics.breaks, 0);
        assert_eq!(
            recovered.metrics.continues, 0,
            "a loop that only falls off its own end needs no continue"
        );

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.follow, None);
    }

    /// BinKit contains no multi-latch loop, so this shape exists only here.
    #[test]
    fn multi_latch_loop_continues_from_every_latch() {
        // Header 1 with two latches, 3 and 4, and an exit at 5.
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 5), (2, 3), (2, 4), (3, 1), (4, 1)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1, "two latches are still one loop");
        assert_eq!(recovered.metrics.gotos, 0, "two latches still structure");

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(looped.follow, Some(NodeId::new(5)));
        // Both latches reach the header by falling off the end of the body, so
        // neither needs to say `continue`.
        assert_eq!(
            looped.body,
            Region::If(Box::new(IfRegion {
                predicate: Predicate::Test {
                    block: NodeId::new(2),
                    negated: false,
                },
                then_branch: Region::Block(NodeId::new(3)),
                else_branch: Some(Region::Block(NodeId::new(4))),
            }))
        );
    }

    /// Also absent from BinKit: a cycle with two entries, which has no header
    /// dominating it and so is not a natural loop at all.
    #[test]
    fn irreducible_cycle_falls_back_to_a_goto() {
        // 0 branches into either side of the 1 <-> 2 cycle.
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (1, 3)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert!(
            recovered.metrics.gotos >= 1,
            "an irreducible cycle needs at least one goto"
        );
        assert!(
            !recovered.labels.is_empty(),
            "the goto target must be labelled"
        );
    }

    #[test]
    fn a_header_that_tests_names_the_follow_when_no_exit_catches_every_path() {
        // Header 1 leaves to 4, body block 2 leaves to 5, and the two never
        // meet — so no exit post-dominates the header and the counting rule
        // would pick 5 for having the lower id. Taking 1's own exit instead is
        // what makes the loop a `while` on 1's test rather than an endless body
        // that opens with `if (!c) break`.
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 4), (2, 3), (2, 5), (3, 1)]);
        let entry = NodeId::new(0);
        let recovered = structure(&graph, entry);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.loops, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.follow, Some(NodeId::new(4)));
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(recovered.metrics.endless_loops, 0);
    }

    #[test]
    fn loop_leaving_to_several_blocks_keeps_one_follow() {
        // Header 1 leaves to 4; body block 2 leaves to 5. Only one of them can
        // be the block control resumes at, and the choice must not depend on
        // the order the edges were added.
        let forward = graph_of(6, &[(0, 1), (1, 2), (1, 4), (2, 3), (2, 5), (3, 1), (4, 5)]);
        let reversed = graph_of(6, &[(4, 5), (3, 1), (2, 5), (2, 3), (1, 4), (1, 2), (0, 1)]);

        let recovered = structure(&forward, NodeId::new(0));
        assert_total(&forward, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        // 5 post-dominates the header: every way out of the loop reaches it,
        // so it is the follow and the exit through 4 stays inside the region.
        assert_eq!(looped.follow, Some(NodeId::new(5)));
        assert_eq!(recovered.metrics.breaks, 1);

        let other = structure(&reversed, NodeId::new(0));
        assert_eq!(
            other.metrics, recovered.metrics,
            "follow selection must not depend on edge order"
        );
    }

    #[test]
    fn multiway_branch_is_a_switch() {
        let graph = graph_of(
            6,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
            ],
        );
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.switches, 1);
        assert_eq!(recovered.metrics.gotos, 0);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Switch(switch)) = top.first() else {
            panic!("expected a switch, got {top:?}");
        };
        assert_eq!(switch.cases.len(), 4);
        assert_eq!(switch.follow, Some(NodeId::new(5)));
    }

    #[test]
    fn shared_case_targets_collapse_to_one_arm() {
        // Selectors 0 and 2 reach block 1; selector 1 reaches block 2. That is
        // `case A: case C:` sharing a body, not two arms with equal contents.
        let graph = graph_of(4, &[(0, 1), (0, 2), (0, 1), (1, 3), (2, 3)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        // Two distinct targets: this reads as a conditional, not a switch.
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Test {
                block: NodeId::new(0),
                negated: false
            }
        );
    }

    #[test]
    fn case_fall_through_reaches_the_next_arm_by_goto() {
        // Arm 1 falls into arm 2 instead of rejoining.
        let graph = graph_of(5, &[(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.switches, 1);
        assert!(
            recovered.labels.contains(&NodeId::new(2)),
            "the arm fallen into must be labelled, got {:?}",
            recovered.labels
        );
    }

    #[test]
    fn nested_loops_break_the_loop_they_name() {
        // Outer header 1, inner header 2. The inner body at 3 leaves straight
        // to the outer follow at 6, which is a break of the *outer* loop.
        let graph = graph_of(
            7,
            &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 3),
                (2, 5),
                (3, 6),
                (3, 4),
                (4, 2),
                (5, 1),
            ],
        );
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 2);

        let mut targets = Vec::new();
        collect_breaks(&recovered.root, &mut targets);
        assert!(
            targets.contains(&NodeId::new(1)),
            "expected a break naming the outer header, got {targets:?}"
        );
    }

    #[test]
    fn unreachable_blocks_are_counted_not_placed() {
        let graph = graph_of(4, &[(0, 1), (2, 3)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_eq!(recovered.metrics.blocks, 2);
        assert_eq!(recovered.metrics.unreached, 2);
        assert_eq!(recovered.blocks().len(), 2);
    }

    #[test]
    fn single_block_function_structures() {
        let graph = graph_of(1, &[]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.root, Region::Block(NodeId::new(0)));
        assert_eq!(recovered.metrics.gotos, 0);
    }

    #[test]
    fn self_loop_is_a_loop() {
        let graph = graph_of(3, &[(0, 1), (1, 1), (1, 2)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1);
    }

    #[test]
    fn both_edges_to_one_block_is_not_a_branch() {
        let graph = graph_of(2, &[(0, 1), (0, 1)]);
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.ifs, 0);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    #[test]
    fn deep_chain_does_not_overflow_the_stack() {
        // A long straight line takes the iterative path: no region is entered,
        // so the walk never descends and the depth bound is never near. The
        // recursive dimension is covered by the two tests above it.
        let count: usize = 20_000;
        let edges: Vec<(usize, usize)> = (0..count.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        let graph = graph_of(count, &edges);
        let recovered = structure(&graph, NodeId::new(0));
        assert_eq!(recovered.metrics.blocks, count);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    #[test]
    fn early_return_from_a_nested_arm_leaves_by_goto_not_by_duplication() {
        // Inner arm 3 returns at 6, which is also where the outer join leads.
        let graph = graph_of(
            7,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (3, 6),
                (4, 5),
                (2, 5),
                (5, 6),
            ],
        );
        let recovered = structure(&graph, NodeId::new(0));
        assert_total(&graph, NodeId::new(0), &recovered);
    }

    #[test]
    fn protected_region_becomes_a_try_with_its_handler() {
        // Body 1 -> 2 -> 5 (follow). Handler 3 -> 4, reachable only by raising.
        let graph = graph_of(6, &[(0, 1), (1, 2), (2, 5), (3, 4)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(6, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(6, &[3, 4]),
            }],
        }];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 1);
        assert_eq!(recovered.metrics.handlers, 1);
        assert_eq!(recovered.metrics.gotos, 0);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        assert_eq!(protected.follow, Some(NodeId::new(5)));
        assert_eq!(
            protected.body,
            Region::Seq(vec![
                Region::Block(NodeId::new(1)),
                Region::Block(NodeId::new(2)),
            ])
        );
        assert_eq!(protected.handlers.len(), 1);
        assert_eq!(
            protected.handlers.first().map(|h| &h.body),
            Some(&Region::Seq(vec![
                Region::Block(NodeId::new(3)),
                Region::Block(NodeId::new(4)),
            ]))
        );
        // The follow is placed after the try, not inside it.
        assert_eq!(top.get(2), Some(&Region::Block(NodeId::new(5))));
    }

    #[test]
    fn several_clauses_on_one_region_are_several_handlers() {
        let graph = graph_of(6, &[(0, 1), (1, 5), (2, 5), (3, 5), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(6, &[1]),
            handlers: vec![
                ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(2),
                    blocks: set_of(6, &[2]),
                },
                ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(3),
                    blocks: set_of(6, &[3]),
                },
                ProtectedHandler {
                    kind: ProtectedHandlerKind::Finally,
                    entry: NodeId::new(4),
                    blocks: set_of(6, &[4]),
                },
            ],
        }];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.handlers, 3);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        let kinds: Vec<HandlerKind> = protected.handlers.iter().map(|h| h.kind).collect();
        assert_eq!(
            kinds,
            vec![HandlerKind::Catch, HandlerKind::Catch, HandlerKind::Finally],
            "clause order is the declared order"
        );
    }

    #[test]
    fn filter_expression_is_recovered_separately_from_the_handler_body() {
        // Filter at 2 decides; handler body at 3.
        let graph = graph_of(5, &[(0, 1), (1, 4), (3, 4)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(5, &[1]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Filter(HandlerFilter {
                    entry: NodeId::new(2),
                    blocks: set_of(5, &[2]),
                }),
                entry: NodeId::new(3),
                blocks: set_of(5, &[3]),
            }],
        }];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        let handler = protected.handlers.first().expect("one handler");
        assert_eq!(handler.filter, Some(Region::Block(NodeId::new(2))));
        assert_eq!(handler.body, Region::Block(NodeId::new(3)));
    }

    #[test]
    fn nested_protected_regions_nest_in_the_tree() {
        // Outer try covers 1..3, inner try covers 2 alone.
        let graph = graph_of(7, &[(0, 1), (1, 2), (2, 3), (3, 6), (4, 6), (5, 6)]);
        let regions = vec![
            ProtectedRegion {
                entry: NodeId::new(1),
                blocks: set_of(7, &[1, 2, 3]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(4),
                    blocks: set_of(7, &[4]),
                }],
            },
            ProtectedRegion {
                entry: NodeId::new(2),
                blocks: set_of(7, &[2]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Finally,
                    entry: NodeId::new(5),
                    blocks: set_of(7, &[5]),
                }],
            },
        ];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 2);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(outer)) = top.get(1) else {
            panic!("expected the outer try, got {top:?}");
        };
        assert_eq!(outer.entry, NodeId::new(1));
        let mut nested = 0usize;
        count_tries(&outer.body, &mut nested);
        assert_eq!(nested, 1, "the inner try is inside the outer body");
    }

    #[test]
    fn leaving_a_protected_region_early_does_not_drag_code_inside_it() {
        // Block 2, inside the try, jumps to 5 while the region's follow is 4.
        // 5 belongs to neither, so it must not be placed within the try.
        let graph = graph_of(7, &[(0, 1), (1, 2), (1, 4), (2, 5), (4, 6), (5, 6)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(7, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Finally,
                entry: NodeId::new(3),
                blocks: set_of(7, &[3]),
            }],
        }];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        let mut inside = Vec::new();
        protected.body.for_each_block(&mut |node| inside.push(node));
        assert!(
            !inside.contains(&NodeId::new(5)),
            "block 5 escaped the region and must not be inside it: {inside:?}"
        );
        assert_eq!(protected.follow, Some(NodeId::new(4)));
        assert!(
            recovered.labels.contains(&NodeId::new(5)),
            "the escape target needs a label"
        );
    }

    #[test]
    fn a_protected_region_around_a_loop_keeps_both() {
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 5), (2, 1), (3, 4), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(6, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(6, &[3, 4]),
            }],
        }];

        let recovered = structure_protected(&graph, NodeId::new(0), &regions);
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 1);
        assert_eq!(recovered.metrics.loops, 1);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    #[test]
    fn no_protected_regions_matches_the_plain_entry_point() {
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let plain = structure(&graph, NodeId::new(0));
        let protected = structure_protected(&graph, NodeId::new(0), &[]);
        assert_eq!(plain, protected);
    }

    #[test]
    fn conjunction_becomes_one_condition() {
        // `if (a && b) X; else Y;` — 0 tests a, 1 tests b, and both false arms
        // reach 2. Without merging, 2 has two predecessors and one of them
        // needs a goto.
        let graph = graph_of(5, &[(0, 1), (0, 2), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let unmerged = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &unmerged);
        assert_eq!(
            unmerged.metrics.gotos, 1,
            "the shared false arm needs a goto when the condition cannot merge"
        );

        let merged = structure_merging(&graph, entry, &[1]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0, "merging removes the goto");
        assert_eq!(merged.metrics.merged_conditions, 1);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::All(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: false,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
        assert_eq!(branch.then_branch, Region::Block(NodeId::new(3)));
        assert_eq!(branch.else_branch, Some(Region::Block(NodeId::new(2))));
    }

    #[test]
    fn a_block_that_computes_still_joins_the_condition_it_tests_in() {
        // The same disjunction, but 1 does work before its test. Short-circuit
        // evaluation runs that work exactly when the edge 0 -> 1 did, so the
        // merge stays sound and the alternative — placing 1 inside an arm — is
        // what leaves 3 reached twice and costs the goto.
        let graph = graph_of(5, &[(0, 3), (0, 1), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let merged = structure_expressing(&graph, entry, &[1]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);
        assert_eq!(merged.metrics.merged_conditions, 1);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: false,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
    }

    #[test]
    fn a_block_neither_moveable_nor_expressible_keeps_its_own_place() {
        // With 1 in neither set the merge must not happen: 1 is a block with
        // effects the recovery cannot restate, so it stays where it ran.
        let graph = graph_of(5, &[(0, 3), (0, 1), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let plain = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &plain);
        assert_eq!(plain.metrics.merged_conditions, 0);
    }

    #[test]
    fn disjunction_becomes_one_condition() {
        // `if (a || b) X; else Y;` — 0's taken arm and 1's taken arm both
        // reach 3.
        let graph = graph_of(5, &[(0, 3), (0, 1), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let merged = structure_merging(&graph, entry, &[1]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: false,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
    }

    #[test]
    fn a_chain_of_tests_collapses_into_one_condition() {
        // `if (a && b && c)`: 1 and 2 both test and both fall to 4.
        let graph = graph_of(
            6,
            &[
                (0, 1),
                (0, 4),
                (1, 2),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 5),
                (4, 5),
            ],
        );
        let entry = NodeId::new(0);

        let merged = structure_merging(&graph, entry, &[1, 2]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate.tests(),
            3,
            "three tests, flattened into one condition: {:?}",
            branch.predicate
        );
        assert!(
            matches!(branch.predicate, Predicate::All(ref parts) if parts.len() == 3),
            "a chain flattens rather than nesting: {:?}",
            branch.predicate
        );
    }

    #[test]
    fn a_block_with_side_effects_is_never_folded_into_a_condition() {
        // The same graph as the conjunction, but block 1 is not declared
        // condition-only, so folding it would move whatever it does inside a
        // condition that may not evaluate it.
        let graph = graph_of(5, &[(0, 1), (0, 2), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let recovered = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.merged_conditions, 0);
        assert_eq!(
            recovered.metrics.gotos, 1,
            "the goto is the price of not moving a block with effects"
        );
    }

    #[test]
    fn a_shared_test_block_is_never_folded_into_a_condition() {
        // Block 2 tests and is condition-only, but block 3 also reaches it.
        // Folding it into 0's condition would delete the path through 3.
        let graph = graph_of(
            6,
            &[
                (0, 2),
                (0, 1),
                (3, 2),
                (2, 4),
                (2, 1),
                (1, 5),
                (4, 5),
                (0, 3),
            ],
        );
        let entry = NodeId::new(0);
        let recovered = structure_merging(&graph, entry, &[2]);
        assert_total(&graph, entry, &recovered);
        assert_eq!(
            recovered.metrics.merged_conditions, 0,
            "a block with two predecessors cannot move into a condition"
        );
    }

    #[test]
    fn a_loop_header_is_never_folded_into_a_condition() {
        // Block 1 is a loop header that also tests and is condition-only.
        // Folding it would lose the loop.
        let graph = graph_of(5, &[(0, 1), (0, 3), (1, 2), (1, 3), (2, 1), (3, 4)]);
        let entry = NodeId::new(0);
        let recovered = structure_merging(&graph, entry, &[1]);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.loops, 1, "the loop survives");
        assert_eq!(recovered.metrics.merged_conditions, 0);
    }

    #[test]
    fn negating_a_merged_condition_distributes_over_its_parts() {
        let predicate = Predicate::All(vec![
            Predicate::Test {
                block: NodeId::new(0),
                negated: false,
            },
            Predicate::Test {
                block: NodeId::new(1),
                negated: true,
            },
        ]);
        assert_eq!(
            predicate.negate(),
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: true,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
        assert_eq!(predicate.negate().negate(), predicate);
    }

    #[test]
    fn a_shared_return_is_repeated_rather_than_jumped_to() {
        // Two paths end at the same `return` at 3, and the function also ends
        // at 4, so no single block post-dominates the branch at 0 and 3 cannot
        // become its join. A goto to a bare return reads worse than the return.
        let graph = graph_of(5, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 3)]);
        let entry = NodeId::new(0);
        let recovered = structure(&graph, entry);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "no goto to a return");
        assert_eq!(recovered.metrics.replicated_tails, 1);
        assert!(
            recovered.labels.is_empty(),
            "a replicated tail needs no label"
        );
    }

    #[test]
    fn a_shared_block_that_continues_is_never_repeated() {
        // The same shape, except 3 flows on to 5. Repeating it would repeat
        // everything after it too, so this one stays a goto.
        let graph = graph_of(6, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (3, 5)]);
        let entry = NodeId::new(0);
        let recovered = structure(&graph, entry);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.replicated_tails, 0);
        assert_eq!(recovered.metrics.gotos, 1);
    }
}
