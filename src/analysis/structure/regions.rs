//! The region tree a recovery returns: its blocks, predicates, regions and
//! counts.

use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};

use crate::{
    bitset::BitSet,
    graph::{NodeId, Successors},
    ir::exception::HandlerKind,
};

/// A set of blocks over a stated index domain.
///
/// The domain is `0..node_count`, and it is part of the value rather than a
/// precondition on the reader: [`BlockSet::contains`] answers for any
/// [`NodeId`] at all, and a node outside the domain is simply not a member.
/// That is what lets a caller hand a set to [`structure`](crate::analysis::structure::structure) without knowing
/// which of the ten places inside the walk will read it.
///
/// A set whose `node_count` disagrees with the graph's is renormalised once, at
/// the [`structure`](crate::analysis::structure::structure) boundary, and the renormalisation is counted in
/// [`StructureMetrics::resized_inputs`]. A non-zero count means a caller built
/// a set for a different function, which is a defect in the caller - the
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

    /// Returns the nodes of `graph` reachable from `start` without entering
    /// a node of `excluded`, over `0..node_count`.
    ///
    /// `start` itself is not reached when excluded. Iterative, so a long chain
    /// does not deepen the call stack.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph whose successor edges are walked.
    /// * `node_count` - The index domain of the result.
    /// * `start` - Where the walk begins.
    /// * `excluded` - Nodes neither visited nor walked through.
    #[must_use]
    pub fn reachable_avoiding<G: Successors>(
        graph: &G,
        node_count: usize,
        start: NodeId,
        excluded: &Self,
    ) -> Self {
        let mut reached = Self::new(node_count);
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if excluded.contains(node) || !reached.insert(node) {
                continue;
            }
            stack.extend(graph.successors(node));
        }
        reached
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
/// place in the tree, so the second reaches it by `goto`. That shape is the
/// commonest source of gotos in recovered machine code: a short-circuit
/// condition is everywhere, and each one leaves a block with exactly two
/// predecessors.
///
/// Recovering the condition as a tree removes the cause rather than the
/// symptom.
///
/// Nesting is bounded. Each fold alternating between [`Predicate::All`] and
/// [`Predicate::Any`] adds one level, and the recovery performs at most
/// [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth) folds per condition, so every recursive
/// traversal of a predicate this module produces - including the derived
/// `Drop`, `Clone` and `PartialEq` - is bounded by that number.
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

    /// The branch at `block`, as [`Predicate::Test`], with the statements of
    /// `edge` run ahead of it: `edge` stands for one of the branch's edges
    /// ([`StructureOptions::edge_blocks`](crate::analysis::structure::StructureOptions::edge_blocks)), and its statements, which the
    /// caller can equally say at the end of `block`, are said there so the
    /// branch joins a condition its edge would have kept it out of.
    Hoisted {
        /// Block whose terminator performs the test.
        block: NodeId,

        /// Whether the test reads inverted.
        negated: bool,

        /// The edge block whose statements run ahead of the test.
        edge: NodeId,
    },

    /// Every part holds. Parts are evaluated in order and later ones only when
    /// the earlier ones held, which is what the control flow does.
    All(Vec<Predicate>),

    /// Any part holds, with the same short-circuit order.
    Any(Vec<Predicate>),
}

impl Predicate {
    /// The blocks this condition spans, as the statements a region places
    /// in evaluation order.
    ///
    /// The blocks a condition spans are placed by the [`Region::If`] that
    /// holds it, so anything that drops the conditional places them through
    /// this or loses them from the tree.
    #[must_use]
    pub fn statements(&self) -> Region {
        let mut parts = Vec::new();
        self.for_each_block(&mut |block| parts.push(Region::Block(block)));
        Region::sequence(parts)
    }

    /// Calls `visit` for every block that takes part in the condition.
    pub fn for_each_block(&self, visit: &mut impl FnMut(NodeId)) {
        match self {
            Predicate::Test { block, .. } => visit(*block),
            Predicate::Hoisted { block, edge, .. } => {
                visit(*block);
                visit(*edge);
            }
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
            Predicate::Hoisted {
                block,
                negated,
                edge,
            } => Predicate::Hoisted {
                block: *block,
                negated: !negated,
                edge: *edge,
            },
            Predicate::All(parts) => Predicate::Any(parts.iter().map(Predicate::negate).collect()),
            Predicate::Any(parts) => Predicate::All(parts.iter().map(Predicate::negate).collect()),
        }
    }

    /// Returns the block the condition starts at.
    ///
    /// Evaluation order is the order the parts appear, so this is the block
    /// control reaches first - the one whose statements run whatever the
    /// condition turns out to be.
    #[must_use]
    pub fn head(&self) -> Option<NodeId> {
        match self {
            Predicate::Test { block, .. } | Predicate::Hoisted { block, .. } => Some(*block),
            Predicate::All(parts) | Predicate::Any(parts) => {
                parts.first().and_then(Predicate::head)
            }
        }
    }

    /// How many blocks say their statements inside the condition: each edge
    /// whose statements run ahead of a test ([`Predicate::Hoisted`]), and
    /// each tested block after the one the condition starts at that
    /// computes more than its test (outside `condition_only`). The first
    /// block's statements run whatever the condition turns out to be, so
    /// they are said ahead of it.
    fn statements_said(&self, condition_only: &BlockSet) -> usize {
        let head = self.head();
        let mut said = 0usize;
        let mut pending = vec![self];
        while let Some(predicate) = pending.pop() {
            match predicate {
                Predicate::Hoisted { .. } => said = said.saturating_add(1),
                Predicate::Test { block, .. } => {
                    if Some(*block) != head && !condition_only.contains(*block) {
                        said = said.saturating_add(1);
                    }
                }
                Predicate::All(parts) | Predicate::Any(parts) => pending.extend(parts.iter()),
            }
        }
        said
    }

    /// States this condition in the order given of the blocks the recovery
    /// read the other way round (`flipped`): a test of one of them negated.
    fn reorient(&mut self, flipped: &BlockSet) {
        match self {
            Predicate::Test { block, negated } | Predicate::Hoisted { block, negated, .. } => {
                if flipped.contains(*block) {
                    *negated = !*negated;
                }
            }
            Predicate::All(parts) | Predicate::Any(parts) => {
                for part in parts {
                    part.reorient(flipped);
                }
            }
        }
    }

    /// Replaces every block the condition names by `original` of it.
    fn renumber(&mut self, original: &impl Fn(NodeId) -> NodeId) {
        match self {
            Predicate::Test { block, .. } => *block = original(*block),
            Predicate::Hoisted { block, edge, .. } => {
                *block = original(*block);
                *edge = original(*edge);
            }
            Predicate::All(parts) | Predicate::Any(parts) => {
                for part in parts {
                    part.renumber(original);
                }
            }
        }
    }

    /// Returns the number of tests the condition performs.
    #[must_use]
    pub fn tests(&self) -> usize {
        match self {
            Predicate::Test { .. } | Predicate::Hoisted { .. } => 1,
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

    /// The values that select this arm in a switch over a value rather than a
    /// table - `case 'a':` - which a tree of comparisons recovers
    /// ([`CaseTest`]). Empty for a table's arm, which [`Self::selectors`]
    /// names.
    pub values: Vec<i64>,

    /// Whether this arm is the switch's `default`: every value no other arm
    /// names takes it.
    pub default: bool,

    /// Whether the arm runs on into the next arm of [`SwitchRegion::cases`]
    /// rather than leaving the switch: C's fall-through, where the arm's
    /// last block runs on to the next arm's first and nothing else reaches
    /// that block but the dispatch.
    pub falls_through: bool,
}

/// The relation a block's branch tests between one value and a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseRelation {
    /// `value == constant`.
    Equal,
    /// `value != constant`.
    NotEqual,
    /// `value < constant`.
    Less {
        /// Whether the values compare as signed.
        signed: bool,
    },
    /// `value <= constant`.
    LessOrEqual {
        /// Whether the values compare as signed.
        signed: bool,
    },
    /// `value > constant`.
    Greater {
        /// Whether the values compare as signed.
        signed: bool,
    },
    /// `value >= constant`.
    GreaterOrEqual {
        /// Whether the values compare as signed.
        signed: bool,
    },
}

impl CaseRelation {
    /// Whether an ordering compares signed; `None` for an equality.
    #[must_use]
    pub const fn signed(self) -> Option<bool> {
        match self {
            Self::Equal | Self::NotEqual => None,
            Self::Less { signed }
            | Self::LessOrEqual { signed }
            | Self::Greater { signed }
            | Self::GreaterOrEqual { signed } => Some(signed),
        }
    }
}

/// What a block's branch tests when it compares one value with a constant:
/// the step a compiler lowers a sparse `switch` to.
///
/// gcc and clang lower `switch (c) { case 'a': .. case 'v': .. }` over values
/// too sparse for a table into a binary search of such tests. The tree is a
/// `switch` when every test reads one value, the blocks between the tests do
/// nothing else, and every path that matches no `case` ends in one block, the
/// `default`; [`StructureOptions::case_tests`](crate::analysis::structure::StructureOptions::case_tests) is how the caller, which reads
/// the instructions, states each test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaseTest {
    /// The tested value, as the caller identifies it: two blocks whose keys
    /// are equal test the same value.
    pub value: u64,

    /// The relation the branch tests.
    pub relation: CaseRelation,

    /// The constant the value is compared with.
    pub constant: i64,

    /// The successor index the branch takes when the relation holds.
    pub holds: usize,

    /// Whether the branch tests the value the block is entered with: nothing
    /// in the block writes it before the branch.
    ///
    /// Only such a test reads, on entry, the value the test before it read:
    /// a test below another in a `switch` tree, and a test a jump threaded
    /// past, whose outcome the jump's own condition decides. A block that
    /// computes the value it tests - a root computing `c - 'a'`, or a
    /// variable that holds another value in each block - tests a value of
    /// its own.
    pub entered: bool,
}

/// How the comparisons of one value read it: as a signed or an unsigned
/// 64-bit number.
///
/// An ordering states which ([`CaseRelation::signed`]); an equality compares
/// alike under either, so a tree or a pair of tests holding no ordering is
/// read signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading {
    /// Whether the value and the constants it is compared with are signed.
    pub signed: bool,
}

impl Reading {
    /// The number `constant` is under this reading.
    pub fn value(self, constant: i64) -> i128 {
        if self.signed {
            i128::from(constant)
        } else {
            i128::from(constant.cast_unsigned())
        }
    }

    /// The least and the greatest value the compared value can hold under
    /// this reading.
    pub fn bounds(self) -> (i128, i128) {
        if self.signed {
            (i128::from(i64::MIN), i128::from(i64::MAX))
        } else {
            (0, i128::from(u64::MAX))
        }
    }
}

impl CaseTest {
    /// Returns the values the tested value can hold on the edge in position
    /// `index` of the block's successors, as ranges under `reading`.
    fn values(&self, index: usize, reading: Reading) -> Vec<(i128, i128)> {
        let (floor, ceiling) = reading.bounds();
        let constant = reading.value(self.constant);
        let below = constant.saturating_sub(1);
        let above = constant.saturating_add(1);
        let ranges = match (self.relation, index == self.holds) {
            (CaseRelation::Equal, true) | (CaseRelation::NotEqual, false) => {
                vec![(constant, constant)]
            }
            (CaseRelation::Equal, false) | (CaseRelation::NotEqual, true) => {
                vec![(floor, below), (above, ceiling)]
            }
            (CaseRelation::Less { .. }, true) | (CaseRelation::GreaterOrEqual { .. }, false) => {
                vec![(floor, below)]
            }
            (CaseRelation::Less { .. }, false) | (CaseRelation::GreaterOrEqual { .. }, true) => {
                vec![(constant, ceiling)]
            }
            (CaseRelation::LessOrEqual { .. }, true) | (CaseRelation::Greater { .. }, false) => {
                vec![(floor, constant)]
            }
            (CaseRelation::LessOrEqual { .. }, false) | (CaseRelation::Greater { .. }, true) => {
                vec![(above, ceiling)]
            }
        };
        ranges
            .into_iter()
            .filter(|(low, high)| low <= high)
            .collect()
    }

    /// Whether every value this test's block sends along the edge in
    /// position `index` is one `skipped`, a test of the same value, sends to
    /// its successor in position `toward`: run on that edge, `skipped` goes
    /// where the edge goes.
    ///
    /// Two orderings that compare under different signedness decide nothing
    /// of each other; an equality compares alike under either.
    #[must_use]
    pub fn decides(&self, index: usize, skipped: &Self, toward: usize) -> bool {
        let reading = Reading {
            signed: match (self.relation.signed(), skipped.relation.signed()) {
                (Some(one), Some(other)) if one != other => return false,
                (Some(signed), _) | (_, Some(signed)) => signed,
                (None, None) => true,
            },
        };
        let sent = skipped.values(toward, reading);
        self.values(index, reading)
            .iter()
            .all(|(low, high)| sent.iter().any(|(from, to)| from <= low && high <= to))
    }
}

/// A recovered multi-way branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchRegion {
    /// Block whose terminator performs the dispatch.
    pub head: NodeId,

    /// The arms, ordered by their lowest selector index as the recovery read
    /// the head's successors ([`StructureOptions::polar`](crate::analysis::structure::StructureOptions::polar)).
    pub cases: Vec<SwitchCase>,

    /// Block control resumes at after the switch, if the arms rejoin.
    pub follow: Option<NodeId>,

    /// The test blocks below the head a switch over a value consumed: each
    /// only narrows the value toward its case, so its work is the dispatch's.
    /// Empty for a table.
    pub tests: Vec<NodeId>,
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

impl ProtectedHandler {
    /// The body of a handler whose extent the exception table does not
    /// record: what its entry reaches that neither the protected code nor the
    /// function entry reaches.
    ///
    /// A handler is entered by the runtime, so its own blocks are unreachable
    /// from the function entry. That is what bounds the walk, and it cuts both
    /// ways: a block the walk finds that the function entry also reaches is
    /// tail code the handler shares with the ordinary path, not the handler.
    /// An unbounded walk runs off the end of the handler into that code and
    /// attributes it to the handler.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph whose successor edges are walked.
    /// * `node_count` - The index domain of the result.
    /// * `handler_entry` - The handler's first block.
    /// * `protected` - The blocks the handler's region protects.
    /// * `entry_reach` - The blocks the function entry reaches, the same for
    ///   every handler of a function.
    ///
    /// # Returns
    ///
    /// The inferred body, empty when the handler entry is itself protected or
    /// reached from the function entry.
    #[must_use]
    pub fn inferred_body<G: Successors>(
        graph: &G,
        node_count: usize,
        handler_entry: NodeId,
        protected: &BlockSet,
        entry_reach: &BlockSet,
    ) -> BlockSet {
        let excluded = BlockSet::from_nodes(node_count, protected.iter().chain(entry_reach.iter()));
        BlockSet::reachable_avoiding(graph, node_count, handler_entry, &excluded)
    }
}

/// A protected region to recover, described by the blocks it occupies.
///
/// Ranges come from the binary's own exception tables, which is why they are an
/// input rather than something structuring infers: the blocks a `try` covers are
/// recorded by the compiler and cannot be recovered from control flow, because
/// the edge into a handler is taken by the runtime and appears nowhere in the
/// code.
///
/// Two regions may share an entry block - `try { try {} catch {} } finally {}`
/// is the ordinary ECMA-335 and JVM encoding of nested clauses - and both open.
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

    /// Go on to the next test of the loop with the named header, as C's
    /// `continue` does: the header's for a loop tested there or nowhere, the
    /// latch's for a [`StructuredLoop::DoWhile`].
    Continue(NodeId),

    /// Transfer to a block that structuring could not reach in tree order.
    ///
    /// The target is recorded in [`Structured::labels`], so a printer knows to
    /// give it a label.
    Goto(NodeId),

    /// Leave the `switch` whose dispatch is the named block, as C's `break`
    /// inside one does. Apart from [`Region::Break`], since the block a
    /// switch dispatches from can also be the header of the loop around it.
    SwitchBreak(NodeId),
}

impl Region {
    /// The simplest region equivalent to `sequence` placed in order.
    ///
    /// Nested sequences are flattened. A `Seq` inside a `Seq` is never a real
    /// nesting level - it only records that one part of the walk returned
    /// several statements - and leaving it in would make the tree's shape
    /// depend on how the walk was decomposed rather than on the code.
    #[must_use]
    pub fn sequence(sequence: Vec<Self>) -> Self {
        let mut flattened: Vec<Self> = Vec::with_capacity(sequence.len());
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

    /// Calls `visit` for every block placed in this region, in printed order.
    ///
    /// A [`Region::Goto`] target is not a placement: the block is placed
    /// wherever structuring reached it, and the goto only transfers there.
    ///
    /// Recursive, and safely so: a tree this module produces nests no deeper
    /// than [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth). See [`Structured`].
    pub fn for_each_block(&self, visit: &mut impl FnMut(NodeId)) {
        match self {
            Region::Empty
            | Region::Break(_)
            | Region::Continue(_)
            | Region::Goto(_)
            | Region::SwitchBreak(_) => {}
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
                for test in &inner.tests {
                    visit(*test);
                }
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

    /// Returns how many transfers in the tree C can only write as a goto
    /// ([`Self::jumps`]).
    #[must_use]
    pub fn gotos(&self) -> usize {
        self.jumps().len()
    }

    /// Returns each transfer in the tree C can only write as a goto, by the
    /// block it goes to where the tree names one: a goto, to its target; a
    /// `break` naming a loop other than the innermost one around it - or one
    /// with a `switch` between, which a bare `break` would leave instead - to
    /// that loop's follow; a `continue` naming a loop other than the
    /// innermost one, to its header; and a switch's `break` with a loop or
    /// another switch between, to the switch's follow. A transfer to a loop
    /// or switch that encloses it and has no follow goes nowhere the tree
    /// names.
    #[must_use]
    pub fn jumps(&self) -> Vec<Option<NodeId>> {
        /// A loop or switch the walk is inside, innermost last.
        #[derive(Clone, Copy)]
        enum Open {
            /// A loop, by its header, and its follow.
            Loop(NodeId, Option<NodeId>),
            /// A switch, by its head, and its follow.
            Switch(NodeId, Option<NodeId>),
        }
        fn walk(region: &Region, open: &mut Vec<Open>, jumps: &mut Vec<Option<NodeId>>) {
            // The innermost loop, and whether a switch is inside it.
            let innermost_loop = || {
                open.iter().rev().find_map(|frame| match frame {
                    Open::Loop(header, _) => Some(*header),
                    Open::Switch(..) => None,
                })
            };
            let switch_inside = || matches!(open.last(), Some(Open::Switch(..)));
            let follow_of_loop = |header: NodeId| {
                open.iter().rev().find_map(|frame| match frame {
                    Open::Loop(at, follow) if *at == header => Some(*follow),
                    _ => None,
                })
            };
            match region {
                Region::Empty | Region::Block(_) => {}
                Region::Goto(target) => jumps.push(Some(*target)),
                Region::Break(header) => {
                    if innermost_loop() != Some(*header) || switch_inside() {
                        jumps.push(follow_of_loop(*header).flatten());
                    }
                }
                Region::SwitchBreak(head) => {
                    if !matches!(open.last(), Some(Open::Switch(at, _)) if at == head) {
                        let follow = open.iter().rev().find_map(|frame| match frame {
                            Open::Switch(at, follow) if at == head => Some(*follow),
                            _ => None,
                        });
                        jumps.push(follow.flatten());
                    }
                }
                Region::Continue(header) => {
                    if innermost_loop() != Some(*header) {
                        jumps.push(Some(*header));
                    }
                }
                Region::Seq(parts) => {
                    for part in parts {
                        walk(part, open, jumps);
                    }
                }
                Region::If(inner) => {
                    walk(&inner.then_branch, open, jumps);
                    if let Some(other) = &inner.else_branch {
                        walk(other, open, jumps);
                    }
                }
                Region::Loop(inner) => {
                    open.push(Open::Loop(inner.header, inner.follow));
                    walk(&inner.body, open, jumps);
                    open.pop();
                }
                Region::Switch(inner) => {
                    open.push(Open::Switch(inner.head, inner.follow));
                    for case in &inner.cases {
                        walk(&case.body, open, jumps);
                    }
                    open.pop();
                }
                Region::Try(inner) => {
                    walk(&inner.body, open, jumps);
                    for handler in &inner.handlers {
                        if let Some(filter) = &handler.filter {
                            walk(filter, open, jumps);
                        }
                        walk(&handler.body, open, jumps);
                    }
                }
            }
        }
        let mut jumps = Vec::new();
        walk(self, &mut Vec::new(), &mut jumps);
        jumps
    }

    /// How many blocks the tree's `if` conditions say the statements of:
    /// each edge whose statements run ahead of a test
    /// ([`Predicate::Hoisted`]), and each tested block after a condition's
    /// first that computes more than its test (outside `condition_only`,
    /// [`StructureOptions::condition_only`](crate::analysis::structure::StructureOptions::condition_only)).
    ///
    /// A loop's condition is not counted: `while ((c = *p++) != 0)` says its
    /// header's statements where the header runs them, every iteration.
    #[must_use]
    pub fn statements_in_conditions(&self, condition_only: &BlockSet) -> usize {
        match self {
            Region::Empty
            | Region::Break(_)
            | Region::Continue(_)
            | Region::Goto(_)
            | Region::SwitchBreak(_)
            | Region::Block(_) => 0,
            Region::Seq(parts) => parts.iter().fold(0usize, |sum, part| {
                sum.saturating_add(part.statements_in_conditions(condition_only))
            }),
            Region::If(inner) => inner
                .predicate
                .statements_said(condition_only)
                .saturating_add(inner.then_branch.statements_in_conditions(condition_only))
                .saturating_add(
                    inner
                        .else_branch
                        .as_ref()
                        .map_or(0, |other| other.statements_in_conditions(condition_only)),
                ),
            Region::Loop(inner) => inner.body.statements_in_conditions(condition_only),
            Region::Switch(inner) => inner.cases.iter().fold(0usize, |sum, case| {
                sum.saturating_add(case.body.statements_in_conditions(condition_only))
            }),
            Region::Try(inner) => inner.handlers.iter().fold(
                inner.body.statements_in_conditions(condition_only),
                |sum, handler| {
                    sum.saturating_add(handler.body.statements_in_conditions(condition_only))
                },
            ),
        }
    }

    /// How deep the tree places its blocks, `depth` its own depth: the sum of
    /// each block's depth, an arm one deeper than what holds it, and an `else`
    /// that is one conditional as deep as the `if` whose chain it continues
    /// (`} else if (..) {`).
    fn nesting(&self, depth: usize) -> usize {
        let deeper = depth.saturating_add(1);
        match self {
            Region::Empty
            | Region::Break(_)
            | Region::Continue(_)
            | Region::Goto(_)
            | Region::SwitchBreak(_) => 0,
            Region::Block(_) => depth,
            Region::Seq(parts) => parts
                .iter()
                .fold(0usize, |sum, part| sum.saturating_add(part.nesting(depth))),
            Region::If(inner) => {
                let mut tests = 0usize;
                inner
                    .predicate
                    .for_each_block(&mut |_| tests = tests.saturating_add(depth));
                let otherwise = match &inner.else_branch {
                    Some(chained @ Region::If(_)) => chained.nesting(depth),
                    Some(other) => other.nesting(deeper),
                    None => 0,
                };
                tests
                    .saturating_add(inner.then_branch.nesting(deeper))
                    .saturating_add(otherwise)
            }
            Region::Loop(inner) => {
                let mut tests = 0usize;
                if let Some(predicate) = &inner.predicate {
                    predicate.for_each_block(&mut |_| tests = tests.saturating_add(depth));
                }
                tests.saturating_add(inner.body.nesting(deeper))
            }
            Region::Switch(inner) => inner.cases.iter().fold(depth, |sum, case| {
                sum.saturating_add(case.body.nesting(deeper))
            }),
            Region::Try(inner) => inner
                .handlers
                .iter()
                .fold(inner.body.nesting(deeper), |sum, handler| {
                    sum.saturating_add(handler.body.nesting(deeper))
                }),
        }
    }

    /// States this tree in the order given of the blocks the recovery read
    /// the other way round (`flipped`, [`StructureOptions::polar`](crate::analysis::structure::StructureOptions::polar)): a test
    /// of one of them negated, and a switch headed by one with its two
    /// successor indices exchanged.
    fn reorient(&mut self, flipped: &BlockSet) {
        match self {
            Region::Empty
            | Region::Block(_)
            | Region::Break(_)
            | Region::Continue(_)
            | Region::Goto(_)
            | Region::SwitchBreak(_) => {}
            Region::Seq(parts) => {
                for part in parts {
                    part.reorient(flipped);
                }
            }
            Region::If(inner) => {
                inner.predicate.reorient(flipped);
                inner.then_branch.reorient(flipped);
                if let Some(other) = &mut inner.else_branch {
                    other.reorient(flipped);
                }
            }
            Region::Loop(inner) => {
                if let Some(predicate) = &mut inner.predicate {
                    predicate.reorient(flipped);
                }
                inner.body.reorient(flipped);
            }
            Region::Switch(inner) => {
                let head = flipped.contains(inner.head);
                for case in &mut inner.cases {
                    if head {
                        for selector in &mut case.selectors {
                            if *selector < 2 {
                                *selector = usize::from(*selector == 0);
                            }
                        }
                    }
                    case.body.reorient(flipped);
                }
            }
            Region::Try(inner) => {
                inner.body.reorient(flipped);
                for handler in &mut inner.handlers {
                    handler.body.reorient(flipped);
                }
            }
        }
    }

    /// Replaces every block the tree names - placed, tested, jumped to, or
    /// naming a loop - by `original` of it.
    fn renumber(&mut self, original: &impl Fn(NodeId) -> NodeId) {
        match self {
            Region::Empty => {}
            Region::Block(node)
            | Region::Break(node)
            | Region::Continue(node)
            | Region::Goto(node)
            | Region::SwitchBreak(node) => {
                *node = original(*node);
            }
            Region::Seq(parts) => {
                for part in parts {
                    part.renumber(original);
                }
            }
            Region::If(inner) => {
                inner.predicate.renumber(original);
                inner.then_branch.renumber(original);
                if let Some(other) = &mut inner.else_branch {
                    other.renumber(original);
                }
            }
            Region::Loop(inner) => {
                inner.header = original(inner.header);
                if let Some(predicate) = &mut inner.predicate {
                    predicate.renumber(original);
                }
                inner.body.renumber(original);
                inner.follow = inner.follow.map(original);
            }
            Region::Switch(inner) => {
                inner.head = original(inner.head);
                for case in &mut inner.cases {
                    case.target = original(case.target);
                    case.body.renumber(original);
                }
                inner.follow = inner.follow.map(original);
                for test in &mut inner.tests {
                    *test = original(*test);
                }
            }
            Region::Try(inner) => {
                inner.entry = original(inner.entry);
                inner.body.renumber(original);
                for handler in &mut inner.handlers {
                    handler.entry = original(handler.entry);
                    if let Some(filter) = &mut handler.filter {
                        filter.renumber(original);
                    }
                    handler.body.renumber(original);
                }
                inner.follow = inner.follow.map(original);
            }
        }
    }

    /// Leaves out each `goto` that only runs on to a `goto` to the same
    /// block: where the statement after a conditional is `goto L`, an arm of
    /// it ending in `goto L` reaches that by running on, as an arm of an `if`
    /// nested at its end does. A loop's body, a switch's cases and a
    /// protected body are tidied within: their ends do not run on to what
    /// follows them.
    fn drop_onward_gotos(&mut self) {
        match self {
            Region::Empty
            | Region::Block(_)
            | Region::Break(_)
            | Region::Continue(_)
            | Region::Goto(_)
            | Region::SwitchBreak(_) => {}
            Region::Seq(parts) => {
                let mut kept: Vec<Region> = Vec::with_capacity(parts.len());
                for mut part in mem::take(parts) {
                    part.drop_onward_gotos();
                    if let Region::Goto(target) = part
                        && let Some(before) = kept.pop()
                    {
                        kept.push(before.ending_without(target));
                    }
                    kept.push(part);
                }
                *self = Region::sequence(kept);
            }
            Region::If(inner) => {
                inner.then_branch.drop_onward_gotos();
                if let Some(other) = &mut inner.else_branch {
                    other.drop_onward_gotos();
                }
            }
            Region::Loop(inner) => inner.body.drop_onward_gotos(),
            Region::Switch(inner) => {
                for case in &mut inner.cases {
                    case.body.drop_onward_gotos();
                }
            }
            Region::Try(inner) => {
                inner.body.drop_onward_gotos();
                for handler in &mut inner.handlers {
                    if let Some(filter) = &mut handler.filter {
                        filter.drop_onward_gotos();
                    }
                    handler.body.drop_onward_gotos();
                }
            }
        }
    }

    /// This region with a `goto target` it ends in left out, where it ends
    /// in one: running on past its end reaches `target` anyway. An `if`
    /// whose arms are left with nothing keeps its test and the one `goto`; one
    /// whose taken arm is left with nothing asks the other way round.
    fn ending_without(self, target: NodeId) -> Self {
        match self {
            Region::Goto(to) if to == target => Region::Empty,
            Region::Seq(mut parts) => {
                if let Some(last) = parts.pop() {
                    parts.push(last.ending_without(target));
                }
                Region::sequence(parts)
            }
            Region::If(inner) => {
                let IfRegion {
                    predicate,
                    then_branch,
                    else_branch,
                } = *inner;
                let then_branch = then_branch.ending_without(target);
                let else_branch = else_branch
                    .map(|other| other.ending_without(target))
                    .filter(|other| *other != Region::Empty);
                let kept = match (then_branch, else_branch) {
                    (Region::Empty, None) => IfRegion {
                        predicate,
                        then_branch: Region::Goto(target),
                        else_branch: None,
                    },
                    (Region::Empty, Some(other)) => IfRegion {
                        predicate: predicate.negate(),
                        then_branch: other,
                        else_branch: None,
                    },
                    (then_branch, else_branch) => IfRegion {
                        predicate,
                        then_branch,
                        else_branch,
                    },
                };
                Region::If(Box::new(kept))
            }
            other => other,
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

    /// Transfers the recovered tree writes that C can only write as a goto
    /// ([`Region::gotos`]): every [`Region::Goto`], and every `break` or
    /// `continue` no bare `break` or `continue` where it stands could write.
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

    /// Of [`Self::switches`], those over a value recovered from a tree of
    /// comparisons ([`CaseTest`]) rather than from a dispatch table.
    pub value_switches: usize,

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

    /// Switch arms that run on into the next arm
    /// ([`SwitchCase::falls_through`]).
    pub fall_throughs: usize,

    /// Placements of a tail beyond its first.
    ///
    /// Straight-line code that ends the function, or runs on to the block
    /// the region reaching it resumes at, is repeated rather than jumped to,
    /// so this counts the blocks of every goto that became a duplicated tail
    /// instead. See [`Structured::blocks`] for what that means
    /// for placement.
    pub replicated_tails: usize,

    /// Blocks copied so that every loop is entered at one block.
    ///
    /// A loop entered at two blocks is no natural loop, and reaching its
    /// second entry takes a goto; copying what that entry runs before it
    /// reaches the first gives the outside path its own blocks. A copy is
    /// placed under its original's id, so this is how many placements beyond
    /// the first that allows. See [`Structured::blocks`].
    pub split_blocks: usize,

    /// Caller-supplied [`BlockSet`]s renormalised at the [`structure`](crate::analysis::structure::structure)
    /// boundary because their `node_count` disagreed with the graph's.
    ///
    /// Zero for every well-formed input. A non-zero count is the loud signal
    /// that a caller built a set for a different function: the recovery still
    /// answers, but a block the caller meant to include may have been dropped.
    pub resized_inputs: usize,

    /// Recoveries cut short by [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth).
    ///
    /// Counts both the nesting bound - a walk abandoned at the limit, which
    /// leaves its blocks reachable by goto or flattened inside their protected
    /// region - and a condition whose folding stopped at the limit.
    pub depth_limited: usize,
}

/// A structured control-flow recovery.
///
/// # Depth
///
/// The recovery descends at most [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth) nested
/// regions, and each descent contributes at most two levels to the tree - the
/// construct and the sequence below it - so no tree this module returns nests
/// deeper than `2 * max_depth + 4`. No [`Predicate`] it returns nests deeper
/// than `max_depth` folds either.
///
/// That bound is what makes every recursive traversal of the result safe on a
/// bounded machine stack - including the ones the caller does not write, the
/// derived `Drop`, `Clone`, `PartialEq` and `Debug`, as well as
/// [`Region::for_each_block`].
///
/// A recovery that would have exceeded the bound degrades instead: see
/// [`StructureMetrics::depth_limited`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Structured {
    /// The statement tree.
    pub root: Region,

    /// Blocks a transfer C can only write as a goto goes to, which need a
    /// label when printed ([`Region::jumps`]).
    ///
    /// Exactly those of [`Structured::root`], and nothing else: a goto's
    /// target, and the follow or header of a loop or switch a `break` or
    /// `continue` names past the one a bare transfer would bind to. A block
    /// the walk placed at top level because no region owned it is reached by
    /// falling into it, not by jumping, so it needs no label.
    pub labels: BTreeSet<NodeId>,

    /// Quality counts for the recovery.
    pub metrics: StructureMetrics,

    /// The blocks the recovery routed to another block's test they end with
    /// ([`StructureOptions::repeated_tests`](crate::analysis::structure::StructureOptions::repeated_tests)),
    /// by that test: each goes on to the test, which runs the rows the block
    /// ends with, so the block's own copy of them is no statement of its
    /// own. Printed in the block as well, they run twice on its path.
    pub repeats: BTreeMap<NodeId, NodeId>,
}

impl Structured {
    /// The recovery `root`, its gotos and the labels they need stated from
    /// the tree ([`Region::jumps`]) over `metrics`' other counts.
    #[must_use]
    pub fn new(root: Region, metrics: StructureMetrics) -> Self {
        let jumps = root.jumps();
        Self {
            labels: jumps.iter().flatten().copied().collect(),
            metrics: StructureMetrics {
                gotos: jumps.len(),
                ..metrics
            },
            root,
            repeats: BTreeMap::new(),
        }
    }

    /// Leaves out each `goto` that only runs on to a `goto` to the same
    /// block, and states the gotos and labels left: `if (c) { b; goto L; }
    /// goto L;` is `if (c) b; goto L;`, inside nested `if` arms too. A loop's
    /// body, a switch's cases and a protected body keep a `goto` they end in,
    /// since their ends do not run on to what follows them.
    ///
    /// [`structure`](crate::analysis::structure::structure) does this to the
    /// recovery it keeps, once it is kept: the recoveries it weighs are
    /// compared as the walk wrote them.
    pub fn drop_onward_gotos(&mut self) {
        self.root.drop_onward_gotos();
        let jumps = self.root.jumps();
        self.labels = jumps.iter().flatten().copied().collect();
        self.metrics.gotos = jumps.len();
    }

    /// Returns every block placed in the tree, in printed order.
    ///
    /// Every block reachable from the entry appears at least once, which is the
    /// invariant a caller checks to know the tree accounts for the whole graph.
    /// A block appears more than once only when it lies on straight-line code
    /// that ends the function or its region and was replicated rather than
    /// jumped to, which [`StructureMetrics::replicated_tails`] counts, or was
    /// copied to give a loop one entry, which
    /// [`StructureMetrics::split_blocks`] counts.
    #[must_use]
    pub fn blocks(&self) -> Vec<NodeId> {
        let mut blocks = Vec::new();
        self.root.for_each_block(&mut |node| blocks.push(node));
        blocks
    }
}

/// One recovery the search weighs against the others: the tree, and what the
/// walk learned that the next trial reads.
#[derive(Debug, Clone)]
pub struct Attempt {
    /// The recovered tree.
    pub structured: Structured,

    /// Blocks the tree places again at an arrival, as the start of
    /// straight-line code that runs on to the block its region resumes at
    /// rather than ending the function.
    ///
    /// Each is a tail the arms of a conditional run on to, which the search
    /// offers as that conditional's join on its next trial along with the
    /// tree's labels: taken there, the code is placed once.
    pub rejoining_tails: BTreeSet<NodeId>,

    /// Whether the recovery kept a block out of a condition only because
    /// statements were not to be folded into conditions. Where none was, the
    /// recovery that folds them makes every choice this one made.
    pub declined_fold: bool,
}

impl Attempt {
    /// The recovery of a graph whose entry reaches nothing: an empty tree,
    /// every one of the `node_count` blocks unreached.
    pub fn unreached(node_count: usize) -> Self {
        Self {
            structured: Structured {
                root: Region::Empty,
                labels: BTreeSet::new(),
                metrics: StructureMetrics {
                    unreached: node_count,
                    ..StructureMetrics::default()
                },
                repeats: BTreeMap::new(),
            },
            rejoining_tails: BTreeSet::new(),
            declined_fold: false,
        }
    }

    /// How many transfers the tree writes that C can only write as a goto.
    pub fn gotos(&self) -> usize {
        self.structured.root.gotos()
    }

    /// What the recovery costs a reader, least first: the gotos it writes,
    /// then the blocks it places more than once, as a tail placed in each
    /// arm or as a copy a split made.
    pub fn cost(&self) -> (usize, usize) {
        let metrics = &self.structured.metrics;
        (
            self.gotos(),
            metrics
                .replicated_tails
                .saturating_add(metrics.split_blocks),
        )
    }

    /// How the recovery ranks against another of the same graph, least
    /// first: its cost, then how many blocks its conditions say the
    /// statements of ([`Region::statements_in_conditions`], over the blocks
    /// computing only their test, `condition_only`), then how deep it places
    /// its blocks, then how often it places a block after one the code lays
    /// out later.
    ///
    /// Statements written inside a condition are the cost of a goto, or of
    /// code placed twice, that they save, and only that: of two recoveries
    /// that cost alike, the one nesting the statements is kept, whichever
    /// trial put them in the other's conditions (folding a block, or
    /// hoisting an edge's). Returned as the cost and the tie-breaks after
    /// it, so a caller can weigh a choice of its own between the two.
    pub fn rank(&self, condition_only: &BlockSet) -> ((usize, usize), (usize, usize, usize)) {
        (
            self.cost(),
            (
                self.structured
                    .root
                    .statements_in_conditions(condition_only),
                self.structured.root.nesting(0),
                self.descents(),
            ),
        )
    }

    /// How many times the tree places a block after one the code lays out
    /// later: the blocks in the order the tree places them, counted where one
    /// has a lower id than the one before it.
    ///
    /// Of two recoveries that cost alike, the one that reads the code in the
    /// order it is laid out reads as a compiler of it wrote it: `if (a > 0 &&
    /// b > 0) return a + b;` ahead of the `return 0` after it, and `else if`
    /// for the test the `else` falls into.
    fn descents(&self) -> usize {
        self.structured
            .blocks()
            .windows(2)
            .filter(|pair| matches!(pair, [before, after] if after < before))
            .count()
    }

    /// Replaces every block the tree and its rejoining tails name by
    /// `original` of it, and states its gotos and labels again
    /// ([`Structured::new`]): a `break` or `continue` binds to its loop by the
    /// header's block, and a split copy of a header is renumbered to the
    /// block its original is.
    pub fn renumber(&mut self, original: &impl Fn(NodeId) -> NodeId) {
        let mut root = mem::replace(&mut self.structured.root, Region::Empty);
        root.renumber(original);
        self.structured = Structured::new(root, self.structured.metrics);
        self.rejoining_tails = self
            .rejoining_tails
            .iter()
            .map(|tail| original(*tail))
            .collect();
    }

    /// Restates the tree for the graph with the branches of `flipped`
    /// reversed back.
    pub fn reorient(&mut self, flipped: &BlockSet) {
        self.structured.root.reorient(flipped);
    }
}
