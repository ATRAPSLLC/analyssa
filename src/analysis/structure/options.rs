//! What a recovery is given beyond the graph, and how it is fitted to the
//! graph's index domain.

use std::collections::BTreeMap;

use crate::{
    analysis::structure::{
        BlockSet, CaseTest, HandlerFilter, ProtectedHandler, ProtectedHandlerKind, ProtectedRegion,
    },
    graph::NodeId,
};

/// Default nesting bound for a recovered tree; see [`StructureOptions::max_depth`].
pub const DEFAULT_MAX_DEPTH: usize = 256;

/// What control-flow recovery needs beyond the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureOptions<'a> {
    /// Protected regions to recover; see [`structure`](crate::analysis::structure::structure).
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
    /// once instead of every iteration - code that does not loop.
    ///
    /// Whether a block is such a block is a question about the instructions it
    /// holds, which this module does not see, so the caller answers it. An
    /// empty set is always safe: it costs merged conditions and turns pre-tested
    /// loops into `Endless` ones with an explicit `break`, both of which are
    /// longer to read and correct.
    pub condition_only: BlockSet,

    /// The blocks of [`Self::condition_only`] whose membership the caller
    /// decided from what they reach.
    ///
    /// A block computes nothing but its own condition either because nothing
    /// in the function reads what it defines - true of any graph - or because
    /// something does and the definition reaches no read of it on this graph.
    /// Only the second kind can be made wrong by a recovery that grows what a
    /// block reaches, so only an edge such a block reaches has to keep the
    /// reachable sets as they were: an edge the block it leaves does not
    /// already reach is refused there, and admitted everywhere else.
    ///
    /// An empty set - the default, and what a caller that cannot tell the two
    /// apart should pass - says no membership depends on reachability, and
    /// every edge is then judged on the graph's shape alone. A caller unsure
    /// of the difference should instead pass the whole of
    /// [`Self::condition_only`], which refuses an edge wherever a folded block
    /// could see it.
    pub condition_reachability: BlockSet,

    /// Blocks whose statements the caller can render *inside* a condition.
    ///
    /// A weaker claim than [`Self::condition_only`], and it answers a different
    /// question. That set asks whether a block can be **moved**; this one asks
    /// whether it can be **said** - because a `while` does not move its header
    /// anywhere. C evaluates a condition before every iteration and lets it
    /// carry the work: `while ((c = *p++) != 0)` runs the assignment exactly
    /// where the header ran it, and the body reads what it assigned.
    ///
    /// Short-circuit merging consults it too, and for the same reason. Joining
    /// a block's test onto a condition that already leads to its own target
    /// puts the block behind an `&&` or an `||`, and C evaluates the right
    /// operand of either exactly when control took the edge that reached the
    /// block - so the work runs on the paths it always ran on, and on no
    /// others.
    ///
    /// An empty set is always safe. It costs merged conditions and leaves
    /// pre-tested loops `Endless`, both of which are longer to read and
    /// correct.
    pub condition_expressible: BlockSet,

    /// Blocks whose only statement is one assignment their own test reads, so
    /// the assignment can be the test's operand: `(c = *p++) != 0`.
    ///
    /// The `while` a loop header heads takes its statements into its
    /// condition. When they are this one assignment, the condition reads as C
    /// writes it; any other statements could go there only as comma operands,
    /// and a loop also tested at its latch reads better as the `do`/`while`
    /// that runs them at the top of its body and leaves by one `break` there.
    /// A block here should also be in [`Self::condition_expressible`]. An
    /// empty set is always safe: it costs `while` forms the loops would have
    /// had, never a loop.
    pub condition_operand: BlockSet,

    /// The value test each block's branch makes, for the blocks whose branch
    /// compares one value with a constant ([`CaseTest`]).
    ///
    /// A tree of such tests over one value, its inner blocks
    /// [`Self::condition_only`], recovers as the `switch` it was compiled from.
    /// An empty map is always safe: every test recovers as its `if`.
    pub case_tests: BTreeMap<NodeId, CaseTest>,

    /// Blocks that end the function without returning: a call that does not
    /// come back, a trap.
    ///
    /// A path into one ends where it is, so it has no say in where the paths
    /// that continue meet: `if (c) abort();` joins at the statement after it,
    /// rather than nowhere, which is what post-dominance over every path
    /// answers and what turns the rest of the function into one arm of the
    /// `if` and a goto. An empty set is always safe: every block without a
    /// successor counts as a return.
    pub no_return: BlockSet,

    /// How deep the recovered tree may nest.
    ///
    /// The walk descends one level per region it enters, and the tree it builds
    /// is as deep as the walk - so this is at once a bound on the machine stack
    /// the recovery uses and a bound on the stack every later traversal of the
    /// result uses, the derived `Drop` and `Clone` included. It also caps the
    /// number of folds one condition may take.
    ///
    /// [`DEFAULT_MAX_DEPTH`] is the value the entry points use. Clamped to at
    /// least 1, so depth 1 is never limited and the recovery always terminates.
    /// Reaching the bound costs recovery quality, never correctness: see
    /// [`StructureMetrics::depth_limited`](crate::analysis::structure::StructureMetrics::depth_limited).
    pub max_depth: usize,

    /// The two-way blocks whose successors' order says only which way their
    /// condition reads: a conditional branch, whose first successor is the
    /// arm taken where the condition holds, never a two-case jump table,
    /// whose order is its cases'.
    ///
    /// The tree recovered from a graph should not depend on which way each
    /// condition is stated, and which way that is depends on the lift: a
    /// fold that reads a flag clear exchanges its branch's targets. So each
    /// such block is read in an order of its own successors' ids, every one
    /// ascending and then every one descending, the cheaper recovery is kept
    /// (of two alike, the one nesting its blocks less deep, an `else if`
    /// chain as deep as its first `if`, and then the one placing its blocks
    /// more nearly in the order their ids lay them out), and its tree is
    /// stated in the order given: a test read the other way
    /// round is negated, and a case test's or a two-way head's successor
    /// indices exchanged. An empty set reads every block in the order given.
    pub polar: BlockSet,

    /// Blocks that stand for one edge: each is entered from one block, goes
    /// on to one, and holds nothing of the program's own code, only what the
    /// caller writes on that edge (the copies a join's values take there).
    ///
    /// Such a block is where its edge's statements are said, not a place of
    /// its own control goes to. A loop that leaves through one reaches the
    /// block it goes on to, so that block resumes the loop where it is the
    /// one every way out reaches, and the edge block is placed with the
    /// `break` that leaves; a condition may say its statements ahead of the
    /// test whose arm it is ([`Predicate::Hoisted`](crate::analysis::structure::Predicate::Hoisted)). An empty set reads every
    /// block as a place of its own.
    pub edge_blocks: BlockSet,

    /// Blocks that end with the whole of another block's test, by that
    /// block: run at the end of the block, the test's rows compute what the
    /// block's last rows compute and its branch goes where the block's goes,
    /// so the block may go to the test instead.
    ///
    /// A compiler that rotates a loop tests its condition again at the end of
    /// a lap and goes straight on to the body, past the test at the loop's
    /// head (dash's `expmeta` re-reads `int_pending()` at the end of one arm
    /// and goes on to `readdir`): the body is then entered at two blocks, and
    /// the loop recovers as two whose outer `continue`s are gotos. Routed to
    /// the test, on trial and kept where that reads with fewer gotos, the lap
    /// ends where the others do. The caller states only what it proved: the
    /// test's rows are the block's last ones, its branch the block's own
    /// closing branch or its negation, taken to the same blocks. An empty map
    /// is always safe.
    pub repeated_tests: BTreeMap<NodeId, NodeId>,

    /// Edges a compiler threaded past a test the path decides, by the block
    /// and the index of the edge among its successors, and the test they
    /// went past.
    ///
    /// A compiler that knows on a path what a test will find sends the path
    /// straight to where the test would: dash's `argstr` inlines `expbackq`,
    /// whose cleanup tests `in.buf` and then `in.fd >= 0`, and on the exit
    /// that left its read loop because `in.fd < 0` the `in.buf` test goes
    /// past the `in.fd` test. The test is then entered from fewer of its ways
    /// in and the code after it reached past it too, so the cleanup recovers
    /// twice over with gotos between. Routed through the test, on trial and
    /// kept where that reads with fewer gotos, the edge reaches what it
    /// reached: the test finds what the path decided and goes where the edge
    /// went. The caller states only what it proved: the path decides the
    /// test, and the test's other rows define nothing the edge's target
    /// reads before it is written again. An empty map is always safe.
    pub threaded_tests: BTreeMap<(NodeId, usize), NodeId>,

    /// The statements each block prints, by node: what placing the block once
    /// more costs a reader. A split copies blocks to give a region one entry
    /// and is kept only where the gotos it saves are worth the statements it
    /// copies. A block past the end of the list counts as one; empty, every
    /// block does.
    pub statements: Vec<usize>,
}

impl<'a> StructureOptions<'a> {
    /// The options of a graph whose nodes carry no statements, with `regions`
    /// its protected regions.
    ///
    /// A caller holding a bare graph has no instruction-level view of it, so
    /// every block is declared [`condition-only`](Self::condition_only): its
    /// nodes carry nothing that folding a condition or hoisting a `while`
    /// header could misplace. Nothing can render a statement, so nothing is
    /// expressible inside a condition either; no block is a case test, none
    /// ends the function, and the recovery nests [`DEFAULT_MAX_DEPTH`] deep. A
    /// caller whose nodes *do* carry statements states the sets it can
    /// justify, or the recovery may move work that cannot move.
    ///
    /// # Arguments
    ///
    /// * `node_count` - The graph's node count.
    /// * `regions` - The protected regions to recover.
    #[must_use]
    pub fn bare(node_count: usize, regions: &'a [ProtectedRegion]) -> Self {
        Self {
            regions,
            condition_only: BlockSet::full(node_count),
            // Every block claimed foldable with nothing said about why, so
            // every edge keeps the reachable sets as they were.
            condition_reachability: BlockSet::full(node_count),
            condition_expressible: BlockSet::new(node_count),
            condition_operand: BlockSet::new(node_count),
            case_tests: BTreeMap::new(),
            no_return: BlockSet::new(node_count),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(node_count),
            edge_blocks: BlockSet::new(node_count),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
            statements: Vec::new(),
        }
    }
}

/// Returns `set` sized to `node_count`, counting any resize into `resized`.
///
/// The single normalisation point. Every [`BlockSet`] the walk reads comes
/// through here, so the walk's own reads need no width check and cannot get a
/// silently wrong answer from a set built for another function.
pub fn normalised(node_count: usize, set: &BlockSet, resized: &mut usize) -> BlockSet {
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
pub fn normalised_region(
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
