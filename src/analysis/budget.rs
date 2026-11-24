//! Work budgets: how much an analysis may do, and what it reports when that is
//! not enough.
//!
//! Two kinds of limit live in this crate and they are kept apart on purpose.
//! A **termination guard** - the depth of an address walk, the size of a
//! symbolic expression - is part of an algorithm's definition: it bounds a
//! recursion that would otherwise be unbounded on a malformed input, and it
//! stays a constant beside the algorithm it guards. A **work budget** bounds
//! how much an analysis may spend on an input that is merely large, which is a
//! policy a host may want to set, so every one of them lives here, in one
//! [`WorkBudget`].
//!
//! A work budget is also required to be **observable**. An analysis that stops
//! early and returns what it had produced a quiet wrong answer - the failure
//! this crate's alias proofs exist to rule out - unless what it returns is the
//! conservative one *and* it says it stopped. So each budgeted analysis falls
//! back to its least informative answer and records an [`Exhaustion`] a caller
//! can read, and the passes built on them report it through their event sink.

use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// One budget [`WorkBudget`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Budget {
    /// The `locations x blocks` product plus may-alias instruction definitions
    /// [`MemorySsa`](crate::analysis::memory::MemorySsa) may build. Renaming
    /// records an entry and an exit version per location per block, so the
    /// retained structure is quadratic when the location count scales with the
    /// block count - which it does for data misread as code. Instruction-local
    /// version queries additionally index each location changed by a write.
    MemorySsaCells,
    /// Steps the [points-to solver](crate::analysis::pointsto::solve) may take,
    /// per constraint. Inclusion-based points-to is cubic in the worst case,
    /// and a crafted function with dense aliasing reaches it.
    PointsToStepsPerConstraint,
    /// Bytes of image contents one function's loads may be folded from.
    ImageBytes,
    /// Instructions a [static execution](crate::analysis::execution) may run,
    /// its calls included. A decode loop runs once per byte it decodes, so this
    /// is what bounds how large a buffer is decoded - and a loop the program
    /// never leaves. A block instruction and a library model cost one more step
    /// per byte they span, as the loop they stand for would.
    ExecutionSteps,
    /// How deep a static execution may nest the calls it executes in place,
    /// the root's activation included. A body entered again while it is on the
    /// stack, by itself or through others, runs as an activation of its own,
    /// and recursion the program fixes the depth of still has to stop
    /// somewhere.
    ExecutionDepth,
    /// Paths a static execution may explore past branches it cannot decide.
    /// Each undecided branch it explores doubles what is left to run.
    ExecutionPaths,
    /// Known prefix bytes retained behind one call argument.
    ExecutionArgumentBytes,
    /// Bytes inspected while a library model searches for a string terminator.
    ExecutionStringBytes,
    /// Bytes interpreted by one native block-memory instruction.
    ExecutionBlockBytes,
    /// Bytes in one typed store, materialized execution value or library-model buffer.
    ExecutionBufferBytes,
    /// Rounds over the function the [range fixpoint](crate::passes::ranges)
    /// may take: it may process this many times the function's blocks plus
    /// variables, and a small fixed allowance, in edges and value updates.
    /// Widening is what terminates the
    /// fixpoint, so this is the backstop for a constraint count that grows
    /// without the widening catching it, and an unconverged run proves
    /// nothing.
    RangeRounds,
    /// Rounds of [dead-code elimination](crate::passes::deadcode) over a
    /// function, each removing what the round before left unread.
    DeadCodeRounds,
    /// Rounds of [copy propagation](crate::passes::copying), each replacing
    /// the copies the round before exposed.
    CopyRounds,
    /// Rounds of [block merging](crate::passes::blockmerge), each merging the
    /// blocks the round before left in line.
    BlockMergeRounds,
    /// Rounds of [control-flow simplification](crate::passes::controlflow),
    /// each folding the branches the round before decided.
    ControlFlowRounds,
    /// Rounds a host's constant folding may take over a function on top of
    /// [constant propagation](crate::analysis::dataflow::sccp), each folding
    /// what the round before exposed.
    FoldRounds,
    /// Rounds of [loop canonicalization](crate::passes::loopcanon): no measure
    /// provably falls with each round, so this is what ends it.
    CanonicalizationRounds,
    /// Events a [moot test](crate::passes::moot) search may run the arms of
    /// one function's tests through, side by side: a pair of loops is gone
    /// round until it repeats, and a function whose tests all lead into
    /// alike code could be followed far.
    MootSteps,
    /// Addresses a host's relation of where a whole sample's values may point
    /// may record over every value: an address handed to a routine reaches
    /// each value downstream of the routine's parameter, so a routine a
    /// thousand tables are handed to holds each of them at every value it
    /// passes them on to.
    HeldAddresses,
    /// Rounds of a [scheduler](crate::scheduling::PassScheduler) pipeline over
    /// every method, each running every layer once.
    PipelineRounds,
    /// Pipeline rounds that change nothing a [scheduler](crate::scheduling::PassScheduler)
    /// waits for before it stops: a round that changes nothing is the
    /// fixpoint, and more of them only confirm it.
    StableRounds,
    /// Rounds of one layer of a [scheduler](crate::scheduling::PassScheduler)
    /// pipeline, its passes run until none changes a method.
    PhaseRounds,
    /// Visits of each method of a recursive group the
    /// [interprocedural solve](crate::interproc::solve) may make
    /// before the group's summaries are taken as they stand.
    SummaryRounds,
    /// Rounds of [taint propagation](crate::analysis::taint) over a function,
    /// each carrying taint one step further.
    TaintRounds,
    /// Root trials and selected call contexts one image's [static execution](crate::analysis::execution)
    /// admits.
    StaticContexts,
    /// Execution steps charged across every root context of one image, beside
    /// each context's own [`Self::ExecutionSteps`].
    StaticSteps,
    /// Decoded payload bytes retained across one image's contexts: call and exit
    /// byte ranges and inline string, array and vector scalars, each repeated
    /// observation at its retained size.
    StaticOutputBytes,
    /// Rounds of one image's static execution: a round executes every
    /// selected root, and the next runs again over the code versions the
    /// previous one recovered, until a round recovers no new version. The
    /// first round always runs.
    StaticRounds,
    /// Distinct producer and edge sites a host submits for edge observations; none
    /// by default.
    StaticEdgeSites,
    /// Byte and scalar operands a host submits for edge observations, duplicates
    /// counted before merging; none by default.
    StaticEdgeRequests,
    /// Instructions plus phi nodes the complete body holds while a
    /// [residual](crate::passes::residual) request applies, the original body
    /// included until repair removes it.
    ResidualOperations,
    /// Variable slots the complete body holds while a
    /// [residual](crate::passes::residual) request applies.
    ResidualVariables,
    /// Block slots, unreachable ones awaiting repair included, the complete body
    /// holds while a [residual](crate::passes::residual) request applies.
    ResidualBlocks,
    /// Incoming phi operands across the complete body while a
    /// [residual](crate::passes::residual) request applies.
    ResidualPhiOperands,
    /// Known instruction-source prefix bytes inspected for one recovered code version.
    CodeVersionBytes,
    /// Source bytes retained by recovered code versions and attempted source keys
    /// across one invocation.
    CodeTotalBytes,
    /// Distinct recovered code versions retained across one invocation.
    CodeVersions,
    /// Code resolution requests, refused and cached ones included.
    CodeRequests,
    /// Native instructions decoded in one recovered block: a longer block splits
    /// and decodes on, and zero establishes no entry.
    CodeBlockInstructions,
}

impl fmt::Display for Budget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MemorySsaCells => "memory SSA cells",
            Self::PointsToStepsPerConstraint => "points-to steps per constraint",
            Self::ImageBytes => "image bytes folded",
            Self::ExecutionSteps => "static execution steps",
            Self::ExecutionDepth => "static execution call depth",
            Self::ExecutionPaths => "static execution paths",
            Self::ExecutionArgumentBytes => "known prefix bytes retained behind one call argument",
            Self::ExecutionStringBytes => {
                "bytes inspected while a library model searches for a string terminator"
            }
            Self::ExecutionBlockBytes => "bytes interpreted by one native block-memory instruction",
            Self::ExecutionBufferBytes => {
                "bytes in one typed store, materialized execution value or library-model buffer"
            }
            Self::RangeRounds => "range fixpoint rounds",
            Self::DeadCodeRounds => "dead-code elimination rounds",
            Self::CopyRounds => "copy propagation rounds",
            Self::BlockMergeRounds => "block merging rounds",
            Self::ControlFlowRounds => "control-flow simplification rounds",
            Self::CanonicalizationRounds => "loop canonicalization rounds",
            Self::MootSteps => "moot test search events",
            Self::HeldAddresses => "addresses held across one sample's values",
            Self::PipelineRounds => "pipeline rounds",
            Self::StableRounds => "stable pipeline rounds",
            Self::PhaseRounds => "phase rounds",
            Self::SummaryRounds => "interprocedural summary rounds",
            Self::FoldRounds => "constant folding rounds",
            Self::TaintRounds => "taint propagation rounds",
            Self::StaticContexts => "static execution contexts in one image",
            Self::StaticSteps => "static execution steps across one image",
            Self::StaticOutputBytes => "decoded bytes retained across one image",
            Self::StaticRounds => "static execution rounds over one image",
            Self::StaticEdgeSites => "submitted static edge sites",
            Self::StaticEdgeRequests => "submitted static edge requests",
            Self::ResidualOperations => "residual instructions and phi nodes",
            Self::ResidualVariables => "residual variable slots",
            Self::ResidualBlocks => "residual block slots",
            Self::ResidualPhiOperands => "residual phi operands",
            Self::CodeVersionBytes => "recovered source bytes per version",
            Self::CodeTotalBytes => "total recovered source bytes",
            Self::CodeVersions => "recovered code versions",
            Self::CodeRequests => "code resolution requests",
            Self::CodeBlockInstructions => "decoded instructions per recovered block",
        })
    }
}

/// The work budgets every analysis in this crate, and every pass a host builds
/// on them, is bounded by.
///
/// The defaults are far above any real function and exist to bound
/// pathological input, not to trade precision for speed on ordinary code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct WorkBudget {
    /// See [`Budget::MemorySsaCells`].
    memory_ssa_cells: usize,
    /// See [`Budget::PointsToStepsPerConstraint`].
    points_to_steps_per_constraint: usize,
    /// See [`Budget::ImageBytes`].
    image_bytes: usize,
    /// See [`Budget::ExecutionSteps`].
    execution_steps: usize,
    /// See [`Budget::ExecutionDepth`].
    execution_depth: usize,
    /// See [`Budget::ExecutionPaths`].
    execution_paths: usize,
    /// See [`Budget::ExecutionArgumentBytes`].
    execution_argument_bytes: usize,
    /// See [`Budget::ExecutionStringBytes`].
    execution_string_bytes: usize,
    /// See [`Budget::ExecutionBlockBytes`].
    execution_block_bytes: usize,
    /// See [`Budget::ExecutionBufferBytes`].
    execution_buffer_bytes: usize,
    /// See [`Budget::RangeRounds`].
    range_rounds: usize,
    /// See [`Budget::DeadCodeRounds`].
    dead_code_rounds: usize,
    /// See [`Budget::CopyRounds`].
    copy_rounds: usize,
    /// See [`Budget::BlockMergeRounds`].
    block_merge_rounds: usize,
    /// See [`Budget::ControlFlowRounds`].
    control_flow_rounds: usize,
    /// See [`Budget::CanonicalizationRounds`].
    canonicalization_rounds: usize,
    /// See [`Budget::MootSteps`].
    moot_steps: usize,
    /// See [`Budget::HeldAddresses`].
    held_addresses: usize,
    /// See [`Budget::PipelineRounds`].
    pipeline_rounds: usize,
    /// See [`Budget::StableRounds`].
    stable_rounds: usize,
    /// See [`Budget::PhaseRounds`].
    phase_rounds: usize,
    /// See [`Budget::SummaryRounds`].
    summary_rounds: usize,
    /// See [`Budget::TaintRounds`].
    taint_rounds: usize,
    /// See [`Budget::FoldRounds`].
    fold_rounds: usize,
    /// See [`Budget::StaticContexts`].
    static_contexts: usize,
    /// See [`Budget::StaticSteps`].
    static_steps: usize,
    /// See [`Budget::StaticOutputBytes`].
    static_output_bytes: usize,
    /// See [`Budget::StaticRounds`].
    static_rounds: usize,
    /// See [`Budget::StaticEdgeSites`].
    static_edge_sites: usize,
    /// See [`Budget::StaticEdgeRequests`].
    static_edge_requests: usize,
    /// See [`Budget::ResidualOperations`].
    residual_operations: usize,
    /// See [`Budget::ResidualVariables`].
    residual_variables: usize,
    /// See [`Budget::ResidualBlocks`].
    residual_blocks: usize,
    /// See [`Budget::ResidualPhiOperands`].
    residual_phi_operands: usize,
    /// See [`Budget::CodeVersionBytes`].
    code_version_bytes: usize,
    /// See [`Budget::CodeTotalBytes`].
    code_total_bytes: usize,
    /// See [`Budget::CodeVersions`].
    code_versions: usize,
    /// See [`Budget::CodeRequests`].
    code_requests: usize,
    /// See [`Budget::CodeBlockInstructions`].
    code_block_instructions: usize,
}

impl WorkBudget {
    /// The default budgets.
    ///
    /// Memory SSA: 4M cells, where a 2000-block function touching 200 distinct
    /// cells is 400k, bounding the structure at tens of MB. Points-to: 64 steps
    /// per constraint, which every ordinary function converges well inside.
    /// Image bytes: 64 KiB per function, more than any real function reads out
    /// of its constant data. Static execution: 64M instruction, control and
    /// phi steps, 64 nested
    /// calls and 4096 explored paths per execution - set high, to be measured
    /// down, so that what decides whether a buffer decodes is the analysis and
    /// not the budget. Data limits bound each materialized buffer, block transfer,
    /// string scan and retained argument prefix at the operation that owns it.
    /// Range fixpoint: 32 rounds, which every ordinary
    /// function converges well inside. The rewriting passes' rounds - dead
    /// code 50, copies, block merging and control flow 10 each, loop
    /// canonicalization 16, 16K moot test search events - and the scheduler's - 10 pipeline rounds, one
    /// stable round to stop, 10 rounds a layer - are the ones its presets ran
    /// with; 8 visits a method of a recursive group, 100 taint rounds and 10
    /// constant folding rounds; 4M addresses held across one sample's values.
    /// One image's static execution: 128 contexts,
    /// 16M steps and 1 MiB of retained decoded bytes, and no submitted edge
    /// requests. A residual body: 64K operations and variables, 16K blocks and
    /// 256K phi operands. Recovered code: 64 KiB per version, 1 MiB in all,
    /// 128 versions, 512 requests and 10,000 instructions a block.
    pub const DEFAULT: Self = Self {
        memory_ssa_cells: 4_000_000,
        points_to_steps_per_constraint: 64,
        image_bytes: 64 * 1024,
        execution_steps: 64_000_000,
        execution_depth: 64,
        execution_paths: 4096,
        execution_argument_bytes: 4096,
        execution_string_bytes: 1048576,
        execution_block_bytes: 1048576,
        execution_buffer_bytes: 16777216,
        range_rounds: 32,
        dead_code_rounds: 50,
        copy_rounds: 10,
        block_merge_rounds: 10,
        control_flow_rounds: 10,
        canonicalization_rounds: 16,
        moot_steps: 1 << 14,
        held_addresses: 1 << 22,
        pipeline_rounds: 10,
        stable_rounds: 1,
        phase_rounds: 10,
        summary_rounds: 8,
        taint_rounds: 100,
        fold_rounds: 10,
        static_contexts: 128,
        static_steps: 1 << 24,
        static_output_bytes: 1 << 20,
        static_rounds: 4,
        static_edge_sites: 0,
        static_edge_requests: 0,
        residual_operations: 1 << 16,
        residual_variables: 1 << 16,
        residual_blocks: 1 << 14,
        residual_phi_operands: 1 << 18,
        code_version_bytes: 1 << 16,
        code_total_bytes: 1 << 20,
        code_versions: 128,
        code_requests: 512,
        code_block_instructions: 10_000,
    };

    /// Iterates every configured limit without serialization or allocation.
    ///
    /// The order follows the budget vocabulary. Exhaustive field projection keeps
    /// canonical execution-policy consumers aware of future budget additions.
    pub fn limits(&self) -> impl ExactSizeIterator<Item = (Budget, usize)> {
        let Self {
            memory_ssa_cells,
            points_to_steps_per_constraint,
            image_bytes,
            execution_steps,
            execution_depth,
            execution_paths,
            execution_argument_bytes,
            execution_string_bytes,
            execution_block_bytes,
            execution_buffer_bytes,
            range_rounds,
            dead_code_rounds,
            copy_rounds,
            block_merge_rounds,
            control_flow_rounds,
            canonicalization_rounds,
            moot_steps,
            held_addresses,
            pipeline_rounds,
            stable_rounds,
            phase_rounds,
            summary_rounds,
            taint_rounds,
            fold_rounds,
            static_contexts,
            static_steps,
            static_output_bytes,
            static_rounds,
            static_edge_sites,
            static_edge_requests,
            residual_operations,
            residual_variables,
            residual_blocks,
            residual_phi_operands,
            code_version_bytes,
            code_total_bytes,
            code_versions,
            code_requests,
            code_block_instructions,
        } = *self;
        [
            (Budget::MemorySsaCells, memory_ssa_cells),
            (
                Budget::PointsToStepsPerConstraint,
                points_to_steps_per_constraint,
            ),
            (Budget::ImageBytes, image_bytes),
            (Budget::ExecutionSteps, execution_steps),
            (Budget::ExecutionDepth, execution_depth),
            (Budget::ExecutionPaths, execution_paths),
            (Budget::ExecutionArgumentBytes, execution_argument_bytes),
            (Budget::ExecutionStringBytes, execution_string_bytes),
            (Budget::ExecutionBlockBytes, execution_block_bytes),
            (Budget::ExecutionBufferBytes, execution_buffer_bytes),
            (Budget::RangeRounds, range_rounds),
            (Budget::DeadCodeRounds, dead_code_rounds),
            (Budget::CopyRounds, copy_rounds),
            (Budget::BlockMergeRounds, block_merge_rounds),
            (Budget::ControlFlowRounds, control_flow_rounds),
            (Budget::CanonicalizationRounds, canonicalization_rounds),
            (Budget::MootSteps, moot_steps),
            (Budget::HeldAddresses, held_addresses),
            (Budget::PipelineRounds, pipeline_rounds),
            (Budget::StableRounds, stable_rounds),
            (Budget::PhaseRounds, phase_rounds),
            (Budget::SummaryRounds, summary_rounds),
            (Budget::TaintRounds, taint_rounds),
            (Budget::FoldRounds, fold_rounds),
            (Budget::StaticContexts, static_contexts),
            (Budget::StaticSteps, static_steps),
            (Budget::StaticOutputBytes, static_output_bytes),
            (Budget::StaticRounds, static_rounds),
            (Budget::StaticEdgeSites, static_edge_sites),
            (Budget::StaticEdgeRequests, static_edge_requests),
            (Budget::ResidualOperations, residual_operations),
            (Budget::ResidualVariables, residual_variables),
            (Budget::ResidualBlocks, residual_blocks),
            (Budget::ResidualPhiOperands, residual_phi_operands),
            (Budget::CodeVersionBytes, code_version_bytes),
            (Budget::CodeTotalBytes, code_total_bytes),
            (Budget::CodeVersions, code_versions),
            (Budget::CodeRequests, code_requests),
            (Budget::CodeBlockInstructions, code_block_instructions),
        ]
        .into_iter()
    }

    /// Returns the rounds a fixpoint loop may run under `budget`'s limit.
    #[must_use]
    pub const fn rounds(&self, budget: Budget) -> Rounds {
        Rounds::new(budget, self.limit(budget))
    }

    /// Returns the limit set for `budget`.
    #[must_use]
    pub const fn limit(&self, budget: Budget) -> usize {
        match budget {
            Budget::MemorySsaCells => self.memory_ssa_cells,
            Budget::PointsToStepsPerConstraint => self.points_to_steps_per_constraint,
            Budget::ImageBytes => self.image_bytes,
            Budget::ExecutionSteps => self.execution_steps,
            Budget::ExecutionDepth => self.execution_depth,
            Budget::ExecutionPaths => self.execution_paths,
            Budget::ExecutionArgumentBytes => self.execution_argument_bytes,
            Budget::ExecutionStringBytes => self.execution_string_bytes,
            Budget::ExecutionBlockBytes => self.execution_block_bytes,
            Budget::ExecutionBufferBytes => self.execution_buffer_bytes,
            Budget::RangeRounds => self.range_rounds,
            Budget::DeadCodeRounds => self.dead_code_rounds,
            Budget::CopyRounds => self.copy_rounds,
            Budget::BlockMergeRounds => self.block_merge_rounds,
            Budget::ControlFlowRounds => self.control_flow_rounds,
            Budget::CanonicalizationRounds => self.canonicalization_rounds,
            Budget::MootSteps => self.moot_steps,
            Budget::HeldAddresses => self.held_addresses,
            Budget::PipelineRounds => self.pipeline_rounds,
            Budget::StableRounds => self.stable_rounds,
            Budget::PhaseRounds => self.phase_rounds,
            Budget::SummaryRounds => self.summary_rounds,
            Budget::TaintRounds => self.taint_rounds,
            Budget::FoldRounds => self.fold_rounds,
            Budget::StaticContexts => self.static_contexts,
            Budget::StaticSteps => self.static_steps,
            Budget::StaticOutputBytes => self.static_output_bytes,
            Budget::StaticRounds => self.static_rounds,
            Budget::StaticEdgeSites => self.static_edge_sites,
            Budget::StaticEdgeRequests => self.static_edge_requests,
            Budget::ResidualOperations => self.residual_operations,
            Budget::ResidualVariables => self.residual_variables,
            Budget::ResidualBlocks => self.residual_blocks,
            Budget::ResidualPhiOperands => self.residual_phi_operands,
            Budget::CodeVersionBytes => self.code_version_bytes,
            Budget::CodeTotalBytes => self.code_total_bytes,
            Budget::CodeVersions => self.code_versions,
            Budget::CodeRequests => self.code_requests,
            Budget::CodeBlockInstructions => self.code_block_instructions,
        }
    }

    /// Returns these budgets with `budget` set to `limit`.
    ///
    /// # Arguments
    ///
    /// * `budget` - Which budget to set.
    /// * `limit` - Its new limit.
    #[must_use]
    pub const fn with_limit(mut self, budget: Budget, limit: usize) -> Self {
        match budget {
            Budget::MemorySsaCells => self.memory_ssa_cells = limit,
            Budget::PointsToStepsPerConstraint => self.points_to_steps_per_constraint = limit,
            Budget::ImageBytes => self.image_bytes = limit,
            Budget::ExecutionSteps => self.execution_steps = limit,
            Budget::ExecutionDepth => self.execution_depth = limit,
            Budget::ExecutionPaths => self.execution_paths = limit,
            Budget::ExecutionArgumentBytes => self.execution_argument_bytes = limit,
            Budget::ExecutionStringBytes => self.execution_string_bytes = limit,
            Budget::ExecutionBlockBytes => self.execution_block_bytes = limit,
            Budget::ExecutionBufferBytes => self.execution_buffer_bytes = limit,
            Budget::RangeRounds => self.range_rounds = limit,
            Budget::DeadCodeRounds => self.dead_code_rounds = limit,
            Budget::CopyRounds => self.copy_rounds = limit,
            Budget::BlockMergeRounds => self.block_merge_rounds = limit,
            Budget::ControlFlowRounds => self.control_flow_rounds = limit,
            Budget::CanonicalizationRounds => self.canonicalization_rounds = limit,
            Budget::MootSteps => self.moot_steps = limit,
            Budget::HeldAddresses => self.held_addresses = limit,
            Budget::PipelineRounds => self.pipeline_rounds = limit,
            Budget::StableRounds => self.stable_rounds = limit,
            Budget::PhaseRounds => self.phase_rounds = limit,
            Budget::SummaryRounds => self.summary_rounds = limit,
            Budget::TaintRounds => self.taint_rounds = limit,
            Budget::FoldRounds => self.fold_rounds = limit,
            Budget::StaticContexts => self.static_contexts = limit,
            Budget::StaticSteps => self.static_steps = limit,
            Budget::StaticOutputBytes => self.static_output_bytes = limit,
            Budget::StaticRounds => self.static_rounds = limit,
            Budget::StaticEdgeSites => self.static_edge_sites = limit,
            Budget::StaticEdgeRequests => self.static_edge_requests = limit,
            Budget::ResidualOperations => self.residual_operations = limit,
            Budget::ResidualVariables => self.residual_variables = limit,
            Budget::ResidualBlocks => self.residual_blocks = limit,
            Budget::ResidualPhiOperands => self.residual_phi_operands = limit,
            Budget::CodeVersionBytes => self.code_version_bytes = limit,
            Budget::CodeTotalBytes => self.code_total_bytes = limit,
            Budget::CodeVersions => self.code_versions = limit,
            Budget::CodeRequests => self.code_requests = limit,
            Budget::CodeBlockInstructions => self.code_block_instructions = limit,
        }
        self
    }

    /// Checks `demand` against `budget`, returning the exhaustion when it is
    /// over.
    ///
    /// # Arguments
    ///
    /// * `budget` - Which budget the demand is measured against.
    /// * `demand` - What the analysis would need.
    #[must_use]
    pub const fn check(&self, budget: Budget, demand: usize) -> Option<Exhaustion> {
        let limit = self.limit(budget);
        if demand > limit {
            Some(Exhaustion {
                budget,
                limit,
                demand,
            })
        } else {
            None
        }
    }
}

impl Default for WorkBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A budget an analysis ran out of, and what it would have needed.
///
/// Recorded by the analysis that stopped, which then answers conservatively;
/// a consumer reports it so the precision lost is visible rather than silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Exhaustion {
    /// The budget exceeded.
    pub budget: Budget,
    /// Its limit.
    pub limit: usize,
    /// What the analysis needed, as far as it measured before stopping.
    pub demand: usize,
}

impl Exhaustion {
    /// Returns whether the demand exceeds the limit.
    #[must_use]
    pub const fn is_exhausted(self) -> bool {
        self.demand > self.limit
    }
}

impl fmt::Display for Exhaustion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} budget exhausted: needed at least {}, limit {}",
            self.budget, self.demand, self.limit
        )
    }
}

/// The rounds a rewriting loop takes out of one of its budgets.
///
/// A loop that rewrites until a round changes nothing ends on its own when the
/// rewrite settles, and its budget only stops a loop that does not. Both ends
/// are kept apart here: a loop takes each round with [`Rounds::next_round`],
/// says it settled with [`Rounds::settle`], and a loop that used every round it
/// was allowed without settling is [`Rounds::exhausted`]. What it produced is
/// still sound, only less simplified than its fixpoint, and the exhaustion is
/// what says so.
///
/// # Examples
///
/// ```
/// use analyssa::analysis::budget::{Budget, Rounds};
///
/// let mut rounds = Rounds::new(Budget::CopyRounds, 2);
/// let mut left = 5;
/// while rounds.next_round() {
///     if left == 0 {
///         rounds.settle();
///     } else {
///         left -= 1;
///     }
/// }
/// let exhausted = rounds.exhausted().expect("two rounds do not settle five");
/// assert_eq!((exhausted.limit, exhausted.demand), (2, 3));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rounds {
    /// The budget the rounds are counted against.
    budget: Budget,
    /// The rounds allowed.
    limit: usize,
    /// The rounds taken so far.
    taken: usize,
    /// Whether a round changed nothing, which ends the loop.
    settled: bool,
}

impl Rounds {
    /// Starts counting the rounds of a loop `budget` allows `limit` of.
    ///
    /// # Arguments
    ///
    /// * `budget` - The budget the rounds are counted against.
    /// * `limit` - The rounds allowed, normally that budget's limit.
    #[must_use]
    pub const fn new(budget: Budget, limit: usize) -> Self {
        Self {
            budget,
            limit,
            taken: 0,
            settled: false,
        }
    }

    /// Takes the next round, returning whether one may run: `false` once a
    /// round settled or every allowed round ran.
    pub const fn next_round(&mut self) -> bool {
        if self.settled || self.taken >= self.limit {
            return false;
        }
        self.taken = self.taken.saturating_add(1);
        true
    }

    /// Records that the round just taken changed nothing: the loop reached its
    /// fixpoint and takes no more rounds.
    pub const fn settle(&mut self) {
        self.settled = true;
    }

    /// The exhaustion of the budget when every allowed round ran without one
    /// settling: the fixpoint needed at least one round more.
    #[must_use]
    pub const fn exhausted(&self) -> Option<Exhaustion> {
        if self.settled || self.taken < self.limit {
            return None;
        }
        Some(Exhaustion {
            budget: self.budget,
            limit: self.limit,
            demand: self.limit.saturating_add(1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loop that settles inside its rounds is not exhausted, even on its
    /// last allowed round; one whose every round changed something is, by one
    /// round; and a zero limit allows no round at all.
    #[test]
    fn rounds_are_exhausted_only_when_every_allowed_round_changed_something() {
        let run = |limit: usize, changing: usize| {
            let mut rounds = Rounds::new(Budget::DeadCodeRounds, limit);
            let mut ran = 0usize;
            while rounds.next_round() {
                ran = ran.saturating_add(1);
                if ran > changing {
                    rounds.settle();
                }
            }
            (ran, rounds.exhausted())
        };
        assert_eq!(run(4, 1), (2, None), "settles on its second round");
        assert_eq!(run(4, 3), (4, None), "settles on its last round");
        assert_eq!(
            run(4, 4),
            (
                4,
                Some(Exhaustion {
                    budget: Budget::DeadCodeRounds,
                    limit: 4,
                    demand: 5,
                })
            ),
            "every round changed something"
        );
        assert_eq!(run(0, 0).0, 0, "a zero limit runs nothing");
        assert!(run(0, 0).1.is_some(), "and says the loop never settled");
    }

    /// A demand at the limit fits; one past it is reported with both numbers.
    #[test]
    fn a_demand_past_the_limit_is_an_exhaustion() {
        let budget = WorkBudget::DEFAULT.with_limit(Budget::ImageBytes, 16);
        assert_eq!(budget.check(Budget::ImageBytes, 16), None);
        assert_eq!(
            budget.check(Budget::ImageBytes, 17),
            Some(Exhaustion {
                budget: Budget::ImageBytes,
                limit: 16,
                demand: 17,
            })
        );
        assert_eq!(
            budget.limit(Budget::MemorySsaCells),
            WorkBudget::DEFAULT.limit(Budget::MemorySsaCells),
            "setting one budget leaves the others"
        );
    }
}
