//! SSA analyses: target-agnostic dataflow, liveness, type/value tracking,
//! constant evaluation, memory model, and symbolic execution.
//!
//! # Architecture
//!
//! The analysis module is organized into several layers:
//!
//! ## Core Analyses
//!
//! - **`cfg`**: The control flow graph view of an SSA function, carrying both
//!   relations control has: the transfers terminators name, and the transfers a
//!   throw makes. The complete relation is what the plain accessors return, so
//!   an analysis is exception-aware unless it asks for the narrow
//!   `terminator_*` view. Bridges the gap between passes (which receive
//!   `SsaFunction`) and dataflow analyses (which require a CFG). Constructed in
//!   O(E) time.
//! - **`exceptions`**: What the graph is built from, and dominance over it.
//!   `FunctionRoots` says where control can enter a function - the entry block
//!   plus every handler and filter entry - and which blocks a throw in reaches
//!   which root; `EhDominance` is dominance over the resulting graph, keeping
//!   apart the three questions callers ask (may a pass rewrite through here,
//!   does this block dominate that one, is this definition well-formed at all).
//! - **`defuse`**: Def-Use index providing O(1) lookups of definition sites, use sites,
//!   and per-location variable queries. Built in O(n) time where n is instruction count.
//! - **`liveness`**: Backward dataflow to compute live-in blocks for each variable group.
//!   Used to prune phi placement (dead-on-arrival phi avoidance).
//! - **`constraints`**: Constraint types derived from branch conditions for path-aware
//!   SSA evaluation. Supports equality, inequality, signed/unsigned ordering constraints.
//! - **`range`**: Interval-based value range analysis with a lattice structure supporting
//!   constant, bounded, half-open, and union ranges for opaque predicate detection and
//!   bounds check elimination.
//! - **`congruence`**: What a value can be *modulo a power of two*, the
//!   independent half of `range`: an interval knows nothing about parity and a
//!   congruence nothing about magnitude. Machine arithmetic is already modular,
//!   so `+`, `-`, `*`, `<<` and the bitwise operations are exact here, which
//!   decides the quadratic opaque predicates no interval settles.
//! - **`address`**: Normalized `base + index*stride + offset` address model recovered from
//!   `PtrAdd` or shredded `Shl`/`Mul`/`Add` chains, plus the cell-identity `AliasKey`
//!   projection. The decoded form is stable under GVN/LICM rewrites of the address
//!   computation, which keying on the address value id is not.
//!
//! ## Evaluation
//!
//! - **`evaluator`**: Hybrid concrete/symbolic SSA interpreter that computes values
//!   for arithmetic and logical operations given known inputs. Supports path-aware
//!   phi evaluation, constraint tracking, and fixed-point loop iteration.
//! - **`consts`**: Constant folding engine with caching and cycle detection. Used by
//!   multiple passes (unflattening, decryption, SCCP). Depth-limited recursive evaluation.
//! - **`resolver`**: Three-tier constant resolver composing ConstEvaluator, PhiAnalyzer,
//!   and optionally SsaEvaluator for demand-driven constant resolution.
//!
//! ## Symbolic Execution
//!
//! - **`symbolic/expr`**: Symbolic expression tree (`SymbolicExpr`) representing SSA
//!   operations as trees with constants, variables, and operation nodes.
//! - **`symbolic/ops`**: Operation types for symbolic expressions (arithmetic, bitwise,
//!   comparison with signed/unsigned variants).
//! - **`symbolic/evaluator`**: Builds symbolic expression trees from SSA operations for
//!   host-side constraint solving.
//!
//! ## SSA Structure Analysis
//!
//! - **`loops`**: Natural loop detection using dominance-based back edge detection.
//!   Computes preheaders, latches, exit edges, loop type classification, nesting
//!   relationships, and induction variable detection.
//! - **`loopanalyzer`**: Convenience wrapper around `detect_loops` providing
//!   SSA-specific loop analysis interface.
//! - **`phis`**: Phi node analysis utilities (trivial phi detection, uniform constant
//!   detection) and pruned phi placement at iterated dominance frontiers.
//! - **`algebraic`**: Algebraic identity simplification (XOR self-cancellation,
//!   identity/absorbing element detection).
//! - **`bits`**: Which bits of an integer value are known clear or set: a
//!   value every bit of which is known is a constant (a field read back after
//!   a write of it into an unknown container), and an `and` whose operand has
//!   no bit outside the mask is its operand.
//! - **`records`**: Record layouts from the indirect accesses measured from one
//!   base - `recover_layouts` reports the fields a base's accesses agree on.
//! - **`flags`**: Reading a flag through `MergeFlags` and `InsertFlag` - which
//!   input the bits a `ReadFlags` selects came from, or the boolean one bit was
//!   set from - and folding a branch on an expanded flag condition back into the
//!   comparison a host recorded for it.
//! - **`convert`**: Integer conversion-chain collapse - `collapse_conversion_chain`
//!   decides when a conversion may read straight past the conversions inside it,
//!   and how far past, from the (width, signedness) pair at each link. Walks the
//!   chain through the guarded def-site lookup, so a stale site costs the
//!   optimization rather than the answer.
//! - **`patterns`**: Obfuscation pattern detection (control flow flattening dispatchers,
//!   opaque predicates, source block identification).
//! - **`taint`**: Generic forward/backward taint propagation with configurable PHI
//!   handling modes. Used for CFF state tracking and cleanup neutralization.
//! - **`recovery`**: The bridge from an `SsaFunction` to `structure`. Reads the
//!   exception table into `ProtectedRegions` (grouping clauses that share a
//!   protected range into one region with several handlers, and returning the
//!   clauses it could not represent rather than dropping them), derives which
//!   blocks are condition-only from their instructions, and hands both to the
//!   structurer. `structure_ssa` is the one call a host holding a function
//!   makes.
//! - **`structure`**: Structured control-flow recovery - turns a control-flow
//!   graph into a tree of statements (`Region`), classifying loops by where they
//!   test, folding multi-block conditions back into `a && b`, and placing
//!   protected regions with their handlers. Total by construction: edges no
//!   structured form expresses become `Region::Goto`, so recovery quality is a
//!   count (`StructureMetrics`) rather than a precondition.
//!
//! ## Memory Analysis
//!
//! - **`memory`**: Memory SSA (MSSA) for tracking versioned memory locations. Supports
//!   static fields, instance fields, array elements, and indirect accesses through
//!   a hierarchical alias analysis.
//! - **`pointsto`**: Inclusion-based (Andersen) points-to analysis - a standalone
//!   constraint solver plus an intraprocedural, field-sensitive extractor. A may-analysis:
//!   field-cell exhaustion falls back to the whole object, and an exhausted step
//!   budget answers every location "may point anywhere".
//! - **`image`**: What a host knows about the image a program was loaded from -
//!   regions, whether each may be written, and their initial bytes - behind the
//!   `ImageData` oracle.
//! - **`bindingtime`**: Which values are known before the program runs.
//!   `Static` under `Dynamic`, over values and the regions loads read from; a
//!   merge also takes the binding time of the branches deciding it. The
//!   sibling of `taint` with the opposite polarity: unknown is the top, and
//!   what is not shown static is dynamic. A host adds what one function cannot
//!   say - what a call does, what a location holds on entry - through a
//!   `BindingOracle`.
//! - **`execution`**: Static execution. Evaluates what the program alone fixes:
//!   constants, the read-only image, and memory addressed as a root plus an
//!   offset. Everything else is carried as unknown.
//!
//! ## Context and Budgets
//!
//! - **`context`**: `AnalysisContext`, the target-instance facts an analysis needs
//!   and a function does not carry: pointer width, the image oracle, the work budget.
//! - **`budget`**: `WorkBudget`, every work budget in one place, and the
//!   `Exhaustion` an analysis records when it runs out.
//!
//! ## Dataflow Framework
//!
//! - **`dataflow/framework`**: Generic dataflow analysis traits (`DataFlowAnalysis`,
//!   `DataFlowCfg`) and direction abstraction.
//! - **`dataflow/lattice`**: Lattice traits (MeetSemiLattice, JoinSemiLattice, Lattice)
//!   with BitSet implementations for may/must analysis.
//! - **`dataflow/solver`**: Worklist-based iterative fixpoint solver using reverse
//!   postorder traversal. Converges in O(n*h) on reducible CFGs.
//! - **`dataflow/liveness`**: Backward live variable analysis computing USE/DEF sets.
//! - **`dataflow/reaching`**: Forward reaching definitions analysis (simplified for SSA).
//! - **`dataflow/sccp`**: Sparse Conditional Constant Propagation combining sparse
//!   def-use analysis with branch condition pruning. Uses edge-based phi evaluation
//!   per Wegman & Zadeck 1991.
//!
//! ## Memoization
//!
//! - **`cache`**: `FunctionAnalyses` - the derived analyses of one SSA function,
//!   each computed at most once and reused while the IR they were derived from
//!   is provably unchanged, since the handle borrows it.
//!
//! ## Verification
//!
//! - **`verifier`**: SSA invariant verifier at three levels (Quick/Standard/Full)
//!   checking single-definition, def-use chains, phi operand coverage, dominance,
//!   and structural integrity.

pub mod address;
pub mod algebraic;
pub mod bindingtime;
pub mod bits;
pub mod budget;
pub mod byteswap;
pub mod cache;
mod cells;
pub mod cfg;
pub mod congruence;
pub mod constraints;
pub mod consts;
pub mod context;
pub mod convert;
pub mod copies;
pub mod dataflow;
pub mod defuse;
pub mod demand;
pub mod distribute;
pub mod doubleword;
pub mod evaluator;
pub mod exceptions;
pub mod execution;
pub mod flags;
pub mod folding;
pub mod frame;
pub mod guards;
pub mod image;
mod inactive;
pub mod liveness;
pub mod loopanalyzer;
pub mod loops;
pub mod mba;
pub mod memory;
mod operands;
pub mod patterns;
pub mod phis;
pub mod pointsto;
pub mod range;
pub mod reach;
pub mod records;
pub mod recovery;
pub mod reshape;
pub mod resolver;
pub mod rotate;
pub mod select;
pub mod structure;
pub mod symbolic;
pub mod taint;
pub mod verifier;
pub mod webs;

pub use address::{
    AbsoluteUse, AddressBase, AddressExpr, AddressUse, AddressWalk, AliasKey, FrameUse,
    alignment_mask,
};
pub use algebraic::{SimplifyResult, executing_nothing, simplify_op};
pub use bindingtime::{BindingTime, BindingTimes};
pub use budget::{Budget, Exhaustion, Rounds, WorkBudget};
pub use cfg::SsaCfg;
pub use congruence::Congruence;
pub use consts::{ConstEvaluator, Destination, FoldedResults, Machine, MisalignedAccess};
pub use context::{AnalysisContext, CallModels};
pub use convert::collapse_conversion_chain;
pub use copies::ParallelCopy;
pub use defuse::{DefUseIndex, Location};
pub use evaluator::{ControlFlow, EvaluatorMark, SsaEvaluator};
pub use exceptions::{EhDominance, ExceptionRoot, FunctionRoots, RootKind};
pub use execution::{StaticExecution, StaticFacts};
pub use folding::{FoldForest, FoldPolicy};
pub use image::{ImageData, ImageRegion, Mutability, ReadOnlyExtents, ReadOnlyOverlay};
pub use loopanalyzer::{LoopAnalyzer, SsaLoopAnalysis};
pub use loops::{InductionVar, LoopForest, LoopInfo, LoopType, detect_loops};
pub use patterns::PatternDetector;
pub use phis::{PhiAnalyzer, place_pruned_phis};
pub use pointsto::{CallEscape, Constraint, PointsTo};
pub use range::ValueRange;
pub use reach::{
    AddressOrigin, IndexBounds, LoadedValues, Reach, address_origins, for_each_reach,
    greatest_common_step,
};
pub use recovery::{
    ClauseRejection, ProtectedRegions, Recovered, RejectedClause, condition_only_blocks,
    structure_ssa,
};
pub use resolver::ValueResolver;
pub use structure::{
    BlockSet, DEFAULT_MAX_DEPTH, HandlerFilter, HandlerKind, HandlerRegion, ProtectedHandler,
    ProtectedHandlerKind, ProtectedRegion, Region, StructureMetrics, StructureOptions, Structured,
    structure,
};
pub use symbolic::{SymbolicEvaluator, SymbolicExpr};
pub use verifier::{SsaVerifier, VerifierError, VerifyLevel};
pub use webs::{Conflicts, Gathered, Gathering, Group, first_fit_partition};
