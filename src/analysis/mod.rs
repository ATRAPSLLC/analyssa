//! SSA analyses: target-agnostic dataflow, liveness, type/value tracking,
//! constant evaluation, memory model, and symbolic execution.
//!
//! # Architecture
//!
//! The analysis module is organized into several layers:
//!
//! ## Core Analyses
//!
//! - **`cfg`**: Lightweight control flow graph view built from SSA function terminators.
//!   Bridges the gap between passes (which receive `SsaFunction`) and dataflow analyses
//!   (which require a CFG). Constructed in O(E) time.
//! - **`exceptions`**: The exception-aware flow view. `FunctionRoots` says where
//!   control can enter a function — the entry block plus every handler and
//!   filter entry; `EhCfg` is the graph in which those entries are reachable;
//!   `EhDominance` is dominance over it, keeping apart the three questions
//!   callers ask (may a pass rewrite through here, does this block dominate
//!   that one, is this definition well-formed at all).
//! - **`defuse`**: Def-Use index providing O(1) lookups of definition sites, use sites,
//!   and per-location variable queries. Built in O(n) time where n is instruction count.
//! - **`liveness`**: Backward dataflow to compute live-in blocks for each variable group.
//!   Used to prune phi placement (dead-on-arrival phi avoidance).
//! - **`constraints`**: Constraint types derived from branch conditions for path-aware
//!   SSA evaluation. Supports equality, inequality, signed/unsigned ordering constraints.
//! - **`range`**: Interval-based value range analysis with a lattice structure supporting
//!   constant, bounded, half-open, and union ranges for opaque predicate detection and
//!   bounds check elimination.
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
//! - **`loop_analyzer`**: Convenience wrapper around `detect_loops` providing
//!   SSA-specific loop analysis interface.
//! - **`phis`**: Phi node analysis utilities (trivial phi detection, uniform constant
//!   detection) and pruned phi placement at iterated dominance frontiers.
//! - **`algebraic`**: Algebraic identity simplification (XOR self-cancellation,
//!   identity/absorbing element detection).
//! - **`convert`**: Integer conversion-chain collapse — `collapse_conversion_chain`
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
//! - **`structure`**: Structured control-flow recovery — turns a control-flow
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
//! - **`pointsto`**: Inclusion-based (Andersen) points-to analysis — a standalone
//!   constraint solver plus an intraprocedural, field-sensitive extractor. A may-analysis:
//!   both the step budget and field-cell exhaustion lose precision rather than soundness.
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
//! - **`cache`**: `FunctionAnalyses` — the derived analyses of one SSA function,
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
pub mod cache;
pub mod cfg;
pub mod constraints;
pub mod consts;
pub mod convert;
pub mod dataflow;
pub mod defuse;
pub mod evaluator;
pub mod exceptions;
pub mod liveness;
pub mod loop_analyzer;
pub mod loops;
pub mod memory;
pub mod patterns;
pub mod phis;
pub mod pointsto;
pub mod range;
pub mod recovery;
pub mod resolver;
pub mod structure;
pub mod symbolic;
pub mod taint;
pub mod verifier;

pub use address::{AddressExpr, AliasKey, normalize_address};
pub use algebraic::{SimplifyResult, simplify_op};
pub use cfg::SsaCfg;
pub use consts::{ConstEvaluator, evaluate_const_op};
pub use convert::collapse_conversion_chain;
pub use defuse::{DefUseIndex, Location};
pub use evaluator::{ControlFlow, EvaluatorMark, SsaEvaluator};
pub use exceptions::{EhCfg, EhDominance, ExceptionRoot, FunctionRoots, RootKind};
pub use loop_analyzer::{LoopAnalyzer, SsaLoopAnalysis};
pub use loops::{InductionVar, LoopForest, LoopInfo, LoopType, detect_loops};
pub use patterns::PatternDetector;
pub use phis::{PhiAnalyzer, place_pruned_phis};
pub use pointsto::{Constraint, PointsTo};
pub use range::ValueRange;
pub use recovery::{
    ClauseRejection, ProtectedRegions, Recovered, RejectedClause, condition_only_blocks,
    structure_ssa,
};
pub use resolver::ValueResolver;
pub use structure::{
    BlockSet, DEFAULT_MAX_DEPTH, HandlerFilter, HandlerKind, HandlerRegion, ProtectedHandler,
    ProtectedHandlerKind, ProtectedRegion, Region, StructureMetrics, StructureOptions, Structured,
    structure, structure_protected, structure_with,
};
pub use symbolic::{SymbolicEvaluator, SymbolicExpr, SymbolicOp};
pub use verifier::{SsaVerifier, VerifierError, VerifyLevel};
