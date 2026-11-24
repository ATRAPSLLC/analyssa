//! Target-agnostic SSA (Static Single Assignment) intermediate representation core.
//!
//! This module defines the generic SSA IR types used by RASSA. The IR is parameterized
//! over a [`crate::Target`] trait, allowing different host instruction sets (CIL, etc.)
//! to reuse the same SSA infrastructure. Hosts extend these types through their own
//! `Target` implementation and host-specific extension traits or inherent impls.
//!
//! # Architecture
//!
//! The SSA IR is organized into several sub-modules, each handling a specific aspect:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`block`] | Basic blocks containing phi nodes and instructions |
//! | [`instruction`] | SSA instruction wrapper with def/use metadata |
//! | [`ops`] | Decomposed SSA operations (`SsaOp` enum with all op variants) |
//! | [`phi`] | Phi nodes for merging values at control flow joins |
//! | [`value`] | Constant values, abstract value lattice, and CSE tracking |
//! | [`variable`] | Variable IDs, origins, def/use sites, and per-function allocators |
//! | [`exception`] | Exception handler preservation through SSA transformations |
//! | [`function`] | Complete SSA function representation with rebuild/repair/canonicalize |
//!
//! # SSA Construction Pipeline
//!
//! 1. **CIL to SSA**: The `SsaConverter` simulates the stack, creates explicit variables,
//!    places phi nodes at dominance frontiers, and renames to achieve single-assignment.
//! 2. **Optimization**: Analysis passes operate on the SSA form (constant propagation,
//!    copy propagation, DCE, GVN, etc.).
//! 3. **Repair**: After passes, `repair_ssa()` handles instruction-only changes; a
//!    pass that modifies the CFG reconstructs the values it disturbed with
//!    `reconstruct_values()`.
//! 4. **Canonicalize**: Final cleanup (strip nops, remove empty blocks, compact indices)
//!    before code generation.
//!
//! # Key Design Decisions
//!
//! - **Target genericity**: All types are parameterized on `T: Target`, allowing
//!   host-specific metadata (types, methods, fields) without losing SSA structure.
//! - **Dense variable IDs**: `SsaVarId` indices are dense per-function (0, 1, 2, ...)
//!   enabling O(1) variable lookup via direct vector indexing.
//! - **Explicit def/use**: Every SSA operation tracks its destination and operands,
//!   enabling direct def-use chain construction without scanning.
//! - **Thread safety**: All IR types implement `Send` and `Sync`.

mod aes;
pub mod block;
/// Canonical visitation of operation content.
pub mod content;
pub mod division;
pub mod evaluation;
pub mod exception;
pub mod float;
pub mod function;
mod ghash;
mod identity;
pub mod instruction;
pub mod integer;
pub mod lanes;
pub mod ops;
pub mod phi;
mod sha;
mod sm3;
mod sm4;
pub mod value;
pub mod variable;
pub mod varstore;

pub use block::{BlockProvenance, ReplaceResult, SsaBlock};
pub use evaluation::{ArithmeticTrap, Evaluation};
pub use exception::{
    BlockRange, ClauseLayout, ClausePart, ExceptionBlocks, ExceptionTableError, HandlerKind,
    LaidOutHandler, SsaExceptionHandler,
};
pub use function::{
    CheckedReplaceResult, FunctionKind, MethodPurity, ReplacementSkipReason, ReturnInfo,
    SkippedReplacement, SsaBlockBuilder, SsaDefSpec, SsaEditOptions, SsaEditReport, SsaEditScope,
    SsaEditor, SsaFunction, SsaFunctionBuilder, SsaRollbackPolicy, TrivialPhiOptions,
    VariableRemap,
};
pub use instruction::{InstrId, InstructionRole, SsaInstruction};
pub use integer::IntegerType;
pub use lanes::VectorConst;
pub use phi::{OperandReplacement, PhiEdge, PhiNode, PhiOperand};
pub use value::{AbstractValue, ComputedOp, ComputedValue, ConstValue};
pub use variable::{
    DefSite, FunctionVarAllocator, SsaVarId, SsaVariable, UseSite, VariableOrigin, VariableRole,
};
pub use varstore::{SharedVarMap, VarMap, VarMapIntoIter, VarMapIter, VarSet};

pub use self::{
    float::{
        DefaultNan, FloatBinary, FloatBits, FloatClassLayout, FloatComparison,
        FloatExceptionLayout, FloatExceptions, FloatFormat, FloatIntegerResult, FloatResult,
        FloatRounding, FloatRules, NanEncoding, NanOperands, NanPropagation, OutOfRangeConversion,
        RaisingArithmetic,
    },
    ops::{
        AccessMode, BinaryOpInfo, BinaryOpKind, CallAccess, CallData, CallEffects, CallInput,
        CallInputKey, CallLocation, CallMemory, CallOutput, CallResult, CallTermination, Callee,
        Capture, CmpKind, FlagCondition, FlagsMask, FrameOrigin, PointeeExtent, ReturnTransfer,
        RuntimeValueKind, SsaOp, StagedValue, UnaryOpInfo, UnaryOpKind,
    },
};
