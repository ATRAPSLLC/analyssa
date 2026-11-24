//! Built-in SSA optimization passes for the analyssa deobfuscation framework.
//!
//! Each pass is a struct implementing [`FunctionPass`] (or [`ModulePass`] for
//! a whole-program one), defined in the module that holds its body. A host
//! runs one directly, `pass.run(&mut ssa, &env)` under a [`PassEnv`], or
//! registers it with [`PassScheduler`].
//!
//! # Pass Categories
//!
//! - **Normalization**: [`deadcode`], [`copying`], [`gvn`], [`blockmerge`] -
//!   clean up the IR after structural transformations. Run between every
//!   pipeline layer's fixpoint iterations.
//! - **Structural**: [`controlflow`], [`threading`], [`loopcanon`], [`blockmerge`] -
//!   simplify and canonicalize the control-flow graph; [`unthread`] merges a
//!   block a compiler duplicated back into the block it copies.
//! - **Value-level**: [`algebraic`], [`strength`], [`reassociate`], [`ranges`],
//!   [`predicates`], [`licm`], [`calltarget`] - replace expensive or redundant
//!   computations with cheaper or constant equivalents; [`passthrough`] drops
//!   the old value of a vector operation no reader observes, and [`demand`]
//!   the operand of a bitwise operation no reader demands the bits of.
//!
//! # Default Layer Conventions
//!
//! The fallback layer numbers follow the deobfuscation-pipeline convention:
//!
//! - **Layer 0**: Structural passes (control flow simplification, jump
//!   threading, loop canonicalization, block merging).
//! - **Layer 1**: Value-level passes (algebraic simplification, strength
//!   reduction, reassociation, value range propagation, opaque predicates,
//!   LICM).
//!
//! Normalize passes (DCE, GVN, copy propagation) are registered via
//! [`PassScheduler::add_normalize`] and run between every layer's fixpoint
//! iterations (their layer number is irrelevant).
//!
//! # Pass Inventory
//!
//! | Struct | Layer | Module | Scope |
//! |--------|-------|--------|-------|
//! | [`AlgebraicSimplificationPass`] | 1 | [`algebraic`] | InstructionsOnly |
//! | [`BlockMergingPass`] | 0 | [`blockmerge`] | CfgModifying |
//! | [`ConstantCallTargetPass`] | 1 | [`calltarget`] | InstructionsOnly |
//! | [`ControlFlowSimplificationPass`] | 0 | [`controlflow`] | CfgModifying |
//! | [`CopyPropagationPass`] | normalize | [`copying`] | InstructionsOnly |
//! | [`DeadCodeEliminationPass`] | normalize | [`deadcode`] | CfgModifying |
//! | [`DeadMethodEliminationPass`] | global | [`deadcode`] | global |
//! | [`DemandedBitsPass`] | 1 | [`demand`] | InstructionsOnly |
//! | [`DuplicateMergingPass`] | 0 | [`unthread`] | CfgModifying |
//! | [`EqualityPropagationPass`] | 1 | [`equalities`] | InstructionsOnly |
//! | [`IdleLoopPass`] | 0 | [`idle`] | CfgModifying |
//! | [`LoopRejoiningPass`] | 0 | [`rejoin`] | CfgModifying |
//! | [`ArmMergingPass`] | 0 | [`arms`] | CfgModifying |
//! | [`MootTestPass`] | 0 | [`moot`] | CfgModifying |
//! | [`GlobalValueNumberingPass`] | normalize | [`gvn`] | InstructionsOnly |
//! | [`JumpThreadingPass`] | 0 | [`threading`] | CfgModifying |
//! | [`LicmPass`] | 1 | [`licm`] | CfgModifying |
//! | [`LoopCanonicalizationPass`] | 0 | [`loopcanon`] | CfgModifying |
//! | [`MemoryOptimizationPass`] | 1 | [`memory`] | InstructionsOnly |
//! | [`OpaquePredicatePass<T>`] | 1 | [`predicates`] | CfgModifying |
//! | [`ReassociationPass`] | 1 | [`reassociate`] | InstructionsOnly |
//! | [`StrengthReductionPass`] | 1 | [`strength`] | InstructionsOnly |
//! | [`SwitchBoundsPass`] | 0 | [`switchbounds`] | CfgModifying |
//! | [`UnreadPassthroughPass`] | 1 | [`passthrough`] | InstructionsOnly |
//! | [`ValueRangePropagationPass`] | 1 | [`ranges`] | CfgModifying |
//!
//! [`FunctionPass`]: crate::scheduling::FunctionPass
//! [`ModulePass`]: crate::scheduling::ModulePass
//! [`PassEnv`]: crate::scheduling::PassEnv
//! [`PassScheduler`]: crate::scheduling::PassScheduler
//! [`PassScheduler::add_normalize`]: crate::scheduling::PassScheduler::add_normalize

pub mod algebraic;
pub mod arms;
pub mod blockmerge;
pub mod calltarget;
pub mod controlflow;
pub mod copying;
pub mod deadcode;
mod decide;
pub mod demand;
pub mod equalities;
mod guarded;
pub mod gvn;
pub mod idle;
pub mod licm;
pub mod loopcanon;
pub mod memory;
pub mod moot;
pub mod passthrough;
pub mod predicates;
pub mod ranges;
pub mod reassociate;
mod redirect;
pub mod rejoin;
pub mod residual;
pub mod strength;
pub mod switchbounds;
pub mod threading;
pub mod unthread;
pub mod utils;

pub use algebraic::AlgebraicSimplificationPass;
pub use arms::ArmMergingPass;
pub use blockmerge::BlockMergingPass;
pub use calltarget::ConstantCallTargetPass;
pub use controlflow::ControlFlowSimplificationPass;
pub use copying::CopyPropagationPass;
pub use deadcode::{DeadCodeEliminationPass, DeadMethodEliminationPass};
pub use demand::DemandedBitsPass;
pub use equalities::EqualityPropagationPass;
pub use gvn::GlobalValueNumberingPass;
pub use idle::IdleLoopPass;
pub use licm::LicmPass;
pub use loopcanon::LoopCanonicalizationPass;
pub use memory::MemoryOptimizationPass;
pub use moot::MootTestPass;
pub use passthrough::UnreadPassthroughPass;
pub use predicates::{OpaquePredicatePass, PredicateResult};
pub use ranges::ValueRangePropagationPass;
pub use reassociate::ReassociationPass;
pub use rejoin::LoopRejoiningPass;
pub use strength::StrengthReductionPass;
pub use switchbounds::SwitchBoundsPass;
pub use threading::JumpThreadingPass;
pub use unthread::DuplicateMergingPass;
