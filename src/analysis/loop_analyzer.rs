//! Loop analyzer for computing comprehensive loop information from SSA.
//!
//! This module provides the [`LoopAnalyzer`] which computes full
//! [`crate::analysis::loops::LoopInfo`] structures from an SSA function,
//! including preheaders, latches, exits, and loop type classification.
//!
//! # Architecture
//!
//! `LoopAnalyzer` is a thin convenience wrapper around the generic `detect_loops`
//! function from the `loops` module:
//!
//! 1. Constructs an `SsaCfg` from the SSA function
//! 2. Computes dominators using `algorithms::compute_dominators`
//! 3. Delegates to `detect_loops` for full loop analysis
//!
//! The separation between `LoopAnalyzer` and `detect_loops` allows the generic
//! loop detection to be used with non-SSA graph types (e.g., CIL CFGs, x86 CFGs)
//! while `LoopAnalyzer` provides a convenient SSA-specific interface.
//!
//! # Complexity
//!
//! Analysis: O(B^2) where B is the number of blocks (dominator computation
//! dominates the runtime). Loop detection is O(E * L) where E is edges and
//! L is the number of loops found.

use crate::{
    analysis::{
        exceptions::EhCfg,
        loops::{LoopForest, detect_loops},
    },
    graph::{NodeId, algorithms},
    ir::function::SsaFunction,
    target::Target,
};

/// Analyzes loops in an SSA function.
///
/// The analyzer computes:
/// - Natural loops using dominance-based back edge detection
/// - Preheader identification for each loop
/// - Latch (back edge source) identification
/// - Exit edge detection
/// - Loop type classification
/// - Loop nesting relationships
///
/// This is a thin wrapper around the generic `detect_loops` function,
/// providing a convenient SSA-specific interface.
pub struct LoopAnalyzer<'a, T: Target> {
    /// The exception-aware flow view used for dominance and back-edge detection.
    ///
    /// A back edge is one whose target dominates its source, and
    /// `DominatorTree::dominates` answers `false` whenever either endpoint is
    /// unreachable. Under a terminator-only graph both endpoints of a loop
    /// inside a handler are unreachable, so such a loop is invisible: no
    /// `LoopInfo` is created and every loop pass silently declines to act on
    /// it. Rooting here makes handler loops ordinary loops.
    eh: EhCfg<'a, T>,
}

impl<'a, T: Target> LoopAnalyzer<'a, T> {
    /// Creates a new loop analyzer for the given SSA function.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>) -> Self {
        Self {
            eh: EhCfg::from_ssa(ssa),
        }
    }

    /// Analyzes all loops and returns a [`LoopForest`].
    ///
    /// Uses the shared `detect_loops` function which implements dominance-based
    /// back edge detection and computes preheaders, exits, loop types, and nesting.
    #[must_use]
    pub fn analyze(&self) -> LoopForest {
        let dominators = algorithms::compute_dominators(&self.eh, NodeId::new(0));
        detect_loops(&self.eh, &dominators)
    }
}

/// Extension trait for SSA functions to easily access loop analysis.
pub trait SsaLoopAnalysis {
    /// Analyzes loops in this function.
    fn analyze_loops(&self) -> LoopForest;
}

impl<T: Target> SsaLoopAnalysis for SsaFunction<T> {
    fn analyze_loops(&self) -> LoopForest {
        LoopAnalyzer::new(self).analyze()
    }
}
