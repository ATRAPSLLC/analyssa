//! Identities of single executions of opaque definitions.
//!
//! An unknown value produced by one execution of a definition is not the same
//! value as the definition's SSA variable, nor as another execution of it. A
//! [`SymbolicInstance`] names one such execution, so expressions over unknown
//! values never equate two runs that could differ.

use std::num::NonZeroU64;

use crate::ir::variable::SsaVarId;

/// One opaque execution of an SSA definition, scoped to its analysis context.
///
/// The source id supplies provenance and type information; it does not bind the
/// instance to that definition's current value. Generations distinguish repeated
/// executions without allocating variable names. A copied evaluator preserves
/// existing tokens and starts a separate continuation context for future ones.
/// Observations from independent contexts retain their context when compared.
/// Equality identifies a token within its producing context. Distinct tokens
/// can still denote equal concrete values.
///
/// Instances originate in [`crate::analysis::evaluator::SsaEvaluator`]. Ordinary
/// SSA bindings cannot resolve them; use
/// [`crate::analysis::symbolic::SymbolicExpr::evaluate_with`] with explicit
/// instance bindings when evaluating a concrete scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolicInstance {
    /// SSA definition whose execution produced the opaque value.
    source: SsaVarId,
    /// Nonzero generation allocated by the producing evaluator.
    generation: NonZeroU64,
}

impl SymbolicInstance {
    /// Creates a token after its evaluator has admitted a fresh generation.
    pub const fn new(source: SsaVarId, generation: NonZeroU64) -> Self {
        Self { source, generation }
    }

    /// Returns the definition that supplies this instance's provenance and type.
    #[must_use]
    pub const fn source(self) -> SsaVarId {
        self.source
    }

    /// Returns this instance's nonzero generation within its producing context.
    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation.get()
    }
}
