//! How many bytes a call's pointer argument is known to cover.
//!
//! The bytes reported behind a pointer argument stop where the call's stated
//! use of that argument ends, when the instruction or selected model proves
//! a length or access extent, and the observation says which proved it
//! ([`UseBound`]). [`CallLimits`] indexes the bounds of every argument of one
//! call once, before any argument is read.

use crate::{
    analysis::execution::{facts::UseBound, host::CallModel, value::Value},
    ir::{
        ops::{CallLocation, CallMemory, PointeeExtent, SsaOp},
        value::ConstValue,
    },
    pointer::PointerSize,
    target::Target,
};

/// Which actual body entries retain separate observation domains.
#[derive(Clone, Copy)]
pub enum CallSelection<'a> {
    /// Explicit positions in the original root body.
    Sites(&'a [(usize, usize)]),
    /// Every actual root call, within a retained-record bound.
    Roots(usize),
    /// Actual calls under the root and its retained structural descendants.
    Descendants(usize),
}

impl CallSelection<'_> {
    /// Returns whether this entry has an admitted observation ancestry.
    pub fn selects(self, root: bool, scoped: bool, explicit: bool) -> bool {
        match self {
            Self::Sites(_) => root && explicit,
            Self::Roots(_) => root,
            Self::Descendants(_) => root || scoped,
        }
    }

    /// Returns the owner-configured retained-record limit.
    pub const fn maximum(self) -> Option<usize> {
        match self {
            Self::Sites(_) => None,
            Self::Roots(maximum) | Self::Descendants(maximum) => Some(maximum),
        }
    }
}

/// The combined extent of the stated uses of one argument.
#[derive(Clone, Copy)]
enum Limit {
    /// The complete contract lists no pointer use for this argument.
    Unstated,
    /// All stated uses fit within this many bytes, as `by` states.
    Bytes(usize, UseBound),
    /// At least one stated use has no proven bound.
    Unbounded,
}

impl Limit {
    /// Unions a contract's uses without allowing a bounded record to hide
    /// an unbounded one.
    fn include(&mut self, bytes: Option<usize>) {
        *self = match (*self, bytes) {
            (Self::Unbounded, _) | (_, None) => Self::Unbounded,
            (Self::Unstated, Some(bytes)) => Self::Bytes(bytes, UseBound::Contract),
            (Self::Bytes(previous, by), Some(bytes)) => Self::Bytes(previous.max(bytes), by),
        };
    }
}

/// Optional use bounds, indexed once for all arguments of one call.
pub struct CallLimits {
    /// `None` preserves prefix reporting; contracts and model caps share one index.
    arguments: Option<Vec<Limit>>,
}

impl CallLimits {
    /// Preserves prefix reporting when no contract names this argument domain.
    pub const fn unstated() -> Self {
        Self { arguments: None }
    }

    /// Reads complete call-site contracts without cloning metadata or argument payloads.
    ///
    /// Callback-capable, unknown or non-pointee memory contracts retain ordinary
    /// prefix reporting. These limits describe pointer uses, never complete
    /// object sizes. An unknown count cannot establish a zero-length use.
    pub fn of<T: Target>(operation: &SsaOp<T>, arguments: &[Value<T>]) -> Self {
        let unknown = || Self { arguments: None };
        let effects = operation.stated_call_effects();
        let Some(effects) = effects.filter(|effects| !effects.calls_back) else {
            return unknown();
        };
        let CallMemory::Only(accesses) = &effects.memory else {
            return unknown();
        };
        if accesses.is_empty() {
            return Self {
                arguments: Some(Vec::new()),
            };
        }
        let mut limits = vec![Limit::Unstated; arguments.len()];
        for access in accesses {
            let CallLocation::Pointee { arg, extent } = access.location else {
                return unknown();
            };
            let Some(limit) = usize::try_from(arg)
                .ok()
                .and_then(|arg| limits.get_mut(arg))
            else {
                return unknown();
            };
            let bytes = match extent {
                PointeeExtent::Bytes(bytes) => Some(bytes),
                PointeeExtent::InArgument(index) => {
                    let Some(value) = usize::try_from(index)
                        .ok()
                        .and_then(|index| arguments.get(index))
                    else {
                        return unknown();
                    };
                    value.constant().and_then(ConstValue::as_u64)
                }
                PointeeExtent::Unbounded => None,
            };
            limit.include(bytes.map(|bytes| usize::try_from(bytes).unwrap_or(usize::MAX)));
        }
        Self {
            arguments: Some(limits),
        }
    }

    /// Intersects selected-model use bounds with any complete instruction contract.
    ///
    /// Each model answer bounds only its named argument. Missing answers retain
    /// the prior prefix policy and never exclude other arguments. A model
    /// contract supplies no memory-effect or object-size evidence.
    pub fn with_model<T: Target>(
        mut self,
        model: &dyn CallModel<T>,
        arguments: &[Value<T>],
        ptr_size: PointerSize,
    ) -> Self {
        for argument in 0..arguments.len() {
            let Some(bytes) = model.observation_bound(argument, arguments, ptr_size) else {
                continue;
            };
            let limits = self
                .arguments
                .get_or_insert_with(|| vec![Limit::Unbounded; arguments.len()]);
            if let Some(limit) = limits.get_mut(argument) {
                *limit = match *limit {
                    Limit::Unbounded => Limit::Bytes(bytes, UseBound::Model),
                    Limit::Bytes(previous, _) if bytes < previous => {
                        Limit::Bytes(bytes, UseBound::Model)
                    }
                    Limit::Bytes(previous, by) => Limit::Bytes(previous, by),
                    Limit::Unstated => Limit::Unstated,
                };
            }
        }
        self
    }

    /// Returns whether a complete contract states that the call touches no memory.
    pub fn no_memory(&self) -> bool {
        matches!(self.arguments.as_deref(), Some([]))
    }

    /// Restricts a known prefix to its proven use bound, when one is available.
    pub fn prefix(&self, argument: usize, maximum: usize) -> usize {
        let Some(arguments) = &self.arguments else {
            return maximum;
        };
        match arguments.get(argument) {
            Some(Limit::Bytes(bytes, _)) => (*bytes).min(maximum),
            Some(Limit::Unbounded) => maximum,
            Some(Limit::Unstated) | None => 0,
        }
    }

    /// The bound a stated use puts on `argument` and what states it, where
    /// one does.
    pub fn use_bound(&self, argument: usize) -> Option<(u64, UseBound)> {
        match self.arguments.as_ref()?.get(argument)? {
            Limit::Bytes(bytes, by) => Some((u64::try_from(*bytes).unwrap_or(u64::MAX), *by)),
            Limit::Unbounded | Limit::Unstated => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            ops::{AccessMode, CallEffects},
            variable::SsaVarId,
        },
        testing::MockTarget,
    };

    /// A call stating `count` reads of four bytes through its first argument.
    fn operation(count: usize) -> SsaOp<MockTarget> {
        let mut effects = CallEffects::pure();
        for _ in 0..count {
            effects = effects.with_access(
                CallLocation::Pointee {
                    arg: 0,
                    extent: PointeeExtent::Bytes(4),
                },
                AccessMode::Read,
            );
        }
        SsaOp::Call(Box::new(crate::ir::CallData {
            dest: None,
            callee: crate::ir::Callee::Direct(1),
            args: vec![SsaVarId::from_index(0)],
            effects: Some(effects),
            tail: false,
            implicit_inputs: Default::default(),
            results: Default::default(),
            staged: Vec::new(),
        }))
    }

    /// A stated four-byte read bounds the prefix reported for its argument,
    /// and an argument no use names reports none.
    #[test]
    fn a_stated_use_bounds_its_arguments_prefix() {
        let limits = CallLimits::of(&operation(2), &[Value::Const(ConstValue::U64(0))]);
        assert!(!limits.no_memory());
        assert_eq!(limits.prefix(0, 64), 4);
        assert_eq!(limits.prefix(1, 64), 0);
        assert_eq!(limits.prefix(0, 2), 2);
    }
}
