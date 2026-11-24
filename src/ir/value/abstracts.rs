//! The abstract value of the constant lattice ([`AbstractValue`]).

use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    ir::{
        value::{ConstValue, computed::ComputedValue},
        variable::SsaVarId,
    },
    target::Target,
};

/// Abstract value for dataflow analysis.
///
/// This represents the abstract state of an SSA variable during analysis.
/// It forms a lattice where values can be refined as more information is gathered.
///
/// Generic over the host `Target` because the `Constant` variant carries a
/// `ConstValue<T>`.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::TypeRef: serde::Serialize, T::MethodRef: serde::Serialize, \
                     T::FieldRef: serde::Serialize, T::SymbolRef: serde::Serialize",
        deserialize = "T::TypeRef: serde::Deserialize<'de>, \
                       T::MethodRef: serde::Deserialize<'de>, \
                       T::FieldRef: serde::Deserialize<'de>, T::SymbolRef: serde::Deserialize<'de>"
    ))
)]
pub enum AbstractValue<T: Target> {
    /// No information yet (top of lattice).
    ///
    /// This is the initial state before any analysis.
    #[default]
    Top,

    /// Known constant value.
    Constant(ConstValue<T>),

    /// Known to be non-null (for reference types).
    NonNull,

    /// Value in a bounded range [min, max].
    Range {
        /// Minimum value (inclusive).
        min: i64,
        /// Maximum value (inclusive).
        max: i64,
    },

    /// Same value as another SSA variable.
    ///
    /// Used for copy propagation.
    SameAs(SsaVarId),

    /// Result of a specific computation (for CSE).
    Computed(ComputedValue),

    /// Multiple possible values (bottom of lattice for constants).
    ///
    /// This means the value cannot be determined at compile time.
    Bottom,
}

impl<T: Target> AbstractValue<T> {
    /// Returns `true` if this is the top element (no information).
    #[must_use]
    pub const fn is_top(&self) -> bool {
        matches!(self, Self::Top)
    }

    /// Returns `true` if this is the bottom element (conflicting info).
    #[must_use]
    pub const fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }

    /// Returns `true` if this is a known constant.
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns the constant value if this is a constant.
    #[must_use]
    pub const fn as_constant(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `true` if this value is known to be non-null.
    #[must_use]
    pub const fn is_non_null(&self) -> bool {
        matches!(self, Self::NonNull | Self::Constant(_))
    }

    /// Meet operation for the lattice (used at control flow joins).
    ///
    /// Returns the greatest lower bound of `self` and `other`.
    #[must_use]
    #[allow(clippy::match_same_arms)] // Arms kept separate for lattice documentation clarity
    pub fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            // Top meets anything yields the other
            (Self::Top, x) | (x, Self::Top) => x.clone(),

            // Bottom meets anything yields Bottom
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,

            // Same constants stay constant
            (Self::Constant(a), Self::Constant(b)) if a == b => Self::Constant(a.clone()),

            // Different constants become Bottom
            (Self::Constant(_), Self::Constant(_)) => Self::Bottom,

            // NonNull meets NonNull stays NonNull
            (Self::NonNull, Self::NonNull) => Self::NonNull,

            // NonNull meets Constant stays Constant (constants are non-null if not null)
            (Self::NonNull, Self::Constant(c)) | (Self::Constant(c), Self::NonNull) => {
                if c.is_null() {
                    Self::Bottom // null is not non-null
                } else {
                    Self::Constant(c.clone())
                }
            }

            // Ranges can be merged
            (
                Self::Range {
                    min: a_min,
                    max: a_max,
                },
                Self::Range {
                    min: b_min,
                    max: b_max,
                },
            ) => {
                let new_min = (*a_min).min(*b_min);
                let new_max = (*a_max).max(*b_max);
                Self::Range {
                    min: new_min,
                    max: new_max,
                }
            }

            // SameAs values must match
            (Self::SameAs(a), Self::SameAs(b)) if a == b => Self::SameAs(*a),

            // Computed values must match exactly
            (Self::Computed(a), Self::Computed(b)) if a == b => Self::Computed(a.clone()),

            // Otherwise, Bottom
            _ => Self::Bottom,
        }
    }

    /// Join operation for the lattice.
    ///
    /// Returns the least upper bound of `self` and `other`.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            // Bottom joins anything yields the other
            (Self::Bottom, x) | (x, Self::Bottom) => x.clone(),

            // Top joins anything yields Top
            (Self::Top, _) | (_, Self::Top) => Self::Top,

            // Same values stay the same
            (a, b) if a == b => a.clone(),

            // Otherwise, Top
            _ => Self::Top,
        }
    }
}

impl<T: Target> fmt::Display for AbstractValue<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
    T::SymbolRef: fmt::Display,
{
    /// Formats the lattice element: `⊤`/`⊥` for the extremal points, the wrapped
    /// constant or computation for the interior ones, `!null` for `NonNull`, and
    /// `[min..max]` for a range.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Top => write!(f, "⊤"),
            Self::Constant(c) => write!(f, "{c}"),
            Self::NonNull => write!(f, "!null"),
            Self::Range { min, max } => write!(f, "[{min}..{max}]"),
            Self::SameAs(v) => write!(f, "={v}"),
            Self::Computed(c) => write!(f, "{c}"),
            Self::Bottom => write!(f, "⊥"),
        }
    }
}
