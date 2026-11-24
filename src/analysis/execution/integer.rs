//! Partial integer values held by static execution.
//!
//! A value whose bits are only partly known, such as a field extracted from an
//! unknown word, is a [`PartialInteger`]: its known zero and one bits, its width
//! and whether it may derive from a stack address. Known bits spanning at most
//! 64 positions are held inline; wider patterns share immutable storage. Both
//! forms keep every known bit, so the representation never changes a result.

use std::{num::NonZeroU16, sync::Arc};

use crate::ir::integer::IntegerBits;

/// Partial scalar integer facts with their width and possible stack provenance.
///
/// Known positions spanning at most 64 bits use an inline window, including
/// windows in the high half of a 128-bit register. Other patterns use immutable
/// shared storage. Both representations preserve every known position and the
/// original width; sharing never changes integer semantics.
///
/// Create an execution value with [`crate::analysis::execution::Value::partial_integer`].
/// Signedness belongs to the register layout, not this bit pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialInteger(IntegerStorage);

/// An inline window or a shared immutable allocation.
///
/// Nonzero geometry leaves the zero encoding available for the shared case.
/// Width, window offset and stack provenance occupy a single scalar header.
#[derive(Debug, Clone, PartialEq, Eq)]
enum IntegerStorage {
    /// Positional masks in one 64-bit window, with unknown bits outside it.
    Window {
        /// Certainly clear positions relative to the window.
        zeros: u64,
        /// Certainly set positions relative to the window.
        ones: u64,
        /// One plus width-minus-one, offset and stack provenance bit fields.
        geometry: NonZeroU16,
    },
    /// Complete facts whose known positions span more than 64 bits.
    Shared {
        /// Immutable masks and width, cloned by sharing this allocation.
        bits: Arc<IntegerBits>,
        /// Whether unknown positions may contain private stack pointers.
        stack_derived: bool,
    },
}

impl PartialInteger {
    /// Preserves a complete pattern in an inline window or shared immutable storage.
    ///
    /// `stack_derived` records possible private-stack aliases independently of
    /// known positions. Returns `None` when geometry cannot be encoded.
    pub fn new(bits: IntegerBits, stack_derived: bool) -> Option<Self> {
        let known = bits.zeros() | bits.ones();
        let offset = if known == 0 {
            0
        } else {
            known.trailing_zeros()
        };
        if known.checked_shr(offset)? <= u128::from(u64::MAX) {
            let width = u16::try_from(bits.width().checked_sub(1)?).ok()?;
            let offset_field = u16::try_from(offset).ok()?.checked_shl(7)?;
            let stack = if stack_derived { 1u16 << 14 } else { 0 };
            let geometry = NonZeroU16::new((width | offset_field | stack).checked_add(1)?)?;
            return Some(Self(IntegerStorage::Window {
                zeros: u64::try_from(bits.zeros().checked_shr(offset)?).ok()?,
                ones: u64::try_from(bits.ones().checked_shr(offset)?).ok()?,
                geometry,
            }));
        }
        Some(Self(IntegerStorage::Shared {
            bits: Arc::new(bits),
            stack_derived,
        }))
    }

    /// Returns the complete known-bit domain without allocating or inspecting host types.
    ///
    /// Returns `None` if encoded geometry cannot establish a supported width or
    /// consistent masks. Public construction accepts only validated integer facts.
    #[must_use]
    pub fn bits(&self) -> Option<IntegerBits> {
        match &self.0 {
            IntegerStorage::Window {
                zeros,
                ones,
                geometry,
            } => {
                let header = geometry.get().checked_sub(1)?;
                let width = u32::from(header & 127).checked_add(1)?;
                let offset = u32::from((header >> 7) & 127);
                IntegerBits::new(
                    width,
                    u128::from(*zeros).checked_shl(offset)?,
                    u128::from(*ones).checked_shl(offset)?,
                )
            }
            IntegerStorage::Shared { bits, .. } => Some(**bits),
        }
    }

    /// Returns whether unknown positions may contain a private stack address.
    #[must_use]
    pub const fn may_reference_stack(&self) -> bool {
        match &self.0 {
            IntegerStorage::Window { geometry, .. } => {
                geometry.get().saturating_sub(1) & (1 << 14) != 0
            }
            IntegerStorage::Shared { stack_derived, .. } => *stack_derived,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    /// Every window position round-trips at every scalar width with both alias states.
    #[test]
    fn partial_integer_windows_preserve_complete_geometry() {
        for width in 1u32..=128 {
            for low in 0..width {
                for span in [1, 2, 32, 64] {
                    let Some(high) = low.checked_add(span).and_then(|end| end.checked_sub(1))
                    else {
                        continue;
                    };
                    if high >= width {
                        continue;
                    }
                    for stack in [false, true] {
                        let zeros = 1u128 << low;
                        let ones = if high == low { 0 } else { 1u128 << high };
                        let bits = IntegerBits::new(width, zeros, ones).unwrap();
                        let value = PartialInteger::new(bits, stack).unwrap();
                        assert!(matches!(value.0, IntegerStorage::Window { .. }));
                        assert_eq!(value.bits(), Some(bits));
                        assert_eq!(value.may_reference_stack(), stack);
                    }
                }
            }
        }
        for width in 1u32..=128 {
            let bits = IntegerBits::unknown(width).unwrap();
            assert_eq!(PartialInteger::new(bits, false).unwrap().bits(), Some(bits));
        }
        assert!(
            size_of::<PartialInteger>() <= 24,
            "partial integer bytes={}",
            size_of::<PartialInteger>()
        );
    }

    /// Wide patterns retain exact bits and share immutable storage when cloned.
    #[test]
    fn partial_integer_wide_patterns_share_storage() {
        let bits = IntegerBits::new(128, 1, 1u128 << 127).unwrap();
        let value = PartialInteger::new(bits, true).unwrap();
        assert!(matches!(value.0, IntegerStorage::Shared { .. }));
        assert_eq!(value.bits(), Some(bits));
        assert!(value.may_reference_stack());
        if let (
            IntegerStorage::Shared { bits: original, .. },
            IntegerStorage::Shared { bits: cloned, .. },
        ) = (&value.0, &value.clone().0)
        {
            assert!(Arc::ptr_eq(original, cloned));
        } else {
            panic!("wide facts retain shared ownership");
        }
    }
}
