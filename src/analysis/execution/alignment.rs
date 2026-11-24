//! Explicit low-bit stack contracts without changing physical address identities.

use std::{fmt, num::NonZeroU8, ops::Range};

use crate::{
    analysis::execution::{Address, Root},
    ir::integer::{IntegerBits, IntegerType},
};

/// Storage an activation reserves below the base it realigned its stack
/// pointer to, where nothing states the pointer's low bits
/// ([`Root::StackAligned`]): the frame `and sp, -2^k` then `sub sp, n` make.
///
/// It names one realignment, the mask of the stack pointer standing `at`
/// relative to entry, and holds for the addresses under that realigned base
/// alone. It promises what a stack extent does
/// ([`ExecutionHost::stack_extent`](crate::analysis::execution::ExecutionHost::stack_extent)):
/// backing and ordinary write permission until the activation returns.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RealignedExtent {
    /// Where the stack pointer the mask realigned stood, relative to entry.
    pub at: i64,
    /// The writable bytes, relative to the realigned base.
    pub range: Range<i64>,
}

/// Known low bits of a physical stack pointer or stack object's base.
///
/// A value with `k` known bits and residue `r` satisfies `pointer mod 2^k = r`.
/// This supplies no absolute high bits, allocation extent, lifetime, readability
/// or writability. A host explicitly declares an entry-pointer contract; the
/// executor translates it into the existing root's coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StackCongruence {
    /// Nonzero number of low bits described by the contract, at most 128.
    bits: NonZeroU8,
    /// Known low-bit pattern in the physical address space.
    residue: u128,
}

impl StackCongruence {
    /// Creates a nonempty low-bit contract with a representable residue.
    ///
    /// Returns `None` for zero or more than 128 bits, or a residue outside the
    /// declared modulus. The execution additionally checks its pointer width.
    #[must_use]
    pub fn from_low_bits(bits: u32, residue: u128) -> Option<Self> {
        let bits = NonZeroU8::new(u8::try_from(bits).ok()?)?;
        let mask = IntegerType::new(u32::from(bits.get()), false).map(IntegerType::mask)?;
        (residue & !mask == 0).then_some(Self { bits, residue })
    }

    /// Returns the number of physical low bits the contract fixes.
    #[must_use]
    pub const fn known_bits(self) -> u32 {
        self.bits.get() as u32
    }

    /// Returns the known physical low-bit pattern.
    #[must_use]
    pub const fn residue(self) -> u128 {
        self.residue
    }

    /// Translates an entry-pointer contract to the same concrete stack root's base.
    pub fn at_root(self, entry: Address) -> Option<Self> {
        if !matches!(entry.root(), Root::Stack | Root::StackFrame(_))
            || self.known_bits() > entry.pointer_size().bits()
        {
            return None;
        }
        let offset = i128::from(entry.offset()).cast_unsigned();
        Self::from_low_bits(
            self.known_bits(),
            self.residue.wrapping_sub(offset)
                & IntegerType::new(self.known_bits(), false).map(IntegerType::mask)?,
        )
    }

    /// Retains exactly the low bits common to two alternatives for the same root.
    pub fn meet(self, other: Self) -> Option<Self> {
        let bits = self
            .known_bits()
            .min(other.known_bits())
            .min((self.residue ^ other.residue).trailing_zeros());
        Self::from_low_bits(
            bits,
            self.residue & IntegerType::new(bits, false).map(IntegerType::mask)?,
        )
    }

    /// Returns the low bits of `address` this contract fixes, read at `width`
    /// bits: the root base's residue moved by the address's offset, every
    /// higher bit unknown.
    ///
    /// `None` for an address of another root, or a width the address is
    /// narrower than.
    pub fn low_bits(self, address: Address, width: u32) -> Option<IntegerBits> {
        if !matches!(address.root(), Root::Stack | Root::StackFrame(_))
            || width > address.pointer_size().bits()
        {
            return None;
        }
        let mask = IntegerType::new(self.known_bits().min(width), false).map(IntegerType::mask)?;
        let ones = self
            .residue
            .wrapping_add(i128::from(address.offset()).cast_unsigned())
            & mask;
        IntegerBits::new(width, mask & !ones, ones)
    }

    /// Clears justified physical low bits while retaining the same root-relative identity.
    pub fn masked(self, address: Address, mask: u128) -> Option<Address> {
        if !matches!(address.root(), Root::Stack | Root::StackFrame(_))
            || self.known_bits() > address.pointer_size().bits()
        {
            return None;
        }
        let target = address.pointer_size().mask_unsigned_128(u128::MAX);
        if mask == 0 || mask & !target != 0 {
            return None;
        }
        let cleared = target ^ mask;
        let modulus = cleared.checked_add(1)?;
        if !modulus.is_power_of_two() || modulus.trailing_zeros() > self.known_bits() {
            return None;
        }
        let removed = self
            .residue
            .wrapping_add(i128::from(address.offset()).cast_unsigned())
            & cleared;
        address.displaced_bits(removed.wrapping_neg(), address.pointer_size())
    }
}

/// Looks up an already established contract for a live concrete stack root.
pub trait StackCongruences {
    /// Returns root-base low bits without making a new entry assumption.
    fn stack_congruence(&self, root: Root) -> Option<StackCongruence>;
}

/// Borrows path-local stack evidence for one pure scalar evaluation.
#[derive(Clone, Copy)]
pub struct CongruenceLookup<'a>(
    /// The live path whose stack roots supply the evidence.
    pub &'a dyn StackCongruences,
);

impl CongruenceLookup<'_> {
    /// Returns an address's low bits only under existing low-bit evidence for its root.
    pub fn low_bits(self, address: Address, width: u32) -> Option<IntegerBits> {
        self.0
            .stack_congruence(address.root())?
            .low_bits(address, width)
    }

    /// Returns a root-relative masking result only under existing low-bit evidence.
    pub fn masked(self, address: Address, mask: u128) -> Option<Address> {
        self.0
            .stack_congruence(address.root())?
            .masked(address, mask)
    }
}

impl fmt::Debug for CongruenceLookup<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StackCongruences")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointer::PointerSize;

    /// Enumerates actual physical bases independently of the root-relative masking formula.
    #[test]
    fn masked_stack_coordinates_match_compatible_concrete_bases() {
        for bits in 1..=8u32 {
            let modulus = 1u128.checked_shl(bits).unwrap();
            let fact_mask = modulus.checked_sub(1).unwrap();
            for residue in 0..modulus {
                let proof = StackCongruence::from_low_bits(bits, residue).unwrap();
                for offset in [
                    -257i64, -129, -33, -16, -9, -8, -1, 0, 1, 8, 15, 16, 31, 127, 255,
                ] {
                    let address = Address::new(Root::Stack, offset, PointerSize::Bit16);
                    for cleared in 0..=8 {
                        let low_mask = 1u128.checked_shl(cleared).unwrap().checked_sub(1).unwrap();
                        let mask = 0xffff ^ low_mask;
                        let result = proof.masked(address, mask);
                        if cleared > bits {
                            assert!(result.is_none());
                            continue;
                        }
                        let result = result.unwrap();
                        assert_eq!(result.root(), Root::Stack);
                        for high in [0, 0x100, 0xff00] {
                            let base = high | residue;
                            assert_eq!(base & fact_mask, residue);
                            let original =
                                base.wrapping_add(i128::from(offset).cast_unsigned()) & 0xffff;
                            let actual = base
                                .wrapping_add(i128::from(result.offset()).cast_unsigned())
                                & 0xffff;
                            assert_eq!(actual, original & mask);
                        }
                    }
                }
            }
        }
    }

    /// Joining loses divergent low bits and never promotes a weaker fact.
    #[test]
    fn congruence_meet_and_entry_mapping_preserve_only_shared_bits() {
        let a = StackCongruence::from_low_bits(4, 8).unwrap();
        assert_eq!(a.meet(a), Some(a));
        assert_eq!(
            a.meet(StackCongruence::from_low_bits(3, 0).unwrap()),
            StackCongruence::from_low_bits(3, 0)
        );
        assert_eq!(
            a.meet(StackCongruence::from_low_bits(4, 12).unwrap()),
            StackCongruence::from_low_bits(2, 0)
        );
        assert_eq!(a.meet(StackCongruence::from_low_bits(4, 9).unwrap()), None);
        let entry = Address::new(Root::Stack, -24, PointerSize::Bit64);
        let base = a.at_root(entry).unwrap();
        assert_eq!(base, StackCongruence::from_low_bits(4, 0).unwrap());
        assert!(StackCongruence::from_low_bits(0, 0).is_none());
        assert!(StackCongruence::from_low_bits(129, 0).is_none());
        assert!(StackCongruence::from_low_bits(4, 16).is_none());
        assert!(a.at_root(Address::of_root(Root::Number)).is_none());
        assert!(
            StackCongruence::from_low_bits(16, 0)
                .unwrap()
                .at_root(Address::new(Root::Stack, 0, PointerSize::Bit8))
                .is_none()
        );
    }
    /// Null-producing and nonalignment masks cannot manufacture a private stack identity.
    #[test]
    fn masks_reject_zero_arbitrary_roots_and_unrepresentable_wide_offsets() {
        let proof = StackCongruence::from_low_bits(4, 8).unwrap();
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let address = Address::new(Root::Stack, -8, width);
            let target = width.mask_unsigned_128(u128::MAX);
            assert!(proof.masked(address, 0).is_none());
            assert!(proof.masked(address, target ^ 10).is_none());
            assert!(
                proof
                    .masked(Address::new(Root::Number, 0, width), target ^ 15)
                    .is_none()
            );
        }
        let proof = StackCongruence::from_low_bits(4, 1).unwrap();
        assert!(
            proof
                .masked(
                    Address::new(Root::Stack, i64::MIN, PointerSize::Bit128),
                    u128::MAX ^ 15
                )
                .is_none()
        );
    }
}
