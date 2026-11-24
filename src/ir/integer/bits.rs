//! Known-bit algebra for fixed-width integers.
//!
//! [`IntegerBits`] holds which bits of a 1- to 128-bit integer are known zero
//! and known one; every other bit may be either. Its operations (bitwise logic,
//! shifts, conversions, joins) are exact transfer functions: a bit is known in
//! the result only when every pair of inputs consistent with the operands'
//! known bits produces it.

use crate::ir::{integer::IntegerType, ops::CmpKind};

/// Known zero and one bits of a scalar integer from one through 128 bits.
///
/// Bits in neither set may be zero or one. Limbs avoid 128-bit alignment in
/// execution frames, retaining inline values without an allocation per result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntegerBits {
    /// Certainly clear bits, low limb first.
    zeros: [u64; 2],
    /// Certainly set bits, low limb first.
    ones: [u64; 2],
    /// Validated width from one through 128 bits.
    width: u8,
}

impl IntegerBits {
    /// Creates consistent facts within a supported width.
    ///
    /// Returns `None` for an unsupported width, conflicting facts or bits
    /// outside that width. No bit may be both known zero and known one.
    #[must_use]
    pub const fn new(width: u32, zeros: u128, ones: u128) -> Option<Self> {
        let all = match IntegerType::new(width, false) {
            Some(integer) => integer.mask(),
            None => return None,
        };
        if zeros & ones != 0 || (zeros | ones) & !all != 0 {
            return None;
        }
        Some(Self::parts(width, zeros, ones))
    }

    /// Creates an unknown scalar integer at a supported width.
    #[must_use]
    pub const fn unknown(width: u32) -> Option<Self> {
        Self::new(width, 0, 0)
    }

    /// Creates a constant, truncating bits outside the supported width.
    #[must_use]
    pub const fn constant(width: u32, pattern: u128) -> Option<Self> {
        let all = match IntegerType::new(width, false) {
            Some(integer) => integer.mask(),
            None => return None,
        };
        Some(Self::parts(width, !pattern & all, pattern & all))
    }

    /// Packs facts whose width and consistency the caller already established.
    const fn parts(width: u32, zeros: u128, ones: u128) -> Self {
        Self {
            zeros: [zeros as u64, (zeros >> 64) as u64],
            ones: [ones as u64, (ones >> 64) as u64],
            width: width as u8,
        }
    }

    /// Returns the validated bit width.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width as u32
    }

    /// Returns the mask of every bit within this width.
    #[must_use]
    pub const fn mask(self) -> u128 {
        match IntegerType::new(self.width(), false) {
            Some(integer) => integer.mask(),
            None => 0,
        }
    }

    /// Returns the certainly clear bits.
    #[must_use]
    pub const fn zeros(self) -> u128 {
        self.zeros[0] as u128 | (self.zeros[1] as u128) << 64
    }

    /// Returns the certainly set bits.
    #[must_use]
    pub const fn ones(self) -> u128 {
        self.ones[0] as u128 | (self.ones[1] as u128) << 64
    }

    /// Returns a pattern only when every bit is known.
    #[must_use]
    pub const fn value(self) -> Option<u128> {
        if self.zeros() | self.ones() == self.mask() {
            Some(self.ones())
        } else {
            None
        }
    }

    /// Returns whether no bit is known.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.zeros() | self.ones() == 0
    }

    /// Returns the bits that may be set.
    #[must_use]
    pub const fn possibly_set(self) -> u128 {
        !self.zeros() & self.mask()
    }

    /// Returns the sign bit where it is known.
    #[must_use]
    pub const fn sign(self) -> Option<bool> {
        let sign = match 1u128.checked_shl(self.width().saturating_sub(1)) {
            Some(sign) => sign,
            None => return None,
        };
        if self.ones() & sign != 0 {
            Some(true)
        } else if self.zeros() & sign != 0 {
            Some(false)
        } else {
            None
        }
    }

    /// Computes bitwise AND, refusing operands of different widths.
    #[must_use]
    pub const fn and(self, other: Self) -> Option<Self> {
        if self.width != other.width {
            return None;
        }
        Some(Self::parts(
            self.width(),
            self.zeros() | other.zeros(),
            self.ones() & other.ones(),
        ))
    }

    /// Computes bitwise OR, refusing operands of different widths.
    #[must_use]
    pub const fn or(self, other: Self) -> Option<Self> {
        if self.width != other.width {
            return None;
        }
        Some(Self::parts(
            self.width(),
            self.zeros() & other.zeros(),
            self.ones() | other.ones(),
        ))
    }

    /// Computes bitwise XOR, refusing operands of different widths.
    #[must_use]
    pub const fn xor(self, other: Self) -> Option<Self> {
        if self.width != other.width {
            return None;
        }
        Some(Self::parts(
            self.width(),
            (self.zeros() & other.zeros()) | (self.ones() & other.ones()),
            (self.ones() & other.zeros()) | (self.zeros() & other.ones()),
        ))
    }

    /// Complements each known bit.
    #[must_use]
    pub const fn inverted(self) -> Self {
        Self::parts(self.width(), self.ones(), self.zeros())
    }

    /// Keeps facts common to two alternatives of the same width.
    #[must_use]
    pub const fn join(self, other: Self) -> Option<Self> {
        if self.width != other.width {
            return None;
        }
        Some(Self::parts(
            self.width(),
            self.zeros() & other.zeros(),
            self.ones() & other.ones(),
        ))
    }

    /// Keeps the facts of both of two descriptions of one value: the value
    /// both hold. `None` for different widths, or for descriptions no one
    /// value satisfies.
    #[must_use]
    pub const fn meet(self, other: Self) -> Option<Self> {
        if self.width != other.width {
            return None;
        }
        Self::new(
            self.width(),
            self.zeros() | other.zeros(),
            self.ones() | other.ones(),
        )
    }

    /// Whether `self cmp other` holds for every pair of values the two
    /// descriptions admit: `Some(true)` when it holds for all of them,
    /// `Some(false)` when for none, `None` when the bits do not settle it or
    /// the widths differ.
    ///
    /// An equality is settled by a bit known set in one and clear in the
    /// other, or by both values known whole; an ordering by the ranges the
    /// known bits bound, compared unsigned or, when `unsigned` is false,
    /// two's complement.
    #[must_use]
    pub fn compare(self, cmp: CmpKind, unsigned: bool, other: Self) -> Option<bool> {
        if self.width != other.width {
            return None;
        }
        let equal = if (self.ones() & other.zeros()) | (self.zeros() & other.ones()) != 0 {
            Some(false)
        } else {
            match (self.value(), other.value()) {
                (Some(left), Some(right)) => Some(left == right),
                _ => None,
            }
        };
        let (left, right) = (self.ordered(unsigned), other.ordered(unsigned));
        let (low, high) = (left.ones(), left.possibly_set());
        let (other_low, other_high) = (right.ones(), right.possibly_set());
        match cmp {
            CmpKind::Eq => equal,
            CmpKind::Ne => equal.map(|equal| !equal),
            CmpKind::Lt if high < other_low => Some(true),
            CmpKind::Lt if low >= other_high => Some(false),
            CmpKind::Le if high <= other_low => Some(true),
            CmpKind::Le if low > other_high => Some(false),
            CmpKind::Gt if low > other_high => Some(true),
            CmpKind::Gt if high <= other_low => Some(false),
            CmpKind::Ge if low >= other_high => Some(true),
            CmpKind::Ge if high < other_low => Some(false),
            CmpKind::Lt | CmpKind::Le | CmpKind::Gt | CmpKind::Ge => None,
        }
    }

    /// What is known of a value once `value cmp bound` is known to hold:
    /// these facts with those the comparison adds, `None` when no value these
    /// facts admit satisfies it.
    ///
    /// The comparison bounds the value to an interval, compared unsigned or,
    /// when `unsigned` is false, two's complement; the bits above the highest
    /// bit in which the interval's two ends differ are those of both ends. An
    /// equality fixes every bit; an inequality adds nothing, short of
    /// excluding the one value these facts already fix. `bound` is read in the
    /// low `width` bits.
    #[must_use]
    pub fn bounded(self, cmp: CmpKind, unsigned: bool, bound: u128) -> Option<Self> {
        let all = self.mask();
        let bound = bound & all;
        if cmp == CmpKind::Ne {
            return (self.value() != Some(bound)).then_some(self);
        }
        let facts = self.ordered(unsigned);
        let bound = Self::parts(self.width(), !bound & all, bound)
            .ordered(unsigned)
            .ones();
        let (low, high) = match cmp {
            CmpKind::Eq => (bound, bound),
            CmpKind::Lt => (0, bound.checked_sub(1)?),
            CmpKind::Le => (0, bound),
            CmpKind::Gt => (bound.checked_add(1).filter(|low| *low <= all)?, all),
            CmpKind::Ge => (bound, all),
            CmpKind::Ne => return Some(self),
        };
        let (low, high) = (low.max(facts.ones()), high.min(facts.possibly_set()));
        if low > high {
            return None;
        }
        let differing = low ^ high;
        let common = if differing == 0 {
            all
        } else {
            let below = 1u128
                .checked_shl(u128::BITS.saturating_sub(differing.leading_zeros()))
                .map_or(u128::MAX, |above| above.wrapping_sub(1));
            all & !below
        };
        facts
            .meet(Self::parts(self.width(), !low & common, low & common))
            .map(|refined| refined.ordered(unsigned))
    }

    /// These facts in the coordinates where `unsigned` order is unsigned
    /// order: the same facts unsigned; with the sign bit inverted for two's
    /// complement, which maps the signed order onto the unsigned one. Its
    /// own inverse.
    const fn ordered(self, unsigned: bool) -> Self {
        if unsigned {
            return self;
        }
        let sign = match 1u128.checked_shl(self.width().saturating_sub(1)) {
            Some(sign) => sign,
            None => return self,
        };
        Self::parts(
            self.width(),
            (self.zeros() & !sign) | (self.ones() & sign),
            (self.ones() & !sign) | (self.zeros() & sign),
        )
    }

    /// Shifts left by an already interpreted count, inserting known zeros.
    ///
    /// Counts at least the width produce zero. Instruction-specific count
    /// interpretation belongs to the caller, as it does for concrete evaluation.
    #[must_use]
    pub const fn shifted_left(self, amount: u32) -> Self {
        if amount >= self.width() {
            return Self::parts(self.width(), self.mask(), 0);
        }
        let zeros = match self.zeros().checked_shl(amount) {
            Some(bits) => bits,
            None => 0,
        };
        let ones = match self.ones().checked_shl(amount) {
            Some(bits) => bits,
            None => 0,
        };
        let retained = match self.mask().checked_shl(amount) {
            Some(bits) => bits,
            None => 0,
        };
        let inserted = self.mask() & !retained;
        Self::parts(
            self.width(),
            zeros & self.mask() | inserted,
            ones & self.mask(),
        )
    }

    /// Shifts right by an interpreted count, inserting zeros or copies of the sign.
    #[must_use]
    pub const fn shifted_right(self, amount: u32, unsigned: bool) -> Self {
        let shift = if amount > self.width() {
            self.width()
        } else {
            amount
        };
        let zeros = match self.zeros().checked_shr(shift) {
            Some(bits) => bits,
            None => 0,
        };
        let ones = match self.ones().checked_shr(shift) {
            Some(bits) => bits,
            None => 0,
        };
        let retained = match self.mask().checked_shr(shift) {
            Some(bits) => bits,
            None => 0,
        };
        let inserted = self.mask() & !retained;
        let (high_zero, high_one) = match (unsigned, self.sign()) {
            (true, _) | (false, Some(false)) => (inserted, 0),
            (false, Some(true)) => (0, inserted),
            (false, None) => (0, 0),
        };
        Self::parts(self.width(), zeros | high_zero, ones | high_one)
    }

    /// Truncates or extends to a supported width with zeros or copies of the sign.
    #[must_use]
    pub const fn converted(self, width: u32, unsigned: bool) -> Option<Self> {
        let all = match IntegerType::new(width, false) {
            Some(integer) => integer.mask(),
            None => return None,
        };
        if width <= self.width() {
            return Some(Self::parts(width, self.zeros() & all, self.ones() & all));
        }
        let inserted = all & !self.mask();
        let (high_zero, high_one) = match (unsigned, self.sign()) {
            (true, _) | (false, Some(false)) => (inserted, 0),
            (false, Some(true)) => (0, inserted),
            (false, None) => (0, 0),
        };
        Some(Self::parts(
            width,
            self.zeros() | high_zero,
            self.ones() | high_one,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::IntegerBits;
    use crate::ir::ops::CmpKind;

    /// An abstract bit's independently enumerated concrete values.
    fn allowed(state: u8) -> &'static [bool] {
        match state {
            0 => &[false],
            1 => &[true],
            _ => &[false, true],
        }
    }

    /// Every bit position obeys the concrete truth table, at every supported width.
    #[test]
    fn bitwise_facts_match_concrete_bit_truth_tables() {
        for width in 1..=128 {
            for position in 0..width {
                let bit = 1u128 << position;
                for a in 0..3 {
                    for b in 0..3 {
                        let facts = |state| {
                            IntegerBits::new(
                                width,
                                if state == 0 { bit } else { 0 },
                                if state == 1 { bit } else { 0 },
                            )
                            .expect("consistent facts")
                        };
                        let (left, right) = (facts(a), facts(b));
                        for (computed, operation) in [
                            (left.and(right), (|a, b| a && b) as fn(bool, bool) -> bool),
                            (left.or(right), (|a, b| a || b) as fn(bool, bool) -> bool),
                            (left.xor(right), (|a, b| a != b) as fn(bool, bool) -> bool),
                        ] {
                            let result = computed.expect("matching widths");
                            let mut seen = [false; 2];
                            for a in allowed(a) {
                                for b in allowed(b) {
                                    seen[usize::from(operation(*a, *b))] = true;
                                }
                            }
                            assert_eq!(result.zeros() & bit != 0, seen == [true, false]);
                            assert_eq!(result.ones() & bit != 0, seen == [false, true]);
                        }
                        let joined = left.join(right).expect("matching widths");
                        assert_eq!(joined.zeros() & bit != 0, a == 0 && b == 0);
                        assert_eq!(joined.ones() & bit != 0, a == 1 && b == 1);
                        assert_eq!(left.inverted().zeros() & bit, left.ones() & bit);
                        assert_eq!(left.inverted().ones() & bit, left.zeros() & bit);
                    }
                }
            }
        }
    }

    /// Enumerates concrete inputs compatible with the cube and compares their common output bits.
    fn check_transfer(source: IntegerBits, result: IntegerBits, evaluate: impl Fn(u128) -> u128) {
        let mut every_one = result.mask();
        let mut any_one = 0;
        for input in 0..=source.mask() {
            if input & source.zeros() != 0 || input & source.ones() != source.ones() {
                continue;
            }
            let value = evaluate(input) & result.mask();
            every_one &= value;
            any_one |= value;
        }
        assert_eq!(result.ones(), every_one);
        assert_eq!(result.zeros(), !any_one & result.mask());
    }

    /// Shifts and extensions match every concrete input of every small partial-bit cube.
    #[test]
    fn shifts_and_conversions_match_concrete_integer_patterns() {
        for width in 1..=6 {
            for mut state in 0..3u32.pow(width) {
                let (mut zeros, mut ones) = (0, 0);
                for position in 0..width {
                    let bit = 1u128 << position;
                    match state % 3 {
                        0 => zeros |= bit,
                        1 => ones |= bit,
                        _ => {}
                    }
                    state /= 3;
                }
                let source = IntegerBits::new(width, zeros, ones).expect("valid cube");
                let sign = 1u128 << (width - 1);
                let signed = |input| {
                    if input & sign == 0 {
                        input as i128
                    } else {
                        (input | !source.mask()) as i128
                    }
                };
                for amount in 0..=8 {
                    check_transfer(source, source.shifted_left(amount), |input| input << amount);
                    check_transfer(source, source.shifted_right(amount, true), |input| {
                        input >> amount
                    });
                    check_transfer(source, source.shifted_right(amount, false), |input| {
                        (signed(input) >> amount) as u128
                    });
                }
                for target in [1, width, 8, 32, 128] {
                    check_transfer(
                        source,
                        source.converted(target, true).expect("supported target"),
                        |input| input,
                    );
                    check_transfer(
                        source,
                        source.converted(target, false).expect("supported target"),
                        |input| signed(input) as u128,
                    );
                }
            }
        }
    }

    /// Constructors reject contradictory geometry; full-width shifts never overflow the host.
    #[test]
    fn scalar_bit_geometry_and_inline_storage_are_explicit() {
        assert!(IntegerBits::unknown(0).is_none());
        assert!(IntegerBits::unknown(129).is_none());
        assert!(IntegerBits::new(8, 1, 1).is_none());
        assert!(IntegerBits::new(8, 256, 0).is_none());
        assert!(
            size_of::<IntegerBits>() <= 40,
            "keep integer facts inline without 128-bit alignment"
        );
        for width in 1..=128 {
            let unknown = IntegerBits::unknown(width).expect("supported width");
            assert!(unknown.is_unknown());
            assert_eq!(unknown.value(), None);
            let high = IntegerBits::constant(width, 1u128 << (width - 1)).expect("supported width");
            assert_eq!(high.sign(), Some(true));
            assert_eq!(high.shifted_left(128).value(), Some(0));
            assert_eq!(high.shifted_right(128, true).value(), Some(0));
            assert_eq!(high.shifted_right(128, false).value(), Some(high.mask()));
            assert_eq!(high.converted(width, false), Some(high));
            assert!(high.converted(0, true).is_none());
            assert!(high.converted(129, true).is_none());
        }
        let bit = IntegerBits::unknown(1).expect("supported");
        let byte = IntegerBits::unknown(8).expect("supported");
        assert!(bit.and(byte).is_none());
        assert!(bit.or(byte).is_none());
        assert!(bit.xor(byte).is_none());
        assert!(bit.join(byte).is_none());
    }

    /// Every description of a value of `width` bits: each bit known clear,
    /// known set or unknown.
    fn descriptions(width: u32) -> Vec<IntegerBits> {
        let count = 3usize.pow(width);
        (0..count)
            .map(|mut index| {
                let (mut zeros, mut ones) = (0u128, 0u128);
                for position in 0..width {
                    match index % 3 {
                        0 => zeros |= 1 << position,
                        1 => ones |= 1 << position,
                        _ => {}
                    }
                    index /= 3;
                }
                IntegerBits::new(width, zeros, ones).expect("consistent facts")
            })
            .collect()
    }

    /// The values a description admits.
    fn admitted(facts: IntegerBits) -> impl Iterator<Item = u128> {
        (0..=facts.mask())
            .filter(move |value| value & facts.zeros() == 0 && value & facts.ones() == facts.ones())
    }

    /// `left cmp right` of two `width`-bit patterns, as the machine compares
    /// them.
    fn holds(width: u32, cmp: CmpKind, unsigned: bool, left: u128, right: u128) -> bool {
        let read = |value: u128| -> i128 {
            if unsigned || value >> (width - 1) == 0 {
                value as i128
            } else {
                value as i128 - (1i128 << width)
            }
        };
        let (left, right) = (read(left), read(right));
        match cmp {
            CmpKind::Eq => left == right,
            CmpKind::Ne => left != right,
            CmpKind::Lt => left < right,
            CmpKind::Le => left <= right,
            CmpKind::Gt => left > right,
            CmpKind::Ge => left >= right,
        }
    }

    const COMPARISONS: [CmpKind; 6] = [
        CmpKind::Eq,
        CmpKind::Ne,
        CmpKind::Lt,
        CmpKind::Le,
        CmpKind::Gt,
        CmpKind::Ge,
    ];

    /// A comparison settled by known bits holds, or fails, for every pair of
    /// values the two descriptions admit, against enumeration at every width
    /// through four bits, signed and unsigned.
    #[test]
    fn a_settled_comparison_holds_for_every_admitted_pair() {
        for width in 1..=4 {
            let all = descriptions(width);
            for (left, right) in all
                .iter()
                .flat_map(|left| all.iter().map(move |right| (*left, *right)))
            {
                for cmp in COMPARISONS {
                    for unsigned in [false, true] {
                        let outcomes: Vec<bool> = admitted(left)
                            .flat_map(|a| admitted(right).map(move |b| (a, b)))
                            .map(|(a, b)| holds(width, cmp, unsigned, a, b))
                            .collect();
                        if let Some(settled) = left.compare(cmp, unsigned, right) {
                            assert!(
                                outcomes.iter().all(|outcome| *outcome == settled),
                                "{left:?} {cmp:?} {right:?} unsigned {unsigned} settled {settled}"
                            );
                        }
                        // Both known whole is always settled.
                        if left.value().is_some() && right.value().is_some() {
                            assert!(left.compare(cmp, unsigned, right).is_some());
                        }
                    }
                }
            }
        }
        let byte = IntegerBits::unknown(8).expect("supported");
        assert_eq!(
            byte.compare(
                CmpKind::Eq,
                true,
                IntegerBits::unknown(4).expect("supported")
            ),
            None
        );
    }

    /// A value bounded by a comparison keeps every value that satisfies it,
    /// is refused only when none does, and from no facts at all knows
    /// exactly the bits every satisfying value shares, against enumeration at
    /// every width through four bits, signed and unsigned.
    #[test]
    fn a_bounded_value_keeps_every_value_that_satisfies_the_bound() {
        for width in 1..=4 {
            for facts in descriptions(width) {
                for bound in 0..=facts.mask() {
                    for cmp in COMPARISONS {
                        for unsigned in [false, true] {
                            let satisfying: Vec<u128> = admitted(facts)
                                .filter(|value| holds(width, cmp, unsigned, *value, bound))
                                .collect();
                            let label = format!("{facts:?} {cmp:?} {bound} unsigned {unsigned}");
                            let Some(refined) = facts.bounded(cmp, unsigned, bound) else {
                                assert!(satisfying.is_empty(), "refused {label}");
                                continue;
                            };
                            let kept: Vec<u128> = admitted(refined).collect();
                            assert!(
                                satisfying.iter().all(|value| kept.contains(value)),
                                "{label} lost a value: kept {refined:?}"
                            );
                            if facts.is_unknown() && cmp != CmpKind::Ne {
                                let shared_ones =
                                    satisfying.iter().fold(facts.mask(), |acc, v| acc & v);
                                let shared_zeros =
                                    satisfying.iter().fold(facts.mask(), |acc, v| acc & !v);
                                assert_eq!(
                                    (refined.ones(), refined.zeros()),
                                    (shared_ones, shared_zeros & facts.mask()),
                                    "{label} is not exact"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// An index known below eight unsigned is known clear above its low
    /// three bits, and so is one known not negative and below eight signed.
    #[test]
    fn a_bounded_index_is_clear_above_its_bound() {
        let index = IntegerBits::unknown(32).expect("supported");
        let below = index.bounded(CmpKind::Lt, true, 8).expect("satisfiable");
        assert_eq!(below.possibly_set(), 7);
        let signed = index.bounded(CmpKind::Lt, false, 8).expect("satisfiable");
        assert_eq!(signed.sign(), None, "a signed bound alone admits negatives");
        let natural = index
            .bounded(CmpKind::Ge, false, 0)
            .and_then(|natural| natural.bounded(CmpKind::Lt, false, 8))
            .expect("satisfiable");
        assert_eq!(natural.possibly_set(), 7);
        assert!(below.bounded(CmpKind::Ge, true, 8).is_none());
        assert_eq!(
            below.meet(IntegerBits::constant(32, 9).expect("supported")),
            None
        );
    }
}
