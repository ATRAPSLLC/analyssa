//! Known bits of integer operands, read the way concrete evaluation reads them.
//!
//! Constant folding and static execution both need what is known of an
//! integer that is not fully constant: a masked field, a value with its high
//! bits cleared. The methods here read a constant or a partial value through
//! its declared scalar integer register ([`IntegerBits`]) with the same carrier
//! extension as concrete evaluation, so a known bit means the same in either
//! consumer. Floating registers and explicit vectors establish no integer bits.
//!
//! [`IntegerBits`]: crate::ir::integer::IntegerBits

use crate::{
    analysis::consts::{Destination, Machine, OperandLayout, TypeQueries},
    ir::{
        integer::{IntegerBits, IntegerType, shift_distance},
        ops::{CmpKind, SsaOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

impl Destination {
    /// Returns a consistent scalar integer declaration, refusing vector geometry.
    pub fn scalar_integer(self) -> Option<IntegerType> {
        let ty = self.integer_type?;
        (self.integer && self.lanes.is_none() && self.bits == Some(ty.bits())).then_some(ty)
    }
}

impl OperandLayout {
    /// Returns the exact scalar bit domain without inspecting target metadata.
    pub fn scalar_integer(self) -> Option<IntegerType> {
        match self {
            Self::Boolean => IntegerType::new(1, false),
            Self::Register(dest) => dest.scalar_integer(),
            Self::Float(_) => None,
        }
    }
}

impl Machine {
    /// Reads a constant through its declared scalar integer register.
    ///
    /// Floating registers and explicit vectors establish no scalar integer facts.
    /// Carrier extension follows the same register read as concrete evaluation.
    pub fn constant_integer_bits<T: Target>(
        self,
        value: &ConstValue<T>,
        layout: OperandLayout,
    ) -> Option<IntegerBits> {
        let ty = layout.scalar_integer()?;
        let pattern = if matches!(layout, OperandLayout::Boolean) {
            u128::from(value.logical_boolean(self.ptr_size)?)
        } else {
            value.scalar_pattern_at_width(ty.bits(), self.ptr_size)?
        };
        IntegerBits::constant(ty.bits(), pattern)
    }

    /// Transfers partial bits once from the operand facts.
    ///
    /// This forward transfer never traverses SSA definitions. Missing facts are
    /// unknown at their declared width. Unsupported operations, inconsistent
    /// layouts and checked conversions establish no partial result.
    pub fn evaluate_integer_bits<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        mut get_bits: impl FnMut(SsaVarId, OperandLayout) -> Option<IntegerBits>,
        mut layout_for: impl FnMut(SsaVarId) -> OperandLayout,
        types: &TypeQueries<'_, T>,
    ) -> Option<IntegerBits> {
        let output = if matches!(op, SsaOp::ReadFlags { .. }) {
            if dest.bits != Some(1) || dest.lanes.is_some() {
                return None;
            }
            IntegerType::new(1, false)?
        } else if matches!(
            op,
            SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. }
        ) {
            // A comparison's result is a truth value whatever its
            // destination declares; only a settled one is materialized.
            IntegerType::new(1, false)?
        } else {
            dest.scalar_integer()?
        };
        let mut read = |variable| {
            let layout = layout_for(variable);
            let ty = layout.scalar_integer()?;
            let bits = get_bits(variable, layout).or_else(|| IntegerBits::unknown(ty.bits()))?;
            (bits.width() == ty.bits()).then_some((bits, layout))
        };
        let result = match op {
            SsaOp::ReadFlags { flags, mask, .. } => {
                let input = read(*flags)?.0;
                let selected = u128::from(mask.bits());
                if !mask.is_single_bit() || selected & !input.mask() != 0 {
                    return None;
                }
                let known = input.ones() & selected != 0 || input.zeros() & selected != 0;
                if known {
                    IntegerBits::constant(1, u128::from(input.ones() & selected != 0))
                } else {
                    IntegerBits::unknown(1)
                }
            }
            SsaOp::MergeFlags {
                base, update, mask, ..
            } => {
                let base = read(*base)?.0;
                let update = read(*update)?.0;
                let selected = u128::from(mask.bits());
                if !mask.is_valid()
                    || base.width() != update.width()
                    || selected & !base.mask() != 0
                {
                    return None;
                }
                IntegerBits::new(
                    base.width(),
                    (base.zeros() & !selected) | (update.zeros() & selected),
                    (base.ones() & !selected) | (update.ones() & selected),
                )
            }
            SsaOp::InsertFlag {
                base, bit, mask, ..
            } => {
                let base = read(*base)?.0;
                let bit = read(*bit)?.0;
                let selected = u128::from(mask.bits());
                if !mask.is_single_bit() || bit.width() != 1 || selected & !base.mask() != 0 {
                    return None;
                }
                let zero = if bit.zeros() & 1 != 0 { selected } else { 0 };
                let one = if bit.ones() & 1 != 0 { selected } else { 0 };
                IntegerBits::new(
                    base.width(),
                    (base.zeros() & !selected) | zero,
                    (base.ones() & !selected) | one,
                )
            }
            SsaOp::Ceq { left, right, .. } => {
                Self::compared(read(*left)?.0, CmpKind::Eq, true, read(*right)?.0)
            }
            SsaOp::Clt {
                left,
                right,
                unsigned,
                ..
            } => Self::compared(read(*left)?.0, CmpKind::Lt, *unsigned, read(*right)?.0),
            SsaOp::Cgt {
                left,
                right,
                unsigned,
                ..
            } => Self::compared(read(*left)?.0, CmpKind::Gt, *unsigned, read(*right)?.0),
            SsaOp::And { left, right, .. } => read(*left)?.0.and(read(*right)?.0),
            SsaOp::Or { left, right, .. } => read(*left)?.0.or(read(*right)?.0),
            SsaOp::Xor { left, right, .. } => read(*left)?.0.xor(read(*right)?.0),
            SsaOp::Not { operand, .. } => Some(read(*operand)?.0.inverted()),
            SsaOp::Shl { value, amount, .. } => {
                let input = read(*value)?.0;
                let count = shift_distance(read(*amount)?.0.value()?, input.width())?;
                Some(input.shifted_left(count))
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned,
                ..
            } => {
                let input = read(*value)?.0;
                let count = shift_distance(read(*amount)?.0.value()?, input.width())?;
                Some(input.shifted_right(count, *unsigned))
            }
            SsaOp::Copy { src, .. } => Some(read(*src)?.0),
            SsaOp::IntConv {
                operand,
                target,
                unsigned,
                overflow_check: false,
                ..
            } => {
                let target = types.integer_type(target)?;
                if target.bits() != output.bits() {
                    return None;
                }
                let (input, layout) = read(*operand)?;
                input.converted(
                    output.bits(),
                    *unsigned || matches!(layout, OperandLayout::Boolean),
                )
            }
            SsaOp::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                let condition = read(*condition)?.0;
                match condition.value() {
                    Some(0) => Some(read(*false_val)?.0),
                    Some(_) => Some(read(*true_val)?.0),
                    None => read(*true_val)?.0.join(read(*false_val)?.0),
                }
            }
            _ => None,
        }?;
        (result.width() == output.bits()).then_some(result)
    }

    /// The truth value of `left cmp right` as one bit: known where the
    /// operands' known bits settle it, unknown where they do not. Operands of
    /// different widths settle nothing.
    fn compared(
        left: IntegerBits,
        cmp: CmpKind,
        unsigned: bool,
        right: IntegerBits,
    ) -> Option<IntegerBits> {
        match left.compare(cmp, unsigned, right) {
            Some(holds) => IntegerBits::constant(1, u128::from(holds)),
            None => IntegerBits::unknown(1),
        }
    }

    /// Materializes the fully known bits [`Self::evaluate_integer_bits`]
    /// found for `op`'s result: a truth value for an operation whose result
    /// is one (a comparison, a flag read), whatever its destination declares;
    /// otherwise a constant of the declared result type.
    pub fn known_result<T: Target>(
        self,
        op: &SsaOp<T>,
        bits: IntegerBits,
        dest: Destination,
    ) -> Option<ConstValue<T>> {
        if matches!(
            op,
            SsaOp::ReadFlags { .. } | SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. }
        ) {
            return (bits.width() == 1)
                .then(|| bits.value())
                .flatten()
                .map(|value| ConstValue::from_bool(value != 0));
        }
        self.integer_bits_constant(bits, dest)
    }

    /// Materializes fully known bits as a constant of the declared result type.
    ///
    /// Signedness and native representation come from the declared result type.
    /// A partial value establishes no concrete constant.
    pub fn integer_bits_constant<T: Target>(
        self,
        bits: IntegerBits,
        dest: Destination,
    ) -> Option<ConstValue<T>> {
        let ty = dest.scalar_integer()?;
        if bits.width() != ty.bits() {
            return None;
        }
        let pattern = bits.value()?;
        ty.constant(pattern, self.ptr_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::FlagsMask,
        testing::{MACHINE, MockTarget, MockType},
    };

    /// Supplies a consistent register declaration for one scalar bit domain.
    fn scalar(width: u32, signed: bool) -> Destination {
        Destination {
            bits: Some(width),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(width, signed),
        }
    }

    /// Each bit position survives a typed masked merge across all supported widths.
    #[test]
    fn typed_integer_bits_all_widths_and_signed_results() {
        let var = SsaVarId::from_index;
        for width in 1..=128 {
            let all = u128::MAX >> (128 - width);
            for signed in [false, true] {
                let dest = scalar(width, signed);
                let types = TypeQueries::<MockTarget>::ordinary(MACHINE.ptr_size);
                for bit in 0..width {
                    let flag = 1u128 << bit;
                    let upper = IntegerBits::new(width, flag, 0).unwrap();
                    let one = IntegerBits::constant(width, flag).unwrap();
                    let op = SsaOp::Or {
                        dest: var(2),
                        left: var(0),
                        right: var(1),
                        flags: None,
                    };
                    let merged = MACHINE
                        .evaluate_integer_bits(
                            &op,
                            dest,
                            |id, _| Some(if id == var(0) { upper } else { one }),
                            |_| OperandLayout::Register(dest),
                            &types,
                        )
                        .unwrap();
                    let op = SsaOp::And {
                        dest: var(3),
                        left: var(2),
                        right: var(1),
                        flags: None,
                    };
                    let exact = MACHINE
                        .evaluate_integer_bits(
                            &op,
                            dest,
                            |id, _| Some(if id == var(2) { merged } else { one }),
                            |_| OperandLayout::Register(dest),
                            &types,
                        )
                        .unwrap();
                    assert_eq!(exact.value(), Some(flag));
                    assert_eq!(exact.zeros(), all ^ flag);
                    let constant = MACHINE
                        .integer_bits_constant::<MockTarget>(exact, dest)
                        .unwrap();
                    assert_eq!(
                        if constant.is_bool() {
                            constant.as_bool().map(u128::from)
                        } else {
                            constant
                                .scalar_integer_pattern(MACHINE.ptr_size)
                                .map(|(bits, _)| bits)
                        },
                        Some(flag),
                        "width={width} signed={signed} bit={bit}"
                    );
                    if signed && matches!(width, 8 | 16 | 32 | 64) && bit == width - 1 {
                        assert!(constant.as_i64().unwrap() < 0);
                    }
                }
            }
        }
    }

    /// Width mismatches, vector declarations and checked conversions establish no bit proof.
    #[test]
    fn typed_integer_bits_refuse_inconsistent_and_checked_layouts() {
        let var = SsaVarId::from_index;
        let dest = scalar(8, false);
        let op = SsaOp::<MockTarget>::And {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        let types = TypeQueries::ordinary(MACHINE.ptr_size);
        for invalid in [
            Destination {
                bits: Some(16),
                ..dest
            },
            Destination {
                lanes: Some((8, 1)),
                ..dest
            },
            Destination {
                integer: false,
                ..dest
            },
            Destination {
                integer_type: None,
                ..dest
            },
        ] {
            assert!(
                MACHINE
                    .evaluate_integer_bits(
                        &op,
                        invalid,
                        |_, _| IntegerBits::constant(8, 0),
                        |_| OperandLayout::Register(dest),
                        &types,
                    )
                    .is_none()
            );
        }
        assert!(
            MACHINE
                .evaluate_integer_bits(
                    &op,
                    dest,
                    |_, _| IntegerBits::constant(16, 0),
                    |_| OperandLayout::Register(dest),
                    &types,
                )
                .is_none()
        );
        let checked = SsaOp::IntConv {
            dest: var(1),
            operand: var(0),
            target: MockType::U8,
            overflow_check: true,
            unsigned: true,
        };
        assert!(
            MACHINE
                .evaluate_integer_bits(
                    &checked,
                    dest,
                    |_, _| IntegerBits::constant(8, 0),
                    |_| OperandLayout::Register(dest),
                    &types,
                )
                .is_none()
        );
    }
    /// Every named partial flag write establishes exactly its selected bit.
    #[test]
    fn partial_flag_writes_reads_and_merges_preserve_unselected_unknown_bits() {
        let var = SsaVarId::from_index;
        let word = scalar(32, false);
        let boolean = scalar(1, false);
        let types = TypeQueries::<MockTarget>::ordinary(MACHINE.ptr_size);
        for &(flag, _) in FlagsMask::NAMED {
            for set in [false, true] {
                let op = SsaOp::InsertFlag {
                    dest: var(2),
                    base: var(0),
                    bit: var(1),
                    mask: flag,
                };
                let bits = MACHINE
                    .evaluate_integer_bits(
                        &op,
                        word,
                        |id, _| {
                            (id == var(1))
                                .then(|| IntegerBits::constant(1, u128::from(set)))
                                .flatten()
                        },
                        |id| {
                            if id == var(1) {
                                OperandLayout::Boolean
                            } else {
                                OperandLayout::Register(word)
                            }
                        },
                        &types,
                    )
                    .unwrap();
                let selected = u128::from(flag.bits());
                assert_eq!(bits.zeros(), if set { 0 } else { selected });
                assert_eq!(bits.ones(), if set { selected } else { 0 });
                let read = SsaOp::ReadFlags {
                    dest: var(3),
                    flags: var(2),
                    mask: flag,
                };
                let read = MACHINE
                    .evaluate_integer_bits(
                        &read,
                        boolean,
                        |_, _| Some(bits),
                        |_| OperandLayout::Register(word),
                        &types,
                    )
                    .unwrap();
                assert_eq!(read.value(), Some(u128::from(set)));
                let merge = SsaOp::MergeFlags {
                    dest: var(4),
                    base: var(0),
                    update: var(2),
                    mask: flag,
                };
                let merged = MACHINE
                    .evaluate_integer_bits(
                        &merge,
                        word,
                        |id, _| (id == var(2)).then_some(bits),
                        |_| OperandLayout::Register(word),
                        &types,
                    )
                    .unwrap();
                assert_eq!(merged, bits);
            }
        }
        let invalid = SsaOp::ReadFlags {
            dest: var(3),
            flags: var(2),
            mask: FlagsMask::from_bits(FlagsMask::CARRY.bits() | FlagsMask::ZERO.bits()),
        };
        assert!(
            MACHINE
                .evaluate_integer_bits(
                    &invalid,
                    boolean,
                    |_, _| IntegerBits::unknown(32),
                    |_| OperandLayout::Register(word),
                    &types
                )
                .is_none()
        );
    }
}
