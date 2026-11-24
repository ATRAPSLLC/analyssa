//! Bits known of a value whose operands are known only in part.

use std::{cell::RefCell, collections::BTreeMap, iter};

use rustc_hash::FxHashMap;

use crate::{
    analysis::{consts::Machine, reshape},
    ir::{
        float::FloatRounding,
        function::SsaFunction,
        integer::BitWidth,
        lanes::VectorConst,
        ops::{SsaOp, VectorElement, VectorShuffleSelector, VectorTernaryKind},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// How many definitions [`Machine::fold_known_bits`] follows from a value to the
/// constants its bits come from: a lane through a lowered register merge is
/// a handful of reshapes, a permute and a mask deep.
pub const KNOWN_BITS_DEPTH: u32 = 16;

/// The bits of values that are known though the values are not constants,
/// by the rules [`Machine::fold_known_bits`] states, each value's bits
/// answered once.
pub struct KnownBits<'a, T: Target> {
    /// The machine the operations compute as.
    pub machine: Machine,
    /// The function, for the definitions.
    pub ssa: &'a SsaFunction<T>,
    /// Known constant values.
    pub constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
    /// Each value's bits answered so far, by value, first bit and width,
    /// with the depth left when they were: an unknown answer holds for no
    /// greater depth.
    pub memo: RefCell<KnownAnswers>,
}

/// The bits [`KnownBits`] has answered, by value, first bit and width, with
/// the depth left when it answered.
pub type KnownAnswers = FxHashMap<(SsaVarId, u32, u32), (u32, Option<u128>)>;

impl<T: Target> KnownBits<'_, T> {
    /// The `width` bits of `var` from bit `start`, when they are known,
    /// following at most `depth` definitions.
    fn bits(&self, var: SsaVarId, start: u32, width: u32, depth: u32) -> Option<u128> {
        if let Some(constant) = self.constants.get(&var) {
            return Some(
                constant
                    .as_lanes(self.machine.ptr_size)?
                    .bits_at(start, width),
            );
        }
        let key = (var, start, width);
        if let Some(&(answered, bits)) = self.memo.borrow().get(&key)
            && (bits.is_some() || answered >= depth)
        {
            return bits;
        }
        let bits = depth
            .checked_sub(1)
            .and_then(|depth| self.defined(self.ssa.get_definition(var)?, start, width, depth));
        self.memo.borrow_mut().insert(key, (depth, bits));
        bits
    }

    /// The `width` bits from bit `start` of the result `op` computes from
    /// known bits, following at most `depth` further definitions.
    pub fn defined(&self, op: &SsaOp<T>, start: u32, width: u32, depth: u32) -> Option<u128> {
        let ssa = self.ssa;
        let known = |var: SsaVarId, start: u32, width: u32| self.bits(var, start, width, depth);
        let mask = BitWidth::new(width).checked_mask_u128()?;
        // The lane holding the bits, of lanes `lane_bits` wide, and their
        // first bit in it.
        let lane_of = |lane_bits: u32| -> Option<(u32, u32)> {
            let (lane, offset) = (start.checked_div(lane_bits)?, start.checked_rem(lane_bits)?);
            (offset.checked_add(width)? <= lane_bits).then_some((lane, offset))
        };
        // The width a lane-wise operation computes in: its element's, or its
        // operand's lanes' where the element states none.
        let lanes = |element: &VectorElement, operand: SsaVarId| -> Option<u32> {
            match element.bits {
                0 => {
                    T::vector_shape(ssa.variable(operand)?.var_type()).map(|shape| shape.lane_bits)
                }
                bits => Some(bits),
            }
        };
        let one = |bits: u128, lane_bits: u32| VectorConst::from_lanes(lane_bits, [bits]);
        // The mode an operation rounding by the environment rounds by: its
        // mode operand's where every bit of it is known.
        let rounding = match op.environment() {
            Some(mode) => match ssa
                .bit_width(mode)
                .and_then(|bits| known(mode, 0, bits.min(64)))
            {
                Some(mode) => FloatRounding::of_environment(u64::try_from(mode).ok()?)?.into(),
                None => T::environment_rounding(),
            },
            None => T::environment_rounding(),
        };
        let rules = self.machine.float;
        // A lane-wise operation's bits: its first operand's above lane zero of
        // a scalar operation, a bitwise one's of its operands' same bits, and
        // otherwise those of the lane computed of its operands' lane.
        let lane_wise = |element: &VectorElement,
                         bitwise: bool,
                         first: SsaVarId,
                         compute: &dyn Fn(&[VectorConst], VectorElement) -> Option<VectorConst>,
                         operands: &[SsaVarId]|
         -> Option<u128> {
            let lane_bits = lanes(element, first)?;
            if element.scalar && start >= lane_bits {
                return known(first, start, width);
            }
            let (lane_start, lane_bits, offset) =
                if bitwise && (!element.scalar || start.checked_add(width)? <= lane_bits) {
                    (start, width, 0)
                } else {
                    let (lane, offset) = lane_of(lane_bits)?;
                    (lane.checked_mul(lane_bits)?, lane_bits, offset)
                };
            let operands = operands
                .iter()
                .map(|operand| one(known(*operand, lane_start, lane_bits)?, lane_bits))
                .collect::<Option<Vec<_>>>()?;
            let element = VectorElement {
                scalar: false,
                bits: lane_bits,
                ..*element
            };
            Some(compute(&operands, element)?.lane(0).checked_shr(offset)? & mask)
        };
        match op {
            SsaOp::Const { value, .. } => {
                Some(value.as_lanes(self.machine.ptr_size)?.bits_at(start, width))
            }
            SsaOp::VectorBinary {
                left,
                right,
                kind,
                element,
                ..
            } => lane_wise(
                element,
                kind.is_bitwise(),
                *left,
                &|operands, element| match operands {
                    [left, right] => left.binary(right, *kind, element, rounding, rules),
                    _ => None,
                },
                &[*left, *right],
            ),
            SsaOp::VectorUnary {
                value,
                kind,
                element,
                ..
            } => lane_wise(
                element,
                kind.is_bitwise(),
                *value,
                &|operands, element| match operands {
                    [value] => value.unary(*kind, element, rounding, rules),
                    _ => None,
                },
                &[*value],
            ),
            // The lane of the second operand where the mask lane's top bit
            // is set, else of the first.
            SsaOp::VectorTernary {
                first,
                second,
                third,
                kind: VectorTernaryKind::Select,
                element,
                ..
            } if !element.scalar => {
                let lane_bits = lanes(element, *first)?;
                let (lane, _) = lane_of(lane_bits)?;
                let top = lane
                    .checked_mul(lane_bits)?
                    .checked_add(lane_bits.checked_sub(1)?)?;
                let chosen = if known(*third, top, 1)? == 1 {
                    *second
                } else {
                    *first
                };
                known(chosen, start, width)
            }
            SsaOp::VectorTernary {
                first,
                second,
                third,
                kind,
                element,
                ..
            } => lane_wise(
                element,
                false,
                *first,
                &|operands, element| match operands {
                    [first, second, third] => {
                        first.ternary(second, third, *kind, element, rounding, rules)
                    }
                    _ => None,
                },
                &[*first, *second, *third],
            ),
            // The inputs laid end to end at the left's lane width, each result
            // lane the one its index lane selects.
            SsaOp::VectorShuffle {
                left,
                right,
                selector: VectorShuffleSelector::Index(index),
                ..
            } => {
                let tables: Vec<SsaVarId> = iter::once(*left).chain(*right).collect();
                let lane_bits = T::vector_shape(ssa.variable(*left)?.var_type())?.lane_bits;
                let index_bits = T::vector_shape(ssa.variable(*index)?.var_type())?.lane_bits;
                let (lane, offset) = lane_of(lane_bits)?;
                let mut select =
                    u32::try_from(known(*index, lane.checked_mul(index_bits)?, index_bits)?)
                        .ok()?;
                for table in tables {
                    let count = ssa.bit_width(table)?.checked_div(lane_bits)?;
                    if select < count {
                        return known(
                            table,
                            select.checked_mul(lane_bits)?.checked_add(offset)?,
                            width,
                        );
                    }
                    select = select.checked_sub(count)?;
                }
                None
            }
            _ => reshape::known_slice(ssa, op, start, width, &known),
        }
    }
}
