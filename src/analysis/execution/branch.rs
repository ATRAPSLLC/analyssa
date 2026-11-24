//! Where control goes from the end of a block.
//!
//! A transfer whose condition the execution knows is decided; one whose
//! condition it does not know may go to any of its targets, and exploring all
//! of them is the explorer's business. A flag-tested branch is decided from the
//! instruction that set the flags, on its result alone: zero and sign are facts
//! about the value, while carry and overflow are facts about how it was
//! computed, whose sense differs between architectures.

use crate::{
    analysis::{
        consts::{Machine, OperandLayout},
        execution::{
            types::TypeLayouts,
            value::{Root, Value},
        },
    },
    ir::{
        block::SsaBlock,
        float::FloatRules,
        function::SsaFunction,
        integer::IntegerBits,
        ops::{CmpKind, FlagCondition, SsaOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// Where control goes from the end of a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// To this block, and nowhere else.
    Goto(usize),
    /// To any of these blocks: the condition is not known.
    Either(Vec<usize>),
    /// Out of the function, by returning.
    Return,
    /// Nowhere: control cannot reach past this point.
    Halt,
    /// Somewhere the execution cannot follow - a thrown exception, or an
    /// indirect jump to a target the function does not list.
    Stuck,
}

/// Decides the transfers of one function.
#[derive(Debug)]
pub struct Branches<'a, T: Target> {
    /// The function, for the instruction a flags value was set by.
    ssa: &'a SsaFunction<T>,
    /// The width of a native integer, for a constant that states no other.
    ptr_size: PointerSize,
    /// The floating input modes allowed by the host.
    float: FloatRules,
    /// Numeric layouts shared by all paths of the body.
    layouts: Option<&'a TypeLayouts<'a, T>>,
}

impl<'a, T: Target> Branches<'a, T> {
    /// Decides the transfers of `ssa`, on a target whose pointers are
    /// `ptr_size` wide.
    pub const fn new(ssa: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self {
            ssa,
            ptr_size,
            float: FloatRules::NONE,
            layouts: None,
        }
    }

    /// Installs the host's floating input-mode contract.
    pub const fn with_float_rules(mut self, rules: FloatRules) -> Self {
        self.float = rules;
        self
    }

    /// Installs this body's shared numeric layouts.
    pub const fn with_layouts(mut self, layouts: &'a TypeLayouts<'a, T>) -> Self {
        self.layouts = Some(layouts);
        self
    }

    /// Reads a comparison input through the same layout cache as scalar
    /// execution, where one is installed.
    fn operand_layout(&self, value: SsaVarId) -> OperandLayout {
        match self.layouts {
            Some(layouts) => layouts.operand(value),
            None => OperandLayout::of(self.ssa, value),
        }
    }

    /// Decides where control goes from the end of `block`.
    ///
    /// # Arguments
    ///
    /// * `block` - The block.
    /// * `value` - What the execution knows about each value.
    pub fn decide(&self, block: &SsaBlock<T>, value: impl Fn(SsaVarId) -> Value<T>) -> Decision {
        let Some(terminator) = block.control_terminator() else {
            return match block.successors().as_slice() {
                [] => Decision::Halt,
                [only] => Decision::Goto(*only),
                many => Decision::Either(many.to_vec()),
            };
        };
        match terminator {
            SsaOp::Jump { target } | SsaOp::Leave { target } => Decision::Goto(*target),
            SsaOp::Branch {
                condition,
                true_target,
                false_target,
            } => Self::two_way(value(*condition).truth(), *true_target, *false_target),
            SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            } => {
                let (left_value, right_value) = (value(*left), value(*right));
                Self::two_way(
                    self.compare(
                        *cmp,
                        *unsigned,
                        &self.comparison_value(*left, left_value.clone()),
                        &self.comparison_value(*right, right_value.clone()),
                    )
                    .or_else(|| {
                        self.compare_bits(
                            *cmp,
                            *unsigned,
                            (*left, &left_value),
                            (*right, &right_value),
                        )
                    }),
                    *true_target,
                    *false_target,
                )
            }
            SsaOp::BranchFlags {
                flags,
                condition,
                true_target,
                false_target,
            } => Self::two_way(
                self.flag(*flags, *condition, &value),
                *true_target,
                *false_target,
            ),
            SsaOp::Switch {
                value: selector,
                targets,
                default,
            } => {
                let held = value(*selector);
                match held.constant().and_then(ConstValue::as_u64) {
                    Some(index) => Decision::Goto(
                        usize::try_from(index)
                            .ok()
                            .and_then(|index| targets.get(index))
                            .copied()
                            .unwrap_or(*default),
                    ),
                    None => Self::cases(held.integer_bits(), targets, *default),
                }
            }
            SsaOp::IndirectBranch {
                resolved_targets, ..
            } => match resolved_targets.as_slice() {
                [] => Decision::Stuck,
                [only] => Decision::Goto(*only),
                many => Decision::Either(many.to_vec()),
            },
            SsaOp::Return { .. } | SsaOp::InterruptReturn => Decision::Return,
            SsaOp::Unreachable => Decision::Halt,
            _ => Decision::Stuck,
        }
    }

    /// A two-way transfer, decided when `taken` is known.
    fn two_way(taken: Option<bool>, true_target: usize, false_target: usize) -> Decision {
        match taken {
            Some(true) => Decision::Goto(true_target),
            Some(false) => Decision::Goto(false_target),
            None if true_target == false_target => Decision::Goto(true_target),
            None => Decision::Either(vec![true_target, false_target]),
        }
    }

    /// Where a switch on a selector not known whole goes: each case whose
    /// index the selector's known bits admit, and the default where the
    /// selector may lie past the last case. No bits admit every case.
    fn cases(bits: Option<IntegerBits>, targets: &[usize], default: usize) -> Decision {
        // A case index past the selector's width is no value it holds.
        let admits = |index: usize| {
            bits.is_none_or(|bits| {
                u128::try_from(index)
                    .ok()
                    .filter(|index| *index <= bits.mask())
                    .and_then(|index| IntegerBits::constant(bits.width(), index))
                    .is_some_and(|case| bits.compare(CmpKind::Eq, true, case) != Some(false))
            })
        };
        let past = bits.is_none_or(|bits| {
            u128::try_from(targets.len()).is_ok_and(|count| bits.possibly_set() >= count)
        });
        let mut all: Vec<usize> = targets
            .iter()
            .enumerate()
            .filter(|(index, _)| admits(*index))
            .map(|(_, target)| *target)
            .chain(past.then_some(default))
            .collect();
        all.sort_unstable();
        all.dedup();
        match all.as_slice() {
            [only] => Decision::Goto(*only),
            _ => Decision::Either(all),
        }
    }

    /// Reads comparison data at the source variable's declared layout.
    fn comparison_value(&self, var: SsaVarId, value: Value<T>) -> Value<T> {
        match value {
            Value::Const(constant) => self
                .operand_layout(var)
                .read_comparison(constant, self.ptr_size)
                .map_or(Value::Unknown, Value::Const),
            Value::Addr(_) if matches!(self.operand_layout(var), OperandLayout::Float(_)) => {
                Value::Unknown
            }
            other => other,
        }
    }

    /// Whether `left cmp right` holds, when the known bits of a partially
    /// known operand settle it: each operand read at its declared integer
    /// layout, a constant whole. Two constants are [`Self::compare`]'s.
    fn compare_bits(
        &self,
        cmp: CmpKind,
        unsigned: bool,
        left: (SsaVarId, &Value<T>),
        right: (SsaVarId, &Value<T>),
    ) -> Option<bool> {
        if !matches!(left.1, Value::Integer(_)) && !matches!(right.1, Value::Integer(_)) {
            return None;
        }
        let machine = Machine::new(self.ptr_size, None, self.float);
        let bits = |(var, value): (SsaVarId, &Value<T>)| match value {
            Value::Integer(_) => value.integer_bits(),
            Value::Const(constant) => {
                machine.constant_integer_bits(constant, self.operand_layout(var))
            }
            _ => None,
        };
        bits(left)?.compare(cmp, unsigned, bits(right)?)
    }

    /// Whether `left cmp right` holds, when that is known.
    fn compare(
        &self,
        cmp: CmpKind,
        unsigned: bool,
        left: &Value<T>,
        right: &Value<T>,
    ) -> Option<bool> {
        match (left, right) {
            (Value::Const(left), Value::Const(right)) => {
                Machine::new(self.ptr_size, None, self.float)
                    .compare_values(left, right, cmp, unsigned)
            }
            (Value::Addr(left), Value::Addr(right)) => {
                if left.root() != right.root() {
                    return None;
                }
                if matches!(cmp, CmpKind::Eq | CmpKind::Ne) {
                    let equal = left.offset() == right.offset();
                    return Some(if cmp == CmpKind::Eq { equal } else { !equal });
                }
                // Relative coordinates do not establish target ordering: an
                // unknown base may place either pointer across the signed or
                // unsigned wrap. Number coordinates are actual target bits.
                if left.root() != Root::Number {
                    return None;
                }
                let numeric_unsigned = unsigned || self.ptr_size.bits() > 64;
                let (left, right) = if numeric_unsigned {
                    (
                        ConstValue::U64(self.ptr_size.mask_unsigned(left.offset().cast_unsigned())),
                        ConstValue::U64(
                            self.ptr_size.mask_unsigned(right.offset().cast_unsigned()),
                        ),
                    )
                } else {
                    (
                        ConstValue::I64(self.ptr_size.mask_signed(left.offset())),
                        ConstValue::I64(self.ptr_size.mask_signed(right.offset())),
                    )
                };
                self.compare(
                    cmp,
                    numeric_unsigned,
                    &Value::Const(left),
                    &Value::Const(right),
                )
            }

            (Value::Addr(address), Value::Const(constant))
            | (Value::Const(constant), Value::Addr(address))
                if constant.is_zero() && address.never_null() =>
            {
                match cmp {
                    CmpKind::Eq => Some(false),
                    CmpKind::Ne => Some(true),
                    CmpKind::Lt | CmpKind::Le | CmpKind::Gt | CmpKind::Ge => None,
                }
            }
            _ => None,
        }
    }

    /// Whether `condition` holds of the flags `flags` names, when the result
    /// of the instruction that set them is known.
    fn flag(
        &self,
        flags: SsaVarId,
        condition: FlagCondition,
        value: &impl Fn(SsaVarId) -> Value<T>,
    ) -> Option<bool> {
        let definition = self.ssa.recorded_definition(flags)?;
        let op = definition.op();
        let result = op.dest().filter(|dest| *dest != flags)?;
        let result = value(result);
        // A partially known result settles what its known bits do: zero is
        // ruled out by a bit known set, the sign by a known sign bit.
        if let Some(bits) = result.integer_bits() {
            let zero = match bits.value() {
                Some(value) => Some(value == 0),
                None => (bits.ones() != 0).then_some(false),
            };
            return match condition {
                FlagCondition::Zero => zero,
                FlagCondition::NotZero => zero.map(|zero| !zero),
                FlagCondition::Negative => bits.sign(),
                FlagCondition::Positive => bits.sign().map(|sign| !sign),
                FlagCondition::Carry
                | FlagCondition::NotCarry
                | FlagCondition::Overflow
                | FlagCondition::NotOverflow
                | FlagCondition::ParityEven
                | FlagCondition::ParityOdd => None,
            };
        }
        let constant = result.constant()?;
        let bits = match self.layouts {
            Some(layouts) => layouts.destination(op).bits,
            None => op.dest().and_then(|var| self.ssa.bit_width(var)),
        }
        .or_else(|| constant.bit_width(self.ptr_size))?;
        let raw = constant
            .as_u64()
            .or_else(|| constant.as_i64().map(i64::cast_unsigned))?;
        let mask = 1u64
            .checked_shl(bits)
            .map_or(u64::MAX, |bound| bound.wrapping_sub(1));
        let masked = raw & mask;
        let sign = bits
            .checked_sub(1)
            .and_then(|top| masked.checked_shr(top))
            .is_some_and(|bit| bit & 1 == 1);
        match condition {
            FlagCondition::Zero => Some(masked == 0),
            FlagCondition::NotZero => Some(masked != 0),
            FlagCondition::Negative => Some(sign),
            FlagCondition::Positive => Some(!sign),
            FlagCondition::Carry
            | FlagCondition::NotCarry
            | FlagCondition::Overflow
            | FlagCondition::NotOverflow
            | FlagCondition::ParityEven
            | FlagCondition::ParityOdd => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        analysis::execution::Address,
        ir::function::{SsaDefSpec, SsaFunctionBuilder},
        testing::{MockTarget, MockType},
    };

    /// Ends the test: a fixture the builder rejects is a broken test.
    fn abort<E, R>(_: E) -> R {
        process::abort()
    }

    /// Every value unknown.
    fn nothing(_: SsaVarId) -> Value<MockTarget> {
        Value::Unknown
    }

    /// A compare-and-branch on two values, and the function holding it.
    fn compare_block(cmp: CmpKind) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let (left, right) = builder
            .in_block(0, |b| {
                let left = b.const_i32(SsaDefSpec::tmp(MockType::I32), 0)?;
                let right = b.const_i32(SsaDefSpec::tmp(MockType::I32), 0)?;
                b.branch_cmp(left, right, cmp, false, 1, 2)?;
                Ok((left, right))
            })
            .unwrap_or_else(abort);
        for block in [1, 2] {
            builder
                .in_block(block, |b| b.ret(None))
                .unwrap_or_else(abort);
        }
        (builder.finish().unwrap_or_else(abort), left, right)
    }

    /// A known comparison is decided; an unknown one may go either way.
    #[test]
    fn a_comparison_decides_when_known() {
        let (ssa, left, right) = compare_block(CmpKind::Lt);
        let block = ssa.block(0).unwrap_or_else(|| abort(()));
        let branches = Branches::new(&ssa, PointerSize::Bit64);
        let values = |var: SsaVarId| {
            if var == left {
                Value::Const(ConstValue::I32(-1))
            } else if var == right {
                Value::Const(ConstValue::I32(3))
            } else {
                Value::Unknown
            }
        };
        assert_eq!(branches.decide(block, values), Decision::Goto(1));
        assert_eq!(
            branches.decide(block, nothing),
            Decision::Either(vec![1, 2])
        );
    }

    /// Equality uses relative offsets; a displaced stack pointer has unknown nullness.
    #[test]
    fn addresses_compare_under_one_root() {
        let (ssa, left, right) = compare_block(CmpKind::Ne);
        let block = ssa.block(0).unwrap_or_else(|| abort(()));
        let branches = Branches::new(&ssa, PointerSize::Bit64);
        let stack = |offset| Value::Addr(Address::new(Root::Stack, offset, PointerSize::Bit64));
        let same = |var: SsaVarId| {
            if var == left || var == right {
                stack(-8)
            } else {
                Value::Unknown
            }
        };
        assert_eq!(branches.decide(block, same), Decision::Goto(2));
        let null_check = |var: SsaVarId| {
            if var == left {
                stack(-8)
            } else if var == right {
                Value::Const(ConstValue::I64(0))
            } else {
                Value::Unknown
            }
        };
        assert_eq!(
            branches.decide(block, null_check),
            Decision::Either(vec![1, 2])
        );
    }

    /// A flag-tested branch reads the zero and sign of the flag-setting
    /// result, and never its carry.
    #[test]
    fn flags_decide_on_the_result_only() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let (sum, flags) = builder
            .in_block(0, |b| {
                let one = b.const_i32(SsaDefSpec::tmp(MockType::I32), 1)?;
                let (sum, flags) = b.add_with_flags(
                    SsaDefSpec::tmp(MockType::I32),
                    SsaDefSpec::tmp(MockType::I32),
                    one,
                    one,
                )?;
                b.branch_flags(flags, FlagCondition::Zero, 1, 2)?;
                Ok((sum, flags))
            })
            .unwrap_or_else(abort);
        for block in [1, 2] {
            builder
                .in_block(block, |b| b.ret(None))
                .unwrap_or_else(abort);
        }
        let ssa = builder.finish().unwrap_or_else(abort);
        let block = ssa.block(0).unwrap_or_else(|| abort(()));
        let branches = Branches::new(&ssa, PointerSize::Bit64);
        let zero = |var: SsaVarId| {
            if var == sum {
                Value::Const(ConstValue::I32(0))
            } else {
                Value::Unknown
            }
        };
        assert_eq!(branches.decide(block, zero), Decision::Goto(1));
        let negative = |var: SsaVarId| {
            if var == sum {
                Value::Const(ConstValue::I32(-5))
            } else {
                Value::Unknown
            }
        };
        assert_eq!(branches.decide(block, negative), Decision::Goto(2));
        assert_ne!(sum, flags);
    }

    /// A known switch selector goes to its case or the default; an unknown
    /// one to every distinct target.
    #[test]
    fn a_switch_decides_by_its_selector() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let selector = builder
            .in_block(0, |b| {
                let selector = b.const_i32(SsaDefSpec::tmp(MockType::I32), 0)?;
                b.switch(selector, vec![1, 2, 1], 3)?;
                Ok(selector)
            })
            .unwrap_or_else(abort);
        for block in [1, 2, 3] {
            builder
                .in_block(block, |b| b.ret(None))
                .unwrap_or_else(abort);
        }
        let ssa = builder.finish().unwrap_or_else(abort);
        let block = ssa.block(0).unwrap_or_else(|| abort(()));
        let branches = Branches::new(&ssa, PointerSize::Bit64);
        let pick = |index: u32| {
            move |var: SsaVarId| {
                if var == selector {
                    Value::Const(ConstValue::U32(index))
                } else {
                    Value::Unknown
                }
            }
        };
        assert_eq!(branches.decide(block, pick(1)), Decision::Goto(2));
        assert_eq!(branches.decide(block, pick(9)), Decision::Goto(3));
        assert_eq!(
            branches.decide(block, nothing),
            Decision::Either(vec![1, 2, 3])
        );
    }

    /// Target order compares concrete bits, while relative offsets cannot establish it.
    #[test]
    fn pointer_order_and_nullness_need_the_correct_coordinate_contract() {
        let (ssa, _, _) = compare_block(CmpKind::Lt);
        let branches = Branches::<MockTarget>::new(&ssa, PointerSize::Bit8);
        let number = |offset| Value::Addr(Address::new(Root::Number, offset, PointerSize::Bit8));
        assert_eq!(
            branches.compare(CmpKind::Lt, true, &number(0), &number(255)),
            Some(true)
        );
        assert_eq!(
            branches.compare(CmpKind::Lt, false, &number(0), &number(255)),
            Some(false)
        );
        assert_eq!(
            branches.compare(CmpKind::Gt, true, &number(255), &number(0)),
            Some(true)
        );
        for root in [
            Root::Symbol,
            Root::Stack,
            Root::StackFrame(1),
            Root::Heap(0),
            Root::Opaque {
                activation: 0,
                value: SsaVarId::from_index(0),
                run: 0,
            },
        ] {
            let first = Value::Addr(Address::new(root, 0, PointerSize::Bit8));
            let last = Value::Addr(Address::new(root, 255, PointerSize::Bit8));
            assert_eq!(branches.compare(CmpKind::Lt, true, &first, &last), None);
            assert_eq!(branches.compare(CmpKind::Lt, false, &first, &last), None);
            assert_eq!(last.truth(), None);
        }
        assert_eq!(
            Value::<MockTarget>::Addr(Address::new(Root::Heap(0), 0, PointerSize::Bit8)).truth(),
            Some(true)
        );
    }
}
