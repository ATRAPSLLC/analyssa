//! One execution of a computation over the lanes of a vector, of constant
//! inputs.
//!
//! A [`Computation`] under lane control (RISC-V's vector extension) runs its
//! kernel for the elements its restart index, active length and governing
//! predicate select, and fills every other element by its policies
//! ([`Computation::lane_action`]). [`LaneExecution`] reads one such execution's
//! constant inputs in the order the computation's runtime interface states
//! and computes its data result element by element. An element the kernel
//! does not execute is never computed, so an inactive element that would
//! trap or decide a NaN its hardware chooses folds the execution all the
//! same.
//!
//! An element whose value the policies leave unspecified (an agnostic tail
//! or inactive element, or any element not executed where the computation
//! takes no passthrough) has no single value, so an execution with one folds
//! no data result. Its status (the exceptions or the saturation its executed
//! elements report) reads the executed elements alone, so it folds all the
//! same.

use std::slice::Iter;

use crate::{
    PointerSize,
    ir::{
        float::{
            FloatBits, FloatExceptions, FloatFormat, FloatRounding, FloatRules, NanEncoding,
            RaisingArithmetic,
        },
        integer::BitWidth,
        lanes::VectorConst,
        ops::{
            execution::{
                Computation, Kernel, VectorFixedKind, VectorLaneAction, VectorMaskPrefix,
                VectorPredicateLayout, VectorRounding, VectorSlideSource, VectorStatus,
            },
            kinds::CarrylessPart,
            vector::{VectorBinaryKind, VectorCastKind, VectorCryptoKind, VectorElement},
        },
        value::ConstValue,
    },
    target::Target,
};

/// A constant input of a lane-controlled computation: a scalar, which a
/// kernel broadcasts to every element it executes, or a vector, whose
/// elements it reads at their own indices.
///
/// A vector's elements are as wide as its type declares, which a constant's
/// own lanes do not say: a vector's bits read as elements of another width
/// are other values. A kernel reads a vector only at the width its
/// declaration states, so one whose declaration is not known folds nothing.
/// So too a float's format: binary16 and bfloat16 are both 16 bits wide, and
/// a kernel reads an operand of a format other than its own elements' only
/// where its type states that format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneOperand {
    /// A scalar of `bits` bits, its value in the low bits of `value`.
    Scalar {
        /// The scalar's bits.
        value: u128,
        /// Its width.
        bits: u32,
        /// Its float format, as its type declares it, `bits` wide.
        format: Option<FloatFormat>,
    },
    /// A vector or register group.
    Vector {
        /// Its bits.
        value: VectorConst,
        /// The width of its elements, as its type declares it.
        lane_bits: Option<u32>,
        /// The format of its float elements, as its type declares it.
        lane_format: Option<FloatFormat>,
    },
}

impl LaneOperand {
    /// The operand `value` is: a vector's bits with the element width
    /// `lane_bits` its type declares, or a scalar's bits at its width, an
    /// integer's or a float's. `format` is the float format the type
    /// declares: a vector's elements', a scalar's own. `None` for a constant
    /// with no bits (a symbol, a string).
    #[must_use]
    pub fn of<T: Target>(
        value: &ConstValue<T>,
        lane_bits: Option<u32>,
        format: Option<FloatFormat>,
        ptr_size: PointerSize,
    ) -> Option<Self> {
        if let ConstValue::Vector(vector) = value {
            return Some(Self::Vector {
                value: vector.clone(),
                lane_bits,
                lane_format: format.filter(|format| lane_bits == Some(format.bits())),
            });
        }
        if let Some((value, bits)) = value.scalar_integer_pattern(ptr_size) {
            return Some(Self::Scalar {
                value,
                bits,
                format: format.filter(|format| format.bits() == bits),
            });
        }
        let float = value.read_float(ptr_size)?;
        Some(Self::Scalar {
            value: float.bits(),
            bits: float.format().bits(),
            format: Some(float.format()),
        })
    }

    /// The value of a control input: a scalar's, or a vector's low 64 bits.
    fn control(&self) -> Option<u64> {
        match self {
            Self::Scalar { value, .. } => u64::try_from(*value).ok(),
            Self::Vector { value, .. } => u64::try_from(value.bits_at(0, 64)).ok(),
        }
    }

    /// Bit `at` of a predicate held in this operand.
    fn bit(&self, at: u64) -> Option<bool> {
        let at = u32::try_from(at).ok()?;
        Some(match self {
            Self::Scalar { value, bits, .. } => {
                at < *bits && value.checked_shr(at).unwrap_or(0) & 1 != 0
            }
            Self::Vector { value, .. } => at < value.bits() && value.bits_at(at, 1) != 0,
        })
    }

    /// Element `index` of `width` bits: a vector's element at that index,
    /// where its elements are declared that wide, or a scalar's low bits
    /// whatever the index. A float scalar wider than its
    /// element is read as a narrower register reads a NaN-boxed one: its low
    /// bits where every bit above them is set, the format's canonical NaN
    /// where not.
    fn element(&self, index: u32, width: u32, format: Option<FloatFormat>) -> Option<u128> {
        let mask = BitWidth::new(width).checked_mask_u128()?;
        match self {
            Self::Vector {
                value, lane_bits, ..
            } => {
                let start = index.checked_mul(width)?;
                (*lane_bits == Some(width) && start.checked_add(width)? <= value.bits())
                    .then(|| value.bits_at(start, width))
            }
            Self::Scalar { value, bits, .. } => match format {
                Some(format) if *bits > width => {
                    let above = BitWidth::new(*bits).checked_mask_u128()? & !mask;
                    Some(if value & above == above {
                        value & mask
                    } else {
                        format.canonical_nan_bits()
                    })
                }
                _ => Some(value & mask),
            },
        }
    }
}

/// One execution of a lane-controlled [`Computation`] of constant inputs,
/// read in its runtime interface's order: the restart index and the active
/// length, the governing predicate where inactive elements are governed,
/// the rounding mode of a computation that rounds by an environment, the
/// mask-tail limit, the passthrough of its data result, then the kernel's
/// data inputs.
#[derive(Debug)]
pub struct LaneExecution<'a> {
    /// What is computed.
    computation: Computation,
    /// The first element the execution may change.
    start: u64,
    /// The active length: elements at and above it are the tail.
    length: u64,
    /// The governing predicate, where inactive elements are governed: `None`
    /// inside where it is not known.
    predicate: Option<Option<&'a LaneOperand>>,
    /// The rounding mode a computation that rounds by an environment reads.
    mode: Option<u64>,
    /// The limit below which a mask tail may hold a computed bit.
    tail_limit: u64,
    /// The old value of the data result, where the computation reads one
    /// and it is known.
    passthrough: Option<&'a LaneOperand>,
    /// The kernel's data inputs, each where it is known.
    data: &'a [Option<LaneOperand>],
    /// The rounding mode the program has set, where the host states it.
    environment: Option<FloatRounding>,
    /// The target's NaN and conversion rules.
    rules: FloatRules,
}

impl<'a> LaneExecution<'a> {
    /// Reads one execution of `computation` from `inputs`, its complete
    /// runtime inputs, each `None` where it is not known: an element that
    /// reads an unknown input folds nothing, and one that does not read it
    /// (a passthrough no element keeps) folds all the same. `environment` is
    /// the floating-point rounding mode the program has set where the host
    /// states it, and `rules` the target's.
    ///
    /// `None` for a computation that does not run under lane control, one
    /// with other than one data result, inputs of another count, or a
    /// restart index, active length, rounding mode or tail limit not known.
    #[must_use]
    pub fn of(
        computation: Computation,
        inputs: &'a [Option<LaneOperand>],
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<Self> {
        let lanes = computation.lanes?;
        if computation.data_outputs() != 1 || inputs.len() != computation.input_count() {
            return None;
        }
        let mut at = inputs.iter();
        let control = |at: &mut Iter<'a, Option<LaneOperand>>| at.next()?.as_ref()?.control();
        let start = control(&mut at)?;
        let length = control(&mut at)?;
        let predicate = match computation.inactive() {
            Some(_) => Some(at.next()?.as_ref()),
            None => None,
        };
        let mode = match computation.rounding {
            VectorRounding::FloatingEnvironment | VectorRounding::FixedEnvironment => {
                Some(control(&mut at)?)
            }
            _ => None,
        };
        let tail_limit = if lanes.mask_tail {
            control(&mut at)?
        } else {
            0
        };
        let passthrough = if lanes.no_passthrough {
            None
        } else {
            at.next()?.as_ref()
        };
        Some(Self {
            computation,
            start,
            length,
            predicate,
            mode,
            tail_limit,
            passthrough,
            data: at.as_slice(),
            environment,
            rules,
        })
    }

    /// The data result, `result_bits` wide, the execution writes: each
    /// element the kernel executes computed, each other element as its
    /// policy fills it.
    ///
    /// `None` where an element's value is unspecified or the kernel's
    /// computation of an executed element is not known (a kernel this
    /// evaluator does not compute, an operand of another shape, or a result
    /// the hardware decides).
    #[must_use]
    pub fn evaluate(&self, result_bits: u32) -> Option<VectorConst> {
        if self.computation.kernel.writes_predicate() {
            return self.evaluate_predicate(result_bits);
        }
        if let Kernel::Crypto(kind) = self.computation.kernel {
            return self.evaluate_groups(kind, result_bits);
        }
        let bits = self.computation.element.bits;
        if bits == 0 || !result_bits.is_multiple_of(bits) {
            return None;
        }
        let capacity = result_bits.checked_div(bits)?;
        let scan = Scan::of(self, capacity)?;
        let mut lanes = Vec::with_capacity(usize::try_from(capacity).ok()?);
        for index in 0..capacity {
            let action = self.action(index, &scan)?;
            lanes.push(match action {
                VectorLaneAction::Execute => match self.element(index, action, &scan)? {
                    Computed::Value(value) => value,
                    Computed::Old => self.passthrough?.element(index, bits, None)?,
                },
                VectorLaneAction::Preserve => self.passthrough?.element(index, bits, None)?,
                VectorLaneAction::Zero => 0,
                VectorLaneAction::Agnostic | VectorLaneAction::MaskTail => return None,
            });
        }
        VectorConst::from_lanes(bits, lanes)
    }

    /// The result, `result_bits` wide, of a cryptographic primitive over
    /// element groups ([`Kernel::Crypto`]): each group the policies execute
    /// is what [`VectorCryptoKind::evaluate`] computes of the inputs' groups
    /// at its index (a vector of one group, RISC-V's `.vs` forms, and a
    /// scalar, a round number, serving every group), a group 128 or 256 bits
    /// wide; each other group filled as for an element.
    fn evaluate_groups(&self, kind: VectorCryptoKind, result_bits: u32) -> Option<VectorConst> {
        let group = self.computation.element.bits;
        if !matches!(group, 128 | 256) || !result_bits.is_multiple_of(group) {
            return None;
        }
        let words_per_group = group / 64;
        let slice = |vector: &VectorConst, index: u32| {
            let start = index.checked_mul(group)?;
            (start.checked_add(group)? <= vector.bits()).then_some(())?;
            VectorConst::from_lanes(
                64,
                (0..words_per_group)
                    .map(|word| vector.bits_at(start.saturating_add(word.saturating_mul(64)), 64)),
            )
        };
        let mut words = Vec::new();
        for index in 0..result_bits.checked_div(group)? {
            match self.action(index, &Scan::None)? {
                VectorLaneAction::Execute => {
                    let inputs = self
                        .data
                        .iter()
                        .map(|input| match input.as_ref()? {
                            LaneOperand::Vector {
                                value, lane_bits, ..
                            } => {
                                let at = if value.bits() == group { 0 } else { index };
                                if *lane_bits != Some(group) && value.bits() != group {
                                    return None;
                                }
                                slice(value, at)
                            }
                            LaneOperand::Scalar { value, bits, .. } => {
                                VectorConst::from_lanes(*bits, [*value])
                            }
                        })
                        .collect::<Option<Vec<_>>>()?;
                    // RISC-V's element-group steps read four elements: words
                    // of a 128-bit group, quadwords of a 256-bit one.
                    let result = kind.evaluate(&inputs, group / 4)?.into_iter().next()?;
                    words.extend(slice(&result, 0)?.lane_values());
                }
                VectorLaneAction::Preserve => {
                    let LaneOperand::Vector { value, .. } = self.passthrough? else {
                        return None;
                    };
                    words.extend(slice(value, index)?.lane_values());
                }
                VectorLaneAction::Zero => words.extend((0..words_per_group).map(|_| 0)),
                VectorLaneAction::Agnostic | VectorLaneAction::MaskTail => return None,
            }
        }
        VectorConst::from_lanes(64, words)
    }

    /// The scalar a mask summary writes ([`Kernel::MaskSummary`]): how many
    /// selected elements of its source predicate are set, or the index of the
    /// first, all ones (`u128::MAX`, for the caller to cut to its width)
    /// where none is. It writes even at a zero active length.
    ///
    /// `None` for another kernel, a restart other than zero, or a source
    /// predicate not packed.
    #[must_use]
    pub fn summary(&self) -> Option<u128> {
        let Kernel::MaskSummary { first } = self.computation.kernel else {
            return None;
        };
        if self.start != 0 || self.computation.predicate != VectorPredicateLayout::Packed {
            return None;
        }
        let source = self.data.first()?.as_ref()?;
        let mut count = 0u128;
        for index in 0..u32::try_from(self.length).ok()? {
            if self.selected(index)? && source.bit(u64::from(index))? {
                if first {
                    return Some(u128::from(index));
                }
                count = count.checked_add(1)?;
            }
        }
        Some(if first { u128::MAX } else { count })
    }

    /// The status the execution writes beside its data result
    /// ([`Computation::status`]), whose width `result_bits` says how many
    /// elements there are: the IEEE exceptions its executed elements raise,
    /// inexact through invalid in bits zero to four, or one where any
    /// executed element saturated. An element the kernel does not execute
    /// reports nothing, so an execution that executes none reports zero.
    ///
    /// `None` for a computation that writes no status, or where what an
    /// executed element reports is not known: an input it reads that is not,
    /// a rounding mode its flags depend on that is not, or a kernel whose
    /// report this evaluator does not compute.
    #[must_use]
    pub fn status(&self, result_bits: u32) -> Option<u128> {
        let status = self.computation.status;
        let elements = if self.computation.kernel.writes_predicate() {
            result_bits
        } else {
            result_bits.checked_div(self.computation.element.bits)?
        };
        let scan = Scan::of(self, elements)?;
        let mut raised = FloatExceptions::NONE;
        let mut saturated = false;
        for index in 0..elements {
            if !matches!(self.action(index, &scan)?, VectorLaneAction::Execute) {
                continue;
            }
            match status {
                VectorStatus::None => return None,
                VectorStatus::FloatingExceptions => raised = raised.union(self.raised(index)?),
                VectorStatus::Saturation => saturated |= self.saturated(index)?,
            }
        }
        Some(match status {
            VectorStatus::None => return None,
            VectorStatus::FloatingExceptions => {
                u128::from(raised.intersection(FloatExceptions::IEEE).bits())
            }
            VectorStatus::Saturation => u128::from(saturated),
        })
    }

    /// The IEEE exceptions executed element `index` raises, by the target's
    /// rules ([`FloatRules::raises`]): those of its arithmetic and of
    /// promoting a narrower source to the computation's format. A kernel
    /// that computes no float (a move, a select, a sign operation, a
    /// classification, integer arithmetic) raises none.
    fn raised(&self, index: u32) -> Option<FloatExceptions> {
        let element = self.computation.element;
        let raises = |arithmetic, values: &[FloatBits], rounding| {
            self.rules
                .raises(arithmetic, values, rounding, FloatExceptions::IEEE)
        };
        let promoted = |inputs: &[usize]| {
            inputs
                .iter()
                .try_fold(FloatExceptions::NONE, |raised, input| {
                    Some(raised.union(self.promotion_raised(*input, index)?))
                })
        };
        match self.computation.kernel {
            Kernel::Move
            | Kernel::Select
            | Kernel::SignInject { .. }
            | Kernel::FloatClassify
            | Kernel::Slide { .. }
            | Kernel::Gather
            | Kernel::Compress
            | Kernel::Index
            | Kernel::InsertScalar
            | Kernel::MaskBinary(_)
            | Kernel::MaskPrefix(_)
            | Kernel::MaskIota => Some(FloatExceptions::NONE),
            Kernel::Unary(kind) if element.kind.is_float() => {
                let raised = raises(
                    kind.raising()?,
                    &[self.flag_float(0, index)?],
                    self.rounding(),
                )?;
                Some(raised.union(promoted(&[0])?))
            }
            Kernel::Binary(kind) | Kernel::WidenBinary { kind, .. } if element.kind.is_float() => {
                let values = [self.flag_float(0, index)?, self.flag_float(1, index)?];
                let raised = raises(kind.raising()?, &values, self.rounding())?;
                Some(raised.union(promoted(&[0, 1])?))
            }
            Kernel::FusedMultiplyAdd {
                negate_product,
                negate_addend,
            } => {
                let left = self.flag_float(0, index)?;
                let addend = self.flag_float(2, index)?;
                let values = [
                    if negate_product { left.negate() } else { left },
                    self.flag_float(1, index)?,
                    if negate_addend {
                        addend.negate()
                    } else {
                        addend
                    },
                ];
                let raised = raises(RaisingArithmetic::MulAdd, &values, self.rounding())?;
                Some(raised.union(promoted(&[0, 1, 2])?))
            }
            Kernel::Compare { signaling, .. } if element.kind.is_float() => raises(
                RaisingArithmetic::Compare { signaling },
                &[self.flag_float(0, index)?, self.flag_float(1, index)?],
                None,
            ),
            Kernel::Cast { from, to } => self.cast_raised(index, from, to),
            Kernel::FloatToInteger { unsigned, .. } => raises(
                RaisingArithmetic::ToInteger {
                    width: element.bits,
                    signed: !unsigned,
                },
                &[self.integer_source(index)?],
                self.rounding(),
            ),
            Kernel::Estimate7 { square_root } => raises(
                RaisingArithmetic::Estimate7 { square_root },
                &[self.flag_float(0, index)?],
                self.rounding(),
            ),
            Kernel::Reduce { kind, ordered } | Kernel::ReduceWide { kind, ordered }
                if element.kind.is_float() =>
            {
                let RaisingArithmetic::Binary(operation) = kind.step()?.raising()? else {
                    return None;
                };
                let own = self.format()?;
                let mut values = vec![FloatBits::new(own, self.operand(1, 0, element.bits)?)?];
                let mut raised = FloatExceptions::NONE;
                let source = if matches!(self.computation.kernel, Kernel::ReduceWide { .. }) {
                    element.bits.checked_div(2)?
                } else {
                    element.bits
                };
                for at in 0..self.capacity(0, source)? {
                    if self.selected(at)? {
                        values.push(self.flag_float(0, at)?);
                        raised = raised.union(self.promotion_raised(0, at)?);
                    }
                }
                let reduced = raises(
                    RaisingArithmetic::Reduce { operation, ordered },
                    &values,
                    self.rounding(),
                )?;
                Some(reduced.union(raised))
            }
            _ if !element.kind.is_float() => Some(FloatExceptions::NONE),
            _ => None,
        }
    }

    /// The IEEE exceptions a conversion ([`Kernel::Cast`]) of element
    /// `index` raises: between float formats as the conversion rounds, of an
    /// integer as rounding it to the float format does, and of a float to an
    /// integer as truncating it does; nothing between integers.
    fn cast_raised(
        &self,
        index: u32,
        from: VectorCastKind,
        to: VectorCastKind,
    ) -> Option<FloatExceptions> {
        let bits = self.computation.element.bits;
        let width = self.declared(0)?;
        let format_of = |kind: VectorCastKind, bits: u32| match kind {
            VectorCastKind::BFloat16 => (bits == 16).then_some(FloatFormat::BFloat16),
            _ => FloatFormat::of_width(bits),
        };
        let source = || self.source_float(0, index, format_of(from, width)?);
        match (from.is_float(), to.is_float()) {
            (false, false) => Some(FloatExceptions::NONE),
            (true, true) => self.rules.raises(
                RaisingArithmetic::Convert(format_of(to, bits)?),
                &[source()?],
                self.rounding(),
                FloatExceptions::IEEE,
            ),
            (true, false) => self.rules.raises(
                RaisingArithmetic::ToInteger {
                    width: bits,
                    signed: to == VectorCastKind::Signed,
                },
                &[source()?],
                Some(FloatRounding::TowardZero),
                FloatExceptions::IEEE,
            ),
            (false, true) => {
                let format = format_of(to, bits)?;
                let value = self.operand(0, index, width)?;
                FloatRounding::decide(self.rounding(), |rounding| {
                    Some(
                        FloatBits::from_integer_rounded(
                            format,
                            value,
                            width,
                            from == VectorCastKind::Signed,
                            rounding,
                        )?
                        .exceptions
                        .intersection(FloatExceptions::IEEE),
                    )
                })
            }
        }
    }

    /// Whether executed element `index` saturated: a saturating add or
    /// subtract whose exact result its element does not hold, or a
    /// fixed-point operation that clamps.
    fn saturated(&self, index: u32) -> Option<bool> {
        let element = self.computation.element;
        let bits = element.bits;
        match self.computation.kernel {
            Kernel::Binary(kind @ (VectorBinaryKind::SatAdd | VectorBinaryKind::SatSub)) => {
                let subtract = kind == VectorBinaryKind::SatSub;
                let (left, right) = (self.operand(0, index, bits)?, self.operand(1, index, bits)?);
                let shift = 128u32.checked_sub(bits)?;
                if element.signed? {
                    let extend =
                        |value: u128| value.checked_shl(shift)?.cast_signed().checked_shr(shift);
                    let (left, right) = (extend(left)?, extend(right)?);
                    let exact = if subtract {
                        left.checked_sub(right)
                    } else {
                        left.checked_add(right)
                    };
                    // An exact result past an i128 is past any element.
                    Some(exact.is_none_or(|exact| {
                        exact
                            .checked_shl(shift)
                            .and_then(|high| high.checked_shr(shift))
                            != Some(exact)
                    }))
                } else if subtract {
                    Some(left < right)
                } else {
                    let mask = BitWidth::new(bits).checked_mask_u128()?;
                    Some(left.checked_add(right).is_none_or(|sum| sum > mask))
                }
            }
            Kernel::Fixed { kind } => Some(self.fixed(index, kind)?.1),
            _ => None,
        }
    }

    /// Element `index` of a fixed-point operation ([`Kernel::Fixed`]) and
    /// whether it saturated ([`VectorConst::fixed_binary`]), by the rounding
    /// mode its environment input names.
    fn fixed(&self, index: u32, kind: VectorFixedKind) -> Option<(u128, bool)> {
        let element = self.computation.element;
        let bits = element.bits;
        let source = if kind == VectorFixedKind::Clip {
            bits.checked_mul(2)?
        } else {
            bits
        };
        let source = VectorConst::from_lanes(source, [self.operand(0, index, source)?])?;
        let mode = u8::try_from(self.mode?).ok()?;
        let (value, saturated) = source.fixed_binary(
            &VectorConst::from_lanes(bits, [self.operand(1, index, bits)?])?,
            kind,
            VectorElement {
                scalar: false,
                ..element
            },
            mode,
        )?;
        Some((value.lane(0), saturated))
    }

    /// The predicate result, `result_bits` wide, of a kernel that writes one
    /// bit per element ([`Kernel::writes_predicate`]): element `i`'s bit is
    /// bit `i` of the packed result, every bit of which is an element the
    /// policies fill where the kernel does not execute it.
    ///
    /// A mask tail permits its old bit, a set bit or the bit computed as if
    /// active, so it has one value only where its old bit and the computed
    /// one are both set.
    fn evaluate_predicate(&self, result_bits: u32) -> Option<VectorConst> {
        if self.computation.predicate != VectorPredicateLayout::Packed {
            return None;
        }
        let scan = Scan::of(self, result_bits)?;
        let mut bits = Vec::with_capacity(usize::try_from(result_bits).ok()?);
        for index in 0..result_bits {
            let old = || self.passthrough?.bit(u64::from(index));
            let action = self.action(index, &scan)?;
            bits.push(u128::from(match action {
                VectorLaneAction::Execute => self.predicate_bit(index, &scan)?,
                VectorLaneAction::Preserve => old()?,
                VectorLaneAction::Zero => false,
                VectorLaneAction::MaskTail if old()? && self.predicate_bit(index, &scan)? => true,
                VectorLaneAction::MaskTail | VectorLaneAction::Agnostic => return None,
            }));
        }
        VectorConst::from_lanes(1, bits)
    }

    /// Whether the governing predicate selects element `index`: set where
    /// the computation governs none.
    fn governed(&self, index: u32) -> Option<bool> {
        match self.predicate {
            Some(predicate) => predicate?.bit(self.computation.predicate_bit(u64::from(index))?),
            None => Some(true),
        }
    }

    /// The action the policies take at element `index`, with the slide
    /// offset or compressed count the kernel's scan found.
    fn action(&self, index: u32, scan: &Scan) -> Option<VectorLaneAction> {
        Some(self.computation.lane_action(
            u64::from(index),
            self.start,
            self.length,
            self.governed(index)?,
            self.tail_limit,
            scan.auxiliary(),
        ))
    }

    /// Whether element `index` is one a scan over the elements reads: below
    /// the active length, at or past the restart, and selected by the
    /// governing predicate.
    fn selected(&self, index: u32) -> Option<bool> {
        let at = u64::from(index);
        Some(at >= self.start && at < self.length && self.governed(index)?)
    }

    /// The bit a predicate-writing kernel computes at element `index`.
    fn predicate_bit(&self, index: u32, scan: &Scan) -> Option<bool> {
        let bits = self.computation.element.bits;
        let lane = VectorElement {
            scalar: false,
            ..self.computation.element
        };
        let one = |value: u128| VectorConst::from_lanes(bits, [value]);
        Some(match self.computation.kernel {
            Kernel::Compare { relation, .. } => {
                one(self.operand(0, index, bits)?)?
                    .compare(&one(self.operand(1, index, bits)?)?, relation, lane)?
                    .lane(0)
                    != 0
            }
            Kernel::MaskBinary(kind) => kind.evaluate(
                self.data.first()?.as_ref()?.bit(u64::from(index))?,
                self.data.get(1)?.as_ref()?.bit(u64::from(index))?,
            ),
            Kernel::IntegerCarry {
                subtract,
                mask_result: true,
            } => {
                let carry = self.carry(index, bits)?;
                let (left, right) = (self.operand(0, index, bits)?, self.operand(1, index, bits)?);
                // The carry out of the add, or the borrow out of the
                // subtract, at the element's width.
                if subtract {
                    left.checked_sub(right)
                        .and_then(|less| less.checked_sub(carry))
                        .is_none()
                } else {
                    let sum = left.checked_add(right)?.checked_add(carry)?;
                    sum.checked_shr(bits)? != 0
                }
            }
            Kernel::MaskPrefix(prefix) => {
                let Scan::First(first) = scan else {
                    return None;
                };
                let at = u64::from(index);
                match (prefix, first) {
                    (VectorMaskPrefix::Before, Some(first)) => at < *first,
                    (VectorMaskPrefix::Including, Some(first)) => at <= *first,
                    (VectorMaskPrefix::Only, Some(first)) => at == *first,
                    (VectorMaskPrefix::Before | VectorMaskPrefix::Including, None) => true,
                    (VectorMaskPrefix::Only, None) => false,
                }
            }
            _ => return None,
        })
    }

    /// The rounding mode a float kernel rounds by: the one its environment
    /// input names, the one it states, or the program's; `None` where the
    /// program's is not stated, for [`FloatRounding::decide`] to fold only
    /// where every mode agrees.
    fn rounding(&self) -> Option<FloatRounding> {
        match self.computation.rounding {
            VectorRounding::FloatingEnvironment => FloatRounding::of_environment(self.mode?),
            VectorRounding::TowardZero => Some(FloatRounding::TowardZero),
            VectorRounding::ToOdd => Some(FloatRounding::ToOdd),
            VectorRounding::None | VectorRounding::FixedEnvironment => self.environment,
        }
    }

    /// The format of the computation's float elements.
    fn format(&self) -> Option<FloatFormat> {
        let element = self.computation.element;
        element
            .kind
            .is_float()
            .then(|| element.kind.float_format(element.bits))?
    }

    /// Element `index` of data input `input`, `width` bits wide.
    fn operand(&self, input: usize, index: u32, width: u32) -> Option<u128> {
        let format = if self.computation.element.kind.is_float() {
            self.computation.element.kind.float_format(width)
        } else {
            None
        };
        self.data
            .get(input)?
            .as_ref()?
            .element(index, width, format)
    }

    /// The float format data input `input`'s elements hold: the one its
    /// type declares, or, where it declares none, the format of `own`, the
    /// computation's elements, where it is as wide as they are. A scalar
    /// wider than them is read at their width, as a narrower register reads
    /// a NaN-boxed one.
    fn source(&self, input: usize, own: FloatFormat) -> Option<FloatFormat> {
        match self.data.get(input)?.as_ref()? {
            LaneOperand::Vector {
                lane_bits,
                lane_format,
                ..
            } => lane_format.or_else(|| (*lane_bits == Some(own.bits())).then_some(own)),
            LaneOperand::Scalar { bits, format, .. } => {
                if *bits > own.bits() {
                    Some(own)
                } else {
                    format.or_else(|| (*bits == own.bits()).then_some(own))
                }
            }
        }
    }

    /// Element `index` of data input `input` as a float of its own format
    /// ([`Self::source`]), the computation's elements' being `own`.
    fn source_float(&self, input: usize, index: u32, own: FloatFormat) -> Option<FloatBits> {
        let format = self.source(input, own)?;
        let operand = self.data.get(input)?.as_ref()?;
        FloatBits::new(format, operand.element(index, format.bits(), Some(format))?)
    }

    /// The float of data input `input` at element `index`, in the format of
    /// the computation's elements: a narrower source promoted to it exactly,
    /// as a widening operation reads it.
    fn float(&self, input: usize, index: u32) -> Option<FloatBits> {
        let own = self.format()?;
        let value = self.source_float(input, index, own)?;
        if value.format() == own {
            return Some(value);
        }
        if value.format().bits() >= own.bits() {
            return None;
        }
        let kind = |format: FloatFormat| {
            if format == FloatFormat::BFloat16 {
                VectorCastKind::BFloat16
            } else {
                VectorCastKind::Float
            }
        };
        FloatBits::new(
            own,
            self.convert(
                value.bits(),
                (kind(value.format()), value.format().bits()),
                (kind(own), own.bits()),
            )?,
        )
    }

    /// The float of data input `input` at element `index` as the arithmetic
    /// whose flags [`Self::raised`] reads takes it: promoted as
    /// [`Self::float`] promotes it, a NaN quiet and of any payload, which no
    /// flag reads. A target whose NaNs are not encoded as IEEE's states
    /// their kind otherwise, so a promoted NaN of one folds no flag.
    fn flag_float(&self, input: usize, index: u32) -> Option<FloatBits> {
        let own = self.format()?;
        let value = self.source_float(input, index, own)?;
        if value.format() == own {
            return Some(value);
        }
        if value.format().bits() >= own.bits()
            || value.is_nan() && self.rules.nan_encoding != NanEncoding::Ieee
        {
            return None;
        }
        Some(
            value
                .convert_rounded(own, FloatRounding::NearestEven, true)?
                .value,
        )
    }

    /// The exceptions promoting data input `input`'s element `index` to the
    /// computation's format raises ([`Self::float`]): invalid for a
    /// signalling NaN, nothing where it is of that format already.
    fn promotion_raised(&self, input: usize, index: u32) -> Option<FloatExceptions> {
        let own = self.format()?;
        let value = self.source_float(input, index, own)?;
        if value.format() == own {
            return Some(FloatExceptions::NONE);
        }
        self.rules.raises(
            RaisingArithmetic::Convert(own),
            &[value],
            self.rounding(),
            FloatExceptions::IEEE,
        )
    }

    /// The format a kernel that reads float elements and writes integers
    /// (a classification, a conversion to an integer) reads its source in:
    /// the one the source's type declares, or the binary format of the
    /// elements' width.
    fn integer_source(&self, index: u32) -> Option<FloatBits> {
        self.source_float(
            0,
            index,
            FloatFormat::of_width(self.computation.element.bits)?,
        )
    }

    /// The element the kernel computes at `index`, which `action` executes,
    /// or the old element where a slide keeps it.
    fn element(&self, index: u32, action: VectorLaneAction, scan: &Scan) -> Option<Computed> {
        let element = self.computation.element;
        let bits = element.bits;
        let lane = VectorElement {
            scalar: false,
            ..element
        };
        let one = |value: u128| VectorConst::from_lanes(bits, [value]);
        let canonical = self.computation.canonicalizes_nan(action);
        let computed = match self.computation.kernel {
            Kernel::Move => self.operand(0, index, bits)?,
            Kernel::Unary(kind) => one(self.operand(0, index, bits)?)?
                .unary(kind, lane, self.rounding(), self.rules)?
                .lane(0),
            // Float arithmetic reads its operands as floats, a narrower one
            // promoted; bitwise operations and scales read the bits.
            Kernel::Binary(kind) if element.kind.is_float() && kind.raising().is_some() => {
                one(self.float(0, index)?.bits())?
                    .binary(
                        &one(self.float(1, index)?.bits())?,
                        kind,
                        lane,
                        self.rounding(),
                        self.rules,
                    )?
                    .lane(0)
            }
            Kernel::Binary(kind) => one(self.operand(0, index, bits)?)?
                .binary(
                    &one(self.operand(1, index, bits)?)?,
                    kind,
                    lane,
                    self.rounding(),
                    self.rules,
                )?
                .lane(0),
            Kernel::IntegerDivide {
                remainder,
                unsigned,
            } => Division::of(
                self.operand(0, index, bits)?,
                self.operand(1, index, bits)?,
                bits,
                !unsigned,
            )?
            .result(remainder),
            Kernel::ModuloShift { kind } => {
                let count = self.operand(1, index, bits)? & u128::from(bits.checked_sub(1)?);
                one(self.operand(0, index, bits)?)?
                    .binary(&one(count)?, kind, lane, None, self.rules)?
                    .lane(0)
            }
            Kernel::MultiplyHigh {
                signed_left,
                signed_right,
            } => multiply_high(
                self.operand(0, index, bits)?,
                self.operand(1, index, bits)?,
                bits,
                (signed_left, signed_right),
            )?,
            Kernel::IntegerMultiplyAdd { subtract_product } => one(self.operand(0, index, bits)?)?
                .integer_multiply_add(
                    &one(self.operand(1, index, bits)?)?,
                    &one(self.operand(2, index, bits)?)?,
                    bits,
                    (false, false),
                    subtract_product,
                )?
                .lane(0),
            Kernel::Fixed { kind } => self.fixed(index, kind)?.0,
            Kernel::SignInject { negate, xor } => self
                .float(0, index)?
                .inject_sign(self.float(1, index)?, negate, xor)?
                .bits(),
            Kernel::Select => {
                let selector = self.computation.predicate.bit(u64::from(index), bits)?;
                let input = if self.data.get(2)?.as_ref()?.bit(selector)? {
                    1
                } else {
                    0
                };
                self.operand(input, index, bits)?
            }
            Kernel::FusedMultiplyAdd {
                negate_product,
                negate_addend,
            } => {
                let left = self.float(0, index)?;
                let left = if negate_product { left.negate() } else { left };
                let addend = self.float(2, index)?;
                let addend = if negate_addend {
                    addend.negate()
                } else {
                    addend
                };
                let right = self.float(1, index)?;
                FloatRounding::decide(self.rounding(), |rounding| {
                    Some(
                        left.mul_add_rounded(right, addend, rounding, canonical)?
                            .value,
                    )
                })?
                .bits()
            }
            Kernel::FloatClassify => u128::from(self.integer_source(index)?.classify()),
            Kernel::FloatToInteger {
                unsigned,
                nan_maximum,
            } => {
                let float = self.integer_source(index)?;
                FloatRounding::decide(self.rounding(), |rounding| {
                    Some(
                        float
                            .to_integer_clamped(bits, !unsigned, nan_maximum, rounding)?
                            .bits,
                    )
                })?
            }
            Kernel::Estimate7 { square_root } => {
                let float = self.float(0, index)?;
                FloatRounding::decide(self.rounding(), |rounding| {
                    Some(float.estimate7(square_root, rounding, canonical)?.value)
                })?
                .bits()
            }
            // A part as wide as the element; the whole product, twice as
            // wide, is no element.
            Kernel::CarrylessMultiply { part } if part != CarrylessPart::Full => self
                .computation
                .kernel
                .evaluate_lanes(
                    &[
                        one(self.operand(0, index, bits)?)?,
                        one(self.operand(1, index, bits)?)?,
                    ],
                    bits,
                )?
                .lane(0),
            Kernel::Index => u128::from(index) & BitWidth::new(bits).checked_mask_u128()?,
            Kernel::InsertScalar => self.operand(0, 0, bits)?,
            Kernel::Gather => {
                let at = self.operand(1, index, self.declared(1)?)?;
                match u32::try_from(at) {
                    Ok(at) if at < self.capacity(0, bits)? => self.operand(0, at, bits)?,
                    // An index past the source's capacity reads zero.
                    _ => 0,
                }
            }
            Kernel::Slide { scalar_fill, .. } => {
                let offset = if scalar_fill {
                    0
                } else {
                    self.data.get(1)?.as_ref()?.control()?
                };
                match self.computation.slide_source(
                    u64::from(index),
                    self.length,
                    u64::from(self.capacity(0, bits)?),
                    offset,
                )? {
                    VectorSlideSource::Element(at) => {
                        self.operand(0, u32::try_from(at).ok()?, bits)?
                    }
                    VectorSlideSource::Scalar => self.operand(1, 0, bits)?,
                    VectorSlideSource::Zero => 0,
                    VectorSlideSource::Preserve => return Some(Computed::Old),
                }
            }
            Kernel::Compress => {
                let Scan::Compressed(selected) = scan else {
                    return None;
                };
                self.operand(0, *selected.get(usize::try_from(index).ok()?)?, bits)?
            }
            Kernel::Reduce { kind, ordered } | Kernel::ReduceWide { kind, ordered } => {
                let wide = matches!(self.computation.kernel, Kernel::ReduceWide { .. });
                let source = if wide {
                    bits.checked_div(2).filter(|_| bits.is_multiple_of(2))?
                } else {
                    bits
                };
                let seed = self.operand(1, 0, bits)?;
                let mut values = Vec::new();
                for at in 0..self.capacity(0, source)? {
                    if !self.selected(at)? {
                        continue;
                    }
                    values.push(if element.kind.is_float() {
                        self.float(0, at)?.bits()
                    } else if source == bits {
                        self.operand(0, at, source)?
                    } else {
                        let class = VectorCastKind::integer(element.signed == Some(true));
                        self.convert(self.operand(0, at, source)?, (class, source), (class, bits))?
                    });
                }
                // An empty selection copies the seed's bits as they are.
                if values.is_empty() {
                    return Some(Computed::Value(seed));
                }
                VectorConst::from_lanes(bits, values)?.reduce(
                    kind,
                    lane,
                    Some(seed),
                    ordered,
                    self.rounding(),
                    self.rules,
                )?
            }
            Kernel::MaskIota => {
                let Scan::Counts(counts) = scan else {
                    return None;
                };
                *counts.get(usize::try_from(index).ok()?)?
                    & BitWidth::new(bits).checked_mask_u128()?
            }
            Kernel::Extend { signed } => {
                let from = self.declared(0)?;
                self.convert(
                    self.operand(0, index, from)?,
                    (VectorCastKind::integer(signed), from),
                    (VectorCastKind::integer(signed), bits),
                )?
            }
            Kernel::Cast { from, to } => {
                let width = self.declared(0)?;
                self.convert(self.operand(0, index, width)?, (from, width), (to, bits))?
            }
            Kernel::WidenBinary {
                wide_left,
                kind,
                signed_left,
                signed_right,
            } => {
                let narrow = bits.checked_div(2).filter(|_| bits.is_multiple_of(2))?;
                let widened = |input: usize, width: u32, signed: bool| -> Option<u128> {
                    if element.kind.is_float() {
                        return Some(self.float(input, index)?.bits());
                    }
                    let value = self.operand(input, index, width)?;
                    if width == bits {
                        return Some(value);
                    }
                    let class = VectorCastKind::integer(signed);
                    self.convert(value, (class, width), (class, bits))
                };
                let left = widened(0, if wide_left { bits } else { narrow }, signed_left)?;
                let right = widened(1, narrow, signed_right)?;
                // A widening logical shift reduces its count modulo the
                // result's width.
                let right = if kind == VectorBinaryKind::Shl {
                    right & u128::from(bits.checked_sub(1)?)
                } else {
                    right
                };
                one(left)?
                    .binary(&one(right)?, kind, lane, self.rounding(), self.rules)?
                    .lane(0)
            }
            Kernel::NarrowShift { unsigned } => {
                let wide = bits.checked_mul(2)?;
                let source = self.operand(0, index, wide)?;
                let count =
                    u32::try_from(self.operand(1, index, bits)? & u128::from(wide.checked_sub(1)?))
                        .ok()?;
                let shifted = if unsigned {
                    source.checked_shr(count)?
                } else {
                    let shift = 128u32.checked_sub(wide)?;
                    source
                        .checked_shl(shift)?
                        .cast_signed()
                        .checked_shr(shift.checked_add(count)?)?
                        .cast_unsigned()
                };
                shifted & BitWidth::new(bits).checked_mask_u128()?
            }
            Kernel::WidenMultiplyAdd {
                signed_left,
                signed_right,
            } => {
                let narrow = bits.checked_div(2).filter(|_| bits.is_multiple_of(2))?;
                let factor = |input: usize| {
                    VectorConst::from_lanes(narrow, [self.operand(input, index, narrow)?])
                };
                factor(0)?
                    .integer_multiply_add(
                        &factor(1)?,
                        &one(self.operand(2, index, bits)?)?,
                        bits,
                        (signed_left, signed_right),
                        false,
                    )?
                    .lane(0)
            }
            Kernel::IntegerCarry {
                subtract,
                mask_result: false,
            } => {
                let carry = self.carry(index, bits)?;
                let (left, right) = (self.operand(0, index, bits)?, self.operand(1, index, bits)?);
                let value = if subtract {
                    left.wrapping_sub(right).wrapping_sub(carry)
                } else {
                    left.wrapping_add(right).wrapping_add(carry)
                };
                value & BitWidth::new(bits).checked_mask_u128()?
            }
            _ => return None,
        };
        Some(Computed::Value(self.canonicalized(computed, canonical)))
    }

    /// How many elements `width` bits wide data input `input` holds: a
    /// vector's capacity, which a gather or a slide reads up to whatever the
    /// active length.
    fn capacity(&self, input: usize, width: u32) -> Option<u32> {
        match self.data.get(input)?.as_ref()? {
            LaneOperand::Vector { value, .. } => value.bits().checked_div(width),
            LaneOperand::Scalar { .. } => None,
        }
    }

    /// The element width data input `input` holds: a vector's as its type
    /// declares it, a scalar's own.
    fn declared(&self, input: usize) -> Option<u32> {
        match self.data.get(input)?.as_ref()? {
            LaneOperand::Scalar { bits, .. } => Some(*bits),
            LaneOperand::Vector { lane_bits, .. } => *lane_bits,
        }
    }

    /// `value`, of the class and width `from` names, converted to `to`'s, as
    /// [`VectorConst::cast`] converts a lane.
    fn convert(
        &self,
        value: u128,
        from: (VectorCastKind, u32),
        to: (VectorCastKind, u32),
    ) -> Option<u128> {
        Some(
            VectorConst::from_lanes(from.1, [value])?
                .cast(from.0, to.0, self.rounding(), to.1, 1, self.rules)?
                .lane(0),
        )
    }

    /// The carry or borrow bit element `index` of an add or subtract with
    /// carry reads: its bit of the third data input, which holds one per
    /// element where a vector, one for all where a scalar.
    fn carry(&self, index: u32, bits: u32) -> Option<u128> {
        let carry = self.data.get(2)?.as_ref()?;
        let at = match carry {
            LaneOperand::Scalar { .. } => 0,
            LaneOperand::Vector { .. } => self.computation.predicate.bit(u64::from(index), bits)?,
        };
        Some(u128::from(carry.bit(at)?))
    }

    /// `value` with a NaN replaced by its format's canonical NaN, where the
    /// computation canonicalizes the NaNs it computes.
    fn canonicalized(&self, value: u128, canonical: bool) -> u128 {
        match self.format() {
            Some(format)
                if canonical && FloatBits::new(format, value).is_some_and(FloatBits::is_nan) =>
            {
                format.canonical_nan_bits()
            }
            _ => value,
        }
    }
}

/// What an executed element holds: the value the kernel computes, or the
/// old element, which an upward slide keeps below its offset.
enum Computed {
    /// The computed value.
    Value(u128),
    /// The passthrough's element.
    Old,
}

/// What a kernel that reads across elements learns in one pass over the
/// selected elements of its source.
enum Scan {
    /// The kernel reads each element alone.
    None,
    /// The count of selected set source bits below each element
    /// ([`Kernel::MaskIota`]).
    Counts(Vec<u128>),
    /// The first selected element whose source bit is set
    /// ([`Kernel::MaskPrefix`]).
    First(Option<u64>),
    /// The source elements a compression keeps, in order
    /// ([`Kernel::Compress`]).
    Compressed(Vec<u32>),
    /// An upward slide's offset, below which it keeps the old elements.
    Offset(u64),
}

impl Scan {
    /// The scan `execution`'s kernel needs over its first `elements`
    /// elements. A prefix or an iota requires a restart of zero and reads a
    /// packed source, bit `i` element `i`'s.
    fn of(execution: &LaneExecution<'_>, elements: u32) -> Option<Self> {
        let kernel = execution.computation.kernel;
        if let Kernel::Slide {
            up: true,
            scalar_fill: false,
        } = kernel
        {
            return Some(Self::Offset(execution.data.get(1)?.as_ref()?.control()?));
        }
        if kernel == Kernel::Compress {
            // The selector, the second data input, picks among the source
            // elements below the active length; there is no governing mask.
            let selector = execution.data.get(1)?.as_ref()?;
            let mut kept = Vec::new();
            for index in 0..u32::try_from(execution.length.min(u64::from(elements))).ok()? {
                if selector.bit(u64::from(index))? {
                    kept.push(index);
                }
            }
            return Some(Self::Compressed(kept));
        }
        if !matches!(kernel, Kernel::MaskIota | Kernel::MaskPrefix(_)) {
            return Some(Self::None);
        }
        if execution.start != 0 || execution.computation.predicate != VectorPredicateLayout::Packed
        {
            return None;
        }
        let source = |index: u32| -> Option<bool> {
            Some(
                execution.selected(index)?
                    && execution.data.first()?.as_ref()?.bit(u64::from(index))?,
            )
        };
        if matches!(kernel, Kernel::MaskPrefix(_)) {
            for index in 0..elements {
                if source(index)? {
                    return Some(Self::First(Some(u64::from(index))));
                }
            }
            return Some(Self::First(None));
        }
        let mut counts = Vec::with_capacity(usize::try_from(elements).ok()?);
        let mut count = 0u128;
        for index in 0..elements {
            counts.push(count);
            // An element past the active length reads no source bit.
            if u64::from(index) < execution.length && source(index)? {
                count = count.checked_add(1)?;
            }
        }
        Some(Self::Counts(counts))
    }

    /// The auxiliary [`Computation::lane_action`] reads: an upward slide's
    /// offset, or how many elements a compression keeps.
    fn auxiliary(&self) -> u64 {
        match self {
            Self::Offset(offset) => *offset,
            Self::Compressed(kept) => u64::try_from(kept.len()).unwrap_or(u64::MAX),
            Self::None | Self::Counts(_) | Self::First(_) => 0,
        }
    }
}

/// An integer division with total semantics, as a vector extension states
/// it: a quotient of all ones and a remainder of the dividend for a zero
/// divisor, and for the least signed dividend over minus one a quotient of
/// the dividend and a remainder of zero; otherwise a quotient truncated
/// toward zero.
struct Division {
    /// The quotient's bits.
    quotient: u128,
    /// The remainder's bits.
    remainder: u128,
}

impl Division {
    /// The division of the `bits`-wide `dividend` by `divisor`, read signed
    /// where `signed` says.
    fn of(dividend: u128, divisor: u128, bits: u32, signed: bool) -> Option<Self> {
        if !(1..=64).contains(&bits) {
            return None;
        }
        let shift = 128u32.checked_sub(bits)?;
        let mask = BitWidth::new(bits).checked_mask_u128()?;
        if divisor & mask == 0 {
            return Some(Self {
                quotient: mask,
                remainder: dividend & mask,
            });
        }
        if !signed {
            return Some(Self {
                quotient: (dividend & mask).checked_div(divisor & mask)?,
                remainder: (dividend & mask).checked_rem(divisor & mask)?,
            });
        }
        let left = dividend
            .checked_shl(shift)?
            .cast_signed()
            .checked_shr(shift)?;
        let right = divisor
            .checked_shl(shift)?
            .cast_signed()
            .checked_shr(shift)?;
        // At most 64 bits wide, the least dividend over minus one does not
        // overflow an i128; it truncates to the dividend at the width.
        Some(Self {
            quotient: left.checked_div(right)?.cast_unsigned() & mask,
            remainder: left.checked_rem(right)?.cast_unsigned() & mask,
        })
    }

    /// The quotient, or the remainder where `remainder` says.
    const fn result(&self, remainder: bool) -> u128 {
        if remainder {
            self.remainder
        } else {
            self.quotient
        }
    }
}

/// The high `bits` of the double-width product of `left` and `right`, each
/// read signed where `signed` says.
fn multiply_high(left: u128, right: u128, bits: u32, signed: (bool, bool)) -> Option<u128> {
    if !(1..=64).contains(&bits) {
        return None;
    }
    let shift = 128u32.checked_sub(bits)?;
    let mask = BitWidth::new(bits).checked_mask_u128()?;
    if !signed.0 && !signed.1 {
        // Two unsigned 64-bit factors overflow an i128, never a u128.
        let product = (left & mask).checked_mul(right & mask)?;
        return Some(product.checked_shr(bits)? & mask);
    }
    // A signed factor is at most 2^63 in magnitude, so a product with one
    // fits an i128.
    let extend = |value: u128, signed: bool| -> Option<i128> {
        Some(if signed {
            value.checked_shl(shift)?.cast_signed().checked_shr(shift)?
        } else {
            (value & mask).cast_signed()
        })
    };
    let product = extend(left, signed.0)?.checked_mul(extend(right, signed.1)?)?;
    Some(product.checked_shr(bits)?.cast_unsigned() & mask)
}

#[cfg(test)]
mod tests;
