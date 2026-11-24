//! Floating exception status of an operation whose numeric result is unknown.
//!
//! A `float.raise` states which IEEE exceptions an operation raises, and that is
//! often decided when the rounded value is not: invalid for a signaling NaN
//! operand, or the same inexact status under every rounding mode. Each active
//! lane is evaluated under every possible rounding mode and only the status
//! bits all of them agree on are folded. Inactive lanes neither compute nor
//! raise. Target NaN encodings and flush modes follow the ordinary folding rules.

use crate::{
    PointerSize,
    analysis::consts::{Destination, OperandLayout, TypeQueries},
    ir::{
        float::{
            FloatBinary, FloatBits, FloatFormat, FloatRounding, FloatRules, RaisingArithmetic,
        },
        integer::{BitWidth, IntegerType},
        lanes::VectorConst,
        ops::{SsaOp, VectorElement, VectorElementKind},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// The fixed operands and semantics of one supported floating computation.
struct Computation {
    /// Scalar or lane inputs; absent entries are never inspected.
    inputs: [Option<SsaVarId>; 3],
    /// Explicit geometry for a lane computation.
    element: Option<VectorElement>,
    /// The canonical target-aware exception operation.
    arithmetic: RaisingArithmetic,
    /// A predicated instruction's own execution mask.
    mask: Option<SsaVarId>,
    /// An explicit scalar rounding override.
    rounding: Option<FloatRounding>,
    /// The environment's rounding mode the computation reads, where it rounds
    /// by it ([`SsaOp::environment`]).
    environment: Option<SsaVarId>,
    /// Whether the fused product changes sign before addition.
    negate_product: bool,
    /// Whether the fused addend changes sign before addition.
    negate_addend: bool,
}

impl Computation {
    /// Selects explicit semantics using the supplied target projection.
    fn of<T: Target>(operation: &SsaOp<T>, types: &TypeQueries<'_, T>) -> Option<Self> {
        let mut computation = Self {
            inputs: [None; 3],
            element: None,
            arithmetic: RaisingArithmetic::Sqrt,
            mask: None,
            rounding: None,
            environment: operation.environment(),
            negate_product: false,
            negate_addend: false,
        };
        let (inputs, arithmetic) = match operation {
            SsaOp::Add { left, right, .. } => (
                [Some(*left), Some(*right), None],
                RaisingArithmetic::Binary(FloatBinary::Add),
            ),
            SsaOp::Sub { left, right, .. } => (
                [Some(*left), Some(*right), None],
                RaisingArithmetic::Binary(FloatBinary::Subtract),
            ),
            SsaOp::Mul { left, right, .. } => (
                [Some(*left), Some(*right), None],
                RaisingArithmetic::Binary(FloatBinary::Multiply),
            ),
            SsaOp::Div { left, right, .. } => (
                [Some(*left), Some(*right), None],
                RaisingArithmetic::Binary(FloatBinary::Divide),
            ),
            SsaOp::Ceq {
                left,
                right,
                signaling,
                ..
            }
            | SsaOp::Clt {
                left,
                right,
                signaling,
                ..
            }
            | SsaOp::Cgt {
                left,
                right,
                signaling,
                ..
            }
            | SsaOp::FloatCompareFlags {
                left,
                right,
                signaling,
                ..
            } => (
                [Some(*left), Some(*right), None],
                RaisingArithmetic::Compare {
                    signaling: *signaling,
                },
            ),
            SsaOp::VectorBinary {
                left,
                right,
                kind,
                element,
                ..
            }
            | SsaOp::VectorPredicatedBinary {
                left,
                right,
                kind,
                element,
                ..
            } => {
                computation.element = Some(*element);
                if let SsaOp::VectorPredicatedBinary { mask, .. } = operation {
                    computation.mask = Some(*mask);
                }
                ([Some(*left), Some(*right), None], kind.raising()?)
            }
            SsaOp::VectorCompare {
                left,
                right,
                element,
                signaling,
                ..
            } => {
                computation.element = Some(*element);
                (
                    [Some(*left), Some(*right), None],
                    RaisingArithmetic::Compare {
                        signaling: *signaling,
                    },
                )
            }
            SsaOp::VectorUnary {
                value,
                kind,
                element,
                ..
            } => {
                computation.element = Some(*element);
                ([Some(*value), None, None], kind.raising()?)
            }
            SsaOp::VectorTernary {
                first,
                second,
                third,
                kind,
                element,
                ..
            } => {
                let (negate_product, negate_addend) = kind.fused_signs()?;
                computation.element = Some(*element);
                computation.negate_product = negate_product;
                computation.negate_addend = negate_addend;
                (
                    [Some(*first), Some(*second), Some(*third)],
                    RaisingArithmetic::MulAdd,
                )
            }
            SsaOp::FloatToInt {
                operand,
                target,
                unsigned,
                ..
            } => (
                [Some(*operand), None, None],
                RaisingArithmetic::ToInteger {
                    width: types.integer_type(target)?.bits(),
                    signed: !*unsigned,
                },
            ),
            SsaOp::FloatConv {
                operand,
                target,
                rounding,
                ..
            } => {
                computation.rounding = *rounding;
                (
                    [Some(*operand), None, None],
                    RaisingArithmetic::Convert(types.float_format(target)?),
                )
            }
            _ => return None,
        };
        computation.inputs = inputs;
        computation.arithmetic = arithmetic;
        Some(computation)
    }
}

/// A declared integer bitset or a packed canonical predicate, without reallocating it.
pub enum Mask {
    /// One integer bit per computed lane, through 128 lanes.
    Bits(u128),
    /// Canonical zero or all-ones predicate lanes.
    Predicate(VectorConst),
}

impl Mask {
    /// Retains the declared mask interpretation even when its carrier is a vector.
    pub fn read<T: Target>(
        (value, layout): (ConstValue<T>, OperandLayout),
        ptr_size: PointerSize,
    ) -> Option<Self> {
        match layout {
            OperandLayout::Boolean => {
                Some(Self::Bits(u128::from(value.logical_boolean(ptr_size)?)))
            }
            OperandLayout::Float(_) => None,
            OperandLayout::Register(dest) if dest.integer && dest.lanes.is_none() => {
                Some(Self::Bits(
                    layout
                        .read_comparison(value, ptr_size)?
                        .scalar_integer_pattern(ptr_size)?
                        .0,
                ))
            }
            OperandLayout::Register(_) => match value {
                ConstValue::Vector(vector) => Some(Self::Predicate(vector)),
                value => Some(Self::Bits(value.scalar_integer_pattern(ptr_size)?.0)),
            },
        }
    }

    /// Returns whether no lane is active, reading predicate storage 128 bits at a time.
    pub fn empty(&self) -> Option<bool> {
        match self {
            Self::Bits(bits) => Some(*bits == 0),
            Self::Predicate(vector) => {
                for bit in (0..vector.bits()).step_by(128) {
                    if vector.bits_at(bit, vector.bits().saturating_sub(bit).min(128)) != 0 {
                        return Some(false);
                    }
                }
                Some(true)
            }
        }
    }

    /// Reads one bitset element or one canonical predicate lane.
    pub fn active(&self, lane: u32, width: u32, count: u32) -> Option<bool> {
        match self {
            Self::Bits(bits) => Some(bits.checked_shr(lane).unwrap_or(0) & 1 != 0),
            Self::Predicate(vector) => {
                if vector.lanes() < count || vector.lane_bits() != 1 && vector.lane_bits() != width
                {
                    return None;
                }
                let value = vector.lane(lane);
                if value == 0 {
                    Some(false)
                } else if value == BitWidth::new(vector.lane_bits()).checked_mask_u128()? {
                    Some(true)
                } else {
                    None
                }
            }
        }
    }
}

/// Reads an absent mask as all active lanes without changing packed predicates.
fn active(mask: Option<&Mask>, lane: u32, width: u32, count: u32) -> Option<bool> {
    mask.map_or(Some(true), |mask| mask.active(lane, width, count))
}

/// Returns the complete numeric carrier extent, including floating and packed values.
fn carrier_bits<T: Target>(value: &ConstValue<T>, ptr_size: PointerSize) -> Option<u32> {
    match value {
        ConstValue::Float(float) => Some(float.format().bits()),
        ConstValue::Vector(vector) => Some(vector.bits()),
        _ => value.bit_width(ptr_size),
    }
}

/// Reads one float lane directly from borrowed bits without constructing lane vectors.
fn lane<T: Target>(
    value: &ConstValue<T>,
    index: u32,
    format: FloatFormat,
    ptr_size: PointerSize,
) -> Option<FloatBits> {
    let start = index.checked_mul(format.bits())?;
    if start.checked_add(format.bits())? > carrier_bits(value, ptr_size)? {
        return None;
    }
    let bits = match value {
        ConstValue::Vector(vector) => vector.bits_at(start, format.bits()),
        ConstValue::Float(value) => value.bits().checked_shr(start)?,
        _ => value
            .scalar_integer_pattern(ptr_size)?
            .0
            .checked_shr(start)?,
    };
    let mask = BitWidth::new(format.bits()).checked_mask_u128()?;
    FloatBits::new(format, bits & mask)
}

/// Evaluates supported scalar and lane computations.
///
/// Agreement compares requested register bits across rounding modes, rather than
/// discarded numeric values. The instruction mask and outer mask intersect; inactive
/// lanes neither compute nor contribute exceptions. The minimum and maximum selects
/// ([`crate::ir::ops::VectorBinaryKind::MinLess`] and
/// [`crate::ir::ops::VectorBinaryKind::MaxGreater`]) use signaling comparisons:
/// any active NaN raises invalid, regardless of the operand the select returns.
/// Target NaN encodings, flush modes and rounded-magnitude status use the same
/// floating rules as ordinary folding. Unknown rounding is joined after the
/// active lanes are combined, so only observed status bits need agree.
pub fn fold<T: Target>(
    raise: &SsaOp<T>,
    destination: Destination,
    get_value: &mut impl FnMut(SsaVarId) -> Option<(ConstValue<T>, OperandLayout)>,
    ptr_size: PointerSize,
    rules: FloatRules,
    types: &TypeQueries<'_, T>,
) -> Option<ConstValue<T>> {
    let SsaOp::FloatRaise {
        op,
        lanes,
        raises,
        layout,
        ..
    } = raise
    else {
        return None;
    };
    let (operation, lanes, raises, layout) = (op.as_ref(), *lanes, *raises, *layout);
    let raises = raises.intersection(layout.holds());
    if raises.is_empty() {
        return result(0, destination, ptr_size);
    }
    let outer = match lanes {
        Some(mask) => Some(Mask::read(get_value(mask)?, ptr_size)?),
        None => None,
    };
    if let Some(mask) = &outer
        && mask.empty()?
    {
        return result(0, destination, ptr_size);
    }
    let computation = Computation::of(operation, types)?;
    let inner = match computation.mask {
        Some(mask) => Some(Mask::read(get_value(mask)?, ptr_size)?),
        None => None,
    };
    if let Some(mask) = &inner
        && mask.empty()?
    {
        return result(0, destination, ptr_size);
    }
    if computation
        .element
        .is_some_and(|element| element.kind == VectorElementKind::Integer)
    {
        return result(0, destination, ptr_size);
    }
    let numeric =
        |(value, layout): (ConstValue<T>, OperandLayout)| layout.read_numeric(value, ptr_size);
    let mut values = [None, None, None];
    for (place, variable) in values.iter_mut().zip(computation.inputs) {
        if let Some(variable) = variable {
            *place = Some(numeric(get_value(variable)?)?);
        }
    }
    let left = values[0].as_ref()?;
    let (format, count, rounding) = match computation.element {
        Some(element) => {
            let bits = if element.bits != 0 {
                element.bits
            } else {
                match left {
                    ConstValue::Vector(vector) => vector.lane_bits(),
                    ConstValue::Float(float) => float.format().bits(),
                    _ => return None,
                }
            };
            let format = match element.kind {
                VectorElementKind::Float => FloatFormat::of_width(bits)?,
                VectorElementKind::BFloat16 if bits == 16 => FloatFormat::BFloat16,
                _ => return None,
            };
            let left_bits = carrier_bits(left, ptr_size)?;
            let valid_sources = values.iter().flatten().all(|value| {
                carrier_bits(value, ptr_size).is_some_and(|bits| {
                    bits >= format.bits()
                        && bits.is_multiple_of(format.bits())
                        && (element.scalar || bits == left_bits)
                })
            });
            if left_bits < bits || !left_bits.is_multiple_of(bits) || !valid_sources {
                return None;
            }
            (
                format,
                if element.scalar {
                    1
                } else {
                    left_bits.checked_div(bits)?
                },
                element.rounding,
            )
        }
        None => {
            let ConstValue::Float(left) = left else {
                return None;
            };
            if !values.iter().flatten().all(|value| matches!(value, ConstValue::Float(value) if value.format() == left.format())) {
                return None;
            }
            (left.format(), 1, computation.rounding)
        }
    };
    // The mode a computation rounding by the environment reads, where it is
    // known; a known mode no rule names raises nothing anyone can compute.
    let environment = match computation
        .environment
        .and_then(|mode| get_value(mode)?.0.as_u64())
    {
        Some(mode) => Some(FloatRounding::of_environment(mode)?),
        None => T::environment_rounding(),
    };
    let rounding = match computation.arithmetic {
        RaisingArithmetic::Compare { .. } => Some(FloatRounding::NearestEven),
        _ => rounding.or(environment),
    };
    let bits = FloatRounding::decide(rounding, |rounding| {
        let mut bits = 0;
        for index in 0..count {
            if !active(outer.as_ref(), index, format.bits(), count)?
                || !active(inner.as_ref(), index, format.bits(), count)?
            {
                continue;
            }
            let mut operands = [FloatBits::new(format, 0)?; 3];
            for (operand, value) in operands.iter_mut().zip(values.iter().flatten()) {
                *operand = lane(value, index, format, ptr_size)?;
            }
            let length = values.iter().flatten().count();
            if computation.negate_product {
                operands[0] = operands[0].negate();
            }
            if computation.negate_addend {
                operands[2] = operands[2].negate();
            }
            let exceptions = rules.raises(
                computation.arithmetic,
                operands.get(..length)?,
                Some(rounding),
                raises,
            )?;
            bits |= layout.place(exceptions.intersection(raises));
        }
        Some(bits)
    })?;
    result(bits, destination, ptr_size)
}

/// Boxes status at its declared register width.
fn result<T: Target>(
    bits: u64,
    destination: Destination,
    ptr_size: PointerSize,
) -> Option<ConstValue<T>> {
    let width = destination.bits.unwrap_or(64);
    let layout = IntegerType::new(width, false)?;
    if width < 64 && bits.checked_shr(width).unwrap_or(0) != 0 {
        return None;
    }
    layout.constant(u128::from(bits), ptr_size)
}
