//! Folding one scalar operation of constant operands.

use std::slice;

use crate::{
    PointerSize,
    analysis::consts::{
        Machine, TypeQueries,
        layout::{Destination, OperandLayout, scalar_zero},
        raises,
        vectors::VectorFold,
    },
    ir::{
        evaluation::NumericOp,
        float::{FloatClassLayout, FloatFormat, FloatRounding, FloatRules},
        lanes::VectorConst,
        ops::{
            CmpKind, Computation, ControlledKind, FloatFunctionKind, Kernel, KindedVecData, SsaOp,
            VectorBinaryKind, VectorElement, VectorElementKind,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{Endianness, Target},
};

/// Encodes a nonnegative count at its result width without changing the input width.
#[inline(always)]
pub fn bit_count_result<T: Target>(
    count: u32,
    width: Option<u32>,
    unsigned: bool,
) -> Option<ConstValue<T>> {
    let width = width.unwrap_or(32);
    if width == 32 {
        return Some(if unsigned {
            ConstValue::U32(count)
        } else {
            ConstValue::I32(i32::try_from(count).ok()?)
        });
    }
    if !(1..=128).contains(&width) || (width < 32 && count >= 1u32 << width) {
        return None;
    }
    match width {
        1 => Some(ConstValue::from_bool(count != 0)),
        _ => ConstValue::of_lane(width, u128::from(count)),
    }
}

/// The rule an operation that rounds by the environment rounds by: that of
/// its mode operand `mode` where the operand is known
/// ([`FloatRounding::of_environment`]), else the target's
/// ([`Target::environment_rounding`]), `None` within for a mode the program
/// sets at run time. `None` for a known mode no rule names (a reserved `frm`),
/// which no operation computes by.
pub fn environment_rounding<T: Target>(
    mode: Option<SsaVarId>,
    get_const: &mut impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
) -> Option<Option<FloatRounding>> {
    let Some(mode) = mode.and_then(|mode| get_const(mode)?.as_u64()) else {
        return Some(T::environment_rounding());
    };
    FloatRounding::of_environment(mode).map(Some)
}

/// The constant a lane operation's computed lanes are: the vector itself, or,
/// for an operation on scalars only, the scalar of lane 0 - the float of the
/// element's format, or the unsigned integer of its width.
pub fn lane_result<T: Target>(
    computed: VectorConst,
    element: VectorElement,
    scalars: bool,
) -> Option<ConstValue<T>> {
    if !scalars {
        return Some(ConstValue::Vector(computed));
    }
    let bits = computed.lane_bits();
    let lane = computed.lane(0);
    if element.kind.is_float() {
        let format = match element.kind {
            VectorElementKind::BFloat16 => FloatFormat::BFloat16,
            _ => FloatFormat::of_width(bits)?,
        };
        return ConstValue::float(format, lane);
    }
    ConstValue::of_lane(bits, lane)
}

/// The constants `values` as vectors: a vector as itself, an integer as one
/// lane of its width.
pub fn as_lanes<T: Target>(
    values: &[ConstValue<T>],
    ptr_size: PointerSize,
) -> Option<Vec<VectorConst>> {
    values
        .iter()
        .map(|value| match value {
            ConstValue::Vector(vector) => Some(vector.clone()),
            scalar => {
                let (raw, width) = scalar.integer_pattern(ptr_size)?;
                VectorConst::from_lanes(width, [u128::from(raw)])
            }
        })
        .collect()
}

/// The constant a computation of a whole value of constant inputs gives,
/// `bits` wide: the integer [`Kernel::evaluate_integers`] computes of the
/// inputs' bits, unsigned, the lanes [`Kernel::evaluate_lanes`] computes of
/// vector inputs or of a result no integer holds, the register
/// [`Kernel::evaluate_registers`] computes of a kernel over a whole register,
/// or the signed integer a modular conversion gives of its float
/// ([`FloatBits::to_integer_modular`]).
///
/// [`FloatBits::to_integer_modular`]: crate::ir::float::FloatBits::to_integer_modular
pub fn fold_compute<T: Target>(
    data: &KindedVecData<Computation>,
    bits: u32,
    get_const: &mut impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ptr_size: PointerSize,
) -> Option<ConstValue<T>> {
    if data.kind.lanes.is_some() {
        // A mask summary writes its scalar whatever the active length: of
        // no element, a count of zero and no first set element, all ones
        // (RISC-V's `vcpop.m` and `vfirst.m` with `vl` zero).
        // Any other computation over lanes reads its vectors at the element
        // widths their types declare, which only the operand layouts state
        // ([`Machine::evaluate_with_type_queries`]).
        let Kernel::MaskSummary { first } = data.kind.kernel else {
            return None;
        };
        let length = get_const(*data.inputs.get(1)?)?.as_u64()?;
        let none = if first { -1 } else { 0 };
        return (length == 0).then(|| ConstValue::integer(bits, false, none))?;
    }
    // The length chosen for an application count of zero is zero, whatever
    // the maximum: the choice is the count where it is no greater than the
    // maximum (RISC-V's `vsetivli zero, 0`).
    if data.kind.kernel == Kernel::VectorLengthChoice
        && data
            .inputs
            .first()
            .and_then(|count| get_const(*count))
            .and_then(|count| count.as_u64())
            == Some(0)
    {
        return ConstValue::integer(bits, false, 0);
    }
    if data.kind.kernel == Kernel::FloatToIntModular {
        let [input] = data.inputs.as_slice() else {
            return None;
        };
        let float = get_const(*input)?.read_float(ptr_size)?;
        let value = u64::try_from(float.to_integer_modular(bits)?).ok()?;
        return ConstValue::integer(bits, true, value.cast_signed());
    }
    let values = data
        .inputs
        .iter()
        .map(|input| get_const(*input))
        .collect::<Option<Vec<_>>>()?;
    // A float lane kernel computes its lanes as floats whatever their
    // operands' shape: a scalar form's operands are floats, and its result
    // one.
    let float_lanes = match data.kind.kernel {
        Kernel::FloatControlled { kind, control } => Some((Some((kind, control)), None)),
        Kernel::FloatHelper(helper) => Some((None, Some(helper))),
        _ => None,
    };
    if let Some((controlled, helper)) = float_lanes {
        // The mode one rounding by the environment reads comes first of its
        // whole-value inputs, and its lanes after it.
        let mode = match data.kind.environment_input() {
            Some(0) if data.kind.lanes.is_none() => data.inputs.first().copied(),
            Some(_) => return None,
            None => None,
        };
        let environment = environment_rounding(mode, get_const)?;
        let lanes = values
            .iter()
            .skip(usize::from(mode.is_some()))
            .map(|value| value.as_lanes(ptr_size))
            .collect::<Option<Vec<_>>>()?;
        let element = data.kind.element;
        let computed = match (controlled, helper) {
            (Some((kind, control)), _) => {
                kind.evaluate_lanes(control, &lanes, element, environment)?
            }
            (None, Some(helper)) => helper.evaluate_lanes(&lanes, element, environment)?,
            (None, None) => return None,
        };
        if computed.bits() != bits {
            return None;
        }
        // A class test's result is a lane mask, never a float.
        let classifies = matches!(controlled, Some((ControlledKind::Classify, _)));
        let scalar = !classifies
            && !values
                .iter()
                .any(|value| matches!(value, ConstValue::Vector(_)));
        return match FloatFormat::of_width(bits).filter(|_| scalar && computed.lanes() == 1) {
            Some(format) => ConstValue::float(format, computed.lane(0)),
            None => Some(ConstValue::Vector(computed)),
        };
    }
    // A kernel over the lanes of a vector computes them element by element,
    // as does one whose result no integer holds (the whole carry-less product
    // of two quadwords), its integer inputs read as a lane each.
    if bits > 64
        || values
            .iter()
            .any(|value| matches!(value, ConstValue::Vector(_)))
    {
        let lanes = as_lanes(&values, ptr_size)?;
        let (kernel, element_bits) = (data.kind.kernel, data.kind.element.bits);
        let computed = kernel.evaluate_lanes(&lanes, element_bits).or_else(|| {
            let [result] =
                <[VectorConst; 1]>::try_from(kernel.evaluate_registers(&lanes, element_bits)?)
                    .ok()?;
            Some(result)
        })?;
        return (computed.bits() == bits).then_some(ConstValue::Vector(computed));
    }
    let inputs = values
        .iter()
        .map(|value| value.integer_pattern(ptr_size))
        .collect::<Option<Vec<_>>>()?;
    let value = data.kind.kernel.evaluate_integers(&inputs, bits)?;
    ConstValue::integer(bits, false, value.cast_signed())
}

/// Folds a scalar comparison to a predicate using shared target and NaN rules.
///
/// A known NaN decides equality and ordered predicates even when the other
/// input is unknown. Unordered comparisons return their stated unordered value.
#[inline]
pub fn fold_comparison<T: Target>(
    op: &SsaOp<T>,
    mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ptr_size: PointerSize,
    rules: FloatRules,
) -> Option<bool> {
    let (left, right, relation, unsigned) = match op {
        SsaOp::Ceq { left, right, .. } => (*left, *right, CmpKind::Eq, false),
        SsaOp::Clt {
            left,
            right,
            unsigned,
            ..
        } => (*left, *right, CmpKind::Lt, *unsigned),
        SsaOp::Cgt {
            left,
            right,
            unsigned,
            ..
        } => (*left, *right, CmpKind::Gt, *unsigned),
        _ => return None,
    };
    let (left, right) = (get_const(left), get_const(right));
    if [&left, &right]
        .into_iter()
        .flatten()
        .any(|value| !comparison_stable(value, rules))
    {
        return None;
    }
    if [&left, &right]
        .into_iter()
        .flatten()
        .any(ConstValue::is_nan)
    {
        return Some(unsigned);
    }
    let (left, right) = (left?, right?);
    Machine::new(ptr_size, None, rules).compare_values(&left, &right, relation, unsigned)
}

/// Returns whether the floating environment preserves this comparison input.
pub fn comparison_stable<T: Target>(value: &ConstValue<T>, rules: FloatRules) -> bool {
    match value {
        ConstValue::Float(value) => rules.unflushed(value.format(), slice::from_ref(value)),
        _ => true,
    }
}

/// Whether two constants are one symbol: the same symbol less or xored with
/// itself is zero, whatever address it is loaded at.
pub fn same_symbol<T: Target>(left: &ConstValue<T>, right: &ConstValue<T>) -> bool {
    matches!((left, right), (ConstValue::Symbol(left), ConstValue::Symbol(right)) if left == right)
}

/// Folds scalar arithmetic through the same floating environment contract as lane operations.
pub fn fold_numeric<T: Target>(
    kind: NumericOp,
    left: &ConstValue<T>,
    right: &ConstValue<T>,
    ptr_size: PointerSize,
    rules: FloatRules,
) -> Option<ConstValue<T>> {
    if let (ConstValue::Float(a), ConstValue::Float(b)) = (left, right) {
        if a.format() != b.format() {
            return None;
        }
        if kind == NumericOp::RemS {
            let result = if a.is_nan() || b.is_nan() {
                rules.propagated(&[*a, *b])
            } else {
                a.checked_rem(*b).or_else(|| {
                    (!a.is_finite() || b.abs().bits() == 0)
                        .then(|| rules.default_nan.and_then(|nan| nan.of(a.format())))
                        .flatten()
                })
            }?;
            return rules
                .unflushed(a.format(), &[*a, *b, result])
                .then_some(ConstValue::Float(result));
        }
        let kind = match kind {
            NumericOp::Add => VectorBinaryKind::Add,
            NumericOp::Sub => VectorBinaryKind::Sub,
            NumericOp::Mul => VectorBinaryKind::Mul,
            NumericOp::DivS => VectorBinaryKind::Div,
            _ => return None,
        };
        let mut element = VectorElement::float(a.format().bits()).as_scalar();
        if a.format() == FloatFormat::BFloat16 {
            element.kind = VectorElementKind::BFloat16;
        }
        let result = left.as_lanes(ptr_size)?.binary(
            &right.as_lanes(ptr_size)?,
            kind,
            element,
            T::environment_rounding(),
            rules,
        )?;
        return lane_result(result, element, true);
    }
    kind.evaluate_binary(left, right, ptr_size)
}

/// The operation-level fold, before the destination-width check.
///
/// `dest_width` is the destination's declared width, which an operation whose
/// result lanes nothing else states reads: an extracted lane is as wide as
/// its destination, and gathered lane bits fill theirs.
pub fn fold_op<T: Target>(
    op: &SsaOp<T>,
    dest: Destination,
    mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ptr_size: PointerSize,
    rules: FloatRules,
    types: &TypeQueries<'_, T>,
) -> Option<ConstValue<T>> {
    let dest_width = dest.bits;
    if let Some(folded) = VectorFold::fold(op, dest_width, &mut get_const, ptr_size, rules, types) {
        return folded;
    }
    let environment = environment_rounding(op.environment(), &mut get_const)?;
    match op {
        SsaOp::Const { value, .. } => Some(value.clone()),
        SsaOp::Copy { src, .. } => get_const(*src),
        SsaOp::FloatRaise { .. } => raises::fold(
            op,
            dest,
            &mut |variable| Some((get_const(variable)?, OperandLayout::default())),
            ptr_size,
            rules,
            types,
        ),

        // A `while` of constant bounds is the predicate it generates, bit `e`
        // element `e`'s.
        SsaOp::VectorPredicateWhile(data) => {
            let [first, last] = data.inputs.as_slice() else {
                return None;
            };
            let mut counter = |var: SsaVarId| -> Option<i128> {
                let (raw, width) = get_const(var)?.integer_pattern(ptr_size)?;
                let shift = 128u32.checked_sub(width)?;
                Some(if data.unsigned {
                    i128::from(raw)
                } else {
                    (i128::from(raw) << shift) >> shift
                })
            };
            let (first, last) = (counter(*first)?, counter(*last)?);
            let bits = data.evaluate(first, last)?;
            let width = dest_width.unwrap_or_else(|| u32::from(data.elements).next_multiple_of(8));
            ConstValue::of_lane(width, bits)
        }

        // The class of a constant float, in the encoding its classification
        // states: a mask of one bit as wide as the destination, or x87's
        // condition codes as a flags word.
        SsaOp::FpClassify {
            operand, layout, ..
        } => {
            let class = layout.class(get_const(*operand)?.read_float(ptr_size)?);
            match layout {
                FloatClassLayout::X87 => Some(ConstValue::U32(u32::try_from(class).ok()?)),
                FloatClassLayout::RiscV | FloatClassLayout::Mips => {
                    ConstValue::integer(dest_width?, false, class.cast_signed())
                }
            }
        }

        // A computation of constants with one result is the constant it
        // computes.
        SsaOp::Compute(data) if data.outputs.len() == 1 => {
            fold_compute(data, dest_width?, &mut get_const, ptr_size)
        }

        // Binary arithmetic
        SsaOp::Add { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            fold_numeric(NumericOp::Add, &l, &r, ptr_size, rules)
        }
        SsaOp::Sub { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            if dest.lanes.is_none() && same_symbol(&l, &r) {
                scalar_zero(dest_width.unwrap_or_else(|| ptr_size.bits()))
            } else {
                fold_numeric(NumericOp::Sub, &l, &r, ptr_size, rules)
            }
        }
        SsaOp::Mul { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            fold_numeric(NumericOp::Mul, &l, &r, ptr_size, rules)
        }
        SsaOp::Div {
            left,
            right,
            unsigned,
            ..
        } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            if *unsigned {
                fold_numeric(NumericOp::DivU, &l, &r, ptr_size, rules)
            } else {
                fold_numeric(NumericOp::DivS, &l, &r, ptr_size, rules)
            }
        }
        SsaOp::Rem {
            left,
            right,
            unsigned,
            ..
        } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            if *unsigned {
                fold_numeric(NumericOp::RemU, &l, &r, ptr_size, rules)
            } else {
                fold_numeric(NumericOp::RemS, &l, &r, ptr_size, rules)
            }
        }

        // Bitwise
        SsaOp::Xor { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            if dest.lanes.is_none() && same_symbol(&l, &r) {
                scalar_zero(dest_width.unwrap_or_else(|| ptr_size.bits()))
            } else {
                NumericOp::Xor.evaluate_binary(&l, &r, ptr_size)
            }
        }
        SsaOp::And { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            NumericOp::And.evaluate_binary(&l, &r, ptr_size)
        }
        SsaOp::Or { left, right, .. } => {
            let l = get_const(*left)?;
            let r = get_const(*right)?;
            NumericOp::Or.evaluate_binary(&l, &r, ptr_size)
        }

        // Boolean logic. Distinct from the bitwise ops: the operands are truth
        // values (`ConstValue::as_bool` reads any zero/non-zero integer as well
        // as `True`/`False`) and the result is a boolean, not a width-carrying
        // integer.
        SsaOp::BoolAnd { left, right, .. } => Some(ConstValue::from_bool(
            get_const(*left)?.as_bool()? && get_const(*right)?.as_bool()?,
        )),
        SsaOp::BoolOr { left, right, .. } => Some(ConstValue::from_bool(
            get_const(*left)?.as_bool()? || get_const(*right)?.as_bool()?,
        )),
        SsaOp::BoolXor { left, right, .. } => Some(ConstValue::from_bool(
            get_const(*left)?.as_bool()? != get_const(*right)?.as_bool()?,
        )),
        SsaOp::BoolNot { value, .. } => Some(ConstValue::from_bool(!get_const(*value)?.as_bool()?)),

        // A constant flags value holds each flag at its bit of `FlagsMask`.
        SsaOp::ReadFlags { flags, mask, .. } => {
            Some(ConstValue::from_bool(get_const(*flags)?.read_flag(*mask)?))
        }
        SsaOp::MergeFlags {
            base, update, mask, ..
        } => get_const(*base)?.merged_flags(&get_const(*update)?, *mask),
        SsaOp::InsertFlag {
            base, bit, mask, ..
        } => get_const(*base)?.with_flag(*mask, get_const(*bit)?.as_bool()?),

        // Shifts
        SsaOp::Shl { value, amount, .. } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            NumericOp::Shl.evaluate_binary(&v, &a, ptr_size)
        }
        SsaOp::Shr {
            value,
            amount,
            unsigned,
            ..
        } => {
            let v = get_const(*value)?;
            let a = get_const(*amount)?;
            if *unsigned {
                NumericOp::ShrU
            } else {
                NumericOp::ShrS
            }
            .evaluate_binary(&v, &a, ptr_size)
        }

        // Unary
        SsaOp::Neg { operand, .. } => {
            let v = get_const(*operand)?;
            NumericOp::Neg.evaluate_unary(&v, ptr_size)
        }
        SsaOp::Not { operand, .. } => {
            let v = get_const(*operand)?;
            NumericOp::Not.evaluate_unary(&v, ptr_size)
        }
        SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. } => {
            fold_comparison(op, get_const, ptr_size, rules).map(ConstValue::from_bool)
        }

        // Type conversions. Integer→integer and float→integer carry both the
        // signedness and the (CIL) overflow-check; integer→float carries only
        // signedness; the pointer and float-width conversions carry neither.
        // A reinterpretation keeps the bits and reads them as its target type:
        // a float moved into an integer register (`fmv.x.w`, `movd`) is its
        // bit pattern.
        SsaOp::Bitcast {
            operand, target, ..
        } => get_const(*operand)?
            .to_bytes(Endianness::Little, ptr_size)
            .and_then(|bytes| types.decode(&bytes, target)),
        // A conversion that states its rounding folds exactly by it.
        SsaOp::IntToFloat {
            operand,
            target,
            unsigned,
            rounding,
            ..
        } => types.rounded(
            &get_const(*operand)?,
            target,
            *unsigned,
            rounding.or(environment),
            rules,
        ),
        // Pointers are unsigned: a pointer-width value must zero-extend, never
        // sign-extend. Passing `false` here would widen a 32-bit `0x8000_0000`
        // to `0xFFFF_FFFF_8000_0000` on a 64-bit target.
        //
        // A symbol reinterpreted at the pointer's width is the same symbol: the
        // bits do not change, only the type that reads them. Kept a conversion,
        // it is a computation over a constant, which a loop's invariant code
        // motion hoists out of the loop into a variable and a printer reads as
        // arithmetic. At any other width it is a slice of an address, which
        // only the address model can say anything about.
        SsaOp::IntToPtr {
            operand, target, ..
        }
        | SsaOp::PtrToInt {
            operand, target, ..
        } => {
            let v = get_const(*operand)?;
            if matches!(v, ConstValue::Symbol(_)) {
                let pointer_wide = types.pointer_wide(target);
                return pointer_wide.then_some(v);
            }
            types.pointer_convert(&v, target)
        }

        // An address that adds nothing to a symbol is the symbol: an import's
        // address held in an integer register and read back as the address of
        // a load. Any displacement of a symbol is arithmetic on an address
        // relocation moves, which is the address model's to say, and so is
        // one formed from null. A plain number is an absolute address nothing
        // relocates: the sum is that address, wrapping at the pointer's width
        // as the machine's arithmetic does.
        SsaOp::PtrAdd {
            base,
            index,
            stride,
            offset,
            result_type,
            ..
        } => {
            let base = get_const(*base)?;
            if index.is_none() && *offset == 0 && matches!(base, ConstValue::Symbol(_)) {
                return Some(base);
            }
            if !types.is_pointer(result_type) {
                return None;
            }
            let index = match index {
                Some(index) => Some(get_const(*index)?),
                None => None,
            };
            let address = base.pointer_add(index.as_ref(), *stride, *offset, ptr_size)?;
            if ptr_size.bits() <= 64 {
                address.as_u64().map(ConstValue::NativeUInt)
            } else {
                Some(address)
            }
        }

        // Between two float formats the conversion is IEEE's on every target,
        // rounding as it states or as the environment does.
        SsaOp::FloatConv {
            operand,
            target,
            rounding,
            source,
            ..
        } => {
            let value = get_const(*operand)?;
            let float = match source {
                Some(format) => value.as_float(*format, ptr_size),
                None => value.read_float(ptr_size),
            }?;
            types.rounded(
                &ConstValue::Float(float),
                target,
                false,
                rounding.or(environment),
                rules,
            )
        }

        // `fscale`, the float function of one result that is exact: the value
        // times two to the truncated power, where that does not round.
        SsaOp::FloatFunction(data) if data.kind == FloatFunctionKind::Scale => {
            let [value, by] = data.inputs.as_slice() else {
                return None;
            };
            let value = get_const(*value)?.read_float(ptr_size)?;
            let by = get_const(*by)?.read_float(ptr_size)?;
            Some(ConstValue::Float(value.scaled(by)?))
        }

        // Rotates
        SsaOp::Rol { value, amount, .. } => {
            NumericOp::Rol.evaluate_binary(&get_const(*value)?, &get_const(*amount)?, ptr_size)
        }
        SsaOp::Ror { value, amount, .. } => {
            NumericOp::Ror.evaluate_binary(&get_const(*value)?, &get_const(*amount)?, ptr_size)
        }

        // Bit permutations retain the operand width; count results use the
        // declared result width, independently of the register being inspected.
        SsaOp::BSwap { src, .. } => NumericOp::BSwap.evaluate_unary(&get_const(*src)?, ptr_size),
        SsaOp::BRev { src, .. } => NumericOp::BRev.evaluate_unary(&get_const(*src)?, ptr_size),
        SsaOp::BitScanForward { src, .. }
        | SsaOp::BitScanReverse { src, .. }
        | SsaOp::Popcount { src, .. }
        | SsaOp::Parity { src, .. } => {
            let kind = match op {
                SsaOp::BitScanForward { .. } => NumericOp::BitScanForward,
                SsaOp::BitScanReverse { .. } => NumericOp::BitScanReverse,
                SsaOp::Popcount { .. } => NumericOp::Popcount,
                SsaOp::Parity { .. } => NumericOp::Parity,
                _ => return None,
            };
            let count = kind.bit_count(&get_const(*src)?, ptr_size)?;
            bit_count_result(count, dest_width, matches!(kind, NumericOp::BitScanReverse))
        }

        // Conditional select
        SsaOp::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            let cond = get_const(*condition)?;
            match cond.as_i64() {
                Some(0) => get_const(*false_val),
                Some(_) => get_const(*true_val),
                None => None,
            }
        }

        // All other operations cannot be evaluated to constants
        _ => None,
    }
}
