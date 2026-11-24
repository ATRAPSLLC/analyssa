//! Folding an operation over vector lanes, of constant operands.

use std::iter;

use crate::{
    PointerSize,
    analysis::consts::{
        ReinterpretShape, TypeQueries,
        fold::{environment_rounding, lane_result},
    },
    ir::{
        float::{FloatRounding, FloatRules},
        integer::BitWidth,
        lanes::VectorConst,
        ops::{
            ComplexProduct, Contraction, ReduceGroups, SsaOp, SveComputeKind, VectorCastKind,
            VectorMaskMode, VectorPackKind, VectorReduceKind, VectorShuffleSelector,
            VectorTernaryKind,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// A family of vector operations a fold knows: the fold of one of them,
/// `None` for an operation of another family.
type Family<F, T> = fn(&mut F, &SsaOp<T>) -> Option<Option<ConstValue<T>>>;

/// One vector operation's fold: the constants its operands are, read as
/// lanes, and what the fold reads of the target.
pub struct VectorFold<'g, 'q, T: Target, G> {
    /// Resolves an operand to its constant.
    get_const: &'g mut G,
    /// The width of a pointer and a native integer.
    ptr_size: PointerSize,
    /// The target's float rules.
    rules: FloatRules,
    /// What the target states of a type.
    types: &'q TypeQueries<'q, T>,
    /// The rounding mode the program has set, where the operation reads it.
    environment: Option<FloatRounding>,
    /// The width of the destination, where it states one.
    dest_width: Option<u32>,
}

impl<'g, 'q, T, G> VectorFold<'g, 'q, T, G>
where
    T: Target,
    G: FnMut(SsaVarId) -> Option<ConstValue<T>>,
{
    /// The families of operations a vector fold knows, tried in order.
    const FAMILIES: [Family<Self, T>; 7] = [
        Self::movement,
        Self::casts,
        Self::arithmetic,
        Self::summaries,
        Self::contractions,
        Self::packing,
        Self::sve,
    ];

    pub fn fold(
        op: &SsaOp<T>,
        dest_width: Option<u32>,
        get_const: &'g mut G,
        ptr_size: PointerSize,
        rules: FloatRules,
        types: &'q TypeQueries<'q, T>,
    ) -> Option<Option<ConstValue<T>>> {
        let Some(environment) = environment_rounding(op.environment(), get_const) else {
            return Some(None);
        };
        let mut fold = Self {
            get_const,
            ptr_size,
            rules,
            types,
            environment,
            dest_width,
        };
        Self::FAMILIES
            .into_iter()
            .find_map(|family| family(&mut fold, op))
    }

    /// Folds an operation that moves lanes: a splat, a slice or its insertion, a
    /// concatenation, a reinterpretation, a shuffle, a lane inserted or extracted.
    fn movement(&mut self, op: &SsaOp<T>) -> Option<Option<ConstValue<T>>> {
        let ptr_size = self.ptr_size;
        let types = self.types;
        let dest_width = self.dest_width;
        let get_const = &mut *self.get_const;
        let mut lanes = |var: SsaVarId| get_const(var).and_then(|value| value.as_lanes(ptr_size));
        let vector = |folded: Option<VectorConst>| folded.map(ConstValue::Vector);
        Some(match op {
            SsaOp::VectorSplat {
                value, vector_type, ..
            } => (|| {
                let shape = types.vector_shape(vector_type)?;
                let value = lanes(*value)?.bits_at(0, shape.lane_bits);
                vector(VectorConst::splat(shape.lane_bits, shape.lane_count, value))
            })(),
            SsaOp::VectorSlice {
                vector: source,
                offset,
                length,
                target_type,
                ..
            } => (|| {
                let shape = types.vector_shape(target_type)?;
                if lanes(*length)?.bits_at(0, 128) != u128::from(shape.total_bits) {
                    return None;
                }
                let source = lanes(*source)?;
                let offset = u64::try_from(lanes(*offset)?.bits_at(0, 128)).ok()?;
                vector(source.slice_bits(offset, shape.lane_bits, shape.lane_count))
            })(),
            SsaOp::VectorInsertSlice {
                vector: source,
                offset,
                value,
                target_type,
                ..
            } => (|| {
                let shape = types.vector_shape(target_type)?;
                let source = lanes(*source)?;
                if source.bits() != shape.total_bits {
                    return None;
                }
                let source = source.relaned(shape.lane_bits)?;
                let offset = u64::try_from(lanes(*offset)?.bits_at(0, 128)).ok()?;
                let value = lanes(*value)?;
                Some(ConstValue::Vector(source.insert_bits(offset, &value)))
            })(),
            SsaOp::VectorConcat {
                left,
                right,
                target_type,
                ..
            } => (|| {
                let shape = types.vector_shape(target_type)?;
                let parts = [lanes(*left)?, lanes(*right)?];
                vector(VectorConst::concatenate(
                    &parts,
                    shape.lane_bits,
                    shape.lane_count,
                ))
            })(),
            SsaOp::VectorReinterpret {
                value, target_type, ..
            } => (|| {
                let value = lanes(*value)?;
                match types.reinterpret_shape(target_type)? {
                    ReinterpretShape::Vector(shape) => {
                        vector(value.resized(shape.lane_bits, shape.lane_count))
                    }
                    ReinterpretShape::Scalar(bits) => {
                        ConstValue::of_lane(bits, value.bits_at(0, bits))
                    }
                }
            })(),
            // The result holds a lane per selector across the destination, and the
            // inputs are read in lanes that wide.
            SsaOp::VectorShuffle {
                left,
                right,
                selector: VectorShuffleSelector::Mask(mask),
                ..
            } => (|| {
                let selectors = u32::try_from(mask.lanes().len()).ok()?;
                let relaid = |vector: VectorConst| match dest_width {
                    Some(bits) => vector.relaned(bits.checked_div(selectors)?),
                    None => Some(vector),
                };
                let left = relaid(lanes(*left)?)?;
                let right = match right {
                    Some(right) => Some(relaid(lanes(*right)?)?),
                    None => None,
                };
                vector(left.shuffle(right.as_ref(), mask.lanes()))
            })(),
            // The lanes are as wide as the value inserted.
            SsaOp::VectorInsert {
                vector: base,
                lane,
                value,
                ..
            } => (|| {
                let value = lanes(*value)?;
                let base = lanes(*base)?.relaned(value.bits())?;
                vector(base.with_lane(*lane, value.bits_at(0, value.bits())))
            })(),
            // The lane is as wide as the destination.
            SsaOp::VectorExtract {
                vector: base, lane, ..
            } => (|| {
                let base = lanes(*base)?;
                let bits = dest_width.unwrap_or(base.lane_bits());
                let base = base.relaned(bits)?;
                (*lane < base.lanes()).then(|| ConstValue::of_lane(bits, base.lane(*lane)))?
            })(),
            // A saturating integer cast clamps each lane, read as `from`, to the
            // range of the destination's lane as `to` reads it.
            _ => return None,
        })
    }

    /// Folds a cast of every lane.
    fn casts(&mut self, op: &SsaOp<T>) -> Option<Option<ConstValue<T>>> {
        let ptr_size = self.ptr_size;
        let rules = self.rules;
        let types = self.types;
        let environment = self.environment;
        let get_const = &mut *self.get_const;
        let mut lanes = |var: SsaVarId| get_const(var).and_then(|value| value.as_lanes(ptr_size));
        let vector = |folded: Option<VectorConst>| folded.map(ConstValue::Vector);
        Some(match op {
            SsaOp::VectorCast {
                value,
                target_type,
                from: from @ (VectorCastKind::Signed | VectorCastKind::Unsigned),
                to: to @ (VectorCastKind::Signed | VectorCastKind::Unsigned),
                saturate: true,
                ..
            } => (|| {
                let shape = types.vector_shape(target_type)?;
                vector(lanes(*value)?.saturated(
                    shape.lane_bits,
                    *from == VectorCastKind::Signed,
                    *to == VectorCastKind::Unsigned,
                ))
            })(),
            SsaOp::VectorCast { saturate: true, .. } => None,
            SsaOp::VectorCast {
                value,
                target_type,
                from,
                to,
                rounding,
                ..
            } => (|| {
                let shape = types.vector_shape(target_type)?;
                // Lanes whose rounding the implementation decides take none of
                // the environment's ([`VectorElementKind::FloatOpen`]).
                let open = from.opens_specials() || to.opens_specials();
                vector(lanes(*value)?.cast(
                    *from,
                    *to,
                    if open {
                        *rounding
                    } else {
                        rounding.or(environment)
                    },
                    shape.lane_bits,
                    shape.lane_count,
                    rules,
                ))
            })(),
            _ => return None,
        })
    }

    /// Folds lane-wise arithmetic, predicated or not.
    fn arithmetic(&mut self, op: &SsaOp<T>) -> Option<Option<ConstValue<T>>> {
        let ptr_size = self.ptr_size;
        let rules = self.rules;
        let environment = self.environment;
        let get_const = &mut *self.get_const;
        let mut lanes = |var: SsaVarId| get_const(var).and_then(|value| value.as_lanes(ptr_size));
        let vector = |folded: Option<VectorConst>| folded.map(ConstValue::Vector);
        Some(match op {
            SsaOp::VectorUnary {
                value,
                kind,
                element,
                ..
            } => (|| {
                let source = get_const(*value)?;
                let computed =
                    source
                        .as_lanes(ptr_size)?
                        .unary(*kind, *element, environment, rules)?;
                lane_result(computed, *element, !source.is_vector())
            })(),
            SsaOp::VectorBinary {
                left,
                right,
                kind,
                element,
                ..
            } => (|| {
                let (l, r) = (get_const(*left)?, get_const(*right)?);
                let computed = l.as_lanes(ptr_size)?.binary(
                    &r.as_lanes(ptr_size)?,
                    *kind,
                    *element,
                    environment,
                    rules,
                )?;
                lane_result(computed, *element, !l.is_vector() && !r.is_vector())
            })(),
            // A select reads only the lanes it chooses: a value it discards may be
            // unknown (a NaN product, whose payload the architecture decides).
            SsaOp::VectorTernary {
                first,
                second,
                third,
                kind: VectorTernaryKind::Select,
                ..
            } => (|| {
                let (clear, set) = (lanes(*first), lanes(*second));
                vector(lanes(*third)?.select(clear.as_ref(), set.as_ref()))
            })(),
            SsaOp::VectorTernary {
                first,
                second,
                third,
                kind,
                element,
                ..
            } => (|| {
                let (first, second, third) =
                    (get_const(*first)?, get_const(*second)?, get_const(*third)?);
                let computed = first.as_lanes(ptr_size)?.ternary(
                    &second.as_lanes(ptr_size)?,
                    &third.as_lanes(ptr_size)?,
                    *kind,
                    *element,
                    environment,
                    rules,
                )?;
                let scalars = !first.is_vector() && !second.is_vector() && !third.is_vector();
                lane_result(computed, *element, scalars)
            })(),
            SsaOp::VectorPredicatedUnary {
                value,
                mask,
                passthrough,
                kind,
                element,
                mode,
                ..
            } => (|| {
                let computed = get_const(*value)?.as_lanes(ptr_size)?.unary(
                    *kind,
                    *element,
                    environment,
                    rules,
                )?;
                let governed = Governed::of(get_const, ptr_size, *mask, *passthrough, *mode);
                vector(governed.merge(&computed, element.bits))
            })(),
            SsaOp::VectorPredicatedBinary {
                left,
                right,
                mask,
                passthrough,
                kind,
                element,
                mode,
                ..
            } => (|| {
                let right = get_const(*right)?.as_lanes(ptr_size)?;
                let computed = get_const(*left)?.as_lanes(ptr_size)?.binary(
                    &right,
                    *kind,
                    *element,
                    environment,
                    rules,
                )?;
                let governed = Governed::of(get_const, ptr_size, *mask, *passthrough, *mode);
                vector(governed.merge(&computed, element.bits))
            })(),
            SsaOp::VectorPredicatedTernary {
                first,
                second,
                third,
                mask,
                passthrough,
                kind,
                element,
                mode,
                ..
            } => (|| {
                let second = get_const(*second)?.as_lanes(ptr_size)?;
                let third = get_const(*third)?.as_lanes(ptr_size)?;
                let computed = get_const(*first)?.as_lanes(ptr_size)?.ternary(
                    &second,
                    &third,
                    *kind,
                    *element,
                    environment,
                    rules,
                )?;
                let governed = Governed::of(get_const, ptr_size, *mask, *passthrough, *mode);
                vector(governed.merge(&computed, element.bits))
            })(),
            _ => return None,
        })
    }

    /// Folds a comparison, a reduction or a bitmask of the lanes.
    fn summaries(&mut self, op: &SsaOp<T>) -> Option<Option<ConstValue<T>>> {
        let ptr_size = self.ptr_size;
        let rules = self.rules;
        let environment = self.environment;
        let dest_width = self.dest_width;
        let get_const = &mut *self.get_const;
        let mut lanes = |var: SsaVarId| get_const(var).and_then(|value| value.as_lanes(ptr_size));
        let vector = |folded: Option<VectorConst>| folded.map(ConstValue::Vector);
        Some(match op {
            SsaOp::VectorCompare {
                left,
                right,
                kind,
                element,
                ..
            } => (|| vector(lanes(*left)?.compare(&lanes(*right)?, *kind, *element)))(),
            // A grouped reduction: each run of lanes summed, or a pair's
            // difference taken, into a wider lane.
            SsaOp::VectorReduce {
                value,
                kind,
                element,
                initial: None,
                groups: Some(ReduceGroups { lanes: group, bits }),
                ..
            } => (|| {
                let subtract = match kind {
                    VectorReduceKind::Add => false,
                    VectorReduceKind::Difference if *group == 2 => true,
                    _ => return None,
                };
                let source = lanes(*value)?;
                let narrow = if element.bits == 0 {
                    source.lane_bits()
                } else {
                    element.bits
                };
                let count = source.bits().checked_div(narrow)?.checked_div(*group)?;
                vector(source.horizontal(
                    narrow,
                    *bits,
                    count,
                    element.signed != Some(true),
                    subtract,
                ))
            })(),
            SsaOp::VectorReduce {
                value,
                kind,
                element,
                initial,
                ordered,
                groups: None,
                ..
            } => (|| {
                let initial = match initial {
                    Some(initial) => Some(lanes(*initial)?.lane(0)),
                    None => None,
                };
                let value = lanes(*value)?;
                let reduced =
                    value.reduce(*kind, *element, initial, *ordered, environment, rules)?;
                match kind {
                    VectorReduceKind::Any | VectorReduceKind::All => {
                        Some(ConstValue::from_bool(reduced != 0))
                    }
                    // One lane of the element: a float of its format, bfloat16
                    // included, or an integer of its width.
                    _ => {
                        let bits = if element.bits == 0 {
                            value.lane_bits()
                        } else {
                            element.bits
                        };
                        lane_result(VectorConst::from_lanes(bits, [reduced])?, *element, true)
                    }
                }
            })(),
            SsaOp::VectorBitmask { value, kind, .. } => (|| {
                let value = lanes(*value)?;
                let bits = dest_width.unwrap_or(value.lanes().next_multiple_of(8));
                ConstValue::of_lane(bits, value.bitmask(*kind)?)
            })(),
            // A contraction: the two factors, then the accumulator where there
            // is one; a float dot folds as its rule adds
            // ([`VectorConst::dot_float`]).
            _ => return None,
        })
    }

    /// Folds a contraction: a dot product, a matrix product or a complex one.
    fn contractions(&mut self, op: &SsaOp<T>) -> Option<Option<ConstValue<T>>> {
        let ptr_size = self.ptr_size;
        let get_const = &mut *self.get_const;
        let mut lanes = |var: SsaVarId| get_const(var).and_then(|value| value.as_lanes(ptr_size));
        let vector = |folded: Option<VectorConst>| folded.map(ConstValue::Vector);
        Some(match op {
            SsaOp::Contract(data) => (|| match data.kind {
                Contraction::Dot(shape) => {
                    let (accumulator, sources) = if shape.accumulates {
                        let (accumulator, sources) = data.inputs.split_last()?;
                        (Some(lanes(*accumulator)?), sources)
                    } else {
                        (None, data.inputs.as_slice())
                    };
                    let [a, b] = sources else {
                        return None;
                    };
                    vector(lanes(*a)?.dot(&lanes(*b)?, accumulator.as_ref(), shape))
                }
                Contraction::FloatDot { elements, rule } => {
                    let [a, b, accumulator] = data.inputs.as_slice() else {
                        return None;
                    };
                    vector(lanes(*accumulator)?.dot_float(&lanes(*a)?, &lanes(*b)?, elements, rule))
                }
                Contraction::MultiplyAdd52 { high } => {
                    let [a, b, accumulator] = data.inputs.as_slice() else {
                        return None;
                    };
                    vector(lanes(*accumulator)?.multiply_add52(&lanes(*a)?, &lanes(*b)?, high))
                }
                // A complex product or rotation folds where every way of
                // computing it gives one result ([`VectorConst::complex`]).
                Contraction::Complex(shape) => {
                    let accumulates = matches!(
                        shape.product,
                        ComplexProduct::Multiply {
                            accumulates: true,
                            ..
                        }
                    );
                    let (accumulator, sources) = if accumulates {
                        let (accumulator, sources) = data.inputs.split_last()?;
                        (Some(lanes(*accumulator)?), sources)
                    } else {
                        (None, data.inputs.as_slice())
                    };
                    let [a, b] = sources else {
                        return None;
                    };
                    vector(lanes(*a)?.complex(&lanes(*b)?, accumulator.as_ref(), shape))
                }
                Contraction::MaskedDot { .. } | Contraction::Matrix(_) | Contraction::Outer(_) => {
                    None
                }
            })(),
            // The index vector selects from the inputs laid end to end.
            _ => return None,
        })
    }

    /// Folds a pack, an extension within lanes, or a shuffle that selects across
    /// lanes of two sources.
    fn packing(&mut self, op: &SsaOp<T>) -> Option<Option<ConstValue<T>>> {
        let ptr_size = self.ptr_size;
        let get_const = &mut *self.get_const;
        let mut lanes = |var: SsaVarId| get_const(var).and_then(|value| value.as_lanes(ptr_size));
        let vector = |folded: Option<VectorConst>| folded.map(ConstValue::Vector);
        Some(match op {
            SsaOp::VectorShuffle {
                left,
                right,
                selector: VectorShuffleSelector::Index(index),
                ..
            } => (|| {
                let tables = iter::once(*left)
                    .chain(*right)
                    .map(&mut lanes)
                    .collect::<Option<Vec<_>>>()?;
                vector(lanes(*index)?.permute(&tables))
            })(),
            // A compress packs the lanes its mask selects low, an expand spreads
            // its low lanes over them; the others keep the passthrough, or zero.
            SsaOp::VectorPack {
                value,
                mask,
                passthrough,
                element_bits,
                kind,
                mode,
                ..
            } => (|| {
                let source = lanes(*value)?.relaned(*element_bits)?;
                let count = source.lanes();
                // A mask of one lane holds a bit per lane (an opmask); one of the
                // operation's lanes is all ones where a lane is active.
                let mask = lanes(*mask)?;
                let active: Vec<bool> = if mask.lanes() == 1 {
                    let bits = mask.lane(0);
                    (0..count)
                        .map(|lane| bits.checked_shr(lane).is_some_and(|bit| bit & 1 == 1))
                        .collect()
                } else {
                    let mask = mask.relaned(*element_bits)?;
                    (0..count)
                        .map(|lane| (mask.lane(lane) >> element_bits.saturating_sub(1)) & 1 == 1)
                        .collect()
                };
                let base = match (mode, passthrough) {
                    (VectorMaskMode::Merge, Some(passthrough)) => {
                        lanes(*passthrough)?.relaned(*element_bits)?
                    }
                    _ => VectorConst::splat(*element_bits, count, 0)?,
                };
                vector(source.packed(&active, &base, *kind == VectorPackKind::Compress))
            })(),
            // `sxtb`/`uxth`/...: each lane's low bits extended to the lane.
            SsaOp::VectorExtendInLane(data) => (|| {
                let [source] = data.inputs.as_slice() else {
                    return None;
                };
                let bits = u32::from(data.element_bits);
                let from = u32::from(data.source_bits);
                if from == 0 || from > bits {
                    return None;
                }
                let source = lanes(*source)?.relaned(bits)?;
                let lane_mask = BitWidth::new(bits).checked_mask_u128()?;
                let low_mask = BitWidth::new(from).checked_mask_u128()?;
                let sign = from.checked_sub(1)?;
                let extended = source.lane_values().map(|lane| {
                    let low = lane & low_mask;
                    if data.signed && low.checked_shr(sign).is_some_and(|bit| bit & 1 == 1) {
                        (low | !low_mask) & lane_mask
                    } else {
                        low
                    }
                });
                vector(VectorConst::from_lanes(bits, extended.collect::<Vec<_>>()))
            })(),
            // `cnot`: one where a lane is zero, zero elsewhere.
            _ => return None,
        })
    }

    /// Folds the SVE computations of the lanes a predicate governs.
    fn sve(&mut self, op: &SsaOp<T>) -> Option<Option<ConstValue<T>>> {
        let ptr_size = self.ptr_size;
        let get_const = &mut *self.get_const;
        let mut lanes = |var: SsaVarId| get_const(var).and_then(|value| value.as_lanes(ptr_size));
        let vector = |folded: Option<VectorConst>| folded.map(ConstValue::Vector);
        Some(match op {
            SsaOp::VectorSveCompute(data) if data.op == SveComputeKind::LogicalNot => (|| {
                let source = lanes(*data.inputs.last()?)?.relaned(u32::from(data.element_bits))?;
                let bits = source.lane_bits();
                let not = source.lane_values().map(|lane| u128::from(lane == 0));
                vector(VectorConst::from_lanes(bits, not.collect::<Vec<_>>()))
            })(
            ),
            // SVE2's carrying adds and subtracts, pair by pair: the even element
            // is the accumulator's plus the bottom or top source element (its
            // complement for a subtract) plus the carry in bit 0 of the second
            // source's odd element; the odd element is the carry out.
            SsaOp::VectorSveCompute(data)
                if matches!(
                    data.op,
                    SveComputeKind::AddCarryBottom
                        | SveComputeKind::AddCarryTop
                        | SveComputeKind::SubCarryBottom
                        | SveComputeKind::SubCarryTop
                ) =>
            {
                (|| {
                    let [accumulator, source, carries] = data.inputs.as_slice() else {
                        return None;
                    };
                    let bits = u32::from(data.element_bits);
                    let accumulator = lanes(*accumulator)?.relaned(bits)?;
                    let source = lanes(*source)?.relaned(bits)?;
                    let carries = lanes(*carries)?.relaned(bits)?;
                    let count = accumulator.lanes();
                    if source.lanes() != count || carries.lanes() != count || bits > 64 {
                        return None;
                    }
                    let top = matches!(
                        data.op,
                        SveComputeKind::AddCarryTop | SveComputeKind::SubCarryTop
                    );
                    let subtract = matches!(
                        data.op,
                        SveComputeKind::SubCarryBottom | SveComputeKind::SubCarryTop
                    );
                    let mask = BitWidth::new(bits).checked_mask_u128()?;
                    let mut result = Vec::new();
                    for pair in 0..count.checked_div(2)? {
                        let even = pair.checked_mul(2)?;
                        let selected = source.lane(even.checked_add(u32::from(top))?);
                        let added = if subtract { !selected & mask } else { selected };
                        let carry = carries.lane(even.checked_add(1)?) & 1;
                        let total = accumulator
                            .lane(even)
                            .checked_add(added)?
                            .checked_add(carry)?;
                        result.push(total & mask);
                        result.push(total.checked_shr(bits)?);
                    }
                    vector(VectorConst::from_lanes(bits, result))
                })()
            }
            // `sclamp` and `uclamp`: each lane of the accumulator at least the
            // first source's and at most the second's, `min(max(a, low), high)`,
            // as signed or unsigned integers.
            SsaOp::VectorSveCompute(data)
                if matches!(
                    data.op,
                    SveComputeKind::ClampSigned | SveComputeKind::ClampUnsigned
                ) =>
            {
                (|| {
                    let [value, low, high] = data.inputs.as_slice() else {
                        return None;
                    };
                    let bits = u32::from(data.element_bits);
                    let value = lanes(*value)?.relaned(bits)?;
                    let low = lanes(*low)?.relaned(bits)?;
                    let high = lanes(*high)?.relaned(bits)?;
                    let count = value.lanes();
                    if low.lanes() != count || high.lanes() != count || bits > 64 {
                        return None;
                    }
                    // A signed lane orders as unsigned with its sign bit flipped.
                    let bias = if data.op == SveComputeKind::ClampSigned {
                        1u128.checked_shl(bits.checked_sub(1)?)?
                    } else {
                        0
                    };
                    let order = |lane: u128| lane ^ bias;
                    let clamped = (0..count).map(|at| {
                        let (lane, least, most) = (value.lane(at), low.lane(at), high.lane(at));
                        let raised = if order(lane) < order(least) {
                            least
                        } else {
                            lane
                        };
                        if order(raised) > order(most) {
                            most
                        } else {
                            raised
                        }
                    });
                    vector(VectorConst::from_lanes(bits, clamped.collect::<Vec<_>>()))
                })()
            }
            SsaOp::VectorSveCompute(data)
                if matches!(
                    data.op,
                    SveComputeKind::SaturatingAbs | SveComputeKind::SaturatingNeg
                ) =>
            {
                (|| {
                    let source =
                        lanes(*data.inputs.last()?)?.relaned(u32::from(data.element_bits))?;
                    vector(source.saturating_sign(data.op == SveComputeKind::SaturatingNeg))
                })()
            }
            _ => return None,
        })
    }
}

/// The mask a predicated lane operation runs under and the value its inactive
/// lanes keep, as the folder knows them ([`SsaOp::VectorPredicatedUnary`]).
struct Governed {
    /// The mask's lanes; `None` where the mask is not known.
    mask: Option<VectorConst>,
    /// Whether the mask is a scalar holding a bit per lane (an AVX-512 opmask)
    /// rather than a vector of the operation's lanes (an SVE predicate).
    bits: bool,
    /// The passthrough of a merging mask; `None` where it is not known or the
    /// operation names none.
    passthrough: Option<VectorConst>,
    /// Whether the inactive lanes are zero.
    zeroing: bool,
}

impl Governed {
    /// Reads the mask and the passthrough of a predicated operation.
    fn of<T: Target>(
        get_const: &mut impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        ptr_size: PointerSize,
        mask: SsaVarId,
        passthrough: Option<SsaVarId>,
        mode: VectorMaskMode,
    ) -> Self {
        let mask = get_const(mask);
        Self {
            bits: mask.as_ref().is_some_and(|mask| !mask.is_vector()),
            mask: mask.and_then(|mask| mask.as_lanes(ptr_size)),
            passthrough: passthrough
                .and_then(&mut *get_const)
                .and_then(|passthrough| passthrough.as_lanes(ptr_size)),
            zeroing: mode == VectorMaskMode::Zero,
        }
    }

    /// Returns each lane of `computed` the mask makes active, and the inactive
    /// lanes' value elsewhere, at `lane_bits` (`computed`'s own width for 0).
    /// Only the lanes chosen are read: an unknown passthrough folds wherever
    /// the mask makes every lane active. `None` where the mask is not known.
    fn merge(&self, computed: &VectorConst, lane_bits: u32) -> Option<VectorConst> {
        let lane_bits = if lane_bits == 0 {
            computed.lane_bits()
        } else {
            lane_bits
        };
        let computed = computed.relaned(lane_bits)?;
        let mask = self.mask.as_ref()?;
        let mask = if self.bits {
            mask.lanes_of_bits(computed.lanes(), lane_bits)?
        } else {
            mask.relaned(lane_bits)?
        };
        let inactive = if self.zeroing {
            Some(VectorConst::splat(lane_bits, computed.lanes(), 0)?)
        } else {
            self.passthrough.clone()
        };
        mask.select(inactive.as_ref(), Some(&computed))
    }
}
