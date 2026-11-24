//! How folding and execution inspect host types at the numeric kernel boundary.
//!
//! The numeric kernel needs only a type's numeric layout: an integer's width
//! and signedness, a float format, a vector shape, whether it is a pointer.
//! [`TypeQueries`] answers those either from the host's ordinary queries or
//! from a coherent projection supplied by the caller, never
//! falling back from one to the other. Host metadata is borrowed, and type
//! graphs no query needs are never visited.

use crate::{
    PointerSize,
    ir::{
        float::{FloatFormat, FloatRounding, FloatRules},
        integer::IntegerType,
        value::ConstValue,
    },
    target::{Endianness, Target, TypeLayout, VectorShape},
};

/// Projects a borrowed host type into its numeric layout.
type TypeProjection<'a, T> = dyn Fn(&<T as Target>::Type) -> Option<TypeLayout> + 'a;

/// The fixed interpretation selected by one target type projection.
pub enum ReinterpretShape {
    /// A vector retaining its complete lane geometry.
    Vector(VectorShape),
    /// One scalar value of the declared bit width.
    Scalar(u32),
}

/// Selects ordinary host queries or a coherent projection without fallback.
pub struct TypeQueries<'a, T: Target> {
    /// Width used by native integer conversions.
    ptr_size: PointerSize,
    /// Host projection supplied for execution.
    inspect: Option<&'a TypeProjection<'a, T>>,
}

impl<'a, T: Target> TypeQueries<'a, T> {
    /// Preserves ordinary host inspection and custom conversion hooks.
    pub const fn ordinary(ptr_size: PointerSize) -> Self {
        Self {
            ptr_size,
            inspect: None,
        }
    }

    /// Uses a supplied projection for every operation target query.
    pub const fn projected(ptr_size: PointerSize, inspect: &'a TypeProjection<'a, T>) -> Self {
        Self {
            ptr_size,
            inspect: Some(inspect),
        }
    }

    /// Returns fixed vector geometry without inventing a runtime extent.
    pub fn vector_shape(&self, ty: &T::Type) -> Option<VectorShape> {
        match self.inspect {
            Some(inspect) => inspect(ty)?.vector_shape(),
            None => T::vector_shape(ty),
        }
    }

    /// Returns one coherent fixed target shape without repeating host inspection.
    pub fn reinterpret_shape(&self, ty: &T::Type) -> Option<ReinterpretShape> {
        match self.inspect {
            Some(inspect) => {
                let layout = inspect(ty)?;
                if layout.is_runtime_sized() {
                    return None;
                }
                match layout.vector_shape() {
                    Some(shape) => Some(ReinterpretShape::Vector(shape)),
                    None => Some(ReinterpretShape::Scalar(layout.bit_width()?)),
                }
            }
            None => match T::vector_shape(ty) {
                Some(shape) => Some(ReinterpretShape::Vector(shape)),
                None => Some(ReinterpretShape::Scalar(T::bit_width(ty)?)),
            },
        }
    }

    /// Returns the exact scalar conversion range and constant representation.
    pub fn integer_type(&self, ty: &T::Type) -> Option<IntegerType> {
        match self.inspect {
            Some(inspect) => {
                let layout = inspect(ty)?;
                (!layout.is_runtime_sized())
                    .then(|| layout.integer_conversion_type())
                    .flatten()
            }
            None => T::integer_conversion_type(ty, self.ptr_size),
        }
    }

    /// Returns a scalar floating format through the selected host projection.
    pub fn float_format(&self, ty: &T::Type) -> Option<FloatFormat> {
        match self.inspect {
            Some(inspect) => inspect(ty)?.float_format(),
            None => T::float_format(ty),
        }
    }

    /// Proves a fixed pointer type through the selected host projection.
    pub fn is_pointer(&self, ty: &T::Type) -> bool {
        match self.inspect {
            Some(inspect) => {
                inspect(ty).is_some_and(|layout| !layout.is_runtime_sized() && layout.is_pointer())
            }
            None => T::is_pointer(ty),
        }
    }

    /// Proves that a symbol conversion preserves the complete machine address.
    pub fn pointer_wide(&self, ty: &T::Type) -> bool {
        match self.inspect {
            Some(inspect) => inspect(ty).is_some_and(|layout| {
                !layout.is_runtime_sized()
                    && layout
                        .bit_width()
                        .map_or_else(|| layout.is_pointer(), |bits| bits == self.ptr_size.bits())
            }),
            None => T::bit_width(ty)
                .map_or_else(|| T::is_pointer(ty), |bits| bits == self.ptr_size.bits()),
        }
    }

    /// Returns the extent preserving an address through a bitcast.
    pub fn bitcast_width(&self, ty: &T::Type, dest: Option<u32>) -> Option<u32> {
        match self.inspect {
            Some(inspect) => {
                let layout = inspect(ty)?;
                if layout.is_runtime_sized() {
                    return None;
                }
                layout
                    .bit_width()
                    .or_else(|| layout.is_pointer().then_some(self.ptr_size.bits()))
                    .or_else(|| layout.is_unknown().then_some(dest).flatten())
            }
            None => T::bit_width(ty)
                .or_else(|| T::is_pointer(ty).then_some(self.ptr_size.bits()))
                .or_else(|| T::is_unknown(ty).then_some(dest).flatten()),
        }
    }

    /// Decodes a bitcast through the active type queries.
    pub fn decode(&self, bytes: &[u8], ty: &T::Type) -> Option<ConstValue<T>> {
        match self.inspect {
            Some(inspect) => ConstValue::from_layout_bytes(bytes, inspect(ty)?, Endianness::Little),
            None => ConstValue::from_typed_bytes(bytes, ty, Endianness::Little),
        }
    }

    /// Converts to an exact floating format under the active type queries and NaN rules.
    ///
    /// A supplied projection is the only source of target format knowledge when
    /// present; refusal and runtime extents return `None` without ordinary host
    /// inspection. Both paths retain `rules`, including unknown NaN propagation.
    pub fn rounded(
        &self,
        value: &ConstValue<T>,
        ty: &T::Type,
        unsigned: bool,
        rounding: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<ConstValue<T>> {
        match self.inspect {
            Some(inspect) => {
                let layout = inspect(ty)?;
                if layout.is_runtime_sized() {
                    return None;
                }
                value.convert_to_format(
                    layout.float_format()?,
                    unsigned,
                    rounding,
                    self.ptr_size,
                    rules,
                )
            }
            None => value.convert_rounded(ty, unsigned, rounding, self.ptr_size, rules),
        }
    }

    /// Converts an address number through the typed integer kernel or ordinary host hook.
    pub fn pointer_convert(&self, value: &ConstValue<T>, ty: &T::Type) -> Option<ConstValue<T>> {
        match self.inspect {
            Some(_) => self
                .integer_type(ty)?
                .convert(value, true, false, self.ptr_size)
                .into_value(),
            None => value.convert_to(ty, true, self.ptr_size.bytes() as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{
        analysis::consts::{Destination, Machine, OperandLayout},
        ir::{
            evaluation::{ArithmeticTrap, Evaluation},
            float::{DefaultNan, FloatBits, FloatFormat, FloatRules, NanPropagation},
            lanes::VectorConst,
            ops::{NativeKindedData, SsaOp, SystemOpKind, TrapOp, VectorCastKind},
            variable::SsaVarId,
        },
        testing::{MockTarget, MockType},
    };

    type Cv = ConstValue<MockTarget>;

    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }
    fn words(values: [u128; 4]) -> Cv {
        Cv::Vector(VectorConst::from_lanes(32, values).unwrap())
    }

    /// Every target-reading fold reads only its projection, refusing where it states no layout.
    #[test]
    fn projected_kernel_targets_cover_every_type_query_family() {
        let machine = Machine::new(PointerSize::Bit64, None, FloatRules::default());
        let dest = var(4);
        let cases = [
            (
                SsaOp::IntConv {
                    dest,
                    operand: var(0),
                    target: MockType::I64,
                    unsigned: false,
                    overflow_check: false,
                },
                64,
                vec![Cv::I8(-1)],
                Cv::I64(-1),
            ),
            (
                SsaOp::FloatToInt {
                    dest,
                    operand: var(0),
                    target: MockType::I32,
                    unsigned: false,
                    overflow_check: false,
                    source: Some(FloatFormat::Double),
                },
                32,
                vec![Cv::from(3.0f64)],
                Cv::I32(3),
            ),
            (
                SsaOp::IntToFloat {
                    dest,
                    operand: var(0),
                    target: MockType::F64,
                    unsigned: false,
                    rounding: Some(FloatRounding::NearestEven),
                    environment: None,
                },
                64,
                vec![Cv::I32(1)],
                Cv::from(1.0f64),
            ),
            (
                SsaOp::FloatConv {
                    dest,
                    operand: var(0),
                    target: MockType::BF16,
                    rounding: Some(FloatRounding::NearestEven),
                    source: Some(FloatFormat::Double),
                    environment: None,
                },
                16,
                vec![Cv::from(1.0f64)],
                Cv::float(FloatFormat::BFloat16, 0x3f80).unwrap(),
            ),
            (
                SsaOp::Bitcast {
                    dest,
                    operand: var(0),
                    target: MockType::F64,
                },
                64,
                vec![Cv::U64(0x3ff0_0000_0000_0000)],
                Cv::from(1.0f64),
            ),
            (
                SsaOp::PtrToInt {
                    dest,
                    operand: var(0),
                    target: MockType::U64,
                },
                64,
                vec![Cv::U64(7)],
                Cv::U64(7),
            ),
            (
                SsaOp::IntToPtr {
                    dest,
                    operand: var(0),
                    target: MockType::Ptr,
                },
                64,
                vec![Cv::U64(7)],
                Cv::NativeUInt(7),
            ),
            (
                SsaOp::PtrToInt {
                    dest,
                    operand: var(0),
                    target: MockType::U64,
                },
                64,
                vec![Cv::Symbol(7)],
                Cv::Symbol(7),
            ),
            (
                SsaOp::VectorSplat {
                    dest,
                    value: var(0),
                    vector_type: MockType::V4I32,
                },
                128,
                vec![Cv::U32(3)],
                words([3; 4]),
            ),
            (
                SsaOp::VectorSlice {
                    dest,
                    vector: var(0),
                    offset: var(1),
                    length: var(2),
                    target_type: MockType::V4I32,
                },
                128,
                vec![words([1, 2, 3, 4]), Cv::U32(0), Cv::U32(128)],
                words([1, 2, 3, 4]),
            ),
            (
                SsaOp::VectorInsertSlice {
                    dest,
                    vector: var(0),
                    offset: var(1),
                    value: var(2),
                    target_type: MockType::V4I32,
                },
                128,
                vec![
                    words([0; 4]),
                    Cv::U32(0),
                    Cv::Vector(VectorConst::from_lanes(32, [7, 8]).unwrap()),
                ],
                words([7, 8, 0, 0]),
            ),
            (
                SsaOp::VectorConcat {
                    dest,
                    left: var(0),
                    right: var(1),
                    target_type: MockType::V4I32,
                },
                128,
                vec![
                    Cv::Vector(VectorConst::from_lanes(32, [1, 2]).unwrap()),
                    Cv::Vector(VectorConst::from_lanes(32, [3, 4]).unwrap()),
                ],
                words([1, 2, 3, 4]),
            ),
            (
                SsaOp::VectorReinterpret {
                    dest,
                    value: var(0),
                    target_type: MockType::U128,
                },
                128,
                vec![Cv::U64(7)],
                Cv::of_lane(128, 7).unwrap(),
            ),
            (
                SsaOp::VectorCast {
                    dest,
                    value: var(0),
                    target_type: MockType::V4I32,
                    from: VectorCastKind::Unsigned,
                    to: VectorCastKind::Unsigned,
                    rounding: None,
                    saturate: false,
                    environment: None,
                    format: None,
                },
                128,
                vec![words([1, 2, 3, 4])],
                words([1, 2, 3, 4]),
            ),
        ];
        for (op, bits, inputs, expected) in cases {
            let calls = Cell::new(0);
            let inspect = |ty: &MockType| {
                calls.set(calls.get() + 1);
                Some(TypeLayout::of::<MockTarget>(ty, machine.ptr_size))
            };
            let types = TypeQueries::projected(machine.ptr_size, &inspect);
            let actual = machine.evaluate_with_type_queries(
                &op,
                Some(bits),
                |v| inputs.get(v.index()).cloned(),
                |_| OperandLayout::default(),
                &types,
            );
            assert_eq!(actual, Some(expected), "{op:?}");
            assert_eq!(calls.get(), 1, "{op:?}");
            let refused = TypeQueries::projected(machine.ptr_size, &|_| None);
            assert_eq!(
                machine.evaluate_with_type_queries(
                    &op,
                    Some(bits),
                    |v| inputs.get(v.index()).cloned(),
                    |_| OperandLayout::default(),
                    &refused
                ),
                None,
                "{op:?}"
            );
        }
    }

    /// Projected float conversions retain explicit source bits, NaN rules and type refusal.
    #[test]
    fn projected_nan_conversions_preserve_rules_and_refusal() {
        for (target, format, width, propagated, canonical, one) in [
            (
                MockType::BF16,
                FloatFormat::BFloat16,
                16,
                0x7fe0,
                0x7fc0,
                0x3f80,
            ),
            (
                MockType::F80,
                FloatFormat::X87Extended,
                80,
                0x7fff_e000_0000_0000_0000,
                0x7fff_c000_0000_0000_0000,
                0x3fff_8000_0000_0000_0000,
            ),
        ] {
            for nan_conversion in [
                None,
                Some(NanPropagation::Operand),
                Some(NanPropagation::Default),
            ] {
                let rules = FloatRules {
                    out_of_range: None,
                    default_nan: Some(DefaultNan::Positive),
                    nan_conversion,
                    ..FloatRules::NONE
                };
                let machine = Machine::new(PointerSize::Bit64, None, rules);
                for nan in [false, true] {
                    let raw = if nan {
                        0x7ff4_0000_0000_0000
                    } else {
                        0x3ff0_0000_0000_0000
                    };
                    let expected = if nan {
                        match nan_conversion {
                            None => None,
                            Some(NanPropagation::Operand) => Some(propagated),
                            Some(NanPropagation::Default) => Some(canonical),
                        }
                    } else {
                        Some(one)
                    }
                    .map(|bits| Cv::Float(FloatBits::new(format, bits).unwrap()));
                    for raw_source in [false, true] {
                        let source = if raw_source {
                            Cv::U64(raw)
                        } else {
                            Cv::Float(FloatBits::new(FloatFormat::Double, u128::from(raw)).unwrap())
                        };
                        let op = SsaOp::FloatConv {
                            dest: var(1),
                            operand: var(0),
                            target,
                            rounding: Some(FloatRounding::NearestEven),
                            source: Some(FloatFormat::Double),
                            environment: None,
                        };
                        let fold = |types: &TypeQueries<'_, MockTarget>| {
                            machine.evaluate_with_type_queries(
                                &op,
                                Some(width),
                                |v| (v == var(0)).then(|| source.clone()),
                                |_| OperandLayout::default(),
                                types,
                            )
                        };
                        assert_eq!(fold(&TypeQueries::ordinary(machine.ptr_size)), expected);
                        let calls = Cell::new(0);
                        let inspect = |ty: &MockType| {
                            calls.set(calls.get() + 1);
                            Some(TypeLayout::of::<MockTarget>(ty, machine.ptr_size))
                        };
                        assert_eq!(
                            fold(&TypeQueries::projected(machine.ptr_size, &inspect)),
                            expected
                        );
                        assert_eq!(calls.get(), 1);
                        let refuse = |_: &MockType| {
                            calls.set(calls.get() + 1);
                            None
                        };
                        assert_eq!(
                            fold(&TypeQueries::projected(machine.ptr_size, &refuse)),
                            None
                        );
                        assert_eq!(calls.get(), 2);
                    }
                }
            }
        }
    }

    /// **A conditional trap fires exactly where its comparison holds.** `teq`
    /// of equal inputs is a fault, of different inputs its false condition, of
    /// an unknown input undecided; an MPX lower-bound check fires where the
    /// bound lies above the value, unsigned. A trap on a flag or on bounds in
    /// memory is undecided, a software interrupt, which always transfers, has
    /// no outcome here, and no trap has a value: it defines nothing.
    #[test]
    fn a_conditional_trap_fires_where_its_comparison_holds() {
        let machine = Machine::new(PointerSize::Bit64, None, FloatRules::default());
        let trap = |op| {
            SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
                kind: SystemOpKind::Trap { op, vector: None },
                mnemonic: String::new(),
                metadata: None,
                outputs: Vec::new(),
                inputs: vec![var(0), var(1)],
            }))
        };
        let constants = |left: Option<u64>, right: u64| {
            move |v: SsaVarId| {
                if v == var(0) {
                    left.map(Cv::U64)
                } else {
                    Some(Cv::U64(right))
                }
            }
        };
        for (op, left, right, expected) in [
            (
                TrapOp::TrapIfEqual,
                Some(3),
                3,
                Some(Evaluation::Trap(ArithmeticTrap::Condition)),
            ),
            (
                TrapOp::TrapIfEqual,
                Some(3),
                4,
                Some(Evaluation::Value(Cv::False)),
            ),
            (TrapOp::TrapIfEqual, None, 4, Some(Evaluation::Unknown)),
            (
                TrapOp::BoundsLower,
                Some(5),
                3,
                Some(Evaluation::Trap(ArithmeticTrap::Condition)),
            ),
            (
                TrapOp::BoundsLower,
                Some(3),
                5,
                Some(Evaluation::Value(Cv::False)),
            ),
            (TrapOp::OverflowTrap, Some(3), 3, Some(Evaluation::Unknown)),
            (TrapOp::BoundsCheck, Some(3), 3, Some(Evaluation::Unknown)),
            (TrapOp::SoftwareInterrupt, Some(3), 3, None),
        ] {
            let op = trap(op);
            assert_eq!(
                machine.evaluate_control(&op, None, constants(left, right)),
                expected,
                "{op:?} {left:?} {right}"
            );
            assert_eq!(machine.evaluate(&op, None, constants(left, right)), None);
        }
    }

    /// A refused conversion target cannot become an overflow proof or a value.
    #[test]
    fn checked_kernel_type_refusal_is_distinct_from_a_proven_trap() {
        let machine = Machine::new(PointerSize::Bit64, None, FloatRules::default());
        let op = SsaOp::IntConv {
            dest: var(1),
            operand: var(0),
            target: MockType::I8,
            unsigned: false,
            overflow_check: true,
        };
        let inspect = |ty: &MockType| Some(TypeLayout::of::<MockTarget>(ty, machine.ptr_size));
        let types = TypeQueries::projected(machine.ptr_size, &inspect);
        assert_eq!(
            machine.evaluate_control_with_type_queries(
                &op,
                Some(8),
                |_| Some(Cv::U16(256)),
                |_| OperandLayout::default(),
                &types
            ),
            Some(Evaluation::Trap(ArithmeticTrap::Overflow))
        );
        let refused = TypeQueries::projected(machine.ptr_size, &|_| None);
        assert_eq!(
            machine.evaluate_control_with_type_queries(
                &op,
                Some(8),
                |_| Some(Cv::U16(256)),
                |_| OperandLayout::default(),
                &refused
            ),
            Some(Evaluation::Unsupported)
        );
    }

    /// Plain arithmetic has no target type to inspect.
    #[test]
    fn primitive_kernel_skips_type_inspection() {
        let machine = Machine::new(PointerSize::Bit64, None, FloatRules::default());
        let op = SsaOp::<MockTarget>::Add {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        let types = TypeQueries::projected(machine.ptr_size, &|_| {
            panic!("primitive arithmetic queried type metadata")
        });
        assert_eq!(
            machine.evaluate_with_type_queries(
                &op,
                Destination {
                    bits: Some(32),
                    lanes: None,
                    lane_format: None,
                    integer: true,
                    integer_type: None,
                },
                |_| Some(Cv::U32(3)),
                |_| OperandLayout::default(),
                &types
            ),
            Some(Cv::U32(6))
        );
    }
}
