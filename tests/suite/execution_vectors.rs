//! Whole-function execution of typed scalar and vector SSA operations.

use std::{error::Error, io, process};

use analyssa::{
    PointerSize,
    analysis::{
        ConstEvaluator, ControlFlow, Machine, SsaEvaluator, SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        constraints::Constraint,
        consts::{CheckedEvaluation, Destination, OperandLayout},
        context::AnalysisContext,
        execution::{Completion, ReplacementProof, StaticExecution, Value},
        image::{ImageData, ImageRegion, Mutability},
        symbolic::{SymbolicEvaluator, SymbolicExpr},
    },
    ir::{
        evaluation::{ArithmeticTrap, Evaluation, NumericOp},
        float::{FloatFormat, FloatRounding},
        function::{SsaBlockBuilder, SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        lanes::VectorConst,
        ops::{
            AccessQualifiers, CmpKind, ReduceGroups, SsaOp, VectorBinaryKind, VectorBitmaskKind,
            VectorCastKind, VectorElement, VectorReduceKind, VectorShuffleSelector,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    passes::residual::fold_constants,
    target::{Endianness, VectorShuffleLane, VectorShuffleMask},
    testing::{MACHINE, MockTarget, MockType},
};

/// Returns a required fixture input; a missing input makes the fixture malformed.
fn operand(inputs: &[SsaVarId], index: usize) -> SsaVarId {
    inputs
        .get(index)
        .copied()
        .unwrap_or_else(|| process::abort())
}

/// Builds a fixed, independently stated lane fixture.
fn vector(bits: u32, lanes: &[u128]) -> Result<ConstValue<MockTarget>, Box<dyn Error>> {
    VectorConst::from_lanes(bits, lanes.iter().copied())
        .map(ConstValue::Vector)
        .ok_or_else(|| io::Error::other("invalid vector fixture").into())
}

/// Executes constants, the supplied operation, and a real return through the shared explorer.
fn execute(
    inputs: &[(MockType, ConstValue<MockTarget>)],
    ty: MockType,
    operation: impl FnOnce(SsaVarId, &[SsaVarId]) -> SsaOp<MockTarget>,
) -> Result<Value<MockTarget>, Box<dyn Error>> {
    execute_machine(inputs, ty, operation, MACHINE)
}

/// Executes the same typed fixture under an explicit target machine.
fn execute_machine(
    inputs: &[(MockType, ConstValue<MockTarget>)],
    ty: MockType,
    operation: impl FnOnce(SsaVarId, &[SsaVarId]) -> SsaOp<MockTarget>,
    machine: Machine,
) -> Result<Value<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let mut returned = None;
    builder.in_block(0, |block| {
        let mut variables = Vec::new();
        for (ty, constant) in inputs {
            variables.push(block.const_value(SsaDefSpec::tmp(*ty), constant.clone())?);
        }
        let result = block.emit_def(SsaDefSpec::tmp(ty), |dest| operation(dest, &variables))?;
        returned = Some(result);
        block.ret(Some(result))
    })?;
    let ssa = builder.finish()?;
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&ssa, &context).run();
    if !facts.is_complete() {
        return Err(io::Error::other(format!(
            "vector execution refused: {:?}",
            facts.completion()
        ))
        .into());
    }
    let value = facts
        .returned()
        .cloned()
        .ok_or_else(|| io::Error::other("missing vector return"))?;
    // The same independently specified execution fixtures cover the symbolic
    // SSA consumer. Its concrete numeric carrier admission is bounded at 128 bits.
    if inputs.iter().all(|(_, value)| {
        value.scalar_integer_pattern(machine.ptr_size).is_some()
            || value.is_float()
            || value.is_bool()
    }) {
        let mut symbolic = SymbolicEvaluator::with_machine(&ssa, machine);
        symbolic.evaluate_block(0);
        assert_eq!(
            returned
                .and_then(|var| symbolic.get_expression(var))
                .and_then(SymbolicExpr::as_constant),
            value.constant(),
            "symbolic SSA evaluation and interpreter disagree",
        );
    }
    let mut constants = ConstEvaluator::new(&ssa, machine);
    assert_eq!(
        returned
            .and_then(|var| constants.evaluate_var(var))
            .as_ref(),
        value.constant(),
        "constant evaluator and interpreter disagree"
    );
    if returned
        .and_then(|var| ssa.recorded_definition(var))
        .is_some_and(|definition| {
            matches!(
                definition.op(),
                SsaOp::Add { .. }
                    | SsaOp::Sub { .. }
                    | SsaOp::Mul { .. }
                    | SsaOp::Div { .. }
                    | SsaOp::Rem { .. }
                    | SsaOp::And { .. }
                    | SsaOp::Or { .. }
                    | SsaOp::Xor { .. }
                    | SsaOp::Shl { .. }
                    | SsaOp::Shr { .. }
                    | SsaOp::Rol { .. }
                    | SsaOp::Ror { .. }
                    | SsaOp::Neg { .. }
                    | SsaOp::Not { .. }
                    | SsaOp::BSwap { .. }
                    | SsaOp::BRev { .. }
                    | SsaOp::AddOvf { .. }
                    | SsaOp::SubOvf { .. }
                    | SsaOp::MulOvf { .. }
                    | SsaOp::Ceq { .. }
                    | SsaOp::Clt { .. }
                    | SsaOp::Cgt { .. }
                    | SsaOp::IntConv { .. }
                    | SsaOp::IntToPtr { .. }
                    | SsaOp::PtrToInt { .. }
                    | SsaOp::IntToFloat { .. }
                    | SsaOp::FloatToInt { .. }
                    | SsaOp::FloatConv { .. }
                    | SsaOp::Ckfinite { .. }
            )
        })
    {
        let mut symbolic = SsaEvaluator::new(&ssa, machine);
        symbolic.evaluate_block(0);
        assert_eq!(
            returned.and_then(|var| symbolic.get_concrete(var)),
            value.constant(),
            "symbolic evaluator and interpreter disagree"
        );
    }
    Ok(value)
}

/// Both mask interpretations select the specified lane bit through SSA dispatch.
#[test]
fn vector_bitmasks_execute_with_independent_lane_bit_oracles() -> Result<(), Box<dyn Error>> {
    for (kind, expected) in [
        (VectorBitmaskKind::LaneMostSignificantBits, 5),
        (VectorBitmaskKind::PredicateBits, 12),
    ] {
        let input = vector(32, &[0x8000_0000, 0, 0xffff_ffff, 1])?;
        let result = execute(&[(MockType::V4I32, input)], MockType::U8, |dest, inputs| {
            SsaOp::VectorBitmask {
                dest,
                value: operand(inputs, 0),
                kind,
            }
        })?;
        assert_eq!(result, Value::Const(ConstValue::U8(expected)));
    }
    Ok(())
}

/// Runtime index vectors select one or two tables; an unstated index remains unknown.
#[test]
fn vector_permutation_executes_tables_and_rejects_out_of_domain_indices()
-> Result<(), Box<dyn Error>> {
    for (tables, indices, expected) in [
        (
            vec![vec![10, 20, 30, 40]],
            vec![3, 0, 3, 1],
            Some(vec![40, 10, 40, 20]),
        ),
        (
            vec![vec![10, 20, 30, 40], vec![50, 60, 70, 80]],
            vec![0, 7, 2, 5],
            Some(vec![10, 80, 30, 60]),
        ),
        (vec![vec![10, 20, 30, 40]], vec![4, 0, 1, 2], None),
    ] {
        let mut inputs = Vec::new();
        for table in tables {
            inputs.push((MockType::V4I32, vector(32, &table)?));
        }
        inputs.push((MockType::V4I32, vector(32, &indices)?));
        let result = execute(&inputs, MockType::V4I32, |dest, inputs| {
            let (&[left, ref right @ ..], Some(&index)) = (
                inputs.split_last().map_or(&[][..], |(_, tables)| tables),
                inputs.last(),
            ) else {
                process::abort()
            };
            SsaOp::VectorShuffle {
                dest,
                left,
                right: right.first().copied(),
                selector: VectorShuffleSelector::Index(index),
            }
        })?;
        match expected {
            Some(values) => assert_eq!(result, Value::Const(vector(32, &values)?)),
            None => assert!(result.constant().is_none()),
        }
    }
    Ok(())
}

/// A grouped reduction of pairs into wider lanes honors sign extension and
/// the pair's difference through the interpreter.
#[test]
fn horizontal_integer_groups_execute_signed_and_unsigned_semantics() -> Result<(), Box<dyn Error>> {
    for (unsigned, subtract, expected) in [
        (false, false, [1, (-2147483641i64) as u64]),
        (true, false, [4294967297, 2147483655]),
        (false, true, [(-3i64) as u64, (-2147483655i64) as u64]),
        (true, true, [4294967293, 2147483641]),
    ] {
        let input = vector(32, &[0xffff_ffff, 2, 0x8000_0000, 7])?;
        let result = execute(
            &[(MockType::V4I32, input)],
            MockType::U128,
            |dest, inputs| match *inputs {
                [value] => SsaOp::VectorReduce {
                    dest,
                    value,
                    kind: if subtract {
                        VectorReduceKind::Difference
                    } else {
                        VectorReduceKind::Add
                    },
                    element: VectorElement::integer(32, Some(!unsigned)),
                    initial: None,
                    ordered: false,
                    groups: Some(ReduceGroups { lanes: 2, bits: 64 }),
                    environment: None,
                },
                // One source; anything else computes nothing.
                _ => SsaOp::Nop,
            },
        )?;
        assert_eq!(result, Value::Const(vector(64, &expected.map(u128::from))?));
    }
    Ok(())
}

/// A saturating cast clamps each lane to the destination lane's range: signed
/// to signed, unsigned to unsigned, and signed to unsigned, which clamps a
/// negative lane to zero.
#[test]
fn saturating_cast_clamps_each_lane_to_the_destination_range() -> Result<(), Box<dyn Error>> {
    for (from, to, source, expected) in [
        (
            VectorCastKind::Signed,
            VectorCastKind::Signed,
            [-32769i64, -32768, 32767, 32768],
            [32768u128, 32768, 32767, 32767],
        ),
        (
            VectorCastKind::Unsigned,
            VectorCastKind::Unsigned,
            [0, 65535, 65536, 4294967295],
            [0, 65535, 65535, 65535],
        ),
        (
            VectorCastKind::Signed,
            VectorCastKind::Unsigned,
            [-1, 0, 65535, 65536],
            [0, 0, 65535, 65535],
        ),
    ] {
        let lanes = source.map(|value| u128::from(value as u32));
        let input = vector(32, &lanes)?;
        let result = execute(
            &[(MockType::V4I32, input)],
            MockType::V4I16,
            |dest, inputs| match *inputs {
                [value] => SsaOp::VectorCast {
                    dest,
                    value,
                    target_type: MockType::V4I16,
                    from,
                    to,
                    rounding: None,
                    saturate: true,
                    environment: None,
                    format: None,
                },
                // One source; anything else computes nothing.
                _ => SsaOp::Nop,
            },
        )?;
        assert_eq!(result, Value::Const(vector(16, &expected)?));
    }
    Ok(())
}

/// A scalar addition carries across the lanes used to encode a packed register constant.
#[test]
fn scalar_arithmetic_obeys_declared_width_for_packed_constant_carriers()
-> Result<(), Box<dyn Error>> {
    let packed = vector(8, &[255, 0, 0, 0])?;
    let result = execute(
        &[(MockType::U32, packed), (MockType::U32, ConstValue::U32(1))],
        MockType::U32,
        |dest, inputs| SsaOp::Add {
            dest,
            left: operand(inputs, 0),
            right: operand(inputs, 1),
            flags: None,
        },
    )?;
    let bytes = result
        .constant()
        .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit64));
    assert_eq!(bytes, Some(256u32.to_le_bytes().to_vec()));
    Ok(())
}

/// Scalar operations whose operands describe a complete integer register.
#[derive(Debug, Clone, Copy)]
enum IntegerOp {
    Add,
    Sub,
    Mul,
    Div(bool),
    Rem(bool),
    And,
    Or,
    Xor,
    Shl,
    Shr(bool),
    Rol,
    Ror,
    Neg,
    Not,
    BSwap,
    BRev,
    AddOvf,
    SubOvf,
    MulOvf,
}

impl IntegerOp {
    /// Builds an operation over the fixture's explicitly supplied operands.
    fn operation(self, dest: SsaVarId, inputs: &[SsaVarId]) -> SsaOp<MockTarget> {
        let (left, right) = (operand(inputs, 0), operand(inputs, 1));
        match self {
            Self::Add => SsaOp::Add {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Sub => SsaOp::Sub {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Mul => SsaOp::Mul {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Div(unsigned) => SsaOp::Div {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            Self::Rem(unsigned) => SsaOp::Rem {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            Self::And => SsaOp::And {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Or => SsaOp::Or {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Xor => SsaOp::Xor {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Shl => SsaOp::Shl {
                dest,
                value: left,
                amount: right,
                flags: None,
            },
            Self::Shr(unsigned) => SsaOp::Shr {
                dest,
                value: left,
                amount: right,
                unsigned,
                flags: None,
            },
            Self::Rol => SsaOp::Rol {
                dest,
                value: left,
                amount: right,
            },
            Self::Ror => SsaOp::Ror {
                dest,
                value: left,
                amount: right,
            },
            Self::Neg => SsaOp::Neg {
                dest,
                operand: left,
                flags: None,
            },
            Self::Not => SsaOp::Not {
                dest,
                operand: left,
                flags: None,
            },
            Self::BSwap => SsaOp::BSwap { dest, src: left },
            Self::BRev => SsaOp::BRev { dest, src: left },
            Self::AddOvf => SsaOp::AddOvf {
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            Self::SubOvf => SsaOp::SubOvf {
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            Self::MulOvf => SsaOp::MulOvf {
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            },
        }
    }
}

/// Packed, primitive and wider literal carriers execute the same scalar integer operations.
#[test]
fn declared_integer_operations_use_whole_register_patterns() -> Result<(), Box<dyn Error>> {
    for (operation, left, right, expected) in [
        (IntegerOp::Add, 0xffff_ffffu32, 1u32, 0u32),
        (IntegerOp::Sub, 0x100, 1, 0xff),
        (IntegerOp::Mul, 0xff, 2, 0x1fe),
        (IntegerOp::Div(false), 0xffff_fff8, 3, 0xffff_fffe),
        (IntegerOp::Div(true), 0xffff_ffff, 3, 0x5555_5555),
        (IntegerOp::Rem(false), 0xffff_fff8, 3, 0xffff_fffe),
        (IntegerOp::Rem(true), 0xffff_ffff, 3, 0),
        (IntegerOp::And, 0x1234_5678, 0xff00_ff00, 0x1200_5600),
        (IntegerOp::Or, 0x1234_0000, 0x5678, 0x1234_5678),
        (IntegerOp::Xor, 0xaaaa_5555, 0xffff_0000, 0x5555_5555),
        (IntegerOp::Shl, 0x0001_0000, 8, 0x0100_0000),
        (IntegerOp::Shr(false), 0x8000_0000, 4, 0xf800_0000),
        (IntegerOp::Shr(true), 0x8000_0000, 4, 0x0800_0000),
        (IntegerOp::Rol, 0x8000_0001, 1, 3),
        (IntegerOp::Ror, 0x8000_0001, 1, 0xc000_0000),
        (IntegerOp::Neg, 0x100, 0, 0xffff_ff00),
        (IntegerOp::Not, 0x0000_ff00, 0, 0xffff_00ff),
        (IntegerOp::BSwap, 0x1234_5678, 0, 0x7856_3412),
        (IntegerOp::BRev, 0x1234_5678, 0, 0x1e6a_2c48),
        (IntegerOp::AddOvf, 40, 2, 42),
        (IntegerOp::SubOvf, 40, 2, 38),
        (IntegerOp::MulOvf, 40, 2, 80),
    ] {
        for carrier in 0..7 {
            let left = match carrier {
                0 => vector(8, &left.to_le_bytes().map(u128::from))?,
                1 => vector(32, &[u128::from(left)])?,
                2 => ConstValue::U32(left),
                3 => ConstValue::U64(u64::from(left)),
                4 => ConstValue::I64(i64::from(left)),
                5 => ConstValue::NativeInt(i64::from(left)),
                _ => ConstValue::NativeUInt(u64::from(left)),
            };
            let result = execute(
                &[
                    (MockType::U32, left),
                    (MockType::U32, ConstValue::U32(right)),
                ],
                MockType::U32,
                |dest, inputs| operation.operation(dest, inputs),
            )?;
            assert_eq!(
                result
                    .constant()
                    .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit64)),
                Some(expected.to_le_bytes().to_vec()),
                "{operation:?}, carrier {carrier}"
            );
        }
    }
    Ok(())
}

/// Comparison operands retain their data width even though their result is one bit wide.
#[test]
fn scalar_comparison_inputs_are_not_reinterpreted_at_boolean_result_width()
-> Result<(), Box<dyn Error>> {
    let result = execute(
        &[
            (MockType::U32, ConstValue::U32(0x100)),
            (MockType::U32, ConstValue::U32(0)),
        ],
        MockType::I1,
        |dest, inputs| SsaOp::Ceq {
            dest,
            left: operand(inputs, 0),
            right: operand(inputs, 1),
            signaling: false,
        },
    )?;
    assert_eq!(result.constant().and_then(ConstValue::as_bool), Some(false));
    Ok(())
}

/// Explicit vector destinations continue to add their lanes independently.
#[test]
fn vector_arithmetic_preserves_lane_boundaries() -> Result<(), Box<dyn Error>> {
    let result = execute(
        &[
            (MockType::V4I32, vector(32, &[0xffff_ffff, 0, 10, 20])?),
            (MockType::V4I32, vector(32, &[1, 2, 3, 4])?),
        ],
        MockType::V4I32,
        |dest, inputs| IntegerOp::Add.operation(dest, inputs),
    )?;
    assert_eq!(result, Value::Const(vector(32, &[0, 2, 13, 24])?));
    Ok(())
}

/// Carries propagate through each scalar register width, including a full 128-bit value.
#[test]
fn packed_register_carries_follow_every_supported_scalar_width() -> Result<(), Box<dyn Error>> {
    for (ty, width, left, expected) in [
        (MockType::U8, 8u32, 0xffu128, 0u128),
        (MockType::U16, 16, 0xff, 0x100),
        (MockType::U32, 32, 0xffff, 0x1_0000),
        (MockType::U64, 64, 0xffff_ffff, 0x1_0000_0000),
        (MockType::U128, 128, u128::from(u64::MAX), 1u128 << 64),
    ] {
        let bytes = width.div_ceil(8) as usize;
        let lanes: Vec<_> = left
            .to_le_bytes()
            .into_iter()
            .take(bytes)
            .map(u128::from)
            .collect();
        let result = execute(
            &[(ty, vector(8, &lanes)?), (ty, ConstValue::U8(1))],
            ty,
            |dest, inputs| IntegerOp::Add.operation(dest, inputs),
        )?;
        assert_eq!(
            result
                .constant()
                .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit64)),
            Some(
                expected
                    .to_le_bytes()
                    .into_iter()
                    .take(bytes)
                    .collect::<Vec<_>>()
            ),
            "width {width}"
        );
    }
    Ok(())
}

/// A rotation count retains its complete value when the register width is not a power of two.
#[test]
fn scalar_rotate_counts_keep_their_own_width() -> Result<(), Box<dyn Error>> {
    let inputs = [vector(24, &[1])?, ConstValue::U32(1 << 24)];
    let vars = [SsaVarId::from_index(0), SsaVarId::from_index(1)];
    let op = IntegerOp::Rol.operation(SsaVarId::from_index(2), &vars);
    let result = MACHINE.evaluate(
        &op,
        Destination {
            bits: Some(24),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: None,
        },
        |var| inputs.get(var.index()).cloned(),
    );
    assert_eq!(
        result.and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit64)),
        Some(vec![0, 0, 1])
    );
    Ok(())
}

/// Checked packed arithmetic proves overflow before returning or publishing its result.
#[test]
fn packed_checked_overflow_ends_the_execution_path() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let packed = vector(2, &[3, 3, 3, 1])?;
    builder.in_block(0, |block| {
        let left = block.const_value(SsaDefSpec::tmp(MockType::I8), packed)?;
        let right = block.const_value(SsaDefSpec::tmp(MockType::I8), ConstValue::I8(1))?;
        let sum = block.emit_def(SsaDefSpec::tmp(MockType::I8), |dest| SsaOp::AddOvf {
            dest,
            left,
            right,
            unsigned: false,
            flags: None,
        })?;
        block.ret(Some(sum))
    })?;
    let ssa = builder.finish()?;
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete());
    assert_eq!(facts.usage().traps, 1);
    assert!(facts.returned().is_none());
    Ok(())
}

/// Native carriers mask and sign-extend at a 32-bit machine's width before arithmetic.
#[test]
fn native_integer_carriers_follow_the_target_pointer_width() -> Result<(), Box<dyn Error>> {
    for unsigned in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let value = if unsigned {
                ConstValue::NativeUInt(0xffff_ffff)
            } else {
                ConstValue::NativeInt(0xffff_ffff)
            };
            let left = block.const_value(SsaDefSpec::tmp(MockType::U32), value)?;
            let amount = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(1))?;
            let result = block.emit_def(SsaDefSpec::tmp(MockType::U32), |dest| SsaOp::Shr {
                dest,
                value: left,
                amount,
                unsigned: false,
                flags: None,
            })?;
            block.ret(Some(result))
        })?;
        let ssa = builder.finish()?;
        let context = AnalysisContext::new(Machine {
            ptr_size: PointerSize::Bit32,
            ..MACHINE
        });
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert_eq!(
            facts
                .returned()
                .and_then(Value::constant)
                .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit32)),
            Some(vec![0xff; 4])
        );
    }
    Ok(())
}

/// Packed literals retain the source lane geometry of an explicitly lane-based consumer.
#[test]
fn packed_integer_literals_preserve_narrowing_geometry() -> Result<(), Box<dyn Error>> {
    let result = execute(
        &[(MockType::U64, vector(16, &[0x7fff, 0x8000, 1, 0xffff])?)],
        MockType::V4I8,
        |dest, inputs| {
            match *inputs {
                [value] => SsaOp::VectorCast {
                    dest,
                    value,
                    target_type: MockType::V4I8,
                    from: VectorCastKind::Signed,
                    to: VectorCastKind::Signed,
                    rounding: None,
                    saturate: true,
                    environment: None,
                    format: None,
                },
                // One source; anything else computes nothing.
                _ => SsaOp::Nop,
            }
        },
    )?;
    assert_eq!(result, Value::Const(vector(8, &[127, 128, 1, 255])?));
    Ok(())
}

/// Primitive literals and copies carry the declared width in either evaluator.
#[test]
fn primitive_literals_obey_declared_scalar_width() -> Result<(), Box<dyn Error>> {
    for (value, expected) in [
        (ConstValue::I8(-1), u32::MAX),
        (ConstValue::U8(255), 255),
        (ConstValue::U64(0x1_0000_0001), 1),
    ] {
        let result = execute(&[(MockType::U32, value)], MockType::U32, |dest, inputs| {
            SsaOp::Copy {
                dest,
                src: operand(inputs, 0),
            }
        })?;
        assert_eq!(
            result
                .constant()
                .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit64)),
            Some(expected.to_le_bytes().to_vec())
        );
    }
    Ok(())
}

/// Wide scalar shift and rotate counts are consumed before narrowing to lane storage.
#[test]
fn scalar_lane_counts_preserve_high_bits_before_range_or_modulo() -> Result<(), Box<dyn Error>> {
    for width in [1u32, 2, 7, 24, 65, 128] {
        let value = vector(width, &[1])?;
        let high = 1u128 << 100;
        for count in [ConstValue::U64(1 << 40), vector(128, &[high])?] {
            let bits = if matches!(count, ConstValue::U64(_)) {
                1u128 << 40
            } else {
                high
            };
            for right in [false, true] {
                let distance = (bits % u128::from(width)) as u32;
                let expected = if right {
                    1u128 << ((width - distance) % width)
                } else {
                    1u128 << distance
                };
                assert_eq!(
                    value.rotate(&count, right, PointerSize::Bit64),
                    Some(vector(width, &[expected])?)
                );
            }
            assert!(value.shl(&count, PointerSize::Bit64).is_none());
            assert!(value.shr(&count, true, PointerSize::Bit64).is_none());
            assert!(value.shr(&count, false, PointerSize::Bit64).is_none());
        }
    }
    Ok(())
}

/// Checked operations use the same attested packed width at their public entry point.
#[test]
fn checked_machine_entry_reads_packed_registers_at_declared_width() -> Result<(), Box<dyn Error>> {
    let vars = [SsaVarId::from_index(0), SsaVarId::from_index(1)];
    let operation = IntegerOp::AddOvf.operation(SsaVarId::from_index(2), &vars);
    for (left, expected) in [
        (0x7eu128, CheckedEvaluation::Value(ConstValue::U8(127))),
        (0x7f, CheckedEvaluation::Overflow),
    ] {
        let lanes: Vec<_> = left
            .to_le_bytes()
            .into_iter()
            .take(1)
            .flat_map(|byte| [u128::from(byte & 15), u128::from(byte >> 4)])
            .collect();
        let inputs = [vector(4, &lanes)?, ConstValue::U8(1)];
        assert_eq!(
            MACHINE.evaluate_checked(&operation, Some(8), |var| inputs.get(var.index()).cloned()),
            Some(expected)
        );
        assert_eq!(
            MACHINE.evaluate_checked(&operation, None, |var| inputs.get(var.index()).cloned()),
            None
        );
    }
    Ok(())
}

/// Proven overflow counts a trap and publishes no normal return value.
#[test]
fn proven_overflow_is_counted_without_a_return_value() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        block.const_value(
            SsaDefSpec::tmp(MockType::Ref),
            ConstValue::DecryptedString("decoded data".into()),
        )?;
        let left = block.const_value(SsaDefSpec::tmp(MockType::I8), ConstValue::I8(127))?;
        let right = block.const_value(SsaDefSpec::tmp(MockType::I8), ConstValue::I8(1))?;
        let sum = block.emit_def(SsaDefSpec::tmp(MockType::I8), |dest| SsaOp::AddOvf {
            dest,
            left,
            right,
            unsigned: false,
            flags: None,
        })?;
        block.ret(Some(sum))
    })?;
    let ssa = builder.finish()?;
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete());
    assert_eq!(facts.usage().traps, 1);
    assert!(facts.returned().is_none());
    Ok(())
}

/// A step-limit stop keeps its reason instead of observing definitions after the stop.
#[test]
fn step_limit_precedes_unnecessary_definition_observation() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        block.const_value(
            SsaDefSpec::tmp(MockType::Ref),
            ConstValue::DecryptedString("decoded data".into()),
        )?;
        let value = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
        block.ret(Some(value))
    })?;
    let ssa = builder.finish()?;
    let context = AnalysisContext::new(MACHINE)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionSteps, 2));
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(matches!(facts.completion(), Completion::Exhausted(reason)
        if reason.budget == Budget::ExecutionSteps));
    assert_eq!(facts.usage().steps, 2);
    assert_eq!(facts.usage().traps, 0);
    assert!(facts.returned().is_none());
    Ok(())
}

/// The scalar register widths represented by the fixture target.
const SCALAR_WIDTHS: [(MockType, u32); 5] = [
    (MockType::U8, 8),
    (MockType::U16, 16),
    (MockType::U32, 32),
    (MockType::U64, 64),
    (MockType::U128, 128),
];

/// An independently constructed little-endian register pattern.
fn register_bytes(value: u128, width: u32) -> Vec<u8> {
    value
        .to_le_bytes()
        .into_iter()
        .take(width.div_ceil(8) as usize)
        .collect()
}

/// Byte and bit reversal execute at every scalar width and preserve all register bits.
#[test]
fn scalar_bit_reversals_cover_every_width_and_carrier() -> Result<(), Box<dyn Error>> {
    for (ty, width) in SCALAR_WIDTHS {
        for bit in 0..width {
            let pattern = 1u128 << bit;
            for packed in [false, true] {
                let input = if packed {
                    vector(
                        8,
                        &register_bytes(pattern, width)
                            .into_iter()
                            .map(u128::from)
                            .collect::<Vec<_>>(),
                    )?
                } else {
                    ConstValue::of_lane(width, pattern)
                        .ok_or_else(|| io::Error::other("invalid width"))?
                };
                for (operation, expected_bit) in [
                    (IntegerOp::BRev, width - 1 - bit),
                    (IntegerOp::BSwap, (width / 8 - 1 - bit / 8) * 8 + bit % 8),
                ] {
                    let result = execute(&[(ty, input.clone())], ty, |dest, inputs| {
                        // The binary fixture builder's second slot is unused by these operations.
                        operation.operation(dest, &[operand(inputs, 0), operand(inputs, 0)])
                    })?;
                    assert_eq!(
                        result.constant().and_then(
                            |value| value.to_bytes(Endianness::Little, PointerSize::Bit64)
                        ),
                        Some(register_bytes(1u128 << expected_bit, width)),
                        "{operation:?}, width {width}, bit {bit}, packed {packed}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Signed and unsigned quotient/remainder preserve full scalar register patterns.
#[test]
fn scalar_division_and_remainder_cover_every_width() -> Result<(), Box<dyn Error>> {
    for (ty, width) in SCALAR_WIDTHS {
        let mask = u128::MAX >> (128 - width);
        for (operation, dividend, divisor, expected) in [
            (IntegerOp::Div(true), mask, 3, mask / 3),
            (IntegerOp::Rem(true), mask, 7, mask % 7),
            (IntegerOp::Div(false), mask - 7, 3, mask - 1),
            (IntegerOp::Rem(false), mask - 7, 3, mask - 1),
        ] {
            let result = execute(
                &[
                    (
                        ty,
                        vector(
                            8,
                            &register_bytes(dividend, width)
                                .into_iter()
                                .map(u128::from)
                                .collect::<Vec<_>>(),
                        )?,
                    ),
                    (
                        ty,
                        ConstValue::of_lane(width, divisor)
                            .ok_or_else(|| io::Error::other("invalid divisor"))?,
                    ),
                ],
                ty,
                |dest, inputs| operation.operation(dest, inputs),
            )?;
            assert_eq!(
                result
                    .constant()
                    .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit64)),
                Some(register_bytes(expected, width)),
                "{operation:?}, width {width}"
            );
        }
    }
    Ok(())
}

/// The scalar division contract refuses its overflowing quotient at every lane width.
#[test]
fn signed_scalar_lane_division_rejects_minimum_over_negative_one() -> Result<(), Box<dyn Error>> {
    for width in [1u32, 7, 8, 16, 24, 32, 64, 65, 127, 128] {
        let minimum = vector(width, &[1u128 << (width - 1)])?;
        let negative_one = vector(width, &[u128::MAX >> (128 - width)])?;
        assert!(
            minimum.div(&negative_one, PointerSize::Bit64).is_none(),
            "div width {width}"
        );
        assert!(
            minimum.rem(&negative_one, PointerSize::Bit64).is_none(),
            "rem width {width}"
        );
    }
    Ok(())
}

/// Scan, population and parity results use their own destination width.
#[test]
fn scalar_bit_counts_cover_input_and_result_widths() -> Result<(), Box<dyn Error>> {
    for (input_ty, input_width) in SCALAR_WIDTHS {
        let pattern = (1u128 << (input_width - 1)) | 2;
        for (result_ty, result_width) in SCALAR_WIDTHS {
            for (kind, expected) in [(0, 1), (1, input_width - 1), (2, 2), (3, 0)] {
                let result = execute(
                    &[(
                        input_ty,
                        ConstValue::of_lane(input_width, pattern)
                            .ok_or_else(|| io::Error::other("invalid input"))?,
                    )],
                    result_ty,
                    |dest, inputs| {
                        let src = operand(inputs, 0);
                        match kind {
                            0 => SsaOp::BitScanForward { dest, src },
                            1 => SsaOp::BitScanReverse { dest, src },
                            2 => SsaOp::Popcount { dest, src },
                            _ => SsaOp::Parity { dest, src },
                        }
                    },
                )?;
                assert_eq!(
                    result
                        .constant()
                        .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit64)),
                    Some(register_bytes(u128::from(expected), result_width)),
                    "kind {kind}, input {input_width}, result {result_width}"
                );
            }
        }
    }
    Ok(())
}

/// No set-bit index is known for a zero register in either scan direction.
#[test]
fn zero_scalar_bit_scans_do_not_invent_a_set_bit() -> Result<(), Box<dyn Error>> {
    for (ty, width) in SCALAR_WIDTHS {
        for reverse in [false, true] {
            let result = execute(
                &[(
                    ty,
                    ConstValue::of_lane(width, 0)
                        .ok_or_else(|| io::Error::other("invalid zero"))?,
                )],
                MockType::U32,
                |dest, inputs| {
                    let src = operand(inputs, 0);
                    if reverse {
                        SsaOp::BitScanReverse { dest, src }
                    } else {
                        SsaOp::BitScanForward { dest, src }
                    }
                },
            )?;
            assert!(
                result.constant().is_none(),
                "width {width}, reverse {reverse}"
            );
        }
    }
    Ok(())
}

/// Native carriers use the target width for permutations, counts and wide division.
#[test]
fn native_register_bits_obey_every_target_pointer_width() -> Result<(), Box<dyn Error>> {
    let src = SsaVarId::from_index(0);
    let right = SsaVarId::from_index(1);
    let dest = SsaVarId::from_index(2);
    for ptr_size in [
        PointerSize::Bit8,
        PointerSize::Bit16,
        PointerSize::Bit32,
        PointerSize::Bit64,
        PointerSize::Bit128,
    ] {
        let machine = Machine {
            ptr_size,
            ..MACHINE
        };
        let width = ptr_size.bits();
        let result_type = Destination {
            bits: Some(width),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: None,
        };
        let result = machine.evaluate(&SsaOp::BSwap { dest, src }, result_type, |_| {
            Some(ConstValue::<MockTarget>::NativeUInt(1))
        });
        assert_eq!(
            result.and_then(|value| value.to_bytes(Endianness::Little, ptr_size)),
            Some(register_bytes(1u128 << (width - 8), width))
        );
        let result = machine.evaluate(&SsaOp::BRev { dest, src }, result_type, |_| {
            Some(ConstValue::<MockTarget>::NativeUInt(1))
        });
        assert_eq!(
            result.and_then(|value| value.to_bytes(Endianness::Little, ptr_size)),
            Some(register_bytes(1u128 << (width - 1), width))
        );
        let result = machine.evaluate(&SsaOp::Popcount { dest, src }, Some(32), |_| {
            Some(ConstValue::<MockTarget>::NativeInt(-1))
        });
        assert_eq!(
            result.and_then(|value| value.as_u64()),
            Some(u64::from(width))
        );
        let result = machine.evaluate(
            &SsaOp::Div {
                dest,
                left: src,
                right,
                unsigned: true,
                flags: None,
            },
            result_type,
            |var| {
                Some(if var == src {
                    ConstValue::<MockTarget>::NativeInt(-1)
                } else {
                    ConstValue::NativeUInt(3)
                })
            },
        );
        let expected = (u128::MAX >> (128 - width)) / 3;
        assert_eq!(
            result.and_then(|value| value.to_bytes(Endianness::Little, ptr_size)),
            Some(register_bytes(expected, width)),
            "pointer width {width}"
        );
    }
    Ok(())
}

/// Unsigned lanes retain their high bit rather than being bounded by signed 128-bit storage.
#[test]
fn unsigned_division_of_wide_vector_lanes_preserves_the_full_domain() -> Result<(), Box<dyn Error>>
{
    let left = vector(128, &[u128::MAX, (1u128 << 127) + 7])?;
    let right = vector(128, &[3, 7])?;
    assert_eq!(
        left.div_un(&right, PointerSize::Bit64),
        Some(vector(128, &[u128::MAX / 3, ((1u128 << 127) + 7) / 7])?)
    );
    assert_eq!(
        left.rem_un(&right, PointerSize::Bit64),
        Some(vector(128, &[u128::MAX % 3, ((1u128 << 127) + 7) % 7])?)
    );
    let zero = vector(128, &[3, 0])?;
    assert!(left.div_un(&zero, PointerSize::Bit64).is_none());
    assert!(left.rem_un(&zero, PointerSize::Bit64).is_none());
    Ok(())
}

/// Packed count inputs keep their source width; one-bit result conditions remain usable.
#[test]
fn packed_bit_counts_and_boolean_parity_retain_their_domains() -> Result<(), Box<dyn Error>> {
    for (ty, width) in SCALAR_WIDTHS {
        let ones = u128::MAX >> (128 - width);
        let lanes = register_bytes(ones, width)
            .into_iter()
            .map(u128::from)
            .collect::<Vec<_>>();
        let input = vector(8, &lanes)?;
        let count = execute(&[(ty, input.clone())], MockType::U8, |dest, inputs| {
            SsaOp::Popcount {
                dest,
                src: operand(inputs, 0),
            }
        })?;
        assert_eq!(
            count.constant().and_then(ConstValue::as_u64),
            Some(u64::from(width))
        );
        let parity = execute(&[(ty, input)], MockType::I1, |dest, inputs| SsaOp::Parity {
            dest,
            src: operand(inputs, 0),
        })?;
        assert_eq!(parity.constant().and_then(ConstValue::as_bool), Some(false));
    }
    let parity = execute(
        &[(MockType::U128, vector(128, &[1u128 << 127])?)],
        MockType::I1,
        |dest, inputs| SsaOp::Parity {
            dest,
            src: operand(inputs, 0),
        },
    )?;
    assert_eq!(parity.constant().and_then(ConstValue::as_bool), Some(true));
    Ok(())
}

/// One scalar register can be carried in native, primitive or differently packed storage.
#[test]
fn scalar_division_accepts_equal_width_mixed_carriers() -> Result<(), Box<dyn Error>> {
    let lhs = vector(32, &[0xffff_fff8])?;
    assert_eq!(
        lhs.div(&ConstValue::I32(3), PointerSize::Bit64),
        Some(vector(32, &[0xffff_fffe])?)
    );
    let rhs = vector(8, &[3, 0, 0, 0])?;
    assert_eq!(
        ConstValue::U32(8).div(&rhs, PointerSize::Bit64),
        Some(ConstValue::U32(2))
    );
    let lhs = vector(128, &[u128::MAX])?;
    assert_eq!(
        lhs.div_un(&ConstValue::NativeUInt(3), PointerSize::Bit128),
        Some(vector(128, &[u128::MAX / 3])?)
    );
    Ok(())
}

/// An independently stated native-register case and its full-width bit pattern.
struct NativeCase {
    /// Operation whose data registers have the target's native width.
    operation: IntegerOp,
    /// Left or unary input, before any SSA literal normalization.
    left: ConstValue<MockTarget>,
    /// Right input or count, before any SSA literal normalization.
    right: ConstValue<MockTarget>,
    /// Exact register pattern at the operation's result.
    expected: u128,
}

/// Covers every width-preserving scalar family with carries beyond 64 bits.
fn native_cases() -> Vec<NativeCase> {
    let unsigned = ConstValue::NativeUInt;
    let signed = ConstValue::NativeInt;
    let low = u128::from(u64::MAX);
    let high = 1u128 << 127;
    [
        (
            IntegerOp::Add,
            unsigned(u64::MAX),
            ConstValue::U64(1),
            1u128 << 64,
        ),
        (
            IntegerOp::Add,
            ConstValue::U64(1),
            unsigned(u64::MAX),
            1u128 << 64,
        ),
        (IntegerOp::Add, unsigned(0), ConstValue::I8(-1), u128::MAX),
        (
            IntegerOp::Sub,
            ConstValue::U8(1),
            unsigned(u64::MAX),
            (u128::MAX ^ low) | 2,
        ),
        (IntegerOp::Xor, ConstValue::U32(0), signed(-1), u128::MAX),
        (IntegerOp::Neg, signed(-1), unsigned(0), 1),
        (IntegerOp::Neg, signed(i64::MIN), unsigned(0), 1u128 << 63),
        (
            IntegerOp::Neg,
            unsigned(u64::MAX),
            unsigned(0),
            (u128::MAX ^ low) | 1,
        ),
        (IntegerOp::Not, signed(-1), unsigned(0), 0),
        (
            IntegerOp::Not,
            unsigned(u64::MAX),
            unsigned(0),
            u128::MAX ^ low,
        ),
        (IntegerOp::Shr(true), signed(-1), unsigned(127), 1),
        (IntegerOp::Shr(false), signed(-1), unsigned(127), u128::MAX),
        (IntegerOp::Shr(true), unsigned(u64::MAX), unsigned(64), 0),
        (IntegerOp::Add, unsigned(u64::MAX), unsigned(1), 1u128 << 64),
        (IntegerOp::Sub, unsigned(0), unsigned(1), u128::MAX),
        (
            IntegerOp::Mul,
            unsigned(u64::MAX),
            unsigned(2),
            (1u128 << 65) - 2,
        ),
        (IntegerOp::And, unsigned(u64::MAX), signed(-1), low),
        (IntegerOp::Or, unsigned(0), signed(-1), u128::MAX),
        (IntegerOp::Xor, unsigned(0), signed(-1), u128::MAX),
        (IntegerOp::Shl, unsigned(1), unsigned(64), 1u128 << 64),
        (IntegerOp::Shl, unsigned(1), unsigned(127), high),
        (IntegerOp::Shr(false), signed(-1), unsigned(1), u128::MAX),
        (
            IntegerOp::Shr(true),
            signed(-1),
            unsigned(1),
            u128::MAX >> 1,
        ),
        (
            IntegerOp::Shr(false),
            unsigned(u64::MAX),
            unsigned(1),
            low >> 1,
        ),
        (IntegerOp::Rol, unsigned(1), unsigned(127), high),
        (IntegerOp::Ror, unsigned(1), unsigned(1), high),
        (IntegerOp::Rol, unsigned(u64::MAX), unsigned(64), low << 64),
        (IntegerOp::Ror, unsigned(u64::MAX), unsigned(64), low << 64),
        (IntegerOp::Neg, unsigned(1), unsigned(0), u128::MAX),
        (IntegerOp::Not, unsigned(0), unsigned(0), u128::MAX),
        (IntegerOp::BSwap, unsigned(1), unsigned(0), 1u128 << 120),
        (IntegerOp::BRev, unsigned(1), unsigned(0), high),
        (
            IntegerOp::Div(false),
            signed(-8),
            unsigned(3),
            u128::MAX - 1,
        ),
        (
            IntegerOp::Rem(false),
            signed(-8),
            unsigned(3),
            u128::MAX - 1,
        ),
        (IntegerOp::Div(true), signed(-1), unsigned(3), u128::MAX / 3),
        (IntegerOp::Rem(true), signed(-1), unsigned(7), u128::MAX % 7),
        (
            IntegerOp::AddOvf,
            unsigned(u64::MAX),
            unsigned(1),
            1u128 << 64,
        ),
        (IntegerOp::SubOvf, unsigned(0), unsigned(1), u128::MAX),
        (
            IntegerOp::MulOvf,
            unsigned(u64::MAX),
            unsigned(2),
            (1u128 << 65) - 2,
        ),
    ]
    .into_iter()
    .map(|(operation, left, right, expected)| NativeCase {
        operation,
        left,
        right,
        expected,
    })
    .collect()
}

/// Native 128-bit inputs supplied directly cannot wrap inside their 64-bit payloads.
#[test]
fn typed_native_128_operations_preserve_the_entire_register() {
    let machine = Machine {
        ptr_size: PointerSize::Bit128,
        ..MACHINE
    };
    let vars = [SsaVarId::from_index(0), SsaVarId::from_index(1)];
    let dest = SsaVarId::from_index(2);
    for case in native_cases() {
        let values = [case.left, case.right];
        let op = case.operation.operation(dest, &vars);
        let result = machine.evaluate(
            &op,
            Destination {
                bits: Some(128),
                lanes: None,
                lane_format: None,
                integer: true,
                integer_type: None,
            },
            |var| values.get(var.index()).cloned(),
        );
        assert_eq!(
            result.and_then(|value| value.to_bytes(Endianness::Little, machine.ptr_size)),
            Some(register_bytes(case.expected, 128)),
            "{:?}",
            case.operation
        );
    }
}

/// A native operand still declares its width when the SSA result type is unknown.
#[test]
fn inferred_native_128_operations_preserve_the_entire_register() {
    let machine = Machine {
        ptr_size: PointerSize::Bit128,
        ..MACHINE
    };
    let vars = [SsaVarId::from_index(0), SsaVarId::from_index(1)];
    let dest = SsaVarId::from_index(2);
    for case in native_cases() {
        let values = [case.left, case.right];
        let op = case.operation.operation(dest, &vars);
        let result = machine.evaluate(&op, None, |var| values.get(var.index()).cloned());
        assert_eq!(
            result.and_then(|value| value.to_bytes(Endianness::Little, machine.ptr_size)),
            Some(register_bytes(case.expected, 128)),
            "{:?}",
            case.operation
        );
    }
}

/// Whole SSA interpretation and constant evaluation agree at the native target width.
#[test]
fn native_128_arithmetic_executes_with_and_without_result_type() -> Result<(), Box<dyn Error>> {
    let machine = Machine {
        ptr_size: PointerSize::Bit128,
        ..MACHINE
    };
    for ty in [MockType::U128, MockType::Unknown] {
        for case in native_cases() {
            let result = execute_machine(
                &[(ty, case.left), (ty, case.right)],
                ty,
                |dest, inputs| case.operation.operation(dest, inputs),
                machine,
            )?;
            assert_eq!(
                result
                    .constant()
                    .and_then(|value| value.to_bytes(Endianness::Little, machine.ptr_size)),
                Some(register_bytes(case.expected, 128)),
                "{:?}, {ty:?}",
                case.operation
            );
        }
    }
    Ok(())
}

/// Native lanes encode target-width patterns, including sign extension beyond the payload.
#[test]
fn native_lanes_retain_the_full_target_pattern() {
    for ptr_size in [
        PointerSize::Bit8,
        PointerSize::Bit16,
        PointerSize::Bit32,
        PointerSize::Bit64,
        PointerSize::Bit128,
    ] {
        let mask = u128::MAX >> (128 - ptr_size.bits());
        for (value, expected) in [
            (ConstValue::<MockTarget>::NativeInt(-1), mask),
            (
                ConstValue::NativeInt(i64::MIN),
                (u128::MAX - (1u128 << 63) + 1) & mask,
            ),
            (
                ConstValue::NativeUInt(u64::MAX),
                u128::from(u64::MAX) & mask,
            ),
        ] {
            let lanes = value.as_lanes(ptr_size);
            assert_eq!(lanes.as_ref().map(VectorConst::bits), Some(ptr_size.bits()));
            assert_eq!(lanes.as_ref().map(|vector| vector.lane(0)), Some(expected));
        }
    }
}

/// Public value kernels share full-width semantics without operation-level coercion.
#[test]
fn native_128_value_kernels_preserve_the_entire_register() {
    let ptr = PointerSize::Bit128;
    for case in native_cases() {
        let left = &case.left;
        let right = &case.right;
        let result = match case.operation {
            IntegerOp::Add => left.add(right, ptr),
            IntegerOp::Sub => left.sub(right, ptr),
            IntegerOp::Mul => left.mul(right, ptr),
            IntegerOp::Div(false) => left.div(right, ptr),
            IntegerOp::Div(true) => left.div_un(right, ptr),
            IntegerOp::Rem(false) => left.rem(right, ptr),
            IntegerOp::Rem(true) => left.rem_un(right, ptr),
            IntegerOp::And => left.bitwise_and(right, ptr),
            IntegerOp::Or => left.bitwise_or(right, ptr),
            IntegerOp::Xor => left.bitwise_xor(right, ptr),
            IntegerOp::Shl => left.shl(right, ptr),
            IntegerOp::Shr(unsigned) => left.shr(right, unsigned, ptr),
            IntegerOp::Rol => left.rotate(right, false, ptr),
            IntegerOp::Ror => left.rotate(right, true, ptr),
            IntegerOp::Neg => left.negate(ptr),
            IntegerOp::Not => left.bitwise_not(ptr),
            IntegerOp::AddOvf => left.add_checked(right, false, ptr),
            IntegerOp::SubOvf => left.sub_checked(right, false, ptr),
            IntegerOp::MulOvf => left.mul_checked(right, false, ptr),
            IntegerOp::BSwap | IntegerOp::BRev => continue,
        };
        assert_eq!(
            result.and_then(|value| value.to_bytes(Endianness::Little, ptr)),
            Some(register_bytes(case.expected, 128)),
            "{:?}",
            case.operation
        );
    }
}

/// Native shifts read the target register before shifting, independently of payload storage.
#[test]
fn native_shifts_read_the_entire_target_register() -> Result<(), Box<dyn Error>> {
    for ptr in [
        PointerSize::Bit8,
        PointerSize::Bit16,
        PointerSize::Bit32,
        PointerSize::Bit64,
        PointerSize::Bit128,
    ] {
        let width = ptr.bits();
        let mask = u128::MAX >> (128 - width);
        let high = 1u128 << (width - 1);
        let raw_signed = if width <= 64 { high as i64 } else { i64::MIN };
        let signed_bits = (i128::from(raw_signed) as u128) & mask;
        for (value, pattern) in [
            (ConstValue::<MockTarget>::NativeInt(-1), mask),
            (
                ConstValue::NativeUInt(u64::MAX),
                u128::from(u64::MAX) & mask,
            ),
            (ConstValue::NativeInt(raw_signed), signed_bits),
        ] {
            for distance in [0, 1, width - 1] {
                for unsigned in [true, false] {
                    let fill = if !unsigned && pattern & high != 0 {
                        mask ^ (mask >> distance)
                    } else {
                        0
                    };
                    let expected = (pattern >> distance) | fill;
                    let count = ConstValue::U32(distance);
                    assert_eq!(
                        value
                            .shr(&count, unsigned, ptr)
                            .and_then(|value| value.to_bytes(Endianness::Little, ptr)),
                        Some(register_bytes(expected, width)),
                        "direct {ptr:?} {value:?} {distance} unsigned={unsigned}"
                    );
                    let result = execute_machine(
                        &[(MockType::Unknown, value.clone()), (MockType::U32, count)],
                        MockType::Unknown,
                        |dest, inputs| IntegerOp::Shr(unsigned).operation(dest, inputs),
                        Machine {
                            ptr_size: ptr,
                            ..MACHINE
                        },
                    )?;
                    assert_eq!(
                        result
                            .constant()
                            .and_then(|value| value.to_bytes(Endianness::Little, ptr)),
                        Some(register_bytes(expected, width)),
                        "SSA {ptr:?} {value:?} {distance} unsigned={unsigned}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Native promotion masks at the native width before extending to a wider primitive carrier.
#[test]
fn native_mixed_promotions_use_target_width_before_extension() {
    for ptr in [PointerSize::Bit8, PointerSize::Bit16, PointerSize::Bit32] {
        let width = ptr.bits();
        let high = 1u64 << (width - 1);
        let mask = (1u64 << width) - 1;
        for (left, promoted) in [
            (
                ConstValue::<MockTarget>::NativeInt(high as i64),
                u64::MAX << (width - 1),
            ),
            (ConstValue::NativeUInt(u64::MAX), mask),
        ] {
            let right = ConstValue::U64(3);
            for (op, expected) in [
                (IntegerOp::Add, promoted.wrapping_add(3)),
                (IntegerOp::Sub, promoted.wrapping_sub(3)),
                (IntegerOp::Mul, promoted.wrapping_mul(3)),
                (IntegerOp::And, promoted & 3),
                (IntegerOp::Or, promoted | 3),
                (IntegerOp::Xor, promoted ^ 3),
            ] {
                let actual = match op {
                    IntegerOp::Add => left.add(&right, ptr),
                    IntegerOp::Sub => left.sub(&right, ptr),
                    IntegerOp::Mul => left.mul(&right, ptr),
                    IntegerOp::And => left.bitwise_and(&right, ptr),
                    IntegerOp::Or => left.bitwise_or(&right, ptr),
                    IntegerOp::Xor => left.bitwise_xor(&right, ptr),
                    _ => process::abort(),
                };
                assert_eq!(
                    actual.and_then(|value| value.to_bytes(Endianness::Little, ptr)),
                    Some(expected.to_le_bytes().to_vec()),
                    "{ptr:?} {left:?} {op:?}"
                );
            }
        }
    }
}

/// Shift counts retain native width and accept a complete single-lane scalar carrier.
#[test]
fn native_and_packed_shift_counts_follow_their_own_widths() -> Result<(), Box<dyn Error>> {
    let value = ConstValue::<MockTarget>::U64(0x80);
    for ptr in [PointerSize::Bit8, PointerSize::Bit16, PointerSize::Bit32] {
        let modulus = 1u64 << ptr.bits();
        for (count, distance) in [
            (ConstValue::NativeUInt(modulus | 1), 1u32),
            (ConstValue::NativeInt(modulus as i64), 0),
            (ConstValue::NativeInt((modulus | 1) as i64), 1),
            (vector(128, &[1])?, 1),
        ] {
            assert_eq!(
                value.shl(&count, ptr),
                Some(ConstValue::U64(0x80u64 << distance)),
                "{ptr:?} {count:?}"
            );
            assert_eq!(
                value.shr(&count, true, ptr),
                Some(ConstValue::U64(0x80u64 >> distance)),
                "{ptr:?} {count:?}"
            );
            assert_eq!(
                value.rotate(&count, false, ptr),
                Some(ConstValue::U64(0x80u64.rotate_left(distance))),
                "{ptr:?} {count:?}"
            );
        }
        assert!(
            value
                .shl(&ConstValue::NativeInt((modulus - 1) as i64), ptr)
                .is_none()
        );
        assert!(
            value
                .shr(&ConstValue::NativeInt((modulus - 1) as i64), true, ptr)
                .is_none()
        );
    }
    let high_count = vector(128, &[(1u128 << 100) | 1])?;
    assert!(value.shl(&high_count, PointerSize::Bit64).is_none());
    assert!(value.shr(&high_count, true, PointerSize::Bit64).is_none());
    assert_eq!(
        value.rotate(&high_count, false, PointerSize::Bit64),
        Some(ConstValue::U64(0x100))
    );
    Ok(())
}

/// Every byte pattern follows primitive shift semantics after native input/count masking.
#[test]
fn native_byte_shifts_match_exhaustive_primitive_oracles() {
    for byte in 0..=u8::MAX {
        for value in [
            ConstValue::<MockTarget>::NativeInt(i64::from(byte) | 0x1_0000),
            ConstValue::NativeUInt(u64::from(byte) | 0x1_0000),
        ] {
            for distance in 0..=8u32 {
                let count = ConstValue::NativeUInt(0x1_0000 | u64::from(distance));
                for unsigned in [false, true] {
                    let expected = if distance == 8 {
                        None
                    } else {
                        Some(if unsigned {
                            byte >> distance
                        } else {
                            ((byte as i8) >> distance) as u8
                        })
                    };
                    assert_eq!(value.shr(&count, unsigned, PointerSize::Bit8)
                        .and_then(|value| value.to_bytes(Endianness::Little, PointerSize::Bit8)),
                        expected.map(|byte| vec![byte]), "{value:?} >> {distance} unsigned={unsigned}");
                }
            }
        }
    }
}

/// Scalar comparisons use data-register widths independently of their boolean result.
#[derive(Debug, Clone, Copy)]
enum Comparison {
    /// Equality of the two values.
    Equal,
    /// Less-than with an explicit unsigned interpretation.
    Less(bool),
    /// Greater-than with an explicit unsigned interpretation.
    Greater(bool),
}

impl Comparison {
    /// Builds one comparison over the fixture's operands.
    fn operation(self, dest: SsaVarId, inputs: &[SsaVarId]) -> SsaOp<MockTarget> {
        let (left, right) = (operand(inputs, 0), operand(inputs, 1));
        match self {
            Self::Equal => SsaOp::Ceq {
                dest,
                left,
                right,
                signaling: false,
            },
            Self::Less(unsigned) => SsaOp::Clt {
                dest,
                left,
                right,
                unsigned,
                signaling: false,
            },
            Self::Greater(unsigned) => SsaOp::Cgt {
                dest,
                left,
                right,
                unsigned,
                signaling: false,
            },
        }
    }
}

/// Equal bit patterns in the same verified integer type cannot differ by carrier signedness.
#[test]
fn verified_integer_equality_ignores_literal_carrier_signedness() -> Result<(), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let mut returned = None;
    builder.in_block(0, |block| {
        let left = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(-1))?;
        let right = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::U32(u32::MAX))?;
        let result = block.emit_def(SsaDefSpec::tmp(MockType::I1), |dest| SsaOp::Ceq {
            dest,
            left,
            right,
            signaling: false,
        })?;
        returned = Some(result);
        block.ret(Some(result))
    })?;
    let ssa = builder.finish()?;
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors.is_empty(),
        "comparison fixture must be legal SSA: {errors:?}"
    );
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete());
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_bool),
        Some(true)
    );
    let mut evaluator = ConstEvaluator::new(&ssa, MACHINE);
    assert_eq!(
        returned
            .and_then(|var| evaluator.evaluate_var(var))
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    Ok(())
}

/// Integer-declared packed registers are scalar comparison inputs, not a boolean-width value.
#[test]
fn scalar_comparisons_read_declared_packed_registers() -> Result<(), Box<dyn Error>> {
    for (left, right, signed_less, unsigned_less) in [
        (u32::MAX, 0u32, true, false),
        (0x8000_0000, 0x7fff_ffff, true, false),
        (42, 42, false, false),
    ] {
        let packed = vector(8, &left.to_le_bytes().map(u128::from))?;
        for (comparison, expected) in [
            (Comparison::Equal, left == right),
            (Comparison::Less(false), signed_less),
            (Comparison::Less(true), unsigned_less),
            (Comparison::Greater(false), (left as i32) > (right as i32)),
            (Comparison::Greater(true), left > right),
        ] {
            let result = execute(
                &[
                    (MockType::U32, packed.clone()),
                    (MockType::U32, ConstValue::U32(right)),
                ],
                MockType::I1,
                |dest, inputs| comparison.operation(dest, inputs),
            )?;
            assert_eq!(
                result.constant().and_then(ConstValue::as_bool),
                Some(expected),
                "{comparison:?} {left:x}/{right:x}"
            );
        }
    }
    Ok(())
}

/// A 128-bit native unsigned payload remains positive in signed 128-bit ordering.
#[test]
fn native_128_ordering_retains_bits_above_signed_64_range() -> Result<(), Box<dyn Error>> {
    let machine = Machine {
        ptr_size: PointerSize::Bit128,
        ..MACHINE
    };
    let values = [
        ConstValue::<MockTarget>::NativeUInt(u64::MAX),
        ConstValue::NativeUInt(0),
    ];
    assert_eq!(
        values[0]
            .clt(&values[1], machine.ptr_size)
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        values[0]
            .cgt(&values[1], machine.ptr_size)
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    for (comparison, expected) in [
        (Comparison::Less(false), false),
        (Comparison::Greater(false), true),
    ] {
        for ty in [MockType::U128, MockType::Unknown] {
            let result = execute_machine(
                &[(ty, values[0].clone()), (ty, values[1].clone())],
                MockType::I1,
                |dest, inputs| comparison.operation(dest, inputs),
                machine,
            )?;
            assert_eq!(
                result.constant().and_then(ConstValue::as_bool),
                Some(expected),
                "{comparison:?} {ty:?}"
            );
        }
    }
    Ok(())
}

/// Equality reads native integer registers at their target width, including raw payloads.
#[test]
fn native_integer_equality_reads_target_width() -> Result<(), Box<dyn Error>> {
    for ptr in [
        PointerSize::Bit8,
        PointerSize::Bit16,
        PointerSize::Bit32,
        PointerSize::Bit64,
        PointerSize::Bit128,
    ] {
        let machine = Machine {
            ptr_size: ptr,
            ..MACHINE
        };
        let input = [
            (MockType::Unknown, ConstValue::NativeInt(-1)),
            (MockType::Unknown, ConstValue::NativeUInt(u64::MAX)),
        ];
        let result = execute_machine(
            &input,
            MockType::I1,
            |dest, inputs| Comparison::Equal.operation(dest, inputs),
            machine,
        )?;
        assert_eq!(
            result.constant().and_then(ConstValue::as_bool),
            Some(ptr.bits() <= 64),
            "{ptr:?}"
        );
        if ptr.bits() < 64 {
            let raw = ConstValue::NativeUInt(1u64 << ptr.bits());
            let result = execute_machine(
                &[
                    (MockType::Unknown, raw),
                    (MockType::Unknown, ConstValue::NativeUInt(0)),
                ],
                MockType::I1,
                |dest, inputs| Comparison::Equal.operation(dest, inputs),
                machine,
            )?;
            assert_eq!(
                result.constant().and_then(ConstValue::as_bool),
                Some(true),
                "wrapped {ptr:?}"
            );
        }
    }
    Ok(())
}

/// Single-lane scalar integers compare at their own widths before extension.
#[test]
fn scalar_lane_ordering_compares_different_carrier_widths() -> Result<(), Box<dyn Error>> {
    for right in [vector(128, &[0])?, ConstValue::U64(0)] {
        let left = vector(8, &[0x80])?;
        for (comparison, expected) in [
            (Comparison::Less(false), true),
            (Comparison::Greater(false), false),
            (Comparison::Less(true), false),
            (Comparison::Greater(true), true),
        ] {
            let result = execute(
                &[
                    (MockType::Unknown, left.clone()),
                    (MockType::Unknown, right.clone()),
                ],
                MockType::I1,
                |dest, inputs| comparison.operation(dest, inputs),
            )?;
            assert_eq!(
                result.constant().and_then(ConstValue::as_bool),
                Some(expected),
                "{comparison:?} {right:?}"
            );
        }
    }
    Ok(())
}

/// A scalar packet and a primitive can name the same positive integer exactly.
#[test]
fn scalar_packet_equality_accepts_positive_mixed_width_values() -> Result<(), Box<dyn Error>> {
    for left in [vector(8, &[42])?, vector(128, &[42])?] {
        let result = execute(
            &[
                (MockType::Unknown, left),
                (MockType::Unknown, ConstValue::U64(42)),
            ],
            MockType::I1,
            |dest, inputs| Comparison::Equal.operation(dest, inputs),
        )?;
        assert_eq!(result.constant().and_then(ConstValue::as_bool), Some(true));
    }
    Ok(())
}

/// Constraint and symbolic readers agree on target width and register bit equality.
#[test]
fn equality_readers_share_target_and_carrier_semantics() -> Result<(), Box<dyn Error>> {
    for (left, right, ptr) in [
        (
            ConstValue::NativeUInt(256),
            ConstValue::NativeUInt(0),
            PointerSize::Bit8,
        ),
        (
            ConstValue::I32(-1),
            ConstValue::U32(u32::MAX),
            PointerSize::Bit64,
        ),
    ] {
        let equal = Constraint::<MockTarget>::Equal(left.clone());
        let unequal = Constraint::NotEqual(left.clone());
        assert!(equal.is_satisfied_by(&right, ptr));
        assert!(!unequal.is_satisfied_by(&right, ptr));
        assert!(!equal.conflicts_with(&Constraint::Equal(right.clone()), ptr));
        assert!(equal.conflicts_with(&Constraint::NotEqual(right.clone()), ptr));
        for (op, expected) in [(NumericOp::Eq, true), (NumericOp::Ne, false)] {
            assert_eq!(
                op.evaluate_binary(&left, &right, ptr)
                    .and_then(|v| v.as_bool()),
                Some(expected)
            );
        }
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let mut variables = None;
        builder.in_block(0, |block| {
            let lhs = block.const_value(SsaDefSpec::tmp(MockType::Unknown), left.clone())?;
            let rhs = block.const_value(SsaDefSpec::tmp(MockType::Unknown), right.clone())?;
            let condition = block.ceq(SsaDefSpec::tmp(MockType::I1), lhs, rhs)?;
            variables = Some((lhs, rhs, condition));
            block.ret(Some(condition))
        })?;
        let ssa = builder.finish()?;
        let (lhs, rhs, condition) =
            variables.ok_or_else(|| io::Error::other("missing comparison"))?;
        for (constraint, expected) in [(equal, true), (unequal, false)] {
            let mut evaluator = SsaEvaluator::new(
                &ssa,
                Machine {
                    ptr_size: ptr,
                    ..MACHINE
                },
            );
            evaluator.set_concrete(rhs, right.clone());
            evaluator.add_constraint(lhs, constraint);
            assert_eq!(
                evaluator.evaluate_condition_with_constraints(condition),
                Some(expected)
            );
        }
    }
    Ok(())
}

/// Branch comparisons use input register layouts and explicit target widths.
#[test]
fn branch_comparisons_read_declared_inputs() -> Result<(), Box<dyn Error>> {
    for (ty, left, right, ptr, cmp) in [
        (
            MockType::Unknown,
            ConstValue::NativeInt(-1),
            ConstValue::NativeUInt(u64::MAX),
            PointerSize::Bit8,
            CmpKind::Eq,
        ),
        (
            MockType::U32,
            vector(8, &[255, 255, 255, 255])?,
            ConstValue::U32(0),
            PointerSize::Bit64,
            CmpKind::Lt,
        ),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let lhs = block.const_value(SsaDefSpec::tmp(ty), left)?;
            let rhs = block.const_value(SsaDefSpec::tmp(ty), right)?;
            block.branch_cmp(lhs, rhs, cmp, false, 1, 2)
        })?;
        for (target, value) in [(1, 42), (2, 7)] {
            builder.in_block(target, |block| {
                let result =
                    block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(value))?;
                block.ret(Some(result))
            })?;
        }
        let ssa = builder.finish()?;
        let context = AnalysisContext::new(Machine {
            ptr_size: ptr,
            ..MACHINE
        });
        let mut symbolic = SsaEvaluator::new(
            &ssa,
            Machine {
                ptr_size: ptr,
                ..MACHINE
            },
        );
        symbolic.evaluate_block(0);
        assert_eq!(symbolic.next_block(0), ControlFlow::Continue(1));
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::U32(42))));
    }
    Ok(())
}

/// Every byte pair agrees with independent signed and unsigned primitive comparisons.
#[test]
fn scalar_byte_comparisons_match_exhaustive_primitive_oracles() {
    for lhs in 0u16..=255 {
        for rhs in 0u16..=255 {
            let expected = [
                lhs == rhs,
                (lhs as u8 as i8) < (rhs as u8 as i8),
                (lhs as u8 as i8) > (rhs as u8 as i8),
                lhs < rhs,
                lhs > rhs,
            ];
            for (left, right) in [
                (
                    ConstValue::<MockTarget>::I8(lhs as i8),
                    ConstValue::U8(rhs as u8),
                ),
                (
                    ConstValue::NativeInt(i64::from(lhs) | 0x10000),
                    ConstValue::NativeUInt(u64::from(rhs) | 0x10000),
                ),
            ] {
                let actual = [
                    left.ceq_at(&right, PointerSize::Bit8),
                    left.clt(&right, PointerSize::Bit8),
                    left.cgt(&right, PointerSize::Bit8),
                    left.clt_un(&right, PointerSize::Bit8),
                    left.cgt_un(&right, PointerSize::Bit8),
                ];
                for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                    assert_eq!(
                        actual.and_then(|v| v.as_bool()),
                        Some(expected),
                        "{left:?} {right:?}, comparison {index}"
                    );
                }
            }
        }
    }
}

/// Equality constraints on typed registers read packed bounds and scalar carriers identically.
#[test]
fn symbolic_equality_constraints_read_attested_input_layouts() -> Result<(), Box<dyn Error>> {
    let packed = vector(8, &[255, 255, 255, 255])?;
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let mut variables = None;
    builder.in_block(0, |block| {
        let left = block.const_value(SsaDefSpec::tmp(MockType::U32), packed.clone())?;
        let right = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(u32::MAX))?;
        let condition = block.ceq(SsaDefSpec::tmp(MockType::I1), left, right)?;
        variables = Some((left, right, condition));
        block.ret(Some(condition))
    })?;
    let ssa = builder.finish()?;
    let (left, right, condition) =
        variables.ok_or_else(|| io::Error::other("missing constraint fixture"))?;
    for (constraint, expected) in [
        (Constraint::Equal(packed.clone()), true),
        (Constraint::NotEqual(packed), false),
    ] {
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.set_concrete(right, ConstValue::U32(u32::MAX));
        evaluator.add_constraint(left, constraint);
        assert_eq!(
            evaluator.evaluate_condition_with_constraints(condition),
            Some(expected)
        );
    }
    Ok(())
}

/// Declared integer comparisons read logical and packed carriers as the same register.
#[test]
fn typed_boolean_comparisons_read_attested_register_patterns() -> Result<(), Box<dyn Error>> {
    for (ty, width) in [
        (MockType::I1, 1),
        (MockType::U8, 8),
        (MockType::U16, 16),
        (MockType::U32, 32),
        (MockType::U64, 64),
        (MockType::U128, 128),
    ] {
        for lhs in [false, true] {
            for rhs in [false, true] {
                for right in [
                    ConstValue::from_bool(rhs),
                    ConstValue::U8(u8::from(rhs)),
                    vector(width, &[u128::from(rhs)])?,
                ] {
                    let left = ConstValue::from_bool(lhs);
                    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
                    builder.in_block(0, |block| {
                        let a = block.const_value(SsaDefSpec::tmp(ty), left.clone())?;
                        let b = block.const_value(SsaDefSpec::tmp(ty), right.clone())?;
                        let result = block.ceq(SsaDefSpec::tmp(MockType::I1), a, b)?;
                        block.ret(Some(result))
                    })?;
                    let ssa = builder.finish()?;
                    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
                    assert!(
                        errors.is_empty(),
                        "legal typed comparison required: {errors:?}"
                    );
                    let signed = |bit| {
                        if width == 1 && bit {
                            -1i8
                        } else {
                            i8::from(bit)
                        }
                    };
                    for (comparison, expected) in [
                        (Comparison::Equal, lhs == rhs),
                        (Comparison::Less(false), signed(lhs) < signed(rhs)),
                        (Comparison::Greater(false), signed(lhs) > signed(rhs)),
                        (Comparison::Less(true), u8::from(lhs) < u8::from(rhs)),
                        (Comparison::Greater(true), u8::from(lhs) > u8::from(rhs)),
                    ] {
                        let result = execute(
                            &[(ty, left.clone()), (ty, right.clone())],
                            MockType::I1,
                            |dest, inputs| comparison.operation(dest, inputs),
                        )?;
                        assert_eq!(
                            result.constant().and_then(ConstValue::as_bool),
                            Some(expected),
                            "{ty:?} {left:?} {right:?} {comparison:?}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        ConstValue::<MockTarget>::True.ceq_at(&ConstValue::U8(1), PointerSize::Bit64),
        None,
        "untyped classes remain distinct"
    );
    Ok(())
}

/// Floating declarations compare register bits by their declared floating format.
#[test]
fn declared_float_comparisons_read_raw_register_bits() -> Result<(), Box<dyn Error>> {
    for (ty, bits, left, right) in [
        (
            MockType::F64,
            64,
            ConstValue::U64(1u64 << 63),
            ConstValue::U64(0),
        ),
        (
            MockType::BF16,
            16,
            ConstValue::U16(1u16 << 15),
            ConstValue::U16(0),
        ),
        (
            MockType::F80,
            80,
            vector(80, &[1u128 << 79])?,
            vector(80, &[0])?,
        ),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let a = block.const_value(SsaDefSpec::tmp(ty), left.clone())?;
            let b = block.const_value(SsaDefSpec::tmp(ty), right.clone())?;
            let result = block.ceq(SsaDefSpec::tmp(MockType::I1), a, b)?;
            block.ret(Some(result))
        })?;
        let ssa = builder.finish()?;
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            errors.is_empty(),
            "legal floating register fixture required: {errors:?}"
        );
        for (comparison, expected) in [
            (Comparison::Equal, true),
            (Comparison::Less(false), false),
            (Comparison::Less(true), false),
            (Comparison::Greater(false), false),
            (Comparison::Greater(true), false),
        ] {
            let result = execute(
                &[(ty, left.clone()), (ty, right.clone())],
                MockType::I1,
                |dest, inputs| comparison.operation(dest, inputs),
            )?;
            assert_eq!(
                result.constant().and_then(ConstValue::as_bool),
                Some(expected),
                "floating signed zeros {bits} bits {comparison:?}"
            );
        }
    }
    Ok(())
}

/// Floating arithmetic reads declared registers instead of adding their encodings.
#[test]
fn declared_float_arithmetic_reads_raw_register_bits() -> Result<(), Box<dyn Error>> {
    let extended = |exponent: u128, significand: u128| (exponent << 64) | significand;
    let integer_bit = 1u128 << 63;
    for (ty, format, left_bits, right_bits, expected_bits) in [
        (
            MockType::F64,
            FloatFormat::Double,
            u128::from(0.5f64.to_bits()),
            u128::from(0.25f64.to_bits()),
            [0.75f64, 0.25, 0.125, 2.0, 0.0, -0.5].map(|value| u128::from(value.to_bits())),
        ),
        (
            MockType::BF16,
            FloatFormat::BFloat16,
            0x3f00,
            0x3e80,
            [0x3f40, 0x3e80, 0x3e00, 0x4000, 0, 0xbf00],
        ),
        (
            MockType::F80,
            FloatFormat::X87Extended,
            extended(16382, integer_bit),
            extended(16381, integer_bit),
            [
                extended(16382, integer_bit | (1u128 << 62)),
                extended(16381, integer_bit),
                extended(16380, integer_bit),
                extended(16384, integer_bit),
                0,
                (1u128 << 79) | extended(16382, integer_bit),
            ],
        ),
    ] {
        for (kind, expected_bits) in expected_bits.into_iter().enumerate() {
            let operation = |dest, left, right| match kind {
                0 => SsaOp::Add {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                1 => SsaOp::Sub {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                2 => SsaOp::Mul {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                3 => SsaOp::Div {
                    dest,
                    left,
                    right,
                    unsigned: false,
                    flags: None,
                },
                4 => SsaOp::Rem {
                    dest,
                    left,
                    right,
                    unsigned: false,
                    flags: None,
                },
                _ => SsaOp::Neg {
                    dest,
                    operand: left,
                    flags: None,
                },
            };
            let left_value = vector(format.bits(), &[left_bits])?;
            let right_value = vector(format.bits(), &[right_bits])?;
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            let mut returned = None;
            builder.in_block(0, |block| {
                let left = block.const_value(SsaDefSpec::tmp(ty), left_value.clone())?;
                let right = block.const_value(SsaDefSpec::tmp(ty), right_value.clone())?;
                let result =
                    block.emit_def(SsaDefSpec::tmp(ty), |dest| operation(dest, left, right))?;
                returned = Some(result);
                block.ret(Some(result))
            })?;
            let ssa = builder.finish()?;
            let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
            assert!(
                errors.is_empty(),
                "legal float arithmetic required: {errors:?}"
            );
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&ssa, &context).run();
            assert!(facts.is_complete());
            let expected = ConstValue::float(format, expected_bits)
                .ok_or_else(|| io::Error::other("invalid float oracle"))?;
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&expected),
                "float arithmetic {kind}"
            );
            let result = returned.ok_or_else(|| io::Error::other("missing arithmetic result"))?;
            assert_eq!(
                ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result),
                Some(expected.clone())
            );
            let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
            symbolic.evaluate_block(0);
            assert_eq!(symbolic.get_concrete(result), Some(&expected));
        }
    }
    Ok(())
}

/// Integer-declared literals expose the same float bits in both evaluators.
#[test]
fn integer_declared_float_literals_share_the_result_bit_view() -> Result<(), Box<dyn Error>> {
    for (ty, literal, expected) in [
        (
            MockType::U64,
            ConstValue::from(-0.0f64),
            ConstValue::U64(1u64 << 63),
        ),
        (
            MockType::U64,
            ConstValue::from(0.5f32),
            ConstValue::U64(0x3f00_0000),
        ),
        (
            MockType::U128,
            ConstValue::float(FloatFormat::X87Extended, 1u128 << 79)
                .ok_or_else(|| io::Error::other("invalid extended literal"))?,
            vector(128, &[1u128 << 79])?,
        ),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let mut returned = None;
        builder.in_block(0, |block| {
            let source = block.const_value(SsaDefSpec::tmp(ty), literal.clone())?;
            let result = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::Copy {
                dest,
                src: source,
            })?;
            returned = Some(result);
            block.ret(Some(result))
        })?;
        let ssa = builder.finish()?;
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            errors.is_empty(),
            "legal integer literal required: {errors:?}"
        );
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&expected),
            "{ty:?} {literal:?}"
        );
        let result = returned.ok_or_else(|| io::Error::other("missing literal result"))?;
        assert_eq!(
            ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result),
            Some(expected.clone())
        );
        let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
        symbolic.evaluate_block(0);
        assert_eq!(symbolic.get_concrete(result), Some(&expected));
    }
    Ok(())
}

/// Floating conversions consume the declared source format, including bfloat16.
#[test]
fn floating_conversions_use_declared_source_formats() -> Result<(), Box<dyn Error>> {
    for (to_integer, bits) in [(false, 0x3f00u16), (true, 0x3fc0)] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let mut returned = None;
        let ty = if to_integer {
            MockType::I32
        } else {
            MockType::F64
        };
        builder.in_block(0, |block| {
            let operand =
                block.const_value(SsaDefSpec::tmp(MockType::BF16), ConstValue::U16(bits))?;
            let result = block.emit_def(SsaDefSpec::tmp(ty), |dest| {
                if to_integer {
                    SsaOp::FloatToInt {
                        dest,
                        operand,
                        target: ty,
                        unsigned: false,
                        overflow_check: false,
                        source: None,
                    }
                } else {
                    SsaOp::FloatConv {
                        dest,
                        operand,
                        target: ty,
                        rounding: None,
                        source: None,
                        environment: None,
                    }
                }
            })?;
            returned = Some(result);
            block.ret(Some(result))
        })?;
        let ssa = builder.finish()?;
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            errors.is_empty(),
            "legal float conversion required: {errors:?}"
        );
        let expected = if to_integer {
            ConstValue::I32(1)
        } else {
            ConstValue::from(0.5f64)
        };
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert_eq!(facts.returned().and_then(Value::constant), Some(&expected));
        let result = returned.ok_or_else(|| io::Error::other("missing conversion"))?;
        assert_eq!(
            ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result),
            Some(expected.clone())
        );
        let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
        symbolic.evaluate_block(0);
        assert_eq!(symbolic.get_concrete(result), Some(&expected));
    }
    Ok(())
}

/// A width or explicit floating-format mismatch cannot become a numeric proof.
#[test]
fn floating_operand_views_refuse_mismatched_formats_and_widths() -> Result<(), Box<dyn Error>> {
    for left in [
        ConstValue::U8(0),
        ConstValue::float(FloatFormat::Half, 0)
            .ok_or_else(|| io::Error::other("invalid half fixture"))?,
        vector(32, &[0])?,
    ] {
        let result = execute(
            &[(MockType::BF16, left), (MockType::BF16, ConstValue::U16(0))],
            MockType::I1,
            |dest, inputs| Comparison::Equal.operation(dest, inputs),
        )?;
        assert!(
            result.constant().is_none(),
            "unsupported view must not establish equality: {result:?}"
        );
    }
    Ok(())
}

/// A restated conversion retains its stated float format after an integer load.
#[test]
fn explicit_float_conversion_formats_survive_integer_restatements() -> Result<(), Box<dyn Error>> {
    for (to_integer, bits) in [(false, 0x3f00u16), (true, 0x3fc0)] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let operand =
                block.const_value(SsaDefSpec::tmp(MockType::U16), ConstValue::U16(bits))?;
            let ty = if to_integer {
                MockType::I32
            } else {
                MockType::F64
            };
            let result = block.emit_def(SsaDefSpec::tmp(ty), |dest| {
                if to_integer {
                    SsaOp::FloatToInt {
                        dest,
                        operand,
                        target: ty,
                        unsigned: false,
                        overflow_check: false,
                        source: Some(FloatFormat::BFloat16),
                    }
                } else {
                    SsaOp::FloatConv {
                        dest,
                        operand,
                        target: ty,
                        rounding: None,
                        source: Some(FloatFormat::BFloat16),
                        environment: None,
                    }
                }
            })?;
            block.ret(Some(result))
        })?;
        let ssa = builder.finish()?;
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            errors.is_empty(),
            "legal restated conversion required: {errors:?}"
        );
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        let expected = if to_integer {
            ConstValue::I32(1)
        } else {
            ConstValue::from(0.5f64)
        };
        assert_eq!(facts.returned().and_then(Value::constant), Some(&expected));
    }
    Ok(())
}

/// Builds a verified floating expression whose parameter can be NaN or infinite.
fn unknown_float_fixture(
    kind: u8,
) -> Result<(SsaFunction<MockTarget>, SsaVarId, SsaVarId), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let mut variables = None;
    builder.in_block(0, |block| {
        let input = block.load_arg(SsaDefSpec::argument(0, MockType::F64), 0)?;
        let zero = block.const_value(SsaDefSpec::tmp(MockType::F64), ConstValue::from(0.0f64))?;
        let ty = if kind == 0 || kind == 4 {
            MockType::I1
        } else {
            MockType::F64
        };
        let result = block.emit_def(SsaDefSpec::tmp(ty), |dest| match kind {
            0 => SsaOp::Ceq {
                dest,
                left: input,
                right: input,
                signaling: false,
            },
            1 => SsaOp::Sub {
                dest,
                left: input,
                right: input,
                flags: None,
            },
            2 => SsaOp::Div {
                dest,
                left: input,
                right: input,
                unsigned: false,
                flags: None,
            },
            3 => SsaOp::Mul {
                dest,
                left: input,
                right: zero,
                flags: None,
            },
            _ => SsaOp::Cgt {
                dest,
                left: input,
                right: input,
                unsigned: true,
                signaling: false,
            },
        })?;
        variables = Some((input, result));
        block.ret(Some(result))
    })?;
    let ssa = builder.finish()?;
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors.is_empty(),
        "legal unknown-float fixture required: {errors:?}"
    );
    let (input, result) = variables.ok_or_else(|| io::Error::other("missing floating fixture"))?;
    Ok((ssa, input, result))
}

/// Unknown floating operands cannot acquire constants from pointer identities.
#[test]
fn unknown_float_execution_preserves_ieee_uncertainty() -> Result<(), Box<dyn Error>> {
    for kind in 0..5 {
        let (ssa, _, _) = unknown_float_fixture(kind)?;
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(
            facts.returned().and_then(Value::constant).is_none(),
            "unknown floating expression {kind}: {:?}",
            facts.returned()
        );
    }
    Ok(())
}

/// Integer symbolic identities cannot decide an unresolved floating expression.
#[test]
fn unknown_float_symbolic_values_preserve_ieee_uncertainty() -> Result<(), Box<dyn Error>> {
    for kind in 0..5 {
        let (ssa, input, result) = unknown_float_fixture(kind)?;
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.set_symbolic(input, "floating input");
        let definition = ssa
            .recorded_definition(result)
            .ok_or_else(|| io::Error::other("missing expression definition"))?;
        evaluator.evaluate_op(definition.op());
        assert!(
            evaluator.get_concrete(result).is_none(),
            "unknown symbolic float {kind}: {:?}",
            evaluator.get_concrete(result)
        );
    }
    Ok(())
}

/// Floating branch comparisons cannot use an unknown value's address identity.
#[test]
fn unknown_float_branch_comparisons_preserve_both_outcomes() -> Result<(), Box<dyn Error>> {
    for (cmp, unsigned) in [
        (CmpKind::Eq, false),
        (CmpKind::Ne, false),
        (CmpKind::Lt, true),
        (CmpKind::Ge, false),
    ] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        builder.in_block(0, |block| {
            let input = block.load_arg(SsaDefSpec::argument(0, MockType::F64), 0)?;
            block.branch_cmp(input, input, cmp, unsigned, 1, 2)
        })?;
        for (target, value) in [(1, 42), (2, 7)] {
            builder.in_block(target, |block| {
                let result =
                    block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(value))?;
                block.ret(Some(result))
            })?;
        }
        let ssa = builder.finish()?;
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            errors.is_empty(),
            "legal float branch fixture required: {errors:?}"
        );
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert!(
            facts.returned().and_then(Value::constant).is_none(),
            "unknown float branch {cmp:?}, unsigned={unsigned}: {:?}",
            facts.returned()
        );
    }
    Ok(())
}

/// Explicit conversion formats also govern the machine API without SSA metadata.
#[test]
fn direct_float_conversion_formats_govern_integer_carriers() {
    let source = SsaVarId::from_index(0);
    let dest = SsaVarId::from_index(1);
    for to_integer in [false, true] {
        let bits = if to_integer { 0x3fc0u16 } else { 0x3f00 };
        let op = if to_integer {
            SsaOp::FloatToInt {
                dest,
                operand: source,
                target: MockType::I32,
                unsigned: false,
                overflow_check: false,
                source: Some(FloatFormat::BFloat16),
            }
        } else {
            SsaOp::FloatConv {
                dest,
                operand: source,
                target: MockType::F64,
                rounding: None,
                source: Some(FloatFormat::BFloat16),
                environment: None,
            }
        };
        let expected = if to_integer {
            ConstValue::I32(1)
        } else {
            ConstValue::from(0.5f64)
        };
        assert_eq!(
            MACHINE.evaluate(&op, Some(if to_integer { 32 } else { 64 }), |var| (var
                == source)
                .then_some(ConstValue::<MockTarget>::U16(bits))),
            Some(expected)
        );
    }
}

/// Checks exact zero bits for cancellation of one named address at every native width.
fn identical_symbol_cancellation(operation: IntegerOp) -> Result<(), Box<dyn Error>> {
    for (ptr_size, ty) in [
        (PointerSize::Bit8, MockType::U8),
        (PointerSize::Bit16, MockType::U16),
        (PointerSize::Bit32, MockType::U32),
        (PointerSize::Bit64, MockType::U64),
        (PointerSize::Bit128, MockType::U128),
    ] {
        let machine = Machine {
            ptr_size,
            ..MACHINE
        };
        let result = execute_machine(
            &[(ty, ConstValue::Symbol(7)), (ty, ConstValue::Symbol(7))],
            ty,
            |dest, inputs| operation.operation(dest, inputs),
            machine,
        )?;
        assert_eq!(
            result
                .constant()
                .and_then(|value| value.to_bytes(Endianness::Little, ptr_size)),
            Some(vec![0; ptr_size.bytes()]),
            "{operation:?}, {ptr_size:?}",
        );
    }
    Ok(())
}

/// A symbol minus itself is zero without requiring a numeric symbol projection.
#[test]
fn identical_symbol_subtraction_shares_the_integer_kernel() -> Result<(), Box<dyn Error>> {
    identical_symbol_cancellation(IntegerOp::Sub)
}

/// XOR cancellation also applies to the stable bits of a named address.
#[test]
fn identical_symbol_xor_shares_the_integer_kernel() -> Result<(), Box<dyn Error>> {
    identical_symbol_cancellation(IntegerOp::Xor)
}

/// Cancellation cannot invent zero for distinct addresses, metadata, floating NaNs or vector layouts.
#[test]
fn address_cancellation_preserves_other_value_domains() {
    let nan = ConstValue::from(f64::NAN);
    let vars = [SsaVarId::from_index(0), SsaVarId::from_index(1)];
    for operation in [IntegerOp::Sub, IntegerOp::Xor] {
        let op = operation.operation(SsaVarId::from_index(2), &vars);
        for values in [
            [ConstValue::Symbol(7), ConstValue::Symbol(8)],
            [ConstValue::Type(7), ConstValue::Type(7)],
            [ConstValue::MethodHandle(7), ConstValue::MethodHandle(7)],
            [ConstValue::FieldHandle(7), ConstValue::FieldHandle(7)],
            [nan.clone(), nan.clone()],
        ] {
            let result = MACHINE.evaluate(&op, Some(64), |var| values.get(var.index()).cloned());
            assert!(
                result.as_ref().is_none_or(|value| !value.is_zero()),
                "{operation:?}: {values:?}"
            );
        }
        let values = [ConstValue::Symbol(7), ConstValue::Symbol(7)];
        let result = MACHINE.evaluate(
            &op,
            Destination {
                bits: Some(64),
                lanes: Some((8, 8)),
                lane_format: None,
                integer: false,
                integer_type: None,
            },
            |var| values.get(var.index()).cloned(),
        );
        assert_eq!(result, None);
    }
}

/// Typed integer conversion arithmetic needs only a declared scalar layout.
#[test]
fn integer_conversions_execute_without_host_arithmetic() -> Result<(), Box<dyn Error>> {
    for (source, value, target, unsigned, expected) in [
        (
            MockType::U8,
            ConstValue::U8(255),
            MockType::I64,
            false,
            ConstValue::I64(-1),
        ),
        (
            MockType::I8,
            ConstValue::I8(-1),
            MockType::U64,
            true,
            ConstValue::U64(255),
        ),
        (
            MockType::U32,
            ConstValue::U32(0x1234_56ff),
            MockType::I8,
            false,
            ConstValue::I8(-1),
        ),
        (
            MockType::I64,
            ConstValue::I64(-2),
            MockType::U128,
            false,
            vector(128, &[u128::MAX - 1])?,
        ),
        (
            MockType::U128,
            vector(128, &[(1u128 << 100) | 0xfe])?,
            MockType::U8,
            true,
            ConstValue::U8(254),
        ),
    ] {
        let result = execute(&[(source, value)], target, |dest, inputs| SsaOp::IntConv {
            dest,
            operand: operand(inputs, 0),
            target,
            unsigned,
            overflow_check: false,
        })?;
        assert_eq!(result.constant(), Some(&expected));
    }
    Ok(())
}

/// Floating-to-integer conversion retains bits beyond the 64-bit scalar carriers.
#[test]
fn floating_to_wide_integers_execute_exact_register_patterns() -> Result<(), Box<dyn Error>> {
    for (input, expected) in [(2f64.powi(100), 1u128 << 100), (-0.0, 0), (1.75, 1)] {
        let result = execute(
            &[(MockType::F64, ConstValue::from(input))],
            MockType::U128,
            |dest, inputs| SsaOp::FloatToInt {
                dest,
                operand: operand(inputs, 0),
                target: MockType::U128,
                unsigned: true,
                overflow_check: false,
                source: Some(FloatFormat::Double),
            },
        )?;
        assert_eq!(result.constant(), Some(&vector(128, &[expected])?));
    }
    Ok(())
}

/// Integer-to-float conversion consumes the entire wide source and the stated rounding.
#[test]
fn integer_to_float_reads_full_wide_sources() -> Result<(), Box<dyn Error>> {
    for (input, unsigned, rounding, expected) in [
        (1u128 << 100, true, None, 2f64.powi(100)),
        (1u128 << 127, false, None, -2f64.powi(127)),
        (
            u128::MAX,
            true,
            Some(FloatRounding::TowardZero),
            f64::from_bits(0x47ef_ffff_ffff_ffff),
        ),
    ] {
        let result = execute(
            &[(MockType::U128, vector(128, &[input])?)],
            MockType::F64,
            |dest, inputs| SsaOp::IntToFloat {
                dest,
                operand: operand(inputs, 0),
                target: MockType::F64,
                unsigned,
                rounding,
                environment: None,
            },
        )?;
        assert_eq!(result.constant(), Some(&ConstValue::from(expected)));
    }
    Ok(())
}

/// A finite check preserves exact floating register bits, including negative zero.
#[test]
fn finite_checks_preserve_finite_values() -> Result<(), Box<dyn Error>> {
    for input in [0.0f64, -0.0, 1.25, f64::MAX] {
        let value = ConstValue::from(input);
        let result = execute(
            &[(MockType::F64, value.clone())],
            MockType::F64,
            |dest, inputs| SsaOp::Ckfinite {
                dest,
                operand: operand(inputs, 0),
            },
        )?;
        assert_eq!(result.constant(), Some(&value));
    }
    Ok(())
}

/// Checks whether execution can reach an independently stated marker after a conversion.
fn conversion_reaches_marker(kind: u8, trapping: bool) -> Result<bool, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let (source, value, target) = match kind {
            0 => (
                MockType::U16,
                ConstValue::U16(if trapping { 256 } else { 127 }),
                MockType::I8,
            ),
            1 => (
                MockType::F64,
                ConstValue::from(if trapping { f64::INFINITY } else { 1.25 }),
                MockType::I32,
            ),
            _ => (
                MockType::F64,
                ConstValue::from(if trapping { f64::NAN } else { 1.25 }),
                MockType::F64,
            ),
        };
        let input = block.const_value(SsaDefSpec::tmp(source), value)?;
        block.emit_def(SsaDefSpec::tmp(target), |dest| match kind {
            0 => SsaOp::IntConv {
                dest,
                operand: input,
                target,
                unsigned: true,
                overflow_check: true,
            },
            1 => SsaOp::FloatToInt {
                dest,
                operand: input,
                target,
                unsigned: false,
                overflow_check: true,
                source: Some(FloatFormat::Double),
            },
            _ => SsaOp::Ckfinite {
                dest,
                operand: input,
            },
        })?;
        let marker = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(42))?;
        block.ret(Some(marker))
    })?;
    let ssa = builder.finish()?;
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
    let facts = StaticExecution::new(&ssa, &AnalysisContext::new(MACHINE)).run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    Ok(facts.returned().and_then(Value::constant) == Some(&ConstValue::U32(42)))
}

/// Proven integer conversion overflow stops before later constants or observations.
#[test]
fn checked_integer_conversion_traps_before_later_work() -> Result<(), Box<dyn Error>> {
    assert!(conversion_reaches_marker(0, false)?);
    assert!(!conversion_reaches_marker(0, true)?);
    Ok(())
}

/// Proven floating conversion overflow cannot acquire a normal return.
#[test]
fn checked_float_conversion_traps_before_later_work() -> Result<(), Box<dyn Error>> {
    assert!(conversion_reaches_marker(1, false)?);
    assert!(!conversion_reaches_marker(1, true)?);
    Ok(())
}

/// A nonfinite check traps even when its result has no consumer.
#[test]
fn nonfinite_checks_trap_before_later_work() -> Result<(), Box<dyn Error>> {
    assert!(conversion_reaches_marker(2, false)?);
    assert!(!conversion_reaches_marker(2, true)?);
    Ok(())
}

/// Runtime uncertainty, unsupported domains and proven overflow are different outcomes.
#[test]
fn conversion_control_outcomes_do_not_conflate_refusal_with_traps() {
    let input = SsaVarId::from_index(0);
    let op = SsaOp::<MockTarget>::IntConv {
        dest: SsaVarId::from_index(1),
        operand: input,
        target: MockType::I8,
        unsigned: false,
        overflow_check: true,
    };
    assert_eq!(
        MACHINE.evaluate_control(&op, Some(8), |_| None),
        Some(Evaluation::Unknown)
    );
    assert_eq!(
        MACHINE.evaluate_control(&op, Some(8), |_| Some(ConstValue::Type(7))),
        Some(Evaluation::Unsupported)
    );
    assert_eq!(
        MACHINE.evaluate_control(&op, Some(8), |_| Some(ConstValue::I16(256))),
        Some(Evaluation::Trap(ArithmeticTrap::Overflow))
    );
    assert_eq!(
        MACHINE.evaluate_control(&op, Some(8), |_| Some(ConstValue::I16(127))),
        Some(Evaluation::Value(ConstValue::I8(127)))
    );
    let unsupported = SsaOp::<MockTarget>::IntConv {
        dest: SsaVarId::from_index(1),
        operand: input,
        target: MockType::Unknown,
        unsigned: false,
        overflow_check: true,
    };
    assert_eq!(
        MACHINE.evaluate_control(&unsupported, Some(8), |_| Some(ConstValue::I16(1))),
        Some(Evaluation::Unsupported)
    );
}

/// Conversion uses the source register width before applying its explicit signedness.
#[test]
fn conversion_reads_attested_source_width_before_widening() {
    for (bits, value, expected) in [
        (8, ConstValue::U64(255), -1),
        (16, ConstValue::U64(0xff80), -128),
        (1, ConstValue::True, -1),
    ] {
        let op = SsaOp::<MockTarget>::IntConv {
            dest: SsaVarId::from_index(1),
            operand: SsaVarId::from_index(0),
            target: MockType::I64,
            unsigned: false,
            overflow_check: false,
        };
        let result = MACHINE.evaluate_control_with_operand_layouts(
            &op,
            Some(64),
            |_| Some(value.clone()),
            |_| {
                OperandLayout::Register(Destination {
                    bits: Some(bits),
                    lanes: None,
                    lane_format: None,
                    integer: true,
                    integer_type: None,
                })
            },
        );
        assert_eq!(result, Some(Evaluation::Value(ConstValue::I64(expected))));
    }
}

/// A sixteen-bit finite check requires its real format, and mismatches cannot prove a trap.
#[test]
fn finite_checks_read_exact_declared_formats() {
    let op = SsaOp::<MockTarget>::Ckfinite {
        dest: SsaVarId::from_index(1),
        operand: SsaVarId::from_index(0),
    };
    let evaluate = |format| {
        MACHINE.evaluate_control_with_operand_layouts(
            &op,
            None,
            |_| Some(ConstValue::U16(0x7c00)),
            |_| OperandLayout::Float(format),
        )
    };
    assert_eq!(
        evaluate(FloatFormat::Half),
        Some(Evaluation::Trap(ArithmeticTrap::NonFinite))
    );
    assert_eq!(
        evaluate(FloatFormat::BFloat16),
        Some(Evaluation::Value(
            ConstValue::float(FloatFormat::BFloat16, 0x7c00).unwrap_or_else(|| process::abort())
        ))
    );
    assert_eq!(
        MACHINE.evaluate_control_with_operand_layouts(
            &op,
            None,
            |_| Some(ConstValue::from(1.0f64)),
            |_| OperandLayout::Float(FloatFormat::Half)
        ),
        Some(Evaluation::Unsupported)
    );
    assert_eq!(
        MACHINE.evaluate_control(&op, None, |_| Some(ConstValue::U16(0x7c00))),
        Some(Evaluation::Unsupported)
    );
}

/// Builds a conversion whose known or runtime input controls its replacement proof.
fn conversion_rewrite_fixture(
    kind: u8,
    known: bool,
) -> Result<(SsaFunction<MockTarget>, SsaVarId), Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let mut returned = None;
    builder.in_block(0, |block| {
        let (source, value, target) = match kind {
            0 => (MockType::I16, ConstValue::I16(127), MockType::I8),
            1 => (MockType::F64, ConstValue::from(1.75f64), MockType::I32),
            _ => (MockType::F64, ConstValue::from(-0.0f64), MockType::F64),
        };
        let input = if known {
            block.const_value(SsaDefSpec::tmp(source), value)?
        } else {
            block.load_arg(SsaDefSpec::argument(0, source), 0)?
        };
        let result = block.emit_def(SsaDefSpec::tmp(target), |dest| match kind {
            0 => SsaOp::IntConv {
                dest,
                operand: input,
                target,
                unsigned: false,
                overflow_check: true,
            },
            1 => SsaOp::FloatToInt {
                dest,
                operand: input,
                target,
                unsigned: false,
                overflow_check: true,
                source: Some(FloatFormat::Double),
            },
            _ => SsaOp::Ckfinite {
                dest,
                operand: input,
            },
        })?;
        returned = Some(result);
        block.ret(Some(result))
    })?;
    let ssa = builder.finish()?;
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
    Ok((
        ssa,
        returned.ok_or_else(|| io::Error::other("missing conversion result"))?,
    ))
}

/// Only a successful check with a value proof can be replaced by its constant.
#[test]
fn conversion_rewrites_require_nontrapping_evidence() -> Result<(), Box<dyn Error>> {
    for kind in 0..3 {
        for known in [false, true] {
            let (mut ssa, result) = conversion_rewrite_fixture(kind, known)?;
            let facts = StaticExecution::new(&ssa, &AnalysisContext::new(MACHINE)).run();
            assert!(facts.is_complete());
            let proof = known.then_some(ReplacementProof::Completed);
            assert_eq!(facts.replacement_proof(result), proof);
            if known {
                let expected = match kind {
                    0 => ConstValue::I8(127),
                    1 => ConstValue::I32(1),
                    _ => ConstValue::from(-0.0f64),
                };
                assert_eq!(facts.returned().and_then(Value::constant), Some(&expected));
            } else {
                assert!(facts.returned().and_then(Value::constant).is_none());
            }
            let count = fold_constants(&mut ssa, &facts)?;
            assert_eq!(count, usize::from(known));
            assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
            assert_eq!(
                ssa.recorded_definition(result)
                    .is_some_and(|definition| matches!(definition.op(), SsaOp::Const { .. })),
                known
            );
        }
    }
    Ok(())
}

/// A successful conversion or finite check cannot erase a later trap at the same definition.
#[test]
fn conversion_proofs_require_every_dynamic_visit() -> Result<(), Box<dyn Error>> {
    for kind in 0..3 {
        let (source_type, target, safe, trapping, expected) = match kind {
            0 => (
                MockType::I16,
                MockType::I8,
                ConstValue::I16(127),
                ConstValue::I16(128),
                ConstValue::I8(127),
            ),
            1 => (
                MockType::F64,
                MockType::I32,
                ConstValue::from(1.0f64),
                ConstValue::from(f64::INFINITY),
                ConstValue::I32(1),
            ),
            _ => (
                MockType::F64,
                MockType::F64,
                ConstValue::from(1.0f64),
                ConstValue::from(f64::INFINITY),
                ConstValue::from(1.0f64),
            ),
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let (safe, trapping, zero, one, two) = builder.in_block(0, |block| {
            let safe = block.const_value(SsaDefSpec::tmp(source_type), safe)?;
            let trapping = block.const_value(SsaDefSpec::tmp(source_type), trapping)?;
            let zero = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
            let one = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
            let two = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
            block.jump(1)?;
            Ok((safe, trapping, zero, one, two))
        })?;
        let (input, index) = builder.in_block(1, |block| {
            Ok((
                block.empty_phi(SsaDefSpec::tmp(source_type))?,
                block.empty_phi(SsaDefSpec::tmp(MockType::U8))?,
            ))
        })?;
        let (result, next) = builder.in_block(1, |block| {
            let result = block.emit_def(SsaDefSpec::tmp(target), |dest| match kind {
                0 => SsaOp::IntConv {
                    dest,
                    operand: input,
                    target,
                    unsigned: false,
                    overflow_check: true,
                },
                1 => SsaOp::FloatToInt {
                    dest,
                    operand: input,
                    target,
                    unsigned: false,
                    overflow_check: true,
                    source: Some(FloatFormat::Double),
                },
                _ => SsaOp::Ckfinite {
                    dest,
                    operand: input,
                },
            })?;
            let next = block.add(SsaDefSpec::tmp(MockType::U8), index, one)?;
            block.branch_cmp(next, two, CmpKind::Lt, true, 1, 2)?;
            Ok((result, next))
        })?;
        builder.add_phi_operand(1, input, 0, safe)?;
        builder.add_phi_operand(1, input, 1, trapping)?;
        builder.add_phi_operand(1, index, 0, zero)?;
        builder.add_phi_operand(1, index, 1, next)?;
        builder.in_block(2, |block| block.ret(Some(result)))?;
        let mut ssa = builder.finish()?;
        assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert_eq!(facts.constant(result), Some(&expected));
        assert_eq!(facts.usage().traps, 1);
        assert!(facts.returned().is_none());
        assert_eq!(facts.replacement_proof(result), None);
        fold_constants(&mut ssa, &facts)?;
        // Repair can compact variable IDs. Check the operation family, then
        // execute the repaired function to validate the preserved trap.
        assert_eq!(
            ssa.all_instructions()
                .filter(|instruction| matches!(
                    (kind, instruction.op()),
                    (
                        0,
                        SsaOp::IntConv {
                            overflow_check: true,
                            ..
                        }
                    ) | (
                        1,
                        SsaOp::FloatToInt {
                            overflow_check: true,
                            ..
                        }
                    ) | (2, SsaOp::Ckfinite { .. })
                ))
                .count(),
            1
        );
        assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        let rewritten = StaticExecution::new(&ssa, &context).run();
        assert!(rewritten.is_complete());
        assert_eq!(rewritten.usage().traps, 1);
        assert!(rewritten.returned().is_none());
    }
    Ok(())
}

/// Builds signed and unsigned widening of one source with an optional exact value.
fn address_conversion_identity_fixture(
    pointer: bool,
    widening: bool,
    known: bool,
) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    builder.in_block(0, |block| {
        let source = if widening {
            MockType::U32
        } else {
            MockType::U64
        };
        let target = if pointer {
            MockType::Ptr
        } else {
            MockType::U64
        };
        let input = if known {
            let value = if widening {
                ConstValue::U32(0x8000_0007)
            } else {
                ConstValue::U64(7)
            };
            block.const_value(SsaDefSpec::tmp(source), value)?
        } else {
            block.load_arg(SsaDefSpec::argument(0, source), 0)?
        };
        let signed = block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::IntConv {
            dest,
            operand: input,
            target: MockType::I64,
            unsigned: false,
            overflow_check: false,
        })?;
        let converted = block.emit_def(SsaDefSpec::tmp(target), |dest| {
            if pointer {
                SsaOp::IntToPtr {
                    dest,
                    operand: input,
                    target,
                }
            } else {
                SsaOp::IntConv {
                    dest,
                    operand: input,
                    target,
                    unsigned: true,
                    overflow_check: false,
                }
            }
        })?;
        let equal = block.emit_def(SsaDefSpec::tmp(MockType::I1), |dest| SsaOp::Ceq {
            dest,
            left: signed,
            right: converted,
            signaling: false,
        })?;
        block.ret(Some(equal))
    })?;
    let ssa = builder.finish()?;
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "conversion fixture: {errors:?}");
    Ok(ssa)
}

/// Signed and unsigned widening of an unresolved 32-bit source can produce different 64-bit values.
#[test]
fn widening_runtime_conversions_preserve_source_signedness() -> Result<(), Box<dyn Error>> {
    for pointer in [false, true] {
        let concrete = address_conversion_identity_fixture(pointer, true, true)?;
        let context = AnalysisContext::new(MACHINE);
        let concrete_facts = StaticExecution::new(&concrete, &context).run();
        assert!(concrete_facts.is_complete());
        assert_eq!(
            concrete_facts.returned().and_then(Value::constant),
            Some(&ConstValue::False)
        );
        let runtime = address_conversion_identity_fixture(pointer, true, false)?;
        let facts = StaticExecution::new(&runtime, &context).run();
        assert!(facts.is_complete());
        assert!(
            facts.returned().and_then(Value::constant).is_none(),
            "pointer={pointer}: {:?}",
            facts.returned()
        );
    }
    Ok(())
}

/// Exact pointer-width conversions preserve an unresolved register's complete identity.
#[test]
fn lossless_runtime_conversions_preserve_identity() -> Result<(), Box<dyn Error>> {
    for pointer in [false, true] {
        for known in [false, true] {
            let ssa = address_conversion_identity_fixture(pointer, false, known)?;
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&ssa, &context).run();
            assert!(facts.is_complete());
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&ConstValue::True)
            );
        }
    }
    Ok(())
}

/// Sixteen read-only bytes at `0x1000`, as a decoder's encoded input lies.
struct EncodedImage;

/// The image's bytes: `0x10, 0x11, ..., 0x1f`.
const ENCODED: [u8; 16] = [
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

impl ImageData for EncodedImage {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        (0x1000..0x1010).contains(&address).then_some(ImageRegion {
            start: 0x1000,
            end: 0x1010,
            mutability: Mutability::ReadOnly,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(address.checked_sub(0x1000)?).ok()?;
        ENCODED.get(start..start.checked_add(len)?)
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}

/// What a function returns that loads from the image at `0x1000` with
/// `load`, of type `ty`, and XORs the result with `key` as x86's `pxor`
/// states it: every lane's bits, of no stated element.
fn loaded_and_keyed(
    ty: MockType,
    key: ConstValue<MockTarget>,
    load: impl FnOnce(SsaVarId, SsaVarId) -> SsaOp<MockTarget>,
) -> Result<Value<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let source = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::U64(0x1000))?;
        let key = block.const_value(SsaDefSpec::tmp(ty), key)?;
        let loaded = block.emit_def(SsaDefSpec::tmp(ty), |dest| load(dest, source))?;
        let plain = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::VectorBinary {
            dest,
            left: loaded,
            right: key,
            kind: VectorBinaryKind::Xor,
            element: VectorElement::default(),
            environment: None,
        })?;
        block.ret(Some(plain))
    })?;
    let ssa = builder.finish()?;
    let context = AnalysisContext::new(MACHINE).with_image(&EncodedImage);
    let facts = StaticExecution::new(&ssa, &context).run();
    if !facts.is_complete() {
        return Err(io::Error::other(format!("refused: {:?}", facts.completion())).into());
    }
    facts
        .returned()
        .cloned()
        .ok_or_else(|| io::Error::other("no return").into())
}

/// **A vector read from known bytes is the vector they spell.** gcc
/// vectorises a decoder into 16-byte loads of the encoded image, a `pxor`
/// with the key broadcast, and 16-byte stores: a load the execution does not
/// read leaves every decoded byte unknown. A whole-vector load is its
/// bytes in the image's order, and a broadcast load its first element in
/// every lane; each is checked against the bytes themselves.
#[test]
fn vector_loads_of_known_bytes_are_their_bytes() -> Result<(), Box<dyn Error>> {
    let held = |value: &Value<MockTarget>| match value.constant() {
        Some(ConstValue::Vector(vector)) => Some(vector.bits_at(0, vector.bits())),
        _ => None,
    };
    let little = |bytes: &[u8]| {
        bytes
            .iter()
            .rev()
            .fold(0u128, |word, byte| (word << 8) | u128::from(*byte))
    };
    let whole = loaded_and_keyed(
        MockType::V4I32,
        vector(128, &[0x5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a])?,
        |dest, addr| SsaOp::VectorLoad {
            dest,
            addr,
            vector_type: MockType::V4I32,
            access: AccessQualifiers::FLAT,
        },
    )?;
    let decoded: Vec<u8> = ENCODED.iter().map(|byte| byte ^ 0x5a).collect();
    assert_eq!(held(&whole), Some(little(&decoded)), "{whole:?}");
    let broadcast = loaded_and_keyed(
        MockType::V4I32,
        vector(32, &[0x5a5a_5a5a; 4])?,
        |dest, addr| SsaOp::VectorBroadcastLoad {
            dest,
            addr,
            vector_type: MockType::V4I32,
            access: AccessQualifiers::FLAT,
        },
    )?;
    let element = little(decoded.get(..4).unwrap_or_default());
    assert_eq!(
        held(&broadcast),
        Some(element | element << 32 | element << 64 | element << 96),
        "{broadcast:?}"
    );
    Ok(())
}

/// Loads at 0x400000 above their module addresses, as a non-PIE ELF does.
struct LoadedAbove;

impl ImageData for LoadedAbove {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, _: u64) -> Option<ImageRegion> {
        None
    }

    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }

    fn number_offset(&self) -> Option<i64> {
        Some(-0x40_0000)
    }
}

/// **A pointer placed in a vector's lanes is its address's bits.** x86-64's
/// `movq %rax, %xmm1` puts a code pointer in a lane, and `punpcklqdq` pairs it
/// with another: two module addresses, one a pointer and one an integer, go
/// into one vector as the numbers the image is loaded at, so a table filled
/// from the vector holds the helpers' addresses. Splatting and inserting a
/// pointer place the same bits. A stack address has no bits to place.
#[test]
fn a_pointer_placed_in_vector_lanes_is_its_address_bits() -> Result<(), Box<dyn Error>> {
    let image = LoadedAbove;
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    let run = |build: &dyn Fn(
        &mut SsaBlockBuilder<'_, MockTarget>,
    ) -> analyssa::Result<SsaVarId>|
     -> Result<Value<MockTarget>, Box<dyn Error>> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |block| {
            let result = build(block)?;
            block.ret(Some(result))
        })?;
        let ssa = builder.finish()?;
        let facts = StaticExecution::new(&ssa, &context).run();
        Ok(facts.returned().cloned().unwrap_or(Value::Unknown))
    };
    let paired = run(&|block| {
        let pointer =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x11c0))?;
        let integer =
            block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::Symbol(0x11d0))?;
        let low = block.emit_def(SsaDefSpec::tmp(MockType::V1I64), |dest| {
            SsaOp::VectorReinterpret {
                dest,
                value: pointer,
                target_type: MockType::V1I64,
            }
        })?;
        let high = block.emit_def(SsaDefSpec::tmp(MockType::V1I64), |dest| {
            SsaOp::VectorReinterpret {
                dest,
                value: integer,
                target_type: MockType::V1I64,
            }
        })?;
        block.emit_def(SsaDefSpec::tmp(MockType::V2I64), |dest| {
            SsaOp::VectorShuffle {
                dest,
                left: low,
                right: Some(high),
                selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                    VectorShuffleLane::Left(0),
                    VectorShuffleLane::Right(0),
                ])),
            }
        })
    })?;
    assert_eq!(
        paired,
        Value::Const(vector(64, &[0x40_11c0, 0x40_11d0])?),
        "the pointer lane is its loaded address"
    );
    let splat = run(&|block| {
        let pointer =
            block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x11c0))?;
        block.emit_def(SsaDefSpec::tmp(MockType::V2I64), |dest| {
            SsaOp::VectorSplat {
                dest,
                value: pointer,
                vector_type: MockType::V2I64,
            }
        })
    })?;
    assert_eq!(splat, Value::Const(vector(64, &[0x40_11c0, 0x40_11c0])?));
    Ok(())
}
