//! Symbolic expression evaluation tests.

use std::{process, sync::Arc};

use analyssa::{
    PointerSize,
    analysis::{
        evaluator::SsaEvaluator,
        symbolic::{SymbolicEvaluator, SymbolicExpr},
    },
    ir::{
        block::SsaBlock,
        evaluation::NumericOp,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MACHINE, MockTarget, MockType},
};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        MockType::I32,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

#[test]
fn symbolic_expr_constant_and_variable_construction() {
    let c = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(42));
    assert!(c.is_constant());
    assert!(!c.is_variable());
    assert_eq!(c.as_constant().and_then(|cv| cv.as_i64()), Some(42));

    let v = SymbolicExpr::<MockTarget>::Variable(SsaVarId::from_index(7));
    assert!(v.is_variable());
    assert!(!v.is_constant());
    assert_eq!(v.as_variable(), Some(SsaVarId::from_index(7)));
}

#[test]
fn symbolic_expr_binary_operation() {
    let a = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(5));
    let b = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(3));

    let sum = SymbolicExpr::<MockTarget>::Binary {
        op: NumericOp::Add,
        left: Arc::new(a),
        right: Arc::new(b),
    };

    assert!(matches!(sum, SymbolicExpr::Binary { .. }));
    assert!(!matches!(sum, SymbolicExpr::Constant(_)));
}

#[test]
fn symbolic_expr_unary_operation() {
    let v = SymbolicExpr::<MockTarget>::Variable(SsaVarId::from_index(0));

    let neg = SymbolicExpr::<MockTarget>::Unary {
        op: NumericOp::Neg,
        operand: Arc::new(v),
    };

    assert!(matches!(neg, SymbolicExpr::Unary { .. }));
    assert!(!matches!(neg, SymbolicExpr::Constant(_)));
}

#[test]
fn symbolic_expr_complex_nesting() {
    let a = SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(1));
    let b = SymbolicExpr::<MockTarget>::Variable(SsaVarId::from_index(0));

    let sum = SymbolicExpr::<MockTarget>::Binary {
        op: NumericOp::Add,
        left: Arc::new(a),
        right: Arc::new(b),
    };
    let _prod = SymbolicExpr::<MockTarget>::Binary {
        op: NumericOp::Mul,
        left: Arc::new(sum),
        right: Arc::new(SymbolicExpr::<MockTarget>::Constant(ConstValue::I32(2))),
    };
}

#[test]
fn symbolic_expr_helpers_constant_and_binary() {
    let c = SymbolicExpr::<MockTarget>::constant(ConstValue::I32(10));
    assert!(c.is_constant());

    let bin = SymbolicExpr::<MockTarget>::binary(
        NumericOp::Add,
        SymbolicExpr::<MockTarget>::constant(ConstValue::I32(1)),
        SymbolicExpr::<MockTarget>::constant(ConstValue::I32(2)),
    );
    assert!(matches!(bin, SymbolicExpr::Binary { .. }));

    let uni = SymbolicExpr::<MockTarget>::unary(
        NumericOp::Neg,
        SymbolicExpr::<MockTarget>::constant(ConstValue::I32(5)),
    );
    assert!(matches!(uni, SymbolicExpr::Unary { .. }));
}

#[test]
fn symbolic_evaluator_tracks_expressions() {
    let mut ssa = SsaFunction::new(0, 0);
    let param = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let const_val = local(&mut ssa, 1, 0, 1);
    let sum = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: const_val,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: param,
        right: const_val,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut sym_eval = SymbolicEvaluator::new(&ssa, PointerSize::Bit64);
    sym_eval.evaluate_block(0);

    let c_expr = sym_eval.get_expression(const_val);
    assert!(c_expr.is_some());
    assert!(some_or_abort(c_expr).is_constant());

    let s_expr = sym_eval.get_expression(sum);
    assert!(s_expr.is_some());
    assert!(matches!(
        some_or_abort(s_expr),
        SymbolicExpr::TypedBinary { .. }
    ));

    // param may or may not have an expression (entry-point variables are
    // not tracked unless explicitly set_symbolic is called)
    let _p_expr = sym_eval.get_expression(param);
}

#[test]
fn symbolic_evaluator_handles_empty_function() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    let mut sym_eval = SymbolicEvaluator::new(&ssa, PointerSize::Bit64);
    sym_eval.evaluate_block(0);
}

#[test]
fn symbolic_expr_tracks_through_copy() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
    block.add_instruction(instr(SsaOp::returns(Some(b))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut sym_eval = SymbolicEvaluator::new(&ssa, PointerSize::Bit64);
    sym_eval.evaluate_block(0);

    let expr = sym_eval.get_expression(b);
    assert!(expr.is_some());
    assert!(some_or_abort(expr).is_constant());
}

#[test]
fn ssa_evaluator_concrete_and_symbolic() {
    let mut ssa = SsaFunction::new(0, 0);
    let param = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let c = local(&mut ssa, 1, 0, 1);
    let prod = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: c,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: prod,
        left: param,
        right: c,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(prod))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);

    assert!(eval.is_concrete(c));
    assert!(eval.is_concrete(c));
    assert_eq!(eval.get_concrete(c).and_then(|cv| cv.as_i64()), Some(2));

    // prod is unknown (param is unknown too) or symbolic
    let _sym = eval.get_symbolic(prod);
}

#[test]
fn evaluate_with_trace_resolves_complex_path() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 0, 2);
    let d = local(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: c,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: d,
        left: c,
        right: a,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(d))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);

    let trace = eval.evaluate_with_trace(d, 10);
    assert_eq!(trace, Some(40));
}

/// Symbolic carrier arithmetic agrees with independent byte-domain bit patterns.
#[test]
fn symbolic_concrete_integer_helpers_cover_wrapping_shifts_and_rotates() {
    for raw in 0u16..=255 {
        let value = ConstValue::<MockTarget>::U8(raw as u8);
        for count in [0u32, 1, 7, 8, 9, 31, 32, 127] {
            let amount = ConstValue::U32(count);
            for (op, expected) in [
                (NumericOp::Rol, Some((raw as u8).rotate_left(count))),
                (NumericOp::Ror, Some((raw as u8).rotate_right(count))),
                (NumericOp::Shl, (count < 8).then(|| (raw as u8) << count)),
                (NumericOp::ShrU, (count < 8).then(|| (raw as u8) >> count)),
                (
                    NumericOp::ShrS,
                    (count < 8).then(|| ((raw as u8 as i8) >> count) as u8),
                ),
            ] {
                let result = op.evaluate_binary(&value, &amount, PointerSize::Bit64);
                assert_eq!(
                    result
                        .and_then(|value| value.scalar_integer_pattern(PointerSize::Bit64))
                        .map(|p| p.0),
                    expected.map(u128::from),
                    "{op:?} {raw} {count}"
                );
            }
        }
        for (op, expected) in [
            (NumericOp::Neg, (raw as u8).wrapping_neg()),
            (NumericOp::Not, !(raw as u8)),
            (NumericOp::BRev, (raw as u8).reverse_bits()),
            (NumericOp::BSwap, raw as u8),
        ] {
            let result = some_or_abort(op.evaluate_unary(&value, PointerSize::Bit64));
            assert_eq!(
                result
                    .scalar_integer_pattern(PointerSize::Bit64)
                    .map(|p| p.0),
                Some(u128::from(expected)),
                "{op:?} {raw}"
            );
        }
    }
}

/// A converted unknown cannot become its unconverted source through substitution.
#[test]
fn symbolic_unknown_conversions_retain_their_target() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let source = local(&mut ssa, 0, 0, 0);
    let dest = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, 1),
        MockType::I8,
    );
    let mut evaluator = SymbolicEvaluator::with_machine(&ssa, MACHINE);
    evaluator.set_symbolic(source, "input");
    evaluator.evaluate_op(&SsaOp::IntConv {
        dest,
        operand: source,
        target: MockType::I8,
        unsigned: false,
        overflow_check: false,
    });
    let expression = some_or_abort(evaluator.get_expression(dest));
    assert_eq!(
        expression
            .substitute_named("input", &SymbolicExpr::constant_i32(511))
            .simplify(PointerSize::Bit64)
            .as_constant(),
        Some(&ConstValue::I8(-1)),
    );
}

/// A known conversion uses the SSA target instead of copying its source carrier.
#[test]
fn symbolic_known_integer_conversions_use_the_shared_machine() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let source = local(&mut ssa, 0, 0, 0);
    let dest = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, 1),
        MockType::I8,
    );
    for (check, expected) in [(false, Some(0xffu128)), (true, None)] {
        let mut evaluator = SymbolicEvaluator::with_machine(&ssa, MACHINE);
        evaluator.set_constant(source, ConstValue::I32(511));
        evaluator.evaluate_op(&SsaOp::IntConv {
            dest,
            operand: source,
            target: MockType::I8,
            unsigned: false,
            overflow_check: check,
        });
        let actual = evaluator
            .get_expression(dest)
            .and_then(SymbolicExpr::as_constant)
            .and_then(|value| value.scalar_integer_pattern(PointerSize::Bit64))
            .map(|p| p.0);
        assert_eq!(actual, expected);
    }
}

/// Every binary integer adapter agrees with an independent mathematical oracle.
#[test]
fn symbolic_binary_integer_adapters_match_byte_domain_oracles() {
    for raw in 0u16..=255 {
        let left = ConstValue::<MockTarget>::U8(raw as u8);
        let signed_left = raw as u8 as i8 as i16;
        for other in [0u16, 1, 7, 127, 128, 255] {
            let right = ConstValue::U8(other as u8);
            let signed_right = other as u8 as i8 as i16;
            let signed_div = if signed_right == 0 || (signed_left == -128 && signed_right == -1) {
                None
            } else {
                Some((signed_left / signed_right).rem_euclid(256) as u128)
            };
            for (op, expected) in [
                (NumericOp::Add, Some(u128::from((raw + other) & 255))),
                (NumericOp::Sub, Some(u128::from((raw + 256 - other) & 255))),
                (NumericOp::Mul, Some(u128::from((raw * other) & 255))),
                (NumericOp::And, Some(u128::from(raw & other))),
                (NumericOp::Or, Some(u128::from(raw | other))),
                (NumericOp::Xor, Some(u128::from(raw ^ other))),
                (
                    NumericOp::DivU,
                    (other != 0).then(|| u128::from(raw / other)),
                ),
                (
                    NumericOp::RemU,
                    (other != 0).then(|| u128::from(raw % other)),
                ),
                (NumericOp::DivS, signed_div),
                (
                    NumericOp::RemS,
                    // The SSA remainder shares the representable-quotient
                    // precondition of signed division, including minimum / -1.
                    signed_div.map(|_| (signed_left % signed_right).rem_euclid(256) as u128),
                ),
            ] {
                let result = op
                    .evaluate_binary(&left, &right, PointerSize::Bit64)
                    .and_then(|value| value.scalar_integer_pattern(PointerSize::Bit64))
                    .map(|p| p.0);
                assert_eq!(result, expected, "{op:?} {raw} {other}");
            }
            for (op, expected) in [
                (NumericOp::Eq, raw == other),
                (NumericOp::Ne, raw != other),
                (NumericOp::LtS, signed_left < signed_right),
                (NumericOp::LtU, raw < other),
                (NumericOp::GtS, signed_left > signed_right),
                (NumericOp::GtU, raw > other),
                (NumericOp::LeS, signed_left <= signed_right),
                (NumericOp::LeU, raw <= other),
                (NumericOp::GeS, signed_left >= signed_right),
                (NumericOp::GeU, raw >= other),
            ] {
                assert_eq!(
                    op.evaluate_binary(&left, &right, PointerSize::Bit64)
                        .and_then(|value| value.as_bool()),
                    Some(expected),
                    "{op:?} {raw} {other}"
                );
            }
        }
        for (op, expected) in [
            (
                NumericOp::BitScanForward,
                (raw != 0).then(|| u128::from(raw.trailing_zeros())),
            ),
            (
                NumericOp::BitScanReverse,
                (raw != 0).then(|| u128::from(15 - raw.leading_zeros())),
            ),
            (NumericOp::Popcount, Some(u128::from(raw.count_ones()))),
            (NumericOp::Parity, Some(u128::from(raw.count_ones() % 2))),
        ] {
            assert_eq!(
                op.evaluate_unary(&left, PointerSize::Bit64)
                    .and_then(|value| value.as_u64())
                    .map(u128::from),
                expected,
                "{op:?} {raw}"
            );
        }
    }
}

/// Full-width symbolic operands retain bit 127 across arithmetic and bit operations.
#[test]
fn symbolic_integer_adapters_preserve_all_128_bits() {
    for raw in [
        0u128,
        1,
        1u128 << 127,
        u128::MAX,
        0x123456789abcdef0123456789abcdef0,
    ] {
        let value = some_or_abort(ConstValue::<MockTarget>::of_lane(128, raw));
        for count in [0u32, 1, 63, 64, 127, 128, 129, 255] {
            let amount = ConstValue::U32(count);
            for (op, expected) in [
                (NumericOp::Rol, Some(raw.rotate_left(count))),
                (NumericOp::Ror, Some(raw.rotate_right(count))),
                (NumericOp::Shl, (count < 128).then(|| raw << count)),
                (NumericOp::ShrU, (count < 128).then(|| raw >> count)),
                (
                    NumericOp::ShrS,
                    (count < 128).then(|| ((raw as i128) >> count) as u128),
                ),
            ] {
                let actual = op
                    .evaluate_binary(&value, &amount, PointerSize::Bit64)
                    .and_then(|value| value.scalar_integer_pattern(PointerSize::Bit64))
                    .map(|p| p.0);
                assert_eq!(actual, expected, "{op:?} {raw} {count}");
            }
        }
        let one = some_or_abort(ConstValue::of_lane(128, 1));
        for (op, expected) in [
            (NumericOp::Add, raw.wrapping_add(1)),
            (NumericOp::Sub, raw.wrapping_sub(1)),
            (NumericOp::Mul, raw),
            (NumericOp::Xor, raw ^ 1),
        ] {
            let actual = op
                .evaluate_binary(&value, &one, PointerSize::Bit64)
                .and_then(|value| value.scalar_integer_pattern(PointerSize::Bit64))
                .map(|p| p.0);
            assert_eq!(actual, Some(expected), "{op:?} {raw}");
        }
    }
}
