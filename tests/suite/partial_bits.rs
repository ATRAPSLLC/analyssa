//! Independent field-mask precision checks shared by integer analysis consumers.

use std::{collections::BTreeMap, num::NonZeroU16, process};

use analyssa::{
    analysis::{
        ConstEvaluator, SsaEvaluator, SsaVerifier, VerifyLevel,
        bits::Reader,
        context::AnalysisContext,
        evaluator::EvaluatorConfig,
        execution::{StaticExecution, Value},
    },
    ir::{
        ConstValue, SsaOp, SsaVarId,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{AccessPermission, AccessQualifiers, CallEffects},
        variable::{DefSite, VariableOrigin},
    },
    testing::{MACHINE, MockTarget, MockType},
};
use rustc_hash::FxHashMap;

/// Builds a field whose unrelated register bits are entirely unknown.
fn field_fixture(width: u32) -> (SsaFunction<MockTarget>, SsaVarId) {
    field_fixture_with_mask(width, 15)
}

/// Queries a configurable field while fixing only the original low four bits.
fn field_fixture_with_mask(width: u32, query_mask: u128) -> (SsaFunction<MockTarget>, SsaVarId) {
    let ty = match width {
        8 => MockType::U8,
        16 => MockType::U16,
        32 => MockType::U32,
        64 => MockType::U64,
        _ => MockType::U128,
    };
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let unknown = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        ty,
    );
    let mask = u128::MAX
        .checked_shr(128u32.saturating_sub(width))
        .unwrap_or(0);
    let constant = |bits| ConstValue::of_lane(width, bits).unwrap_or_else(|| process::abort());
    let result = builder
        .in_block(0, |block| {
            let upper_mask = block.const_value(SsaDefSpec::tmp(ty), constant(mask & !15))?;
            let lower = block.const_value(SsaDefSpec::tmp(ty), constant(9))?;
            let field_mask = block.const_value(SsaDefSpec::tmp(ty), constant(query_mask))?;
            let upper = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::And {
                dest,
                left: unknown,
                right: upper_mask,
                flags: None,
            })?;
            let combined = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::Or {
                dest,
                left: upper,
                right: lower,
                flags: None,
            })?;
            let field = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::And {
                dest,
                left: combined,
                right: field_mask,
                flags: None,
            })?;
            block.ret(Some(field))?;
            Ok(field)
        })
        .unwrap_or_else(|_| process::abort());
    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    (ssa, result)
}

/// The independent bit analysis proves a low field without assigning its unknown upper bits.
#[test]
fn partial_bits_analysis_tracks_the_field() {
    for width in [8, 16, 32, 64] {
        let (ssa, result) = field_fixture(width);
        let constants = ssa
            .iter_instructions()
            .filter_map(|(_, _, instruction)| match instruction.op() {
                SsaOp::Const { dest, value } => Some((*dest, value.clone())),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            Reader::new(&ssa, &constants)
                .known_bits(result)
                .and_then(|bits| bits.value()),
            Some(9),
            "width={width}"
        );
    }
}

/// Execution retains known field bits through a masked merge at every scalar width.
#[test]
fn partial_bits_execution_tracks_the_field() {
    for width in [8, 16, 32, 64, 128] {
        let (ssa, _) = field_fixture(width);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert_eq!(
            facts
                .returned()
                .and_then(Value::constant)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits),
            Some(9),
            "width={width}"
        );
    }
}

/// Recursive constant evaluation shares the field proof with straight-line execution.
#[test]
fn partial_bits_constant_evaluator_tracks_the_field() {
    for width in [8, 16, 32, 64, 128] {
        let (ssa, result) = field_fixture(width);
        assert_eq!(
            ConstEvaluator::new(&ssa, MACHINE)
                .evaluate_var(result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits),
            Some(9),
            "width={width}"
        );
    }
}

/// Concrete symbolic results retain the same field certainty.
#[test]
fn partial_bits_symbolic_evaluator_tracks_the_field() {
    for width in [8, 16, 32, 64, 128] {
        let (ssa, result) = field_fixture(width);
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.evaluate_block(0);
        assert_eq!(
            evaluator
                .get_concrete(result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits),
            Some(9),
            "width={width}"
        );
    }
}

/// Querying one additional runtime bit must not manufacture a complete field value.
#[test]
fn partial_bits_consumers_keep_unfixed_bits_unknown() {
    for width in [8, 16, 32, 64, 128] {
        let (ssa, result) = field_fixture_with_mask(width, 31);
        assert!(
            ConstEvaluator::new(&ssa, MACHINE)
                .evaluate_var(result)
                .is_none()
        );
        let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
        symbolic.evaluate_block(0);
        assert!(symbolic.get_concrete(result).is_none());
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert!(facts.returned().and_then(Value::constant).is_none());
    }
}

/// Reassigning and clearing values invalidates bits; rollback restores both domains.
#[test]
fn partial_bits_symbolic_state_rolls_back_and_invalidates() {
    let (ssa, result) = field_fixture(64);
    let op = ssa
        .get_definition(result)
        .unwrap_or_else(|| process::abort());
    let combined = match op {
        SsaOp::And { left, .. } => *left,
        _ => process::abort(),
    };
    let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
    evaluator.evaluate_block(0);
    let mark = evaluator.checkpoint();
    evaluator.set_unknown(combined);
    evaluator.evaluate_op(op);
    assert!(evaluator.get_concrete(result).is_none());
    evaluator.rollback(mark);
    evaluator.evaluate_op(op);
    assert_eq!(
        evaluator.get_concrete(result).and_then(ConstValue::as_u64),
        Some(9)
    );
    let mark = evaluator.checkpoint();
    evaluator.clear();
    evaluator.evaluate_op(op);
    assert!(evaluator.get_concrete(result).is_none());
    evaluator.rollback(mark);
    evaluator.evaluate_op(op);
    assert_eq!(
        evaluator.get_concrete(result).and_then(ConstValue::as_u64),
        Some(9)
    );
}

/// Builds a masked field after one partial-bit transfer with literal expected results.
fn transfer_fixture(width: u32, kind: u8) -> (SsaFunction<MockTarget>, SsaVarId, Option<u128>) {
    let ty = match width {
        8 => MockType::U8,
        16 => MockType::U16,
        32 => MockType::U32,
        64 => MockType::U64,
        _ => MockType::U128,
    };
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let unknown = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        ty,
    );
    let condition = builder.function_mut().create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::Bool,
    );
    let mask = u128::MAX
        .checked_shr(128u32.saturating_sub(width))
        .unwrap_or(0);
    let constant = |bits| ConstValue::of_lane(width, bits).unwrap_or_else(|| process::abort());
    let output_ty = if kind == 6 {
        MockType::U8
    } else if kind == 7 {
        MockType::I8
    } else {
        ty
    };
    let expected = match kind {
        1 => Some(6),
        2 => Some(10),
        3 => Some(2),
        4 | 5 => Some(4),
        9 => None,
        _ => Some(9),
    };
    let result = builder
        .in_block(0, |block| {
            let upper_mask = block.const_value(SsaDefSpec::tmp(ty), constant(mask & !15))?;
            let low = block.const_value(SsaDefSpec::tmp(ty), constant(9))?;
            let upper = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::And {
                dest,
                left: unknown,
                right: upper_mask,
                flags: None,
            })?;
            let combined = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::Or {
                dest,
                left: upper,
                right: low,
                flags: None,
            })?;
            let amount = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
            let other = block.const_value(
                SsaDefSpec::tmp(ty),
                constant(if kind == 8 { 64 } else { 3 }),
            )?;
            let alternate = block.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::Or {
                dest,
                left: combined,
                right: other,
                flags: None,
            })?;
            let value = block.emit_def(SsaDefSpec::tmp(output_ty), |dest| match kind {
                1 => SsaOp::Not {
                    dest,
                    operand: combined,
                    flags: None,
                },
                2 => SsaOp::Xor {
                    dest,
                    left: combined,
                    right: other,
                    flags: None,
                },
                3 => SsaOp::Shl {
                    dest,
                    value: combined,
                    amount,
                    flags: None,
                },
                4 | 5 => SsaOp::Shr {
                    dest,
                    value: combined,
                    amount,
                    unsigned: kind == 4,
                    flags: None,
                },
                6 | 7 => SsaOp::IntConv {
                    dest,
                    operand: combined,
                    target: output_ty,
                    overflow_check: false,
                    unsigned: kind == 6,
                },
                8 => SsaOp::Select {
                    dest,
                    condition,
                    true_val: combined,
                    false_val: alternate,
                },
                9 => SsaOp::Shl {
                    dest,
                    value: combined,
                    amount: unknown,
                    flags: None,
                },
                _ => SsaOp::Copy {
                    dest,
                    src: combined,
                },
            })?;
            let field_mask = if matches!(kind, 4 | 5) { 7 } else { 15 };
            let field_mask = block.const_value(
                SsaDefSpec::tmp(output_ty),
                ConstValue::of_lane(if matches!(kind, 6 | 7) { 8 } else { width }, field_mask)
                    .unwrap_or_else(|| process::abort()),
            )?;
            let result = block.emit_def(SsaDefSpec::tmp(output_ty), |dest| SsaOp::And {
                dest,
                left: value,
                right: field_mask,
                flags: None,
            })?;
            block.ret(Some(result))?;
            Ok(result)
        })
        .unwrap_or_else(|_| process::abort());
    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    (ssa, result, expected)
}

/// Bitwise, shift, conversion, copy and select transfers agree across all three consumers.
#[test]
fn partial_bits_transfer_family_shared_consumers() {
    for width in [8, 16, 32, 64, 128] {
        for kind in 0..10 {
            let (ssa, result, expected) = transfer_fixture(width, kind);
            let pattern = |value: &ConstValue<MockTarget>| {
                value
                    .scalar_integer_pattern(MACHINE.ptr_size)
                    .map(|(bits, _)| bits)
            };
            let constant = ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result);
            assert_eq!(
                constant.as_ref().and_then(pattern),
                expected,
                "constant width={width} kind={kind}"
            );
            let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
            symbolic.evaluate_block(0);
            assert_eq!(
                symbolic.get_concrete(result).and_then(pattern),
                expected,
                "symbolic width={width} kind={kind}"
            );
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&ssa, &context).run();
            assert!(facts.is_complete());
            assert_eq!(
                facts.returned().and_then(Value::constant).and_then(pattern),
                expected,
                "execution width={width} kind={kind}"
            );
        }
    }
}

/// Invalidating an assigned argument or local must also invalidate its mirrored slot.
#[test]
fn partial_bits_symbolic_slot_invalidation() {
    for local in [false, true] {
        let mut ssa = SsaFunction::<MockTarget>::new(1, 1);
        let origin = if local {
            VariableOrigin::Local(0)
        } else {
            VariableOrigin::Argument(0)
        };
        let source = ssa.create_variable(origin, 0, DefSite::entry(), MockType::U64);
        let dest = ssa.create_variable(VariableOrigin::Phi, 1, DefSite::entry(), MockType::U64);
        let op = if local {
            SsaOp::LoadLocal {
                dest,
                local_index: 0,
            }
        } else {
            SsaOp::LoadArg { dest, arg_index: 0 }
        };
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.set_concrete(source, ConstValue::U64(7));
        evaluator.evaluate_op(&op);
        assert_eq!(
            evaluator.get_concrete(dest).and_then(ConstValue::as_u64),
            Some(7)
        );
        let mark = evaluator.checkpoint();
        evaluator.set_unknown(source);
        evaluator.evaluate_op(&op);
        assert!(evaluator.get_concrete(dest).is_none(), "local={local}");
        evaluator.rollback(mark);
        evaluator.evaluate_op(&op);
        assert_eq!(
            evaluator.get_concrete(dest).and_then(ConstValue::as_u64),
            Some(7)
        );
    }
}

/// Replacing an injected input invalidates complete dependent values at every width.
#[test]
fn constant_evaluator_updates_injected_context() {
    for width in [8, 16, 32, 64, 128] {
        let (ssa, result) = field_fixture_with_mask(width, 31);
        let input = SsaVarId::from_index(0);
        assert_eq!(
            ssa.variable(input).map(|var| var.origin()),
            Some(VariableOrigin::Argument(0))
        );
        let mut evaluator = ConstEvaluator::new(&ssa, MACHINE);
        evaluator.set_known(
            input,
            ConstValue::of_lane(width, 16).unwrap_or_else(|| process::abort()),
        );
        assert_eq!(
            evaluator
                .evaluate_var(result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits),
            Some(25)
        );
        evaluator.set_known(
            input,
            ConstValue::of_lane(width, 0).unwrap_or_else(|| process::abort()),
        );
        assert_eq!(
            evaluator
                .evaluate_var(result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits),
            Some(9),
            "width={width}"
        );
    }
}

/// A previously unknown field is recomputed after its missing input becomes known.
#[test]
fn constant_evaluator_injected_context_revisits_unknowns() {
    for width in [8, 16, 32, 64, 128] {
        let (ssa, result) = field_fixture_with_mask(width, 31);
        let mut evaluator = ConstEvaluator::new(&ssa, MACHINE);
        assert!(evaluator.evaluate_var(result).is_none());
        evaluator.set_known(
            SsaVarId::from_index(0),
            ConstValue::of_lane(width, 0).unwrap_or_else(|| process::abort()),
        );
        assert_eq!(
            evaluator
                .evaluate_var(result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits),
            Some(9),
            "width={width}"
        );
    }
}

/// Builds two independent external bindings and one derived scalar sum.
fn binding_fixture() -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId, SsaVarId) {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let left = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U32,
    );
    let right = builder.function_mut().create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::U32,
    );
    let result = builder
        .in_block(0, |block| {
            let result = block.emit_def(SsaDefSpec::tmp(MockType::U32), |dest| SsaOp::Add {
                dest,
                left,
                right,
                flags: None,
            })?;
            block.ret(Some(result))?;
            Ok(result)
        })
        .unwrap_or_else(|_| process::abort());
    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    (ssa, left, right, result)
}

/// Updating one assumption preserves other supplied bindings; a full reset drops all of them.
#[test]
fn constant_evaluator_injected_context_preserves_other_bindings() {
    let (ssa, left, right, result) = binding_fixture();
    let mut evaluator = ConstEvaluator::new(&ssa, MACHINE);
    evaluator.set_known(left, ConstValue::U32(3));
    evaluator.set_known(right, ConstValue::U32(4));
    assert_eq!(
        evaluator
            .evaluate_var(result)
            .and_then(|value| value.as_u64()),
        Some(7)
    );
    evaluator.set_known(left, ConstValue::U32(10));
    assert_eq!(
        evaluator
            .evaluate_var(result)
            .and_then(|value| value.as_u64()),
        Some(14)
    );
    evaluator.set_known(right, ConstValue::U32(20));
    assert_eq!(
        evaluator
            .evaluate_var(result)
            .and_then(|value| value.as_u64()),
        Some(30)
    );
    let facts = evaluator.into_results();
    assert_eq!(facts.get(&left).and_then(ConstValue::as_u64), Some(10));
    assert_eq!(facts.get(&right).and_then(ConstValue::as_u64), Some(20));
    assert_eq!(facts.get(&result).and_then(ConstValue::as_u64), Some(30));
    let mut evaluator = ConstEvaluator::new(&ssa, MACHINE);
    evaluator.set_known(left, ConstValue::U32(3));
    evaluator.set_known(right, ConstValue::U32(4));
    assert!(evaluator.evaluate_var(result).is_some());
    evaluator.clear_cache();
    assert!(evaluator.evaluate_var(result).is_none());
}

/// A value boundary that must retain facts without assuming its unknown bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartialTransport {
    /// The incoming operand selected by a recorded predecessor.
    Phi,
    /// The current contents of a CIL local slot.
    Local,
    /// The current contents of a CIL argument slot.
    Argument,
}

/// Valid SSA and the definitions needed to observe a transported low field.
struct PartialTransportFixture {
    /// The complete verified function.
    ssa: SsaFunction<MockTarget>,
    /// The original partial value before its transport.
    partial: SsaVarId,
    /// The phi or local/argument assignment receiving the partial value.
    assigned: SsaVarId,
    /// The transported value immediately before the final mask.
    loaded: SsaVarId,
    /// The low-field extraction whose independent oracle is nine.
    result: SsaVarId,
    /// The block containing the extraction.
    final_block: usize,
}

/// Builds an unknown upper register and a fixed low nibble across one value boundary.
fn partial_transport_fixture(
    width: u32,
    route: PartialTransport,
    taken: bool,
    same_origin: bool,
) -> PartialTransportFixture {
    let ty = match width {
        8 => MockType::U8,
        16 => MockType::U16,
        32 => MockType::U32,
        64 => MockType::U64,
        _ => MockType::U128,
    };
    let arguments = if route == PartialTransport::Argument {
        2
    } else {
        1
    };
    let locals = usize::from(route == PartialTransport::Local);
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(arguments, locals);
    let final_block = usize::from(route == PartialTransport::Phi);
    builder.ensure_block(final_block);
    let unknown = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        ty,
    );
    let mask = u128::MAX
        .checked_shr(128u32.saturating_sub(width))
        .unwrap_or(0);
    let constant = |value| ConstValue::of_lane(width, value).unwrap_or_else(|| process::abort());
    let (partial, assigned, loaded, initial_result, field_mask) = builder
        .in_block(0, |block| {
            let upper_mask = block.const_value(SsaDefSpec::tmp(ty), constant(mask & !15))?;
            let lower = block.const_value(SsaDefSpec::tmp(ty), constant(9))?;
            let field_mask = block.const_value(SsaDefSpec::tmp(ty), constant(15))?;
            let upper = block.and(SsaDefSpec::tmp(ty), unknown, upper_mask)?;
            let partial = block.or(SsaDefSpec::tmp(ty), upper, lower)?;
            if route == PartialTransport::Phi {
                block.jump(1)?;
                return Ok((partial, partial, partial, partial, field_mask));
            }
            let assignment = if route == PartialTransport::Local {
                SsaDefSpec::local(0, ty)
            } else {
                SsaDefSpec::argument(1, ty)
            };
            let assigned = block.copy(assignment.clone(), partial)?;
            if taken {
                if route == PartialTransport::Local {
                    let _ = block.load_local_addr(SsaDefSpec::tmp(MockType::Ptr), 0)?;
                } else {
                    let _ = block.load_arg_addr(SsaDefSpec::tmp(MockType::Ptr), 1)?;
                }
            }
            let definition = if same_origin {
                assignment
            } else {
                SsaDefSpec::tmp(ty)
            };
            let loaded = if route == PartialTransport::Local {
                block.load_local(definition, 0)?
            } else {
                block.load_arg(definition, 1)?
            };
            let result = block.and(SsaDefSpec::tmp(ty), loaded, field_mask)?;
            block.ret(Some(result))?;
            Ok((partial, assigned, loaded, result, field_mask))
        })
        .unwrap_or_else(|_| process::abort());
    let (assigned, loaded, result) = if route == PartialTransport::Phi {
        builder
            .in_block(1, |block| {
                let joined = block.phi(SsaDefSpec::tmp(ty), [(0, partial)])?;
                let result = block.and(SsaDefSpec::tmp(ty), joined, field_mask)?;
                block.ret(Some(result))?;
                Ok((joined, joined, result))
            })
            .unwrap_or_else(|_| process::abort())
    } else {
        (assigned, loaded, initial_result)
    };
    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.is_empty(),
        "width={width}, route={route:?}: {errors:?}"
    );
    PartialTransportFixture {
        ssa,
        partial,
        assigned,
        loaded,
        result,
        final_block,
    }
}

/// Selected phis preserve low-field facts at every scalar width and reject absent edges.
#[test]
fn partial_bits_selected_phis_preserve_the_field() {
    let mut failures = Vec::new();
    for width in [8, 16, 32, 64, 128] {
        let fixture = partial_transport_fixture(width, PartialTransport::Phi, false, false);
        let mut evaluator = SsaEvaluator::new(&fixture.ssa, MACHINE);
        evaluator.evaluate_block(0);
        evaluator.set_predecessor(Some(0));
        evaluator.evaluate_block(fixture.final_block);
        let actual = evaluator
            .get_concrete(fixture.result)
            .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
            .map(|(bits, _)| bits);
        if actual != Some(9) {
            failures.push((width, actual));
        }
        for predecessor in [None, Some(99)] {
            evaluator.set_predecessor(predecessor);
            evaluator.evaluate_block(fixture.final_block);
            assert!(
                evaluator.get_concrete(fixture.result).is_none(),
                "width={width}, predecessor={predecessor:?}"
            );
        }
    }
    assert!(failures.is_empty(), "lost selected phi facts: {failures:?}");
}

/// Local slot reads retain the low field, including when the load defines the same origin.
#[test]
fn partial_bits_local_mirrors_preserve_the_field() {
    let mut failures = Vec::new();
    for width in [8, 16, 32, 64, 128] {
        for same_origin in [false, true] {
            let fixture =
                partial_transport_fixture(width, PartialTransport::Local, false, same_origin);
            let mut evaluator = SsaEvaluator::new(&fixture.ssa, MACHINE);
            evaluator.evaluate_block(0);
            let actual = evaluator
                .get_concrete(fixture.result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits);
            if actual != Some(9) {
                failures.push((width, same_origin, actual));
            }
        }
    }
    assert!(failures.is_empty(), "lost local facts: {failures:?}");
}

/// Argument slot reads retain the low field without rereading their original entry value.
#[test]
fn partial_bits_argument_mirrors_preserve_the_field() {
    let mut failures = Vec::new();
    for width in [8, 16, 32, 64, 128] {
        for same_origin in [false, true] {
            let fixture =
                partial_transport_fixture(width, PartialTransport::Argument, false, same_origin);
            let mut evaluator = SsaEvaluator::new(&fixture.ssa, MACHINE);
            evaluator.evaluate_block(0);
            let actual = evaluator
                .get_concrete(fixture.result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits);
            if actual != Some(9) {
                failures.push((width, same_origin, actual));
            }
        }
    }
    assert!(failures.is_empty(), "lost argument facts: {failures:?}");
}

/// Address-taken slots establish no constant field at any width or origin spelling.
#[test]
fn partial_bits_address_taken_mirrors_stay_unknown() {
    for width in [8, 16, 32, 64, 128] {
        for route in [PartialTransport::Local, PartialTransport::Argument] {
            for same_origin in [false, true] {
                let fixture = partial_transport_fixture(width, route, true, same_origin);
                let mut evaluator = SsaEvaluator::new(&fixture.ssa, MACHINE);
                evaluator.evaluate_block(0);
                assert!(
                    evaluator.get_concrete(fixture.result).is_none(),
                    "width={width}, route={route:?}"
                );
            }
        }
    }
}

/// Slot facts are stored values, survive value-only reset, and participate in speculative undo.
#[test]
fn partial_bits_mirrors_snapshot_clear_and_rollback() {
    for width in [8, 16, 32, 64, 128] {
        for route in [PartialTransport::Local, PartialTransport::Argument] {
            for same_origin in [false, true] {
                let fixture = partial_transport_fixture(width, route, false, same_origin);
                let load = fixture
                    .ssa
                    .get_definition(fixture.loaded)
                    .unwrap_or_else(|| process::abort());
                let query = fixture
                    .ssa
                    .get_definition(fixture.result)
                    .unwrap_or_else(|| process::abort());
                let mask_var = match query {
                    SsaOp::And { right, .. } => *right,
                    _ => process::abort(),
                };
                let mask = fixture
                    .ssa
                    .get_definition(mask_var)
                    .unwrap_or_else(|| process::abort());
                let field = |evaluator: &SsaEvaluator<'_, MockTarget>| {
                    evaluator
                        .get_concrete(fixture.result)
                        .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                        .map(|(bits, _)| bits)
                };
                let mut evaluator = SsaEvaluator::new(&fixture.ssa, MACHINE);
                evaluator.evaluate_block(0);
                // Reassigning the original temporary cannot change a copy already held in a slot.
                evaluator.set_unknown(fixture.partial);
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert_eq!(
                    field(&evaluator),
                    Some(9),
                    "snapshot width={width}, route={route:?}"
                );
                let mark = evaluator.checkpoint();
                evaluator.set_unknown(fixture.assigned);
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert!(
                    field(&evaluator).is_none(),
                    "invalidation width={width}, route={route:?}"
                );
                evaluator.rollback(mark);
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert_eq!(
                    field(&evaluator),
                    Some(9),
                    "rollback width={width}, route={route:?}"
                );
                evaluator.clear();
                evaluator.evaluate_op(mask);
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert_eq!(
                    field(&evaluator),
                    Some(9),
                    "value reset width={width}, route={route:?}"
                );
            }
        }
    }
}

/// Partial and concrete writes replace complete slot facts, and rollback restores the old value.
#[test]
fn partial_bits_mirrors_follow_partial_and_concrete_overwrites() {
    for width in [8, 16, 32, 64, 128] {
        for route in [PartialTransport::Local, PartialTransport::Argument] {
            for same_origin in [false, true] {
                let fixture = partial_transport_fixture(width, route, false, same_origin);
                let partial = fixture
                    .ssa
                    .get_definition(fixture.partial)
                    .unwrap_or_else(|| process::abort());
                let assignment = fixture
                    .ssa
                    .get_definition(fixture.assigned)
                    .unwrap_or_else(|| process::abort());
                let load = fixture
                    .ssa
                    .get_definition(fixture.loaded)
                    .unwrap_or_else(|| process::abort());
                let query = fixture
                    .ssa
                    .get_definition(fixture.result)
                    .unwrap_or_else(|| process::abort());
                let lower = match partial {
                    SsaOp::Or { right, .. } => *right,
                    _ => process::abort(),
                };
                let constant =
                    |value| ConstValue::of_lane(width, value).unwrap_or_else(|| process::abort());
                let field = |evaluator: &SsaEvaluator<'_, MockTarget>| {
                    evaluator
                        .get_concrete(fixture.result)
                        .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                        .map(|(bits, _)| bits)
                };
                let mut evaluator = SsaEvaluator::new(&fixture.ssa, MACHINE);
                evaluator.evaluate_block(0);
                assert_eq!(field(&evaluator), Some(9));
                let mark = evaluator.checkpoint();
                evaluator.set_concrete(lower, constant(6));
                evaluator.evaluate_op(partial);
                evaluator.evaluate_op(assignment);
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert_eq!(
                    field(&evaluator),
                    Some(6),
                    "partial overwrite width={width}, route={route:?}"
                );
                evaluator.rollback(mark);
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert_eq!(
                    field(&evaluator),
                    Some(9),
                    "partial undo width={width}, route={route:?}"
                );
                evaluator.set_concrete(fixture.assigned, constant(3));
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert_eq!(
                    field(&evaluator),
                    Some(3),
                    "concrete overwrite width={width}, route={route:?}"
                );
                evaluator.set_symbolic(fixture.assigned, "replacement");
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert!(
                    field(&evaluator).is_none(),
                    "symbolic overwrite width={width}, route={route:?}"
                );
            }
        }
    }
}

/// Loop phis read both partial operands before either assignment can overwrite a source.
#[test]
fn partial_bits_phis_swap_partial_values_simultaneously() {
    for width in [8, 16, 32, 64, 128] {
        let ty = match width {
            8 => MockType::U8,
            16 => MockType::U16,
            32 => MockType::U32,
            64 => MockType::U64,
            _ => MockType::U128,
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        builder.ensure_block(1);
        let unknown = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            ty,
        );
        let constant = |bits| ConstValue::of_lane(width, bits).unwrap_or_else(|| process::abort());
        let mask = u128::MAX
            .checked_shr(128u32.saturating_sub(width))
            .unwrap_or(0);
        let (nine, six, mask) = builder
            .in_block(0, |block| {
                let upper_mask = block.const_value(SsaDefSpec::tmp(ty), constant(mask & !15))?;
                let nine = block.const_value(SsaDefSpec::tmp(ty), constant(9))?;
                let six = block.const_value(SsaDefSpec::tmp(ty), constant(6))?;
                let mask = block.const_value(SsaDefSpec::tmp(ty), constant(15))?;
                let upper = block.and(SsaDefSpec::tmp(ty), unknown, upper_mask)?;
                let nine = block.or(SsaDefSpec::tmp(ty), upper, nine)?;
                let six = block.or(SsaDefSpec::tmp(ty), upper, six)?;
                block.jump(1)?;
                Ok((nine, six, mask))
            })
            .unwrap_or_else(|_| process::abort());
        let (left, right, left_field, right_field) = builder
            .in_block(1, |block| {
                let left = block.phi(SsaDefSpec::tmp(ty), [(0, nine)])?;
                let right = block.phi(SsaDefSpec::tmp(ty), [(0, six)])?;
                let left_field = block.and(SsaDefSpec::tmp(ty), left, mask)?;
                let right_field = block.and(SsaDefSpec::tmp(ty), right, mask)?;
                block.jump(1)?;
                Ok((left, right, left_field, right_field))
            })
            .unwrap_or_else(|_| process::abort());
        builder
            .add_phi_operand(1, left, 1, right)
            .unwrap_or_else(|_| process::abort());
        builder
            .add_phi_operand(1, right, 1, left)
            .unwrap_or_else(|_| process::abort());
        let ssa = builder.finish().unwrap_or_else(|_| process::abort());
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(errors.is_empty(), "width={width}: {errors:?}");
        let fields = |evaluator: &SsaEvaluator<'_, MockTarget>| {
            [left_field, right_field].map(|var| {
                evaluator
                    .get_concrete(var)
                    .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                    .map(|(bits, _)| bits)
            })
        };
        let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
        evaluator.evaluate_block(0);
        evaluator.set_predecessor(Some(0));
        evaluator.evaluate_block(1);
        assert_eq!(fields(&evaluator), [Some(9), Some(6)]);
        let mark = evaluator.checkpoint();
        evaluator.set_predecessor(Some(1));
        evaluator.evaluate_block(1);
        assert_eq!(fields(&evaluator), [Some(6), Some(9)]);
        evaluator.rollback(mark);
        assert_eq!(fields(&evaluator), [Some(9), Some(6)]);
        evaluator.set_predecessor(Some(1));
        evaluator.evaluate_block(1);
        evaluator.evaluate_block(1);
        assert_eq!(fields(&evaluator), [Some(9), Some(6)]);
    }
}

/// Every slot and field load replaces its destination mirror and rolls it back as one assignment.
#[test]
fn evaluator_loads_update_destination_slot_mirrors() {
    for width in [8, 16, 32, 64, 128] {
        let ty = match width {
            8 => MockType::U8,
            16 => MockType::U16,
            32 => MockType::U32,
            64 => MockType::U64,
            _ => MockType::U128,
        };
        let constant = |bits| ConstValue::of_lane(width, bits).unwrap_or_else(|| process::abort());
        for source in 0..4 {
            for argument in [false, true] {
                let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 2);
                let object = builder.function_mut().create_variable(
                    VariableOrigin::Argument(0),
                    0,
                    DefSite::entry(),
                    MockType::Ref,
                );
                let target = if argument {
                    SsaDefSpec::argument(2, ty)
                } else {
                    SsaDefSpec::local(1, ty)
                };
                let (seed, loaded, result) = builder
                    .in_block(0, |block| {
                        let seed = block.const_value(target.clone(), constant(3))?;
                        let value = block.const_value(SsaDefSpec::tmp(ty), constant(9))?;
                        let loaded = match source {
                            0 => {
                                let _ = block.copy(SsaDefSpec::local(0, ty), value)?;
                                block.load_local(target.clone(), 0)?
                            }
                            1 => {
                                let _ = block.copy(SsaDefSpec::argument(1, ty), value)?;
                                block.load_arg(target.clone(), 1)?
                            }
                            2 => {
                                block.store_field(object, 1, value)?;
                                block.load_field(target.clone(), object, 1)?
                            }
                            _ => {
                                block.store_static_field(1, value)?;
                                block.load_static_field(target.clone(), 1)?
                            }
                        };
                        let result = if argument {
                            block.load_arg(SsaDefSpec::tmp(ty), 2)?
                        } else {
                            block.load_local(SsaDefSpec::tmp(ty), 1)?
                        };
                        block.ret(Some(result))?;
                        Ok((seed, loaded, result))
                    })
                    .unwrap_or_else(|_| process::abort());
                let ssa = builder.finish().unwrap_or_else(|_| process::abort());
                let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
                assert!(
                    errors.is_empty(),
                    "width={width}, source={source}: {errors:?}"
                );
                let seed = ssa.get_definition(seed).unwrap_or_else(|| process::abort());
                let load = ssa
                    .get_definition(loaded)
                    .unwrap_or_else(|| process::abort());
                let query = ssa
                    .get_definition(result)
                    .unwrap_or_else(|| process::abort());
                let field = |evaluator: &SsaEvaluator<'_, MockTarget>| {
                    evaluator
                        .get_concrete(result)
                        .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                        .map(|(bits, _)| bits)
                };
                let mut evaluator =
                    SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
                evaluator.evaluate_block(0);
                assert_eq!(
                    field(&evaluator),
                    Some(9),
                    "width={width}, source={source}, argument={argument}"
                );
                evaluator.evaluate_op(seed);
                let mark = evaluator.checkpoint();
                evaluator.evaluate_op(load);
                evaluator.evaluate_op(query);
                assert_eq!(field(&evaluator), Some(9));
                evaluator.rollback(mark);
                evaluator.evaluate_op(query);
                assert_eq!(
                    field(&evaluator),
                    Some(3),
                    "mirror rollback source={source}"
                );
                // With no tracked field state, the load must also forget the old slot value.
                if source >= 2 {
                    let mut unknown = SsaEvaluator::new(&ssa, MACHINE);
                    unknown.evaluate_block(0);
                    assert!(field(&unknown).is_none(), "untracked field source={source}");
                }
            }
        }
    }
}

/// Builds repeated field accesses whose stored phi source changes on the next iteration.
fn field_loop_fixture(
    width: u32,
    static_field: bool,
    partial: bool,
    changing_object: bool,
) -> (SsaFunction<MockTarget>, SsaVarId) {
    let ty = match width {
        8 => MockType::U8,
        16 => MockType::U16,
        32 => MockType::U32,
        64 => MockType::U64,
        _ => MockType::U128,
    };
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
    builder.ensure_block(1);
    let entry = |builder: &mut SsaFunctionBuilder<MockTarget>, index, ty| {
        builder.function_mut().create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            ty,
        )
    };
    let unknown = entry(&mut builder, 0, ty);
    let object = entry(&mut builder, 1, MockType::Ref);
    let other = entry(&mut builder, 2, MockType::Ref);
    let constant = |bits| ConstValue::of_lane(width, bits).unwrap_or_else(|| process::abort());
    let (nine, six, mask) = builder
        .in_block(0, |block| {
            let nine = block.const_value(SsaDefSpec::tmp(ty), constant(9))?;
            let six = block.const_value(SsaDefSpec::tmp(ty), constant(6))?;
            let mask = block.const_value(SsaDefSpec::tmp(ty), constant(15))?;
            let (nine, six) = if partial {
                let all = u128::MAX
                    .checked_shr(128u32.saturating_sub(width))
                    .unwrap_or(0);
                let upper_mask = block.const_value(SsaDefSpec::tmp(ty), constant(all & !15))?;
                let upper = block.and(SsaDefSpec::tmp(ty), unknown, upper_mask)?;
                (
                    block.or(SsaDefSpec::tmp(ty), upper, nine)?,
                    block.or(SsaDefSpec::tmp(ty), upper, six)?,
                )
            } else {
                (nine, six)
            };
            block.jump(1)?;
            Ok((nine, six, mask))
        })
        .unwrap_or_else(|_| process::abort());
    let (source, current_object, result) = builder
        .in_block(1, |block| {
            let source = block.phi(SsaDefSpec::tmp(ty), [(0, nine)])?;
            let current_object = if changing_object {
                block.phi(SsaDefSpec::tmp(MockType::Ref), [(0, object)])?
            } else {
                object
            };
            let loaded = if static_field {
                block.load_static_field(SsaDefSpec::tmp(ty), 1)?
            } else {
                block.load_field(SsaDefSpec::tmp(ty), current_object, 1)?
            };
            let result = block.and(SsaDefSpec::tmp(ty), loaded, mask)?;
            if static_field {
                block.store_static_field(1, source)?;
            } else {
                block.store_field(current_object, 1, source)?;
            }
            block.jump(1)?;
            Ok((source, current_object, result))
        })
        .unwrap_or_else(|_| process::abort());
    builder
        .add_phi_operand(1, source, 1, six)
        .unwrap_or_else(|_| process::abort());
    if changing_object {
        builder
            .add_phi_operand(1, current_object, 1, other)
            .unwrap_or_else(|_| process::abort());
    }
    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "width={width}: {errors:?}");
    (ssa, result)
}

/// Stored constants retain their value after a loop phi selects the next source.
#[test]
fn field_stores_snapshot_concrete_phi_sources() {
    let mut failures = Vec::new();
    for width in [8, 16, 32, 64, 128] {
        for static_field in [false, true] {
            let (ssa, result) = field_loop_fixture(width, static_field, false, false);
            let mut evaluator =
                SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
            evaluator.evaluate_block(0);
            evaluator.set_predecessor(Some(0));
            evaluator.evaluate_block(1);
            assert!(evaluator.get_concrete(result).is_none());
            evaluator.set_predecessor(Some(1));
            evaluator.evaluate_block(1);
            let actual = evaluator
                .get_concrete(result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits);
            if actual != Some(9) {
                failures.push((width, static_field, actual));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "field store-time value lost: {failures:?}"
    );
}

/// Field stores preserve partial facts independently of the next phi value.
#[test]
fn field_stores_snapshot_partial_phi_sources() {
    let mut failures = Vec::new();
    for width in [8, 16, 32, 64, 128] {
        for static_field in [false, true] {
            let (ssa, result) = field_loop_fixture(width, static_field, true, false);
            let mut evaluator =
                SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
            evaluator.evaluate_block(0);
            evaluator.set_predecessor(Some(0));
            evaluator.evaluate_block(1);
            assert!(evaluator.get_concrete(result).is_none());
            evaluator.set_predecessor(Some(1));
            evaluator.evaluate_block(1);
            let actual = evaluator
                .get_concrete(result)
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
                .map(|(bits, _)| bits);
            if actual != Some(9) {
                failures.push((width, static_field, actual));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "partial field store-time value lost: {failures:?}"
    );
}

/// A different object on a loop edge cannot inherit the previous object's stored field.
#[test]
fn field_stores_do_not_alias_redefined_object_ids() {
    let mut failures = Vec::new();
    for width in [8, 16, 32, 64, 128] {
        let (ssa, result) = field_loop_fixture(width, false, false, true);
        let mut evaluator =
            SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
        evaluator.evaluate_block(0);
        evaluator.set_predecessor(Some(0));
        evaluator.evaluate_block(1);
        evaluator.set_predecessor(Some(1));
        evaluator.evaluate_block(1);
        if evaluator.get_concrete(result).is_some() {
            failures.push(width);
        }
    }
    assert!(
        failures.is_empty(),
        "redefined object gained a field fact: {failures:?}"
    );
}

/// Instructions needed to observe and overwrite one verified loop field.
struct FieldAccess<'a> {
    /// Load of the field's current contents.
    load: &'a SsaOp<MockTarget>,
    /// Store of the loop's phi value.
    store: &'a SsaOp<MockTarget>,
    /// Projection of the low four bits.
    query: &'a SsaOp<MockTarget>,
    /// Field-mask constant, for value-only resets.
    mask: &'a SsaOp<MockTarget>,
    /// Phi used as the store's source.
    source: SsaVarId,
    /// Entry object for instance fields.
    object: Option<SsaVarId>,
    /// Value carrying the observed low field.
    result: SsaVarId,
}

/// Finds accesses in the independently verified field-loop fixture.
fn field_access(ssa: &SsaFunction<MockTarget>, result: SsaVarId) -> FieldAccess<'_> {
    let query = ssa
        .get_definition(result)
        .unwrap_or_else(|| process::abort());
    let (loaded, mask) = match query {
        SsaOp::And { left, right, .. } => (*left, *right),
        _ => process::abort(),
    };
    let store = ssa
        .block(1)
        .unwrap_or_else(|| process::abort())
        .instructions()
        .iter()
        .map(|instruction| instruction.op())
        .find(|op| {
            matches!(
                op,
                SsaOp::StoreField { .. } | SsaOp::StoreStaticField { .. }
            )
        })
        .unwrap_or_else(|| process::abort());
    let (source, object) = match store {
        SsaOp::StoreField { value, object, .. } => (*value, Some(*object)),
        SsaOp::StoreStaticField { value, .. } => (*value, None),
        _ => process::abort(),
    };
    FieldAccess {
        load: ssa
            .get_definition(loaded)
            .unwrap_or_else(|| process::abort()),
        store,
        query,
        mask: ssa.get_definition(mask).unwrap_or_else(|| process::abort()),
        source,
        object,
        result,
    }
}

/// Reads the low field without replaying the store.
fn read_stored_field(
    evaluator: &mut SsaEvaluator<'_, MockTarget>,
    access: &FieldAccess<'_>,
) -> Option<u128> {
    evaluator.evaluate_op(access.load);
    evaluator.evaluate_op(access.query);
    evaluator
        .get_concrete(access.result)
        .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size))
        .map(|(bits, _)| bits)
}

/// Concrete and partial field values survive source invalidation, reset and speculative stores.
#[test]
fn field_snapshots_preserve_reset_and_rollback_contracts() {
    for width in [8, 16, 32, 64, 128] {
        for static_field in [false, true] {
            for partial in [false, true] {
                let (ssa, result) = field_loop_fixture(width, static_field, partial, false);
                let access = field_access(&ssa, result);
                let mut evaluator =
                    SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
                evaluator.evaluate_block(0);
                evaluator.set_predecessor(Some(0));
                evaluator.evaluate_block(1);
                assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
                evaluator.set_unknown(access.source);
                assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
                let mark = evaluator.checkpoint();
                evaluator.evaluate_op(access.store);
                assert!(read_stored_field(&mut evaluator, &access).is_none());
                evaluator.set_concrete(
                    access.source,
                    ConstValue::of_lane(width, 6).unwrap_or_else(|| process::abort()),
                );
                assert!(
                    read_stored_field(&mut evaluator, &access).is_none(),
                    "an unknown store learned a later source"
                );
                let later = ConstValue::of_lane(width, 6).unwrap_or_else(|| process::abort());
                let bindings = FxHashMap::from_iter([(access.source, later)]);
                let inferred = evaluator
                    .get(access.result)
                    .and_then(|expression| expression.evaluate(&bindings, MACHINE.ptr_size));
                assert!(
                    inferred.is_none(),
                    "stored expression learned its source's later value"
                );
                evaluator.rollback(mark);
                assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
                evaluator.clear();
                evaluator.evaluate_op(access.mask);
                assert_eq!(
                    read_stored_field(&mut evaluator, &access),
                    Some(9),
                    "reset width={width}, partial={partial}"
                );
            }
        }
    }
}

/// Rolling back a clobbered object restores both the field snapshot and its dependency index.
#[test]
fn field_snapshot_object_dependencies_roll_back() {
    for width in [8, 16, 32, 64, 128] {
        for partial in [false, true] {
            let (ssa, result) = field_loop_fixture(width, false, partial, false);
            let access = field_access(&ssa, result);
            let object = access.object.unwrap_or_else(|| process::abort());
            let mut evaluator =
                SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
            evaluator.evaluate_block(0);
            evaluator.set_predecessor(Some(0));
            evaluator.evaluate_block(1);
            let mark = evaluator.checkpoint();
            evaluator.set_unknown(object);
            assert!(read_stored_field(&mut evaluator, &access).is_none());
            evaluator.rollback(mark);
            assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
            evaluator.set_unknown(object);
            assert!(
                read_stored_field(&mut evaluator, &access).is_none(),
                "restored dependency missing"
            );
        }
    }
}

/// Pure calls preserve both domains, while an unanswered call clobbers them together.
#[test]
fn field_snapshot_call_clobbers_roll_back() {
    for width in [8, 16, 32, 64, 128] {
        for static_field in [false, true] {
            for partial in [false, true] {
                let (ssa, result) = field_loop_fixture(width, static_field, partial, false);
                let access = field_access(&ssa, result);
                let mut evaluator =
                    SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
                evaluator.evaluate_block(0);
                evaluator.set_predecessor(Some(0));
                evaluator.evaluate_block(1);
                let mark = evaluator.checkpoint();
                evaluator.evaluate_op(&SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: None,
                    callee: analyssa::ir::Callee::Direct(9),
                    args: Vec::new(),
                    effects: Some(CallEffects::pure()),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })));
                assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
                evaluator.evaluate_op(&SsaOp::Call(Box::new(analyssa::ir::CallData {
                    dest: None,
                    callee: analyssa::ir::Callee::Direct(9),
                    args: Vec::new(),
                    effects: None,
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: Vec::new(),
                })));
                assert!(read_stored_field(&mut evaluator, &access).is_none());
                evaluator.rollback(mark);
                assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
            }
        }
    }
}

/// Re-enabling field tracking cannot resurrect facts from an untracked interval.
#[test]
fn field_snapshot_tracking_changes_forget_old_state() {
    for width in [8, 16, 32, 64, 128] {
        for static_field in [false, true] {
            let (ssa, result) = field_loop_fixture(width, static_field, false, false);
            let access = field_access(&ssa, result);
            let mut evaluator =
                SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
            evaluator.evaluate_block(0);
            evaluator.set_predecessor(Some(0));
            evaluator.evaluate_block(1);
            assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
            evaluator = evaluator.with_config(EvaluatorConfig::new());
            evaluator.set_unknown(access.source);
            evaluator.evaluate_op(access.store);
            evaluator = evaluator.with_config(EvaluatorConfig::with_memory());
            assert!(
                read_stored_field(&mut evaluator, &access).is_none(),
                "resurrected field width={width}"
            );
        }
    }
}

/// Configuration changes and newly tracked stores belong to the speculative transaction.
#[test]
fn field_snapshot_rollback_restores_tracking_configuration() {
    let (ssa, result) = field_loop_fixture(32, true, false, false);
    let access = field_access(&ssa, result);
    let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
    let mark = evaluator.checkpoint();
    evaluator = evaluator.with_config(EvaluatorConfig::path_aware());
    evaluator.evaluate_block(0);
    evaluator.set_predecessor(Some(0));
    evaluator.evaluate_block(1);
    assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
    assert!(!evaluator.path().is_empty());
    evaluator.rollback(mark);
    assert!(evaluator.path().is_empty());
    assert!(!evaluator.config().track_memory);
    assert!(!evaluator.config().track_path);
    evaluator = evaluator.with_config(EvaluatorConfig::with_memory());
    evaluator.evaluate_op(access.mask);
    assert!(read_stored_field(&mut evaluator, &access).is_none());
}

/// A field store preserves an unknown input's identity while that input remains unchanged.
#[test]
fn field_snapshots_preserve_stored_symbolic_identity() {
    for static_field in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let object = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ref,
        );
        let result = builder
            .in_block(0, |block| {
                if static_field {
                    block.store_static_field(1, object)?;
                } else {
                    block.store_field(object, 1, object)?;
                }
                let loaded = if static_field {
                    block.load_static_field(SsaDefSpec::tmp(MockType::Ref), 1)?
                } else {
                    block.load_field(SsaDefSpec::tmp(MockType::Ref), object, 1)?
                };
                block.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(|_| process::abort());
        let ssa = builder.finish().unwrap_or_else(|_| process::abort());
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(errors.is_empty(), "{errors:?}");
        let mut evaluator =
            SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
        evaluator.evaluate_block(0);
        let expected = ConstValue::Symbol(0x1000);
        let bindings = FxHashMap::from_iter([(object, expected.clone())]);
        let actual = evaluator
            .get(result)
            .and_then(|expression| expression.evaluate(&bindings, MACHINE.ptr_size));
        assert_eq!(
            actual,
            Some(expected),
            "stored input identity static={static_field}"
        );
    }
}

/// Previously loaded opaque values must stay distinct from a later call result.
#[test]
fn field_values_loaded_before_a_redefinition_keep_their_instance() {
    let mut failures = Vec::new();
    for width in [8, 16, 32, 64, 128] {
        let ty = match width {
            8 => MockType::U8,
            16 => MockType::U16,
            32 => MockType::U32,
            64 => MockType::U64,
            _ => MockType::U128,
        };
        for static_field in [false, true] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
            builder.ensure_block(1);
            let initial = builder.function_mut().create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                ty,
            );
            let object = builder.function_mut().create_variable(
                VariableOrigin::Argument(1),
                0,
                DefSite::entry(),
                MockType::Ref,
            );
            builder
                .in_block(0, |block| {
                    if static_field {
                        block.store_static_field(1, initial)?;
                    } else {
                        block.store_field(object, 1, initial)?;
                    }
                    block.jump(1)?;
                    Ok(())
                })
                .unwrap_or_else(|_| process::abort());
            let difference = builder
                .in_block(1, |block| {
                    let previous = if static_field {
                        block.load_static_field(SsaDefSpec::tmp(ty), 1)?
                    } else {
                        block.load_field(SsaDefSpec::tmp(ty), object, 1)?
                    };
                    let mut effects = CallEffects::pure();
                    // A clock-like source can vary while preserving every memory cell.
                    effects.world = true;
                    let current = block.emit_def(SsaDefSpec::tmp(ty), |dest| {
                        SsaOp::Call(Box::new(analyssa::ir::CallData {
                            dest: Some(dest),
                            callee: analyssa::ir::Callee::Direct(9),
                            args: Vec::new(),
                            effects: Some(effects),
                            tail: false,
                            implicit_inputs: Default::default(),
                            results: Default::default(),
                            staged: Vec::new(),
                        }))
                    })?;
                    if static_field {
                        block.store_static_field(1, current)?;
                    } else {
                        block.store_field(object, 1, current)?;
                    }
                    let latest = if static_field {
                        block.load_static_field(SsaDefSpec::tmp(ty), 1)?
                    } else {
                        block.load_field(SsaDefSpec::tmp(ty), object, 1)?
                    };
                    let difference = block.sub(SsaDefSpec::tmp(ty), previous, latest)?;
                    block.jump(1)?;
                    Ok(difference)
                })
                .unwrap_or_else(|_| process::abort());
            let ssa = builder.finish().unwrap_or_else(|_| process::abort());
            let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
            assert!(errors.is_empty(), "width={width}: {errors:?}");
            let mut evaluator =
                SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
            evaluator.evaluate_block(0);
            evaluator.evaluate_block(1);
            assert!(evaluator.get_concrete(difference).is_none());
            evaluator.evaluate_block(1);
            if evaluator.get_concrete(difference).is_some() {
                failures.push((width, static_field));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "distinct call results gained a constant difference: {failures:?}"
    );
}

/// A shared opaque instance stays equal through fields, copies, phis and both slot kinds.
#[test]
fn opaque_instances_preserve_copies_phis_and_slots() {
    for static_field in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 1);
        builder.ensure_block(1);
        let object = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ref,
        );
        let loaded = builder
            .in_block(0, |block| {
                let current = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                    SsaOp::Call(Box::new(analyssa::ir::CallData {
                        dest: Some(dest),
                        callee: analyssa::ir::Callee::Direct(9),
                        args: Vec::new(),
                        effects: None,
                        tail: false,
                        implicit_inputs: Default::default(),
                        results: Default::default(),
                        staged: Vec::new(),
                    }))
                })?;
                if static_field {
                    block.store_static_field(1, current)?;
                } else {
                    block.store_field(object, 1, current)?;
                }
                let loaded = if static_field {
                    block.load_static_field(SsaDefSpec::tmp(MockType::I32), 1)?
                } else {
                    block.load_field(SsaDefSpec::tmp(MockType::I32), object, 1)?
                };
                let _ = block.copy(SsaDefSpec::local(0, MockType::I32), loaded)?;
                let _ = block.copy(SsaDefSpec::argument(1, MockType::I32), loaded)?;
                block.jump(1)?;
                Ok(loaded)
            })
            .unwrap_or_else(|_| process::abort());
        let result = builder
            .in_block(1, |block| {
                let joined = block.phi(SsaDefSpec::tmp(MockType::I32), [(0, loaded)])?;
                let local = block.load_local(SsaDefSpec::tmp(MockType::I32), 0)?;
                let argument = block.load_arg(SsaDefSpec::tmp(MockType::I32), 1)?;
                let a = block.sub(SsaDefSpec::tmp(MockType::I32), joined, local)?;
                let b = block.sub(SsaDefSpec::tmp(MockType::I32), joined, argument)?;
                let result = block.add(SsaDefSpec::tmp(MockType::I32), a, b)?;
                block.ret(Some(result))?;
                Ok(result)
            })
            .unwrap_or_else(|_| process::abort());
        let ssa = builder.finish().unwrap_or_else(|_| process::abort());
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(errors.is_empty(), "{errors:?}");
        let mut evaluator =
            SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
        evaluator.evaluate_block(0);
        evaluator.set_predecessor(Some(0));
        evaluator.evaluate_block(1);
        assert_eq!(
            evaluator.get_concrete(result).and_then(ConstValue::as_i64),
            Some(0)
        );
    }
}

/// Qualified accesses cannot leak either exact or partial field snapshots at any scalar width.
#[test]
fn qualified_field_accesses_forget_exact_and_partial_masks() {
    for width in [8, 16, 32, 64, 128] {
        for static_field in [false, true] {
            for partial in [false, true] {
                let (ssa, result) = field_loop_fixture(width, static_field, partial, false);
                let access = field_access(&ssa, result);
                let mut evaluator =
                    SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
                evaluator.evaluate_block(0);
                evaluator.set_predecessor(Some(0));
                evaluator.evaluate_block(1);
                assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
                for qualifier in [
                    AccessQualifiers::FLAT.made_volatile(),
                    AccessQualifiers::new(NonZeroU16::new(1), AccessPermission::Current),
                    AccessQualifiers::new(None, AccessPermission::Unprivileged),
                ] {
                    let mark = evaluator.checkpoint();
                    let mut load = access.load.clone();
                    load.visit_qualifiers_mut(|access| *access = qualifier);
                    evaluator.evaluate_op(&load);
                    evaluator.evaluate_op(access.query);
                    assert!(
                        evaluator.get_concrete(result).is_none(),
                        "load width={width}, static={static_field}, partial={partial}, qualifier={qualifier:?}"
                    );
                    evaluator.rollback(mark);
                    assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
                    let mark = evaluator.checkpoint();
                    let mut store = access.store.clone();
                    store.visit_qualifiers_mut(|access| *access = qualifier);
                    evaluator.evaluate_op(&store);
                    assert!(
                        read_stored_field(&mut evaluator, &access).is_none(),
                        "store width={width}, static={static_field}, partial={partial}, qualifier={qualifier:?}"
                    );
                    evaluator.rollback(mark);
                    assert_eq!(read_stored_field(&mut evaluator, &access), Some(9));
                }
            }
        }
    }
}
