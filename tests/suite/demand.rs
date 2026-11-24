//! Observation requests retain relevant unknown identities and ordered effects.

use std::{cell::Cell, collections::BTreeSet, process, rc::Rc};

use analyssa::{
    PointerSize,
    analysis::{
        address::AddressBase,
        bindingtime::{BindingOracle, BindingTime, BindingTimes, CallBinding},
        context::AnalysisContext,
        defuse::DefUseIndex,
        demand::{
            Dependencies, Dependency, DependencyRefusal, DependencyRole, DependencySignature,
            MemoryOrigin, Observation,
        },
        memory::{IndirectLocation, MemoryLocation, MemoryOp, MemorySsa},
    },
    ir::{
        Callee,
        exception::{BlockRange, SsaExceptionHandler},
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{AccessQualifiers, AtomicAccessWidth, AtomicOrdering, CallData, CallInputKey},
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MACHINE, MockTarget, MockType},
};

/// Terminates a malformed fixture without relaxing integration-test lint rules.
fn abort<E, T>(_: E) -> T {
    process::abort()
}

/// Creates an ordinary scalar definition.
fn tmp() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::I32)
}

/// Creates an incoming scalar or pointer at its actual argument index.
fn parameter(builder: &mut SsaFunctionBuilder<MockTarget>, index: u16, ty: MockType) -> SsaVarId {
    builder
        .function_mut()
        .create_variable(VariableOrigin::Argument(index), 0, DefSite::entry(), ty)
}

/// Completes an independently constructed fixture.
fn finish(builder: SsaFunctionBuilder<MockTarget>) -> SsaFunction<MockTarget> {
    builder.finish().unwrap_or_else(abort)
}

/// Projects incoming requirements without allowing analysis-local borrows to escape.
fn inputs(
    analysis: &mut Dependencies<'_, MockTarget>,
    value: SsaVarId,
    role: DependencyRole,
) -> Vec<(SsaVarId, DependencyRole)> {
    {
        analysis
            .request(Observation::Value(value, role))
            .unwrap_or_else(abort)
            .dependencies()
            .cloned()
            .filter_map(|dependency| match dependency {
                Dependency::Input { value, role } => Some((value, role)),
                _ => None,
            })
            .collect()
    }
}

/// Only the key used by the observation is required; unrelated arguments stay dynamic.
#[test]
fn demand_retains_relevant_parameters_and_roles() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let key = parameter(&mut builder, 0, MockType::I32);
    let unrelated = parameter(&mut builder, 1, MockType::I32);
    let result = builder
        .in_block(0, |block| {
            let fixed = block.const_i32(tmp(), 7)?;
            let result = block.add(tmp(), key, fixed)?;
            block.ret(Some(result))?;
            Ok(result)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    assert_eq!(
        inputs(&mut analysis, result, DependencyRole::Value),
        vec![(key, DependencyRole::Value)]
    );
    assert_eq!(
        inputs(&mut analysis, result, DependencyRole::Bound),
        vec![(key, DependencyRole::Bound)]
    );
    assert_eq!(
        inputs(&mut analysis, unrelated, DependencyRole::Value),
        vec![(unrelated, DependencyRole::Value)]
    );
}

/// A pointer's value and its entry contents are independent requirements.
#[test]
fn demand_distinguishes_addresses_from_pointee_contents() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let result = builder
        .in_block(0, |block| {
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    assert_eq!(
        inputs(&mut analysis, result, DependencyRole::Value),
        vec![(pointer, DependencyRole::Address)]
    );
    let memory = {
        analysis
            .request(Observation::Value(result, DependencyRole::Value))
            .unwrap_or_else(abort)
            .dependencies()
            .filter(|dependency| {
                matches!(
                    dependency,
                    Dependency::Memory {
                        origin: MemoryOrigin::Entry,
                        ..
                    }
                )
            })
            .count()
    };
    assert_eq!(memory, 1);
}

/// Exact forwarding keeps its address requirement without making its scalar dynamic.
#[test]
fn demand_observes_ordered_memory_without_changing_scalar_binding_time() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let loaded = builder
        .in_block(0, |block| {
            let fixed = block.const_i32(tmp(), 42)?;
            block.store_indirect(pointer, fixed, MockType::I32)?;
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    assert!(BindingTimes::analyze(&ssa, &context, &()).is_static(loaded));
    let memory = MemorySsa::build(&ssa, &context);
    let location = match memory.operation_at(0, 2) {
        Some(MemoryOp::Load { location, .. }) => location,
        _ => abort(()),
    };
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    assert_eq!(
        inputs(&mut analysis, loaded, DependencyRole::Value),
        vec![(pointer, DependencyRole::Address)]
    );
    for (instruction, expected) in [
        (1, MemoryOrigin::Entry),
        (
            2,
            MemoryOrigin::Write {
                block: 0,
                instruction: 1,
            },
        ),
        (
            4,
            MemoryOrigin::Write {
                block: 0,
                instruction: 1,
            },
        ),
    ] {
        let origins = {
            analysis
                .request(Observation::MemoryBefore {
                    location,
                    block: 0,
                    instruction,
                })
                .unwrap_or_else(abort)
                .dependencies()
                .cloned()
                .filter_map(|dependency| match dependency {
                    Dependency::Memory { origin, .. } => Some(origin),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(origins, vec![expected]);
    }
}

/// An opaque arm must not discard the other arm's required value or the selector.
#[test]
fn demand_keeps_other_requirements_after_an_unknown_phi_arm() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let condition = parameter(&mut builder, 0, MockType::I32);
    let key = parameter(&mut builder, 1, MockType::I32);
    builder.ensure_block(3);
    builder
        .in_block(0, |block| block.branch(condition, 1, 2))
        .unwrap_or_else(abort);
    let unknown = builder
        .in_block(1, |block| {
            let result = block.call(tmp(), Callee::Direct(7), vec![])?;
            block.jump(3)?;
            Ok(result)
        })
        .unwrap_or_else(abort);
    builder
        .in_block(2, |block| block.jump(3))
        .unwrap_or_else(abort);
    let result = builder
        .in_block(3, |block| {
            let result = block.empty_phi(tmp())?;
            block.ret(Some(result))?;
            Ok(result)
        })
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(3, result, 1, unknown)
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(3, result, 2, key)
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let required = inputs(&mut analysis, result, DependencyRole::Value);
    assert!(required.contains(&(condition, DependencyRole::Control)));
    assert!(required.contains(&(key, DependencyRole::Value)));
    let effects = {
        analysis
            .request(Observation::Value(result, DependencyRole::Value))
            .unwrap_or_else(abort)
            .dependencies()
            .cloned()
            .filter_map(|dependency| match dependency {
                Dependency::Effect { block, instruction } => Some((block, instruction)),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(effects, vec![(1, 0)]);
}

/// A host binding contract is retained independently from its argument dependencies.
struct Arguments;

impl BindingOracle<MockTarget> for Arguments {
    fn call(&self, _method: &u32) -> CallBinding {
        CallBinding::OfArguments
    }
}

/// Contract-modeled values still depend on the host model, not just constant arguments.
#[test]
fn demand_keeps_call_contract_assumptions() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let key = parameter(&mut builder, 0, MockType::I32);
    let result = builder
        .in_block(0, |block| {
            let result = block.call(tmp(), Callee::Direct(7), vec![key])?;
            block.ret(Some(result))?;
            Ok(result)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &Arguments);
    assert_eq!(
        inputs(&mut analysis, result, DependencyRole::Value),
        vec![(key, DependencyRole::Value)]
    );
    let contracts = {
        analysis
            .request(Observation::Value(result, DependencyRole::Value))
            .unwrap_or_else(abort)
            .dependencies()
            .filter(|dependency| {
                matches!(
                    dependency,
                    Dependency::CallContract {
                        block: 0,
                        instruction: 0
                    }
                )
            })
            .count()
    };
    assert_eq!(contracts, 1);
}

/// Invalid positions and unknown canonical cells refuse a dependency signature.
#[test]
fn demand_refuses_invalid_observations() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let key = parameter(&mut builder, 0, MockType::I32);
    builder
        .in_block(0, |block| block.ret(Some(key)))
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let invalid = { analysis.request(Observation::Control(99)).map(|_| ()) };
    assert_eq!(invalid, Err(DependencyRefusal::InvalidObservation));
    let unknown = MemoryLocation::StaticField(77);
    let invalid = {
        analysis
            .request(Observation::MemoryBefore {
                location: &unknown,
                block: 0,
                instruction: 0,
            })
            .map(|_| ())
    };
    assert_eq!(invalid, Err(DependencyRefusal::InvalidObservation));
}

/// Unknown memory in one arm does not erase a stored key in the other arm.
#[test]
fn demand_keeps_every_memory_phi_arm_and_its_control() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
    let condition = parameter(&mut builder, 0, MockType::I32);
    let pointer = parameter(&mut builder, 1, MockType::Ptr);
    let key = parameter(&mut builder, 2, MockType::I32);
    builder.ensure_block(3);
    builder
        .in_block(0, |block| block.branch(condition, 1, 2))
        .unwrap_or_else(abort);
    builder
        .in_block(1, |block| {
            block.call_void(Callee::Direct(7), vec![pointer])?;
            block.jump(3)
        })
        .unwrap_or_else(abort);
    builder
        .in_block(2, |block| {
            block.store_indirect(pointer, key, MockType::I32)?;
            block.jump(3)
        })
        .unwrap_or_else(abort);
    let loaded = builder
        .in_block(3, |block| {
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let required = inputs(&mut analysis, loaded, DependencyRole::Value);
    assert!(required.contains(&(key, DependencyRole::Value)));
    assert!(required.contains(&(condition, DependencyRole::Control)));
    assert!(required.contains(&(pointer, DependencyRole::Address)));
    let (effects, origins) = {
        let signature = analysis
            .request(Observation::Value(loaded, DependencyRole::Value))
            .unwrap_or_else(abort);
        let mut effects = Vec::new();
        let mut origins = Vec::new();
        for dependency in signature.dependencies().cloned() {
            match dependency {
                Dependency::Memory { origin, .. } => origins.push(origin),
                Dependency::Effect { block, instruction } => effects.push((block, instruction)),
                _ => {}
            }
        }
        (effects, origins)
    };
    assert_eq!(effects, vec![(1, 0)]);
    assert!(origins.contains(&MemoryOrigin::Merge { block: 3 }));
    assert!(origins.contains(&MemoryOrigin::Write {
        block: 1,
        instruction: 0
    }));
    assert!(origins.contains(&MemoryOrigin::Write {
        block: 2,
        instruction: 0
    }));
}

/// An intervening may-alias store becomes the dependency, rather than a stale value.
#[test]
fn demand_retains_alias_clobbers_and_their_actual_inputs() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let alias = parameter(&mut builder, 1, MockType::Ptr);
    let key = parameter(&mut builder, 2, MockType::I32);
    let loaded = builder
        .in_block(0, |block| {
            let fixed = block.const_i32(tmp(), 42)?;
            block.store_indirect(pointer, fixed, MockType::I32)?;
            block.store_indirect(alias, key, MockType::I32)?;
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let required = inputs(&mut analysis, loaded, DependencyRole::Value);
    assert!(required.contains(&(pointer, DependencyRole::Address)));
    assert!(required.contains(&(alias, DependencyRole::Address)));
    assert!(!required.contains(&(alias, DependencyRole::Value)));
    assert!(required.contains(&(key, DependencyRole::Value)));
    let effects = {
        analysis
            .request(Observation::Value(loaded, DependencyRole::Value))
            .unwrap_or_else(abort)
            .dependencies()
            .cloned()
            .filter_map(|dependency| match dependency {
                Dependency::Effect { block, instruction } => Some((block, instruction)),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(effects, vec![(0, 2)]);
}

/// Cyclic SSA requirements terminate, retaining both the bound and arithmetic input.
#[test]
fn demand_walks_loop_cycles_and_caches_roles_independently() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let bound = parameter(&mut builder, 0, MockType::I32);
    let key = parameter(&mut builder, 1, MockType::I32);
    builder.ensure_block(2);
    let start = builder
        .in_block(0, |block| {
            let start = block.const_i32(tmp(), 0)?;
            block.jump(1)?;
            Ok(start)
        })
        .unwrap_or_else(abort);
    let (counter, next) = builder
        .in_block(1, |block| {
            let counter = block.empty_phi(tmp())?;
            let next = block.add(tmp(), counter, key)?;
            let more = block.clt(tmp(), next, bound, false)?;
            block.branch(more, 1, 2)?;
            Ok((counter, next))
        })
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, counter, 0, start)
        .unwrap_or_else(abort);
    builder
        .add_phi_operand(1, counter, 1, next)
        .unwrap_or_else(abort);
    builder
        .in_block(2, |block| block.ret(Some(next)))
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let first = inputs(&mut analysis, next, DependencyRole::Value);
    assert!(first.contains(&(bound, DependencyRole::Control)));
    assert!(first.contains(&(key, DependencyRole::Value)));
    assert_eq!(first, inputs(&mut analysis, next, DependencyRole::Value));
}

/// A may-alias element write requires its selecting index as address state.
#[test]
fn demand_retains_alias_store_indices_as_address_requirements() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
    let array = parameter(&mut builder, 0, MockType::Ptr);
    let index = parameter(&mut builder, 1, MockType::I32);
    let key = parameter(&mut builder, 2, MockType::I32);
    let unrelated = parameter(&mut builder, 3, MockType::I32);
    let loaded = builder
        .in_block(0, |block| {
            let zero = block.const_i32(tmp(), 0)?;
            let fixed = block.const_i32(tmp(), 42)?;
            block.store_element(array, zero, fixed, MockType::I32)?;
            block.store_element(array, index, key, MockType::I32)?;
            let loaded = block.load_element(tmp(), array, zero, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let required = inputs(&mut analysis, loaded, DependencyRole::Value);
    assert!(required.contains(&(array, DependencyRole::Address)));
    assert!(required.contains(&(index, DependencyRole::Address)));
    assert!(required.contains(&(key, DependencyRole::Value)));
    assert!(!required.contains(&(index, DependencyRole::Value)));
    assert!(!required.iter().any(|(value, _)| *value == unrelated));
}

/// A possible overwrite retains the old bytes; exact and disjoint writes prune correctly.
#[test]
fn demand_keeps_previous_contents_across_possible_overwrites() {
    for relation in 0..3 {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
        let pointer = parameter(&mut builder, 0, MockType::Ptr);
        let other = parameter(&mut builder, 1, MockType::Ptr);
        let before = parameter(&mut builder, 2, MockType::I32);
        let after = parameter(&mut builder, 3, MockType::I32);
        let loaded = builder
            .in_block(0, |block| {
                let zero = block.const_i32(tmp(), 0)?;
                let one = block.const_i32(tmp(), 1)?;
                block.store_element(pointer, zero, before, MockType::I32)?;
                let (object, index) = match relation {
                    0 => (other, zero),
                    1 => (pointer, zero),
                    _ => (pointer, one),
                };
                block.store_element(object, index, after, MockType::I32)?;
                let loaded = block.load_element(tmp(), pointer, zero, MockType::I32)?;
                block.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        let ssa = finish(builder);
        let context = AnalysisContext::new(MACHINE);
        let mut analysis = Dependencies::analyze(&ssa, &context, &());
        let required = inputs(&mut analysis, loaded, DependencyRole::Value);
        assert_eq!(
            required.contains(&(before, DependencyRole::Value)),
            relation != 1,
            "relation {relation}: {required:?}"
        );
        assert_eq!(
            required.contains(&(after, DependencyRole::Value)),
            relation != 2,
            "relation {relation}: {required:?}"
        );
        assert_eq!(
            required.contains(&(other, DependencyRole::Address)),
            relation == 0
        );
    }
}

/// Read-modify-write results depend on the reaching bytes, not only their scalar operands.
#[test]
fn demand_keeps_atomic_read_contents_and_effect_identity() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let before = parameter(&mut builder, 1, MockType::I32);
    let after = parameter(&mut builder, 2, MockType::I32);
    let loaded = builder
        .in_block(0, |block| {
            block.store_indirect(pointer, before, MockType::I32)?;
            let loaded = block.atomic_exchange(
                tmp(),
                pointer,
                after,
                AtomicOrdering::Relaxed,
                AtomicAccessWidth::Bits32,
                AccessQualifiers::default(),
            )?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let required = inputs(&mut analysis, loaded, DependencyRole::Value);
    assert!(
        required.contains(&(before, DependencyRole::Value)),
        "{required:?}"
    );
    let effects = {
        analysis
            .request(Observation::Value(loaded, DependencyRole::Value))
            .unwrap_or_else(abort)
            .dependencies()
            .cloned()
            .filter_map(|dependency| match dependency {
                Dependency::Effect { block, instruction } => Some((block, instruction)),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    // The canonical atomic footprint is Unknown, so the preceding possible
    // store remains unresolved too; neither effect can be discarded.
    assert_eq!(effects, vec![(0, 0), (0, 1)]);
}

/// Reaching a nested observation depends on its enclosing guard across a straight-line block.
#[test]
fn demand_keeps_enclosing_control_across_single_predecessors() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = parameter(&mut builder, 0, MockType::I32);
    builder.ensure_block(3);
    builder
        .in_block(0, |block| block.branch(condition, 1, 3))
        .unwrap_or_else(abort);
    builder
        .in_block(1, |block| block.jump(2))
        .unwrap_or_else(abort);
    let observed = builder
        .in_block(2, |block| {
            let observed = block.const_i32(tmp(), 42)?;
            block.ret(Some(observed))?;
            Ok(observed)
        })
        .unwrap_or_else(abort);
    builder
        .in_block(3, |block| block.ret(None))
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    assert_eq!(
        inputs(&mut analysis, observed, DependencyRole::Value),
        vec![(condition, DependencyRole::Control)]
    );
}

/// A guard whose arms already rejoin is irrelevant to a later independent value.
#[test]
fn demand_drops_reconverged_control_from_independent_observations() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let condition = parameter(&mut builder, 0, MockType::I32);
    builder.ensure_block(3);
    builder
        .in_block(0, |block| block.branch(condition, 1, 2))
        .unwrap_or_else(abort);
    builder
        .in_block(1, |block| block.jump(3))
        .unwrap_or_else(abort);
    builder
        .in_block(2, |block| block.jump(3))
        .unwrap_or_else(abort);
    let observed = builder
        .in_block(3, |block| {
            let observed = block.const_i32(tmp(), 42)?;
            block.ret(Some(observed))?;
            Ok(observed)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    assert!(inputs(&mut analysis, observed, DependencyRole::Value).is_empty());
}

/// A partial write preserves the untouched bytes and therefore their original input.
#[test]
fn demand_keeps_unwritten_bytes_of_an_overlapping_store() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let before = parameter(&mut builder, 1, MockType::I32);
    let after = parameter(&mut builder, 2, MockType::I8);
    let loaded = builder
        .in_block(0, |block| {
            block.store_indirect(pointer, before, MockType::I32)?;
            block.store_indirect(pointer, after, MockType::I8)?;
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let required = inputs(&mut analysis, loaded, DependencyRole::Value);
    assert!(required.contains(&(before, DependencyRole::Value)));
    assert!(required.contains(&(after, DependencyRole::Value)));
    assert!(required.contains(&(pointer, DependencyRole::Address)));
}

/// Counts actual entry-content visits to expose work hidden before signature traversal.
struct EntryVisits(Cell<usize>);

impl BindingOracle<MockTarget> for EntryVisits {
    fn contents_at_entry(&self, _location: &MemoryLocation<MockTarget>) -> Option<BindingTime> {
        self.0.set(self.0.get().saturating_add(1));
        None
    }
}

/// A long memory history retains every relevant input and reuses the complete source.
#[test]
fn demand_source_derivation_preserves_long_memory_history() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let alias = parameter(&mut builder, 1, MockType::Ptr);
    let key = parameter(&mut builder, 2, MockType::I32);
    let loaded = builder
        .in_block(0, |block| {
            for _ in 0..4096 {
                block.store_indirect(alias, key, MockType::I32)?;
            }
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let oracle = EntryVisits(Cell::new(0));
    let mut analysis = Dependencies::analyze(&ssa, &context, &oracle);
    let mut required = inputs(&mut analysis, loaded, DependencyRole::Value);
    required.sort_unstable();
    let mut expected = vec![
        (pointer, DependencyRole::Address),
        (alias, DependencyRole::Address),
        (key, DependencyRole::Value),
    ];
    expected.sort_unstable();
    assert_eq!(required, expected);
    assert_eq!(
        oracle.0.get(),
        1,
        "the complete history reaches the entry contract"
    );
    assert_eq!(
        inputs(&mut analysis, loaded, DependencyRole::Value).len(),
        3
    );
    assert_eq!(oracle.0.get(), 1, "complete source may be reused");
}

/// A host's static-content promise stays explicit; it is not confused with an address.
struct StaticContents;

impl BindingOracle<MockTarget> for StaticContents {
    fn contents_at_entry(&self, _location: &MemoryLocation<MockTarget>) -> Option<BindingTime> {
        Some(BindingTime::Static)
    }
}

/// Caller pointer and host contents remain distinct requirements even when contents are static.
#[test]
fn demand_retains_host_content_contracts_and_hidden_call_inputs() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let hidden = parameter(&mut builder, 1, MockType::I32);
    let (loaded, called) = builder
        .in_block(0, |block| {
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            let fixed = block.const_i32(tmp(), 7)?;
            let called = block.emit_def(tmp(), |dest| {
                CallData::new(Some(dest), Callee::Direct(7), vec![fixed])
                    .with_implicit_input(CallInputKey::new(17), hidden)
                    .into()
            })?;
            block.ret(Some(called))?;
            Ok((loaded, called))
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &StaticContents);
    assert_eq!(
        inputs(&mut analysis, loaded, DependencyRole::Value),
        vec![(pointer, DependencyRole::Address)]
    );
    let origins = {
        analysis
            .request(Observation::Value(loaded, DependencyRole::Value))
            .unwrap_or_else(abort)
            .dependencies()
            .cloned()
            .filter_map(|dependency| match dependency {
                Dependency::Memory { origin, .. } => Some(origin),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(origins, vec![MemoryOrigin::Host]);
    let mut analysis = Dependencies::analyze(&ssa, &context, &Arguments);
    assert_eq!(
        inputs(&mut analysis, called, DependencyRole::Value),
        vec![(hidden, DependencyRole::Value)]
    );
}

/// A protected store cannot establish contents at exceptional entry.
#[test]
fn demand_keeps_exception_contents_instead_of_a_preempted_store() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let key = parameter(&mut builder, 1, MockType::I32);
    builder.ensure_block(1);
    builder
        .in_block(0, |block| {
            block.store_indirect(pointer, key, MockType::I32)?;
            block.ret(None)
        })
        .unwrap_or_else(abort);
    let loaded = builder
        .in_block(1, |block| {
            let loaded = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let mut ssa = finish(builder);
    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 2),
        filter_range: None,
        selector: None,
    }]);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    assert_eq!(
        inputs(&mut analysis, loaded, DependencyRole::Value),
        vec![(pointer, DependencyRole::Address)]
    );
    let (origins, exceptions) = {
        let signature = analysis
            .request(Observation::Value(loaded, DependencyRole::Value))
            .unwrap_or_else(abort);
        let mut origins = Vec::new();
        let mut exceptions = Vec::new();
        for dependency in signature.dependencies().cloned() {
            match dependency {
                Dependency::Memory { origin, .. } => origins.push(origin),
                Dependency::Exception { block } => exceptions.push(block),
                _ => {}
            }
        }
        (origins, exceptions)
    };
    assert_eq!(origins, vec![MemoryOrigin::Exception { block: 1 }]);
    assert_eq!(exceptions, vec![1]);
}

/// Exceptional reach remains a requirement across handler blocks, but not after both paths rejoin.
#[test]
fn demand_keeps_exceptional_scalar_reach_without_tainting_a_rejoined_value() {
    for inside_handler in [true, false] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let observed_block = if inside_handler { 2 } else { 3 };
        builder.ensure_block(observed_block);
        builder
            .in_block(0, |block| {
                block.call_void(Callee::Direct(7), vec![])?;
                if inside_handler {
                    block.ret(None)
                } else {
                    block.jump(2)
                }
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |block| block.jump(observed_block))
            .unwrap_or_else(abort);
        if !inside_handler {
            builder
                .in_block(2, |block| block.jump(3))
                .unwrap_or_else(abort);
        }
        let observed = builder
            .in_block(observed_block, |block| {
                let observed = block.const_i32(tmp(), 42)?;
                block.ret(Some(observed))?;
                Ok(observed)
            })
            .unwrap_or_else(abort);
        let mut ssa = finish(builder);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 1),
            handler_range: BlockRange::new(1, if inside_handler { 3 } else { 2 }),
            filter_range: None,
            selector: None,
        }]);
        let context = AnalysisContext::new(MACHINE);
        assert!(BindingTimes::analyze(&ssa, &context, &()).is_static(observed));
        let mut analysis = Dependencies::analyze(&ssa, &context, &());
        let exceptions = {
            analysis
                .request(Observation::Value(observed, DependencyRole::Value))
                .unwrap_or_else(abort)
                .dependencies()
                .cloned()
                .filter_map(|dependency| match dependency {
                    Dependency::Exception { block } => Some(block),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            exceptions,
            if inside_handler { vec![1] } else { vec![] },
            "inside handler: {inside_handler}"
        );
    }
}

/// Historical definition metadata cannot establish a missing value's dependencies.
#[test]
fn demand_keeps_inputs_and_unknowns_when_only_a_stale_definition_site_remains() {
    for caller in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(usize::from(caller), 1);
        let origin = if caller {
            VariableOrigin::Argument(0)
        } else {
            VariableOrigin::Local(0)
        };
        let missing =
            builder
                .function_mut()
                .create_variable(origin, 0, DefSite::entry(), MockType::I32);
        builder
            .in_block(0, |block| {
                block.const_i32(tmp(), 42)?;
                block.ret(Some(missing))
            })
            .unwrap_or_else(abort);
        let mut ssa = finish(builder);
        ssa.variable_mut(missing)
            .unwrap_or_else(|| abort(()))
            .set_def_site(DefSite::instruction(0, 0));
        assert_eq!(
            DefUseIndex::build(&ssa).def_site(missing),
            Some(DefSite::instruction(0, 0))
        );
        let context = AnalysisContext::new(MACHINE);
        let mut analysis = Dependencies::analyze(&ssa, &context, &());
        let (inputs, unknowns) = {
            let signature = analysis
                .request(Observation::Value(missing, DependencyRole::Value))
                .unwrap_or_else(abort);
            let mut inputs = Vec::new();
            let mut unknowns = Vec::new();
            for dependency in signature.dependencies().cloned() {
                match dependency {
                    Dependency::Input { value, role } => inputs.push((value, role)),
                    Dependency::Unknown { value } => unknowns.push(value),
                    _ => {}
                }
            }
            (inputs, unknowns)
        };
        assert_eq!(
            inputs,
            if caller {
                vec![(missing, DependencyRole::Value)]
            } else {
                vec![]
            }
        );
        assert_eq!(unknowns, if caller { vec![] } else { vec![Some(missing)] });
    }
}

/// Grouped roots retain semantic roles while visiting a shared producer once.
#[test]
fn demand_group_requests_keep_roles_and_share_traversal() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let key = parameter(&mut builder, 0, MockType::I32);
    let pointer = parameter(&mut builder, 1, MockType::Ptr);
    let value = builder
        .in_block(0, |block| {
            let fixed = block.const_i32(tmp(), 7)?;
            let value = block.add(tmp(), key, fixed)?;
            block.ret(Some(value))?;
            Ok(value)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    ssa.validate().unwrap_or_else(abort);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    let signature = analysis
        .request_all(&[
            Observation::Value(value, DependencyRole::Value),
            Observation::Value(value, DependencyRole::Value),
            Observation::Value(key, DependencyRole::Bound),
            Observation::Value(pointer, DependencyRole::Address),
        ])
        .unwrap_or_else(abort);
    let inputs: BTreeSet<_> = signature
        .dependencies()
        .filter_map(|dependency| match dependency {
            Dependency::Input { value, role } => Some((*value, *role)),
            _ => None,
        })
        .collect();
    assert_eq!(
        inputs,
        BTreeSet::from([
            (key, DependencyRole::Value),
            (key, DependencyRole::Bound),
            (pointer, DependencyRole::Address),
        ])
    );
    // Each input once, in its role, however many roots read it.
    assert_eq!(signature.len(), 3);
}

/// A refused member publishes no group; empty groups describe no requested output.
#[test]
fn demand_group_refusal_does_not_publish_a_partial_signature() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let key = parameter(&mut builder, 0, MockType::I32);
    builder
        .in_block(0, |block| block.ret(Some(key)))
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    ssa.validate().unwrap_or_else(abort);
    let context = AnalysisContext::new(MACHINE);
    let mut analysis = Dependencies::analyze(&ssa, &context, &());
    assert_eq!(
        analysis
            .request_all(&[
                Observation::Value(key, DependencyRole::Value),
                Observation::Value(SsaVarId::from_index(9999), DependencyRole::Value),
            ])
            .err(),
        Some(DependencyRefusal::InvalidObservation)
    );
    let valid = analysis
        .request_all(&[Observation::Value(key, DependencyRole::Value)])
        .unwrap_or_else(abort);
    assert_eq!(valid.len(), 1);
    drop(valid);
    let empty = analysis.request_all(&[]).unwrap_or_else(abort);
    assert!(empty.is_empty());
}

/// Ownership detaches canonical locations while preserving the precise input and memory version.
#[test]
fn demand_owned_signature_outlives_its_analysis() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let pointer = parameter(&mut builder, 0, MockType::Ptr);
    let value = builder
        .in_block(0, |block| {
            let value = block.load_indirect(tmp(), pointer, MockType::I32)?;
            block.ret(Some(value))?;
            Ok(value)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    ssa.validate().unwrap_or_else(abort);
    let context = AnalysisContext::new(MACHINE);
    let (signature, before): (DependencySignature<MockTarget>, String) = {
        let mut analysis = Dependencies::analyze(&ssa, &context, &());
        let signature = analysis
            .request_all(&[
                Observation::Value(value, DependencyRole::Value),
                Observation::Control(0),
            ])
            .unwrap_or_else(abort);
        let before = format!("{signature:?}");
        (signature, before)
    };
    assert_eq!(format!("{signature:?}"), before);
    let expected = MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
        AddressBase::Value(pointer),
        None,
        0,
        0,
        Some(32),
        None,
        PointerSize::Bit64,
    ));
    assert!(signature.dependencies().any(|dependency| matches!(dependency,
        Dependency::Memory { location, origin: MemoryOrigin::Entry, .. } if **location == expected)));
    assert!(
        signature
            .dependencies()
            .any(|dependency| matches!(dependency,
        Dependency::Input { value, role: DependencyRole::Address } if *value == pointer))
    );
}

/// A signature of stated dependencies keeps every kind and its order, and a
/// dependency stated beside a request's comes after the request's own.
#[test]
fn demand_signature_retains_all_dependency_kinds() {
    let location = Rc::new(MemoryLocation::<MockTarget>::StaticField(42));
    let signature = DependencySignature::of(vec![
        Dependency::Input {
            value: SsaVarId::from_index(1),
            role: DependencyRole::Control,
        },
        Dependency::Memory {
            location: Rc::clone(&location),
            version: 7,
            origin: MemoryOrigin::Write {
                block: 2,
                instruction: 3,
            },
        },
        Dependency::Effect {
            block: 4,
            instruction: 5,
        },
        Dependency::Exception { block: 6 },
        Dependency::CallContract {
            block: 7,
            instruction: 8,
        },
        Dependency::Unknown {
            value: Some(SsaVarId::from_index(9)),
        },
    ])
    .with(Dependency::Unknown { value: None });
    assert_eq!(signature.len(), 7);
    assert!(matches!(
        signature.dependencies().nth(1),
        Some(Dependency::Memory {
            location,
            version: 7,
            origin: MemoryOrigin::Write {
                block: 2,
                instruction: 3
            }
        }) if **location == MemoryLocation::StaticField(42)
    ));
    assert!(matches!(
        signature.dependencies().last(),
        Some(Dependency::Unknown { value: None })
    ));
}
