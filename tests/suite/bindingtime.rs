//! Binding time: which values are known before the program runs.
//!
//! Each rule is pinned from both sides - the case it makes static and the case
//! it must not - because the failure that matters is one direction only: a
//! dynamic value called static folds a wrong constant into the output.

use std::process;

use analyssa::{
    analysis::{
        bindingtime::{BindingOracle, BindingTime, BindingTimes, CallBinding},
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        image::{ImageData, ImageRegion, Mutability},
        memory::MemoryLocation,
    },
    ir::{
        CallData, CallInputKey, Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Read-only bytes at `0x1000`, writable ones at `0x2000`.
struct TestImage;

impl ImageData for TestImage {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        [
            (0x1000, 0x1010, Mutability::ReadOnly),
            (0x2000, 0x2010, Mutability::Writable),
        ]
        .into_iter()
        .find(|(start, end, _)| (*start..*end).contains(&address))
        .map(|(start, end, mutability)| ImageRegion {
            start,
            end,
            mutability,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        const BYTES: [u8; 0x10] = [0x2a; 0x10];
        let region = self.region(address)?;
        let start = usize::try_from(address.checked_sub(region.start)?).ok()?;
        BYTES.get(start..start.checked_add(len)?)
    }
}

/// Ends the test: a fixture the builder rejects is a broken test, not a
/// finding.
fn abort<E, T>(_: E) -> T {
    process::abort()
}

/// A 32-bit temporary.
fn tmp() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::I32)
}

/// A pointer temporary.
fn ptr() -> SsaDefSpec<MockTarget> {
    SsaDefSpec::tmp(MockType::Ptr)
}

/// The function's first parameter: a value nothing in it defines.
fn parameter(builder: &mut SsaFunctionBuilder<MockTarget>) -> SsaVarId {
    builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    )
}

/// Finishes `builder`, panicking on a malformed fixture.
fn finish(builder: SsaFunctionBuilder<MockTarget>) -> SsaFunction<MockTarget> {
    builder.finish().unwrap_or_else(abort)
}

/// Binding times with the test image, default budgets, and no host knowledge.
fn times(ssa: &SsaFunction<MockTarget>) -> BindingTimes {
    let image = TestImage;
    BindingTimes::analyze(ssa, &AnalysisContext::new(MACHINE).with_image(&image), &())
}

/// Arithmetic over constants is static; anything a parameter reaches is not.
#[test]
fn a_parameter_is_dynamic_and_so_is_everything_it_reaches() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let p = parameter(&mut builder);
    let (five, ten, mixed) = builder
        .in_block(0, |b| {
            let five = b.const_i32(tmp(), 5)?;
            let ten = b.add(tmp(), five, five)?;
            let mixed = b.add(tmp(), ten, p)?;
            b.ret(Some(mixed))?;
            Ok((five, ten, mixed))
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let times = times(&ssa);
    assert!(times.is_static(five));
    assert!(times.is_static(ten));
    assert_eq!(times.of(p), BindingTime::Dynamic);
    assert_eq!(times.of(mixed), BindingTime::Dynamic);
}

/// A call's result is dynamic, even over static arguments: nothing says what
/// the callee returns.
#[test]
fn a_call_result_is_dynamic() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let (argument, result) = builder
        .in_block(0, |b| {
            let argument = b.const_i32(tmp(), 1)?;
            let result = b.call(tmp(), Callee::Direct(7), vec![argument])?;
            b.ret(Some(result))?;
            Ok((argument, result))
        })
        .unwrap_or_else(abort);
    let times = times(&finish(builder));
    assert!(times.is_static(argument));
    assert!(!times.is_static(result));
}

/// A load of read-only image data is static; of writable data, or with no
/// image at all, it is not.
#[test]
fn only_read_only_image_data_loads_static() {
    let load = |address: u32| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let loaded = builder
            .in_block(0, |b| {
                let at = b.const_value(ptr(), ConstValue::Symbol(address))?;
                let loaded = b.load_indirect(tmp(), at, MockType::I32)?;
                b.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        (finish(builder), loaded)
    };
    let (ssa, loaded) = load(0x1000);
    assert!(times(&ssa).is_static(loaded));
    assert!(
        !BindingTimes::analyze(&ssa, &AnalysisContext::new(MACHINE), &()).is_static(loaded),
        "with no image the program's memory is unknown"
    );
    let (ssa, loaded) = load(0x2000);
    assert!(
        !times(&ssa).is_static(loaded),
        "writable data may have been written before the function ran"
    );
}

/// A load reads what the covering store before it wrote; a call in between
/// may have written anything.
#[test]
fn a_load_reads_the_binding_time_of_what_was_stored() {
    let store_then_load = |dynamic_value: bool, call_between: bool| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let p = parameter(&mut builder);
        let loaded = builder
            .in_block(0, |b| {
                let at = b.const_value(ptr(), ConstValue::Symbol(0x2004))?;
                let value = if dynamic_value {
                    p
                } else {
                    b.const_i32(tmp(), 3)?
                };
                b.store_indirect(at, value, MockType::I32)?;
                if call_between {
                    b.call_void(Callee::Direct(7), Vec::new())?;
                }
                let loaded = b.load_indirect(tmp(), at, MockType::I32)?;
                b.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        times(&finish(builder)).is_static(loaded)
    };
    assert!(store_then_load(false, false));
    assert!(!store_then_load(true, false));
    assert!(!store_then_load(false, true));
}

/// Exact writes restore knowledge after opaque clobbers throughout a long observation sequence.
#[test]
fn binding_time_tracks_long_interleaved_memory_observations() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let mut observed = Vec::new();
    builder
        .in_block(0, |block| {
            let address = block.const_value(ptr(), ConstValue::Symbol(0x2004))?;
            let fixed = block.const_i32(tmp(), 3)?;
            for _ in 0..2048 {
                block.store_indirect(address, fixed, MockType::I32)?;
                let known = block.load_indirect(tmp(), address, MockType::I32)?;
                block.call_void(Callee::Direct(7), Vec::new())?;
                let unknown = block.load_indirect(tmp(), address, MockType::I32)?;
                observed.push((known, unknown));
            }
            block.ret(None)
        })
        .unwrap_or_else(abort);
    let binding = times(&finish(builder));
    for (known, unknown) in observed {
        assert!(
            binding.is_static(known),
            "the immediately preceding exact write"
        );
        assert!(!binding.is_static(unknown), "the intervening unknown call");
    }
}

/// A merge of two constants is static only when what picks between them is.
///
/// `x = c ? 1 : 2`: each arm is a constant, and the merge is one only when `c`
/// is.
#[test]
fn a_merge_is_as_dynamic_as_the_branch_deciding_it() {
    let diamond = |dynamic_condition: bool| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let p = parameter(&mut builder);
        builder.ensure_block(3);
        builder
            .in_block(0, |b| {
                let condition = if dynamic_condition {
                    p
                } else {
                    b.const_bool(tmp(), true)?
                };
                b.branch(condition, 1, 2)
            })
            .unwrap_or_else(abort);
        let one = builder
            .in_block(1, |b| {
                let one = b.const_i32(tmp(), 1)?;
                b.jump(3)?;
                Ok(one)
            })
            .unwrap_or_else(abort);
        let two = builder
            .in_block(2, |b| {
                let two = b.const_i32(tmp(), 2)?;
                b.jump(3)?;
                Ok(two)
            })
            .unwrap_or_else(abort);
        let merged = builder
            .in_block(3, |b| {
                let merged = b.phi(tmp(), [(1, one), (2, two)])?;
                b.ret(Some(merged))?;
                Ok(merged)
            })
            .unwrap_or_else(abort);
        let times = times(&finish(builder));
        assert!(times.is_static(one) && times.is_static(two));
        times.is_static(merged)
    };
    assert!(diamond(false));
    assert!(!diamond(true));
}

/// A loop counter is static when its start, step and bound are: the branch
/// that decides whether the loop goes round again is.
#[test]
fn a_loop_counter_with_a_static_bound_is_static() {
    let counter = |dynamic_bound: bool| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let p = parameter(&mut builder);
        builder.ensure_block(2);
        let (start, step, bound) = builder
            .in_block(0, |b| {
                let start = b.const_i32(tmp(), 0)?;
                let step = b.const_i32(tmp(), 1)?;
                let bound = if dynamic_bound {
                    p
                } else {
                    b.const_i32(tmp(), 10)?
                };
                b.jump(1)?;
                Ok((start, step, bound))
            })
            .unwrap_or_else(abort);
        let (i, next) = builder
            .in_block(1, |b| {
                let i = b.empty_phi(tmp())?;
                let next = b.add(tmp(), i, step)?;
                let more = b.clt(tmp(), next, bound, false)?;
                b.branch(more, 1, 2)?;
                Ok((i, next))
            })
            .unwrap_or_else(abort);
        builder
            .add_phi_operand(1, i, 0, start)
            .unwrap_or_else(abort);
        builder.add_phi_operand(1, i, 1, next).unwrap_or_else(abort);
        builder
            .in_block(2, |b| b.ret(Some(next)))
            .unwrap_or_else(abort);
        let times = times(&finish(builder));
        (times.is_static(i), times.is_static(next))
    };
    assert_eq!(counter(false), (true, true));
    assert_eq!(counter(true), (false, false));
}

/// A memory merge follows the same rule as a value merge: stores of two
/// constants in the arms reach a load after the join as one value only when
/// the branch is static.
#[test]
fn a_memory_merge_is_as_dynamic_as_the_branch_deciding_it() {
    let diamond = |dynamic_condition: bool| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let p = parameter(&mut builder);
        builder.ensure_block(3);
        let at = builder
            .in_block(0, |b| {
                let at = b.const_value(ptr(), ConstValue::Symbol(0x2008))?;
                let condition = if dynamic_condition {
                    p
                } else {
                    b.const_bool(tmp(), false)?
                };
                b.branch(condition, 1, 2)?;
                Ok(at)
            })
            .unwrap_or_else(abort);
        for (block, value) in [(1, 1), (2, 2)] {
            builder
                .in_block(block, |b| {
                    let value = b.const_i32(tmp(), value)?;
                    b.store_indirect(at, value, MockType::I32)?;
                    b.jump(3)
                })
                .unwrap_or_else(abort);
        }
        let loaded = builder
            .in_block(3, |b| {
                let loaded = b.load_indirect(tmp(), at, MockType::I32)?;
                b.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        times(&finish(builder)).is_static(loaded)
    };
    assert!(diamond(false));
    assert!(!diamond(true));
}

/// When Memory SSA runs out of budget no load is classified, so every load is
/// dynamic - even one of read-only data - and the analysis says why.
#[test]
fn an_exhausted_memory_budget_makes_every_load_dynamic_and_says_so() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let loaded = builder
        .in_block(0, |b| {
            let at = b.const_value(ptr(), ConstValue::Symbol(0x1000))?;
            let loaded = b.load_indirect(tmp(), at, MockType::I32)?;
            b.ret(Some(loaded))?;
            Ok(loaded)
        })
        .unwrap_or_else(abort);
    let ssa = finish(builder);
    let image = TestImage;
    let context = AnalysisContext::new(MACHINE)
        .with_image(&image)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MemorySsaCells, 0));
    let times = BindingTimes::analyze(&ssa, &context, &());
    assert!(!times.is_static(loaded));
    assert!(times.exhaustion().is_some());
}

/// A host that knows three things: method 7 returns a function of its
/// arguments, method 8 fills the object passed first from its other
/// arguments, and the writable cell at `0x200c` is set once from constants.
struct Host;

impl BindingOracle<MockTarget> for Host {
    fn call(&self, method: &u32) -> CallBinding {
        match method {
            7 => CallBinding::OfArguments,
            8 => CallBinding::Fills { argument: 0 },
            _ => CallBinding::Opaque,
        }
    }

    fn contents_at_entry(&self, location: &MemoryLocation<MockTarget>) -> Option<BindingTime> {
        matches!(location, MemoryLocation::Indirect(cell) if cell.offset_bits == 0x200c * 8)
            .then_some(BindingTime::Static)
    }
}

/// Binding times with the test image and [`Host`].
fn times_with_host(ssa: &SsaFunction<MockTarget>) -> BindingTimes {
    let image = TestImage;
    BindingTimes::analyze(
        ssa,
        &AnalysisContext::new(MACHINE).with_image(&image),
        &Host,
    )
}

/// A call the host says is a function of its arguments is as static as they
/// are; any other call stays dynamic.
#[test]
fn a_call_of_its_arguments_is_as_static_as_they_are() {
    let call = |method: u32, dynamic_argument: bool| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let p = parameter(&mut builder);
        let result = builder
            .in_block(0, |b| {
                let argument = if dynamic_argument {
                    p
                } else {
                    b.const_i32(tmp(), 4)?
                };
                let result = b.call(tmp(), Callee::Direct(method), vec![argument])?;
                b.ret(Some(result))?;
                Ok(result)
            })
            .unwrap_or_else(abort);
        times_with_host(&finish(builder)).is_static(result)
    };
    assert!(call(7, false));
    assert!(!call(7, true));
    assert!(!call(9, false), "a call the host says nothing about");
}

/// An object a call fills holds what the call's other arguments fix, wherever
/// the object came from, and only its own cells do.
///
/// The buffer comes from an opaque allocation, so its address is dynamic; the
/// fill is in the entry block and the reads in the next, so what reaches them
/// is the version Memory SSA says the call defined.
#[test]
fn a_filled_object_holds_what_the_filling_arguments_fix() {
    #[derive(Clone, Copy, PartialEq)]
    enum Read {
        Filled,
        FilledAtIndex { dynamic: bool },
        Unrelated,
    }
    let fill_then_read = |dynamic_key: bool, read: Read| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let p = parameter(&mut builder);
        builder.ensure_block(1);
        let (buffer, other) = builder
            .in_block(0, |b| {
                let buffer = b.call(ptr(), Callee::Direct(1), Vec::new())?;
                let other = b.call(ptr(), Callee::Direct(1), Vec::new())?;
                let key = if dynamic_key {
                    p
                } else {
                    b.const_i32(tmp(), 9)?
                };
                b.call_void(Callee::Direct(8), vec![buffer, key])?;
                b.jump(1)?;
                Ok((buffer, other))
            })
            .unwrap_or_else(abort);
        let loaded = builder
            .in_block(1, |b| {
                let at = match read {
                    Read::Filled => buffer,
                    // `buffer + i*4`, which names `buffer` as the object and
                    // `i` as the index within it.
                    Read::FilledAtIndex { dynamic } => {
                        let index = if dynamic { p } else { b.const_i32(tmp(), 2)? };
                        let four = b.const_i32(tmp(), 4)?;
                        let scaled = b.mul(tmp(), index, four)?;
                        b.add(ptr(), buffer, scaled)?
                    }
                    Read::Unrelated => other,
                };
                let loaded = b.load_indirect(tmp(), at, MockType::I32)?;
                b.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        let ssa = finish(builder);
        let with_host = times_with_host(&ssa).is_static(loaded);
        assert!(!times(&ssa).is_static(loaded), "a host that says nothing");
        with_host
    };
    assert!(fill_then_read(false, Read::Filled));
    assert!(!fill_then_read(true, Read::Filled), "the key is dynamic");
    assert!(fill_then_read(
        false,
        Read::FilledAtIndex { dynamic: false }
    ));
    assert!(
        !fill_then_read(false, Read::FilledAtIndex { dynamic: true }),
        "which element is read is dynamic"
    );
    assert!(
        !fill_then_read(false, Read::Unrelated),
        "another object is clobbered as by any call"
    );
}

/// A location the host says is static on entry loads static; its neighbour,
/// which the host says nothing about, is writable data and does not.
#[test]
fn a_location_the_host_knows_is_static_on_entry_loads_static() {
    let load = |address: u32| {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let loaded = builder
            .in_block(0, |b| {
                let at = b.const_value(ptr(), ConstValue::Symbol(address))?;
                let loaded = b.load_indirect(tmp(), at, MockType::I32)?;
                b.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        times_with_host(&finish(builder)).is_static(loaded)
    };
    assert!(load(0x200c));
    assert!(!load(0x2008));
}

/// Hidden dynamic inputs prevent static returns and filled contents; the
/// filled-object index continues to address only ordinary arguments.
#[test]
fn implicit_inputs_contribute_to_return_and_filled_content_binding_times() {
    for dynamic in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let incoming = parameter(&mut builder);
        builder.ensure_block(1);
        let (returned, buffer) = builder
            .in_block(0, |b| {
                let explicit = b.const_i32(tmp(), 4)?;
                let hidden = if dynamic {
                    incoming
                } else {
                    b.const_i32(tmp(), 9)?
                };
                let returned = b.emit_def(tmp(), |dest| {
                    CallData::new(Some(dest), Callee::Direct(7), vec![explicit])
                        .with_implicit_input(CallInputKey::new(17), hidden)
                        .into()
                })?;
                let buffer = b.call(ptr(), Callee::Direct(1), Vec::new())?;
                b.emit_no_defs(
                    CallData::new(None, Callee::Direct(8), vec![buffer, explicit])
                        .with_implicit_input(CallInputKey::new(0), hidden)
                        .into(),
                )?;
                b.jump(1)?;
                Ok((returned, buffer))
            })
            .unwrap_or_else(abort);
        let loaded = builder
            .in_block(1, |b| {
                let loaded = b.load_indirect(tmp(), buffer, MockType::I32)?;
                b.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        let times = times_with_host(&finish(builder));
        assert_eq!(
            times.is_static(returned),
            !dynamic,
            "return, dynamic={dynamic}"
        );
        assert_eq!(
            times.is_static(loaded),
            !dynamic,
            "contents, dynamic={dynamic}"
        );
    }
}
