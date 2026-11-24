//! Reading the host's image: regions, and loads that fold to its bytes.
//!
//! A load of a cell the image reports read-only is the one load whose value is
//! known without knowing what ran before: a write to that cell faults, so no
//! execution continues past one. These pin that rule and every way a load
//! falls outside it - a writable region, an extent the region does not cover,
//! an address in a space the host cannot relate to its symbols, a type a
//! constant cannot state, an exhausted budget.

use analyssa::{
    PointerSize,
    analysis::{
        ConstEvaluator,
        address::AddressBase,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
        image::{ImageData, ImageRegion, Mutability},
        memory::{
            IndirectLocation, MemoryDefSite, MemoryLocation, MemoryOp, MemoryRegion, MemorySsa,
            MemoryVersion,
        },
    },
    events::{EventKind, EventLog, NullListener},
    ir::{
        CallData, Callee, CmpKind,
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        lanes::VectorConst,
        ops::{AccessQualifiers, SsaOp, VectorMaskMode, VectorPackKind, VectorSegmentLayout},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::MemoryOptimizationPass,
    scheduling::{FunctionPass, PassEnv},
    target::{Endianness, Target},
    testing::{FactfulTarget, MACHINE, MockTarget, MockType},
};

/// A two-region image: read-only bytes at `0x1000`, writable at `0x2000`.
struct TestImage {
    /// Byte order the image is read in.
    endianness: Endianness,
    /// What an integer address adds to reach the symbol space, if known.
    number_offset: Option<i64>,
    /// The bytes at `0x1000`.
    rodata: Vec<u8>,
    /// The bytes at `0x2000`.
    data: Vec<u8>,
    /// Cells the host proves hold an address: `(cell, width, address)`.
    addresses: Vec<(u64, u64, u64)>,
}

impl TestImage {
    /// A little-endian image whose integers and symbols coincide.
    fn new() -> Self {
        Self {
            endianness: Endianness::Little,
            number_offset: Some(0),
            rodata: vec![0x78, 0x56, 0x34, 0x12, 0xff, 0xff, 0xff, 0xff],
            data: vec![0x11, 0x22, 0x33, 0x44],
            addresses: Vec::new(),
        }
    }

    /// The region `address` falls in, with its bytes.
    fn region_of(&self, address: u64) -> Option<(ImageRegion, &[u8])> {
        let regions = [
            (0x1000u64, &self.rodata, Mutability::ReadOnly),
            (0x2000u64, &self.data, Mutability::Writable),
        ];
        regions.into_iter().find_map(|(start, bytes, mutability)| {
            let end = start.checked_add(u64::try_from(bytes.len()).ok()?)?;
            (address >= start && address < end).then_some((
                ImageRegion {
                    start,
                    end,
                    mutability,
                },
                bytes.as_slice(),
            ))
        })
    }
}

impl ImageData for TestImage {
    fn endianness(&self) -> Endianness {
        self.endianness
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        self.region_of(address).map(|(region, _)| region)
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let (region, bytes) = self.region_of(address)?;
        let start = usize::try_from(address.checked_sub(region.start)?).ok()?;
        bytes.get(start..start.checked_add(len)?)
    }

    fn number_offset(&self) -> Option<i64> {
        self.number_offset
    }

    fn address_at(&self, address: u64, len: u64) -> Option<u64> {
        self.addresses
            .iter()
            .find(|(cell, width, _)| (*cell, *width) == (address, len))
            .map(|(_, _, named)| *named)
    }
}

/// Wraps `op` in an instruction.
fn instr<T: Target>(op: SsaOp<T>) -> SsaInstruction<T> {
    SsaInstruction::synthetic(op)
}

/// One load of `address`, stated as `constant`, typed `ty`, with `between`
/// placed after the constant and before the load, returning what it loaded.
///
/// Returns the function and the value the load defines.
fn load_of<T: Target<Type = MockType>>(
    constant: ConstValue<T>,
    ty: MockType,
    between: Vec<SsaOp<T>>,
) -> (SsaFunction<T>, SsaVarId) {
    let mut ssa = SsaFunction::new(0, 8);
    let address = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::Ptr,
    );
    let load_at = between.len().saturating_add(1);
    let loaded = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, load_at),
        ty,
    );
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: address,
        value: constant,
    }));
    for op in between {
        b0.add_instruction(instr(op));
    }
    b0.add_instruction(instr(SsaOp::LoadIndirect {
        dest: loaded,
        addr: address,
        value_type: ty,
        access: AccessQualifiers::FLAT,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    (ssa, loaded)
}

/// **A cell a call goes through is a read the program enters.** The cell at
/// `0x1000` holds the address `0x2000`; loaded and called through, the fold
/// tells the host the value it read is entered, which makes the cell a code
/// pointer rather than data the function reads. Loaded and only returned, it
/// is not.
#[test]
fn a_cell_a_call_goes_through_is_read_as_entered() {
    let mut image = TestImage::new();
    image.addresses.push((0x1000, 8, 0x2000));
    for called in [true, false] {
        let (mut ssa, loaded) =
            load_of::<FactfulTarget>(ConstValue::Symbol(0x1000), MockType::U64, Vec::new());
        if called && let Some(block) = ssa.block_mut(0) {
            let at = block.instructions().len().saturating_sub(1);
            block.instructions_mut().insert(
                at,
                instr(SsaOp::from(CallData::new(
                    None,
                    Callee::Indirect {
                        fptr: loaded,
                        signature: 0,
                    },
                    Vec::new(),
                ))),
            );
        }
        ssa.refresh_def_sites();
        ssa.recompute_uses();
        assert_eq!(ssa.entered_values().contains(&loaded), called);
        fold(&mut ssa, &image);
        assert_eq!(
            ssa.facts().image_reads,
            vec![(0x1000, 8)],
            "called {called}"
        );
        let entered: &[u64] = if called { &[0x1000] } else { &[] };
        assert_eq!(ssa.facts().entered_reads, entered, "called {called}");
    }
}

/// What defines `value` after the pass, when an instruction does.
fn definition<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> Option<SsaOp<T>> {
    ssa.blocks().iter().find_map(|block| {
        block
            .instructions()
            .iter()
            .map(SsaInstruction::op)
            .find(|op| op.dest() == Some(value))
            .cloned()
    })
}

/// Runs the memory pass over `ssa` with `image`, returning the events.
fn fold<T: Target<MethodRef = u32>>(ssa: &mut SsaFunction<T>, image: &TestImage) -> EventLog<T> {
    let events = EventLog::new();
    let context = AnalysisContext::new(MACHINE).with_image(image);
    assert!(
        MemoryOptimizationPass
            .run(ssa, &PassEnv::new(&0u32, &events, context))
            .is_ok()
    );
    events
}

/// A load of read-only image data becomes the constant its bytes encode.
#[test]
fn a_load_of_read_only_data_folds_to_its_bytes() {
    let (mut ssa, loaded) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U32, Vec::new());
    let events = fold(&mut ssa, &TestImage::new());
    assert_eq!(
        definition(&ssa, loaded),
        Some(SsaOp::Const {
            dest: loaded,
            value: ConstValue::U32(0x1234_5678),
        })
    );
    assert_eq!(
        events.count_by_kind().get(&EventKind::ImageLoadFolded),
        Some(&1)
    );
    assert_eq!(ssa.validate(), Ok(()));
}

/// **A cell the host proves holds an address folds to the location, typed as
/// one.** Its bytes say only a number; read as the integer the load was
/// typed, a GOT entry became `0x400000` and the pointer built from it an
/// integer sum. A cell of another width is not the address, and folds to its
/// bytes as before.
#[test]
fn a_cell_holding_an_address_folds_to_the_location_it_names() {
    let mut image = TestImage::new();
    image.addresses.push((0x1000, 8, 0x2000));

    let (mut ssa, loaded) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U64, Vec::new());
    fold(&mut ssa, &image);
    assert_eq!(
        definition(&ssa, loaded),
        Some(SsaOp::Const {
            dest: loaded,
            value: ConstValue::Symbol(0x2000),
        })
    );
    assert_eq!(
        ssa.variable(loaded).map(|variable| *variable.var_type()),
        Some(MockType::Ptr),
        "the destination is retyped as what it now holds"
    );
    assert_eq!(ssa.validate(), Ok(()));

    // A host that keeps facts about its functions is handed what the folded
    // load read: the load is a constant now, and nothing else says the
    // function read the image there.
    let (mut factful, _) =
        load_of::<FactfulTarget>(ConstValue::Symbol(0x1000), MockType::U64, Vec::new());
    fold(&mut factful, &image);
    assert_eq!(factful.facts().image_reads, vec![(0x1000, 8)]);

    let (mut narrow, part) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U32, Vec::new());
    fold(&mut narrow, &image);
    assert_eq!(
        definition(&narrow, part),
        Some(SsaOp::Const {
            dest: part,
            value: ConstValue::U32(0x1234_5678),
        }),
        "half a pointer is not the address"
    );
}

/// The image's byte order decides the value, and a signed type a signed one.
#[test]
fn the_image_byte_order_and_the_type_sign_decide_the_constant() {
    let mut big = TestImage::new();
    big.endianness = Endianness::Big;
    let (mut ssa, loaded) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U32, Vec::new());
    fold(&mut ssa, &big);
    assert_eq!(
        definition(&ssa, loaded),
        Some(SsaOp::Const {
            dest: loaded,
            value: ConstValue::U32(0x7856_3412),
        })
    );

    let (mut ssa, loaded) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1004), MockType::I32, Vec::new());
    fold(&mut ssa, &TestImage::new());
    assert_eq!(
        definition(&ssa, loaded),
        Some(SsaOp::Const {
            dest: loaded,
            value: ConstValue::I32(-1),
        })
    );
}

/// A call before the load changes nothing: a callee cannot write read-only
/// data without faulting, so the bytes are still the image's.
#[test]
fn a_call_does_not_clobber_read_only_data() {
    let call = SsaOp::from(CallData::new(None, Callee::Direct(7u32), Vec::new()));
    let (mut ssa, loaded) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U32, vec![call]);
    fold(&mut ssa, &TestImage::new());
    assert!(matches!(
        definition(&ssa, loaded),
        Some(SsaOp::Const {
            value: ConstValue::U32(0x1234_5678),
            ..
        })
    ));
}

/// Each way a load falls outside the rule leaves it a load.
#[test]
fn a_load_outside_the_rule_stays_a_load() {
    let still_a_load = |constant: ConstValue<MockTarget>, ty: MockType, image: &TestImage| {
        let (mut ssa, loaded) = load_of::<MockTarget>(constant, ty, Vec::new());
        fold(&mut ssa, image);
        matches!(definition(&ssa, loaded), Some(SsaOp::LoadIndirect { .. }))
    };
    let image = TestImage::new();
    assert!(
        still_a_load(ConstValue::Symbol(0x2000), MockType::U32, &image),
        "writable data may have been written before the function ran"
    );
    assert!(
        still_a_load(ConstValue::Symbol(0x1006), MockType::U32, &image),
        "the extent runs past the end of the region"
    );
    assert!(
        still_a_load(ConstValue::Symbol(0x3000), MockType::U32, &image),
        "no region holds it"
    );
    assert!(
        still_a_load(ConstValue::Symbol(0x1000), MockType::Ref, &image),
        "a constant of this type is not stated from bytes"
    );
    assert!(
        !still_a_load(ConstValue::Symbol(0x1000), MockType::F64, &image),
        "a float is: its bytes are the IEEE value its type names"
    );
    let unrelated = TestImage {
        number_offset: None,
        ..TestImage::new()
    };
    assert!(
        still_a_load(ConstValue::U64(0x1000), MockType::U32, &unrelated),
        "an integer address in a space the host cannot relate to its symbols"
    );
    assert!(
        !still_a_load(ConstValue::U64(0x1000), MockType::U32, &image),
        "the same address folds when the host says the spaces coincide"
    );
}

/// **A dead load of a cell the image places goes.** A load is kept when its
/// result is unused only because it may fault, and a cell the loader maps
/// cannot: kept, it held up the branch it sat under. A cell no region holds
/// may fault, and its load stays.
#[test]
fn a_dead_load_of_a_placed_cell_is_removed() {
    let dead_load_of = |address: u32| {
        let (mut ssa, loaded) =
            load_of::<MockTarget>(ConstValue::Symbol(address), MockType::U32, Vec::new());
        if let Some(block) = ssa.block_mut(0)
            && let Some(last) = block.instructions_mut().last_mut()
        {
            last.set_op(SsaOp::returns(None));
        }
        ssa.recompute_uses();
        let events = fold(&mut ssa, &TestImage::new());
        (ssa, loaded, events)
    };
    let (ssa, loaded, events) = dead_load_of(0x2000);
    assert_eq!(definition(&ssa, loaded), None, "writable, but mapped");
    assert_eq!(
        events.count_by_kind().get(&EventKind::InstructionRemoved),
        Some(&1)
    );
    assert_eq!(ssa.validate(), Ok(()));

    let (ssa, loaded, _) = dead_load_of(0x3000);
    assert!(
        matches!(definition(&ssa, loaded), Some(SsaOp::LoadIndirect { .. })),
        "no region holds it, so it may fault"
    );

    // The load is gone and the read is not: a host keeping facts is handed it
    // as it is handed a fold's, so the object stays one the function reads.
    let (mut factful, loaded) =
        load_of::<FactfulTarget>(ConstValue::Symbol(0x2000), MockType::U32, Vec::new());
    if let Some(block) = factful.block_mut(0)
        && let Some(last) = block.instructions_mut().last_mut()
    {
        last.set_op(SsaOp::returns(None));
    }
    factful.recompute_uses();
    fold(&mut factful, &TestImage::new());
    assert_eq!(definition(&factful, loaded), None);
    assert_eq!(factful.facts().image_reads, vec![(0x2000, 4)]);
}

/// **A dead load of a bound import's first word goes.** An import has no
/// address in the image, so no region places it; the host proves the loader
/// binds `__stack_chk_guard`, and a function that never returns loaded it for
/// a canary nothing checks. The controls: the same import the host says
/// nothing of may be null, and so may another import beside a bound one, so
/// both loads stay.
#[test]
fn a_dead_load_of_a_bound_imports_first_word_is_removed() {
    let import = FactfulTarget::IMPORTED + 1;
    let dead_load = |address: ConstValue<FactfulTarget>, bound: bool| {
        let (mut ssa, loaded) = load_of::<FactfulTarget>(address, MockType::U32, Vec::new());
        if let Some(block) = ssa.block_mut(0)
            && let Some(last) = block.instructions_mut().last_mut()
        {
            last.set_op(SsaOp::returns(None));
        }
        if bound {
            ssa.facts_mut().bound_symbols.push(import);
        }
        ssa.recompute_uses();
        fold(&mut ssa, &TestImage::new());
        assert_eq!(ssa.validate(), Ok(()));
        matches!(definition(&ssa, loaded), Some(SsaOp::LoadIndirect { .. }))
    };
    assert!(
        !dead_load(ConstValue::Symbol(import), true),
        "bound: it goes"
    );
    assert!(
        dead_load(ConstValue::Symbol(import), false),
        "control: an import nothing binds may be null"
    );
    assert!(
        dead_load(ConstValue::Symbol(FactfulTarget::IMPORTED + 2), true),
        "control: another import"
    );
}

/// A dead read of a halfword of the table at `0x1000`, at a word index
/// zero-extended to the pointer's width, as x86-64 dispatches: a guard
/// `bound <u index` branching away dominates the read when `bound` is
/// given, the read unguarded otherwise.
fn table_read<T: Target<Type = MockType, SymbolRef = u32>>(
    bound: Option<u32>,
) -> (SsaFunction<T>, SsaVarId) {
    let mut ssa = SsaFunction::new(1, 8);
    let index = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U32,
    );
    let limit = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::U32,
    );
    let wide = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(1, 0),
        MockType::U64,
    );
    let table = ssa.create_variable(
        VariableOrigin::Local(2),
        0,
        DefSite::instruction(1, 1),
        MockType::Ptr,
    );
    let entry = ssa.create_variable(
        VariableOrigin::Local(3),
        0,
        DefSite::instruction(1, 2),
        MockType::Ptr,
    );
    let read = ssa.create_variable(
        VariableOrigin::Local(4),
        0,
        DefSite::instruction(1, 3),
        MockType::U16,
    );
    let mut b0 = SsaBlock::new(0);
    match bound {
        Some(bound) => {
            b0.add_instruction(instr(SsaOp::Const {
                dest: limit,
                value: ConstValue::U32(bound),
            }));
            b0.add_instruction(instr(SsaOp::BranchCmp {
                left: limit,
                right: index,
                cmp: CmpKind::Lt,
                unsigned: true,
                true_target: 2,
                false_target: 1,
            }));
        }
        None => b0.add_instruction(instr(SsaOp::Jump { target: 1 })),
    }
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::IntConv {
        dest: wide,
        operand: index,
        target: MockType::U64,
        overflow_check: false,
        unsigned: true,
    }));
    b1.add_instruction(instr(SsaOp::Const {
        dest: table,
        value: ConstValue::Symbol(0x1000),
    }));
    b1.add_instruction(instr(SsaOp::PtrAdd {
        dest: entry,
        base: table,
        index: Some(wide),
        stride: 2,
        offset: 0,
        result_type: MockType::Ptr,
    }));
    b1.add_instruction(instr(SsaOp::LoadIndirect {
        dest: read,
        addr: entry,
        value_type: MockType::U16,
        access: AccessQualifiers::FLAT,
    }));
    b1.add_instruction(instr(SsaOp::returns(None)));
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b0);
    ssa.add_block(b1);
    ssa.add_block(b2);
    ssa.recompute_uses();
    (ssa, read)
}

/// **A dead read of a table at a guarded index goes.** A jump table's entry
/// is read at the dispatch's index, and once a switch on the index replaces
/// the jump nothing reads the entry; every byte the guard lets the index
/// reach lies in the image, so the read cannot fault, and the host is handed
/// the bytes it read, the whole table. Unguarded, or guarded past the
/// image's bytes, the read may fault, and it stays.
#[test]
fn a_dead_table_read_at_a_guarded_index_is_removed() {
    let (mut ssa, read) = table_read::<MockTarget>(Some(3));
    fold(&mut ssa, &TestImage::new());
    assert_eq!(
        definition(&ssa, read),
        None,
        "four entries, all in the image"
    );
    assert_eq!(ssa.validate(), Ok(()));

    let (mut factful, read) = table_read::<FactfulTarget>(Some(3));
    fold(&mut factful, &TestImage::new());
    assert_eq!(definition(&factful, read), None);
    assert_eq!(factful.facts().image_reads, vec![(0x1000, 8)]);

    for (bound, why) in [
        (None, "unguarded"),
        (Some(4), "a fifth entry past the image's bytes"),
    ] {
        let (mut ssa, read) = table_read::<MockTarget>(bound);
        fold(&mut ssa, &TestImage::new());
        assert!(
            matches!(definition(&ssa, read), Some(SsaOp::LoadIndirect { .. })),
            "{why}"
        );
    }
}

/// **A store and a load through two folds of one literal-pool word meet.**
/// Each copy of the pool word folds to the same address, but the forward was
/// planned before the fold, when the two addresses were two loads: `uidget`
/// stored `0` to a cell and read it back. Folding runs to its own fixpoint
/// first, so one run forwards the store.
#[test]
fn a_store_through_one_folded_pool_word_reaches_a_load_through_another() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 8);
    let mut values = Vec::new();
    for (at, ty) in [
        MockType::Ptr,
        MockType::U32,
        MockType::Ptr,
        MockType::U32,
        MockType::Ptr,
        MockType::U32,
        MockType::Ptr,
        MockType::U32,
    ]
    .into_iter()
    .enumerate()
    {
        values.push(ssa.create_variable(
            VariableOrigin::Local(u16::try_from(at).unwrap_or(u16::MAX)),
            0,
            DefSite::instruction(0, at),
            ty,
        ));
    }
    let [
        pool,
        first,
        stored_at,
        value,
        pool_again,
        second,
        loaded_at,
        loaded,
    ] = values[..]
    else {
        unreachable!("eight values")
    };
    let mut b0 = SsaBlock::new(0);
    for (pool, address, at) in [(pool, first, stored_at), (pool_again, second, loaded_at)] {
        b0.add_instruction(instr(SsaOp::Const {
            dest: pool,
            value: ConstValue::Symbol(0x1000),
        }));
        b0.add_instruction(instr(SsaOp::LoadIndirect {
            dest: address,
            addr: pool,
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        }));
        b0.add_instruction(instr(SsaOp::PtrAdd {
            dest: at,
            base: address,
            index: None,
            stride: 0,
            offset: 2,
            result_type: MockType::Ptr,
        }));
        if at == stored_at {
            b0.add_instruction(instr(SsaOp::Const {
                dest: value,
                value: ConstValue::U32(7),
            }));
            b0.add_instruction(instr(SsaOp::StoreIndirect {
                addr: stored_at,
                value,
                value_type: MockType::U32,
                access: AccessQualifiers::FLAT,
            }));
        }
    }
    b0.add_instruction(instr(SsaOp::LoadIndirect {
        dest: loaded,
        addr: loaded_at,
        value_type: MockType::U32,
        access: AccessQualifiers::FLAT,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    // The pool word at `0x1000` holds `0x2000`, the writable cell's address.
    let image = TestImage {
        rodata: vec![0x00, 0x20, 0x00, 0x00, 0, 0, 0, 0],
        ..TestImage::new()
    };
    let events = fold(&mut ssa, &image);
    let returned = ssa.blocks().iter().find_map(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.op() {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            })
    });
    assert_eq!(returned, Some(value), "the load reads what was stored");
    assert_eq!(
        events.count_by_kind().get(&EventKind::LoadForwarded),
        Some(&1)
    );
    assert_eq!(ssa.validate(), Ok(()));
}

/// **A value stored at the width a load reads is forwarded whatever its own
/// type.** `mov r2, #0; str r2, [cell]` stores an integer the load reads back
/// as `u32`; the forward was refused for the two types, so `uidget` stored
/// `0` and read it back. The load becomes the stored value reinterpreted. A
/// value wider than the store, which the store truncates, is forwarded as its
/// low bits (BOUNDARIES D69).
#[test]
fn a_stored_value_of_another_type_is_forwarded_at_one_width() {
    let forwarded = |held: MockType, value: ConstValue<MockTarget>, access: MockType| {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 8);
        let mut values = Vec::new();
        for (at, ty) in [MockType::Ptr, held, access].into_iter().enumerate() {
            let at_instr = if at == 2 { 3 } else { at };
            values.push(ssa.create_variable(
                VariableOrigin::Local(u16::try_from(at).unwrap_or(u16::MAX)),
                0,
                DefSite::instruction(0, at_instr),
                ty,
            ));
        }
        let [cell, stored, loaded] = values[..] else {
            unreachable!("three values")
        };
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: cell,
            value: ConstValue::Symbol(0x2000),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: stored,
            value,
        }));
        b0.add_instruction(instr(SsaOp::StoreIndirect {
            addr: cell,
            value: stored,
            value_type: access,
            access: AccessQualifiers::FLAT,
        }));
        b0.add_instruction(instr(SsaOp::LoadIndirect {
            dest: loaded,
            addr: cell,
            value_type: access,
            access: AccessQualifiers::FLAT,
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(loaded))));
        ssa.add_block(b0);
        ssa.recompute_uses();
        fold(&mut ssa, &TestImage::new());
        assert_eq!(ssa.validate(), Ok(()));
        (definition(&ssa, loaded), stored, loaded)
    };
    let (definition, stored, loaded) = forwarded(MockType::I32, ConstValue::I32(0), MockType::U32);
    assert_eq!(
        definition,
        Some(SsaOp::IntConv {
            dest: loaded,
            operand: stored,
            target: MockType::U32,
            overflow_check: false,
            unsigned: true,
        })
    );
    // Wider than the store, the value is truncated by it: the load reads the
    // value's low 32 bits, never the value itself.
    let (definition, stored, loaded) = forwarded(MockType::I64, ConstValue::I64(0), MockType::U32);
    assert!(
        !matches!(definition, Some(SsaOp::LoadIndirect { .. })),
        "a truncating store's low bits are the load's"
    );
    assert_ne!(
        definition,
        Some(SsaOp::IntConv {
            dest: loaded,
            operand: stored,
            target: MockType::U32,
            overflow_check: false,
            unsigned: true,
        }),
        "and are read through a narrowing, not as the stored value reinterpreted"
    );
}

/// Folding stops at the byte budget, and says so.
#[test]
fn folding_stops_at_the_byte_budget_and_reports_it() {
    let (mut ssa, loaded) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U32, Vec::new());
    let events = EventLog::new();
    let image = TestImage::new();
    let context = AnalysisContext::new(MACHINE)
        .with_image(&image)
        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ImageBytes, 3));
    assert!(
        MemoryOptimizationPass
            .run(&mut ssa, &PassEnv::new(&0u32, &events, context))
            .is_ok()
    );
    assert!(matches!(
        definition(&ssa, loaded),
        Some(SsaOp::LoadIndirect { .. })
    ));
    assert_eq!(
        events.count_by_kind().get(&EventKind::BudgetExhausted),
        Some(&1)
    );
}

/// With no image nothing folds: the program's memory is unknown.
#[test]
fn without_an_image_nothing_folds() {
    let (mut ssa, loaded) =
        load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U32, Vec::new());
    assert!(
        MemoryOptimizationPass
            .run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    assert!(matches!(
        definition(&ssa, loaded),
        Some(SsaOp::LoadIndirect { .. })
    ));
}

/// Memory SSA places each location in its region, and gives a read-only image
/// cell the image as its only definition.
#[test]
fn memory_ssa_places_locations_in_their_regions() {
    let image = TestImage::new();
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    let cell = |base: AddressBase, address: i64| {
        MemoryLocation::<MockTarget>::Indirect(IndirectLocation::new(
            base,
            None,
            0,
            address * 8,
            Some(32),
            None,
            PointerSize::Bit64,
        ))
    };
    assert_eq!(
        cell(AddressBase::Symbol, 0x1000).region(&context, None),
        MemoryRegion::ImageReadOnly
    );
    assert_eq!(
        cell(AddressBase::Symbol, 0x2000).region(&context, None),
        MemoryRegion::ImageWritable
    );
    let sp = SsaVarId::from_index(3);
    assert_eq!(
        cell(AddressBase::Value(sp), -8).region(&context, Some(sp)),
        MemoryRegion::Frame
    );
    assert_eq!(
        cell(AddressBase::Symbol, 0x1000).region(&AnalysisContext::new(MACHINE), None),
        MemoryRegion::Unknown,
        "no image, no region"
    );

    let (ssa, _) = load_of::<MockTarget>(ConstValue::Symbol(0x1000), MockType::U32, Vec::new());
    let mem_ssa = MemorySsa::build(&ssa, &context);
    let location = cell(AddressBase::Symbol, 0x1000);
    assert!(mem_ssa.is_immutable(&location));
    let version = mem_ssa.version_at_entry(&location, 0);
    assert!(version.is_some(), "the location is tracked");
    let Some(version) = version else {
        return;
    };
    assert_eq!(
        mem_ssa.definition(&MemoryVersion::new(location, version)),
        Some(MemoryDefSite::Image)
    );
}

/// **A composed load reads each piece's surviving value.** The low piece is
/// `x`, itself a load the store before it serves: the forward removes `x`'s
/// load and hands its readers the stored value, and the composition, built
/// after the forwards, has to read that value too. Built from `x` it read a
/// value nothing defines any more, the verifier rejected the pass, and its
/// whole group was discarded on every iteration.
#[test]
fn a_composed_load_reads_the_forwarded_value_of_a_piece() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 12);
    let mut values = Vec::new();
    for (at, ty) in [
        MockType::Ptr,
        MockType::U32,
        MockType::U32,
        MockType::Ptr,
        MockType::Ptr,
        MockType::U32,
        MockType::U64,
    ]
    .into_iter()
    .enumerate()
    {
        values.push(ssa.create_variable(
            VariableOrigin::Local(u16::try_from(at).unwrap_or(u16::MAX)),
            0,
            DefSite::instruction(0, at),
            ty,
        ));
    }
    let [base, stored, high, low_at, high_at, x, loaded] = values[..] else {
        unreachable!("seven values")
    };
    let mut b0 = SsaBlock::new(0);
    for op in [
        SsaOp::Const {
            dest: base,
            value: ConstValue::Symbol(0x2000),
        },
        SsaOp::Const {
            dest: stored,
            value: ConstValue::U32(0x1111_1111),
        },
        SsaOp::Const {
            dest: high,
            value: ConstValue::U32(0x2222_2222),
        },
        SsaOp::PtrAdd {
            dest: low_at,
            base,
            index: None,
            stride: 0,
            offset: 8,
            result_type: MockType::Ptr,
        },
        SsaOp::PtrAdd {
            dest: high_at,
            base,
            index: None,
            stride: 0,
            offset: 12,
            result_type: MockType::Ptr,
        },
        SsaOp::StoreIndirect {
            addr: base,
            value: stored,
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::LoadIndirect {
            dest: x,
            addr: base,
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::StoreIndirect {
            addr: low_at,
            value: x,
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::StoreIndirect {
            addr: high_at,
            value: high,
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::LoadIndirect {
            dest: loaded,
            addr: low_at,
            value_type: MockType::U64,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::returns(Some(loaded)),
    ] {
        b0.add_instruction(instr(op));
    }
    ssa.add_block(b0);
    ssa.recompute_uses();
    fold(&mut ssa, &TestImage::new());
    assert_eq!(
        ssa.validate(),
        Ok(()),
        "no piece reads the removed load of `x`"
    );
    assert!(
        !matches!(definition(&ssa, loaded), Some(SsaOp::LoadIndirect { .. })),
        "the wide load reads the two stores"
    );
    assert!(
        !matches!(definition(&ssa, x), Some(SsaOp::LoadIndirect { .. })),
        "`x` is forwarded"
    );
}

/// The load of `ssa`'s local after two 32-bit stores of entry values typed
/// `piece` tile it, `gap` bytes apart, and the shifts and ors combining them.
fn tiled(piece: MockType, gap: i64) -> (Option<SsaOp<MockTarget>>, usize) {
    let mut ssa = SsaFunction::<MockTarget>::new(2, 8);
    let low = ssa.create_variable(VariableOrigin::Argument(0), 0, DefSite::entry(), piece);
    let high = ssa.create_variable(VariableOrigin::Argument(1), 0, DefSite::entry(), piece);
    let mut values = Vec::new();
    for (at, ty) in [MockType::Ptr, MockType::Ptr, MockType::U64]
        .into_iter()
        .enumerate()
    {
        values.push(ssa.create_variable(
            VariableOrigin::Local(u16::try_from(at).unwrap_or(u16::MAX)),
            0,
            DefSite::instruction(0, at),
            ty,
        ));
    }
    let [base, high_at, loaded] = values[..] else {
        unreachable!("three values")
    };
    let mut b0 = SsaBlock::new(0);
    for op in [
        SsaOp::Const {
            dest: base,
            value: ConstValue::Symbol(0x2000),
        },
        SsaOp::PtrAdd {
            dest: high_at,
            base,
            index: None,
            stride: 0,
            offset: gap.saturating_add(4),
            result_type: MockType::Ptr,
        },
        SsaOp::StoreIndirect {
            addr: base,
            value: low,
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::StoreIndirect {
            addr: high_at,
            value: high,
            value_type: MockType::U32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::LoadIndirect {
            dest: loaded,
            addr: base,
            value_type: MockType::U64,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::returns(Some(loaded)),
    ] {
        b0.add_instruction(instr(op));
    }
    ssa.add_block(b0);
    ssa.recompute_uses();
    fold(&mut ssa, &TestImage::new());
    assert_eq!(ssa.validate(), Ok(()));
    let combining = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter(|instruction| matches!(instruction.op(), SsaOp::Shl { .. } | SsaOp::Or { .. }))
        .count();
    (definition(&ssa, loaded), combining)
}

/// **A load several narrower stores tile reads what they compose.** `-O0` x86
/// copies a `double` argument into a local with two 32-bit moves and reads the
/// local back with one 64-bit load; nothing stored the whole, so nothing was
/// forwarded, and the load read memory the analysis could not follow. On a
/// little-endian image the lower address is the low half. The control leaves
/// a gap between the pieces, which compose nothing.
#[test]
fn a_load_two_stores_tile_reads_their_composition() {
    let (definition, combining) = tiled(MockType::U32, 0);
    assert!(
        !matches!(definition, Some(SsaOp::LoadIndirect { .. })),
        "the load reads the two stores: {definition:?}"
    );
    assert_eq!(combining, 2, "the high half shifted into place, then or-ed");

    let (definition, combining) = tiled(MockType::U32, 4);
    assert!(matches!(definition, Some(SsaOp::LoadIndirect { .. })));
    assert_eq!(combining, 0);
}

/// **A stored value of no stated type is the bits its store writes.** A move
/// between memory and a register states a width and nothing more, so the
/// halves it copies are untyped; each store still says how many bits it
/// wrote, which is all a composition reads. The control leaves a gap.
#[test]
fn untyped_stores_that_tile_a_load_compose_at_their_store_widths() {
    let (definition, combining) = tiled(MockType::Unknown, 0);
    assert!(
        !matches!(definition, Some(SsaOp::LoadIndirect { .. })),
        "the load reads the two stores: {definition:?}"
    );
    assert_eq!(combining, 2, "the high half shifted into place, then or-ed");

    let (definition, combining) = tiled(MockType::Unknown, 4);
    assert!(matches!(definition, Some(SsaOp::LoadIndirect { .. })));
    assert_eq!(combining, 0);
}

/// One block of `ops` over values `0..types.len()` typed `types`, returning
/// `returned`.
fn function_of(
    types: &[MockType],
    ops: Vec<SsaOp<MockTarget>>,
    returned: usize,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(1, types.len());
    for (index, ty) in types.iter().enumerate() {
        ssa.create_variable(
            VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
            0,
            DefSite::instruction(0, index),
            *ty,
        );
    }
    let mut b0 = SsaBlock::new(0);
    for op in ops {
        b0.add_instruction(instr(op));
    }
    b0.add_instruction(instr(SsaOp::returns(Some(SsaVarId::from_index(returned)))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// The constant `value` of `ssa` folds to after the pass.
fn folded(ssa: &SsaFunction<MockTarget>, value: usize) -> Option<ConstValue<MockTarget>> {
    ConstEvaluator::new(ssa, MACHINE).evaluate_var(SsaVarId::from_index(value))
}

/// The constant the value `value` of `ssa` is converted from after the pass,
/// through its conversions: the mock target converts no constant, which is
/// the host's to do.
fn converted(ssa: &SsaFunction<MockTarget>, value: usize) -> Option<ConstValue<MockTarget>> {
    let mut source = SsaVarId::from_index(value);
    while let Some(SsaOp::IntConv { operand, .. } | SsaOp::Copy { src: operand, .. }) =
        definition(ssa, source)
    {
        source = operand;
    }
    ConstEvaluator::new(ssa, MACHINE).evaluate_var(source)
}

/// Words `0x03020100`, `0x07060504`, ... of `count` words of counting bytes.
fn counting(count: usize) -> Vec<u8> {
    (0..count.saturating_mul(4))
        .map(|byte| u8::try_from(byte).unwrap_or(0))
        .collect()
}

/// The vector of `words`, four 32-bit lanes.
fn words(values: [u32; 4]) -> ConstValue<MockTarget> {
    let vector = VectorConst::from_lanes(32, values.map(u128::from));
    assert!(vector.is_some(), "four words are a vector");
    vector.map_or(ConstValue::Null, ConstValue::Vector)
}

/// **A vector read of read-only data folds to its bytes**: a whole vector,
/// one element broadcast to every lane, and two interleaved segments, each
/// lane `j` of segment `m` element `2j + m` (AArch64 `ld2`, ARM `vld2`).
#[test]
fn a_vector_read_of_read_only_data_folds_to_its_bytes() {
    let image = TestImage {
        rodata: counting(8),
        ..TestImage::new()
    };
    let address = || SsaOp::Const {
        dest: SsaVarId::from_index(0),
        value: ConstValue::Symbol(0x1000),
    };
    let v = SsaVarId::from_index;
    let mut whole = function_of(
        &[MockType::Ptr, MockType::V4I32],
        vec![
            address(),
            SsaOp::VectorLoad {
                dest: v(1),
                addr: v(0),
                vector_type: MockType::V4I32,
                access: AccessQualifiers::FLAT,
            },
        ],
        1,
    );
    fold(&mut whole, &image);
    assert_eq!(
        folded(&whole, 1),
        Some(words([0x0302_0100, 0x0706_0504, 0x0b0a_0908, 0x0f0e_0d0c]))
    );
    let mut broadcast = function_of(
        &[MockType::Ptr, MockType::V4I32],
        vec![
            address(),
            SsaOp::VectorBroadcastLoad {
                dest: v(1),
                addr: v(0),
                vector_type: MockType::V4I32,
                access: AccessQualifiers::FLAT,
            },
        ],
        1,
    );
    fold(&mut broadcast, &image);
    assert_eq!(folded(&broadcast, 1), Some(words([0x0302_0100; 4])));
    let mut segments = function_of(
        &[MockType::Ptr, MockType::V4I32, MockType::V4I32],
        vec![
            address(),
            SsaOp::VectorSegmentLoad {
                dests: vec![v(1), v(2)],
                base: v(0),
                mask: None,
                vector_type: MockType::V4I32,
                segments: 2,
                layout: VectorSegmentLayout::Interleaved,
                access: AccessQualifiers::FLAT,
            },
        ],
        2,
    );
    fold(&mut segments, &image);
    assert_eq!(
        folded(&segments, 1),
        Some(words([0x0302_0100, 0x0b0a_0908, 0x1312_1110, 0x1b1a_1918]))
    );
    assert_eq!(
        folded(&segments, 2),
        Some(words([0x0706_0504, 0x0f0e_0d0c, 0x1716_1514, 0x1f1e_1d1c]))
    );
    assert_eq!(segments.validate(), Ok(()));
    // A segment load past what the region holds stays.
    let short = TestImage {
        rodata: counting(6),
        ..TestImage::new()
    };
    let mut past = function_of(
        &[MockType::Ptr, MockType::V4I32, MockType::V4I32],
        vec![
            address(),
            SsaOp::VectorSegmentLoad {
                dests: vec![v(1), v(2)],
                base: v(0),
                mask: None,
                vector_type: MockType::V4I32,
                segments: 2,
                layout: VectorSegmentLayout::Interleaved,
                access: AccessQualifiers::FLAT,
            },
        ],
        2,
    );
    fold(&mut past, &short);
    assert_eq!(folded(&past, 1), None);
}

/// **A load of part of a stored value reads that part**: a word of a stored
/// vector is its lane (ARM `vst1.32 {d0, d1}` then `ldr r0, [r4, #8]`), a
/// word of two interleaved segments the lane of the segment it falls in
/// (`vst2.32`), the high half of a stored doubleword its shifted bits; and a
/// vector store no longer hides a store to other memory from a load of it.
#[test]
fn a_load_of_part_of_a_stored_value_reads_that_part() {
    let v = SsaVarId::from_index;
    let base = || SsaOp::LoadArg {
        dest: v(0),
        arg_index: 0,
    };
    let at = |dest: usize, offset: i64| SsaOp::PtrAdd {
        dest: v(dest),
        base: v(0),
        index: None,
        stride: 1,
        offset,
        result_type: MockType::Ptr,
    };
    let mut vector = function_of(
        &[MockType::Ptr, MockType::V4I32, MockType::Ptr, MockType::U32],
        vec![
            base(),
            SsaOp::Const {
                dest: v(1),
                value: words([10, 11, 12, 13]),
            },
            SsaOp::VectorStore {
                addr: v(0),
                value: v(1),
                vector_type: MockType::V4I32,
                access: AccessQualifiers::FLAT,
            },
            at(2, 8),
            SsaOp::LoadIndirect {
                dest: v(3),
                addr: v(2),
                value_type: MockType::U32,
                access: AccessQualifiers::FLAT,
            },
        ],
        3,
    );
    fold(&mut vector, &TestImage::new());
    assert_eq!(converted(&vector, 3), Some(ConstValue::U32(12)), "{vector}");
    assert_eq!(vector.validate(), Ok(()));
    let mut doubleword = function_of(
        &[MockType::Ptr, MockType::U64, MockType::Ptr, MockType::U32],
        vec![
            base(),
            SsaOp::Const {
                dest: v(1),
                value: ConstValue::U64(0x1111_2222_3333_4444),
            },
            SsaOp::StoreIndirect {
                addr: v(0),
                value: v(1),
                value_type: MockType::U64,
                access: AccessQualifiers::FLAT,
            },
            at(2, 4),
            SsaOp::LoadIndirect {
                dest: v(3),
                addr: v(2),
                value_type: MockType::U32,
                access: AccessQualifiers::FLAT,
            },
        ],
        3,
    );
    // Two interleaved segments: word 3 is lane 1 of the second.
    let mut interleaved = function_of(
        &[
            MockType::Ptr,
            MockType::V4I32,
            MockType::V4I32,
            MockType::Ptr,
            MockType::U32,
        ],
        vec![
            base(),
            SsaOp::Const {
                dest: v(1),
                value: words([10, 11, 12, 13]),
            },
            SsaOp::Const {
                dest: v(2),
                value: words([20, 21, 22, 23]),
            },
            SsaOp::VectorSegmentStore {
                base: v(0),
                values: vec![v(1), v(2)],
                mask: None,
                vector_type: MockType::V4I32,
                segments: 2,
                layout: VectorSegmentLayout::Interleaved,
                access: AccessQualifiers::FLAT,
            },
            at(3, 12),
            SsaOp::LoadIndirect {
                dest: v(4),
                addr: v(3),
                value_type: MockType::U32,
                access: AccessQualifiers::FLAT,
            },
        ],
        4,
    );
    fold(&mut interleaved, &TestImage::new());
    assert_eq!(
        converted(&interleaved, 4),
        Some(ConstValue::U32(21)),
        "{interleaved}"
    );
    assert_eq!(interleaved.validate(), Ok(()));
    fold(&mut doubleword, &TestImage::new());
    assert_eq!(
        converted(&doubleword, 3),
        Some(ConstValue::U64(0x1111_2222)),
        "{doubleword}"
    );
    let mut beside = function_of(
        &[
            MockType::Ptr,
            MockType::U32,
            MockType::Ptr,
            MockType::V4I32,
            MockType::U32,
        ],
        vec![
            base(),
            SsaOp::Const {
                dest: v(1),
                value: ConstValue::U32(7),
            },
            SsaOp::StoreIndirect {
                addr: v(0),
                value: v(1),
                value_type: MockType::U32,
                access: AccessQualifiers::FLAT,
            },
            at(2, 16),
            SsaOp::Const {
                dest: v(3),
                value: words([1, 2, 3, 4]),
            },
            SsaOp::VectorStore {
                addr: v(2),
                value: v(3),
                vector_type: MockType::V4I32,
                access: AccessQualifiers::FLAT,
            },
            SsaOp::LoadIndirect {
                dest: v(4),
                addr: v(0),
                value_type: MockType::U32,
                access: AccessQualifiers::FLAT,
            },
        ],
        4,
    );
    fold(&mut beside, &TestImage::new());
    let returned = beside.blocks().iter().find_map(|block| {
        block
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.op() {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            })
    });
    assert_eq!(
        returned.and_then(|value| ConstEvaluator::new(&beside, MACHINE).evaluate_var(value)),
        Some(ConstValue::U32(7)),
        "{beside}"
    );
}

/// **A whole read of lanes wider than any one integer folds to its bits.** A
/// 512-bit read typed two 256-bit lanes, as a ZMM register moved whole is
/// typed one 512-bit lane, folds to the 64 bytes it reads: read again in
/// 32-bit lanes, the counting words.
#[test]
fn a_read_of_lanes_wider_than_an_integer_folds_to_its_bits() {
    let image = TestImage {
        rodata: counting(16),
        ..TestImage::new()
    };
    let v = SsaVarId::from_index;
    let mut wide = function_of(
        &[MockType::Ptr, MockType::V2I256],
        vec![
            SsaOp::Const {
                dest: v(0),
                value: ConstValue::Symbol(0x1000),
            },
            SsaOp::VectorLoad {
                dest: v(1),
                addr: v(0),
                vector_type: MockType::V2I256,
                access: AccessQualifiers::FLAT,
            },
        ],
        1,
    );
    fold(&mut wide, &image);
    let read = folded(&wide, 1);
    assert!(
        matches!(read, Some(ConstValue::Vector(_))),
        "the wide read folds: {read:?}"
    );
    let Some(ConstValue::Vector(vector)) = read else {
        return;
    };
    let words: Option<Vec<u128>> = vector
        .relaned(32)
        .map(|lanes| lanes.lane_values().collect());
    let bytes = counting(16);
    let expected: Vec<u128> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u128::from(u32::from_le_bytes(*word)))
        .collect();
    assert_eq!(words, Some(expected), "sixteen words");
}

/// **A masked read of read-only data folds to the lanes its mask enables.**
/// Under the mask `0b0101`, lanes 0 and 2 are the image's words at their own
/// places and lanes 1 and 3 the passthrough's, or zero where the read zeroes;
/// an expanding read under `0b1010` takes the first two words into lanes 1
/// and 3. A mask nothing states keeps the read.
#[test]
fn a_masked_read_of_read_only_data_folds_to_the_lanes_it_enables() {
    let image = TestImage {
        rodata: counting(4),
        ..TestImage::new()
    };
    let v = SsaVarId::from_index;
    let kept = 0xaaaa_aaaa;
    let read = |op: SsaOp<MockTarget>, mask: Option<u8>| {
        let mut ops = vec![
            SsaOp::Const {
                dest: v(0),
                value: ConstValue::Symbol(0x1000),
            },
            SsaOp::Const {
                dest: v(3),
                value: words([kept; 4]),
            },
        ];
        if let Some(mask) = mask {
            ops.push(SsaOp::Const {
                dest: v(2),
                value: ConstValue::U8(mask),
            });
        }
        ops.push(op);
        let mut ssa = function_of(
            &[
                MockType::Ptr,
                MockType::V4I32,
                MockType::U8,
                MockType::V4I32,
            ],
            ops,
            1,
        );
        fold(&mut ssa, &image);
        folded(&ssa, 1)
    };
    let masked = |mode: VectorMaskMode, passthrough: Option<SsaVarId>| SsaOp::VectorMaskedLoad {
        dest: v(1),
        addr: v(0),
        mask: v(2),
        passthrough,
        vector_type: MockType::V4I32,
        mode,
        access: AccessQualifiers::FLAT,
    };
    assert_eq!(
        read(masked(VectorMaskMode::Merge, Some(v(3))), Some(0b0101)),
        Some(words([0x0302_0100, kept, 0x0b0a_0908, kept]))
    );
    assert_eq!(
        read(masked(VectorMaskMode::Zero, None), Some(0b0101)),
        Some(words([0x0302_0100, 0, 0x0b0a_0908, 0]))
    );
    let expand = SsaOp::VectorPackLoad {
        dest: v(1),
        addr: v(0),
        mask: v(2),
        passthrough: Some(v(3)),
        vector_type: MockType::V4I32,
        element_bits: 32,
        kind: VectorPackKind::Expand,
        mode: VectorMaskMode::Merge,
        access: AccessQualifiers::FLAT,
    };
    assert_eq!(
        read(expand, Some(0b1010)),
        Some(words([kept, 0x0302_0100, kept, 0x0706_0504]))
    );
    assert_eq!(
        read(masked(VectorMaskMode::Merge, Some(v(3))), None),
        None,
        "a mask nothing states"
    );
}

/// **A packing access reads or writes within its vector at its address.**
/// An expanding load is a read, and a compressing store a write, of the
/// vector's extent at the address, as their masked forms are: a location
/// memory SSA tracks, not an unknown one every access may alias.
#[test]
fn a_packing_access_is_placed_at_its_address() {
    let image = TestImage::new();
    let context = AnalysisContext::new(MACHINE).with_image(&image);
    let v = SsaVarId::from_index;
    let ssa = function_of(
        &[
            MockType::Ptr,
            MockType::V4I32,
            MockType::U8,
            MockType::V4I32,
        ],
        vec![
            SsaOp::Const {
                dest: v(0),
                value: ConstValue::Symbol(0x2000),
            },
            SsaOp::Const {
                dest: v(2),
                value: ConstValue::U8(0b0011),
            },
            SsaOp::VectorPackLoad {
                dest: v(1),
                addr: v(0),
                mask: v(2),
                passthrough: None,
                vector_type: MockType::V4I32,
                element_bits: 32,
                kind: VectorPackKind::Expand,
                mode: VectorMaskMode::Zero,
                access: AccessQualifiers::FLAT,
            },
            SsaOp::VectorPackStore {
                addr: v(0),
                value: v(1),
                mask: v(2),
                vector_type: MockType::V4I32,
                element_bits: 32,
                kind: VectorPackKind::Compress,
                access: AccessQualifiers::FLAT,
            },
        ],
        1,
    );
    let mem_ssa = MemorySsa::build(&ssa, &context);
    let placed = |instr: usize| {
        mem_ssa
            .operations()
            .iter()
            .filter(|op| op.instr() == instr)
            .map(|op| match op {
                MemoryOp::Read { location, .. }
                | MemoryOp::Load { location, .. }
                | MemoryOp::ReadWrite { location, .. } => {
                    matches!(location, MemoryLocation::Indirect(_))
                }
                _ => false,
            })
            .collect::<Vec<bool>>()
    };
    assert_eq!(placed(2), [true], "the expanding load");
    assert_eq!(placed(3), [true], "the compressing store");
}
