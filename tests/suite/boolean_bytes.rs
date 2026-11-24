//! Explicit Boolean memory representations and independent byte-codec oracles.

use std::process;

use analyssa::{
    PointerSize, Target,
    analysis::{
        Machine, SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{StaticExecution, Value},
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        ConstValue, SsaOp,
        exception::HandlerKind,
        function::{SsaDefSpec, SsaFunctionBuilder},
    },
    target::{BooleanStorage, Endianness, TypeLayout},
};

/// A logical type whose memory representation is independently attested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MemoryType {
    /// A placeholder without a value interpretation.
    Unknown,
    /// A Boolean with a requested canonical storage extent.
    Boolean(u8),
    /// A signed one-bit integer without a declared byte representation.
    SignedBit,
    /// A numeric pointer with a machine-specific width.
    Pointer(u32),
    /// An unsigned storage byte.
    Byte,
}

/// A writable image whose bytes are established only by the executed stores.
struct BlockFillImage;

impl ImageData for BlockFillImage {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
    fn region(&self, address: u64) -> Option<ImageRegion> {
        (0x1000..0x1100).contains(&address).then_some(ImageRegion {
            start: 0x1000,
            end: 0x1100,
            mutability: Mutability::Writable,
        })
    }
    fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
        None
    }
}

/// A flat host with Boolean storage declarations from zero through sixteen bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StorageTarget;

impl Target for StorageTarget {
    type TypeRef = ();
    type MethodRef = ();
    type FieldRef = ();
    type SigRef = ();
    type SymbolRef = ();
    type ExceptionKind = ();
    type Type = MemoryType;
    type OriginalInstruction = ();
    type LocalSignature = ();
    type VariableFacts = ();
    type InstructionFacts = ();
    type FunctionFacts = ();
    type Capability = ();

    fn ptr_bytes(&self) -> u32 {
        8
    }
    fn synthetic_instruction() {}
    fn unknown_type() -> MemoryType {
        MemoryType::Unknown
    }
    fn is_boolean(ty: &MemoryType) -> bool {
        matches!(ty, MemoryType::Boolean(_))
    }
    fn boolean_storage(ty: &MemoryType) -> Option<BooleanStorage> {
        match ty {
            MemoryType::Boolean(width) => BooleanStorage::new(*width),
            _ => None,
        }
    }
    fn is_integer(ty: &MemoryType) -> bool {
        matches!(
            ty,
            MemoryType::Boolean(_) | MemoryType::SignedBit | MemoryType::Byte
        )
    }
    fn is_floating(_: &MemoryType) -> bool {
        false
    }
    fn is_signed(ty: &MemoryType) -> bool {
        matches!(ty, MemoryType::SignedBit)
    }
    fn is_pointer(ty: &MemoryType) -> bool {
        matches!(ty, MemoryType::Pointer(_))
    }
    fn is_reference(_: &MemoryType) -> bool {
        false
    }
    fn is_unknown(ty: &MemoryType) -> bool {
        matches!(ty, MemoryType::Unknown)
    }
    fn bit_width(ty: &MemoryType) -> Option<u32> {
        match ty {
            MemoryType::Boolean(_) | MemoryType::SignedBit => Some(1),
            MemoryType::Byte => Some(8),
            MemoryType::Pointer(bits) => Some(*bits),
            MemoryType::Unknown => None,
        }
    }
    fn type_layout(ty: &MemoryType, ptr_size: PointerSize) -> Option<TypeLayout> {
        Some(TypeLayout::of::<Self>(ty, ptr_size))
    }
    fn instruction_mnemonic(_: &()) -> &'static str {
        "test"
    }
    fn instruction_rva(_: &()) -> u64 {
        0
    }
    fn handler_kind(_: &()) -> HandlerKind {
        HandlerKind::Catch
    }
}

/// Requires a fixture value whose absence indicates an invalid test construction.
fn required<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

/// Logical width remains one bit across every valid or refused storage declaration.
#[test]
fn boolean_value_width_and_storage_extent_are_independent() {
    for width in 0..=17 {
        let ty = MemoryType::Boolean(width);
        let layout = TypeLayout::of::<StorageTarget>(&ty, PointerSize::Bit64);
        assert_eq!(layout.bit_width(), Some(1));
        assert!(layout.is_boolean());
        let expected = (1..=16).contains(&width).then_some(usize::from(width));
        for ptr_size in [PointerSize::Bit32, PointerSize::Bit64, PointerSize::Bit128] {
            assert_eq!(layout.storage_bytes(ptr_size), expected);
        }
    }
    assert_eq!(BooleanStorage::new(0), None);
    assert_eq!(BooleanStorage::new(17), None);
    assert_eq!(BooleanStorage::new(u8::MAX), None);
    assert_eq!(BooleanStorage::BYTE.bytes(), 1);
    let integer = TypeLayout::of::<StorageTarget>(&MemoryType::SignedBit, PointerSize::Bit64);
    assert_eq!(integer.bit_width(), Some(1));
    assert!(!integer.is_boolean());
    assert_eq!(integer.storage_bytes(PointerSize::Bit64), None);
}

/// Owned typed encoding and both decoding views agree with independently constructed bytes.
#[test]
fn canonical_boolean_bytes_round_trip_across_extents_orders_and_carriers() {
    for width in 1..=16 {
        let ty = MemoryType::Boolean(width);
        for ptr_size in [PointerSize::Bit32, PointerSize::Bit64, PointerSize::Bit128] {
            let layout = TypeLayout::of::<StorageTarget>(&ty, ptr_size);
            for order in [Endianness::Little, Endianness::Big] {
                for value in [false, true] {
                    let mut expected = vec![0; usize::from(width)];
                    let position = match order {
                        Endianness::Little => 0,
                        Endianness::Big => required(expected.len().checked_sub(1)),
                    };
                    *required(expected.get_mut(position)) = u8::from(value);
                    for constant in [
                        ConstValue::<StorageTarget>::U64(0xffff_ffff_ffff_fffe | u64::from(value)),
                        required(ConstValue::of_lane(128, (1u128 << 127) | u128::from(value))),
                        ConstValue::<StorageTarget>::from_bool(value),
                        ConstValue::U64(u64::from(value)),
                        required(ConstValue::of_lane(1, u128::from(value))),
                        required(ConstValue::of_lane(128, u128::from(value))),
                        ConstValue::NativeUInt(u64::from(value)),
                    ] {
                        assert_eq!(
                            constant.to_layout_bytes(layout, order, ptr_size),
                            Some(expected.clone()),
                            "width={width} {order:?} {ptr_size:?} {constant:?}"
                        );
                    }
                    assert_eq!(
                        ConstValue::<StorageTarget>::from_typed_bytes(&expected, &ty, order),
                        Some(ConstValue::from_bool(value))
                    );
                    assert_eq!(
                        ConstValue::<StorageTarget>::from_layout_bytes(&expected, layout, order),
                        Some(ConstValue::from_bool(value))
                    );
                }
            }
        }
    }
}

/// Every noncanonical single-byte pattern is refused instead of guessing Boolean padding.
#[test]
fn noncanonical_boolean_bytes_and_wrong_extents_are_refused() {
    for width in 1..=16 {
        let ty = MemoryType::Boolean(width);
        let layout = TypeLayout::of::<StorageTarget>(&ty, PointerSize::Bit64);
        for order in [Endianness::Little, Endianness::Big] {
            let bits = u32::from(width)
                .checked_mul(8)
                .unwrap_or_else(|| process::abort());
            let high = required(1u128.checked_shl(required(bits.checked_sub(1))));
            let all_ones = required(u128::MAX.checked_shr(required(128u32.checked_sub(bits))));
            for raw in (0..=255u128).chain([high, all_ones]) {
                let source = match order {
                    Endianness::Little => raw.to_le_bytes(),
                    Endianness::Big => raw.to_be_bytes(),
                };
                let start = match order {
                    Endianness::Little => 0,
                    Endianness::Big => required(16usize.checked_sub(usize::from(width))),
                };
                let end = required(start.checked_add(usize::from(width)));
                let bytes = required(source.get(start..end));
                let expected = match raw {
                    0 => Some(ConstValue::<StorageTarget>::False),
                    1 => Some(ConstValue::True),
                    _ => None,
                };
                assert_eq!(
                    ConstValue::<StorageTarget>::from_typed_bytes(bytes, &ty, order),
                    expected,
                    "width={width} {order:?} raw={raw}"
                );
                assert_eq!(
                    ConstValue::<StorageTarget>::from_layout_bytes(bytes, layout, order),
                    expected,
                    "width={width} {order:?} raw={raw}"
                );
            }
            for length in [
                0,
                required(usize::from(width).checked_sub(1)),
                required(usize::from(width).checked_add(1)),
            ] {
                let bytes = vec![0; length];
                assert_eq!(
                    ConstValue::<StorageTarget>::from_typed_bytes(&bytes, &ty, order),
                    None
                );
                assert_eq!(
                    ConstValue::<StorageTarget>::from_layout_bytes(&bytes, layout, order),
                    None
                );
            }
            for constant in [
                ConstValue::<StorageTarget>::Null,
                ConstValue::Symbol(()),
                ConstValue::from(1.0f64),
            ] {
                assert_eq!(
                    constant.to_layout_bytes(layout, order, PointerSize::Bit64),
                    None
                );
            }
        }
    }
}

/// Logical classification and untyped constants do not invent a memory representation.
#[test]
fn unattested_booleans_and_signed_bits_have_no_implicit_encoding() {
    for ty in [MemoryType::Boolean(0), MemoryType::SignedBit] {
        let layout = TypeLayout::of::<StorageTarget>(&ty, PointerSize::Bit64);
        assert_eq!(
            ConstValue::<StorageTarget>::True.to_layout_bytes(
                layout,
                Endianness::Little,
                PointerSize::Bit64
            ),
            None
        );
        assert_eq!(
            ConstValue::<StorageTarget>::from_layout_bytes(&[1], layout, Endianness::Little),
            None
        );
        assert_eq!(
            ConstValue::<StorageTarget>::from_typed_bytes(&[1], &ty, Endianness::Little),
            None
        );
    }
    assert_eq!(
        ConstValue::<StorageTarget>::True.to_bytes(Endianness::Little, PointerSize::Bit64),
        None
    );
    assert_eq!(
        ConstValue::<StorageTarget>::False.to_bytes(Endianness::Little, PointerSize::Bit64),
        None
    );
    assert_eq!(
        ConstValue::<StorageTarget>::U8(1).to_bytes(Endianness::Little, PointerSize::Bit64),
        Some(vec![1])
    );
}

/// Real SSA memory transfers use every attested extent and retain adjacent bytes.
#[test]
fn boolean_memory_uses_each_attested_extent_and_preserves_guards() {
    for width in 1..=16 {
        for ptr_size in [PointerSize::Bit32, PointerSize::Bit64, PointerSize::Bit128] {
            for order in [Endianness::Little, Endianness::Big] {
                for value in [false, true] {
                    let ty = MemoryType::Boolean(width);
                    let pointer_type = MemoryType::Pointer(ptr_size.bits());
                    let mut builder = SsaFunctionBuilder::<StorageTarget>::new(1, 0);
                    let mut raw = Vec::new();
                    let mut sides = None;
                    builder
                        .in_block(0, |block| {
                            let pointer = block.const_value(
                                SsaDefSpec::tmp(pointer_type),
                                ConstValue::NativeUInt(0x1000),
                            )?;
                            let before = block.const_value(
                                SsaDefSpec::tmp(pointer_type),
                                ConstValue::NativeUInt(0xfff),
                            )?;
                            let after = block.const_value(
                                SsaDefSpec::tmp(pointer_type),
                                ConstValue::NativeUInt(required(
                                    0x1000u64.checked_add(u64::from(width)),
                                )),
                            )?;
                            for (guard, byte) in [(before, 0x96), (after, 0xa5)] {
                                let input = block.const_value(
                                    SsaDefSpec::tmp(MemoryType::Byte),
                                    ConstValue::U8(byte),
                                )?;
                                block.store_indirect(guard, input, MemoryType::Byte)?;
                            }
                            let input = block
                                .const_value(SsaDefSpec::tmp(ty), ConstValue::from_bool(value))?;
                            block.store_indirect(pointer, input, ty)?;
                            let loaded = block.load_indirect(SsaDefSpec::tmp(ty), pointer, ty)?;
                            for offset in 0..width {
                                let address = block.const_value(
                                    SsaDefSpec::tmp(pointer_type),
                                    ConstValue::NativeUInt(required(
                                        0x1000u64.checked_add(u64::from(offset)),
                                    )),
                                )?;
                                raw.push(block.load_indirect(
                                    SsaDefSpec::tmp(MemoryType::Byte),
                                    address,
                                    MemoryType::Byte,
                                )?);
                            }
                            sides = Some((
                                block.load_indirect(
                                    SsaDefSpec::tmp(MemoryType::Byte),
                                    before,
                                    MemoryType::Byte,
                                )?,
                                block.load_indirect(
                                    SsaDefSpec::tmp(MemoryType::Byte),
                                    after,
                                    MemoryType::Byte,
                                )?,
                            ));
                            block.ret(Some(loaded))
                        })
                        .unwrap_or_else(|_| process::abort());
                    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
                    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
                    let machine = Machine::new(ptr_size, Some(order), Default::default());
                    let facts = StaticExecution::new(&ssa, &AnalysisContext::new(machine)).run();
                    assert!(
                        facts.is_complete(),
                        "width={width} {ptr_size:?} {order:?} {:?}",
                        facts.completion()
                    );
                    assert_eq!(
                        facts.returned().and_then(Value::constant),
                        Some(&ConstValue::from_bool(value)),
                        "width={width} {ptr_size:?} {order:?}"
                    );
                    let mut expected = vec![0; usize::from(width)];
                    let position = match order {
                        Endianness::Little => 0,
                        Endianness::Big => required(expected.len().checked_sub(1)),
                    };
                    *required(expected.get_mut(position)) = u8::from(value);
                    let actual: Option<Vec<_>> = raw
                        .into_iter()
                        .map(|id| match facts.constant(id) {
                            Some(ConstValue::U8(byte)) => Some(*byte),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(
                        actual,
                        Some(expected),
                        "wire bytes: width={width} {ptr_size:?} {order:?}"
                    );
                    let (before, after) = required(sides);
                    assert_eq!(facts.constant(before), Some(&ConstValue::U8(0x96)));
                    assert_eq!(facts.constant(after), Some(&ConstValue::U8(0xa5)));
                }
            }
        }
    }
}

/// Executes three complete Boolean elements with independently located guard bytes.
fn check_boolean_block_fill(reverse: bool) {
    for width in 1u8..=16 {
        for ptr_size in [PointerSize::Bit32, PointerSize::Bit64, PointerSize::Bit128] {
            for order in [Endianness::Little, Endianness::Big] {
                for value in [false, true] {
                    let extent = required(width.checked_mul(3));
                    let pointer_type = MemoryType::Pointer(ptr_size.bits());
                    let base = 0x1000u64;
                    let mut builder = SsaFunctionBuilder::<StorageTarget>::new(0, 0);
                    let mut observed = Vec::new();
                    builder
                        .in_block(0, |block| {
                            for (offset, byte) in [
                                (0xfffu64, 0x96),
                                (base.saturating_add(u64::from(extent)), 0xa5),
                            ] {
                                let address = block.const_value(
                                    SsaDefSpec::tmp(pointer_type),
                                    ConstValue::NativeUInt(offset),
                                )?;
                                let guard = block.const_value(
                                    SsaDefSpec::tmp(MemoryType::Byte),
                                    ConstValue::U8(byte),
                                )?;
                                block.store_indirect(address, guard, MemoryType::Byte)?;
                            }
                            let first = if reverse {
                                required(
                                    u64::from(width)
                                        .checked_mul(2)
                                        .and_then(|delta| base.checked_add(delta)),
                                )
                            } else {
                                base
                            };
                            let pointer = block.const_value(
                                SsaDefSpec::tmp(pointer_type),
                                ConstValue::NativeUInt(first),
                            )?;
                            let input = block.const_value(
                                SsaDefSpec::tmp(MemoryType::Boolean(width)),
                                ConstValue::from_bool(value),
                            )?;
                            let size = block.const_value(
                                SsaDefSpec::tmp(MemoryType::Byte),
                                ConstValue::U8(extent),
                            )?;
                            let backward = if reverse {
                                Some(block.const_value(
                                    SsaDefSpec::tmp(MemoryType::Boolean(1)),
                                    ConstValue::True,
                                )?)
                            } else {
                                None
                            };
                            block.init_blk(pointer, input, size, backward)?;
                            for address in 0xfff..=base.saturating_add(u64::from(extent)) {
                                let pointer = block.const_value(
                                    SsaDefSpec::tmp(pointer_type),
                                    ConstValue::NativeUInt(address),
                                )?;
                                observed.push(block.load_indirect(
                                    SsaDefSpec::tmp(MemoryType::Byte),
                                    pointer,
                                    MemoryType::Byte,
                                )?);
                            }
                            block.ret(None)
                        })
                        .unwrap_or_else(|_| process::abort());
                    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
                    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
                    let machine = Machine::new(ptr_size, Some(order), Default::default());
                    let context = AnalysisContext::new(machine);
                    let facts = StaticExecution::new(&ssa, &context).run();
                    assert!(
                        facts.is_complete(),
                        "width={width} {ptr_size:?} {order:?} reverse={reverse} {:?}",
                        facts.completion()
                    );
                    let actual: Option<Vec<_>> = observed
                        .into_iter()
                        .map(|id| match facts.constant(id) {
                            Some(ConstValue::U8(byte)) => Some(*byte),
                            _ => None,
                        })
                        .collect();
                    let mut expected = vec![0x96];
                    for _ in 0..3 {
                        for index in 0..width {
                            let unit = match order {
                                Endianness::Little => 0,
                                Endianness::Big => required(width.checked_sub(1)),
                            };
                            expected.push(if index == unit { u8::from(value) } else { 0 });
                        }
                    }
                    expected.push(0xa5);
                    assert_eq!(
                        actual,
                        Some(expected),
                        "width={width} {ptr_size:?} {order:?} value={value} reverse={reverse}"
                    );
                }
            }
        }
    }
}

/// Forward block fills use declared Boolean storage rather than logical value width.
#[test]
fn boolean_block_fill_forward_uses_attested_storage() {
    check_boolean_block_fill(false);
}

/// Reverse block fills subtract whole storage elements while keeping their byte order.
#[test]
fn boolean_block_fill_reverse_uses_attested_storage() {
    check_boolean_block_fill(true);
}

/// Unknown or unattested layouts do not turn logical width into concrete fill bytes.
#[test]
fn boolean_block_fill_unattested_storage_never_invents_bytes() {
    for ty in [
        MemoryType::Unknown,
        MemoryType::SignedBit,
        MemoryType::Boolean(0),
        MemoryType::Boolean(17),
    ] {
        let mut builder = SsaFunctionBuilder::<StorageTarget>::new(0, 0);
        let mut bytes = Vec::new();
        builder
            .in_block(0, |block| {
                let pointer = block.const_value(
                    SsaDefSpec::tmp(MemoryType::Pointer(64)),
                    ConstValue::NativeUInt(0x1000),
                )?;
                let input = block.const_value(SsaDefSpec::tmp(ty), ConstValue::U8(0x5a))?;
                let size =
                    block.const_value(SsaDefSpec::tmp(MemoryType::Byte), ConstValue::U8(4))?;
                block.init_blk(pointer, input, size, None)?;
                for address in 0x1000..0x1004 {
                    let pointer = block.const_value(
                        SsaDefSpec::tmp(MemoryType::Pointer(64)),
                        ConstValue::NativeUInt(address),
                    )?;
                    bytes.push(block.load_indirect(
                        SsaDefSpec::tmp(MemoryType::Byte),
                        pointer,
                        MemoryType::Byte,
                    )?);
                }
                block.ret(None)
            })
            .unwrap_or_else(|_| process::abort());
        let ssa = builder.finish().unwrap_or_else(|_| process::abort());
        let context = AnalysisContext::new(Machine::new(
            PointerSize::Bit64,
            Some(Endianness::Little),
            Default::default(),
        ));
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete(), "{ty:?}: {:?}", facts.completion());
        for byte in bytes {
            assert_eq!(facts.constant(byte), None, "{ty:?} invented a byte");
        }
    }
}

/// A zero-byte fill preserves memory with an opaque pointer or unattested layout.
#[test]
fn boolean_block_fill_zero_length_preserves_memory_for_opaque_inputs() {
    for reverse in [false, true] {
        for ty in [
            MemoryType::Unknown,
            MemoryType::Boolean(0),
            MemoryType::Boolean(2),
        ] {
            let mut builder = SsaFunctionBuilder::<StorageTarget>::new(1, 0);
            let mut unknown_id = None;
            builder
                .in_block(0, |block| {
                    let guard = block.const_value(
                        SsaDefSpec::tmp(MemoryType::Pointer(64)),
                        ConstValue::NativeUInt(0x1000),
                    )?;
                    let byte = block
                        .const_value(SsaDefSpec::tmp(MemoryType::Byte), ConstValue::U8(0x96))?;
                    block.store_indirect(guard, byte, MemoryType::Byte)?;
                    let unknown =
                        block.load_arg(SsaDefSpec::argument(0, MemoryType::Pointer(64)), 0)?;
                    unknown_id = Some(unknown);
                    let input = block.const_value(SsaDefSpec::tmp(ty), ConstValue::False)?;
                    let size =
                        block.const_value(SsaDefSpec::tmp(MemoryType::Byte), ConstValue::U8(0))?;
                    let backward = if reverse {
                        Some(block.const_value(
                            SsaDefSpec::tmp(MemoryType::Boolean(1)),
                            ConstValue::True,
                        )?)
                    } else {
                        None
                    };
                    block.init_blk(unknown, input, size, backward)?;
                    let observed = block.load_indirect(
                        SsaDefSpec::tmp(MemoryType::Byte),
                        guard,
                        MemoryType::Byte,
                    )?;
                    block.ret(Some(observed))
                })
                .unwrap_or_else(|_| process::abort());
            let ssa = builder.finish().unwrap_or_else(|_| process::abort());
            assert!(
                ssa.iter_instructions()
                    .any(|(_, _, instruction)| matches!(instruction.op(), SsaOp::InitBlk { .. }))
            );
            let image = BlockFillImage;
            let context = AnalysisContext::new(Machine::new(
                PointerSize::Bit64,
                Some(Endianness::Little),
                Default::default(),
            ))
            .with_image(&image);
            let facts = StaticExecution::new(&ssa, &context).run();
            assert!(matches!(
                facts.value(required(unknown_id)),
                Some(Value::Addr(_))
            ));
            assert!(
                facts.is_complete(),
                "{ty:?} reverse={reverse}: {:?}",
                facts.completion()
            );
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&ConstValue::U8(0x96)),
                "{ty:?} reverse={reverse}"
            );
        }
    }
}
