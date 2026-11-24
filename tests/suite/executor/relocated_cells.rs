//! Typed image-cell evidence preserves named pointers without guessing from numeric bits.

use std::{
    error::Error,
    sync::atomic::{AtomicUsize, Ordering},
};

use analyssa::{
    PointerSize,
    analysis::{
        Machine, SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{Completion, FindingKind, Root, StaticExecution, StaticFacts, Value},
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{ConstValue, SsaDefSpec, SsaFunctionBuilder},
    target::Endianness,
    testing::{MACHINE, MockTarget, MockType},
};

/// Serves independently encoded physical bits and an optional named-cell fact.
struct Image {
    /// Complete physical cell bytes.
    bytes: Vec<u8>,
    /// Image byte order.
    order: Endianness,
    /// Whether the backing has the immutable lifetime contract.
    immutable: bool,
    /// Maximum available prefix of the backing.
    available: usize,
    /// Whether two adjacent mappings split the cell.
    split: bool,
    /// Optional module coordinate explicitly vouched for by the host.
    target: Option<u64>,
    /// Evidence queries, to check that unsupported accesses do not request it.
    queries: AtomicUsize,
}

impl Image {
    /// Encodes the actual loaded number independently of the named address.
    fn new(width: usize, order: Endianness, target: u64) -> Result<Self, Box<dyn Error>> {
        let physical = 0x400000u128
            .checked_add(u128::from(target))
            .ok_or("physical address")?;
        let bytes = match order {
            Endianness::Little => physical
                .to_le_bytes()
                .get(..width)
                .ok_or("cell width")?
                .to_vec(),
            Endianness::Big => physical
                .to_be_bytes()
                .get(16usize.checked_sub(width).ok_or("cell width")?..)
                .ok_or("cell width")?
                .to_vec(),
        };
        Ok(Self {
            bytes,
            order,
            immutable: true,
            available: width,
            split: false,
            target: Some(target),
            queries: AtomicUsize::new(0),
        })
    }
}

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        self.order
    }
    fn region(&self, at: u64) -> Option<ImageRegion> {
        let end = 0x3000u64.checked_add(u64::try_from(self.bytes.len()).ok()?)?;
        if !(0x3000..end).contains(&at) {
            return None;
        }
        let middle =
            0x3000u64.checked_add(u64::try_from(self.bytes.len().checked_div(2)?).ok()?)?;
        let (start, end) = if self.split && at < middle {
            (0x3000, middle)
        } else if self.split {
            (middle, end)
        } else {
            (0x3000, end)
        };
        Some(ImageRegion {
            start,
            end,
            mutability: if self.immutable {
                Mutability::ReadOnly
            } else {
                Mutability::Writable
            },
        })
    }
    fn bytes(&self, at: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(at.checked_sub(0x3000)?).ok()?;
        let end = start.checked_add(len)?;
        (end <= self.available).then_some(())?;
        self.bytes.get(start..end)
    }
    fn number_offset(&self) -> Option<i64> {
        Some(-0x400000)
    }
    fn address_at(&self, at: u64, len: u64) -> Option<u64> {
        self.queries.fetch_add(1, Ordering::Relaxed);
        (at == 0x3000 && usize::try_from(len).ok() == Some(self.bytes.len()))
            .then_some(self.target)
            .flatten()
    }
}

/// Interprets one verified load, optionally preceded by a forbidden partial overwrite.
fn run(
    image: &Image,
    ty: MockType,
    ptr_size: PointerSize,
    order: Endianness,
    overwrite: bool,
) -> Result<StaticFacts<MockTarget>, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let slot = block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x3000))?;
        if overwrite {
            let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x7f))?;
            block.store_indirect(slot, byte, MockType::U8)?;
        }
        let value = block.load_indirect(SsaDefSpec::tmp(ty), slot, ty)?;
        block.ret(Some(value))
    })?;
    let function = builder.finish()?;
    let errors = SsaVerifier::new(&function).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    let context = AnalysisContext::new(Machine {
        ptr_size,
        byte_order: Some(order),
        ..MACHINE
    })
    .with_image(image);
    Ok(StaticExecution::new(&function, &context).run())
}

/// Both pointer and integer register views retain the exact named root, including module zero.
#[test]
fn complete_immutable_cells_preserve_pointer_and_integer_address_views()
-> Result<(), Box<dyn Error>> {
    for order in [Endianness::Little, Endianness::Big] {
        for (ptr, width, integer, signed) in [
            (PointerSize::Bit32, 4, MockType::U32, MockType::I32),
            (PointerSize::Bit64, 8, MockType::U64, MockType::I64),
        ] {
            for target in [0, 0x1200] {
                for ty in [MockType::Ptr, integer, signed] {
                    let image = Image::new(width, order, target)?;
                    let facts = run(&image, ty, ptr, order, false)?;
                    assert_eq!(facts.completion(), Completion::Complete);
                    let address = facts
                        .returned()
                        .and_then(Value::address)
                        .ok_or("named returned address")?;
                    assert_eq!(address.root(), Root::Symbol);
                    assert_eq!(address.image_address(), Some(target));
                    assert_eq!(address.pointer_size(), ptr);
                    assert_eq!(image.queries.load(Ordering::Relaxed), 1);
                }
            }
        }
    }
    Ok(())
}

/// A numeric word gains no pointer identity from its magnitude alone.
#[test]
fn untagged_cells_remain_exact_physical_integers() -> Result<(), Box<dyn Error>> {
    for order in [Endianness::Little, Endianness::Big] {
        for (ptr, width, ty) in [
            (PointerSize::Bit32, 4, MockType::U32),
            (PointerSize::Bit64, 8, MockType::U64),
        ] {
            let mut image = Image::new(width, order, 0x1200)?;
            image.target = None;
            let facts = run(&image, ty, ptr, order, false)?;
            assert_eq!(
                facts
                    .returned()
                    .and_then(Value::constant)
                    .and_then(ConstValue::as_u64_bits),
                Some(0x401200)
            );
        }
    }
    Ok(())
}

/// Float, vector, narrow, wider and unspecified views never request a named pointer.
#[test]
fn nonpointer_register_layouts_keep_their_ordinary_typed_reads() -> Result<(), Box<dyn Error>> {
    for order in [Endianness::Little, Endianness::Big] {
        for ty in [
            MockType::F64,
            MockType::V4I16,
            MockType::U32,
            MockType::U128,
            MockType::Unknown,
        ] {
            let image = Image::new(8, order, 0x1200)?;
            let facts = run(&image, ty, PointerSize::Bit64, order, false)?;
            assert_eq!(facts.completion(), Completion::Complete);
            assert_ne!(
                facts
                    .returned()
                    .and_then(Value::address)
                    .and_then(|address| address.image_address()),
                Some(0x1200)
            );
            assert_eq!(image.queries.load(Ordering::Relaxed), 0);
        }
    }
    Ok(())
}

/// Missing, mutable, split or wrongly interpreted backing supplies no named-cell fact.
#[test]
fn backing_and_byte_order_must_support_the_complete_interpretation() -> Result<(), Box<dyn Error>> {
    for (available, immutable, split, order) in [
        (7, true, false, Endianness::Little),
        (8, false, false, Endianness::Little),
        (8, true, true, Endianness::Little),
        (8, true, false, Endianness::Big),
        (0, true, false, Endianness::Little),
    ] {
        let mut image = Image::new(8, Endianness::Little, 0x1200)?;
        image.available = available;
        image.immutable = immutable;
        image.split = split;
        let facts = run(&image, MockType::U64, PointerSize::Bit64, order, false)?;
        assert_eq!(facts.completion(), Completion::Complete);
        assert_ne!(
            facts
                .returned()
                .and_then(Value::address)
                .and_then(|address| address.image_address()),
            Some(0x1200)
        );
        assert_eq!(image.queries.load(Ordering::Relaxed), 0);
    }
    Ok(())
}

/// An unrepresentable host coordinate is refused instead of wrapping to another symbol.
#[test]
fn named_addresses_must_fit_the_target_pointer_width() -> Result<(), Box<dyn Error>> {
    let mut image = Image::new(4, Endianness::Little, 0x1200)?;
    image.target = Some(0x1_0000_1200);
    let facts = run(
        &image,
        MockType::U32,
        PointerSize::Bit32,
        Endianness::Little,
        false,
    )?;
    assert_eq!(
        facts
            .returned()
            .and_then(Value::constant)
            .and_then(ConstValue::as_u64_bits),
        Some(0x401200)
    );
    Ok(())
}

/// A partial overwrite of immutable backing faults before the named cell is
/// observed: the path ends at the store, found there, and returns nothing.
#[test]
fn a_readonly_write_faults_before_named_cell_observation() -> Result<(), Box<dyn Error>> {
    let image = Image::new(8, Endianness::Little, 0x1200)?;
    let facts = run(
        &image,
        MockType::U64,
        PointerSize::Bit64,
        Endianness::Little,
        true,
    )?;
    assert_eq!(facts.completion(), Completion::Complete);
    assert!(
        facts
            .findings()
            .iter()
            .any(|finding| finding.kind == FindingKind::ReadOnlyWrite)
    );
    assert!(facts.returned().is_none());
    assert_eq!(image.queries.load(Ordering::Relaxed), 0);
    Ok(())
}
