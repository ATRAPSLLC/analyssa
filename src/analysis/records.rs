//! Record layouts from the accesses measured from one base.
//!
//! Where nothing declares the type a pointer points to, its accesses still
//! say what is there: a load of four bytes eight bytes past the base, a store
//! of eight bytes at the base. [`recover_layouts`] groups a function's
//! indirect accesses by the value they are measured from and reports, for each
//! base whose accesses describe a record, the fields they agree on.
//!
//! # What counts as a record
//!
//! - **Two or more distinct offsets survive.** One offset is a dereference,
//!   not a layout.
//! - **An index makes an array of records, or of something else.** A base
//!   indexed by a stride (`p + i * 24 + 8`) is an array of elements that wide,
//!   and every access from it, indexed or not, reads the place in an element
//!   its offset leaves past a whole number of elements (`p[i].f`, `p[1].g`).
//!   Where each access lies within one element at two or more places no other
//!   reads at another width, the elements are the record, its size the
//!   stride. An element of one place says nothing either way: `a[0]`, `a[1]`
//!   and `a[i]` of an `int *` read it, and so do a record of `int`s and the
//!   `int` array member its index reads (dash's `parsefile`, whose `lastc` an
//!   `unget` indexes). Where an access crosses an element or two contest a
//!   place, the indexed accesses read something inside a record instead (an
//!   array member after a header). Either way the record is the accesses'
//!   that add no index, its size unstated. A pointer a loop walks in
//!   constant steps from one start reads the start's array, the walk an
//!   index of its step ([`AddressExpr::measured_from_walk`]): `p = phi(s, p + 12)` read at `p +
//!   8` and `s` read at `s + 4` read one array of 12-byte records.
//! - **No negative offset.** A record is addressed forward from its own base;
//!   a frame is addressed backwards into itself. An index brings an earlier
//!   element forward (`p[i - 1]`); a fixed displacement before the base does
//!   not.
//! - **No absolute base.** An access measured from a constant address names a
//!   global at a known address, not a field, and the offsets from one constant
//!   name separate objects.
//! - **A contested offset is dropped, not the record.** Two widths, or two
//!   types of different shape, at one offset name no field there: a union, a
//!   partial read, or a value reused for two objects. The other offsets still
//!   describe what they describe.
//! - **A literal copied in is no record.** A base whose every access stores a
//!   constant, the constants laid out at their offsets in the machine's byte
//!   order spelling text and the NUL that ends it, is a string an inlined
//!   copy of a literal filled (`strcpy(p, "/bin/sh")` as two word stores),
//!   not a record of words.
//!
//! Which bases may hold a record at all is the host's: a frame root addressed
//! forward (an incoming-argument area) is not one, and a base the host has
//! typed from a declaration is better described by the declaration.

use std::{cmp::Reverse, collections::BTreeMap, iter};

use crate::{
    analysis::{
        address::{AddressExpr, AddressWalk},
        consts::Machine,
        reach::greatest_common_step,
    },
    ir::{SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    pointer::PointerSize,
    target::{Endianness, Target},
};

/// One field of a record: the place its accesses read and write at one
/// offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordField<T: Target> {
    /// Offset from the record's base, in bits.
    pub offset_bits: u64,
    /// Access width in bits: the widest reading's, where there are several.
    pub bit_width: u64,
    /// The type the first access at this offset read or wrote, of the widest
    /// reading where there are several.
    pub ty: T::Type,
    /// The readings of a field accessed at two widths or as two shapes of
    /// type, each once, the widest first: a union of them, as C's `union
    /// node` holds an `int` in one variant and a pointer in another at one
    /// offset. Empty where every access agrees.
    pub arms: Vec<RecordArm<T>>,
}

/// One reading of a field accessed several ways ([`RecordField::arms`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordArm<T: Target> {
    /// Access width in bits.
    pub bit_width: u64,
    /// The type the first access of this reading read or wrote.
    pub ty: T::Type,
}

/// The record one value is the base of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordLayout<T: Target> {
    /// The value every field is measured from.
    pub base: SsaVarId,
    /// The fields in ascending offset order, each offset once.
    pub fields: Vec<RecordField<T>>,
    /// The size of one record, where the base is indexed as an array of them:
    /// the step every indexed access from it moves by, each field within it.
    pub stride_bytes: Option<u64>,
    /// The pointers a loop walks over these records from the start of one
    /// ([`AddressExpr::measured_from_walk`]), in ascending order: each points
    /// at a record wherever the walk has taken it, as the base does.
    pub walkers: Vec<SsaVarId>,
}

impl<T: Target> RecordLayout<T> {
    /// The offset of the field an access at `address` reads, when the address
    /// is measured from this record's base.
    ///
    /// The displacement from the base, laid into one record where the base is
    /// indexed as an array of them ([`Self::stride_bytes`]): `p + i * 24 + 32`
    /// reads the field at 8 of the element after `p[i]`. `None` for an address
    /// from another base or an absolute one, an index this record's stride
    /// does not move by, one into a record no array is made of, and a
    /// displacement before the base that no index brings forward.
    #[must_use]
    pub fn place(&self, address: &AddressExpr) -> Option<u64> {
        if address.root != self.base || address.base.is_absolute() {
            return None;
        }
        let Some(stride) = self.stride_bytes else {
            return address
                .index
                .is_none()
                .then(|| u64::try_from(address.root_offset_bits).ok())
                .flatten();
        };
        let element_bits = stride
            .checked_mul(8)
            .and_then(|bits| i64::try_from(bits).ok())
            .filter(|bits| *bits > 0)?;
        let moved = match address.index {
            Some(_) => address.stride_bytes > 0 && address.stride_bytes.is_multiple_of(stride),
            None => address.root_offset_bits >= 0,
        };
        moved
            .then(|| u64::try_from(address.root_offset_bits.rem_euclid(element_bits)).ok())
            .flatten()
    }
}

/// The records recovered for a function, by base: what an access through a
/// base, or through a walk over the array it starts, reads a field of.
#[derive(Debug, Clone)]
pub struct RecordLayouts<'a, T: Target> {
    /// Each record, under its base.
    by_base: BTreeMap<SsaVarId, &'a RecordLayout<T>>,
}

impl<'a, T: Target> RecordLayouts<'a, T> {
    /// Indexes `layouts` by base.
    ///
    /// # Arguments
    ///
    /// * `layouts` - The records recovered for one function
    ///   ([`recover_layouts`]), each base once.
    #[must_use]
    pub fn new(layouts: &'a [RecordLayout<T>]) -> Self {
        Self {
            by_base: layouts.iter().map(|layout| (layout.base, layout)).collect(),
        }
    }

    /// Returns the record an access at `addr` of `ir` reads, by its base, and
    /// the offset of the field it reads there ([`RecordLayout::place`]):
    /// measured from where the walk its pointer takes starts, where the
    /// start's record is the array the walk moves through, and from the
    /// pointer itself otherwise, as [`recover_layouts`] measures it.
    ///
    /// # Arguments
    ///
    /// * `ir` - The function the access is in.
    /// * `addr` - The address the access is made at.
    /// * `ptr_size` - The target pointer width.
    #[must_use]
    pub fn field_of(
        &self,
        ir: &SsaFunction<T>,
        addr: SsaVarId,
        ptr_size: PointerSize,
    ) -> Option<(SsaVarId, u64)> {
        let walk = AddressWalk::new(ir, ptr_size);
        let plain = walk.normalize(addr);
        [plain.measured_from_walk(&walk), plain]
            .into_iter()
            .find_map(|address| {
                let layout = self.by_base.get(&address.root)?;
                Some((address.root, layout.place(&address)?))
            })
    }
}

/// Where a declared field lies in its record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldExtent {
    /// Offset from the record's base, in bits.
    pub offset_bits: u64,
    /// Width in bits, when the declaration states one.
    pub bit_width: Option<u64>,
}

impl FieldExtent {
    /// Whether an access of `access_bits` at `offset_bits` lies within the
    /// field.
    ///
    /// An access at the field's own offset names it only at its full width;
    /// one starting inside it names it when it ends inside it too. A field of
    /// no stated width is named only by an access at its offset.
    #[must_use]
    pub fn contains(&self, offset_bits: u64, access_bits: u64) -> bool {
        let Some(width) = self.bit_width else {
            return self.offset_bits == offset_bits;
        };
        if self.offset_bits == offset_bits {
            return width == access_bits;
        }
        let (Some(access_end), Some(field_end)) = (
            offset_bits.checked_add(access_bits),
            self.offset_bits.checked_add(width),
        ) else {
            return false;
        };
        offset_bits >= self.offset_bits && access_end <= field_end
    }
}

/// Which declared field an access names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldMatch {
    /// Exactly one field contains the access: its position in the fields
    /// given.
    Field(usize),
    /// More than one field contains it, so no single field names it (a
    /// union).
    Overlapping,
    /// No field contains it, but one starts at its offset with another width:
    /// a partial read, or a stale declaration.
    WidthMismatch,
    /// No field contains it or starts where it does.
    NoField,
}

/// Names the field of a declared record an access reads or writes.
///
/// # Arguments
///
/// * `fields` - The record's fields, in any order.
/// * `offset_bits` - The access's offset from the record's base.
/// * `access_bits` - The access width.
///
/// # Returns
///
/// The field containing the access, or why there is none.
#[must_use]
pub fn match_field(
    fields: impl IntoIterator<Item = FieldExtent>,
    offset_bits: u64,
    access_bits: u64,
) -> FieldMatch {
    let mut found = None;
    let mut starts_here = false;
    for (index, field) in fields.into_iter().enumerate() {
        starts_here |= field.offset_bits == offset_bits;
        if field.contains(offset_bits, access_bits) {
            if found.is_some() {
                return FieldMatch::Overlapping;
            }
            found = Some(index);
        }
    }
    match found {
        Some(index) => FieldMatch::Field(index),
        None if starts_here => FieldMatch::WidthMismatch,
        None => FieldMatch::NoField,
    }
}

/// The bytes the constant stores measured from one base write, while every
/// access measured from it is one: the evidence that it is a literal copied
/// in rather than a record.
#[derive(Debug, Default)]
struct Fill {
    /// Each byte written, by its offset from the base.
    bytes: BTreeMap<u64, u8>,
}

impl Fill {
    /// Writes the `bits`-wide constant `raw` at `offset_bits` in `order`, or
    /// returns `false` where the store is no whole bytes of a known order or
    /// rewrites a byte with another value.
    fn write(&mut self, offset_bits: u64, bits: u64, raw: u64, order: Option<Endianness>) -> bool {
        let Some(order) = order else {
            return false;
        };
        if !offset_bits.is_multiple_of(8) || !bits.is_multiple_of(8) || bits > 64 {
            return false;
        }
        let width = bits / 8;
        let start = offset_bits / 8;
        let little = raw.to_le_bytes();
        (0..width).all(|index| {
            // The byte at `index` from the store's address is the number's
            // `index`th lowest in little-endian order, its `index`th highest
            // of the stored width in big-endian order.
            let significance = match order {
                Endianness::Little => index,
                Endianness::Big => width.saturating_sub(1).saturating_sub(index),
            };
            let Some(&byte) = usize::try_from(significance)
                .ok()
                .and_then(|at| little.get(at))
            else {
                return false;
            };
            self.bytes
                .insert(start.saturating_add(index), byte)
                .is_none_or(|written| written == byte)
        })
    }

    /// Whether the bytes written are text and the NUL that ends it: every
    /// byte from the base on, at least one printable character, and nothing
    /// but NULs from the first NUL to the last byte, which is one.
    fn is_text(&self) -> bool {
        let contiguous = self.bytes.keys().copied().eq(0..self.bytes.len() as u64);
        let mut bytes = self.bytes.values().copied();
        let text = bytes
            .by_ref()
            .take_while(|byte| *byte != 0)
            .map(|byte| matches!(byte, 0x20..=0x7e | b'\t' | b'\n' | b'\r'))
            .try_fold(0usize, |count, printable| {
                printable.then_some(count.saturating_add(1))
            });
        let terminated = self.bytes.values().last() == Some(&0);
        contiguous
            && terminated
            && text.is_some_and(|count| count > 0)
            && bytes.all(|byte| byte == 0)
    }
}

/// Recovers the layout of every record `ir`'s indirect accesses describe.
///
/// # Arguments
///
/// * `ir` - The function, normalized so that address arithmetic is visible to
///   [`AddressExpr::measured_from_walk`].
/// * `machine` - The target: its pointer width, and the byte order a
///   literal's constants are laid out in.
/// * `admits` - Whether a base may hold a record; a base it refuses is
///   skipped.
/// * `same_shape` - Whether two accesses at one offset name the same field.
/// * `least_fields` - The fewest uncontested fields a base that adds no
///   index must describe: two make a record, and one is a part of the
///   record of a pointer the base is one with, which only a caller relating
///   them can join (a frame cell reloaded and read at one offset each time).
///   An indexed base or a walked one is an array of records at two or more
///   whatever this says: at one place it is an array of that one value.
///
/// # Returns
///
/// One layout per base with at least `least_fields` uncontested fields, an
/// array's stating the size of its records, in ascending base order, so the
/// result is the same on every run over the same function.
#[must_use]
pub fn recover_layouts<T: Target>(
    ir: &SsaFunction<T>,
    machine: &Machine,
    admits: impl Fn(SsaVarId) -> bool,
    same_shape: impl Fn(&T::Type, &T::Type) -> bool,
    least_fields: usize,
) -> Vec<RecordLayout<T>> {
    let mut observed: BTreeMap<SsaVarId, Observed<'_, T>> = BTreeMap::new();
    for block in ir.blocks() {
        for instruction in block.instructions() {
            let (addr, value_type, stored) = match instruction.op() {
                SsaOp::LoadIndirect {
                    addr, value_type, ..
                } => (*addr, value_type, None),
                SsaOp::StoreIndirect {
                    addr,
                    value_type,
                    value,
                    ..
                } => (
                    *addr,
                    value_type,
                    ir.get_var_constant(*value)
                        .and_then(ConstValue::as_u64_bits),
                ),
                _ => continue,
            };
            let Some(bit_width) = T::bit_width(value_type).map(u64::from) else {
                continue;
            };
            // A walk from a constant address walks a global, which is the
            // image's object and no record's, and one from a base that holds
            // no record leaves the walk the only base its accesses have: each
            // is measured from the walker then, as from any pointer.
            let walk = AddressWalk::new(ir, machine.ptr_size);
            let plain = walk.normalize(addr);
            let address = Some(plain.measured_from_walk(&walk))
                .filter(|walked| !walked.base.is_absolute() && admits(walked.root))
                .unwrap_or(plain);
            if address.base.is_absolute() || !admits(address.root) {
                continue;
            }
            observed
                .entry(address.root)
                .or_insert_with(|| Observed {
                    accesses: Vec::new(),
                    fills: BTreeMap::new(),
                    walkers: BTreeMap::new(),
                })
                .observe(
                    Access {
                        address,
                        plain,
                        bit_width,
                        ty: value_type,
                    },
                    stored,
                    machine.byte_order,
                );
        }
    }
    let mut layouts: Vec<RecordLayout<T>> = observed
        .into_iter()
        .flat_map(|(base, observed)| observed.layouts(base, &same_shape, least_fields))
        .collect();
    layouts.sort_by_key(|layout| layout.base);
    layouts
}

/// One access to memory, measured from where the walk its pointer takes
/// starts ([`AddressExpr::measured_from_walk`]) and from the pointer itself.
struct Access<'a, T: Target> {
    /// Its address, measured from the start of the walk its pointer takes.
    address: AddressExpr,
    /// Its address, measured from the pointer it is made from.
    plain: AddressExpr,
    /// The width it reads or writes, in bits.
    bit_width: u64,
    /// The type it reads or writes.
    ty: &'a T::Type,
}

impl<T: Target> Access<'_, T> {
    /// The walker the access is made through, where it is made through one.
    fn walker(&self) -> Option<SsaVarId> {
        (self.address.index != self.plain.index)
            .then(|| self.plain.base.value())
            .flatten()
    }
}

/// What the accesses measured from one base say, before they are read as one
/// record or as an array of them.
struct Observed<'a, T: Target> {
    /// Each access measured from the base or from a walk it starts.
    accesses: Vec<Access<'a, T>>,
    /// The constant stores at the fixed offsets of the base (`None`) and of
    /// each walker, while every access at one is such a store.
    fills: BTreeMap<Option<SsaVarId>, Option<Fill>>,
    /// Each pointer a loop walks from the base whose accesses are among
    /// these ([`AddressExpr::measured_from_walk`]), with how far past the
    /// base, in bits, its walk starts.
    walkers: BTreeMap<SsaVarId, i64>,
}

impl<'a, T: Target> Observed<'a, T> {
    /// Adds `access`, storing the constant `stored` where it stores one,
    /// laid out in `order`.
    fn observe(&mut self, access: Access<'a, T>, stored: Option<u64>, order: Option<Endianness>) {
        let walker = access.walker();
        if let (Some(walker), Some(start)) = (
            walker,
            access
                .address
                .root_offset_bits
                .checked_sub(access.plain.root_offset_bits),
        ) {
            self.walkers.insert(walker, start);
        }
        if access.plain.index.is_none()
            && let Ok(offset_bits) = u64::try_from(access.plain.root_offset_bits)
        {
            let fill = self
                .fills
                .entry(walker)
                .or_insert_with(|| Some(Fill::default()));
            let filled = fill
                .as_mut()
                .zip(stored)
                .is_some_and(|(fill, raw)| fill.write(offset_bits, access.bit_width, raw, order));
            if !filled {
                *fill = None;
            }
        }
        self.accesses.push(access);
    }

    /// The records the accesses describe, measured from `base`.
    ///
    /// Where the base is indexed, or a loop walks from it, the array of
    /// records it makes, when every access lies within one element at two or
    /// more places no other contests or overlaps. Otherwise the record the
    /// accesses that add no index describe, a contested offset dropped, and
    /// for each walker, the one its own accesses describe measured from it,
    /// each where it has `least_fields` fields or more. A base or walker
    /// whose accesses copy a literal in describes none.
    fn layouts(
        &self,
        base: SsaVarId,
        same_shape: &impl Fn(&T::Type, &T::Type) -> bool,
        least_fields: usize,
    ) -> Vec<RecordLayout<T>> {
        let literal = |pointer: Option<SsaVarId>| {
            self.fills
                .get(&pointer)
                .and_then(Option::as_ref)
                .is_some_and(Fill::is_text)
        };
        let stride = self
            .accesses
            .iter()
            .filter(|access| access.address.index.is_some())
            .map(|access| access.address.stride_bytes)
            .reduce(greatest_common_step)
            .filter(|stride| *stride > 0);
        if let Some(stride) = stride
            && !literal(None)
        {
            let mut array = RecordLayout {
                base,
                fields: Vec::new(),
                stride_bytes: Some(stride),
                walkers: Vec::new(),
            };
            if let Some(fields) = self.element(&array, same_shape)
                && fields.len() >= 2
            {
                array.fields = fields;
                let element_bits = stride
                    .checked_mul(8)
                    .and_then(|bits| i64::try_from(bits).ok());
                array.walkers = self
                    .walkers
                    .iter()
                    .filter(|(_, start)| {
                        element_bits.is_some_and(|bits| start.rem_euclid(bits) == 0)
                    })
                    .map(|(walker, _)| *walker)
                    .collect();
                return vec![array];
            }
        }
        iter::once(base)
            .chain(self.walkers.keys().copied())
            .filter(|pointer| !literal((*pointer != base).then_some(*pointer)))
            .filter_map(|pointer| {
                let mut record = RecordLayout {
                    base: pointer,
                    fields: Vec::new(),
                    stride_bytes: None,
                    walkers: Vec::new(),
                };
                record.fields = self
                    .fields(&record, same_shape, pointer != base)
                    .into_values()
                    .collect();
                (record.fields.len() >= least_fields.max(1)).then_some(record)
            })
            .collect()
    }

    /// The fields the accesses `layout` places read, by offset: one reading
    /// where every access at an offset agrees, and the union of them where
    /// two contest it, at two widths or as two types of different shape
    /// ([`RecordField::arms`]). Each access is measured from the pointer it
    /// is made from where `plain`, and from the start of its walk otherwise.
    fn fields(
        &self,
        layout: &RecordLayout<T>,
        same_shape: &impl Fn(&T::Type, &T::Type) -> bool,
        plain: bool,
    ) -> BTreeMap<u64, RecordField<T>> {
        let mut readings: BTreeMap<u64, Vec<RecordArm<T>>> = BTreeMap::new();
        for access in &self.accesses {
            let measured = if plain {
                &access.plain
            } else {
                &access.address
            };
            let Some(offset_bits) = layout.place(measured) else {
                continue;
            };
            let held = readings.entry(offset_bits).or_default();
            if !held
                .iter()
                .any(|arm| arm.bit_width == access.bit_width && same_shape(&arm.ty, access.ty))
            {
                held.push(RecordArm {
                    bit_width: access.bit_width,
                    ty: access.ty.clone(),
                });
            }
        }
        readings
            .into_iter()
            .filter_map(|(offset_bits, mut arms)| {
                // The widest first, the order they were read in among equals.
                arms.sort_by_key(|arm| Reverse(arm.bit_width));
                let widest = arms.first()?.clone();
                if arms.len() < 2 {
                    arms.clear();
                }
                Some((
                    offset_bits,
                    RecordField {
                        offset_bits,
                        bit_width: widest.bit_width,
                        ty: widest.ty,
                        arms,
                    },
                ))
            })
            .collect()
    }

    /// The fields of one element of the array `array` reads the base as,
    /// where every access lies within one element at a place no other access
    /// contests or overlaps.
    fn element(
        &self,
        array: &RecordLayout<T>,
        same_shape: &impl Fn(&T::Type, &T::Type) -> bool,
    ) -> Option<Vec<RecordField<T>>> {
        let element_bits = array.stride_bytes?.checked_mul(8)?;
        if !self
            .accesses
            .iter()
            .all(|access| array.place(&access.address).is_some())
        {
            return None;
        }
        let fields: Vec<RecordField<T>> = self
            .fields(array, same_shape, false)
            .into_values()
            .collect();
        // A place two readings contest proves no element: the array's
        // records are read alike everywhere or are no array of them.
        let contested = fields.iter().any(|field| !field.arms.is_empty());
        let end = |field: &RecordField<T>| field.offset_bits.checked_add(field.bit_width);
        let apart = fields
            .iter()
            .zip(fields.iter().skip(1))
            .all(|(one, next)| end(one).is_some_and(|end| end <= next.offset_bits));
        let within = fields
            .last()
            .is_none_or(|last| end(last).is_some_and(|end| end <= element_bits));
        (!contested && apart && within).then_some(fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            PhiNode, PhiOperand, SsaBlock, SsaInstruction,
            float::FloatRules,
            ops::AccessQualifiers,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// A function entered with a pointer `p` and an integer `i`, whose block
    /// reads `p + offset`, or `p + i * stride + offset` where a stride is
    /// given, for each `(offset, stride, type)` in `accesses`.
    fn accessing(accesses: &[(i64, Option<u64>, MockType)]) -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let base = ir.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let index = ir.create_variable(
            VariableOrigin::EntryLiveIn,
            1,
            DefSite::entry(),
            MockType::I64,
        );
        let mut block = SsaBlock::new(0);
        let define = |ir: &mut SsaFunction<MockTarget>,
                      block: &mut SsaBlock<MockTarget>,
                      ty: MockType,
                      op: &dyn Fn(SsaVarId) -> SsaOp<MockTarget>| {
            let dest = ir.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, block.instructions().len()),
                ty,
            );
            block.add_instruction(SsaInstruction::synthetic(op(dest)));
            dest
        };
        for (offset, stride, ty) in accesses {
            let row = match stride {
                Some(stride) => {
                    let stride = define(&mut ir, &mut block, MockType::I64, &|dest| SsaOp::Const {
                        dest,
                        value: ConstValue::I64((*stride).cast_signed()),
                    });
                    let scaled = define(&mut ir, &mut block, MockType::I64, &|dest| SsaOp::Mul {
                        dest,
                        left: index,
                        right: stride,
                        flags: None,
                    });
                    define(&mut ir, &mut block, MockType::Ptr, &|dest| SsaOp::Add {
                        dest,
                        left: base,
                        right: scaled,
                        flags: None,
                    })
                }
                None => base,
            };
            let constant = define(&mut ir, &mut block, MockType::I64, &|dest| SsaOp::Const {
                dest,
                value: ConstValue::I64(*offset),
            });
            let address = define(&mut ir, &mut block, MockType::Ptr, &|dest| SsaOp::Add {
                dest,
                left: row,
                right: constant,
                flags: None,
            });
            define(&mut ir, &mut block, *ty, &|dest| SsaOp::LoadIndirect {
                dest,
                addr: address,
                value_type: *ty,
                access: AccessQualifiers::FLAT,
            });
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ir.add_block(block);
        ir.recompute_uses();
        (ir, base)
    }

    /// A 64-bit machine of `order`.
    fn machine(order: Endianness) -> Machine {
        Machine::new(PointerSize::Bit64, Some(order), FloatRules::default())
    }

    fn layouts(ir: &SsaFunction<MockTarget>) -> Vec<RecordLayout<MockTarget>> {
        recover_layouts(ir, &machine(Endianness::Little), |_| true, |a, b| a == b, 2)
    }

    /// A function entered with a pointer `p`, whose block stores at `p +
    /// offset` each `(offset, type, constant)` of `stores`, then reads `p +
    /// read` where `read` is given.
    fn storing(
        stores: &[(i64, MockType, ConstValue<MockTarget>)],
        read: Option<i64>,
    ) -> SsaFunction<MockTarget> {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let base = ir.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let mut block = SsaBlock::new(0);
        let address =
            |ir: &mut SsaFunction<MockTarget>, block: &mut SsaBlock<MockTarget>, offset: i64| {
                let index = block.instructions().len();
                let constant = ir.create_variable(
                    VariableOrigin::Local(0),
                    0,
                    DefSite::instruction(0, index),
                    MockType::I64,
                );
                let address = ir.create_variable(
                    VariableOrigin::Local(0),
                    0,
                    DefSite::instruction(0, index + 1),
                    MockType::Ptr,
                );
                block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                    dest: constant,
                    value: ConstValue::I64(offset),
                }));
                block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
                    dest: address,
                    left: base,
                    right: constant,
                    flags: None,
                }));
                address
            };
        for (offset, ty, constant) in stores {
            let addr = address(&mut ir, &mut block, *offset);
            let index = block.instructions().len();
            let value = ir.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, index),
                *ty,
            );
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: value,
                value: constant.clone(),
            }));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::StoreIndirect {
                addr,
                value,
                value_type: *ty,
                access: AccessQualifiers::FLAT,
            }));
        }
        if let Some(offset) = read {
            let addr = address(&mut ir, &mut block, offset);
            let index = block.instructions().len();
            let loaded = ir.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, index),
                MockType::I8,
            );
            block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadIndirect {
                dest: loaded,
                addr,
                value_type: MockType::I8,
                access: AccessQualifiers::FLAT,
            }));
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ir.add_block(block);
        ir.recompute_uses();
        ir
    }

    /// **A literal copied in is no record.** `strcpy(p, "/bin/sh")` inlined
    /// as two word stores writes text and its NUL from `p` on: a string, in
    /// either byte order its constants are laid out in. The controls: the
    /// same words read in the other byte order spell no text, words that
    /// leave a byte between them unwritten, words that end in no NUL, and
    /// the literal read back through `p`, are each the record of two words
    /// they look like.
    #[test]
    fn a_literal_copied_in_is_no_record() {
        let words = |first: u32, second: u32| {
            [
                (0, MockType::I32, ConstValue::I32(first.cast_signed())),
                (4, MockType::I32, ConstValue::I32(second.cast_signed())),
            ]
        };
        let found = |stores: &[(i64, MockType, ConstValue<MockTarget>)],
                     read: Option<i64>,
                     order: Endianness| {
            recover_layouts(
                &storing(stores, read),
                &machine(order),
                |_| true,
                |a, b| a == b,
                2,
            )
            .len()
        };
        let little = words(0x6e69_622f, 0x0068_732f);
        let big = words(0x2f62_696e, 0x2f73_6800);
        assert_eq!(found(&little, None, Endianness::Little), 0, "\"/bin/sh\"");
        assert_eq!(
            found(&big, None, Endianness::Big),
            0,
            "\"/bin/sh\" big-endian"
        );
        assert_eq!(found(&little, None, Endianness::Big), 1, "the other order");
        assert_eq!(
            found(
                &[
                    (0, MockType::I32, ConstValue::I32(0x6e69_622f)),
                    (8, MockType::I32, ConstValue::I32(0x0068_732f)),
                ],
                None,
                Endianness::Little
            ),
            1,
            "a gap"
        );
        assert_eq!(
            found(&words(0x6e69_622f, 0x6873_2f2f), None, Endianness::Little),
            1,
            "no NUL"
        );
        assert_eq!(found(&little, Some(1), Endianness::Little), 1, "read back");
    }

    /// Accesses at two offsets from one base are a record of two fields, in
    /// offset order.
    #[test]
    fn two_offsets_from_one_base_are_a_record() {
        let (ir, base) = accessing(&[(8, None, MockType::I64), (0, None, MockType::I32)]);
        let found = layouts(&ir);
        assert_eq!(found.len(), 1);
        let record = found.first().unwrap();
        assert_eq!(record.base, base);
        let fields: Vec<(u64, u64)> = record
            .fields
            .iter()
            .map(|field| (field.offset_bits, field.bit_width))
            .collect();
        assert_eq!(fields, [(0, 32), (64, 64)]);
    }

    /// One offset is a dereference and a negative one addresses a frame:
    /// neither makes a record.
    #[test]
    fn a_dereference_and_a_frame_are_no_record() {
        let (ir, _) = accessing(&[(0, None, MockType::I32), (0, None, MockType::I32)]);
        assert!(layouts(&ir).is_empty(), "one offset");
        let (ir, _) = accessing(&[(-8, None, MockType::I32), (-4, None, MockType::I32)]);
        assert!(layouts(&ir).is_empty(), "negative offsets");
    }

    /// **An offset read at two widths is a union of its readings.** `union
    /// node` holds an `int` in one variant and a pointer in another at one
    /// offset: the field there is both, the widest first, beside the field at
    /// `8`. Read one way only, a field has no arms. An array's element read
    /// two ways at one place proves no element.
    #[test]
    fn an_offset_read_at_two_widths_is_a_union_of_its_readings() {
        let (ir, _) = accessing(&[
            (0, None, MockType::I32),
            (0, None, MockType::I64),
            (8, None, MockType::I32),
        ]);
        let found = layouts(&ir);
        let [record] = found.as_slice() else {
            panic!("one record: {found:?}");
        };
        let fields: Vec<(u64, u64, Vec<u64>)> = record
            .fields
            .iter()
            .map(|field| {
                (
                    field.offset_bits,
                    field.bit_width,
                    field.arms.iter().map(|arm| arm.bit_width).collect(),
                )
            })
            .collect();
        assert_eq!(fields, [(0, 64, vec![64, 32]), (64, 32, Vec::new())]);
        let (ir, _) = accessing(&[
            (0, Some(16), MockType::I32),
            (0, Some(16), MockType::I64),
            (8, Some(16), MockType::I32),
        ]);
        assert!(
            layouts(&ir)
                .iter()
                .all(|layout| layout.stride_bytes.is_none()),
            "no array of an element read two ways"
        );
    }

    /// The places of `ir`'s one record, each `(offset, width)` in bits, and
    /// the size it states.
    fn shape(ir: &SsaFunction<MockTarget>) -> (Vec<(u64, u64)>, Option<u64>) {
        let found = layouts(ir);
        let [record] = found.as_slice() else {
            panic!("one record expected, found {}", found.len());
        };
        let places = record
            .fields
            .iter()
            .map(|field| (field.offset_bits, field.bit_width))
            .collect();
        (places, record.stride_bytes)
    }

    /// **An index makes an array of records.** `p[i].a` and `p[i].b` of
    /// 16-byte records are two fields of a record 16 bytes long, and so are
    /// `p[1].b` (a fixed offset past the first element), `p[i + 1].a` and
    /// `p[i - 1].a` (an index's offset past or before its element). Two
    /// strides from one base move by the step they share.
    #[test]
    fn an_indexed_base_is_an_array_of_records() {
        let (ir, _) = accessing(&[(0, Some(16), MockType::I64), (8, Some(16), MockType::I32)]);
        assert_eq!(shape(&ir), (vec![(0, 64), (64, 32)], Some(16)));
        let (ir, _) = accessing(&[
            (0, Some(16), MockType::I64),
            (24, None, MockType::I32),
            (16, Some(16), MockType::I64),
            (-16, Some(16), MockType::I64),
        ]);
        assert_eq!(shape(&ir), (vec![(0, 64), (64, 32)], Some(16)), "folded");
        let (ir, _) = accessing(&[(0, Some(32), MockType::I64), (8, Some(16), MockType::I32)]);
        assert_eq!(
            shape(&ir),
            (vec![(0, 64), (64, 32)], Some(16)),
            "a shared step"
        );
    }

    /// **An element of one place says nothing.** `a[0]`, `a[1]` and `a[i]`
    /// of an `int *` read one place of four-byte elements, and so do a
    /// record of two `int`s and an `int` array member after them that an
    /// index reads (`p->lastc[p->unget]`): the fixed accesses are the record
    /// they are without the index, its size unstated.
    #[test]
    fn an_element_of_one_place_says_nothing() {
        let (ir, _) = accessing(&[
            (0, None, MockType::I32),
            (4, None, MockType::I32),
            (8, Some(4), MockType::I32),
        ]);
        assert_eq!(shape(&ir), (vec![(0, 32), (32, 32)], None));
    }

    /// **An index inside a record leaves the record.** A header of two `int`s
    /// and an array of `long` after it folds the array onto the first `int`
    /// at another width, and a header of two `long`s and a byte buffer after
    /// it leaves each `long` across many one-byte elements: each is the
    /// record of its header, no size stated. A union inside the elements, two
    /// places overlapping, refutes the array the same way.
    #[test]
    fn an_index_inside_a_record_leaves_the_record() {
        let (ir, _) = accessing(&[
            (0, None, MockType::I32),
            (4, None, MockType::I32),
            (8, Some(8), MockType::I64),
        ]);
        assert_eq!(
            shape(&ir),
            (vec![(0, 32), (32, 32)], None),
            "a contested place"
        );
        let (ir, _) = accessing(&[
            (0, None, MockType::I64),
            (8, None, MockType::I64),
            (16, Some(1), MockType::I8),
        ]);
        assert_eq!(
            shape(&ir),
            (vec![(0, 64), (64, 64)], None),
            "across elements"
        );
        let (ir, _) = accessing(&[
            (0, None, MockType::I64),
            (4, None, MockType::I32),
            (0, Some(16), MockType::I64),
        ]);
        assert_eq!(
            shape(&ir),
            (vec![(0, 64), (32, 32)], None),
            "overlapping places"
        );
    }

    /// A function entered with a pointer `s` that reads each `(offset, type)`
    /// of `starts` at `s + offset`, then loops over `p = phi(s, p + step)`
    /// reading each of `walks` at `p + offset`. Returns it, `s` and `p`.
    fn walking(
        starts: &[(i64, MockType)],
        walks: &[(i64, MockType)],
        step: i64,
    ) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let start = ir.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let walked = ir.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::Ptr);
        let define = |ir: &mut SsaFunction<MockTarget>,
                      block: &mut SsaBlock<MockTarget>,
                      ty: MockType,
                      op: &dyn Fn(SsaVarId) -> SsaOp<MockTarget>| {
            let dest = ir.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(block.id(), block.instructions().len()),
                ty,
            );
            block.add_instruction(SsaInstruction::synthetic(op(dest)));
            dest
        };
        let moved = |ir: &mut SsaFunction<MockTarget>,
                     block: &mut SsaBlock<MockTarget>,
                     base: SsaVarId,
                     offset: i64| {
            define(ir, block, MockType::Ptr, &|dest| SsaOp::PtrAdd {
                dest,
                base,
                index: None,
                stride: 0,
                offset,
                result_type: MockType::Ptr,
            })
        };
        let mut entry = SsaBlock::new(0);
        let mut body = SsaBlock::new(1);
        let mut reads = Vec::new();
        for (block, base, accesses) in [(&mut entry, start, starts), (&mut body, walked, walks)] {
            for (offset, ty) in accesses {
                let address = moved(&mut ir, block, base, *offset);
                reads.push((block.id(), address, *ty));
            }
        }
        let next = moved(&mut ir, &mut body, walked, step);
        for (block, address, ty) in reads {
            let loaded = ir.create_variable(
                VariableOrigin::Local(1),
                0,
                DefSite::instruction(block, 0),
                ty,
            );
            let read = SsaInstruction::synthetic(SsaOp::LoadIndirect {
                dest: loaded,
                addr: address,
                value_type: ty,
                access: AccessQualifiers::FLAT,
            });
            if block == 0 {
                entry.add_instruction(read);
            } else {
                body.add_instruction(read);
            }
        }
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        let mut phi = PhiNode::new(walked, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(start, 0));
        phi.add_operand(PhiOperand::new(next, 1));
        body.add_phi(phi);
        body.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ir.add_block(entry);
        ir.add_block(body);
        ir.recompute_uses();
        (ir, start, walked)
    }

    /// A record as the walk tests read it: its base, each field's `(offset,
    /// width)` in bits, its size and its walkers.
    type Shape = (SsaVarId, Vec<(u64, u64)>, Option<u64>, Vec<SsaVarId>);

    /// The base, places and size of each record `layouts` reads in `ir`, and
    /// the walkers of each.
    fn records(ir: &SsaFunction<MockTarget>) -> Vec<Shape> {
        layouts(ir)
            .into_iter()
            .map(|record| {
                let places = record
                    .fields
                    .iter()
                    .map(|field| (field.offset_bits, field.bit_width))
                    .collect();
                (record.base, places, record.stride_bytes, record.walkers)
            })
            .collect()
    }

    /// **A walk reads its start's array.** `s[0]` and `s[1]` of a start
    /// `s`, and `p + 8` of `p = phi(s, p + 12)`, are one record of three
    /// `int`s 12 bytes long, measured from `s`, and `p` points at one of them
    /// wherever it has walked. The controls: `p + 6` of a walk moving by 8
    /// lies across two elements, and the record is the one the start's
    /// accesses alone describe, with no size; and a walk by 8 reading an
    /// `int` and a `long` at one place of its elements reads a record of its
    /// own, measured from it, as a length and the pointers after it are,
    /// and so does a walk from a start that holds no record.
    #[test]
    fn a_walk_reads_its_starts_array() {
        let ints = [(0, MockType::I32), (4, MockType::I32)];
        let (ir, start, walked) = walking(&ints, &[(8, MockType::I32)], 12);
        assert_eq!(
            records(&ir),
            [(
                start,
                vec![(0, 32), (32, 32), (64, 32)],
                Some(12),
                vec![walked]
            )]
        );
        let found = layouts(&ir);
        let read = ir
            .blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .filter_map(|instruction| match instruction.op() {
                SsaOp::LoadIndirect { addr, .. } => Some(*addr),
                _ => None,
            })
            .next_back()
            .expect("the walk's read");
        assert_eq!(
            RecordLayouts::new(&found).field_of(&ir, read, PointerSize::Bit64),
            Some((start, 64)),
            "the walk reads the start's third member"
        );
        let (ir, start, walked) = walking(&ints, &[(0, MockType::I32), (8, MockType::I32)], 12);
        let unadmitted = recover_layouts(
            &ir,
            &machine(Endianness::Little),
            |base| base != start,
            |a, b| a == b,
            2,
        );
        assert_eq!(
            unadmitted
                .iter()
                .map(|layout| (layout.base, layout.stride_bytes))
                .collect::<Vec<_>>(),
            [(walked, None)],
            "a start that holds no record"
        );
        let (ir, start, _) = walking(&ints, &[(6, MockType::I32)], 8);
        assert_eq!(
            records(&ir),
            [(start, vec![(0, 32), (32, 32)], None, Vec::new())],
            "across elements"
        );
        let (ir, _, walked) = walking(&[], &[(0, MockType::I32), (8, MockType::I64)], 8);
        assert_eq!(
            records(&ir),
            [(walked, vec![(0, 32), (64, 64)], None, Vec::new())],
            "the walk's own record"
        );
    }

    /// An access names the one field containing it; two containing it, a
    /// field at its offset of another width, and nothing at all are each
    /// reported as such.
    #[test]
    fn an_access_names_the_field_containing_it() {
        let field = |offset_bits, bit_width| FieldExtent {
            offset_bits,
            bit_width,
        };
        let record = [field(0, Some(32)), field(32, Some(32)), field(64, None)];
        assert_eq!(match_field(record, 32, 32), FieldMatch::Field(1));
        assert_eq!(match_field(record, 40, 8), FieldMatch::Field(1), "inside");
        assert_eq!(match_field(record, 64, 16), FieldMatch::Field(2), "unsized");
        assert_eq!(match_field(record, 32, 16), FieldMatch::WidthMismatch);
        assert_eq!(match_field(record, 28, 8), FieldMatch::NoField, "straddles");
        assert_eq!(match_field(record, 96, 32), FieldMatch::NoField);

        let union = [field(0, Some(32)), field(0, Some(64))];
        assert_eq!(match_field(union, 8, 8), FieldMatch::Overlapping);
        assert!(
            !field(u64::MAX - 4, Some(8)).contains(u64::MAX - 2, 8),
            "overflow"
        );
    }

    /// **One field is a part of a record only where the caller joins it.**
    /// A base read at one offset describes no record by itself; asked for
    /// parts, the layout states its one field, which a caller relating the
    /// base to another pointer adds to that pointer's record. An indexed
    /// base read at one place is an array of that value, no record's part,
    /// whatever the caller asks.
    #[test]
    fn one_field_is_a_part_of_a_record_only_where_the_caller_joins_it() {
        let found = |accesses: &[(i64, Option<u64>, MockType)], least_fields: usize| {
            let (ir, base) = accessing(accesses);
            recover_layouts(
                &ir,
                &machine(Endianness::Little),
                |_| true,
                |a, b| a == b,
                least_fields,
            )
            .into_iter()
            .filter(|layout| layout.base == base)
            .map(|layout| layout.fields.len())
            .collect::<Vec<_>>()
        };
        let single = [(8, None, MockType::I32)];
        assert_eq!(found(&single, 2), Vec::<usize>::new(), "no record alone");
        assert_eq!(
            found(&single, 1),
            vec![1],
            "a part where parts are asked for"
        );
        let indexed = [(8, Some(4), MockType::I32)];
        assert_eq!(
            found(&indexed, 1),
            Vec::<usize>::new(),
            "an array of one value"
        );
    }

    /// A base the host refuses is skipped.
    #[test]
    fn a_base_the_host_refuses_is_skipped() {
        let (ir, base) = accessing(&[(0, None, MockType::I32), (4, None, MockType::I32)]);
        let found = recover_layouts(
            &ir,
            &machine(Endianness::Little),
            |b| b != base,
            |a, b| a == b,
            2,
        );
        assert!(found.is_empty());
    }
}
