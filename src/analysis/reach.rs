//! The bytes an access moving through a constant address may touch.
//!
//! An access at a fixed displacement from a constant address says nothing of
//! how far the object there extends: a compiler forms one anchor and reaches
//! several neighbouring objects from it at fixed offsets. An access that moves
//! does say it. A table read by a byte index through `&table + 129` touches
//! every byte the index can reach from there, and a loop walking a pointer
//! from `&names` to the bound `&names + 256` touches every element on the way.
//! Each such access proves the bytes it may touch belong to one object.
//!
//! [`for_each_reach`] states those bytes for every access whose address
//! moves - by an index the value ranges bound, or along a walk a comparison
//! ends - measured from the constant the address is built on. What objects the
//! reaches make is the host's question: only it knows which constants are
//! addresses of its image and what the image already states about them.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Range,
};

use crate::{
    analysis::{
        address::{AddressBase, AddressWalk},
        budget::WorkBudget,
        cfg::SsaCfg,
        image::{ImageData, Mutability},
        memory::MemorySsa,
        pointsto::{PointsTo, UNKNOWN_LOC},
        range::ValueRange,
    },
    graph::{
        RootedGraph,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{CmpKind, SsaFunction, SsaOp, SsaVarId, ops::AddressedAccess},
    passes::ranges::{self, GuardedRanges, ValueRanges},
    pointer::PointerSize,
    target::Target,
};

/// The bytes one access may touch, measured from the constant address its
/// address is built on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reach {
    /// The address the constant the access is built on holds.
    pub root: u64,
    /// The bytes the access may touch, as offsets from [`Self::root`]: the
    /// first, and one past the last.
    pub bytes: Range<i64>,
    /// The step in bytes between the places the access may land: the stride
    /// of its index, or of the walk it moves along.
    pub stride: u64,
    /// Where the access moves along a walk a comparison ends, at constant
    /// displacements from the walker, rather than by an index, the block
    /// whose closing comparison ends it: its bytes end where the walk stops,
    /// and that comparison is what bounds it there. `None` for an index.
    pub walk: Option<usize>,
    /// The access.
    pub access: AddressedAccess,
}

/// Where a value may point: a constant address moved by any of the multiples
/// of `stride` from `low` to `high`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Origin {
    /// The constant address the displacements are measured from.
    root: u64,
    /// The least displacement from the root's address, in bytes.
    low: i64,
    /// The greatest.
    high: i64,
    /// The step between the displacements; `0` for a value that does not
    /// move.
    stride: u64,
    /// Where the displacements are a walk's, a pointer a loop steps until
    /// a comparison ends it ([`Reaches::walked`]), moved by constants alone:
    /// the block whose closing comparison ends it.
    walked: Option<usize>,
}

impl Origin {
    /// The origin `bytes` further on.
    fn moved(self, bytes: i64) -> Option<Self> {
        Some(Self {
            low: self.low.checked_add(bytes)?,
            high: self.high.checked_add(bytes)?,
            ..self
        })
    }

    /// The origin widened by an index from `low` to `high` scaled by `stride`,
    /// which lands where no walk alone does.
    fn indexed(self, low: i64, high: i64, stride: u64) -> Option<Self> {
        let scale = i64::try_from(stride).ok()?;
        Some(Self {
            root: self.root,
            low: self.low.checked_add(low.checked_mul(scale)?)?,
            high: self.high.checked_add(high.checked_mul(scale)?)?,
            stride: greatest_common_step(self.stride, stride),
            walked: None,
        })
    }
}

/// Where an induction starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Start {
    /// A constant address.
    Address {
        /// The value holding the constant address.
        root: SsaVarId,
        /// The displacement from the root's address, in bytes.
        offset: i64,
        /// The address.
        at: u64,
        /// The space it is in.
        space: AddressBase,
    },
    /// A number.
    Number(i64),
}

impl Start {
    /// Whether `other` starts at the same place, however it is materialised.
    fn same_place(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Address { at, space, .. },
                Self::Address {
                    at: other_at,
                    space: other_space,
                    ..
                },
            ) => at == other_at && space == other_space,
            (Self::Number(one), Self::Number(other)) => one == other,
            _ => false,
        }
    }
}

/// A value a loop steps by a constant: `phi(start, phi + step)`.
#[derive(Debug, Clone, Copy)]
struct Induction {
    /// The block whose phi it is: the loop's header.
    header: usize,
    /// Where it starts.
    start: Start,
    /// What each step adds.
    step: i64,
    /// The stepped value, the phi's operand from the loop.
    stepped: SsaVarId,
    /// The block the stepped value comes back from.
    latch: usize,
}

/// The step two movements together land on: the greatest common divisor of
/// their strides, where a stride of `0` does not move. Every place a value
/// moved by both reaches lies a whole number of these from where it started.
#[must_use]
pub fn greatest_common_step(one: u64, other: u64) -> u64 {
    let (mut one, mut other) = (one, other);
    while other != 0 {
        (one, other) = (other, one.checked_rem(other).unwrap_or(0));
    }
    one
}

/// The values an index may add to an address, from one function's value
/// ranges, computed at the first index asked about.
pub struct IndexBounds<'a, T: Target> {
    ir: &'a SsaFunction<T>,
    ptr_size: PointerSize,
    budget: &'a WorkBudget,
    /// The range a host states for each call's result
    /// ([`ranges::analyze`]).
    results: Option<&'a BTreeMap<SsaVarId, ValueRange>>,
    /// The function's value ranges and dominators once asked: `None` inside
    /// when the fixpoint did not converge, which bounds nothing.
    ranges: Option<Option<(ValueRanges, DominatorTree)>>,
}

impl<'a, T: Target> IndexBounds<'a, T> {
    /// The bounds of `ir`'s indices, its value ranges to run within
    /// `budget`'s [`Budget::RangeRounds`](crate::analysis::budget::Budget::RangeRounds)
    /// once asked.
    ///
    /// # Arguments
    ///
    /// * `ir` - The function.
    /// * `ptr_size` - The target pointer width, at which an address adds an
    ///   index.
    /// * `budget` - The work budget the value ranges run in.
    #[must_use]
    pub const fn new(
        ir: &'a SsaFunction<T>,
        ptr_size: PointerSize,
        budget: &'a WorkBudget,
    ) -> Self {
        Self {
            ir,
            ptr_size,
            budget,
            results: None,
            ranges: None,
        }
    }

    /// These bounds, with `results` the range the host states for each
    /// call's result ([`ranges::analyze`]).
    #[must_use]
    pub const fn with_results(mut self, results: &'a BTreeMap<SsaVarId, ValueRange>) -> Self {
        self.results = Some(results);
        self
    }

    /// The least and greatest value `index` may add to an address at
    /// `block`: its range there, where the guards that dominate it bound it
    /// ([`ValueRanges::range_at`]), read at the pointer's width, as an
    /// address adds it. `None` where nothing bounds it, or the ranges did not
    /// converge.
    ///
    /// An index as wide as a pointer wraps with the address, so its range is
    /// read in the signed window, where a byte sign-extended into a word,
    /// `[0, 127] ∪ [2^32 - 128, 2^32 - 1]`, is the `[-128, 127]` it is. One
    /// narrower is bounded only where both of its extensions agree: where it
    /// lies below its sign bit.
    pub fn at(&mut self, index: SsaVarId, block: usize) -> Option<(i64, i64)> {
        let width = self.ir.bit_width(index)?;
        let pointer = self.ptr_size.bits();
        let (ir, budget, results) = (self.ir, self.budget, self.results);
        let (ranges, dominators) = self
            .ranges
            .get_or_insert_with(|| {
                let none = BTreeMap::new();
                let ranges = ranges::analyze(ir, budget, results.unwrap_or(&none)).ok()?;
                let cfg = SsaCfg::from_ssa(ir);
                Some((ranges, compute_dominators(&cfg, cfg.entry())))
            })
            .as_ref()?;
        let range = ranges.range_at(ir, dominators, index, block);
        let read = if width == pointer {
            range.wrap(pointer, true)
        } else if width < pointer {
            let unsigned = range.wrap(width, false);
            let sign = 1i64.checked_shl(width.checked_sub(1)?)?;
            if unsigned.max()? >= sign {
                return None;
            }
            unsigned
        } else {
            return None;
        };
        if matches!(read, ValueRange::Bottom) {
            return None;
        }
        Some((read.min()?, read.max()?))
    }
}

/// Where a value may point: a constant address, and the displacements from
/// it it may be at ([`address_origins`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddressOrigin {
    /// The constant address the displacements are measured from.
    pub root: u64,
    /// The least displacement, in bytes.
    pub low: i64,
    /// The greatest.
    pub high: i64,
    /// The step between the displacements; `0` for one that does not move.
    pub stride: u64,
}

/// What a function's values hold beyond what its own operations define, as
/// the caller can say: the address a cell nothing but the loader writes holds
/// ([`ImageData::address_at`]), the stores that may reach a load
/// ([`MemorySsa::stored_values`]), what a pointer may point at
/// ([`PointsTo`]), asked in that order, and where the host states a value
/// may point ([`AddressOrigin`]): a parameter every call reaching the function
/// passes a constant address, or a load of what the program kept somewhere
/// the function's own memory does not follow.
#[derive(Clone, Copy)]
pub struct LoadedValues<'a, T: Target> {
    /// The image the function's code reads: a load from a read-only cell
    /// holding an address reads that address.
    pub image: Option<&'a dyn ImageData>,
    /// The function's memory versions.
    pub memory: Option<&'a MemorySsa<T>>,
    /// The function's points-to relation.
    pub points_to: Option<&'a PointsTo>,
    /// Where the host states each of these values may point, asked before
    /// anything else: a value the function is entered with, where every call
    /// reaching it is known and passes it a constant address, or a load of an
    /// address the program kept in a record or handed through its calls.
    pub stated: Option<&'a BTreeMap<SsaVarId, Vec<AddressOrigin>>>,
    /// The range the host states for each call's result, which bounds an
    /// index computed from it ([`ranges::analyze`]).
    pub results: Option<&'a BTreeMap<SsaVarId, ValueRange>>,
}

impl<T: Target> LoadedValues<'_, T> {
    /// Nothing known of what a load reads.
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            image: None,
            memory: None,
            points_to: None,
            stated: None,
            results: None,
        }
    }
}

impl<T: Target> fmt::Debug for LoadedValues<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadedValues")
            .field("image", &self.image.is_some())
            .field("memory", &self.memory.is_some())
            .field("points_to", &self.points_to)
            .field("stated", &self.stated)
            .field("results", &self.results)
            .finish()
    }
}

/// The most constants an index may be for the addresses it moves to to be
/// told apart ([`Reaches::choices`]).
const MOST_CHOICES: usize = 16;

/// The most cells a load's address may be at for the addresses they hold to
/// be read one by one ([`Reaches::sealed`]).
const MOST_SEALED_CELLS: u64 = 256;

/// One function's addresses and the bounds of its indices, asked lazily.
struct Reaches<'a, T: Target> {
    ir: &'a SsaFunction<T>,
    constants: &'a BTreeMap<SsaVarId, u64>,
    ptr_size: PointerSize,
    bounds: IndexBounds<'a, T>,
    /// What the function's loads may read.
    loaded: &'a LoadedValues<'a, T>,
    /// The block of the access being resolved, where its index is bounded by
    /// the guards that dominate it.
    block: usize,
    /// How many resolutions ([`Self::origins`]) are open, each inside the
    /// last: a chain of loads or sums the binary makes as long as it likes.
    depth: usize,
}

/// How many resolutions may be open, each inside the last, before what is
/// asked of the deepest is unknown: deeper, a chain the binary makes as long
/// as it likes would exhaust the stack.
const MOST_DEPTH: usize = 64;

impl<T: Target> Reaches<'_, T> {
    /// Where `value` may point, or `None` where it may point anywhere.
    ///
    /// A constant address points at itself. A phi points wherever each of its
    /// operands does, or - where it walks - wherever the walk goes
    /// ([`Self::walked`]). A load points wherever each value the stores that
    /// may reach it wrote does ([`MemorySsa::stored_values`]): a table's
    /// address kept in a cell of the frame and read back. A sum of two values
    /// no scale tells apart points wherever one of them does, moved by what
    /// the other's range allows. Anything else is unknown, as is anything
    /// asked more than [`MOST_DEPTH`] resolutions deep. `seen` holds the phis
    /// being resolved, so a cycle that is no walk resolves to nothing.
    fn origins(&mut self, value: SsaVarId, seen: &mut Vec<SsaVarId>) -> Option<Vec<Origin>> {
        if self.depth >= MOST_DEPTH {
            return None;
        }
        self.depth = self.depth.saturating_add(1);
        let found = self.resolve(value, seen);
        self.depth = self.depth.saturating_sub(1);
        found
    }

    /// [`Self::origins`] of `value`, one resolution deeper.
    fn resolve(&mut self, value: SsaVarId, seen: &mut Vec<SsaVarId>) -> Option<Vec<Origin>> {
        let address = AddressWalk::new(self.ir, self.ptr_size).normalize(value);
        // An index that is one of a few constants moves the address by each.
        let choices = match address.index {
            Some(index) => {
                let stride = i64::try_from(address.stride_bytes).ok()?;
                self.choices(index, &mut Vec::new())?
                    .into_iter()
                    .map(|choice| choice.checked_mul(stride))
                    .collect::<Option<Vec<i64>>>()?
            }
            None => vec![0],
        };
        let origins = if address.base.is_absolute() {
            let root = *self.constants.get(&address.root)?;
            let low = whole_bytes(address.root_offset_bits)?;
            vec![Origin {
                root,
                low,
                high: low,
                stride: 0,
                walked: None,
            }]
        } else {
            let AddressBase::Value(base) = address.base else {
                return None;
            };
            let moved = whole_bytes(address.offset_bits)?;
            self.base_origins(base, seen)?
                .into_iter()
                .map(|origin| origin.moved(moved))
                .collect::<Option<Vec<Origin>>>()?
        };
        origins
            .iter()
            .flat_map(|origin| choices.iter().map(|choice| origin.moved(*choice)))
            .collect()
    }

    /// The constants `value` may be, where it is one or a phi of a few
    /// ([`MOST_CHOICES`] at most), phis inside phis at most [`MOST_DEPTH`]
    /// deep; `None` where it may be anything else.
    fn choices(&self, value: SsaVarId, seen: &mut Vec<SsaVarId>) -> Option<Vec<i64>> {
        if let Some(constant) = self.ir.const_i64(value) {
            return Some(vec![constant]);
        }
        if seen.len() >= MOST_DEPTH {
            return None;
        }
        if seen.contains(&value) {
            return Some(Vec::new());
        }
        let (_, phi) = self.ir.find_phi_defining(value)?;
        seen.push(value);
        let mut choices = Vec::new();
        for operand in phi.operands() {
            choices.extend(self.choices(operand.value(), seen)?);
            if choices.len() > MOST_CHOICES {
                return None;
            }
        }
        seen.pop();
        choices.sort_unstable();
        choices.dedup();
        (!choices.is_empty()).then_some(choices)
    }

    /// The values the load defining `base` may read, where the points-to
    /// relation says ([`PointsTo`]): those each store that may write a cell it
    /// reads wrote, a field of a record read back through a pointer to the
    /// record. `None` where a cell it reads escaped, which holds anything
    /// ([`UNKNOWN_LOC`]), where a write that may reach one stores no stated
    /// value, or where the cells hold something those stores did not write (a
    /// block copy).
    fn written(&self, base: SsaVarId) -> Option<Vec<SsaVarId>> {
        let points_to = self.loaded.points_to?;
        let site = self.ir.variable(base)?.def_site();
        let load = self
            .ir
            .block(site.block())?
            .instructions()
            .get(site.index()?)?;
        let mut address = None;
        load.op().for_each_addressed_access(|access| {
            if access.reads && access.value == Some(base) {
                address = Some(access.addr);
            }
        });
        let cells = points_to.get(address?.as_u32());
        if cells.is_empty() || cells.iter().any(|cell| points_to.cell(*cell).is_none()) {
            return None;
        }
        let mut held: BTreeSet<u32> = BTreeSet::new();
        for cell in cells {
            held.extend(points_to.get(*cell).iter().copied());
        }
        if held.contains(&UNKNOWN_LOC) {
            return None;
        }
        let mut written = Vec::new();
        let mut unknown = false;
        for instruction in self
            .ir
            .blocks()
            .iter()
            .flat_map(|block| block.instructions())
        {
            instruction.op().for_each_addressed_access(|access| {
                if !access.writes {
                    return;
                }
                // A pointer that may be anywhere reaches what escaped, which
                // holds anything and was turned away above: a cell held in no
                // escaped memory is written only through pointers to it.
                if points_to.get(access.addr.as_u32()).is_disjoint(cells) {
                    return;
                }
                match access.value.filter(|_| !access.reads) {
                    Some(value) => written.push(value),
                    None => unknown = true,
                }
            });
        }
        if unknown {
            return None;
        }
        let stored: BTreeSet<u32> = written
            .iter()
            .flat_map(|value| points_to.get(value.as_u32()).iter().copied())
            .collect();
        if !held.is_subset(&stored) {
            return None;
        }
        written.sort_unstable();
        written.dedup();
        Some(written)
    }

    /// Where `base` may point where a load defines it from cells nothing but
    /// the loader writes: the address each cell its address may be at holds
    /// ([`ImageData::address_at`]), where every such cell lies in read-only
    /// memory and holds one, and there are at most [`MOST_SEALED_CELLS`] of
    /// them. `None` otherwise.
    fn sealed(&mut self, base: SsaVarId, seen: &mut Vec<SsaVarId>) -> Option<Vec<Origin>> {
        let image = self.loaded.image?;
        let site = self.ir.variable(base)?.def_site();
        let load = self
            .ir
            .block(site.block())?
            .instructions()
            .get(site.index()?)?;
        let mut read = None;
        load.op().for_each_addressed_access(|access| {
            if access.reads && !access.writes && access.value == Some(base) {
                read = Some((access.addr, access.width_bits));
            }
        });
        let (address, bits) = read?;
        let width = u64::from(bits?.checked_div(8)?);
        let cells = self.origins(address, seen)?;
        let mut origins = Vec::new();
        let mut count = 0u64;
        for cell in cells {
            if cell.walked.is_some() {
                return None;
            }
            let span = u64::try_from(cell.high.checked_sub(cell.low)?).ok()?;
            // A fixed cell is one; a stepped one is every stride from its
            // least displacement to its greatest, a whole number of strides
            // apart.
            let steps = match cell.stride {
                0 if span == 0 => 0,
                0 => return None,
                stride if span.checked_rem(stride)? == 0 => span.checked_div(stride)?,
                _ => return None,
            };
            count = count.checked_add(steps.checked_add(1)?)?;
            if count > MOST_SEALED_CELLS {
                return None;
            }
            for step in 0..=steps {
                let displacement = cell
                    .low
                    .checked_add(i64::try_from(step.checked_mul(cell.stride)?).ok()?)?;
                let at = cell.root.checked_add_signed(displacement)?;
                let region = image.region(at)?;
                if region.mutability != Mutability::ReadOnly || !region.covers(at, width) {
                    return None;
                }
                origins.push(Origin {
                    root: image.address_at(at, width)?,
                    low: 0,
                    high: 0,
                    stride: 0,
                    walked: None,
                });
            }
        }
        (!origins.is_empty()).then_some(origins)
    }

    /// The values `base` may hold where a load defines it: what the stores
    /// that may reach the load wrote ([`MemorySsa::stored_values`]), or
    /// `None` where no load defines it or anything else may.
    fn stored(&self, base: SsaVarId) -> Option<Vec<SsaVarId>> {
        let memory = self.loaded.memory?;
        let site = self.ir.variable(base)?.def_site();
        memory.stored_values(self.ir, site.block(), site.index()?)
    }

    /// Where the value an address is measured from may point
    /// ([`Self::origins`]).
    fn base_origins(&mut self, base: SsaVarId, seen: &mut Vec<SsaVarId>) -> Option<Vec<Origin>> {
        // A value points where the host says it does.
        if let Some(origins) = self.loaded.stated.and_then(|stated| stated.get(&base)) {
            return Some(
                origins
                    .iter()
                    .map(|origin| Origin {
                        root: origin.root,
                        low: origin.low,
                        high: origin.high,
                        stride: origin.stride,
                        walked: None,
                    })
                    .collect(),
            );
        }
        if let Some(SsaOp::Add { left, right, .. }) = self.ir.get_definition(base) {
            let (left, right) = (*left, *right);
            for (pointer, index) in [(left, right), (right, left)] {
                let Some((low, high)) = self.bounds.at(index, self.block) else {
                    continue;
                };
                if let Some(origins) = self.origins(pointer, seen) {
                    return origins
                        .into_iter()
                        .map(|origin| origin.indexed(low, high, 1))
                        .collect();
                }
            }
            return None;
        }
        if seen.contains(&base) {
            return None;
        }
        // Memory nothing but the loader writes holds what the image says.
        if let Some(origins) = self.sealed(base, seen) {
            return Some(origins);
        }
        if let Some(stored) = self.stored(base) {
            seen.push(base);
            let mut origins = Vec::new();
            for value in stored {
                let Some(more) = self.origins(value, seen) else {
                    seen.pop();
                    return None;
                };
                origins.extend(more);
            }
            seen.pop();
            return (!origins.is_empty()).then_some(origins);
        }
        let Some((_, phi)) = self.ir.find_phi_defining(base) else {
            let written = self.written(base)?;
            seen.push(base);
            let mut origins = Vec::new();
            for value in written {
                let Some(more) = self.origins(value, seen) else {
                    seen.pop();
                    return None;
                };
                origins.extend(more);
            }
            seen.pop();
            return (!origins.is_empty()).then_some(origins);
        };
        if let Some(walk) = self.walked(base) {
            return Some(vec![walk]);
        }
        seen.push(base);
        let mut origins = Vec::new();
        for operand in phi.operands() {
            if operand.value() == base {
                continue;
            }
            let Some(more) = self.origins(operand.value(), seen) else {
                seen.pop();
                return None;
            };
            origins.extend(more);
        }
        seen.pop();
        (!origins.is_empty()).then_some(origins)
    }

    /// The induction `value` is, where it is one: `phi(start, value + step)`,
    /// every operand but one the one start, a constant address or a number,
    /// and the one left stepping the phi itself by a constant from the block
    /// that closes the loop.
    fn induction(&self, value: SsaVarId) -> Option<Induction> {
        let (header, phi) = self.ir.find_phi_defining(value)?;
        let mut start: Option<Start> = None;
        let mut step: Option<(SsaVarId, usize, i64)> = None;
        for operand in phi.operands() {
            let address = AddressWalk::new(self.ir, self.ptr_size).normalize(operand.value());
            if address.index.is_some() {
                return None;
            }
            if address.base == AddressBase::Value(value) {
                if step.is_some() {
                    return None;
                }
                let bytes = whole_bytes(address.offset_bits).filter(|bytes| *bytes != 0)?;
                step = Some((operand.value(), operand.predecessor(), bytes));
                continue;
            }
            let from = if address.base.is_absolute()
                && let Some(held) = self.constants.get(&address.root)
            {
                let offset = whole_bytes(address.root_offset_bits)?;
                Start::Address {
                    root: address.root,
                    offset,
                    at: held.checked_add_signed(offset)?,
                    space: address.base,
                }
            } else {
                Start::Number(self.ir.const_i64(operand.value())?)
            };
            match start {
                Some(known) if !known.same_place(&from) => return None,
                Some(_) => {}
                None => start = Some(from),
            }
        }
        let (stepped, latch, step) = step?;
        Some(Induction {
            header,
            start: start?,
            step,
            stepped,
            latch,
        })
    }

    /// The most steps `walk` takes past its start, where the block that
    /// closes its loop ends it by a comparison of an induction stepped there
    /// with a constant bound: the walk itself, compared with an address, or a
    /// counter stepped beside it, compared with a number.
    ///
    /// The loop goes back to its header only while the comparison lets it, so
    /// every step is one it let through: up to the one before a bound met on
    /// inequality, which the induction must meet exactly, or before one an
    /// order stops it at. An order of two addresses is read unsigned, as the
    /// displacements' order only where no step wraps the space, and an order
    /// of numbers only where none leaves the counter's window. Anything else
    /// bounds nothing.
    fn trip(&self, walk: &Induction) -> Option<i64> {
        let Some(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned,
            true_target,
            false_target,
        }) = self.ir.block(walk.latch)?.control_terminator()
        else {
            return None;
        };
        let (counter, bound, cmp) = [(*left, *right, *cmp), (*right, *left, cmp.swapped())]
            .into_iter()
            .find_map(|(tested, bound, cmp)| {
                let address = AddressWalk::new(self.ir, self.ptr_size).normalize(tested);
                let AddressBase::Value(stepped) = address.base else {
                    return None;
                };
                let counter = self.induction(stepped)?;
                (address.index.is_none()
                    && counter.stepped == tested
                    && counter.header == walk.header
                    && counter.latch == walk.latch)
                    .then_some((counter, bound, cmp))
            })?;
        // The relation under which the loop goes on.
        let goes_on = match (*true_target == walk.header, *false_target == walk.header) {
            (true, false) => cmp,
            (false, true) => cmp.negated(),
            _ => return None,
        };
        // Where the counter starts and where its bound is, as numbers, and
        // the window its values must stay in for an order to be theirs.
        let (first, end, window) = match counter.start {
            Start::Address { at, space, .. } => {
                let end = AddressWalk::new(self.ir, self.ptr_size).normalize(bound);
                if end.index.is_some() || end.base != space {
                    return None;
                }
                let end_at = self
                    .constants
                    .get(&end.root)?
                    .checked_add_signed(whole_bytes(end.root_offset_bits)?)?;
                if !*unsigned && goes_on != CmpKind::Ne {
                    return None;
                }
                let top = i128::from(self.ptr_size.mask_unsigned(u64::MAX));
                (i128::from(at), i128::from(end_at), (0, top))
            }
            Start::Number(first) => {
                let width = self
                    .ir
                    .bit_width(counter.stepped)
                    .filter(|width| (1..=64).contains(width))
                    .unwrap_or_else(|| self.ptr_size.bits());
                let span = 1i128.checked_shl(width)?;
                let window = if *unsigned {
                    (0, span.checked_sub(1)?)
                } else {
                    let half = span.checked_div(2)?;
                    (half.checked_neg()?, half.checked_sub(1)?)
                };
                let end = i128::from(self.ir.const_i64(bound)?);
                // A number read in the other window is the same bits.
                let read = |value: i128| {
                    if value < window.0 {
                        value.checked_add(span)
                    } else if value > window.1 {
                        value.checked_sub(span)
                    } else {
                        Some(value)
                    }
                };
                (read(i128::from(first))?, read(end)?, window)
            }
        };
        // Mirrored, a count down is a count up.
        let step = i128::from(counter.step);
        let (distance, step, goes_on) = if step < 0 {
            (
                first.checked_sub(end)?,
                step.checked_neg()?,
                goes_on.swapped(),
            )
        } else {
            (end.checked_sub(first)?, step, goes_on)
        };
        let steps = match goes_on {
            CmpKind::Ne if distance > 0 && distance.checked_rem(step)? == 0 => {
                distance.checked_div(step)?.checked_sub(1)?
            }
            CmpKind::Lt if distance > 0 => distance.checked_sub(1)?.checked_div(step)?,
            CmpKind::Le if distance >= 0 => distance.checked_div(step)?,
            CmpKind::Lt | CmpKind::Le => 0,
            _ => return None,
        };
        // Every value the counter is compared at stays in its window, the
        // last one the step past the walk's furthest.
        if goes_on != CmpKind::Ne {
            let last = first.checked_add(
                steps
                    .checked_add(1)?
                    .checked_mul(i128::from(counter.step))?,
            )?;
            if [first, last, end]
                .iter()
                .any(|value| *value < window.0 || *value > window.1)
            {
                return None;
            }
        }
        i64::try_from(steps).ok()
    }

    /// Where the pointer `walker` walks, where a loop moves it from one
    /// constant address in constant steps a comparison ends ([`Self::trip`]).
    fn walked(&self, walker: SsaVarId) -> Option<Origin> {
        let walk = self.induction(walker)?;
        let Start::Address { root, offset, .. } = walk.start else {
            return None;
        };
        let root = *self.constants.get(&root)?;
        let furthest = self.trip(&walk)?.checked_mul(walk.step)?;
        let (low, high) = if furthest < 0 {
            (offset.checked_add(furthest)?, offset)
        } else {
            (offset, offset.checked_add(furthest)?)
        };
        Some(Origin {
            root,
            low,
            high,
            stride: walk.step.unsigned_abs(),
            walked: Some(walk.latch),
        })
    }
}

/// Where `value` of `ir` may point ([`AddressOrigin`]): a constant address
/// `constants` holds, through what its phis, sums, loads and parameters
/// reach, as [`for_each_reach`] resolves an access's address; `None` where
/// anything else may define it. A host asks it of the arguments a call
/// passes, to say where the callee's parameters point
/// ([`LoadedValues::stated`]).
///
/// # Arguments
///
/// * `ir` - The function.
/// * `constants` - The values known to hold constant addresses, and the
///   address each holds.
/// * `ptr_size` - The target pointer width.
/// * `budget` - The work budget the value ranges run in.
/// * `loaded` - What the function's loads and parameters hold.
/// * `value` - The value asked about.
/// * `block` - The block it is asked at, whose guards bound an index in it.
#[must_use]
pub fn address_origins<T: Target>(
    ir: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, u64>,
    ptr_size: PointerSize,
    budget: &WorkBudget,
    loaded: &LoadedValues<'_, T>,
    value: SsaVarId,
    block: usize,
) -> Option<Vec<AddressOrigin>> {
    let mut reaches = Reaches {
        ir,
        constants,
        ptr_size,
        bounds: match loaded.results {
            Some(results) => IndexBounds::new(ir, ptr_size, budget).with_results(results),
            None => IndexBounds::new(ir, ptr_size, budget),
        },
        loaded,
        block,
        depth: 0,
    };
    let mut origins: Vec<AddressOrigin> = reaches
        .origins(value, &mut Vec::new())?
        .into_iter()
        .map(|origin| AddressOrigin {
            root: origin.root,
            low: origin.low,
            high: origin.high,
            stride: origin.stride,
        })
        .collect();
    origins.sort_unstable();
    origins.dedup();
    Some(origins)
}

/// `bits` as whole bytes, or `None` where they are not.
fn whole_bytes(bits: i64) -> Option<i64> {
    (bits.checked_rem(8)? == 0)
        .then(|| bits.checked_div(8))
        .flatten()
}

/// Visits every access of `ir` that moves through a constant address, with
/// the bytes it may touch ([`Reach`]): one built on such an address, or on a
/// value read back from where the function stored one, which `memory` says
/// ([`MemorySsa::stored_values`]).
///
/// An access moves where its address adds an index the function's value
/// ranges bound, or is measured from a pointer a loop walks from one constant
/// address to a constant bound. Its address is built on a value `constants`
/// holds an address for, directly or through phis whose every operand is.
/// Where the index is unbounded, an operand of a phi is not built on a
/// constant, or the access does not say how many bytes it moves, it is not
/// visited: the bytes it may touch are not known. An access that does not
/// move is not visited either: it proves nothing about the bytes around it.
///
/// The value ranges are computed once, at the first index asked about, within
/// `budget`'s [`Budget::RangeRounds`](crate::analysis::budget::Budget::RangeRounds);
/// a fixpoint that does not converge bounds no index.
///
/// # Arguments
///
/// * `ir` - The function.
/// * `constants` - The values known to hold constant addresses, and the
///   address each holds.
/// * `ptr_size` - The target pointer width.
/// * `budget` - The work budget the value ranges run in.
/// * `loaded` - What the function's loads may read ([`LoadedValues`]):
///   without it a value read from memory is built on nothing known.
/// * `visit` - Called for each access that moves, once per constant it may
///   be built on, in block and instruction order.
pub fn for_each_reach<T: Target>(
    ir: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, u64>,
    ptr_size: PointerSize,
    budget: &WorkBudget,
    loaded: &LoadedValues<'_, T>,
    mut visit: impl FnMut(Reach),
) {
    if constants.is_empty() && loaded.stated.is_none_or(BTreeMap::is_empty) {
        return;
    }
    let mut reaches = Reaches {
        ir,
        constants,
        ptr_size,
        bounds: match loaded.results {
            Some(results) => IndexBounds::new(ir, ptr_size, budget).with_results(results),
            None => IndexBounds::new(ir, ptr_size, budget),
        },
        loaded,
        block: 0,
        depth: 0,
    };
    let mut accesses = Vec::new();
    for (index, block) in ir.blocks().iter().enumerate() {
        for instruction in block.instructions() {
            instruction
                .op()
                .for_each_addressed_access(|access| accesses.push((index, access)));
        }
    }
    for (block, access) in accesses {
        reaches.block = block;
        let Some(width) = access
            .width_bits
            .filter(|bits| *bits > 0 && bits.is_multiple_of(8))
            .and_then(|bits| i64::from(bits).checked_div(8))
        else {
            continue;
        };
        let address = AddressWalk::new(ir, ptr_size).normalize(access.addr);
        let origins = match address.base {
            base if base.is_absolute() => {
                let Some(&root) = constants.get(&address.root) else {
                    continue;
                };
                let Some(low) = whole_bytes(address.root_offset_bits) else {
                    continue;
                };
                vec![Origin {
                    root,
                    low,
                    high: low,
                    stride: 0,
                    walked: None,
                }]
            }
            AddressBase::Value(base) => {
                let Some(moved) = whole_bytes(address.offset_bits) else {
                    continue;
                };
                let Some(origins) = reaches.base_origins(base, &mut Vec::new()) else {
                    continue;
                };
                let Some(origins) = origins
                    .into_iter()
                    .map(|origin| origin.moved(moved))
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                origins
            }
            _ => continue,
        };
        let origins = match address.index {
            Some(index) => {
                let Some((low, high)) = reaches.bounds.at(index, reaches.block) else {
                    continue;
                };
                let Some(origins) = origins
                    .into_iter()
                    .map(|origin| origin.indexed(low, high, address.stride_bytes))
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                origins
            }
            None => origins,
        };
        for origin in origins {
            if origin.stride == 0 || origin.low == origin.high {
                continue;
            }
            let Some(end) = origin.high.checked_add(width) else {
                continue;
            };
            visit(Reach {
                root: origin.root,
                bytes: origin.low..end,
                stride: origin.stride,
                walk: origin.walked,
                access,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::{
        analysis::{AnalysisContext, image::ImageRegion, pointsto::analyze_function},
        ir::{
            CallData, Callee, ConstValue, PhiNode, PhiOperand, SsaBlock, SsaInstruction,
            ops::AccessQualifiers,
            variable::{DefSite, VariableOrigin},
        },
        target::Endianness,
        testing::{MACHINE, MockTarget, MockType},
    };

    const TABLE: u32 = 0x1_142c;

    /// A function under construction: its blocks, and the values it declares.
    struct Built {
        ir: SsaFunction<MockTarget>,
        blocks: Vec<SsaBlock<MockTarget>>,
        /// Where the function's parameters point, as a host bound them.
        parameters: BTreeMap<SsaVarId, Vec<AddressOrigin>>,
        /// The image the function reads, where it reads one.
        image: Option<Got>,
        /// The range a host states for each call's result.
        results: BTreeMap<SsaVarId, ValueRange>,
    }

    /// A global offset table at [`GOT`]: words holding addresses, read-only
    /// where `sealed`, and writable otherwise.
    struct Got {
        cells: BTreeMap<u64, u64>,
        sealed: bool,
    }

    const GOT: u32 = 0x2_fd54;

    impl ImageData for Got {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            let start = u64::from(GOT);
            (start..start + 0x2ac)
                .contains(&address)
                .then(|| ImageRegion {
                    start,
                    end: start + 0x2ac,
                    mutability: if self.sealed {
                        Mutability::ReadOnly
                    } else {
                        Mutability::Writable
                    },
                })
        }

        fn bytes(&self, _address: u64, _len: usize) -> Option<&[u8]> {
            None
        }

        fn address_at(&self, address: u64, len: u64) -> Option<u64> {
            self.cells.get(&address).copied().filter(|_| len == 4)
        }
    }

    impl Built {
        fn new(blocks: usize) -> Self {
            Self {
                ir: SsaFunction::<MockTarget>::with_capacity(0, 0, blocks, 16),
                blocks: (0..blocks).map(SsaBlock::new).collect(),
                parameters: BTreeMap::new(),
                image: None,
                results: BTreeMap::new(),
            }
        }

        fn value(&mut self, ty: MockType) -> SsaVarId {
            self.ir
                .create_variable(VariableOrigin::Phi, 0, DefSite::entry(), ty)
        }

        fn push(&mut self, block: usize, op: SsaOp<MockTarget>) {
            if let Some(block) = self.blocks.get_mut(block) {
                block.add_instruction(SsaInstruction::new((), op));
            }
        }

        /// `dest = op(dest)` in `block`, for a fresh `dest` of type `ty`.
        fn define(
            &mut self,
            block: usize,
            ty: MockType,
            op: impl FnOnce(SsaVarId) -> SsaOp<MockTarget>,
        ) -> SsaVarId {
            let position = self
                .blocks
                .get(block)
                .map_or(0, |block| block.instructions().len());
            let dest = self.ir.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(block, position),
                ty,
            );
            self.push(block, op(dest));
            dest
        }

        /// The pointer a phi of `block` defines, its operands stated later
        /// ([`Self::close`]).
        fn phi(&mut self, block: usize) -> SsaVarId {
            self.ir
                .create_variable(VariableOrigin::Phi, 0, DefSite::phi(block), MockType::Ptr)
        }

        /// The phi of `block` defining `dest`, over `operands`, each a value
        /// and the block it comes from.
        fn close(&mut self, block: usize, dest: SsaVarId, operands: &[(SsaVarId, usize)]) {
            let mut phi = PhiNode::new(dest, VariableOrigin::Phi);
            for (value, from) in operands {
                phi.add_operand(PhiOperand::new(*value, *from));
            }
            if let Some(block) = self.blocks.get_mut(block) {
                block.add_phi(phi);
            }
        }

        fn symbol(&mut self, block: usize, address: u32) -> SsaVarId {
            self.define(block, MockType::Ptr, |dest| SsaOp::Const {
                dest,
                value: ConstValue::Symbol(address),
            })
        }

        fn number(&mut self, block: usize, value: i32) -> SsaVarId {
            self.define(block, MockType::I32, |dest| SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            })
        }

        fn step(
            &mut self,
            block: usize,
            base: SsaVarId,
            index: Option<SsaVarId>,
            stride: u64,
            offset: i64,
        ) -> SsaVarId {
            self.define(block, MockType::Ptr, |dest| SsaOp::PtrAdd {
                dest,
                base,
                index,
                stride,
                offset,
                result_type: MockType::Ptr,
            })
        }

        fn load(&mut self, block: usize, addr: SsaVarId, ty: MockType) -> SsaVarId {
            self.define(block, ty, |dest| SsaOp::LoadIndirect {
                dest,
                addr,
                value_type: ty,
                access: AccessQualifiers::FLAT,
            })
        }

        /// The reaches of the finished function, its symbols the constants,
        /// with what its loads read known.
        fn reaches(self) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
            self.reaches_knowing(true)
        }

        /// The reaches of the finished function, its symbols the constants,
        /// with the points-to relation where `points_to`.
        fn reaches_knowing(
            mut self,
            points_to: bool,
        ) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
            for block in self.blocks {
                self.ir.add_block(block);
            }
            self.ir.recompute_uses();
            let constants: BTreeMap<SsaVarId, u64> = self
                .ir
                .blocks()
                .iter()
                .flat_map(|block| block.instructions())
                .filter_map(|instruction| match instruction.op() {
                    SsaOp::Const {
                        dest,
                        value: ConstValue::Symbol(address),
                    } => Some((*dest, u64::from(*address))),
                    _ => None,
                })
                .collect();
            let memory = MemorySsa::build(&self.ir, &AnalysisContext::new(MACHINE));
            let relation = analyze_function(&self.ir, &WorkBudget::DEFAULT, None);
            let mut found = Vec::new();
            for_each_reach(
                &self.ir,
                &constants,
                PointerSize::Bit32,
                &WorkBudget::DEFAULT,
                &LoadedValues {
                    image: self.image.as_ref().map(|image| image as &dyn ImageData),
                    memory: Some(&memory),
                    points_to: points_to.then_some(&relation),
                    stated: Some(&self.parameters).filter(|bound| !bound.is_empty()),
                    results: Some(&self.results).filter(|stated| !stated.is_empty()),
                },
                |reach| found.push((reach.root, reach.bytes, reach.stride, reach.walk)),
            );
            found
        }
    }

    /// A byte read through `&table + 129` by a signed character, as dash
    /// reads its syntax tables: `(int32_t)(int8_t)c` is `[-128, 127]`, and
    /// the read touches the bytes from `1` to `256` past the table's address.
    /// The same read through a phi of two tables touches the same bytes of
    /// each.
    fn syntax_read(
        tables: &[u32],
        index_type: MockType,
    ) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
        let mut built = Built::new(tables.len().saturating_add(1));
        let join = tables.len();
        let mut entered = Vec::new();
        for (block, table) in tables.iter().enumerate() {
            let address = built.symbol(block, *table);
            let base = built.step(block, address, None, 0, 129);
            built.push(block, SsaOp::Jump { target: join });
            entered.push((base, block));
        }
        let base = match entered.as_slice() {
            [(only, _)] => *only,
            _ => {
                let merged = built.phi(join);
                built.close(join, merged, &entered);
                merged
            }
        };
        let character = built.value(MockType::I8);
        let index = built.define(join, index_type, |dest| SsaOp::IntConv {
            dest,
            operand: character,
            target: index_type,
            overflow_check: false,
            unsigned: false,
        });
        let at = built.step(join, base, Some(index), 1, 0);
        built.load(join, at, MockType::U8);
        built.push(join, SsaOp::returns(None));
        built.reaches()
    }

    #[test]
    fn a_table_read_by_a_character_reaches_every_byte_the_character_can() {
        assert_eq!(
            syntax_read(&[TABLE], MockType::I32),
            vec![(u64::from(TABLE), 1..257, 1, None)]
        );
        assert_eq!(
            syntax_read(&[0x1_1328, TABLE], MockType::I32),
            vec![
                (0x1_1328, 1..257, 1, None),
                (u64::from(TABLE), 1..257, 1, None)
            ]
        );
    }

    /// A byte read by a signed character through `&table + 129` kept in a
    /// cell and read back, as dash's `readtoken1` keeps its syntax table in a
    /// local; with `clobbered`, a store through another pointer, which may be
    /// the cell, between the two.
    fn kept_syntax_read(clobbered: bool) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
        let mut built = Built::new(1);
        let cell = built.value(MockType::Ptr);
        let other = built.value(MockType::Ptr);
        let address = built.symbol(0, TABLE);
        let base = built.step(0, address, None, 0, 129);
        let store = |addr, value| SsaOp::StoreIndirect {
            addr,
            value,
            value_type: MockType::Ptr,
            access: AccessQualifiers::FLAT,
        };
        built.push(0, store(cell, base));
        if clobbered {
            let junk = built.value(MockType::Ptr);
            built.push(0, store(other, junk));
        }
        let kept = built.load(0, cell, MockType::Ptr);
        let character = built.value(MockType::I8);
        let index = built.define(0, MockType::I32, |dest| SsaOp::IntConv {
            dest,
            operand: character,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        });
        let at = built.step(0, kept, Some(index), 1, 0);
        built.load(0, at, MockType::U8);
        built.push(0, SsaOp::returns(None));
        built.reaches()
    }

    /// **A table kept in a cell and read back reaches what it would
    /// directly.** The stores that may reach the read wrote `&table + 129`,
    /// so the address read is that, and the character's read touches the
    /// table's bytes from `1` to `256`. The control: a store through a
    /// pointer that may be the cell leaves what the cell holds unknown, and
    /// the read reaches nothing.
    #[test]
    fn a_table_kept_in_a_cell_reaches_what_it_would_directly() {
        assert_eq!(
            kept_syntax_read(false),
            vec![(u64::from(TABLE), 1..257, 1, None)]
        );
        assert_eq!(kept_syntax_read(true), vec![], "the control");
    }

    /// A byte read by a signed character through `->syntax` of a record that
    /// is one of two the function allocates on its stack, each holding one
    /// table's `+ 129`, as dash's `readtoken1` reads `synstack->syntax`.
    fn record_syntax_read(points_to: bool) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
        let mut built = Built::new(3);
        let size = built.number(0, 40);
        let mut entered = Vec::new();
        for (block, table) in [(0, TABLE), (1, 0x1_1328)] {
            let record = built.define(block, MockType::Ptr, |dest| SsaOp::LocalAlloc {
                dest,
                size,
                zeroed: false,
            });
            let address = built.symbol(block, table);
            let base = built.step(block, address, None, 0, 129);
            built.push(
                block,
                SsaOp::StoreIndirect {
                    addr: record,
                    value: base,
                    value_type: MockType::Ptr,
                    access: AccessQualifiers::FLAT,
                },
            );
            built.push(block, SsaOp::Jump { target: 2 });
            entered.push((record, block));
        }
        let record = built.phi(2);
        built.close(2, record, &entered);
        let syntax = built.load(2, record, MockType::Ptr);
        let character = built.value(MockType::I8);
        let index = built.define(2, MockType::I32, |dest| SsaOp::IntConv {
            dest,
            operand: character,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        });
        let at = built.step(2, syntax, Some(index), 1, 0);
        built.load(2, at, MockType::U8);
        built.push(2, SsaOp::returns(None));
        built.reaches_knowing(points_to)
    }

    /// **A table a record holds reaches what its address would.** The record
    /// read through is one of two the function allocated, and what its first
    /// field holds is one of the two tables' `+ 129`: the read touches each
    /// table's bytes from `1` to `256`. The control: without the points-to
    /// relation, the field read back is built on nothing known.
    #[test]
    fn a_table_a_record_holds_reaches_what_its_address_would() {
        assert_eq!(
            record_syntax_read(true),
            vec![
                (u64::from(TABLE), 1..257, 1, None),
                (0x1_1328, 1..257, 1, None)
            ]
        );
        assert_eq!(record_syntax_read(false), vec![], "the control");
    }

    /// **A call returns what its callee is stated to.** `readtoken1` reads
    /// `syntax[c]` through `&table + 129`, `c` what `pgetc` returns: a
    /// character or one of the two markers below it, `[-129, 127]` as the
    /// host states it. The read reaches the table's 257 bytes. The control:
    /// stated nothing, the result is any word, and the read reaches nothing.
    #[test]
    fn a_call_returns_what_its_callee_is_stated_to() {
        let read = |stated: bool| {
            let mut built = Built::new(1);
            let character = built.define(0, MockType::I32, |dest| {
                SsaOp::from(CallData::new(Some(dest), Callee::Direct(7), Vec::new()))
            });
            if stated {
                built
                    .results
                    .insert(character, ValueRange::bounded(-129, 127));
            }
            let table = built.symbol(0, TABLE);
            let at = built.step(0, table, Some(character), 1, 129);
            built.load(0, at, MockType::U8);
            built.push(0, SsaOp::returns(None));
            built.reaches()
        };
        assert_eq!(read(true), vec![(u64::from(TABLE), 0..257, 1, None)]);
        assert_eq!(read(false), vec![], "the control");
    }

    /// A read of `table[c]` through a chain of `links` resolutions, each
    /// inside the last: pointer phis each choosing the last one where
    /// `pointers`, phis of an index each choosing the last one or `4`
    /// otherwise.
    fn read_through_a_chain(
        links: usize,
        pointers: bool,
    ) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
        let mut built = Built::new(2);
        let table = built.symbol(0, TABLE);
        let mut chained = if pointers { table } else { built.number(0, 0) };
        let four = built.number(0, 4);
        built.push(0, SsaOp::Jump { target: 1 });
        for _ in 0..links {
            let ty = if pointers {
                MockType::Ptr
            } else {
                MockType::I32
            };
            let phi = built
                .ir
                .create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), ty);
            let operands = if pointers {
                vec![(chained, 0)]
            } else {
                vec![(chained, 0), (four, 0)]
            };
            built.close(1, phi, &operands);
            chained = phi;
        }
        let base = if pointers {
            chained
        } else {
            let moved = built.step(1, table, Some(chained), 1, 0);
            let kept = built.phi(1);
            built.close(1, kept, &[(moved, 0)]);
            kept
        };
        let character = built.value(MockType::I8);
        let index = built.define(1, MockType::I32, |dest| SsaOp::IntConv {
            dest,
            operand: character,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        });
        let at = built.step(1, base, Some(index), 1, 129);
        built.load(1, at, MockType::U8);
        built.push(1, SsaOp::returns(None));
        built.reaches()
    }

    /// **A chain the binary makes as long as it likes resolves within the
    /// stack.** A read through a chain of pointer phis, and through an index
    /// chosen by a chain of phis, reaches its table where the chain is short;
    /// a hundred thousand links long, resolved on a thread of 256 KiB, the
    /// deepest is unknown and nothing reaches.
    #[test]
    fn a_chain_as_long_as_the_binary_likes_resolves_within_the_stack() {
        assert_eq!(
            read_through_a_chain(3, true),
            vec![(u64::from(TABLE), 1..257, 1, None)]
        );
        assert_eq!(read_through_a_chain(3, false).len(), 2);
        let reached = thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                (
                    read_through_a_chain(100_000, true),
                    read_through_a_chain(100_000, false),
                )
            })
            .expect("the thread starts")
            .join()
            .expect("the resolutions keep within the stack");
        assert_eq!(reached, (vec![], vec![]));
    }

    /// **A word cut to a byte indexes as the byte.** `readtoken1` reads
    /// `is_type[(signed char)c + 129]` with `c` the word `pgetc` returns, the
    /// byte added to the table's address and the read `129` past the sum:
    /// the truncation to a byte and its sign extension bound the index to
    /// `[-128, 127]`, and the read reaches the table's bytes from `1` to
    /// `256`.
    #[test]
    fn a_word_cut_to_a_byte_indexes_as_the_byte() {
        let mut built = Built::new(1);
        let word = built.value(MockType::I32);
        let byte = built.define(0, MockType::I8, |dest| SsaOp::IntConv {
            dest,
            operand: word,
            target: MockType::I8,
            overflow_check: false,
            unsigned: false,
        });
        let index = built.define(0, MockType::I32, |dest| SsaOp::IntConv {
            dest,
            operand: byte,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        });
        let table = built.symbol(0, TABLE);
        let moved = built.define(0, MockType::I32, |dest| SsaOp::Add {
            dest,
            left: table,
            right: index,
            flags: None,
        });
        let at = built.step(0, moved, None, 1, 129);
        built.load(0, at, MockType::U8);
        built.push(0, SsaOp::returns(None));
        assert_eq!(built.reaches(), vec![(u64::from(TABLE), 1..257, 1, None)]);
    }

    /// **A cell only the loader writes holds the address it binds.**
    /// `parseheredoc` hands `readtoken1` `&sqsyntax + 129` or `&dqsyntax +
    /// 129`, reading the table's address from one of two global offset table
    /// cells by an offset a phi chooses, `400` or `476`: each read through
    /// the address reaches its table's bytes from `1` to `256`. The control:
    /// in a table the program may write, the cells may hold anything.
    #[test]
    fn a_cell_only_the_loader_writes_holds_its_address() {
        const OTHER: u32 = 0x1_1224;
        let read = |sealed: bool| {
            let mut built = Built::new(4);
            built.image = Some(Got {
                cells: BTreeMap::from([
                    (u64::from(GOT) + 400, u64::from(TABLE)),
                    (u64::from(GOT) + 476, u64::from(OTHER)),
                ]),
                sealed,
            });
            let selector = built.value(MockType::I32);
            let zero = built.number(0, 0);
            built.push(
                0,
                SsaOp::BranchCmp {
                    left: selector,
                    right: zero,
                    cmp: CmpKind::Eq,
                    unsigned: false,
                    true_target: 1,
                    false_target: 2,
                },
            );
            let first = built.number(1, 400);
            built.push(1, SsaOp::Jump { target: 3 });
            let second = built.number(2, 476);
            built.push(2, SsaOp::Jump { target: 3 });
            let offset =
                built
                    .ir
                    .create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
            built.close(3, offset, &[(first, 1), (second, 2)]);
            let got = built.symbol(3, GOT);
            let cell = built.step(3, got, Some(offset), 1, 0);
            let table = built.load(3, cell, MockType::U32);
            let syntax = built.step(3, table, None, 1, 129);
            let character = built.value(MockType::I8);
            let index = built.define(3, MockType::I32, |dest| SsaOp::IntConv {
                dest,
                operand: character,
                target: MockType::I32,
                overflow_check: false,
                unsigned: false,
            });
            let at = built.step(3, syntax, Some(index), 1, 0);
            built.load(3, at, MockType::U8);
            built.push(3, SsaOp::returns(None));
            let mut reaches: Vec<_> = built
                .reaches()
                .into_iter()
                .filter(|(root, ..)| *root != u64::from(GOT))
                .collect();
            reaches.sort_unstable_by_key(|(root, ..)| *root);
            reaches
        };
        assert_eq!(
            read(true),
            vec![
                (u64::from(OTHER), 1..257, 1, None),
                (u64::from(TABLE), 1..257, 1, None)
            ]
        );
        assert_eq!(read(false), vec![], "the control");
    }

    /// **A parameter points where its callers said.** A function stores the
    /// table its caller hands it into a record it allocates and reads the
    /// table through the record, as `readtoken1` keeps its `syntax` argument
    /// in `synbase`: bound to the caller's `&table + 129`, the read reaches
    /// the table's bytes from `1` to `256`. The control: unbound, the
    /// parameter is built on nothing known.
    #[test]
    fn a_parameter_points_where_its_callers_said() {
        let read = |bound: bool| {
            let mut built = Built::new(1);
            let syntax = built.value(MockType::Ptr);
            if bound {
                built.parameters.insert(
                    syntax,
                    vec![AddressOrigin {
                        root: u64::from(TABLE),
                        low: 129,
                        high: 129,
                        stride: 0,
                    }],
                );
            }
            let size = built.number(0, 40);
            let record = built.define(0, MockType::Ptr, |dest| SsaOp::LocalAlloc {
                dest,
                size,
                zeroed: false,
            });
            built.push(
                0,
                SsaOp::StoreIndirect {
                    addr: record,
                    value: syntax,
                    value_type: MockType::Ptr,
                    access: AccessQualifiers::FLAT,
                },
            );
            let kept = built.load(0, record, MockType::Ptr);
            let character = built.value(MockType::I8);
            let index = built.define(0, MockType::I32, |dest| SsaOp::IntConv {
                dest,
                operand: character,
                target: MockType::I32,
                overflow_check: false,
                unsigned: false,
            });
            let at = built.step(0, kept, Some(index), 1, 0);
            built.load(0, at, MockType::U8);
            built.push(0, SsaOp::returns(None));
            built.reaches()
        };
        assert_eq!(read(true), vec![(u64::from(TABLE), 1..257, 1, None)]);
        assert_eq!(read(false), vec![], "the control");
    }

    /// A switch's table read by the byte it dispatches on, widened before the
    /// guard `126 <u c` that sends larger bytes to the default: the read
    /// touches the 127 words the guard admits. Without the guard it may touch
    /// all 256 a byte names.
    fn switch_table(guarded: bool) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
        let mut built = Built::new(3);
        let byte = built.value(MockType::U8);
        let wide = built.define(0, MockType::U32, |dest| SsaOp::IntConv {
            dest,
            operand: byte,
            target: MockType::U32,
            overflow_check: false,
            unsigned: true,
        });
        let limit = built.define(0, MockType::U8, |dest| SsaOp::Const {
            dest,
            value: ConstValue::U8(126),
        });
        let table = built.symbol(0, 0x1_5558);
        if guarded {
            built.push(
                0,
                SsaOp::BranchCmp {
                    left: limit,
                    right: byte,
                    cmp: CmpKind::Lt,
                    unsigned: true,
                    true_target: 2,
                    false_target: 1,
                },
            );
        } else {
            built.push(0, SsaOp::Jump { target: 1 });
        }
        let at = built.step(1, table, Some(wide), 4, 0);
        built.load(1, at, MockType::I32);
        built.push(1, SsaOp::returns(None));
        built.push(2, SsaOp::returns(None));
        built.reaches()
    }

    #[test]
    fn a_switch_table_reaches_only_what_its_guard_admits() {
        assert_eq!(switch_table(true), vec![(0x1_5558, 0..508, 4, None)]);
        assert_eq!(switch_table(false), vec![(0x1_5558, 0..1024, 4, None)]);
    }

    /// Control: an index the ranges do not bound - a word read from memory -
    /// reaches nothing anyone can state, and an index narrower than the
    /// pointer that may be negative could be extended either way.
    #[test]
    fn an_unbounded_index_reaches_nothing() {
        let mut built = Built::new(1);
        let table = built.symbol(0, TABLE);
        let cell = built.symbol(0, 0x2_0000);
        let index = built.load(0, cell, MockType::I32);
        let at = built.step(0, table, Some(index), 4, 0);
        built.load(0, at, MockType::I32);
        built.push(0, SsaOp::returns(None));
        assert!(built.reaches().is_empty());
        assert!(syntax_read(&[TABLE], MockType::I16).is_empty());
    }

    /// An access at a fixed displacement moves nothing: a section anchor
    /// reaches its neighbours that way, and the access says nothing of an
    /// object's extent.
    #[test]
    fn a_fixed_access_reaches_nothing() {
        let mut built = Built::new(1);
        let anchor = built.symbol(0, TABLE);
        let at = built.step(0, anchor, None, 0, 8);
        built.load(0, at, MockType::I32);
        built.push(0, SsaOp::returns(None));
        assert!(built.reaches().is_empty());
    }

    /// `kill -l`'s walk over dash's signal names: `p = phi(&names, p + 4)`
    /// reads `p[1]` and goes on while `p + 4 != &names + 256`, so `p` takes
    /// every word from `&names` to `&names + 252`, and the read touches the
    /// bytes from `4` to `260`.
    fn names_walk(
        bound: i32,
        cmp: CmpKind,
        unsigned: bool,
    ) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
        let mut built = Built::new(3);
        let names = built.symbol(0, 0x2_f8ac);
        let other = built.symbol(0, 0x2_f8ac);
        let distance = built.number(0, bound);
        let end = built.define(0, MockType::Ptr, |dest| SsaOp::Add {
            dest,
            left: other,
            right: distance,
            flags: None,
        });
        built.push(0, SsaOp::Jump { target: 1 });
        let walker = built.phi(1);
        let element = built.step(1, walker, None, 0, 4);
        built.load(1, element, MockType::U32);
        let stepped = built.step(1, walker, None, 0, 4);
        built.push(
            1,
            SsaOp::BranchCmp {
                left: stepped,
                right: end,
                cmp,
                unsigned,
                true_target: 1,
                false_target: 2,
            },
        );
        built.push(2, SsaOp::returns(None));
        built.close(1, walker, &[(names, 0), (stepped, 1)]);
        built.reaches()
    }

    #[test]
    fn a_walk_to_a_bound_reaches_every_element_on_the_way() {
        assert_eq!(
            names_walk(256, CmpKind::Ne, false),
            vec![(0x2_f8ac, 4..260, 4, Some(1))]
        );
        assert_eq!(
            names_walk(256, CmpKind::Lt, true),
            vec![(0x2_f8ac, 4..260, 4, Some(1))]
        );
        assert_eq!(
            names_walk(255, CmpKind::Lt, true),
            vec![(0x2_f8ac, 4..260, 4, Some(1))]
        );
    }

    /// dash's `dotrap` over `gotsig`: `p = phi(&gotsig - 1, p + 1)` reads
    /// `p[1]` beside `i = phi(0, i + 1)`, and the loop goes on while `i + 1
    /// != 64` (or `< 64`), so `p` takes 64 bytes from `&gotsig - 1` and the
    /// read touches `gotsig`'s 64. The anchor is `trap`, the object below.
    fn counted_walk(
        bound: Option<i32>,
        cmp: CmpKind,
    ) -> Vec<(u64, Range<i64>, u64, Option<usize>)> {
        let mut built = Built::new(3);
        let trap = built.symbol(0, 0x3_2798);
        let start = built.step(0, trap, None, 0, 0x103);
        let zero = built.number(0, 0);
        let limit = match bound {
            Some(bound) => built.number(0, bound),
            None => built.load(0, trap, MockType::I32),
        };
        built.push(0, SsaOp::Jump { target: 1 });
        let walker = built.phi(1);
        let count =
            built
                .ir
                .create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::I32);
        built.step(1, walker, None, 0, 1);
        let next = built.step(1, walker, None, 0, 1);
        built.load(1, next, MockType::U8);
        let one = built.number(1, 1);
        let counted = built.define(1, MockType::I32, |dest| SsaOp::Add {
            dest,
            left: count,
            right: one,
            flags: None,
        });
        built.push(
            1,
            SsaOp::BranchCmp {
                left: counted,
                right: limit,
                cmp,
                unsigned: false,
                true_target: 1,
                false_target: 2,
            },
        );
        built.push(2, SsaOp::returns(None));
        built.close(1, walker, &[(start, 0), (next, 1)]);
        let mut phi = PhiNode::new(count, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(zero, 0));
        phi.add_operand(PhiOperand::new(counted, 1));
        if let Some(block) = built.blocks.get_mut(1) {
            block.add_phi(phi);
        }
        built.reaches()
    }

    #[test]
    fn a_walk_a_counter_ends_reaches_every_step_the_counter_allows() {
        assert_eq!(
            counted_walk(Some(64), CmpKind::Ne),
            vec![(0x3_2798, 0x104..0x144, 1, Some(1))]
        );
        assert_eq!(
            counted_walk(Some(64), CmpKind::Lt),
            vec![(0x3_2798, 0x104..0x144, 1, Some(1))]
        );
        assert!(counted_walk(None, CmpKind::Ne).is_empty());
        assert!(counted_walk(Some(64), CmpKind::Gt).is_empty());
    }

    /// Control: a bound the walk steps over on inequality never stops it, and
    /// a signed order of two addresses or a bound the walk goes on past bounds
    /// nothing.
    #[test]
    fn a_walk_its_comparison_does_not_end_reaches_nothing() {
        assert!(names_walk(255, CmpKind::Ne, false).is_empty());
        assert!(names_walk(256, CmpKind::Lt, false).is_empty());
        assert!(names_walk(256, CmpKind::Gt, true).is_empty());
    }
}
