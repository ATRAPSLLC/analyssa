//! Folding a double-width sum computed in halves back to the sum.
//!
//! A target whose registers are half as wide as a value computes the value's
//! sum or difference a half at a time: the low halves first, then the high
//! halves with the carry or borrow out of the low one. x86 spells the carry
//! with `adc` and `sbb`, MIPS and RISC-V with `sltu`; once the flags are
//! resolved, every one of them is the same shape:
//!
//! ```text
//! lo = al + bl            lo = al - bl
//! c  = lo <u al           b  = al <u bl
//! hi = ah + bh + c        hi = ah - bh - b
//! ```
//!
//! # What a fold requires
//!
//! The fold is driven by a reader of both halves, not by the arithmetic: a
//! value composed of them, `(hi << w) | lo` with `lo` zero-extended, or two
//! stores writing them to the two adjacent words of one cell ([`Doubleword::store_pair`]),
//! the halves of a `long long` a 32-bit target keeps in memory. Either is the
//! sum of the values the operand halves compose and is rewritten to it. A pair
//! no reader takes as one value is left as it is, because restating its halves
//! as parts of a wider sum would add work no reader collects. The operands'
//! compositions are built from their halves, and fold away in turn wherever the
//! halves are themselves the parts of one wider value
//! ([`Reader::copied_value`](crate::analysis::bits::Reader::copied_value)).
//!
//! The halves are matched on values: each carry must compare its step's low
//! sum against one of that sum's own addends, unsigned, and each borrow the
//! minuend's low half against the subtrahend's. Every half has one width `w`,
//! the composition twice that, and the shift is by `w`. The identity holds
//! modulo `2^2w` whatever the halves hold, so no flag or range is consulted.
//!
//! # Chains
//!
//! `a + b + c` computed in halves carries once per step, and its high half
//! adds every operand's high half and every carry. Once reassociation has
//! flattened it, the steps' high halves are no values of their own, only terms
//! of one sum. The high half is therefore read as the terms it adds and
//! subtracts, and the steps are peeled off the low half from the outside in:
//! each step's carry or borrow is one of the terms, and the step's own operand
//! takes the term that is its high half (its sign fill, the high half of a
//! pair it is the low half of, the other half of one wider value, the adjacent
//! word of a load). What is left is the high half of the value the chain began
//! from.
//!
//! Which term is whose does not change the value: `lo` and the carries fix the
//! low half and what crosses into the high one, and the high half reads only
//! the terms' sum, so `(hi << w) | lo` is the sum of the operands' compositions
//! for any split of the other terms among them. The split decides only which
//! compositions fold further.

use std::collections::BTreeMap;

use crate::{
    analysis::{
        address::{AddressBase, AddressWalk},
        bits::Reader,
        consts::Machine,
        exceptions::FunctionRoots,
        operands::{const_amount, peel},
    },
    ir::{
        SsaFunction, VarSet,
        integer::BitWidth,
        ops::{AccessQualifiers, SsaOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::{Endianness, Target},
};

/// The most terms a high half is read as; a sum of more is left unmatched.
///
/// A chain of `n` steps adds about two terms per step, a carry and the
/// operand's high half, and a `-O0` accumulator summing thirty calls' results
/// is sixty of them. The bound is what stops a tree sharing its subterms
/// (`x + x` added to itself) from being read as the exponentially many terms
/// it sums.
const MOST_TERMS: usize = 256;

/// The most sums in halves one is read through nested: an operand of a step
/// that is itself the low half of a sum in halves is read as that sum's chain,
/// and its operands in turn, this deep. The bound keeps the walk off the
/// stack of whatever thread runs it: a sum nested inside a sum as deep as the
/// code that built it would otherwise recurse once per level.
const MOST_NESTED: usize = 16;

/// A term of a sum: a value, added or subtracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Term {
    /// The value.
    pub value: SsaVarId,
    /// Whether it is subtracted.
    pub negative: bool,
}

/// An operand of a double-width sum given by its halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    /// The low half.
    pub low: SsaVarId,
    /// The terms the high half sums, each added or subtracted; none where the
    /// high half is zero.
    pub high: Vec<Term>,
    /// Whether the operand is subtracted from the ones before it.
    pub negative: bool,
}

/// A composition of two halves that is the double-width sum of operands given
/// by their halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoublewordFold {
    /// The operands, in the order they are summed: the value the chain began
    /// from first, which is always added.
    pub operands: Vec<Operand>,
    /// The shift amount the composition places the high half with, which the
    /// operands' compositions reuse.
    pub amount: SsaVarId,
    /// The low and the high half the composition reads.
    pub halves: [SsaVarId; 2],
    /// The target's byte order, where it is stated: which half of a value
    /// kept in memory is at the lower address ([`Self::in_memory_order`]).
    pub byte_order: Option<Endianness>,
    /// The halves of each partial sum the chain computes, in the order the
    /// operands are summed: the low and the high half of the sum of the
    /// first two operands, then of that and the third, and so on, the last
    /// the whole sum's. The high half is `None` where no single value of the
    /// chain is it.
    pub partial_halves: Vec<(SsaVarId, Option<SsaVarId>)>,
}

/// The operands of a double-width sum in halves and the halves of each sum
/// along the way ([`Doubleword::chain`]).
struct Chain {
    /// The operands, the value the chain began from first.
    operands: Vec<Operand>,
    /// The halves of each partial sum, as [`DoublewordFold::partial_halves`].
    partial_halves: Vec<(SsaVarId, Option<SsaVarId>)>,
}

impl DoublewordFold {
    /// Of a value given by its `low` and `high` half, the one kept at the
    /// lower address where the value is kept in memory, and the other: the
    /// low half first on a little-endian target, the high half on a
    /// big-endian one. `None` when the byte order is not stated.
    #[must_use]
    pub fn in_memory_order(&self, low: SsaVarId, high: SsaVarId) -> Option<[SsaVarId; 2]> {
        self.byte_order.map(|order| match order {
            Endianness::Little => [low, high],
            Endianness::Big => [high, low],
        })
    }
}

/// A double-width sum that adds the two halves of wider values apart
/// ([`Doubleword::regroup`]): each value's low half zero-extended is a term of
/// the sum, and its high half a term of a high half the sum shifts into place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Regrouping {
    /// The values whose halves the sum adds apart, each added or subtracted
    /// as both its halves were.
    pub whole: Vec<Term>,
    /// The high halves the sum shifts into place, each with what the paired
    /// halves leave of it and the low half it is composed with, if any.
    pub composed: Vec<Composed>,
    /// The sum's other terms, as they are.
    pub kept: Vec<Term>,
    /// The shift amount the compositions place their high halves with.
    pub amount: SsaVarId,
}

/// A high half a double-width sum shifts into place, composed with a low half
/// or alone ([`Regrouping::composed`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composed {
    /// The low half the composition zero-extends, where it has one.
    pub low: Option<SsaVarId>,
    /// The terms the high half sums, each added or subtracted.
    pub high: Vec<Term>,
    /// Whether the composition is subtracted from the sum.
    pub negative: bool,
}

/// Two stores of the halves of one double-width value to the two adjacent
/// words of one cell ([`Doubleword::store_pair`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePair {
    /// The position, in the block, of the earlier store; the later one is
    /// the store the pair was found at.
    pub first: usize,
    /// The address of the cell: the lower of the two.
    pub address: SsaVarId,
    /// The space both stores write, and the permission they write it with.
    pub space: AccessQualifiers,
    /// The value the low half is stored from.
    pub low: SsaVarId,
    /// The value the high half is stored from.
    pub high: SsaVarId,
    /// The width of each half, in bits.
    pub half: u32,
}

/// The load of both words a composition of two word loads is ([`Doubleword::wide_load`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WideLoad<T: Target> {
    /// The load, defining the composition's destination.
    pub load: SsaOp<T>,
    /// The positions, in the composition's block, of the two word loads it
    /// replaces, which nothing else reads.
    pub loads: [usize; 2],
    /// The results of the word loads: the one at the address the load reads
    /// from, the start of the value it loads whole, and the other.
    pub words: [SsaVarId; 2],
}

/// The two halves a composition places side by side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Halves {
    /// The low half, as the value the composition zero-extends: `half` bits
    /// wide, or narrower where its own zero-extension is folded into the
    /// composition's.
    pub low: SsaVarId,
    /// The high half, as the value the composition extends and shifts: `half`
    /// bits wide, or narrower where its extension is folded in.
    pub high: SsaVarId,
    /// The width of a half, half the composition's.
    pub half: u32,
    /// The shift amount that places the high half.
    pub amount: SsaVarId,
    /// The values between the halves and the composition: the zero-extended
    /// low half, the extended high half and that shifted into place.
    pub between: [SsaVarId; 3],
}

/// The double-width arithmetic of one function: sums, products, shifts,
/// loads and stores a target too narrow for a value builds in two halves, and
/// the one double-width operation each is, under one machine and with
/// `constants` known ([`Self::fold`]).
#[derive(Clone, Copy)]
pub struct Doubleword<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Known constant values.
    constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
    /// The machine the halves are computed on.
    machine: Machine,
}

impl<'a, T: Target> Doubleword<'a, T> {
    /// Reads the double-width arithmetic of `ssa` on `machine`, with
    /// `constants` known.
    #[must_use]
    pub const fn new(
        ssa: &'a SsaFunction<T>,
        constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
        machine: Machine,
    ) -> Self {
        Self {
            ssa,
            constants,
            machine,
        }
    }

    /// The double-width sum the composition `op` collects from its halves, when it
    /// is one.
    ///
    /// # Arguments
    ///
    /// * `op` - The candidate composition: an `or`, `add` or `xor` of a
    ///   zero-extended low half and a high half shifted by the half width.
    ///
    /// # Returns
    ///
    /// The operands' halves, or `None` when `op` composes no such sum.
    #[must_use]
    pub fn fold(&self, op: &SsaOp<T>) -> Option<DoublewordFold> {
        let halves =
            halves(self.ssa, op, self.constants).filter(|halves| halves.whole(self.ssa))?;
        let Halves {
            low, high, amount, ..
        } = halves;
        let Chain {
            operands,
            partial_halves,
        } = self.chain(low, high, MOST_NESTED)?;
        Some(DoublewordFold {
            operands,
            amount,
            halves: [low, high],
            byte_order: self.machine.byte_order,
            partial_halves,
        })
    }

    /// The double-width sum `op` is, regrouped so that each value whose two
    /// halves it adds apart is a term of its own.
    ///
    /// A sum in halves folded while its operands' halves were not yet known to
    /// be one value's ([`Self::fold`]) adds each operand's low half
    /// zero-extended as a term and collects its high half among the terms of
    /// the high half the sum shifts into place: `((ah + bh) << w | x) + al +
    /// bl`. Once `al` and `ah` are known to be `a`'s halves (a callee's two
    /// results found to be one integer), `al + (ah << w)` is `a`, modulo
    /// `2^2w`: the high half's terms shift into place one by one, as a carry
    /// out of the high half leaves the double width, and the composition's
    /// `or` is an addition of disjoint bits. Each low half pairs with a high
    /// half of the same value and the same sign, and the two are the value.
    ///
    /// # Arguments
    ///
    /// * `op` - The candidate sum: a double-width `add` or `sub`.
    ///
    /// # Returns
    ///
    /// The sum regrouped, or `None` when it adds no value's halves apart.
    #[must_use]
    pub fn regroup(&self, op: &SsaOp<T>) -> Option<Regrouping> {
        let (SsaOp::Add { dest, .. } | SsaOp::Sub { dest, .. }) = op else {
            return None;
        };
        let ssa = self.ssa;
        let wide = ssa.bit_width(*dest)?;
        let half = wide
            .checked_div(2)
            .filter(|half| *half > 0 && half.checked_mul(2) == Some(wide))?;
        let mut composed: Vec<Composed> = Vec::new();
        let mut amount = None;
        let mut rest: Vec<Term> = Vec::new();
        for term in terms(ssa, *dest, wide)? {
            let definition = ssa.get_definition(term.value);
            // A high half re-placed by another extension than the one it had
            // would change the bits above it, so only a high half a half wide,
            // whose extension the shift discards, is read through.
            let placed = definition
                .and_then(|op| halves(ssa, op, self.constants))
                .filter(|halves| halves.half == half && ssa.bit_width(halves.high) == Some(half))
                .map(|halves| (Some(halves.low), halves.high, halves.amount))
                .or_else(|| match definition? {
                    SsaOp::Shl { value, amount, .. }
                        if const_amount(ssa, self.constants, *amount) == Some(half) =>
                    {
                        let high = extended(ssa, *value, false)?;
                        (ssa.bit_width(high) == Some(half)).then_some((None, high, *amount))
                    }
                    _ => None,
                });
            match placed {
                Some((low, high, shift)) => {
                    amount.get_or_insert(shift);
                    composed.push(Composed {
                        low,
                        high: terms(ssa, high, half)?,
                        negative: term.negative,
                    });
                }
                None => rest.push(term),
            }
        }
        let amount = amount?;
        let mut whole = Vec::new();
        let mut kept = Vec::new();
        for term in rest {
            let paired = extended(ssa, term.value, true)
                .filter(|low| ssa.bit_width(*low) == Some(half))
                .and_then(|low| {
                    let SsaOp::IntConv { operand, .. } = ssa.get_definition(low)? else {
                        return None;
                    };
                    composed.iter_mut().find_map(|composition| {
                        let at = composition.high.iter().position(|high| {
                            (high.negative != composition.negative) == term.negative
                                && split(ssa, self.constants, low, high.value, half)
                        })?;
                        composition.high.remove(at);
                        Some(peel(ssa, *operand))
                    })
                });
            match paired {
                Some(value) => whole.push(Term {
                    value,
                    negative: term.negative,
                }),
                None => kept.push(term),
            }
        }
        if whole.is_empty() {
            return None;
        }
        composed.retain(|composition| composition.low.is_some() || !composition.high.is_empty());
        Some(Regrouping {
            whole,
            composed,
            kept,
            amount,
        })
    }

    /// The store pair the store at position `at` of `block` closes: it and the
    /// latest store before it in the block write the two adjacent words of one
    /// cell, in the halves `byte_order` places a double-width value's low and high
    /// halves in, and what they write is one double-width value a composition of
    /// the two reduces - a sum computed in halves, a value with its sign above it,
    /// or the two halves of one wider value - or two constants where the function
    /// accesses the cell whole elsewhere.
    ///
    /// The two are then one store of that value: a 32-bit target keeps a
    /// `long long` in memory as two words, and every update of it is a sum in
    /// halves stored back as two words, which no composition ever reads.
    ///
    /// Only where every access between the two is a load or store through the
    /// same base that misses the earlier store's word, and, in a block a handler
    /// protects, nothing between them may throw: then one store at the later's
    /// place writes what the two did, and no access, and no handler a fault
    /// reaches, can tell the difference. A fault outside a protected region ends
    /// the function where nothing of it reads memory again, the latitude the
    /// memory pass's dead store elimination takes too.
    ///
    /// # Arguments
    ///
    /// * `block` - The block holding the store.
    /// * `at` - The store's position in the block.
    ///
    /// # Returns
    ///
    /// The pair, or `None` when the store closes none or the byte order is not
    /// stated.
    #[must_use]
    pub fn store_pair(&self, block: usize, at: usize) -> Option<StorePair> {
        let Machine {
            ptr_size,
            byte_order,
            ..
        } = self.machine;
        let byte_order = byte_order?;
        let instructions = self.ssa.block(block)?.instructions();
        let store = |position: usize| match instructions.get(position)?.op() {
            SsaOp::StoreIndirect {
                addr,
                value,
                value_type,
                access: space,
            } => Some((*addr, *value, T::bit_width(value_type)?, *space)),
            _ => None,
        };
        let (second_addr, second_value, half, space) = store(at)?;
        if half == 0 || T::integer_type(half.checked_mul(2)?, false).is_none() {
            return None;
        }
        let (second_base, second_offset) = address(self.ssa, second_addr, ptr_size)?;
        let step = i64::from(half / 8);
        // The latest store to a word beside the later one's, with no access that
        // may overlap that word, and nothing that may throw, in between.
        let mut between: Vec<(i64, i64)> = Vec::new();
        let mut first = None;
        let mut protected = None;
        for position in (0..at).rev() {
            let op = instructions.get(position)?.op();
            if op.may_throw()
                && *protected.get_or_insert_with(|| {
                    !FunctionRoots::of(self.ssa)
                        .exceptional_successors(block)
                        .is_empty()
                })
            {
                return None;
            }
            let effects = op.effects();
            if !(effects.reads_memory() || effects.writes_memory()) {
                continue;
            }
            let (addr, bits, access_space, stored) = match op {
                SsaOp::StoreIndirect {
                    addr,
                    value_type,
                    access: space,
                    ..
                } => (*addr, T::bit_width(value_type)?, *space, true),
                SsaOp::LoadIndirect {
                    addr,
                    value_type,
                    access: space,
                    ..
                } => (*addr, T::bit_width(value_type)?, *space, false),
                _ => return None,
            };
            let (base, offset) = address(self.ssa, addr, ptr_size)?;
            if base != second_base || access_space != space {
                return None;
            }
            let partner = offset.checked_add(step) == Some(second_offset)
                || second_offset.checked_add(step) == Some(offset);
            if stored && bits == half && partner {
                first = Some(position);
                break;
            }
            between.push((offset, offset.checked_add(i64::from(bits / 8))?));
        }
        let first = first?;
        let (first_addr, first_value, _, _) = store(first)?;
        let (_, first_offset) = address(self.ssa, first_addr, ptr_size)?;
        // The earlier store's word moves down to the later one's place, so no
        // access in between may touch it.
        let first_end = first_offset.checked_add(step)?;
        if between
            .iter()
            .any(|(start, end)| *start < first_end && first_offset < *end)
        {
            return None;
        }
        if [first_value, second_value]
            .iter()
            .any(|value| self.ssa.bit_width(*value) != Some(half))
        {
            return None;
        }
        let (lower, upper, lower_addr) = if first_offset < second_offset {
            (first_value, second_value, first_addr)
        } else {
            (second_value, first_value, second_addr)
        };
        let (low, high) = match byte_order {
            Endianness::Little => (lower, upper),
            Endianness::Big => (upper, lower),
        };
        // Two self.constants are one value's halves where the function reads or writes
        // the cell whole as well: that access says what the cell holds.
        let constant = |value: SsaVarId| self.constants.contains_key(&peel(self.ssa, value));
        let whole = (constant(low) && constant(high))
            && accessed_whole(self.ssa, ptr_size, lower_addr, space, half.checked_mul(2)?);
        (whole || self.reduces([low, high], half)).then_some(StorePair {
            first,
            address: lower_addr,
            space,
            low,
            high,
            half,
        })
    }

    /// The double-width load a composition of two word loads is: `(hi << w) |
    /// zext(lo)` where `lo` and `hi` load the two adjacent words of one address,
    /// `lo` the one `byte_order` puts the low half in, and nothing between the
    /// first load and the composition can write memory, is the load of both words
    /// at the lower address - a 32-bit target reading a `uint64_t` field.
    ///
    /// Only where nothing but the composition reads either word, and the two word
    /// loads go with it ([`WideLoad::loads`]): the memory pass forwards a later
    /// load of bytes already loaded from the earlier loads, so a word load left
    /// beside the wide one, read apart or merely kept because a load may fault,
    /// would compose the wide load back out of them, and the two would undo each
    /// other at every round. The wide load reads every byte either did, so it
    /// faults wherever one of them would; in a block a handler protects, nothing
    /// between the first load and the composition may throw, so no other fault
    /// can come first.
    ///
    /// # Arguments
    ///
    /// * `op` - The candidate composition, as [`Doubleword::fold`] takes it.
    /// * `uses` - Use counts for the whole function, as
    ///   [`SsaFunction::count_uses`] reports them.
    ///
    /// # Returns
    ///
    /// The load and the word loads it takes, or `None` when `op` is no such
    /// composition or the byte order is not stated.
    #[must_use]
    pub fn wide_load(
        &self,
        op: &SsaOp<T>,
        uses: &BTreeMap<SsaVarId, usize>,
    ) -> Option<WideLoad<T>> {
        let Machine {
            ptr_size,
            byte_order,
            ..
        } = self.machine;
        let byte_order = byte_order?;
        let halves =
            halves(self.ssa, op, self.constants).filter(|halves| halves.whole(self.ssa))?;
        let Halves {
            low,
            high,
            half,
            between,
            ..
        } = halves;
        let dest = op.dest()?;
        let once = |var: SsaVarId| uses.get(&var).copied().unwrap_or(0) == 1;
        if !between.into_iter().chain([low, high]).all(once) {
            return None;
        }
        let WordLoads {
            first,
            space,
            words,
            bits,
        } = word_loads(self.ssa, ptr_size, byte_order, low, high)?;
        if bits != half {
            return None;
        }
        let block = self.ssa.variable(dest)?.def_site().block();
        let end = position_in(self.ssa, block, dest)?;
        let loads = unwritten(self.ssa, [low, high], block, end)?;
        Some(WideLoad {
            load: SsaOp::LoadIndirect {
                dest,
                addr: first,
                value_type: self.ssa.variable(dest)?.var_type().clone(),
                access: space,
            },
            loads,
            words,
        })
    }

    /// Whether `halves`, the low and the high half of a value, each `half` bits
    /// wide, are the halves of one double-width value that a composition of them
    /// read at `at`, a block and a position in it, reduces to: a sum computed in
    /// halves, `low` with its sign above it, the two halves of one wider value,
    /// or the loads of one address's two adjacent words, each read nowhere else,
    /// of a cell the function accesses whole elsewhere, with nothing that may
    /// write memory from the first load to `at` - then a load of both words there
    /// is the value ([`Doubleword::wide_load`]). Two words side by side are no value of their
    /// own where nothing accesses them whole: `main` passes `argc` and `argv` on
    /// as two adjacent words.
    ///
    /// What a reader asks that is about to take the two as one value: a 32-bit
    /// target passes a `double` or a `long long` in two argument registers or two
    /// stack words, and only the halves say the two are one argument.
    ///
    /// # Arguments
    ///
    /// * `uses` - Use counts for the whole function, as
    ///   [`SsaFunction::count_uses`] reports them.
    /// * `halves` - The value's low half, then its high half.
    /// * `half` - The width of each half, in bits: what the places a reader
    ///   takes them from say, where a half's own type does not say it yet (an
    ///   incoming argument a 32-bit target passes on as it arrived).
    /// * `at` - The block the composition would be read in, and its position
    ///   there.
    #[must_use]
    pub fn one_value(
        &self,
        uses: &BTreeMap<SsaVarId, usize>,
        halves: [SsaVarId; 2],
        half: u32,
        at: (usize, usize),
    ) -> bool {
        let [low, high] = halves;
        let (block, at) = at;
        if half == 0
            || halves
                .iter()
                .any(|value| self.ssa.bit_width(*value).is_some_and(|bits| bits != half))
        {
            return false;
        }
        if self.reduces(halves, half) {
            return true;
        }
        // Two words loaded side by side are two values unless the function
        // accesses their cell whole as well, as two words stored side by side are
        // ([`Doubleword::store_pair`]): `argc` and `argv` lie side by side too.
        let once = |var: SsaVarId| uses.get(&var).copied().unwrap_or(0) == 1;
        once(low)
            && once(high)
            && self.machine.byte_order.is_some_and(|byte_order| {
                word_loads(self.ssa, self.machine.ptr_size, byte_order, low, high).is_some_and(
                    |loads| {
                        loads.bits == half
                            && half.checked_mul(2).is_some_and(|bits| {
                                accessed_whole(
                                    self.ssa,
                                    self.machine.ptr_size,
                                    loads.first,
                                    loads.space,
                                    bits,
                                )
                            })
                    },
                ) && unwritten(self.ssa, [low, high], block, at).is_some()
            })
    }

    /// Whether `low` and `high` are the halves of one double-width value a
    /// composition of them reduces to: a sum computed in halves, `low` with its
    /// sign above it, or the two halves of one wider value.
    ///
    /// Two constants alone are no such evidence: an `int` array or a structure
    /// stored a word at a time is two values, and a double-width read of the cell
    /// composes the two and folds the composition however they were stored. Only
    /// a whole access of the cell elsewhere says the two are one value
    /// ([`Doubleword::store_pair`]).
    fn reduces(&self, [low, high]: [SsaVarId; 2], half: u32) -> bool {
        sign_fill(self.ssa, self.constants, low, high, half)
            || split(self.ssa, self.constants, low, high, half)
            || self.chain(low, high, MOST_NESTED).is_some()
    }

    /// The operands of the double-width sum whose low half is `low` and high half
    /// `high`, the value the chain began from first: `None` where no step of a sum
    /// or difference computed in halves ends at `low`.
    ///
    /// The steps are peeled off `low` from the outside in, each by the carry or
    /// borrow it leaves among `high`'s terms; each step's operand then takes the
    /// term that is its own high half, and the value the chain began from keeps
    /// the rest (see the module documentation for why any split is exact).
    ///
    /// `nesting` is how many sums deeper an operand may still be read through
    /// ([`MOST_NESTED`]).
    fn chain(&self, low: SsaVarId, high: SsaVarId, nesting: usize) -> Option<Chain> {
        let half = self.ssa.bit_width(low)?;
        let mut rest = terms(self.ssa, high, half)?;
        let mut low = peel(self.ssa, low);
        let mut steps: Vec<Operand> = Vec::new();
        // Each step's own low half, the low half of the sum it ends, and the
        // carry or borrow it leaves in the high half's terms.
        let mut ends: Vec<(SsaVarId, Term)> = Vec::new();
        loop {
            if self.ssa.bit_width(low) != Some(half) {
                break;
            }
            let (start, operand, negative) = match self.ssa.get_definition(low) {
                Some(SsaOp::Add { left, right, .. }) => {
                    let addends = [peel(self.ssa, *left), peel(self.ssa, *right)];
                    let Some(at) = rest.iter().position(|term| {
                        !term.negative && is_carry(self.ssa, term.value, low, addends)
                    }) else {
                        break;
                    };
                    ends.push((low, rest.remove(at)));
                    // The addend the chain goes on through is the one whose own
                    // step left its carry among the terms.
                    if carried(self.ssa, addends[1], &rest) && !carried(self.ssa, addends[0], &rest)
                    {
                        (addends[1], addends[0], false)
                    } else {
                        (addends[0], addends[1], false)
                    }
                }
                Some(SsaOp::Sub { left, right, .. }) => {
                    let (minuend, subtrahend) = (peel(self.ssa, *left), peel(self.ssa, *right));
                    let Some(at) = rest.iter().position(|term| {
                        term.negative && is_borrow(self.ssa, term.value, minuend, subtrahend)
                    }) else {
                        break;
                    };
                    ends.push((low, rest.remove(at)));
                    (minuend, subtrahend, true)
                }
                _ => break,
            };
            steps.push(Operand {
                low: operand,
                high: Vec::new(),
                negative,
            });
            low = start;
        }
        if steps.is_empty() {
            return None;
        }
        // A subtracted operand's high half is subtracted too, so it is a term of
        // the operand's own sign, and the operand holds it added.
        for step in &mut steps {
            if let Some(at) = rest.iter().position(|term| {
                term.negative == step.negative && self.fits(step.low, term.value, nesting)
            }) {
                let term = rest.remove(at);
                step.high.push(Term {
                    value: term.value,
                    negative: false,
                });
            }
        }
        // One step of two whole operands, `ah + bh + c` or `ah - bh - b`: where no
        // term told whose it is, the start keeps the added term that fits it, or
        // else the first the sum names, and the operand takes the other.
        if let [step] = steps.as_mut_slice()
            && step.high.is_empty()
            && let [first, second] = rest.as_slice()
        {
            let keep = usize::from(
                !first.negative
                    && !self.fits(low, first.value, nesting)
                    && !second.negative
                    && self.fits(low, second.value, nesting),
            );
            let other = 1usize.saturating_sub(keep);
            if let (Some(kept), Some(taken)) = (rest.get(keep).copied(), rest.get(other).copied())
                && !kept.negative
                && taken.negative == step.negative
            {
                rest.remove(other);
                step.high.push(Term {
                    value: taken.value,
                    negative: false,
                });
            }
        }
        // Each partial sum's high half is the value of the high half's tree that
        // sums what the chain began from and each step so far: its carry and,
        // where it has one, its operand's high half.
        let nodes = sum_nodes(self.ssa, high, half);
        let mut summed: Vec<Term> = rest.clone();
        let mut partial_halves = Vec::with_capacity(steps.len());
        for (step, (end, carry)) in steps.iter().zip(&ends).rev() {
            summed.push(*carry);
            summed.extend(step.high.iter().map(|term| Term {
                value: term.value,
                negative: step.negative,
            }));
            let high = nodes
                .iter()
                .find(|(_, terms)| same_terms(terms, &summed))
                .map(|(node, _)| *node);
            partial_halves.push((*end, high));
        }
        // The last sum is the whole one, whose high half is the one given.
        if let Some(last) = partial_halves.last_mut() {
            last.1 = Some(high);
        }
        let mut operands = Vec::with_capacity(steps.len().saturating_add(1));
        operands.push(Operand {
            low,
            high: rest,
            negative: false,
        });
        operands.extend(steps.into_iter().rev());
        Some(Chain {
            operands,
            partial_halves,
        })
    }

    /// Whether `high` is a high half the composition with `low` reduces: `low`'s
    /// sign fill, the other half of one wider value, the adjacent word of a load
    /// `low` is, or the high half of a sum in halves `low` is the low half of,
    /// read `nesting` sums deep at most.
    fn fits(&self, low: SsaVarId, high: SsaVarId, nesting: usize) -> bool {
        self.ssa.bit_width(low).is_some_and(|half| {
            sign_fill(self.ssa, self.constants, low, high, half)
                || split(self.ssa, self.constants, low, high, half)
        }) || self.machine.byte_order.is_some_and(|order| {
            word_loads(
                self.ssa,
                self.machine.ptr_size,
                order,
                peel(self.ssa, low),
                peel(self.ssa, high),
            )
            .is_some()
        }) || nesting
            .checked_sub(1)
            .and_then(|deeper| self.chain(low, high, deeper))
            .is_some()
    }
}

/// The halves `op` composes: `(hi << w) | lo`, as an `or`, `add` or `xor`,
/// with `lo` zero-extended, `hi` extended either way (the shift moves its
/// extension out), each at most `w` wide and the composition twice that.
///
/// The shape every double-width value a target computes in halves takes once
/// a reader wants it whole ([`Doubleword::fold`], [`extension`] and [`Doubleword::wide_load`], which
/// read halves `w` wide, [`Halves::whole`]): a composition no fold rewrites
/// is two values side by side, not one. A half narrower than `w` is one whose
/// own extension to `w` the composition's extension subsumes: a 32-bit value
/// a 64-bit register holds, composed into 128 bits.
///
/// # Arguments
///
/// * `ssa` - The function, for resolving each operand's definition.
/// * `op` - The candidate composition.
/// * `constants` - Known constant values, for the shift amount.
///
/// # Returns
///
/// The halves, or `None` when `op` is no such composition.
#[must_use]
pub fn halves<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<Halves> {
    let (SsaOp::Or {
        dest, left, right, ..
    }
    | SsaOp::Add {
        dest, left, right, ..
    }
    | SsaOp::Xor {
        dest, left, right, ..
    }) = op
    else {
        return None;
    };
    let wide = ssa.bit_width(*dest)?;
    let half = wide.checked_div(2).filter(|half| *half > 0)?;
    let halves =
        [(*left, *right), (*right, *left)]
            .into_iter()
            .find_map(|(widened, shifted)| {
                let SsaOp::Shl { value, amount, .. } = ssa.get_definition(shifted)? else {
                    return None;
                };
                (const_amount(ssa, constants, *amount) == Some(half)).then_some(())?;
                Some(Halves {
                    low: extended(ssa, widened, true)?,
                    high: extended(ssa, *value, false)?,
                    half,
                    amount: *amount,
                    between: [widened, *value, shifted],
                })
            })?;
    let fits = |value: SsaVarId| ssa.bit_width(value).is_some_and(|bits| bits <= half);
    (fits(halves.low) && fits(halves.high)).then_some(halves)
}

impl Halves {
    /// Whether both halves are a half wide, so the composition is made of
    /// them with no extension between: the halves a double-width sum or a
    /// wide load reads.
    #[must_use]
    pub fn whole<T: Target>(&self, ssa: &SsaFunction<T>) -> bool {
        ssa.bit_width(self.low) == Some(self.half) && ssa.bit_width(self.high) == Some(self.half)
    }
}

/// The extension a composition of a value and its sign is: `(hi << w) |
/// zext(lo)` where `hi` is `lo` shifted arithmetically right by `w - 1` - the
/// high word a 32-bit target passes for an `int` promoted to `long long` - is
/// `lo` sign-extended to the composition's width.
///
/// # Arguments
///
/// * `ssa` - The function, for resolving each operand's definition.
/// * `op` - The candidate composition, as [`Doubleword::fold`] takes it.
/// * `constants` - Known constant values, for the shift amounts.
///
/// # Returns
///
/// The sign extension, defining `op`'s destination, or `None` when `op` is no
/// such composition.
#[must_use]
pub fn extension<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<SsaOp<T>> {
    let Halves {
        low, high, half, ..
    } = halves(ssa, op, constants)?;
    let dest = op.dest()?;
    if ssa.bit_width(low)? != half {
        return None;
    }
    let SsaOp::Shr {
        value: signed,
        amount,
        unsigned: false,
        ..
    } = ssa.get_definition(peel(ssa, high))?
    else {
        return None;
    };
    let sign = peel(ssa, *signed) == peel(ssa, low)
        && const_amount(ssa, constants, *amount) == half.checked_sub(1);
    sign.then_some(())?;
    Some(SsaOp::IntConv {
        dest,
        operand: low,
        target: ssa.variable(dest)?.var_type().clone(),
        overflow_check: false,
        unsigned: false,
    })
}

/// Two loads of one address's adjacent words ([`word_loads`]).
struct WordLoads {
    /// The address of the lower word, where a load of both begins.
    first: SsaVarId,
    /// The space both loads read, and the permission they read it with.
    space: AccessQualifiers,
    /// The loads' results, the lower word's first.
    words: [SsaVarId; 2],
    /// The width each loads, in bits.
    bits: u32,
}

/// The loads `low` and `high` are, where both load one width from one base,
/// in one space, at adjacent words: `low` the one `byte_order` puts a value's
/// low half in.
fn word_loads<T: Target>(
    ssa: &SsaFunction<T>,
    ptr_size: PointerSize,
    byte_order: Endianness,
    low: SsaVarId,
    high: SsaVarId,
) -> Option<WordLoads> {
    let load = |value: SsaVarId| match ssa.get_definition(value)? {
        SsaOp::LoadIndirect {
            addr,
            value_type,
            access: space,
            ..
        } => Some((*addr, T::bit_width(value_type)?, *space)),
        _ => None,
    };
    let ((low_addr, half, low_space), (high_addr, high_bits, high_space)) =
        (load(low)?, load(high)?);
    if low_space != high_space || half != high_bits || half == 0 {
        return None;
    }
    let (low_base, low_offset) = address(ssa, low_addr, ptr_size)?;
    let (high_base, high_offset) = address(ssa, high_addr, ptr_size)?;
    if low_base != high_base {
        return None;
    }
    let step = i64::from(half / 8);
    let (first, words) = match byte_order {
        Endianness::Little if low_offset.checked_add(step) == Some(high_offset) => {
            (low_addr, [low, high])
        }
        Endianness::Big if high_offset.checked_add(step) == Some(low_offset) => {
            (high_addr, [high, low])
        }
        _ => return None,
    };
    Some(WordLoads {
        first,
        space: low_space,
        words,
        bits: half,
    })
}

/// The position of the instruction defining `var` in `block`, found in the
/// block rather than read off the definition site, which an edit that
/// inserted instructions earlier in the block has not moved.
fn position_in<T: Target>(ssa: &SsaFunction<T>, block: usize, var: SsaVarId) -> Option<usize> {
    ssa.block(block)?
        .instructions()
        .iter()
        .position(|instruction| instruction.op().dest() == Some(var))
}

/// The base and constant byte offset `addr` is measured as, by the address
/// model ([`AddressWalk::normalize`](crate::analysis::address::AddressWalk::normalize)): `None` for an indexed address, or one not
/// a whole number of bytes from its base.
fn address<T: Target>(
    ssa: &SsaFunction<T>,
    addr: SsaVarId,
    ptr_size: PointerSize,
) -> Option<(AddressBase, i64)> {
    let expr = AddressWalk::new(ssa, ptr_size).normalize(addr);
    if expr.index.is_some() || expr.offset_bits % 8 != 0 {
        return None;
    }
    Some((expr.base, expr.offset_bits / 8))
}

/// The positions of `loads` in `block_idx`, when both lie in it before
/// position `end` and no instruction from the first load to `end` may write
/// memory - then a load at `end` reads what they did - nor, in a block a
/// handler protects, throw ([`position_in`]).
fn unwritten<T: Target>(
    ssa: &SsaFunction<T>,
    loads: [SsaVarId; 2],
    block_idx: usize,
    end: usize,
) -> Option<[usize; 2]> {
    let instructions = ssa.block(block_idx)?.instructions();
    let (first, second) = (
        position_in(ssa, block_idx, loads[0])?,
        position_in(ssa, block_idx, loads[1])?,
    );
    if first.max(second) >= end {
        return None;
    }
    let start = first.min(second);
    let between = instructions.get(start..end)?;
    if between
        .iter()
        .any(|instruction| instruction.op().effects().writes_memory())
    {
        return None;
    }
    // The word loads themselves may throw: the wide one does where they did.
    let throws = between.iter().enumerate().any(|(offset, instruction)| {
        let at = start.saturating_add(offset);
        at != first && at != second && instruction.op().may_throw()
    });
    if throws
        && !FunctionRoots::of(ssa)
            .exceptional_successors(block_idx)
            .is_empty()
    {
        return None;
    }
    Some([first, second])
}

/// The value `var` widens, by a zero-extension where `zero` is asked and by
/// either extension otherwise (a high half's sign bits are shifted out).
fn extended<T: Target>(ssa: &SsaFunction<T>, var: SsaVarId, zero: bool) -> Option<SsaVarId> {
    let SsaOp::IntConv {
        operand, unsigned, ..
    } = ssa.get_definition(var)?
    else {
        return None;
    };
    (*unsigned || !zero).then_some(*operand)
}

/// Whether some load or store of `ssa` accesses the `bits` at `addr`, in
/// `space`'s address space, whole: the same bytes, however aligned or hinted
/// the access is, as the memory pass names a cell. A Thumb `vstr d0` that
/// states its word alignment stores the cell two plain `ldr`s read.
fn accessed_whole<T: Target>(
    ssa: &SsaFunction<T>,
    ptr_size: PointerSize,
    addr: SsaVarId,
    space: AccessQualifiers,
    bits: u32,
) -> bool {
    let Some(cell) = address(ssa, addr, ptr_size) else {
        return false;
    };
    ssa.iter_instructions().any(|(_, _, instruction)| {
        let (accessed, value_type, access_space) = match instruction.op() {
            SsaOp::LoadIndirect {
                addr,
                value_type,
                access: space,
                ..
            }
            | SsaOp::StoreIndirect {
                addr,
                value_type,
                access: space,
                ..
            } => (*addr, value_type, *space),
            _ => return false,
        };
        access_space.address_space == space.address_space
            && T::bit_width(value_type) == Some(bits)
            && address(ssa, accessed, ptr_size) == Some(cell)
    })
}

/// Every value of `var`'s sum tree at width `half` that sums others - `var`
/// itself among them - with the terms it sums, as [`terms`] flattens them.
fn sum_nodes<T: Target>(
    ssa: &SsaFunction<T>,
    var: SsaVarId,
    half: u32,
) -> Vec<(SsaVarId, Vec<Term>)> {
    let mut nodes = Vec::new();
    // A value the tree reaches again is read once: a tree sharing a sum
    // would read it on every path down to it, and code no edge reaches can
    // define a sum through itself (`a = b + c; b = a + d`), which no count of
    // the nodes found would end.
    let mut visited = VarSet::new(ssa.var_id_bound());
    let mut pending = vec![peel(ssa, var)];
    while let Some(value) = pending.pop() {
        if !visited.insert(value) || nodes.len() > MOST_TERMS || ssa.bit_width(value) != Some(half)
        {
            continue;
        }
        let Some(SsaOp::Add { left, right, .. } | SsaOp::Sub { left, right, .. }) =
            ssa.get_definition(value)
        else {
            continue;
        };
        if let Some(terms) = terms(ssa, value, half) {
            nodes.push((value, terms));
        }
        pending.push(peel(ssa, *left));
        pending.push(peel(ssa, *right));
    }
    nodes
}

/// Whether two lists of terms sum the same values with the same signs, in
/// whatever order.
fn same_terms(left: &[Term], right: &[Term]) -> bool {
    let key = |term: &Term| (term.value.index(), term.negative);
    let mut left: Vec<_> = left.iter().map(key).collect();
    let mut right: Vec<_> = right.iter().map(key).collect();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

/// Whether `value` is the low half of a step of a sum or difference whose
/// carry or borrow is among `terms`.
fn carried<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId, terms: &[Term]) -> bool {
    match ssa.get_definition(value) {
        Some(SsaOp::Add { left, right, .. }) => {
            let addends = [peel(ssa, *left), peel(ssa, *right)];
            terms
                .iter()
                .any(|term| !term.negative && is_carry(ssa, term.value, value, addends))
        }
        Some(SsaOp::Sub { left, right, .. }) => {
            let (minuend, subtrahend) = (peel(ssa, *left), peel(ssa, *right));
            terms
                .iter()
                .any(|term| term.negative && is_borrow(ssa, term.value, minuend, subtrahend))
        }
        _ => false,
    }
}

/// The terms `var` adds and subtracts, through every addition and subtraction
/// at width `half`: `None` beyond [`MOST_TERMS`].
fn terms<T: Target>(ssa: &SsaFunction<T>, var: SsaVarId, half: u32) -> Option<Vec<Term>> {
    let mut pending = vec![Term {
        value: peel(ssa, var),
        negative: false,
    }];
    let mut terms = Vec::new();
    while let Some(term) = pending.pop() {
        let at_width = ssa.bit_width(term.value) == Some(half);
        match ssa.get_definition(term.value) {
            Some(SsaOp::Add { left, right, .. }) if at_width => {
                pending.push(Term {
                    value: peel(ssa, *right),
                    negative: term.negative,
                });
                pending.push(Term {
                    value: peel(ssa, *left),
                    negative: term.negative,
                });
            }
            Some(SsaOp::Sub { left, right, .. }) if at_width => {
                pending.push(Term {
                    value: peel(ssa, *right),
                    negative: !term.negative,
                });
                pending.push(Term {
                    value: peel(ssa, *left),
                    negative: term.negative,
                });
            }
            _ => terms.push(term),
        }
        if terms.len().saturating_add(pending.len()) > MOST_TERMS {
            return None;
        }
    }
    Some(terms)
}

/// Whether `high` is `low` shifted arithmetically right by one less than its
/// width: the high half of `low` sign-extended.
fn sign_fill<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    low: SsaVarId,
    high: SsaVarId,
    half: u32,
) -> bool {
    matches!(
        ssa.get_definition(peel(ssa, high)),
        Some(SsaOp::Shr { value, amount, unsigned: false, .. })
            if peel(ssa, *value) == peel(ssa, low)
                && half.checked_sub(1) == const_amount(ssa, constants, *amount)
    )
}

/// Whether `low` and `high` are the two halves of one value twice their
/// width: its truncation, and its truncation shifted down by the half width.
fn split<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    low: SsaVarId,
    high: SsaVarId,
    half: u32,
) -> bool {
    let narrowed = |var: SsaVarId| match ssa.get_definition(var)? {
        SsaOp::IntConv { operand, .. } => Some(*operand),
        _ => None,
    };
    let (Some(whole), Some(shifted)) = (narrowed(low), narrowed(high)) else {
        return false;
    };
    let Some(SsaOp::Shr { value, amount, .. }) = ssa.get_definition(shifted) else {
        return false;
    };
    peel(ssa, *value) == peel(ssa, whole)
        && ssa.bit_width(whole) == half.checked_mul(2)
        && const_amount(ssa, constants, *amount) == Some(half)
}

/// The comparison whose zero or one `term` adds: the term itself, or the
/// boolean its zero-extension to the halves' width widens.
fn flag<T: Target>(ssa: &SsaFunction<T>, term: SsaVarId) -> SsaVarId {
    match ssa.get_definition(term) {
        Some(SsaOp::IntConv {
            operand,
            unsigned: true,
            ..
        }) => peel(ssa, *operand),
        _ => term,
    }
}

/// Whether `term` is the carry out of `sum = addends[0] + addends[1]`: `sum <u
/// addend`, or `addend >u sum`, for either addend.
fn is_carry<T: Target>(
    ssa: &SsaFunction<T>,
    term: SsaVarId,
    sum: SsaVarId,
    addends: [SsaVarId; 2],
) -> bool {
    match ssa.get_definition(flag(ssa, term)) {
        Some(SsaOp::Clt {
            left,
            right,
            unsigned: true,
            ..
        }) => peel(ssa, *left) == sum && addends.contains(&peel(ssa, *right)),
        Some(SsaOp::Cgt {
            left,
            right,
            unsigned: true,
            ..
        }) => peel(ssa, *right) == sum && addends.contains(&peel(ssa, *left)),
        _ => false,
    }
}

/// Whether `term` is the borrow out of `minuend - subtrahend`: `minuend <u
/// subtrahend`, or `subtrahend >u minuend`.
fn is_borrow<T: Target>(
    ssa: &SsaFunction<T>,
    term: SsaVarId,
    minuend: SsaVarId,
    subtrahend: SsaVarId,
) -> bool {
    match ssa.get_definition(flag(ssa, term)) {
        Some(SsaOp::Clt {
            left,
            right,
            unsigned: true,
            ..
        }) => peel(ssa, *left) == minuend && peel(ssa, *right) == subtrahend,
        Some(SsaOp::Cgt {
            left,
            right,
            unsigned: true,
            ..
        }) => peel(ssa, *left) == subtrahend && peel(ssa, *right) == minuend,
        _ => false,
    }
}

/// A comparison of two double-width values a target computes in halves,
/// read whole ([`comparison`]).
#[derive(Debug, Clone)]
pub struct WideComparison<T: Target> {
    /// The comparison of the whole values, defining what the comparison of
    /// the high halves defined.
    pub op: SsaOp<T>,
    /// Where one side is two constants: the operand of [`Self::op`] the
    /// constant stands in for, the constant at the double width, and that
    /// width's type. A caller defines the constant and reads it there.
    pub constant: Option<(SsaVarId, ConstValue<T>, T::Type)>,
}

/// One side of a comparison in halves: a value twice a half wide, or a
/// constant of that width.
enum Whole {
    /// A value whose halves the comparison reads.
    Value(SsaVarId),
    /// The constant its halves make.
    Constant(u128),
}

/// The comparison of two double-width values `op` computes in halves, where
/// each side's halves are those of one value twice their width, or constants.
///
/// A target whose registers are half as wide as a value compares it a half at
/// a time: the borrow out of the low halves, then the high halves with the
/// borrow taken from the second's (ARM's `subs` and `sbcs`, x86's `cmp` and
/// `sbb`, RISC-V's `sltu`). Once the flags are resolved every one of them is
///
/// ```text
/// b = al <u bl
/// a < b   is   ah < bh + b
/// ```
///
/// with the high halves compared unsigned for an unsigned comparison and
/// signed for a signed one: the borrow is one exactly where the low halves
/// order `a` below `b`, so the high halves decide with it added. The identity
/// needs `bh + b` not to wrap, which holds where `bh` is zero (no addition) or
/// a constant below the largest the comparison reads. `dash`'s `number` tests
/// `n > INT_MAX` of its 64-bit `n` as `(uint32_t)(n >> 32) >= (uint32_t)n <u
/// 0x80000000` on ARM, which is `n >=u 0x80000000`.
///
/// Like the sums ([`Doubleword::fold`]), a comparison is read whole only where its halves
/// are whole already: two halves no value composes stay two halves.
///
/// # Arguments
///
/// * `ssa` - The function, for the definitions and types.
/// * `op` - The candidate: the comparison of the high halves.
/// * `constants` - Known constant values.
///
/// # Returns
///
/// The whole comparison, or `None` when `op` is no comparison in halves of
/// whole values, or both sides are constants.
#[must_use]
pub fn comparison<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<WideComparison<T>> {
    let (dest, high, carried, unsigned) = match op {
        SsaOp::Clt {
            dest,
            left,
            right,
            unsigned,
            signaling: false,
        } => (*dest, *left, *right, *unsigned),
        SsaOp::Cgt {
            dest,
            left,
            right,
            unsigned,
            signaling: false,
        } => (*dest, *right, *left, *unsigned),
        _ => return None,
    };
    let half = ssa.bit_width(high)?;
    if half == 0 || half > 64 || ssa.bit_width(carried)? != half {
        return None;
    }
    let constant = |var: SsaVarId| constants.get(&var).and_then(ConstValue::as_u64_bits);
    // The borrow as a value a half wide, and the second's high half it is
    // added to: none, or a constant the borrow cannot carry past.
    let all = BitWidth::new(half).mask();
    let largest = if unsigned { all } else { all >> 1 };
    let (borrow, second_high) = match ssa.get_definition(carried)? {
        SsaOp::IntConv {
            operand,
            unsigned: true,
            overflow_check: false,
            ..
        } => (*operand, 0),
        SsaOp::Add { left, right, .. } => {
            let extended = |var: SsaVarId| match ssa.get_definition(var)? {
                SsaOp::IntConv {
                    operand,
                    unsigned: true,
                    overflow_check: false,
                    ..
                } => Some(*operand),
                _ => None,
            };
            let (borrow, high) = [(*left, *right), (*right, *left)]
                .into_iter()
                .find_map(|(borrow, high)| Some((extended(borrow)?, constant(high)?)))?;
            (high < largest).then_some(())?;
            (borrow, high)
        }
        _ => return None,
    };
    // The borrow out of the low halves, through copies and double negations.
    let mut borrow = borrow;
    let (first_low, second_low) = loop {
        match ssa.get_definition(borrow)? {
            SsaOp::Copy { src, .. } => borrow = *src,
            SsaOp::BoolNot { value, .. } => match ssa.get_definition(*value)? {
                SsaOp::BoolNot { value, .. } => borrow = *value,
                _ => return None,
            },
            SsaOp::Clt {
                left,
                right,
                unsigned: true,
                signaling: false,
                ..
            } => break (*left, *right),
            SsaOp::Cgt {
                left,
                right,
                unsigned: true,
                signaling: false,
                ..
            } => break (*right, *left),
            _ => return None,
        }
    };
    if ssa.bit_width(first_low)? != half || ssa.bit_width(second_low)? != half {
        return None;
    }
    // Each side whole: its halves a value's, or constants.
    let whole = |high: Option<SsaVarId>, high_bits: Option<u64>, low: SsaVarId| {
        let high_bits = high_bits.or_else(|| high.and_then(constant));
        match (high_bits, constant(low)) {
            (Some(high), Some(low)) => Some(Whole::Constant(
                (u128::from(high) << half) | u128::from(low),
            )),
            (None, None) => {
                let (value, at_high) = Reader::new(ssa, constants).copied_field(high?)?;
                let (low_value, at_low) = Reader::new(ssa, constants).copied_field(low)?;
                (value == low_value
                    && at_high == half
                    && at_low == 0
                    && ssa.bit_width(value) == Some(half.checked_mul(2)?))
                .then_some(Whole::Value(value))
            }
            _ => None,
        }
    };
    let first = whole(Some(high), None, first_low)?;
    let second = whole(None, Some(second_high), second_low)?;
    let wide_type = |value: SsaVarId| ssa.variable(value).map(|var| var.var_type().clone());
    let wide_constant = |value: u128, ty: &T::Type| {
        let bytes = usize::try_from(half.checked_mul(2)? / 8).ok()?;
        ConstValue::from_typed_bytes(value.to_le_bytes().get(..bytes)?, ty, Endianness::Little)
    };
    // A constant stands in the place of the operand the high comparison read
    // on its side, which the comparison no longer reads.
    let (left, right, constant) = match (first, second) {
        (Whole::Value(first), Whole::Value(second)) => (first, second, None),
        (Whole::Value(first), Whole::Constant(value)) => {
            let ty = wide_type(first)?;
            (
                first,
                carried,
                Some((carried, wide_constant(value, &ty)?, ty)),
            )
        }
        (Whole::Constant(value), Whole::Value(second)) => {
            let ty = wide_type(second)?;
            (high, second, Some((high, wide_constant(value, &ty)?, ty)))
        }
        (Whole::Constant(_), Whole::Constant(_)) => return None,
    };
    Some(WideComparison {
        op: SsaOp::Clt {
            dest,
            left,
            right,
            unsigned,
            signaling: false,
        },
        constant,
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU8;

    use super::*;
    use crate::{
        ir::{
            SsaBlock, SsaBlockBuilder, SsaDefSpec, SsaFunctionBuilder, SsaInstruction,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// **A sum in halves is read through only while the nesting budget
    /// lasts.** `low = al + bl` and `high = ah + bh + (low <u al)`: with a
    /// level to spend, `high` fits `low` as the high half of their sum; with
    /// none left it fits as nothing, so a sum nested in sums as deep as the
    /// code that built it is read no deeper than [`MOST_NESTED`].
    #[test]
    fn a_sum_in_halves_is_read_through_only_while_the_budget_lasts() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
        let mut halves = None;
        builder
            .in_block(0, |block| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let al = block.load_arg(u32d(), 0)?;
                let ah = block.load_arg(u32d(), 1)?;
                let bl = block.load_arg(u32d(), 2)?;
                let bh = block.load_arg(u32d(), 3)?;
                let low = block.add(u32d(), al, bl)?;
                let carry = block.clt(SsaDefSpec::tmp(MockType::I1), low, al, true)?;
                let carry = block.conv(u32d(), carry, MockType::U32, false, true)?;
                let partial = block.add(u32d(), ah, bh)?;
                let high = block.add(u32d(), partial, carry)?;
                halves = Some((low, high));
                block.ret(Some(high))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let (low, high) = halves.expect("halves");
        let constants = BTreeMap::new();
        assert!(Doubleword::new(&ssa, &constants, MACHINE).fits(low, high, 1));
        assert!(!Doubleword::new(&ssa, &constants, MACHINE).fits(low, high, 0));
    }

    /// **Two words are one value's halves where a composition of them
    /// reduces.** A 32-bit target passes a `double` it keeps in memory as its
    /// two words, loaded from `p` and `p + 4`: one value on a little-endian
    /// target where the function stores the cell whole as well, not on a
    /// big-endian one, not with a store between the first load and the
    /// reader, not where a word is read apart as well, and not where nothing
    /// accesses the cell whole (`main`'s `argc` and `argv`). The
    /// two halves of one wider value, `(u32)x` and `(u32)(x >> 32)`, and a
    /// value with its sign above it, `lo` and `lo >> 31`, are one value; two
    /// unrelated words are not.
    #[test]
    fn two_words_are_one_value_where_a_composition_of_them_reduces() {
        // `(stored, shared, whole)`: a store before the reader, the low word
        // read apart, the cell stored whole after the reader; answers for
        // little- and big-endian targets.
        let loaded = |stored: bool, shared: bool, whole: bool| -> [bool; 2] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            let mut halves = None;
            builder
                .in_block(0, |block| {
                    let pointer = block.load_arg(SsaDefSpec::tmp(MockType::Ptr), 0)?;
                    let word = |block: &mut SsaBlockBuilder<'_, MockTarget>, offset| {
                        let addr = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| {
                            SsaOp::PtrAdd {
                                dest,
                                base: pointer,
                                index: None,
                                stride: 0,
                                offset,
                                result_type: MockType::Ptr,
                            }
                        })?;
                        block.emit_def(SsaDefSpec::tmp(MockType::U32), |dest| SsaOp::LoadIndirect {
                            dest,
                            addr,
                            value_type: MockType::U32,
                            access: AccessQualifiers::FLAT,
                        })
                    };
                    let lo = word(block, 0)?;
                    let hi = word(block, 4)?;
                    if shared {
                        block.add(SsaDefSpec::tmp(MockType::U32), lo, lo)?;
                    }
                    if stored {
                        block.emit_no_defs(SsaOp::StoreIndirect {
                            addr: pointer,
                            value: lo,
                            value_type: MockType::U32,
                            access: AccessQualifiers::FLAT,
                        })?;
                    }
                    // The reader taking the two as one value, as a call takes
                    // its arguments.
                    let reader = block.add(SsaDefSpec::tmp(MockType::U32), lo, hi)?;
                    if whole {
                        // Stated word-aligned, as a Thumb `vstr` is: the
                        // same cell the plain word loads read.
                        let value = block.const_i64(SsaDefSpec::tmp(MockType::U64), 0)?;
                        block.emit_no_defs(SsaOp::StoreIndirect {
                            addr: pointer,
                            value,
                            value_type: MockType::U64,
                            access: AccessQualifiers::FLAT
                                .with_alignment(NonZeroU8::new(4).expect("four")),
                        })?;
                    }
                    halves = Some(([lo, hi], reader));
                    block.ret(None)
                })
                .expect("build block");
            let ssa = builder.finish().expect("finish");
            let (halves, reader) = halves.expect("halves");
            let at = ssa
                .block(0)
                .and_then(|block| {
                    block
                        .instructions()
                        .iter()
                        .position(|instruction| instruction.op().dest() == Some(reader))
                })
                .map(|position| (0, position))
                .expect("the reader");
            let uses = ssa.count_uses();
            [Endianness::Little, Endianness::Big].map(|order| {
                let machine = Machine {
                    byte_order: Some(order),
                    ..MACHINE
                };
                Doubleword::new(&ssa, &BTreeMap::new(), machine).one_value(&uses, halves, 32, at)
            })
        };
        assert_eq!(
            loaded(false, false, true),
            [true, false],
            "adjacent words of a cell accessed whole"
        );
        assert_eq!(
            loaded(false, false, false),
            [false, false],
            "adjacent words of nothing accessed whole"
        );
        assert_eq!(loaded(true, false, true), [false, false], "a store between");
        assert_eq!(
            loaded(false, true, true),
            [false, false],
            "a word read apart"
        );

        // The halves of one 64-bit argument, of the low word with its sign,
        // and two unrelated arguments.
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut cases = None;
        builder
            .in_block(0, |block| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let x = block.load_arg(SsaDefSpec::tmp(MockType::U64), 0)?;
                let other = block.load_arg(u32d(), 1)?;
                let thirty_two =
                    block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let low = block.conv(u32d(), x, MockType::U32, false, true)?;
                let shifted = block.shr(SsaDefSpec::tmp(MockType::U64), x, thirty_two, true)?;
                let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                let thirty_one =
                    block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(31))?;
                let sign = block.shr(u32d(), other, thirty_one, false)?;
                cases = Some(([low, high], [other, sign], [low, other]));
                block.ret(None)
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let (split, signed, apart) = cases.expect("cases");
        let constants = ssa.find_constants();
        let uses = ssa.count_uses();
        let machine = Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        };
        let one = |halves| {
            Doubleword::new(&ssa, &constants, machine).one_value(&uses, halves, 32, (0, 0))
        };
        assert!(one(split), "the halves of one wider value");
        assert!(one(signed), "a value with its sign above it");
        assert!(!one(apart), "two unrelated words");
    }

    /// **A sum defined through itself is read once.** Code no edge reaches
    /// can sum a value into its own definition without a phi: `a = b + c; b =
    /// a + d`. Neither is a tree of terms (each flattens without end), so no
    /// node is ever found, and no count of the nodes found ends the walk. Read
    /// once per value, it ends with none.
    #[test]
    fn a_sum_defined_through_itself_is_read_once() {
        let mut ssa = SsaFunction::<MockTarget>::with_capacity(2, 0, 1, 4);
        let defined = |ssa: &mut SsaFunction<MockTarget>, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(u16::try_from(at).unwrap_or(0)),
                0,
                DefSite::instruction(0, at),
                MockType::U32,
            )
        };
        let c = defined(&mut ssa, 0);
        let d = defined(&mut ssa, 1);
        let a = defined(&mut ssa, 2);
        let b = defined(&mut ssa, 3);
        let mut block = SsaBlock::with_capacity(0, 0, 4);
        for (dest, arg_index) in [(c, 0), (d, 1)] {
            block.add_instruction(SsaInstruction::new((), SsaOp::LoadArg { dest, arg_index }));
        }
        for (dest, left, right) in [(a, b, c), (b, a, d)] {
            block.add_instruction(SsaInstruction::new(
                (),
                SsaOp::Add {
                    dest,
                    left,
                    right,
                    flags: None,
                },
            ));
        }
        ssa.add_block(block);
        ssa.recompute_uses();

        assert!(sum_nodes(&ssa, a, 32).is_empty());
    }
}
