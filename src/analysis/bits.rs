//! Which bits of an integer value are known: certainly clear, certainly set.
//!
//! A lift spells a field of a register as the register's bits: a write of the
//! field clears its place in the container and ors the new value in, a read
//! shifts the container down and masks. RISC-V's `fsrmi 3` followed by `frrm
//! a0` is `((fcsr & 0x1f) | ((3 & 7) << 5)) >> 5 & 7`: three whatever the
//! container held, since every bit the read keeps came from the write. No
//! algebraic identity sees that - the old container is unknown - but its bits
//! that matter are known, and so are all of the result's.
//!
//! So each value's bits are read from its definition: a constant's all, an
//! `and`, `or`, `xor` or `not` of its operands', a shift by a constant amount
//! of its operand's, moved, a conversion's, kept, cut or extended, and a
//! select's, those its two values agree on. And each bit that is not known
//! has an origin: the bit of another value it is a copy of, carried through
//! the same copies, conversions, constant shifts and masks, and through an
//! `or` or `xor` whose other terms are known clear there. The rules that
//! follow:
//!
//! - **A value every bit of which is known is a constant**
//!   ([`Reader::known_constant`]).
//! - **An `and` whose operand has no bit outside the mask is its operand**
//!   ([`Reader::redundant_mask`]): a shift count masked twice, by the program and by
//!   the ISA's count rule, as `-O0` code and a lift that states the count rule
//!   each do. The second `and` changes nothing, but stands between the count
//!   and the rotate idiom that looks for `r & 31` against `(-r) & 31`.
//! - **A value every unknown bit of which is the same bit of one other value,
//!   which has no other bit, is that value** ([`Reader::copied_value`]):
//!   `((c & ~(1 << 27)) | (q << 27)) >> 27` is `q` where `q` is zero or one,
//!   through any extension or truncation between: a one-bit flag written
//!   into a status register and read back out of it.
//! - **A field read back is the value written into it** ([`Reader::demanded_field`]):
//!   `((c & ~m) | ((v & 15) << 24)) >> 24 & 15` reads only bits the write put
//!   there, so it is `(v & 15) & 15`, whatever the container `c` held. A
//!   MIPS `cmpu.lt.qb` writes the DSP condition field this way and `pick.qb`
//!   reads it, and neither `c` nor `v` need be known.
//!
//! The other direction is what a value's readers demand of it
//! ([`DemandedBits`]): how many of its low bits any of them can tell apart.
//! A truncation keeps its operand's low bits, an addition, subtraction,
//! multiplication, bitwise operation or left shift reads its operands' low
//! bits up to the highest its own readers demand, and anything else reads the
//! whole value. `char c` passed in a 64-bit register and read only as `(c -
//! '0') & 0xff` and `(int)c` demands eight bits: it is a byte, whatever the
//! register's width.
//!
//! # Complexity
//!
//! Each value's bits are read once per query and remembered, through at most
//! sixteen definitions from the one asked about.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use crate::{
    ir::{
        SsaFunction,
        integer::{BitWidth, IntegerBits},
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// How many of each value's low bits the function's readers of it demand: the
/// bits any of them can tell apart ([`DemandedBits::of`]).
///
/// The demand flows from each reader to the values it reads. A truncation or
/// extension reads as many of its operand's low bits as its own readers demand
/// of its result, up to the operand's width; an addition, subtraction,
/// multiplication, `and`, `or`, `xor`, `not`, negation or left shift, none of
/// whose low bits depends on a higher one, reads as many of its operands' as
/// its readers demand of it; a copy and a phi pass the demand on. Every other
/// reader demands the whole value: a comparison, a right shift, a division, an
/// address, a stored value, an argument, a returned value, and an operation
/// whose flags are read, since a carry or a sign depends on the high bits.
///
/// Solved with an explicit worklist, each value's demand only rising and at
/// most to its width, so a function's chains of any length and its loops cost
/// no stack and end.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DemandedBits {
    /// The low bits demanded of each value something reads; a value nothing
    /// reads is absent and demands none.
    demanded: FxHashMap<SsaVarId, u32>,
}

impl DemandedBits {
    /// What `ssa`'s readers demand of each of its values.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let mut solver = DemandSolver {
            ssa,
            demanded: FxHashMap::default(),
            pending: Vec::new(),
            phis: FxHashMap::default(),
        };
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                solver.phis.insert(
                    phi.result(),
                    phi.operands()
                        .iter()
                        .map(|operand| operand.value())
                        .collect(),
                );
            }
        }
        for block in ssa.blocks() {
            for instruction in block.instructions() {
                solver.read(instruction.op());
            }
        }
        while let Some(value) = solver.pending.pop() {
            if let Some(operands) = solver.phis.get(&value).cloned() {
                let demand = solver.demand(value);
                for operand in operands {
                    solver.raise(operand, demand);
                }
            } else if let Some(op) = ssa.get_definition(value) {
                solver.read(op);
            }
        }
        Self {
            demanded: solver.demanded,
        }
    }

    /// How many of `value`'s low bits its readers demand: `0` for a value
    /// nothing reads, its width for one something reads whole.
    #[must_use]
    pub fn of_value(&self, value: SsaVarId) -> u32 {
        self.demanded.get(&value).copied().unwrap_or(0)
    }
}

/// [`DemandedBits::of`]'s state: the demand found so far, and the values whose
/// demand rose since their definitions were last read.
struct DemandSolver<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// The low bits demanded of each value so far.
    demanded: FxHashMap<SsaVarId, u32>,
    /// The values whose demand rose and whose definitions are to be read
    /// again.
    pending: Vec<SsaVarId>,
    /// Each phi's result and the values it merges.
    phis: FxHashMap<SsaVarId, Vec<SsaVarId>>,
}

impl<T: Target> DemandSolver<'_, T> {
    /// The low bits demanded of `value` so far.
    fn demand(&self, value: SsaVarId) -> u32 {
        self.demanded.get(&value).copied().unwrap_or(0)
    }

    /// Records that a reader demands `bits` of `value`'s low bits, at most its
    /// width, and queues its definition where that raised its demand.
    fn raise(&mut self, value: SsaVarId, bits: u32) {
        let bits = self
            .ssa
            .bit_width(value)
            .map_or(bits, |width| bits.min(width));
        if bits > self.demand(value) {
            self.demanded.insert(value, bits);
            self.pending.push(value);
        }
    }

    /// Records what `op` demands of each value it reads, given what its
    /// readers demand of its result.
    fn read(&mut self, op: &SsaOp<T>) {
        match op {
            SsaOp::Add {
                dest,
                left,
                right,
                flags: None,
            }
            | SsaOp::Sub {
                dest,
                left,
                right,
                flags: None,
            }
            | SsaOp::Mul {
                dest,
                left,
                right,
                flags: None,
            }
            | SsaOp::And {
                dest,
                left,
                right,
                flags: None,
            }
            | SsaOp::Or {
                dest,
                left,
                right,
                flags: None,
            }
            | SsaOp::Xor {
                dest,
                left,
                right,
                flags: None,
            } => {
                let demand = self.demand(*dest);
                self.raise(*left, demand);
                self.raise(*right, demand);
            }
            SsaOp::Not {
                dest,
                operand,
                flags: None,
            }
            | SsaOp::Neg {
                dest,
                operand,
                flags: None,
            } => {
                let demand = self.demand(*dest);
                self.raise(*operand, demand);
            }
            SsaOp::Shl {
                dest,
                value,
                amount,
                flags: None,
            } => {
                let demand = self.demand(*dest);
                self.raise(*value, demand);
                self.raise(*amount, u32::MAX);
            }
            SsaOp::IntConv {
                dest,
                operand,
                overflow_check: false,
                ..
            } => {
                let demand = self.demand(*dest);
                self.raise(*operand, demand);
            }
            SsaOp::Copy { dest, src } => {
                let demand = self.demand(*dest);
                self.raise(*src, demand);
            }
            other => other.for_each_use(|value| self.raise(value, u32::MAX)),
        }
    }
}

/// How many definitions a query reads through from the value it asks about.
const MOST_LINKS_CROSSED: usize = 16;

/// What is known of an integer value's bits at its width: which are certainly
/// clear and which certainly set. A bit in neither may be either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownBits {
    /// The bits certainly clear.
    zeros: u64,
    /// The bits certainly set.
    ones: u64,
    /// The value's width, at most 64 bits.
    width: u32,
}

impl KnownBits {
    /// Nothing known of a value `width` bits wide.
    const fn unknown(width: u32) -> Self {
        Self {
            zeros: 0,
            ones: 0,
            width,
        }
    }

    /// The value `bits` at `width`, every bit known.
    const fn constant(bits: u64, width: u32) -> Self {
        let mask = BitWidth::new(width).mask();
        Self {
            zeros: !bits & mask,
            ones: bits & mask,
            width,
        }
    }

    /// The value, when every bit of it is known.
    #[must_use]
    pub const fn value(self) -> Option<u64> {
        if self.zeros | self.ones == BitWidth::new(self.width).mask() {
            Some(self.ones)
        } else {
            None
        }
    }

    /// The bits the value can have set.
    #[must_use]
    pub const fn possibly_set(self) -> u64 {
        BitWidth::new(self.width).low_bits(!self.zeros)
    }

    /// Whether the value is never zero: some bit of it is known set.
    #[must_use]
    pub const fn never_zero(self) -> bool {
        self.ones != 0
    }

    /// Whether bit `bit` is known clear.
    const fn clear_at(self, bit: u32) -> bool {
        match 1u64.checked_shl(bit) {
            Some(at) => self.zeros & at != 0,
            None => false,
        }
    }

    /// Whether bit `bit` is known set.
    const fn set_at(self, bit: u32) -> bool {
        match 1u64.checked_shl(bit) {
            Some(at) => self.ones & at != 0,
            None => false,
        }
    }

    /// Projects this bounded public analysis view into the shared transfer core.
    fn wide(self) -> Option<IntegerBits> {
        IntegerBits::new(self.width, u128::from(self.zeros), u128::from(self.ones))
    }

    /// Projects shared facts back without broadening the analysis's public width contract.
    fn narrow(bits: IntegerBits) -> Option<Self> {
        if bits.width() > 64 {
            return None;
        }
        Some(Self {
            zeros: u64::try_from(bits.zeros()).ok()?,
            ones: u64::try_from(bits.ones()).ok()?,
            width: bits.width(),
        })
    }

    /// Applies one shared binary transfer at the attested common width.
    fn combined(
        self,
        other: Self,
        width: u32,
        transfer: impl FnOnce(IntegerBits, IntegerBits) -> Option<IntegerBits>,
    ) -> Option<Self> {
        if self.width != width || other.width != width {
            return None;
        }
        Self::narrow(transfer(self.wide()?, other.wide()?)?)
    }

    /// Moves known bits left using the shared fixed-width transfer.
    fn shifted_left(self, amount: u32) -> Option<Self> {
        Self::narrow(self.wide()?.shifted_left(amount))
    }

    /// Moves known bits right with the declared zero or sign fill.
    fn shifted_right(self, amount: u32, unsigned: bool) -> Option<Self> {
        Self::narrow(self.wide()?.shifted_right(amount, unsigned))
    }

    /// Truncates or extends known bits through the shared integer layout.
    fn converted(self, width: u32, unsigned: bool) -> Option<Self> {
        Self::narrow(self.wide()?.converted(width, unsigned)?)
    }
}

/// Where one bit of a value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BitFrom {
    /// Known clear or set ([`KnownBits`] says which).
    Known,
    /// A copy of bit `bit` of `source`.
    Copy {
        /// The value the bit is copied from.
        source: SsaVarId,
        /// Its position there.
        bit: u32,
    },
    /// Computed from more than one bit.
    Mixed,
}

/// What one function's values hold bit by bit: the bits of a value known
/// set or clear, and where each bit of a value is copied from, with
/// `constants` known ([`Self::known_bits`]).
///
/// Each question is answered afresh: what one reads is not kept for the next.
#[derive(Clone, Copy)]
pub struct Reader<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Known constant values.
    constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
}

impl<'a, T: Target> Reader<'a, T> {
    /// Reads the values of `ssa`, with `constants` known.
    #[must_use]
    pub const fn new(
        ssa: &'a SsaFunction<T>,
        constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
    ) -> Self {
        Self { ssa, constants }
    }

    /// Returns the bits known of `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value asked about.
    ///
    /// # Returns
    ///
    /// What is known of `value`'s bits, or `None` when it is not an integer of at
    /// most 64 bits.
    #[must_use]
    pub fn known_bits(&self, value: SsaVarId) -> Option<KnownBits> {
        Reads::new(self.ssa, self.constants).bits(value, MOST_LINKS_CROSSED)
    }

    /// Returns the constant `op`'s result is, when every bit of it is known
    /// though its operands are not all constants: an `and`, `or`, `xor`, `not`,
    /// shift or conversion.
    ///
    /// # Arguments
    ///
    /// * `op` - The candidate operation.
    ///
    /// # Returns
    ///
    /// The constant, of the result's type, or `None` when some bit of the result
    /// is not known or the operation is none of those.
    #[must_use]
    pub fn known_constant(&self, op: &SsaOp<T>) -> Option<ConstValue<T>> {
        let dest = match op {
            SsaOp::And { dest, .. }
            | SsaOp::Or { dest, .. }
            | SsaOp::Xor { dest, .. }
            | SsaOp::Not { dest, .. }
            | SsaOp::Shl { dest, .. }
            | SsaOp::Shr { dest, .. }
            | SsaOp::IntConv {
                dest,
                overflow_check: false,
                ..
            } => *dest,
            _ => return None,
        };
        let mut reader = Reads::new(self.ssa, self.constants);
        let width = reader.width(dest)?;
        let bits = reader.result(op, width, MOST_LINKS_CROSSED)?.value()?;
        ConstValue::from_bits_of_type(bits, self.ssa.variable(dest)?.var_type())
    }

    /// Returns `op` rewritten to the one value every bit of its result that is not
    /// known copies, at the same position, when that value has no bit the result
    /// does not copy: the value itself, or its zero-extension or truncation to the
    /// result's width.
    ///
    /// A one-bit flag written into a status register and read back out is the
    /// flag, whatever else the register held, through every shift, mask and
    /// conversion between: `((c & ~(1 << 27)) | (q << 27)) >> 27` is `q` where `q`
    /// is zero or one.
    ///
    /// # Arguments
    ///
    /// * `op` - The candidate operation: an `and`, `or`, `xor`, shift or
    ///   conversion.
    ///
    /// # Returns
    ///
    /// The copy or conversion of the value, or `None` when a bit of the result is
    /// computed from more than one, copies a bit of another value or of the same
    /// value at another position, is fixed to other than the value's bit there,
    /// the value has a bit the result drops, or `op` already is a conversion of
    /// it.
    #[must_use]
    pub fn copied_value(&self, op: &SsaOp<T>) -> Option<SsaOp<T>> {
        let dest = match op {
            SsaOp::And { dest, .. }
            | SsaOp::Or { dest, .. }
            | SsaOp::Xor { dest, .. }
            | SsaOp::Shl { dest, .. }
            | SsaOp::Shr { dest, .. }
            | SsaOp::IntConv {
                dest,
                overflow_check: false,
                ..
            } => *dest,
            _ => return None,
        };
        let mut reader = Reads::new(self.ssa, self.constants);
        let width = reader.width(dest)?;
        let known = reader.result(op, width, MOST_LINKS_CROSSED)?;
        let origins = reader.result_origins(op, width, MOST_LINKS_CROSSED)?;
        let copies = |bit: u32| {
            let origin = usize::try_from(bit).ok().and_then(|at| origins.get(at));
            match origin {
                _ if known.clear_at(bit) || known.set_at(bit) => None,
                Some(BitFrom::Copy { source, bit: from }) if *from == bit => Some(*source),
                _ => Some(dest),
            }
        };
        let mut source = None;
        for bit in 0..width {
            match (copies(bit), source) {
                (None, _) => {}
                (Some(copied), None) if copied != dest => source = Some(copied),
                (Some(copied), Some(found)) if copied == found => {}
                _ => return None,
            }
        }
        let source = source?;
        // A conversion of the value already is the value at the result's width.
        if matches!(op, SsaOp::IntConv { operand, .. } if *operand == source) {
            return None;
        }
        let held = reader.bits(source, MOST_LINKS_CROSSED)?;
        let source_width = reader.width(source)?;
        // Every bit the value can have is one the result copies.
        if (0..source_width).any(|bit| !held.clear_at(bit) && copies(bit) != Some(source)) {
            return None;
        }
        // A bit the operation fixes is the value's own there, and clear past it:
        // `zext(b) | -2` sets bits `b` has clear, and is no copy of it.
        let agrees = |bit: u32| {
            if bit >= source_width {
                return !known.set_at(bit);
            }
            (!known.clear_at(bit) || held.clear_at(bit)) && (!known.set_at(bit) || held.set_at(bit))
        };
        if !(0..width).all(agrees) {
            return None;
        }
        let rewrite = if source_width == width {
            SsaOp::Copy { dest, src: source }
        } else {
            SsaOp::IntConv {
                dest,
                operand: source,
                target: self.ssa.variable(dest)?.var_type().clone(),
                overflow_check: false,
                unsigned: true,
            }
        };
        (rewrite != *op).then_some(rewrite)
    }

    /// Returns the value whose consecutive bits `value` copies, and the position
    /// in it of the bit `value` starts with: every bit `i` of `value` is bit
    /// `offset + i` of that value.
    ///
    /// The halves of a double-width value read apart are such fields: the low
    /// half its truncation (offset `0`), the high half the truncation of its
    /// shift right by a half's width (offset `w`).
    ///
    /// # Arguments
    ///
    /// * `value` - The candidate field.
    ///
    /// # Returns
    ///
    /// The value and the offset, or `None` when a bit of `value` is known, mixed,
    /// or copied from another value or another position.
    #[must_use]
    pub fn copied_field(&self, value: SsaVarId) -> Option<(SsaVarId, u32)> {
        let op = self.ssa.get_definition(value)?;
        let mut reader = Reads::new(self.ssa, self.constants);
        let width = reader.width(value)?;
        let origins = reader.result_origins(op, width, MOST_LINKS_CROSSED)?;
        if origins.len() != usize::try_from(width).ok()? {
            return None;
        }
        let mut field: Option<(SsaVarId, u32)> = None;
        for (bit, origin) in (0..width).zip(origins) {
            let BitFrom::Copy { source, bit: from } = origin else {
                return None;
            };
            let at = (source, from.checked_sub(bit)?);
            match field {
                None => field = Some(at),
                Some(found) if found == at => {}
                Some(_) => return None,
            }
        }
        field
    }

    /// Returns the operand of `op`, an `and` with a constant mask, when the operand
    /// has no bit the mask clears that it can have set.
    ///
    /// # Arguments
    ///
    /// * `op` - The candidate operation.
    ///
    /// # Returns
    ///
    /// The variable `op` may be replaced by a copy of, or `None` when `op` is not
    /// an `and` with a constant mask, or its operand can have a bit outside it.
    #[must_use]
    pub fn redundant_mask(&self, op: &SsaOp<T>) -> Option<SsaVarId> {
        let SsaOp::And { left, right, .. } = op else {
            return None;
        };
        let mask_of = |var: SsaVarId| self.constants.get(&var).and_then(ConstValue::as_u64_bits);
        let (operand, mask) = match (mask_of(*right), mask_of(*left)) {
            (Some(mask), _) => (*left, mask),
            (None, Some(mask)) => (*right, mask),
            (None, None) => return None,
        };
        let bits = self.known_bits(operand)?;
        (bits.possibly_set() & !mask == 0).then_some(operand)
    }

    /// Returns `op`, an `and` with a constant mask of a value or of that value
    /// shifted right, rewritten to mask the one term of the value that holds the
    /// bits the mask keeps, when every other term of it is known clear there: the
    /// field a write put into a container, read back.
    ///
    /// The value is an `or` of terms, nested or not. Masked as it is, `(a | b) &
    /// m` is `b & m` where `a` has no bit of `m` set; shifted right by `k`
    /// first, `((a | (b << k)) >> k) & m` is `b & m` where `a` has no bit of `m
    /// << k` set and `m` fits below the bits the left shift dropped.
    ///
    /// # Arguments
    ///
    /// * `op` - The candidate operation.
    ///
    /// # Returns
    ///
    /// The `and` of the term and the mask, or `None` when `op` is no such `and`,
    /// more than one term can hold a kept bit, or nothing is dropped.
    #[must_use]
    pub fn demanded_field(&self, op: &SsaOp<T>) -> Option<SsaOp<T>> {
        /// The most `or` definitions a value's terms are read through.
        const MOST_TERMS: usize = 16;
        let SsaOp::And {
            dest,
            left,
            right,
            flags,
        } = op
        else {
            return None;
        };
        let constant = |var: SsaVarId| self.constants.get(&var).and_then(ConstValue::as_u64_bits);
        let (operand, mask_var, mask) = match (constant(*right), constant(*left)) {
            (Some(mask), _) => (*left, *right, mask),
            (None, Some(mask)) => (*right, *left, mask),
            (None, None) => return None,
        };
        let width = |var: SsaVarId| self.ssa.bit_width(var);
        let (container, shift) = match self.ssa.get_definition(operand) {
            Some(SsaOp::Shr {
                value,
                amount,
                unsigned: true,
                ..
            }) => (*value, u32::try_from(constant(*amount)?).ok()?),
            _ => (operand, 0),
        };
        let bits = width(container)?;
        if shift >= bits || width(*dest)? != bits {
            return None;
        }
        let demanded = mask.checked_shl(shift)?;
        // The terms of the container's `or`s, read down to the first that is not
        // one.
        let mut terms = Vec::new();
        let mut pending = vec![container];
        while let Some(term) = pending.pop() {
            match self.ssa.get_definition(term) {
                Some(SsaOp::Or { left, right, .. })
                    if terms.len().saturating_add(pending.len()) < MOST_TERMS =>
                {
                    pending.extend([*left, *right]);
                }
                _ => terms.push(term),
            }
        }
        let mut holding = terms.iter().copied().filter(|&term| {
            self.known_bits(term)
                .is_none_or(|known| known.possibly_set() & demanded != 0)
        });
        let term = holding.next()?;
        if holding.next().is_some() {
            return None;
        }
        let source = if shift == 0 {
            if terms.len() < 2 {
                return None;
            }
            term
        } else {
            let Some(SsaOp::Shl { value, amount, .. }) = self.ssa.get_definition(term) else {
                return None;
            };
            // What the left shift dropped off the top is gone, so the mask must
            // keep nothing above what survives it.
            let kept = BitWidth::new(bits.saturating_sub(shift)).mask();
            if u32::try_from(constant(*amount)?).ok()? != shift || mask & !kept != 0 {
                return None;
            }
            *value
        };
        (width(source)? == bits).then(|| SsaOp::And {
            dest: *dest,
            left: source,
            right: mask_var,
            flags: *flags,
        })
    }
}

/// One query's reads of the values' bits, each remembered.
struct Reads<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Known constant values.
    constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
    /// The bits read so far, by value.
    known: BTreeMap<SsaVarId, Option<KnownBits>>,
    /// Where each bit of a value comes from, low bit first, read so far.
    origins: BTreeMap<SsaVarId, Option<Vec<BitFrom>>>,
}

impl<'a, T: Target> Reads<'a, T> {
    /// A reader of `ssa`'s values, with `constants` known.
    const fn new(
        ssa: &'a SsaFunction<T>,
        constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
    ) -> Self {
        Self {
            ssa,
            constants,
            known: BTreeMap::new(),
            origins: BTreeMap::new(),
        }
    }

    /// The width of `value`'s integer type, at most 64 bits.
    fn width(&self, value: SsaVarId) -> Option<u32> {
        let ty = self.ssa.variable(value)?.var_type();
        if !T::is_integer(ty) {
            return None;
        }
        T::bit_width(ty).filter(|width| (1..=64).contains(width))
    }

    /// The bits known of `value`, reading at most `depth` definitions down.
    fn bits(&mut self, value: SsaVarId, depth: usize) -> Option<KnownBits> {
        if let Some(known) = self.known.get(&value) {
            return *known;
        }
        let width = self.width(value)?;
        let known = match (self.constants.get(&value), self.ssa.get_definition(value)) {
            (Some(constant), _) => constant
                .as_u64_bits()
                .map(|bits| KnownBits::constant(bits, width)),
            (None, Some(op)) if depth > 0 => Some(
                self.result(op, width, depth)
                    .unwrap_or(KnownBits::unknown(width)),
            ),
            _ => Some(KnownBits::unknown(width)),
        };
        self.known.insert(value, known);
        known
    }

    /// The bits known of `op`'s result, `width` bits wide, reading its operands
    /// at most `depth` definitions down.
    fn result(&mut self, op: &SsaOp<T>, width: u32, depth: usize) -> Option<KnownBits> {
        let below = depth.saturating_sub(1);
        Some(match op {
            SsaOp::Const { value, .. } => KnownBits::constant(value.as_u64_bits()?, width),
            SsaOp::Copy { src, .. } => self.bits(*src, below)?,
            SsaOp::And { left, right, .. } => {
                let (left, right) = (self.bits(*left, below)?, self.bits(*right, below)?);
                left.combined(right, width, IntegerBits::and)?
            }
            SsaOp::Or { left, right, .. } => {
                let (left, right) = (self.bits(*left, below)?, self.bits(*right, below)?);
                left.combined(right, width, IntegerBits::or)?
            }
            SsaOp::Xor { left, right, .. } => {
                let (left, right) = (self.bits(*left, below)?, self.bits(*right, below)?);
                left.combined(right, width, IntegerBits::xor)?
            }
            SsaOp::Not { operand, .. } => {
                let operand = self.bits(*operand, below)?;
                if operand.width != width {
                    return None;
                }
                KnownBits::narrow(operand.wide()?.inverted())?
            }
            SsaOp::Shl { value, amount, .. } => {
                let amount = self.amount(*amount, width, below)?;
                self.bits(*value, below)?.shifted_left(amount)?
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned,
                ..
            } => {
                let amount = self.amount(*amount, width, below)?;
                self.bits(*value, below)?.shifted_right(amount, *unsigned)?
            }
            SsaOp::IntConv {
                operand,
                overflow_check: false,
                unsigned,
                ..
            } => self.bits(*operand, below)?.converted(width, *unsigned)?,
            // Either value: a bit is known where both know it alike.
            SsaOp::Select {
                true_val,
                false_val,
                ..
            } => {
                let (chosen, other) = (self.bits(*true_val, below)?, self.bits(*false_val, below)?);
                chosen.combined(other, width, IntegerBits::join)?
            }
            _ => return None,
        })
    }

    /// Where each bit of `value` comes from, low bit first, reading at most
    /// `depth` definitions down: a known bit is [`BitFrom::Known`], and a value
    /// whose definition moves no bit it can name is its own bits' origin.
    fn origins(&mut self, value: SsaVarId, depth: usize) -> Option<Vec<BitFrom>> {
        if let Some(origins) = self.origins.get(&value) {
            return origins.clone();
        }
        let width = self.width(value)?;
        let known = self.bits(value, depth)?;
        let traced = match self.ssa.get_definition(value) {
            Some(op) if depth > 0 && !self.constants.contains_key(&value) => {
                self.result_origins(op, width, depth)
            }
            _ => None,
        };
        let mut origins = traced.unwrap_or_else(|| {
            (0..width)
                .map(|bit| BitFrom::Copy { source: value, bit })
                .collect()
        });
        for (bit, origin) in (0..width).zip(origins.iter_mut()) {
            if known.clear_at(bit) || known.set_at(bit) {
                *origin = BitFrom::Known;
            }
        }
        self.origins.insert(value, Some(origins.clone()));
        Some(origins)
    }

    /// Where each bit of `op`'s result, `width` bits wide, comes from: through
    /// a copy, a conversion, a shift by a constant, and an `and`, `or` or `xor`
    /// whose other operand fixes the bit; `None` for any other operation.
    fn result_origins(&mut self, op: &SsaOp<T>, width: u32, depth: usize) -> Option<Vec<BitFrom>> {
        let below = depth.saturating_sub(1);
        let at = |origins: &[BitFrom], bit: u32| {
            usize::try_from(bit)
                .ok()
                .and_then(|bit| origins.get(bit).copied())
        };
        Some(match op {
            SsaOp::Copy { src, .. } => self.origins(*src, below)?,
            SsaOp::IntConv {
                operand,
                overflow_check: false,
                unsigned,
                ..
            } => {
                let inner = self.origins(*operand, below)?;
                // Bits past the operand are clear, or copies of its sign.
                let extension = if *unsigned {
                    BitFrom::Known
                } else {
                    inner.last().copied().unwrap_or(BitFrom::Mixed)
                };
                (0..width)
                    .map(|bit| at(&inner, bit).unwrap_or(extension))
                    .collect()
            }
            SsaOp::Shl { value, amount, .. } => {
                let shift = self.amount(*amount, width, below)?;
                let inner = self.origins(*value, below)?;
                (0..width)
                    .map(|bit| match bit.checked_sub(shift) {
                        Some(from) => at(&inner, from).unwrap_or(BitFrom::Mixed),
                        None => BitFrom::Known,
                    })
                    .collect()
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned: true,
                ..
            } => {
                let shift = self.amount(*amount, width, below)?;
                let inner = self.origins(*value, below)?;
                (0..width)
                    .map(|bit| {
                        bit.checked_add(shift)
                            .and_then(|from| at(&inner, from))
                            .unwrap_or(BitFrom::Known)
                    })
                    .collect()
            }
            SsaOp::And { left, right, .. }
            | SsaOp::Or { left, right, .. }
            | SsaOp::Xor { left, right, .. } => {
                let (left_from, right_from) =
                    (self.origins(*left, below)?, self.origins(*right, below)?);
                let (left_known, right_known) =
                    (self.bits(*left, below)?, self.bits(*right, below)?);
                let and = matches!(op, SsaOp::And { .. });
                (0..width)
                    .map(|bit| {
                        let (l, r) = (
                            at(&left_from, bit).unwrap_or(BitFrom::Mixed),
                            at(&right_from, bit).unwrap_or(BitFrom::Mixed),
                        );
                        // A clear bit passes the other through an `or` or `xor`
                        // and clears an `and`; a set one passes it through an
                        // `and`.
                        match (
                            left_known.clear_at(bit),
                            right_known.clear_at(bit),
                            left_known.set_at(bit),
                            right_known.set_at(bit),
                        ) {
                            (true, _, _, _) | (_, true, _, _) if and => BitFrom::Known,
                            (true, _, _, _) => r,
                            (_, true, _, _) => l,
                            (_, _, true, _) if and => r,
                            (_, _, _, true) if and => l,
                            _ => BitFrom::Mixed,
                        }
                    })
                    .collect()
            }
            _ => return None,
        })
    }

    /// The amount `amount` names, when it is known and shifts by less than
    /// `width`.
    fn amount(&mut self, amount: SsaVarId, width: u32, depth: usize) -> Option<u32> {
        let amount = u32::try_from(self.bits(amount, depth)?.value()?).ok()?;
        (amount < width).then_some(amount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{SsaDefSpec, SsaFunctionBuilder},
        testing::{MockTarget, MockType, mock_i32},
    };

    /// The constants of `ssa`, by the value each defines.
    fn constants_of(ssa: &SsaFunction<MockTarget>) -> BTreeMap<SsaVarId, ConstValue<MockTarget>> {
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        constants
    }

    /// Builds `(conv (r & inner)) & outer` for an argument `r` nothing is
    /// known of, with the conversion to `to`, returning the function, the
    /// outer `and`, the conversion's result, and the constants.
    fn fixture(
        inner: i32,
        to: MockType,
        unsigned: bool,
        outer: i32,
    ) -> (
        SsaFunction<MockTarget>,
        SsaOp<MockTarget>,
        SsaVarId,
        BTreeMap<SsaVarId, ConstValue<MockTarget>>,
    ) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 1);
        let mut found = None;
        builder
            .in_block(0, |block| {
                let r = block.emit_def(SsaDefSpec::argument(0, MockType::I32), |dest| {
                    SsaOp::LoadArg { dest, arg_index: 0 }
                })?;
                let inner_mask = block.const_i32(mock_i32(), inner)?;
                let masked = block.and(mock_i32(), r, inner_mask)?;
                let converted = block.conv(SsaDefSpec::tmp(to), masked, to, false, unsigned)?;
                let outer_mask = block.const_i32(mock_i32(), outer)?;
                let result = block.and(SsaDefSpec::tmp(to), converted, outer_mask)?;
                found = Some((
                    SsaOp::And {
                        dest: result,
                        left: converted,
                        right: outer_mask,
                        flags: None,
                    },
                    converted,
                ));
                block.ret(Some(result))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let constants = constants_of(&ssa);
        let (op, converted) = found.expect("the outer and");
        (ssa, op, converted, constants)
    }

    /// **A value read only through its low bits demands only those.**
    /// `digit_value(char c)` gets `c` in a 64-bit register and reads it as
    /// `(uint8_t)((uint32_t)c - 0x30)` and `(int32_t)(int8_t)c - 0x30`: every
    /// reader keeps the low byte, through the subtractions and the
    /// truncations, so the register's value demands eight bits. The
    /// controls: a right shift, a value returned whole, and an addition whose
    /// flags are produced each read the whole value.
    #[test]
    fn a_value_read_only_through_its_low_bits_demands_only_those() {
        /// How the argument is read besides its bytes.
        #[derive(Clone, Copy)]
        enum Also {
            Nothing,
            ShiftedRight,
            Returned,
            AddedWithFlags,
        }
        let demanded = |also: Also| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            let mut argument = None;
            builder
                .in_block(0, |block| {
                    let c = block.emit_def(SsaDefSpec::argument(0, MockType::I64), |dest| {
                        SsaOp::LoadArg { dest, arg_index: 0 }
                    })?;
                    argument = Some(c);
                    let low = block.conv(
                        SsaDefSpec::tmp(MockType::U32),
                        c,
                        MockType::U32,
                        false,
                        true,
                    )?;
                    let zero = block.const_i32(mock_i32(), 0x30)?;
                    let less = block.sub(SsaDefSpec::tmp(MockType::U32), low, zero)?;
                    let byte = block.conv(
                        SsaDefSpec::tmp(MockType::U8),
                        less,
                        MockType::U8,
                        false,
                        true,
                    )?;
                    let nine =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(9))?;
                    block.cgt(SsaDefSpec::tmp(MockType::I1), byte, nine, true)?;
                    let char_ =
                        block.conv(SsaDefSpec::tmp(MockType::I8), c, MockType::I8, false, true)?;
                    let int = block.conv(
                        SsaDefSpec::tmp(MockType::I32),
                        char_,
                        MockType::I32,
                        false,
                        false,
                    )?;
                    let digit = block.sub(SsaDefSpec::tmp(MockType::I32), int, zero)?;
                    let result = match also {
                        Also::Nothing => digit,
                        Also::ShiftedRight => {
                            let one = block.const_i64(SsaDefSpec::tmp(MockType::I64), 1)?;
                            let high = block.shr(SsaDefSpec::tmp(MockType::I64), c, one, true)?;
                            let cut = block.conv(
                                SsaDefSpec::tmp(MockType::I32),
                                high,
                                MockType::I32,
                                false,
                                true,
                            )?;
                            block.add(SsaDefSpec::tmp(MockType::I32), digit, cut)?
                        }
                        Also::Returned => {
                            block.ret(Some(c))?;
                            return Ok(());
                        }
                        Also::AddedWithFlags => {
                            let (sum, _) = block.add_with_flags(
                                SsaDefSpec::tmp(MockType::I64),
                                SsaDefSpec::tmp(MockType::I64),
                                c,
                                c,
                            )?;
                            let cut = block.conv(
                                SsaDefSpec::tmp(MockType::I32),
                                sum,
                                MockType::I32,
                                false,
                                true,
                            )?;
                            block.add(SsaDefSpec::tmp(MockType::I32), digit, cut)?
                        }
                    };
                    block.ret(Some(result))?;
                    Ok(())
                })
                .expect("build block");
            let ssa = builder.finish().expect("finish");
            DemandedBits::of(&ssa).of_value(argument.expect("the argument"))
        };
        assert_eq!(demanded(Also::Nothing), 8, "the low byte");
        assert_eq!(
            demanded(Also::ShiftedRight),
            64,
            "a right shift reads the high bits"
        );
        assert_eq!(demanded(Also::Returned), 64, "returned whole");
        assert_eq!(
            demanded(Also::AddedWithFlags),
            64,
            "a carry reads the high bits"
        );
    }

    /// **A bit an operation fixes keeps it from being a copy.** `zext(r == 0)
    /// | -2` sets every bit above the first, which the extended truth value
    /// has clear: it is no copy of it (RISC-V `seqz a1, a1; ori a1, a1, -2`).
    /// The control: `| 0` fixes nothing, and is the extension.
    #[test]
    fn a_bit_the_operation_sets_is_no_copy() {
        let built = |mask: i64| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            let mut found = None;
            builder
                .in_block(0, |block| {
                    let r = block.emit_def(SsaDefSpec::argument(0, MockType::U8), |dest| {
                        SsaOp::LoadArg { dest, arg_index: 0 }
                    })?;
                    let zero =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                    let truth = block.ceq(SsaDefSpec::tmp(MockType::I1), r, zero)?;
                    let long = || SsaDefSpec::tmp(MockType::I64);
                    let wide = block.conv(long(), truth, MockType::I64, false, true)?;
                    let constant = block.const_value(long(), ConstValue::I64(mask))?;
                    let result = block.or(long(), wide, constant)?;
                    found = Some(SsaOp::Or {
                        dest: result,
                        left: wide,
                        right: constant,
                        flags: None,
                    });
                    block.ret(Some(result))?;
                    Ok(())
                })
                .expect("build block");
            let ssa = builder.finish().expect("finish");
            let constants = constants_of(&ssa);
            (ssa, found.expect("the or"), constants)
        };
        let (ssa, op, constants) = built(-2);
        assert_eq!(Reader::new(&ssa, &constants).copied_value(&op), None);
        let (ssa, op, constants) = built(0);
        assert!(
            matches!(
                Reader::new(&ssa, &constants).copied_value(&op),
                Some(SsaOp::IntConv { .. })
            ),
            "the control: or 0 is the extension"
        );
    }

    /// A shift count masked, narrowed to a byte and masked again by the same
    /// bound is the narrowed count: the second mask keeps every bit it can
    /// have.
    #[test]
    fn a_mask_over_a_smaller_mask_is_its_operand() {
        let (ssa, op, converted, constants) = fixture(31, MockType::U8, true, 31);
        assert_eq!(
            Reader::new(&ssa, &constants).redundant_mask(&op),
            Some(converted)
        );
    }

    /// The control: a mask narrower than the one inside it clears bits the
    /// operand can hold, and stays.
    #[test]
    fn a_mask_narrower_than_its_operand_stays() {
        let (ssa, op, _, constants) = fixture(31, MockType::U8, true, 15);
        assert_eq!(Reader::new(&ssa, &constants).redundant_mask(&op), None);
    }

    /// A sign-extension of a value whose sign bit the inner mask can set fills
    /// the high bits with it, so a mask of the low bits is not redundant.
    #[test]
    fn a_sign_extension_that_can_fill_ones_keeps_the_mask() {
        let (ssa, op, _, constants) = fixture(-1, MockType::I64, false, 0xff);
        assert_eq!(Reader::new(&ssa, &constants).redundant_mask(&op), None);
        let (ssa, op, converted, constants) = fixture(0x7f, MockType::I64, false, 0xff);
        assert_eq!(
            Reader::new(&ssa, &constants).redundant_mask(&op),
            Some(converted)
        );
    }

    /// Builds `select(c, a, b) & mask` over bytes, `c` a comparison of an
    /// argument nothing is known of; returns the function, the `and` and the
    /// select's result.
    fn masked_select(
        a: u8,
        b: u8,
        mask: u8,
    ) -> (
        SsaFunction<MockTarget>,
        SsaOp<MockTarget>,
        SsaVarId,
        BTreeMap<SsaVarId, ConstValue<MockTarget>>,
    ) {
        let byte = || SsaDefSpec::tmp(MockType::U8);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let mut found = None;
        builder
            .in_block(0, |block| {
                let r = block.emit_def(SsaDefSpec::argument(0, MockType::U8), |dest| {
                    SsaOp::LoadArg { dest, arg_index: 0 }
                })?;
                let zero = block.const_value(byte(), ConstValue::U8(0))?;
                let condition = block.ceq(SsaDefSpec::tmp(MockType::I1), r, zero)?;
                let a = block.const_value(byte(), ConstValue::U8(a))?;
                let b = block.const_value(byte(), ConstValue::U8(b))?;
                let chosen = block.select(byte(), condition, a, b)?;
                let mask = block.const_value(byte(), ConstValue::U8(mask))?;
                let result = block.and(byte(), chosen, mask)?;
                found = Some((
                    SsaOp::And {
                        dest: result,
                        left: chosen,
                        right: mask,
                        flags: None,
                    },
                    chosen,
                ));
                block.ret(Some(result))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let constants = constants_of(&ssa);
        let (op, chosen) = found.expect("the and");
        (ssa, op, chosen, constants)
    }

    /// A select of one or zero has no bit above the lowest, so a mask of that
    /// bit is its operand; a select whose other value has a bit outside the
    /// mask keeps it.
    #[test]
    fn a_select_is_known_where_both_values_agree() {
        let (ssa, op, chosen, constants) = masked_select(1, 0, 1);
        assert_eq!(
            Reader::new(&ssa, &constants).redundant_mask(&op),
            Some(chosen)
        );
        let (ssa, op, _, constants) = masked_select(1, 2, 1);
        assert_eq!(Reader::new(&ssa, &constants).redundant_mask(&op), None);
        // Both values set the bit, so it is known set.
        let (ssa, _, chosen, constants) = masked_select(3, 1, 1);
        let known = Reader::new(&ssa, &constants)
            .known_bits(chosen)
            .expect("an integer");
        assert_eq!(known.possibly_set(), 3);
        assert_eq!(known.ones, 1);
    }

    /// The shapes a flag written into a status register takes when it is read
    /// back: `select(r == 0, flag, 0)` as a byte, written to bit 27 of a word
    /// whose bits `clear` clears, then read back as `shape` builds it.
    #[derive(Clone, Copy)]
    enum ReadBack {
        /// `((c & clear) | (ext(q) << 27)) >> 27`, at 32 bits.
        Word,
        /// `(ext64((c & clear) | (ext(q) << 27)) >> 27) & 1`.
        Extended,
        /// `(ext(trunc8(((c & clear) | (ext(q) << 27)) >> 24)) << 24) >> 27`.
        ThroughAByte,
    }

    /// Builds the flag written and read back as `shape`; returns the function,
    /// the final operation and the flag.
    fn flag_read_back(
        flag: u8,
        clear: u32,
        shape: ReadBack,
    ) -> (
        SsaFunction<MockTarget>,
        SsaOp<MockTarget>,
        SsaVarId,
        BTreeMap<SsaVarId, ConstValue<MockTarget>>,
    ) {
        let word = || SsaDefSpec::tmp(MockType::U32);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut found = None;
        builder
            .in_block(0, |block| {
                let container = block.emit_def(SsaDefSpec::argument(0, MockType::U32), |dest| {
                    SsaOp::LoadArg { dest, arg_index: 0 }
                })?;
                let r = block.emit_def(SsaDefSpec::argument(1, MockType::U8), |dest| {
                    SsaOp::LoadArg { dest, arg_index: 1 }
                })?;
                let zero = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                let condition = block.ceq(SsaDefSpec::tmp(MockType::I1), r, zero)?;
                let set = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(flag))?;
                let q = block.select(SsaDefSpec::tmp(MockType::U8), condition, set, zero)?;
                let mask = block.const_value(word(), ConstValue::U32(clear))?;
                let kept = block.and(word(), container, mask)?;
                let wide = block.conv(word(), q, MockType::U32, false, true)?;
                let at = block.const_value(word(), ConstValue::U32(27))?;
                let placed = block.shl(word(), wide, at)?;
                let written = block.or(word(), kept, placed)?;
                let (op, result) = match shape {
                    ReadBack::Word => {
                        let read = block.shr(word(), written, at, true)?;
                        (
                            SsaOp::Shr {
                                dest: read,
                                value: written,
                                amount: at,
                                unsigned: true,
                                flags: None,
                            },
                            read,
                        )
                    }
                    ReadBack::Extended => {
                        let long = || SsaDefSpec::tmp(MockType::U64);
                        let extended = block.conv(long(), written, MockType::U64, false, true)?;
                        let at = block.const_value(long(), ConstValue::U64(27))?;
                        let down = block.shr(long(), extended, at, true)?;
                        let one = block.const_value(long(), ConstValue::U64(1))?;
                        let read = block.and(long(), down, one)?;
                        (
                            SsaOp::And {
                                dest: read,
                                left: down,
                                right: one,
                                flags: None,
                            },
                            read,
                        )
                    }
                    ReadBack::ThroughAByte => {
                        let field_at = block.const_value(word(), ConstValue::U32(24))?;
                        let down = block.shr(word(), written, field_at, true)?;
                        let byte = block.conv(
                            SsaDefSpec::tmp(MockType::U8),
                            down,
                            MockType::U8,
                            false,
                            true,
                        )?;
                        let back = block.conv(word(), byte, MockType::U32, false, true)?;
                        let up = block.shl(word(), back, field_at)?;
                        let read = block.shr(word(), up, at, true)?;
                        (
                            SsaOp::Shr {
                                dest: read,
                                value: up,
                                amount: at,
                                unsigned: true,
                                flags: None,
                            },
                            read,
                        )
                    }
                };
                found = Some((op, q));
                block.ret(Some(result))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let constants = constants_of(&ssa);
        let (op, q) = found.expect("the read");
        (ssa, op, q, constants)
    }

    /// **A flag written into a status register and read back is the flag**,
    /// through the extension, truncation, shifts and mask between, whatever
    /// else the register held below it: the read is the flag zero-extended to
    /// its width. The register's bits above the flag are cleared (`0x07ffffff`)
    /// or masked off by the read (`& 1`). The controls: a register whose bit 27
    /// is not cleared, and a value of `0x20`, whose bit 5 the write to bit 27
    /// shifts out of the word.
    #[test]
    fn a_flag_read_back_out_of_a_register_is_the_flag() {
        let clear = 0x07ff_ffff;
        for shape in [ReadBack::Word, ReadBack::Extended, ReadBack::ThroughAByte] {
            let (ssa, op, q, constants) = flag_read_back(1, clear, shape);
            let rewrite = Reader::new(&ssa, &constants).copied_value(&op);
            let Some(SsaOp::IntConv {
                operand,
                unsigned: true,
                ..
            }) = rewrite
            else {
                panic!("the read is the flag extended: {rewrite:?}");
            };
            assert_eq!(operand, q);
            let (ssa, op, _, constants) = flag_read_back(1, u32::MAX, shape);
            assert_eq!(
                Reader::new(&ssa, &constants).copied_value(&op),
                None,
                "control: bit 27 kept"
            );
            let (ssa, op, _, constants) = flag_read_back(0x20, clear, shape);
            assert_eq!(
                Reader::new(&ssa, &constants).copied_value(&op),
                None,
                "control: a bit dropped"
            );
        }
    }

    /// Builds a field read back after a write of it into a byte nothing is
    /// known of, `((c & 0x1f) | ((v & 7) << 5)) >> 5 & 7` with `c` an argument
    /// and `v` a constant (`Some`) or a second argument (`None`), as a lift
    /// spells RISC-V's `fsrmi`/`fsrm` followed by `frrm`; returns the function
    /// and the final `and`.
    fn field_round_trip(value: Option<u8>) -> (SsaFunction<MockTarget>, SsaOp<MockTarget>) {
        let byte = || SsaDefSpec::tmp(MockType::U8);
        let constant = |bits: u8| ConstValue::<MockTarget>::U8(bits);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut found = None;
        builder
            .in_block(0, |block| {
                let container = block.emit_def(SsaDefSpec::argument(0, MockType::U8), |dest| {
                    SsaOp::LoadArg { dest, arg_index: 0 }
                })?;
                let written = match value {
                    Some(bits) => block.const_value(byte(), constant(bits))?,
                    None => block.emit_def(SsaDefSpec::argument(1, MockType::U8), |dest| {
                        SsaOp::LoadArg { dest, arg_index: 1 }
                    })?,
                };
                let low = block.const_value(byte(), constant(0x1f))?;
                let kept = block.and(byte(), container, low)?;
                let field = block.const_value(byte(), constant(7))?;
                let narrowed = block.and(byte(), written, field)?;
                let at = block.const_value(byte(), constant(5))?;
                let placed = block.shl(byte(), narrowed, at)?;
                let inserted = block.or(byte(), kept, placed)?;
                let down = block.shr(byte(), inserted, at, true)?;
                let read = block.and(byte(), down, field)?;
                found = Some(SsaOp::And {
                    dest: read,
                    left: down,
                    right: field,
                    flags: None,
                });
                block.ret(Some(read))?;
                Ok(())
            })
            .expect("build block");
        (builder.finish().expect("finish"), found.expect("the read"))
    }

    /// **A field read back after a write of it is the value written,
    /// whatever its container held.** Every bit the read keeps came from the
    /// write, so the read is the constant 3 though the container is unknown.
    /// The control: written from an unknown value, the read stays.
    #[test]
    fn a_field_read_back_after_a_write_is_the_value_written() {
        let (ssa, read) = field_round_trip(Some(3));
        assert_eq!(
            Reader::new(&ssa, &constants_of(&ssa)).known_constant(&read),
            Some(ConstValue::U8(3))
        );
        let (ssa, read) = field_round_trip(None);
        assert_eq!(
            Reader::new(&ssa, &constants_of(&ssa)).known_constant(&read),
            None
        );
    }

    /// Builds `((c & clear) | ((v & field) << at)) >> at & field` over two
    /// arguments of a byte nothing is known of, without the shifts where `at`
    /// is 0; returns the function, the final `and` and `v & field`.
    fn container_field(
        clear: u8,
        at: u8,
        field: u8,
    ) -> (SsaFunction<MockTarget>, SsaOp<MockTarget>, SsaVarId) {
        let byte = || SsaDefSpec::tmp(MockType::U8);
        let constant = |bits: u8| ConstValue::<MockTarget>::U8(bits);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut found = None;
        builder
            .in_block(0, |block| {
                let container = block.emit_def(SsaDefSpec::argument(0, MockType::U8), |dest| {
                    SsaOp::LoadArg { dest, arg_index: 0 }
                })?;
                let written = block.emit_def(SsaDefSpec::argument(1, MockType::U8), |dest| {
                    SsaOp::LoadArg { dest, arg_index: 1 }
                })?;
                let clear = block.const_value(byte(), constant(clear))?;
                let kept = block.and(byte(), container, clear)?;
                let mask = block.const_value(byte(), constant(field))?;
                let narrowed = block.and(byte(), written, mask)?;
                let read_from = if at == 0 {
                    block.or(byte(), kept, narrowed)?
                } else {
                    let amount = block.const_value(byte(), constant(at))?;
                    let shifted = block.shl(byte(), narrowed, amount)?;
                    let inserted = block.or(byte(), kept, shifted)?;
                    block.shr(byte(), inserted, amount, true)?
                };
                let read = block.and(byte(), read_from, mask)?;
                found = Some((
                    SsaOp::And {
                        dest: read,
                        left: read_from,
                        right: mask,
                        flags: None,
                    },
                    narrowed,
                ));
                block.ret(Some(read))?;
                Ok(())
            })
            .expect("build block");
        let (read, narrowed) = found.expect("the read");
        (builder.finish().expect("finish"), read, narrowed)
    }

    /// **A field read back is the value written into it, whatever its
    /// container held**, the value unknown: `((c & 0x1f) | ((v & 7) << 5)) >> 5
    /// & 7` is `(v & 7) & 7`, and with the field at bit 0, `((c & 0xf8) | (v &
    /// 7)) & 7` is too. The controls: a container whose field's bits are not
    /// cleared (`c & 0x3f` keeps bit 5), and a mask reaching above what the
    /// left shift kept (`<< 5` then `& 0xf` on a byte).
    #[test]
    fn a_field_read_back_is_the_value_written_into_it() {
        let rewritten = |clear, at, field| {
            let (ssa, read, narrowed) = container_field(clear, at, field);
            let rewrite = Reader::new(&ssa, &constants_of(&ssa)).demanded_field(&read);
            let SsaOp::And { dest, right, .. } = read else {
                unreachable!("the read is an and")
            };
            (
                rewrite,
                SsaOp::And {
                    dest,
                    left: narrowed,
                    right,
                    flags: None,
                },
            )
        };
        let (rewrite, expected) = rewritten(0x1f, 5, 7);
        assert_eq!(rewrite, Some(expected), "shifted into bits 5 to 7");
        let (rewrite, expected) = rewritten(0xf8, 0, 7);
        assert_eq!(rewrite, Some(expected), "at bit 0");
        assert_eq!(rewritten(0x3f, 5, 7).0, None, "control: bit 5 not cleared");
        assert_eq!(
            rewritten(0x1f, 5, 0xf).0,
            None,
            "control: the mask reaches past the byte"
        );
    }

    /// The bits each operation keeps, moves or fills: an `or`'s known ones,
    /// an `xor` of known bits, a `not`, a shift's cleared and sign-filled
    /// bits, and a conversion's cut and extension.
    #[test]
    fn each_operation_carries_the_bits_it_keeps() {
        let known = |zeros: u64, ones: u64, width: u32| KnownBits { zeros, ones, width };
        let byte = known(0x0f, 0x30, 8);
        assert_eq!(byte.possibly_set(), 0xf0);
        assert_eq!(byte.value(), None);
        assert_eq!(
            byte.shifted_left(4).expect("valid scalar geometry"),
            known(0xff, 0x00, 8)
        );
        assert_eq!(
            byte.shifted_right(4, true).expect("valid scalar geometry"),
            known(0xf0, 0x03, 8)
        );
        // The sign is unknown, so an arithmetic shift fills nothing known.
        assert_eq!(
            byte.shifted_right(4, false).expect("valid scalar geometry"),
            known(0x00, 0x03, 8)
        );
        let negative = known(0x00, 0x80, 8);
        assert_eq!(
            negative
                .shifted_right(4, false)
                .expect("valid scalar geometry"),
            known(0x00, 0xf8, 8)
        );
        assert_eq!(
            negative
                .converted(16, false)
                .expect("valid scalar geometry"),
            known(0x0000, 0xff80, 16)
        );
        assert_eq!(
            negative.converted(16, true).expect("valid scalar geometry"),
            known(0xff00, 0x0080, 16)
        );
        assert_eq!(
            known(0xff00, 0x00ff, 16)
                .converted(8, true)
                .expect("valid scalar geometry"),
            known(0x00, 0xff, 8)
        );
        assert_eq!(KnownBits::constant(0xa5, 8).value(), Some(0xa5));
        assert_eq!(KnownBits::constant(u64::MAX, 64).value(), Some(u64::MAX));
    }
}
