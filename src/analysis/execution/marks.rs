//! The bits of a value that came from memory no write had reached.
//!
//! Copying such bits is legal (a record's padding, a partly filled buffer),
//! and using them is the defect: the domain memcheck states with its
//! definedness bits. A value carries the never-written bits of what it was
//! computed from, propagated at bit precision where the operation states
//! which input bits reach which output bits (an `and` with a written zero
//! writes that bit, an addition carries upward from its lowest marked bit),
//! and to every bit of the result otherwise.
//!
//! A mark says the bit was certainly never written on the path that
//! computed it. Nothing marks a bit that may have been written, so a missing
//! rule loses a finding rather than inventing one.

use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    analysis::execution::{
        findings::{Finding, FindingKind, Findings, Witness},
        state::State,
        value::{Address, Value},
    },
    ir::{
        function::SsaFunction, integer::IntegerType, ops::SsaOp, value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// The never-written bits of one value, byte by byte from its least
/// significant: bit `i` of the value is bit `i % 8` of byte `i / 8`. Never
/// empty: a value with no such bit carries no marks at all.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Marks(Arc<[u8]>);

impl Marks {
    /// The marks `bytes` state, `None` where no bit is marked.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Option<Self> {
        bytes
            .iter()
            .any(|byte| *byte != 0)
            .then(|| Self(bytes.into()))
    }

    /// Every bit of a `bits`-wide value marked.
    #[must_use]
    pub fn all(bits: u32) -> Option<Self> {
        let bytes = usize::try_from(bits.div_ceil(8)).ok()?;
        let mut marks = vec![u8::MAX; bytes];
        if let Some(last) = marks.last_mut()
            && !bits.is_multiple_of(8)
        {
            *last = 1u8
                .checked_shl(bits % 8)
                .map_or(u8::MAX, |bit| bit.wrapping_sub(1));
        }
        Self::from_bytes(marks)
    }

    /// The marks of the low `bits` bits of `mask`.
    #[must_use]
    fn from_mask(mask: u128, bits: u32) -> Option<Self> {
        let mask = mask & IntegerType::new(bits, false).map(IntegerType::mask)?;
        let bytes = usize::try_from(bits.div_ceil(8)).ok()?;
        Self::from_bytes(mask.to_le_bytes().get(..bytes)?.to_vec())
    }

    /// The marks as a mask, for a value at most 128 bits wide.
    #[must_use]
    pub fn mask(&self) -> Option<u128> {
        if self.0.len() > 16 {
            return None;
        }
        let mut bytes = [0u8; 16];
        bytes.get_mut(..self.0.len())?.copy_from_slice(&self.0);
        Some(u128::from_le_bytes(bytes))
    }

    /// The marks byte by byte from the least significant.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// The bits marked on both paths: certainly never written on each.
    #[must_use]
    pub fn join(&self, other: &Self) -> Option<Self> {
        Self::from_bytes(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(left, right)| left & right)
                .collect(),
        )
    }

    /// The marks of a value stored in memory, byte by byte from the lowest
    /// address, `bytes` of it, in the target's byte order.
    #[must_use]
    pub fn in_memory(&self, bytes: usize, big_endian: bool) -> Vec<u8> {
        let mut stored: Vec<u8> = (0..bytes)
            .map(|index| self.0.get(index).copied().unwrap_or(0))
            .collect();
        if big_endian {
            stored.reverse();
        }
        stored
    }

    /// The marks of a value loaded from memory bytes `bits`, byte by byte
    /// from the lowest address, in the target's byte order.
    #[must_use]
    fn from_memory(mut bits: Vec<u8>, big_endian: bool) -> Option<Self> {
        if big_endian {
            bits.reverse();
        }
        Self::from_bytes(bits)
    }
}

/// How a use reads bits never written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UnwrittenUseKind {
    /// A branch decides on them.
    Branch,
    /// An address is formed from them.
    Address,
    /// A call is handed them: as an argument value, or as bytes a model
    /// reads through a pointer argument.
    Argument,
    /// The root returns them.
    Return,
}

/// The marks below a value's lowest marked bit cleared and every bit from it
/// up set: what an addition, a subtraction or a product carries, whose low
/// result bits depend only on the operands' bits at and below them.
fn upward(mask: u128) -> u128 {
    if mask == 0 {
        0
    } else {
        !(mask & mask.wrapping_neg()).wrapping_sub(1)
    }
}

/// What one operand of an operation holds: its marks, and the bits known
/// written zero and written one.
#[derive(Debug, Clone, Default)]
pub struct Operand {
    /// Its never-written bits.
    pub marks: Option<Marks>,
    /// Bits known to be zero.
    pub zeros: u128,
    /// Bits known to be one.
    pub ones: u128,
}

impl Operand {
    /// The operand holding `value` with `marks`, its known bits read from a
    /// constant or partial integer `bits` wide.
    #[must_use]
    pub fn of<T: Target>(value: &Value<T>, marks: Option<Marks>, bits: u32) -> Self {
        let (zeros, ones) = match value {
            Value::Integer(integer) => integer
                .bits()
                .map_or((0, 0), |known| (known.zeros(), known.ones())),
            Value::Const(constant) => constant_bits(constant, bits),
            _ => (0, 0),
        };
        Self { marks, zeros, ones }
    }

    /// Its marks as a mask, all of a `bits`-wide value where it is too wide
    /// for one.
    fn mask(&self, bits: u32) -> u128 {
        self.marks.as_ref().map_or(0, |marks| {
            marks.mask().unwrap_or_else(|| {
                IntegerType::new(bits, false)
                    .map(IntegerType::mask)
                    .unwrap_or(u128::MAX)
            })
        })
    }

    /// The bits known zero and written.
    fn written_zeros(&self, bits: u32) -> u128 {
        self.zeros & !self.mask(bits)
    }

    /// The bits known one and written.
    fn written_ones(&self, bits: u32) -> u128 {
        self.ones & !self.mask(bits)
    }
}

/// The known zero and one bits of an integer constant `bits` wide.
fn constant_bits<T: Target>(constant: &ConstValue<T>, bits: u32) -> (u128, u128) {
    let (Some(all), Some(value)) = (
        IntegerType::new(bits, false).map(IntegerType::mask),
        constant.as_u64_bits(),
    ) else {
        return (0, 0);
    };
    let value = u128::from(value) & all;
    (!value & all, value)
}

/// The marks of the result of `op`, `bits` wide, from its operands in the
/// order [`SsaOp::uses`] states them.
///
/// Bitwise operations, shifts by a written constant amount, additions,
/// subtractions and products, extensions, truncations and bitcasts carry the
/// bits their inputs reach; a comparison and every other operation mark the
/// whole result where any input bit is marked.
#[must_use]
pub fn result<T: Target>(op: &SsaOp<T>, operands: &[Operand], bits: u32) -> Option<Marks> {
    let marked = operands.iter().any(|operand| operand.marks.is_some());
    if !marked {
        return None;
    }
    let all = IntegerType::new(bits, false).map(IntegerType::mask);
    let first = operands.first();
    let second = operands.get(1);
    let precise = all.and_then(|all| {
        let mask = |operand: Option<&Operand>| operand.map_or(0, |operand| operand.mask(bits));
        let (a, b) = (mask(first), mask(second));
        let mask = match op {
            SsaOp::Copy { .. } | SsaOp::Bitcast { .. } => a,
            SsaOp::And { .. } => {
                let written_zero = first.map_or(0, |first| first.written_zeros(bits))
                    | second.map_or(0, |second| second.written_zeros(bits));
                (a | b) & !written_zero
            }
            SsaOp::Or { .. } => {
                let written_one = first.map_or(0, |first| first.written_ones(bits))
                    | second.map_or(0, |second| second.written_ones(bits));
                (a | b) & !written_one
            }
            SsaOp::Xor { .. } => a | b,
            SsaOp::Not { .. } => a,
            SsaOp::Neg { .. } => upward(a),
            SsaOp::Add { .. } | SsaOp::Sub { .. } | SsaOp::Mul { .. } => upward(a | b),
            SsaOp::Shl { .. } | SsaOp::Shr { .. } => {
                // A shift by a marked or unknown amount moves every bit.
                let amount = second
                    .filter(|amount| amount.marks.is_none() && amount.zeros | amount.ones == all)?;
                let shift = u32::try_from(amount.ones)
                    .ok()
                    .filter(|shift| *shift < bits)?;
                match op {
                    SsaOp::Shl { .. } => a << shift,
                    SsaOp::Shr { unsigned: true, .. } => a >> shift,
                    _ => {
                        let sign = a.checked_shr(bits.saturating_sub(1)).unwrap_or(0) & 1;
                        let shifted = a >> shift;
                        if sign == 0 {
                            shifted
                        } else {
                            shifted | (all & !(all >> shift))
                        }
                    }
                }
            }
            SsaOp::IntConv { unsigned, .. } => {
                // The operand's own width is not stated here: its marks
                // reach the low bits, and a sign extension carries its
                // marked sign bit, which the marks' last byte holds, up.
                let operand_bits = first
                    .and_then(|first| first.marks.as_ref())
                    .map_or(0, |marks| {
                        u32::try_from(marks.bytes().len().saturating_mul(8)).unwrap_or(128)
                    });
                let low = a & all;
                if *unsigned || operand_bits >= bits || operand_bits == 0 {
                    low
                } else {
                    let sign = a.checked_shr(operand_bits.saturating_sub(1)).unwrap_or(0) & 1;
                    if sign == 0 {
                        low
                    } else {
                        low | (all
                            & !IntegerType::new(operand_bits, false).map(IntegerType::mask)?)
                    }
                }
            }
            _ => return None,
        };
        Some(mask & all)
    });
    match precise {
        Some(mask) => Marks::from_mask(mask, bits),
        // Too wide for a mask, or no rule: a copy keeps its operand's marks
        // as they are, and anything else marks its whole result.
        None => match op {
            SsaOp::Copy { .. } | SsaOp::Bitcast { .. } => {
                first.and_then(|first| first.marks.clone())
            }
            _ => Marks::all(bits),
        },
    }
}

/// Carries marks through the instructions of one body, and records the uses
/// they make of marked bits.
pub struct Marker<'a, T: Target> {
    /// The body run.
    ssa: &'a SsaFunction<T>,
    /// Whether memory holds a value's most significant byte first; `None`
    /// where the target states no order, and no marks cross memory.
    big_endian: Option<bool>,
    /// Where the run records the uses it finds.
    findings: &'a Findings,
}

/// A marker is references and a flag, whatever the target.
impl<T: Target> Clone for Marker<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Target> Copy for Marker<'_, T> {}

impl<'a, T: Target> Marker<'a, T> {
    /// Carries marks through `ssa`'s instructions, records into `findings`, and
    /// reads memory in the byte order `big_endian` states.
    #[must_use]
    pub const fn new(
        ssa: &'a SsaFunction<T>,
        big_endian: Option<bool>,
        findings: &'a Findings,
    ) -> Self {
        Self {
            ssa,
            big_endian,
            findings,
        }
    }

    /// Records a use of `kind` at `site`, `(body, block, instruction)`, on
    /// the path `witness` says.
    pub fn record(
        &self,
        (body, block, instruction): (usize, usize, usize),
        kind: UnwrittenUseKind,
        witness: Witness,
    ) {
        self.findings.record(Finding {
            body,
            block,
            instruction,
            kind: FindingKind::Unwritten(kind),
            access: None,
            object: None,
            witness,
        });
    }

    /// States the marks of what `op` defined, once it has run at `site`,
    /// `(body, block, instruction)`, and records the uses it made: a load's
    /// result takes the never-written bits of the bytes it read (`loaded`:
    /// the span read and the value it defined), a store carries its value's
    /// marks into the bytes it wrote, and an address formed from marked bits
    /// is a use.
    pub fn after(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        loaded: Option<((Address, usize), SsaVarId)>,
        site: (usize, usize, usize),
    ) {
        let addressed = match op {
            SsaOp::LoadIndirect { addr, .. }
            | SsaOp::StoreIndirect { addr, .. }
            | SsaOp::VectorLoad { addr, .. }
            | SsaOp::VectorStore { addr, .. }
            | SsaOp::VectorBroadcastLoad { addr, .. } => Some(*addr),
            _ => None,
        };
        if addressed.is_some_and(|addr| state.marks(addr).is_some()) {
            self.record(
                site,
                UnwrittenUseKind::Address,
                state.memory().decisions().witness(),
            );
        }
        match op {
            SsaOp::LoadIndirect { dest, .. } | SsaOp::VectorLoad { dest, .. } => {
                let marks = loaded
                    .filter(|(_, loaded)| loaded == dest)
                    .zip(self.big_endian)
                    .and_then(|(((address, bytes), _), big_endian)| {
                        let bits = state.memory().unwritten_bits(address, bytes)?;
                        Marks::from_memory(bits, big_endian)
                    });
                state.set_marks(*dest, marks);
            }
            SsaOp::StoreIndirect { addr, value, .. } | SsaOp::VectorStore { addr, value, .. } => {
                if let (Some(marks), Value::Addr(address), Some(big_endian)) = (
                    state.marks(*value).cloned(),
                    state.value(*addr),
                    self.big_endian,
                ) {
                    let bits = marks.in_memory(marks.bytes().len(), big_endian);
                    state.memory_mut().carry_unwritten(address, &bits);
                }
            }
            _ => {
                let uses = op.uses();
                if uses.iter().all(|used| state.marks(*used).is_none()) {
                    return;
                }
                let dest = op.dest();
                if let Some(dest) = dest
                    && let Some(bits) = self.ssa.bit_width(dest)
                {
                    let operands: Vec<Operand> = uses
                        .iter()
                        .map(|used| {
                            Operand::of(&state.value(*used), state.marks(*used).cloned(), bits)
                        })
                        .collect();
                    state.set_marks(dest, result(op, &operands, bits));
                }
                // Any other result (the flags an arithmetic operation leaves)
                // is marked whole: which of its bits the marked ones reach is
                // not stated.
                for defined in op.defs().filter(|defined| Some(*defined) != dest) {
                    if let Some(bits) = self.ssa.bit_width(defined) {
                        state.set_marks(defined, Marks::all(bits));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::variable::SsaVarId,
        testing::{MockTarget, MockType},
    };

    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    fn marked(mask: u128) -> Operand {
        Operand {
            marks: Marks::from_mask(mask, 32),
            zeros: 0,
            ones: 0,
        }
    }

    fn constant(value: u128) -> Operand {
        Operand {
            marks: None,
            zeros: !value & 0xffff_ffff,
            ones: value,
        }
    }

    fn mask(marks: Option<Marks>) -> u128 {
        marks.and_then(|marks| marks.mask()).unwrap_or(0)
    }

    /// **A bit is marked where a marked input bit reaches it.** An `and`
    /// with a written zero writes the bit and one with a written one keeps
    /// it marked; an `or` the reverse; an addition carries upward from the
    /// lowest marked bit; a shift by a written constant moves the marks; an
    /// operation with no rule marks its whole result, and one with no
    /// marked input marks nothing.
    #[test]
    fn marks_follow_the_bits_an_operation_reads() {
        let and = SsaOp::<MockTarget>::And {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        assert_eq!(mask(result(&and, &[marked(0xff), constant(0)], 32)), 0);
        assert_eq!(
            mask(result(&and, &[marked(0xff), constant(0x0f)], 32)),
            0x0f
        );
        let or = SsaOp::<MockTarget>::Or {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        assert_eq!(mask(result(&or, &[marked(0xff), constant(0xf0)], 32)), 0x0f);
        let add = SsaOp::<MockTarget>::Add {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        assert_eq!(
            mask(result(&add, &[marked(0x100), constant(1)], 32)),
            0xffff_ff00
        );
        let shl = SsaOp::<MockTarget>::Shl {
            dest: var(2),
            value: var(0),
            amount: var(1),
            flags: None,
        };
        assert_eq!(mask(result(&shl, &[marked(0x0f), constant(4)], 32)), 0xf0);
        let ceq = SsaOp::<MockTarget>::Ceq {
            dest: var(2),
            left: var(0),
            right: var(1),
            signaling: false,
        };
        assert_eq!(mask(result(&ceq, &[marked(1), constant(0)], 1)), 1);
        assert_eq!(result(&add, &[constant(3), constant(4)], 32), None);
        let _ = MockType::U32;
    }
}
