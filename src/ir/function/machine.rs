//! What a native function's machine registers hold where execution crosses a
//! boundary its optimized SSA does not state.
//!
//! Normalized SSA keeps no register map: a value is a value, wherever the
//! machine kept it. Execution still crosses boundaries where registers, not
//! values, are what passes:
//!
//! - a write that changes code the function has yet to run in the block it
//!   runs, after which execution goes on in a lift of what the bytes hold
//!   now, entered with the registers as the writing instruction left them (a
//!   *resume* point);
//! - control entering a host block whose code a write changed, where
//!   execution goes on in a lift of what that block holds now, entered with
//!   the registers as control brings them there (a host block's *entry*);
//! - a call, whose callee finds the caller's registers, the ones its call
//!   states nothing of included;
//! - a return, which leaves the caller the registers as the callee left
//!   them, the ones its convention calls preserved included.
//!
//! The lift records the register map at each such point, the registers its
//! code names, and the entry value of each, under keys the host defines. A
//! register the function never names it leaves as it found it.
//!
//! The record keeps nothing alive. Normalization runs as if it were not
//! there, and the record follows what it does: a value a substitution
//! replaced is read as its replacement, a renumbering renames its values, a
//! removed instruction drops its points. A host block's entry is no
//! instruction and outlives every pass: a block nothing carries any more is
//! one control never enters. A value nothing defines any more is
//! not stated, and execution goes on with that register unknown.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, mem,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::{
    identity::InstrId,
    ops::calls::{CallInput, CallInputKey},
    variable::SsaVarId,
};

/// Which register map of an instruction a record holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MachinePoint {
    /// After the instruction's source instruction: what execution goes on
    /// with where a write it made changed the code ([`SsaFunction::resume_state`](crate::ir::function::SsaFunction::resume_state)).
    After,
    /// What the instruction hands on as it transfers: a call's callee
    /// entering, a return's caller resuming
    /// ([`SsaFunction::handover_state`](crate::ir::function::SsaFunction::handover_state)).
    Handover,
}

/// The extension a stated register's value takes to fill the register: a
/// value a point names became narrower than the register where the
/// conversion that widened it was retired, and the register holds that
/// conversion of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Widening {
    /// The register value's width, in bits.
    pub bits: u32,
    /// Whether the value is sign-extended to it; zero-extended otherwise.
    pub signed: bool,
}

impl Widening {
    /// `self` applied over `inner`, the widening of the value `self` widens,
    /// as one widening: a zero-extended value has no sign bit set for a
    /// sign extension to copy, so both read as the inner extension. A
    /// zero extension of a sign-extended value is no one widening.
    #[must_use]
    pub const fn over(self, inner: Self) -> Option<Self> {
        if !inner.signed || self.signed {
            Some(Self {
                bits: self.bits,
                signed: inner.signed,
            })
        } else {
            None
        }
    }
}

/// One register a machine state names: its key, its value, and the
/// widening the value takes to fill the register, where it is narrower.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatedRegister {
    /// The register's key, as the host defines it.
    pub key: CallInputKey,
    /// The value the register holds, or the one it holds widened.
    pub value: SsaVarId,
    /// How `value` fills the register, where it is narrower.
    pub widened: Option<Widening>,
}

impl From<CallInput> for StatedRegister {
    fn from(input: CallInput) -> Self {
        Self {
            key: input.key,
            value: input.value,
            widened: None,
        }
    }
}

/// The value a substitution replaced each value with, and the widening the
/// replacement takes to be the value it replaced.
///
/// Printed as the map it holds, as the record printed it before the map had a
/// type of its own: hosts digest a function's `Debug` rendering.
#[derive(Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
struct Substitutes(
    #[cfg_attr(feature = "serde", serde(with = "crate::ir::function::map_as_pairs"))]
    BTreeMap<SsaVarId, (SsaVarId, Option<Widening>)>,
);

impl fmt::Debug for Substitutes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Substitutes {
    /// Records that `new` replaced `old`, read as `old` through `widening`.
    fn insert(&mut self, old: SsaVarId, new: SsaVarId, widening: Option<Widening>) {
        self.0.insert(old, (new, widening));
    }

    /// `register` followed through the substitutions: the value it reads now
    /// and the widening that value takes, or `None` where the widenings on
    /// the way compose to none.
    fn follow(&self, register: StatedRegister) -> Option<StatedRegister> {
        let mut followed = register;
        // Each step names a value substituted later than the one before; the
        // bound only guards a malformed record.
        for _ in 0..=self.0.len() {
            let Some((next, widening)) = self.0.get(&followed.value) else {
                break;
            };
            followed.value = *next;
            followed.widened = match (followed.widened, widening) {
                (outer, None) => outer,
                (None, Some(inner)) => Some(*inner),
                (Some(outer), Some(inner)) => Some(outer.over(*inner)?),
            };
        }
        Some(followed)
    }
}

/// A native function's register maps at the points execution crosses, the
/// registers it names, and its entry value of each.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MachineStates {
    /// The keys of every register the function's code reads or writes.
    named: Box<[CallInputKey]>,
    /// The function's entry value of each register it reads before
    /// writing, under the register's key.
    entries: Box<[CallInput]>,
    /// Each point's registers.
    #[cfg_attr(feature = "serde", serde(with = "crate::ir::function::map_as_pairs"))]
    states: BTreeMap<(InstrId, MachinePoint), Box<[StatedRegister]>>,
    /// Each host block's registers where control enters it, by the host's
    /// identity of the block ([`BlockProvenance`](crate::ir::block::BlockProvenance)).
    #[cfg_attr(feature = "serde", serde(with = "crate::ir::function::map_as_pairs"))]
    hosts: BTreeMap<u64, Box<[StatedRegister]>>,
    /// The value a substitution replaced each value a point names with, and
    /// the widening the replacement takes to be the value it replaced.
    substitutes: Substitutes,
    /// Every value a point names, directly or as a substitute.
    referenced: BTreeSet<SsaVarId>,
}

impl MachineStates {
    /// Whether nothing is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty() && self.hosts.is_empty() && self.named.is_empty()
    }

    /// Records the registers the function names and its entry values.
    pub fn set_registers(&mut self, named: Box<[CallInputKey]>, entries: Box<[CallInput]>) {
        self.named = named;
        self.entries = entries;
    }

    /// The keys of every register the function's code reads or writes.
    #[must_use]
    pub fn named(&self) -> &[CallInputKey] {
        &self.named
    }

    /// The function's entry value of each register it reads before writing.
    #[must_use]
    pub fn entries(&self) -> &[CallInput] {
        &self.entries
    }

    /// Records the register map of `point` at `at`.
    pub fn insert(&mut self, at: InstrId, point: MachinePoint, state: Box<[CallInput]>) {
        let state = self.stated(&state);
        self.states.insert((at, point), state);
    }

    /// Records the register map control enters the host block `host` with.
    pub fn insert_host(&mut self, host: u64, state: Box<[CallInput]>) {
        let state = self.stated(&state);
        self.hosts.insert(host, state);
    }

    /// `state` as the record keeps it, its values noted as referenced.
    fn stated(&mut self, state: &[CallInput]) -> Box<[StatedRegister]> {
        self.referenced
            .extend(state.iter().map(|input| input.value));
        state.iter().copied().map(StatedRegister::from).collect()
    }

    /// Notes that `new` replaced `old` everywhere it was read: a point
    /// naming `old` now reads `new`.
    pub fn substituted(&mut self, old: SsaVarId, new: SsaVarId) {
        self.substitute(old, new, None);
    }

    /// Notes that `old`, a conversion `widening` makes of `new`, was retired:
    /// a point naming `old` now reads `new`, widened.
    pub fn widened(&mut self, old: SsaVarId, new: SsaVarId, widening: Widening) {
        self.substitute(old, new, Some(widening));
    }

    /// Records the substitution of `new`, widened by `widening` where it is
    /// narrower, for `old`.
    fn substitute(&mut self, old: SsaVarId, new: SsaVarId, widening: Option<Widening>) {
        if old != new && self.referenced.contains(&old) {
            self.substitutes.insert(old, new, widening);
            self.referenced.insert(new);
        }
    }

    /// Renames every value after a renumbering, by `survivor`: a value's
    /// id after it, or `None` for a value it removed, which no point states
    /// any more.
    pub fn renumbered(&mut self, survivor: impl Fn(SsaVarId) -> Option<SsaVarId>) {
        if self.is_empty() {
            return;
        }
        // Each value is read as its last substitute first: a value a
        // substitution replaced survives no renumbering, its substitute may.
        let substitutes = mem::take(&mut self.substitutes);
        let rename = |registers: &[StatedRegister]| -> Box<[StatedRegister]> {
            registers
                .iter()
                .filter_map(|register| {
                    let followed = substitutes.follow(*register)?;
                    Some(StatedRegister {
                        value: survivor(followed.value)?,
                        ..followed
                    })
                })
                .collect()
        };
        // An entry value is the register itself, never a narrower one.
        self.entries = self
            .entries
            .iter()
            .filter_map(|input| {
                let followed = substitutes.follow(StatedRegister::from(*input))?;
                followed.widened.is_none().then_some(())?;
                Some(CallInput {
                    key: input.key,
                    value: survivor(followed.value)?,
                })
            })
            .collect();
        for state in self.states.values_mut().chain(self.hosts.values_mut()) {
            *state = rename(state);
        }
        self.referenced = self
            .states
            .values()
            .chain(self.hosts.values())
            .flat_map(|state| state.iter().map(|register| register.value))
            .collect();
    }

    /// Drops the points whose instructions are gone: those not in `live`.
    pub fn retain_instructions(&mut self, live: &BTreeSet<InstrId>) {
        self.states.retain(|(at, _), _| live.contains(at));
    }

    /// Drops everything.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Every recorded point, in order.
    pub fn points(&self) -> impl Iterator<Item = (InstrId, MachinePoint)> + '_ {
        self.states.keys().copied()
    }

    /// The register map of `point` at `at`: each register's value, followed
    /// through the substitutions made since, where `defined` says something
    /// still defines it.
    #[must_use]
    pub fn state(
        &self,
        at: InstrId,
        point: MachinePoint,
        defined: impl Fn(SsaVarId) -> bool,
    ) -> Vec<StatedRegister> {
        self.states
            .get(&(at, point))
            .map_or_else(Vec::new, |state| self.followed(state, defined))
    }

    /// The register map control enters the host block `host` with, read as
    /// [`Self::state`] reads a point's.
    #[must_use]
    pub fn host_state(&self, host: u64, defined: impl Fn(SsaVarId) -> bool) -> Vec<StatedRegister> {
        self.hosts
            .get(&host)
            .map_or_else(Vec::new, |state| self.followed(state, defined))
    }

    /// `state`'s registers, each followed through the substitutions made
    /// since, where `defined` says something still defines its value.
    fn followed(
        &self,
        state: &[StatedRegister],
        defined: impl Fn(SsaVarId) -> bool,
    ) -> Vec<StatedRegister> {
        state
            .iter()
            .filter_map(|register| self.substitutes.follow(*register))
            .filter(|register| defined(register.value))
            .collect()
    }

    /// Every host block with an entry map, in order.
    pub fn hosts(&self) -> impl Iterator<Item = u64> + '_ {
        self.hosts.keys().copied()
    }

    /// Whether `at` has a register map for `point`.
    #[must_use]
    pub fn contains(&self, at: InstrId, point: MachinePoint) -> bool {
        self.states.contains_key(&(at, point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: CallInputKey = CallInputKey::new(7);

    fn value(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    fn record(states: &mut MachineStates, stated: SsaVarId) -> InstrId {
        let at = InstrId::from_raw(1).expect("nonzero");
        states.insert(
            at,
            MachinePoint::After,
            vec![CallInput {
                key: KEY,
                value: stated,
            }]
            .into(),
        );
        states.insert_host(
            0x40,
            vec![CallInput {
                key: KEY,
                value: stated,
            }]
            .into(),
        );
        at
    }

    const ZERO_32_TO_64: Widening = Widening {
        bits: 64,
        signed: false,
    };

    #[test]
    fn a_retired_widening_is_read_as_its_operand_widened() {
        let mut states = MachineStates::default();
        let at = record(&mut states, value(5));
        states.widened(value(5), value(3), ZERO_32_TO_64);
        let expected = vec![StatedRegister {
            key: KEY,
            value: value(3),
            widened: Some(ZERO_32_TO_64),
        }];
        assert_eq!(states.state(at, MachinePoint::After, |_| true), expected);
        assert_eq!(states.host_state(0x40, |_| true), expected);
        // A plain substitution after it keeps the widening.
        states.substituted(value(3), value(2));
        assert_eq!(
            states.host_state(0x40, |_| true),
            vec![StatedRegister {
                value: value(2),
                ..expected[0]
            }]
        );
    }

    #[test]
    fn widenings_compose_where_one_extension_states_both() {
        let sign_16_to_32 = Widening {
            bits: 32,
            signed: true,
        };
        let zero_16_to_32 = Widening {
            bits: 32,
            signed: false,
        };
        // A zero extension of a zero-extended value, and a sign extension of
        // one, are the inner zero extension to the outer width.
        assert_eq!(ZERO_32_TO_64.over(zero_16_to_32), Some(ZERO_32_TO_64));
        let sign_32_to_64 = Widening {
            bits: 64,
            signed: true,
        };
        assert_eq!(sign_32_to_64.over(zero_16_to_32), Some(ZERO_32_TO_64));
        assert_eq!(sign_32_to_64.over(sign_16_to_32), Some(sign_32_to_64));
        // A zero extension of a sign-extended value is none.
        assert_eq!(ZERO_32_TO_64.over(sign_16_to_32), None);
        let mut states = MachineStates::default();
        let at = record(&mut states, value(5));
        states.widened(value(5), value(3), ZERO_32_TO_64);
        states.widened(value(3), value(1), sign_16_to_32);
        assert!(states.state(at, MachinePoint::After, |_| true).is_empty());
        assert!(states.host_state(0x40, |_| true).is_empty());
    }

    #[test]
    fn renumbering_keeps_widenings_and_host_entries() {
        let mut states = MachineStates::default();
        let at = record(&mut states, value(5));
        states.widened(value(5), value(3), ZERO_32_TO_64);
        // Value 3 becomes value 0; value 5 is gone.
        states.renumbered(|old| (old == value(3)).then_some(value(0)));
        let expected = vec![StatedRegister {
            key: KEY,
            value: value(0),
            widened: Some(ZERO_32_TO_64),
        }];
        assert_eq!(states.state(at, MachinePoint::After, |_| true), expected);
        assert_eq!(states.host_state(0x40, |_| true), expected);
        assert_eq!(states.hosts().collect::<Vec<_>>(), vec![0x40]);
    }

    #[test]
    fn a_value_nothing_defines_is_not_stated() {
        let mut states = MachineStates::default();
        let at = record(&mut states, value(5));
        assert!(states.state(at, MachinePoint::After, |_| false).is_empty());
        assert!(states.host_state(0x40, |_| false).is_empty());
        assert!(states.host_state(0x41, |_| true).is_empty());
    }
}
