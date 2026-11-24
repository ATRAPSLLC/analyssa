//! Execution-local admission of known and unresolved memory writes before their effects.

use std::{collections::BTreeMap, fmt, ops::Range};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::analysis::execution::{Address, MemoryRefusal, Root};

/// A host-local identity for one instruction source within a root trial.
///
/// Hosts use the same identity for every entry and known write affecting that
/// source. Distinct sources require distinct identities; overlapping sources
/// may each be affected by the same write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstructionSourceId(u128);

impl InstructionSourceId {
    /// Creates an identity in the host's instruction-source namespace.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value as u128)
    }

    /// Creates an active-version identity distinct from every fixed-source identity.
    ///
    /// The value is local to this host's bounded version registry. Entering it
    /// requires a fresh source-state check rather than a whole-run fixed predicate.
    #[must_use]
    pub const fn versioned(value: u64) -> Self {
        Self((1u128 << 64) | value as u128)
    }

    /// Creates an original-image identity protected by exact instruction fetches.
    ///
    /// It is distinct from both whole-run fixed predicates and the recovered
    /// version registry. Hosts must verify actual retained fetch coverage and
    /// initial instruction contents before admitting this contract.
    #[must_use]
    pub const fn fetched(value: u64) -> Self {
        Self((2u128 << 64) | value as u128)
    }
}

/// Path-local history of entered and written instruction sources.
///
/// Cloning an execution path clones this history; it is part of memory-state
/// equality and hashing. Restoration or unknown memory clobbers do not erase
/// a known write. Source identities, rather than individual store events, bound
/// storage by the host's finite set of available instruction sources. Active
/// versions count live activations separately, after a fresh byte check at each
/// entry; retiring an activation preserves its historical source events.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct InstructionSources {
    /// Bit zero records entry; bit one records any known write.
    states: BTreeMap<InstructionSourceId, u8>,
    /// Live activation counts for source versions checked at each entry.
    active: BTreeMap<InstructionSourceId, usize>,
    /// Whether a proven local fence covers all reaching instruction writes.
    synchronized: bool,
    /// What the path's protection changes left on the storage they named.
    protections: Protections,
    /// How many writes changed code an activation had entered, numbering
    /// each ([`Self::record_change`]).
    generation: u64,
    /// For each source such a write changed and each byte range it wrote,
    /// in the coordinates the source's instructions name, the number of the
    /// last write to it: one entry per range however often it is written.
    changes: BTreeMap<(InstructionSourceId, u64, u64), u64>,
    /// The most bytes one recorded range spans: a range ending past a byte
    /// starts no further before it.
    widest: u64,
}

/// What a path's protection changes left on the storage they named: the
/// permissions in force over each range a change installed them on, and every
/// byte any change covered. A later change over part of a range replaces that
/// part; the rest keeps what it held.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
struct Protections {
    /// The permissions in force, by the start of the disjoint range they hold
    /// over, with its end in the same root's coordinates.
    current: BTreeMap<Address, (i64, InstructionPermissions)>,
    /// Every range a change covered, by start, with its end: disjoint and
    /// never adjacent.
    changed: BTreeMap<Address, i64>,
}

impl Protections {
    /// The ranges of `map` that overlap `start..end` in `start`'s root, by
    /// start, with their ends.
    fn overlapping<V: Copy>(
        map: &BTreeMap<Address, (i64, V)>,
        start: Address,
        end: i64,
    ) -> Vec<(Address, i64, V)> {
        let before = map
            .range(..start)
            .next_back()
            .filter(|(key, (until, _))| key.root() == start.root() && *until > start.offset());
        let from = map
            .range(start..)
            .take_while(|(key, _)| key.root() == start.root() && key.offset() < end);
        before
            .into_iter()
            .chain(from)
            .map(|(key, (until, value))| (*key, *until, *value))
            .collect()
    }

    /// Installs `permissions` over `len` bytes from `start`, replacing what
    /// was in force there and keeping it on either side.
    fn set(&mut self, start: Address, len: u64, permissions: InstructionPermissions) {
        let Some(end) = i64::try_from(len)
            .ok()
            .and_then(|len| start.offset().checked_add(len))
        else {
            return;
        };
        if len == 0 {
            return;
        }
        for (key, until, held) in Self::overlapping(&self.current, start, end) {
            self.current.remove(&key);
            if key.offset() < start.offset() {
                self.current.insert(key, (start.offset(), held));
            }
            if until > end {
                let rest = Address::new(start.root(), end, start.pointer_size());
                self.current.insert(rest, (until, held));
            }
        }
        self.current.insert(start, (end, permissions));
        self.cover(start, end);
    }

    /// Adds `start..end` to the bytes a change covered, joining the ranges
    /// it meets or touches.
    fn cover(&mut self, start: Address, end: i64) {
        let (mut from, mut to) = (start, end);
        // The ranges are disjoint, so walking back from the last one starting
        // at or before `end`, each ends before the next: the first to end
        // before `start` leaves every earlier one apart.
        let last = Address::new(start.root(), end, start.pointer_size());
        let meets: Vec<(Address, i64)> = self
            .changed
            .range(..=last)
            .rev()
            .take_while(|(key, until)| key.root() == start.root() && **until >= start.offset())
            .map(|(key, until)| (*key, *until))
            .collect();
        for (key, until) in meets {
            self.changed.remove(&key);
            if key.offset() < from.offset() {
                from = key;
            }
            to = to.max(until);
        }
        self.changed.insert(from, to);
    }

    /// The permissions in force at `address`, where a change installed any.
    fn at(&self, address: Address) -> Option<InstructionPermissions> {
        let end = address.offset().checked_add(1)?;
        Self::overlapping(&self.current, address, end)
            .first()
            .map(|(_, _, permissions)| *permissions)
    }

    /// The `len` bytes from `address`, cut where what is in force changes:
    /// each piece's start, length and the permissions a change installed
    /// over it, `None` where none did.
    fn over(
        &self,
        address: Address,
        len: u64,
    ) -> Vec<(Address, u64, Option<InstructionPermissions>)> {
        let Some(end) = i64::try_from(len)
            .ok()
            .and_then(|len| address.offset().checked_add(len))
        else {
            return vec![(address, len, None)];
        };
        let mut pieces = Vec::new();
        let mut at = address.offset();
        let piece = |from: i64, to: i64, permissions| {
            (
                Address::new(address.root(), from, address.pointer_size()),
                u64::try_from(to.saturating_sub(from)).unwrap_or(0),
                permissions,
            )
        };
        for (key, until, permissions) in Self::overlapping(&self.current, address, end) {
            let from = key.offset().max(address.offset());
            if from > at {
                pieces.push(piece(at, from, None));
            }
            let to = until.min(end);
            pieces.push(piece(from, to, Some(permissions)));
            at = to;
        }
        if at < end {
            pieces.push(piece(at, end, None));
        }
        pieces
    }

    /// Whether a change covered any of the `len` bytes from `address`.
    fn changed(&self, address: Address, len: u64) -> bool {
        let Some(end) = i64::try_from(len)
            .ok()
            .and_then(|len| address.offset().checked_add(len))
        else {
            return true;
        };
        if len == 0 {
            return false;
        }
        // Of disjoint ranges, the last one starting before `end` reaches
        // furthest: the span meets a range only if it meets that one.
        let last = Address::new(
            address.root(),
            end.saturating_sub(1),
            address.pointer_size(),
        );
        self.changed
            .range(..=last)
            .next_back()
            .is_some_and(|(key, until)| key.root() == address.root() && *until > address.offset())
    }

    /// Joins `other`: the permissions both paths hold in force over the
    /// same range stay, and every byte either changed stays changed.
    fn merge(&mut self, other: &Self) {
        self.current
            .retain(|start, held| other.current.get(start) == Some(held));
        for (start, end) in &other.changed {
            self.cover(*start, *end);
        }
    }
}

/// Permissions installed by an established instruction-storage transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstructionPermissions {
    /// Whether the mapping admits ordinary writes.
    pub writable: bool,
    /// Whether the mapping admits instruction fetch after visibility checks.
    pub executable: bool,
}

/// How a protection routine spells the access it installs, as its operating
/// system defines the argument.
///
/// Every spelling here keeps the bytes readable: an access that removes read
/// permission, or adds a modifier such as a guard page, states nothing
/// [`InstructionPermissions`] can hold and decodes to none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProtectionEncoding {
    /// POSIX `PROT_READ` (1), `PROT_WRITE` (2) and `PROT_EXEC` (4) as bits,
    /// as `mprotect` takes them.
    Posix,
    /// Windows `PAGE_*` constants: one access value (`PAGE_READONLY` 0x02,
    /// `PAGE_READWRITE` 0x04, `PAGE_WRITECOPY` 0x08, `PAGE_EXECUTE_READ`
    /// 0x20, `PAGE_EXECUTE_READWRITE` 0x40, `PAGE_EXECUTE_WRITECOPY` 0x80) in
    /// the low byte and modifiers above it, as `VirtualProtect` takes them.
    Windows,
}

impl ProtectionEncoding {
    /// The permissions `value` installs, or `None` where it removes read
    /// access, names a modifier, or is no access value of this encoding.
    #[must_use]
    pub const fn decode(self, value: u64) -> Option<InstructionPermissions> {
        let (writable, executable) = match self {
            Self::Posix => {
                if value & !0b111 != 0 || value & 1 == 0 {
                    return None;
                }
                (value & 2 != 0, value & 4 != 0)
            }
            Self::Windows => match value {
                0x02 => (false, false),
                0x04 | 0x08 => (true, false),
                0x20 => (false, true),
                0x40 | 0x80 => (true, true),
                _ => return None,
            },
        };
        Some(InstructionPermissions {
            writable,
            executable,
        })
    }

    /// The value this encoding spells readable `permissions` with: the one
    /// a protection routine reports as a mapping's previous access.
    #[must_use]
    pub const fn encode(self, permissions: InstructionPermissions) -> u64 {
        let InstructionPermissions {
            writable,
            executable,
        } = permissions;
        match self {
            Self::Posix => 1 | if writable { 2 } else { 0 } | if executable { 4 } else { 0 },
            Self::Windows => match (writable, executable) {
                (false, false) => 0x02,
                (true, false) => 0x04,
                (false, true) => 0x20,
                (true, true) => 0x40,
            },
        }
    }
}

impl InstructionSources {
    /// Returns whether this trial has recorded any instruction-source events.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Retains every source event without cloning the existing destination map.
    pub fn merge(&mut self, other: &Self) {
        self.synchronized &= other.synchronized;
        self.protections.merge(&other.protections);
        for (source, state) in &other.states {
            *self.states.entry(*source).or_default() |= state;
        }
        for (source, count) in &other.active {
            let active = self.active.entry(*source).or_default();
            *active = (*active).max(*count);
        }
        self.generation = self.generation.max(other.generation);
        for (change, generation) in &other.changes {
            let changed = self.changes.entry(*change).or_default();
            *changed = (*changed).max(*generation);
        }
    }

    /// Retains every source event while combining normal-exit observations.
    pub fn union(&self, other: &Self) -> Self {
        let mut joined = self.clone();
        joined.merge(other);
        joined
    }

    /// Records a local instruction fence that makes preceding stores visible.
    pub fn synchronize_fetch(&mut self) {
        self.synchronized = true;
    }

    /// Withdraws the fence guarantee before a write requiring synchronization.
    ///
    /// An instruction-write guard calls this before the write takes effect.
    /// Writes proven unrelated to synchronized instruction storage need not
    /// withdraw the guarantee.
    pub fn dirty_fetch(&mut self) {
        self.synchronized = false;
    }

    /// Returns whether the reaching instruction writes are synchronized.
    pub const fn fetch_synchronized(&self) -> bool {
        self.synchronized
    }

    /// Installs `permissions` over the `len` bytes from `start` a guard
    /// admitted a change of: they replace what was in force there.
    pub fn set_permissions(
        &mut self,
        start: Address,
        len: u64,
        permissions: InstructionPermissions,
    ) {
        self.protections.set(start, len, permissions);
    }

    /// The permissions a change installed in force at `address`, if any.
    #[must_use]
    pub fn permissions(&self, address: Address) -> Option<InstructionPermissions> {
        self.protections.at(address)
    }

    /// The `len` bytes from `address`, cut where what is in force changes:
    /// each piece's start, length and the permissions a change installed
    /// over it, `None` where the storage's own permissions hold.
    #[must_use]
    pub fn permissions_over(
        &self,
        address: Address,
        len: u64,
    ) -> Vec<(Address, u64, Option<InstructionPermissions>)> {
        self.protections.over(address, len)
    }

    /// Whether a change ever covered any of the `len` bytes from `address`
    /// on this path: what was read-only there may have been written.
    #[must_use]
    pub fn protection_changed(&self, address: Address, len: u64) -> bool {
        self.protections.changed(address, len)
    }

    /// Enters an activation whose exact source version was checked in reaching state.
    ///
    /// Earlier writes do not disqualify freshly checked bytes. A host's write
    /// guard protects this version while [`Self::is_active`] is true; every normal
    /// return or tail retirement balances one [`Self::retire_active`] call.
    pub fn enter_active(&mut self, source: InstructionSourceId) -> bool {
        let current = self.active.get(&source).copied().unwrap_or(0);
        let Some(next) = current.checked_add(1) else {
            return false;
        };
        self.active.insert(source, next);
        *self.states.entry(source).or_default() |= 1;
        true
    }

    /// Retires one checked activation, preserving enclosing uses of the same version.
    ///
    /// Returns `false` for an unmatched retirement. Fixed-source predicates and
    /// historical entry/write evidence remain unchanged.
    pub fn retire_active(&mut self, source: InstructionSourceId) -> bool {
        let Some(count) = self.active.get_mut(&source) else {
            return false;
        };
        if *count > 1 {
            *count = count.saturating_sub(1);
        } else {
            self.active.remove(&source);
        }
        true
    }

    /// Returns whether a live activation currently depends on this source version.
    #[must_use]
    pub fn is_active(&self, source: InstructionSourceId) -> bool {
        self.active.contains_key(&source)
    }

    /// Admits a fixed source only when no preceding known write touched it.
    ///
    /// Successful entry remains recorded after the activation returns, because
    /// a whole-run fixed-source predicate also constrains later writes.
    pub fn enter(&mut self, source: InstructionSourceId) -> bool {
        let state = self.states.entry(source).or_default();
        if *state & 2 != 0 {
            return false;
        }
        *state |= 1;
        true
    }

    /// Returns whether this path has entered the source.
    #[must_use]
    pub fn has_entered(&self, source: InstructionSourceId) -> bool {
        self.states.get(&source).is_some_and(|state| state & 1 != 0)
    }

    /// Returns whether this trial recorded a known write affecting the source.
    ///
    /// Restoring bytes, returning from an activation and clobbering data do not
    /// erase the event. A final union may contain both an entry and a write
    /// from different paths; the publication owner checks their conjunction.
    #[must_use]
    pub fn has_written(&self, source: InstructionSourceId) -> bool {
        self.states.get(&source).is_some_and(|state| state & 2 != 0)
    }

    /// Records a write admitted by a version-aware instruction-fetch guard.
    ///
    /// The host must protect every future fetch or reject overlap with an active
    /// version before using this method. The event survives retirement and joins;
    /// fresh admission still checks reaching bytes. Fixed-source identities cannot
    /// use this contract to weaken their whole-run predicates.
    ///
    /// # Errors
    /// Returns [`MemoryRefusal::InstructionSource`] for a fixed-source identity.
    pub fn record_version_write(
        &mut self,
        source: InstructionSourceId,
    ) -> Result<(), MemoryRefusal> {
        if !matches!(source.0 >> 64, 1 | 2) {
            return Err(MemoryRefusal::InstructionSource);
        }
        *self.states.entry(source).or_default() |= 2;
        Ok(())
    }

    /// Records a write that changed `bytes` of `source`, code an activation
    /// has entered and may still run, in the coordinates the source's
    /// instructions name: those bytes are now something its lift is not. An
    /// activation that may still run them goes on in a lift of what they
    /// hold now ([`Self::changed_within`]); entering the source again is
    /// refused ([`Self::enter`]), so no later activation runs the old lift.
    pub fn record_change(&mut self, source: InstructionSourceId, bytes: Range<u64>) {
        *self.states.entry(source).or_default() |= 2;
        self.generation = self.generation.saturating_add(1);
        self.widest = self.widest.max(bytes.end.saturating_sub(bytes.start));
        self.changes
            .insert((source, bytes.start, bytes.end), self.generation);
    }

    /// The number of the last write that changed entered code, `0` before
    /// any.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether a write numbered after `since` changed any of `bytes` of
    /// `source`, in the coordinates the source's instructions name. Only the
    /// ranges that can reach `bytes` are visited: those starting at most the
    /// widest range's length before them.
    #[must_use]
    pub fn changed_within(
        &self,
        source: InstructionSourceId,
        since: u64,
        bytes: Range<u64>,
    ) -> bool {
        if bytes.is_empty() {
            return false;
        }
        let first = bytes.start.saturating_sub(self.widest);
        self.changes
            .range((source, first, 0)..(source, bytes.end, 0))
            .any(|((_, start, end), generation)| {
                *generation > since && *start < bytes.end && bytes.start < *end
            })
    }

    /// Records a known write and reports whether it contradicts an entered fixed source.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryRefusal::InstructionSource`] when the source was already
    /// entered. A write to an unentered source remains recorded for later entry.
    pub fn record_write(&mut self, source: InstructionSourceId) -> Result<(), MemoryRefusal> {
        let state = self.states.entry(source).or_default();
        *state |= 2;
        if *state & 1 != 0 {
            Err(MemoryRefusal::InstructionSource)
        } else {
            Ok(())
        }
    }
}

/// A host's execution-local constraint on memory-write footprints.
///
/// The executor checks the full destination before storing any byte, changing
/// aliases or exposing a stack address. Scalar stores, block operations,
/// atomics, ABI staging and call-model writes use the same guard. Empty writes
/// do not call it. Unresolved physical footprints use
/// [`Self::check_unresolved_write`] without inventing destinations.
///
/// A guard maps destinations to its finite instruction-source identities and
/// records changes in the supplied path-local history. It must not keep
/// mutable write history shared between paths or independent root trials.
/// Refused preflights can remain recorded: an execution with a memory refusal
/// publishes no observations. This interface supplies no read-only image data.
pub trait MemoryWriteGuard: fmt::Debug + Send + Sync {
    /// Checks a possible write whose complete physical footprint is unavailable.
    ///
    /// `pointer` and `len` preserve independently known inputs; they do not
    /// describe a checked linear range when direction, layout or placement is
    /// unresolved. A known empty write does not invoke this method. The default
    /// permits ordinary conservative value execution. A source-effect audit
    /// can refuse rather than treating unknown writes as evaluated effects.
    /// Checked destinations with unknown payloads use [`Self::check_write`].
    ///
    /// # Errors
    ///
    /// Returns the semantic reason the configured policy cannot admit the
    /// missing footprint, such as [`MemoryRefusal::UnresolvedWrite`].
    fn check_unresolved_write(
        &self,
        _pointer: Option<Address>,
        _len: Option<usize>,
    ) -> Result<(), MemoryRefusal> {
        Ok(())
    }

    /// Checks the writes of a routine the host states writes no code, to
    /// data whose place is unresolved: what the program let escape, which
    /// the routine may write without the program naming where.
    ///
    /// The routine's own statement rules out code, so a guard that refuses
    /// unresolved writes only because they may reach code admits these. The
    /// default asks [`Self::check_unresolved_write`], so a guard refuses them
    /// unless it says why data cannot matter to it.
    ///
    /// # Errors
    ///
    /// Returns the semantic reason the configured policy cannot admit them.
    fn check_unresolved_data_write(&self) -> Result<(), MemoryRefusal> {
        self.check_unresolved_write(None, None)
    }

    /// Checks alias authority for an exact numeric write span.
    ///
    /// The default treats a numeric address as an unresolved alias. A host may
    /// admit the span under an established physical mapping. Incomplete write
    /// footprints still use [`Self::check_unresolved_write`].
    ///
    /// # Errors
    /// Returns a refusal when the mapping does not establish the required write.
    fn check_numeric_write(&self, address: Address, len: usize) -> Result<(), MemoryRefusal> {
        self.check_unresolved_write(Some(address), Some(len))
    }

    /// Applies an established protection change to exactly one declared
    /// mapping, returning the permissions the mapping held before it, where
    /// those are established.
    ///
    /// The host validates the extent, supported permissions and writable-storage
    /// contract before changing path-local state. The transition preserves bytes.
    /// For stack storage the executor supplies the unique live activation, if
    /// it can prove backing for the entire span.
    ///
    /// # Errors
    /// Returns an instruction-source refusal when no exact transition is known.
    fn change_instruction_permissions(
        &self,
        _sources: &mut InstructionSources,
        _address: Address,
        _len: usize,
        _stack_owner: Option<u32>,
        _permissions: InstructionPermissions,
    ) -> Result<Option<InstructionPermissions>, MemoryRefusal> {
        Err(MemoryRefusal::InstructionSource)
    }

    /// Checks a known whole destination before any write or pointer escape.
    ///
    /// The address retains its target pointer width. A guard must account for
    /// wrapping spans, including writes covering a complete narrow address
    /// space. Returning success permits the executor's ordinary memory checks
    /// and operation semantics; it does not supply unknown values.
    ///
    /// # Errors
    ///
    /// Returns the semantic reason this execution cannot admit the write.
    fn check_write(
        &self,
        sources: &mut InstructionSources,
        address: Address,
        len: usize,
    ) -> Result<(), MemoryRefusal>;
}

/// Checks unresolved physical aliases before the guard sees a placed write.
///
/// Numeric addresses without a symbol mapping and opaque or stack-derived
/// identities can name image storage. Concrete stack frames and fresh heap
/// allocations retain the memory model's disjoint storage domains.
pub fn check_write(
    guard: &dyn MemoryWriteGuard,
    sources: &mut InstructionSources,
    address: Address,
    len: usize,
) -> Result<(), MemoryRefusal> {
    if len == 0 {
        return Ok(());
    }
    match address.root() {
        Root::Number => guard.check_numeric_write(address, len)?,
        Root::Opaque { .. } | Root::StackDerived { .. } => {
            guard.check_unresolved_write(Some(address), Some(len))?;
        }
        _ => {}
    }
    guard.check_write(sources, address, len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PointerSize, analysis::execution::Root};

    /// **A protection change holds over exactly the bytes it names.** A
    /// change over a page's middle splits what was in force around it; each
    /// piece of a span reads what holds there, and the storage's own
    /// permissions where nothing was installed. Every byte any change covered
    /// stays changed, even once a later change restores what it was. At a
    /// join, only permissions both paths hold over the same range stay in
    /// force, while every changed byte stays changed.
    #[test]
    fn a_protection_change_holds_over_exactly_the_bytes_it_names() {
        let at = |offset| Address::new(Root::Symbol, offset, PointerSize::Bit64);
        let rw = InstructionPermissions {
            writable: true,
            executable: false,
        };
        let rx = InstructionPermissions {
            writable: false,
            executable: true,
        };
        let mut history = InstructionSources::default();
        assert!(!history.protection_changed(at(0x2000), 0x1000));
        history.set_permissions(at(0x2000), 0x1000, rw);
        history.set_permissions(at(0x2400), 0x100, rx);
        assert_eq!(history.permissions(at(0x23ff)), Some(rw));
        assert_eq!(history.permissions(at(0x2400)), Some(rx));
        assert_eq!(history.permissions(at(0x2500)), Some(rw));
        assert_eq!(history.permissions(at(0x3000)), None);
        assert_eq!(
            history.permissions_over(at(0x1ff0), 0x420),
            vec![
                (at(0x1ff0), 0x10, None),
                (at(0x2000), 0x400, Some(rw)),
                (at(0x2400), 0x10, Some(rx)),
            ]
        );
        assert!(history.protection_changed(at(0x2fff), 1));
        assert!(!history.protection_changed(at(0x3000), 0x10));
        assert!(!history.protection_changed(at(0x1000), 0x1000));

        let mut other = InstructionSources::default();
        other.set_permissions(at(0x2000), 0x1000, rw);
        other.set_permissions(at(0x5000), 0x10, rx);
        let joined = history.union(&other);
        assert_eq!(joined.permissions(at(0x2000)), None);
        assert_eq!(joined.permissions(at(0x5000)), None);
        assert!(joined.protection_changed(at(0x2400), 1));
        assert!(joined.protection_changed(at(0x5008), 1));
        let same = other.union(&other.clone());
        assert_eq!(same.permissions(at(0x2000)), Some(rw));
    }

    #[test]
    fn changes_are_found_by_the_bytes_they_reach_and_their_number() {
        let source = InstructionSourceId::new(0x1000);
        let other = InstructionSourceId::new(0x2000);
        let mut history = InstructionSources::default();
        assert!(!history.changed_within(source, 0, 0..u64::MAX));
        // A wide write long before the bytes asked about still reaches them.
        history.record_change(source, 0x10..0x30);
        history.record_change(source, 0x40..0x41);
        history.record_change(other, 0x20..0x28);
        assert_eq!(history.generation(), 3);
        assert!(history.changed_within(source, 0, 0x2f..0x31));
        assert!(history.changed_within(source, 0, 0x40..0x41));
        assert!(!history.changed_within(source, 0, 0x30..0x40));
        assert!(!history.changed_within(source, 0, 0x41..0x50));
        assert!(!history.changed_within(source, 0, 0x20..0x20));
        // Only writes numbered after `since` count, and only the source's own.
        assert!(!history.changed_within(source, 2, 0x10..0x30));
        assert!(history.changed_within(source, 1, 0x40..0x41));
        assert!(history.changed_within(other, 2, 0x27..0x28));
        // A range written again carries the later number.
        history.record_change(source, 0x10..0x30);
        assert!(history.changed_within(source, 3, 0x2f..0x30));
    }

    #[test]
    fn active_versions_balance_nested_entries_without_weakening_fixed_sources() {
        let fixed = InstructionSourceId::new(0);
        let version = InstructionSourceId::versioned(0);
        assert_ne!(fixed, version);
        let mut history = InstructionSources::default();
        assert!(history.enter(fixed));
        assert!(history.enter_active(version));
        assert!(history.enter_active(version));
        let other_path = history.clone();
        assert!(history.retire_active(version));
        assert!(history.is_active(version));
        assert!(history.retire_active(version));
        assert!(!history.is_active(version));
        assert!(other_path.is_active(version));
        assert!(!history.retire_active(version));
        assert!(history.has_entered(version));
        assert_eq!(
            history.record_write(fixed),
            Err(MemoryRefusal::InstructionSource)
        );
    }

    #[test]
    fn instruction_fence_is_required_on_every_path_after_writes() {
        let mut left = InstructionSources::default();
        assert!(!left.fetch_synchronized());
        left.synchronize_fetch();
        let mut right = left.clone();
        assert!(left.union(&right).fetch_synchronized());
        right.dirty_fetch();
        assert!(!left.union(&right).fetch_synchronized());
        assert!(left.fetch_synchronized());
        right.synchronize_fetch();
        assert!(right.union(&left).fetch_synchronized());
    }

    #[test]
    fn refreshed_versions_and_joins_keep_path_local_active_counts() {
        let source = InstructionSourceId::versioned(1);
        let mut history = InstructionSources::default();
        history.record_write(source).expect("unentered write");
        assert!(
            history.enter_active(source),
            "freshly verified bytes follow earlier writes"
        );
        let mut other = history.clone();
        assert!(other.enter_active(source));
        history.merge(&other);
        assert!(history.retire_active(source));
        assert!(history.is_active(source));
        assert!(history.retire_active(source));
        assert!(!history.is_active(source));
        assert!(history.has_written(source));
        assert!(history.enter_active(source));
    }
    #[test]
    fn checked_version_writes_retain_events_without_weakening_fixed_predicates() {
        let fixed = InstructionSourceId::new(7);
        let version = InstructionSourceId::versioned(7);
        let mut history = InstructionSources::default();
        assert!(history.enter(fixed));
        assert_eq!(
            history.record_version_write(fixed),
            Err(MemoryRefusal::InstructionSource)
        );
        assert!(!history.has_written(fixed));
        assert!(history.enter_active(version));
        assert_eq!(history.record_version_write(version), Ok(()));
        assert!(history.has_entered(version));
        assert!(history.has_written(version));
        assert!(history.retire_active(version));
        assert!(history.has_written(version));
        assert_eq!(
            history.record_write(fixed),
            Err(MemoryRefusal::InstructionSource)
        );
    }
    #[test]
    fn fetched_image_sources_have_distinct_balanced_write_histories() {
        let source = InstructionSourceId::fetched(7);
        assert_ne!(source, InstructionSourceId::new(7));
        assert_ne!(source, InstructionSourceId::versioned(7));
        let mut history = InstructionSources::default();
        assert!(history.enter_active(source));
        assert_eq!(history.record_version_write(source), Ok(()));
        assert!(history.has_written(source));
        assert!(history.retire_active(source));
        assert!(!history.is_active(source));
        assert!(history.has_entered(source));
        assert!(history.has_written(source));
    }
}
