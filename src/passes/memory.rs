//! Memory optimization pass — store-to-load forwarding, redundant load
//! elimination, and dead store elimination.
//!
//! Every rewrite here is gated on a Memory SSA alias *proof*
//! ([`MemoryLocation::must_alias`] / [`MemoryLocation::may_alias`]), never on
//! syntactic shape. That is what makes the pass safe on native code, where two
//! textually unrelated addresses routinely name one cell and one address
//! expression routinely names several.
//!
//! # Transforms
//!
//! - **Store-to-load forwarding.** A load whose location must-alias an earlier
//!   store's, with nothing in between that may-alias it, is replaced by the
//!   stored value.
//! - **Redundant load elimination.** The same rule with an earlier *load* as
//!   the source: the second load is replaced by the first load's result.
//! - **Dead store elimination.** A store overwritten by a later must-aliasing
//!   store, with no possible read in between, is removed.
//!
//! # Scope
//!
//! Forwarding and redundant-load elimination reach **across blocks**: a load
//! can be served by a store in a dominating block, read out of Memory SSA's
//! block-entry version. Dead store elimination is deliberately **block-local**
//! — proving a store dead across a merge needs memory liveness, which does not
//! exist yet, and guessing there silently deletes writes.
//!
//! # Barriers
//!
//! Calls, fences, atomics, volatile accesses, and opaque native operations
//! classify to a barrier or an `Unknown` location. Both invalidate the whole
//! availability map and every pending dead-store candidate, so no value is
//! carried across an operation whose memory effect is not modelled.
//!
//! # What this does not model
//!
//! Removing a store is safe against anything *in-process*: a call is a barrier,
//! so a callee can never observe a store this pass removed, and a store with no
//! later must-aliasing store in the same block is never a candidate. It is not
//! safe against a concurrent observer of a non-atomic, non-volatile cell — a
//! racing thread or a signal handler could read the intermediate value. That is
//! the same latitude an optimizing compiler takes, and the reason atomics,
//! volatile accesses, and fences are all barriers here.
//!
//! # Faults
//!
//! Removing an access changes *where* a memory fault would be raised, though
//! not whether one is. Forwarding a load past a must-aliasing store cannot
//! introduce a fault: the store already proved the address writable. Dead store
//! elimination drops a write whose address is proven written again later on the
//! same straight-line path, so the surviving store faults where the removed one
//! would have.

use std::collections::HashMap;

use crate::{
    analysis::memory::{MemoryDefSite, MemoryLocation, MemoryOp, MemorySsa, MemoryVersion},
    events::{Event, EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// A rewrite the analysis proved safe, applied in one editor session.
enum Rewrite {
    /// Replace every use of `redundant` with `available`, then nop the
    /// instruction at `(block, instr)` that defined it.
    Forward {
        /// Value produced by the redundant access.
        redundant: SsaVarId,
        /// Value already holding that memory's contents.
        available: SsaVarId,
        /// Block of the access to remove.
        block: usize,
        /// Instruction index of the access to remove.
        instr: usize,
    },
    /// Remove the dead store at `(block, instr)`.
    DeadStore {
        /// Block of the store to remove.
        block: usize,
        /// Instruction index of the store to remove.
        instr: usize,
    },
}

/// Runs memory optimization on `ssa`.
///
/// # Arguments
///
/// * `ssa` — The SSA function to transform in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::LoadForwarded`] and
///   [`EventKind::DeadStoreRemoved`].
/// * `ptr_size` — Target pointer width. Address arithmetic wraps at this width,
///   and every alias proof this pass acts on depends on folded displacements
///   being canonicalised into it: without it, a sign-extended `-8` and a
///   zero-extended `0xFFFF_FFF8` name the same cell but decode 4 GiB apart, and
///   the pass would prove them disjoint.
///
/// # Returns
///
/// `true` if any load was forwarded or any dead store removed.
pub fn run<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    ptr_size: PointerSize,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let rewrites = {
        let mem_ssa = MemorySsa::build(ssa, ptr_size);
        plan_rewrites(ssa, &mem_ssa)
    };
    if rewrites.is_empty() {
        return false;
    }
    apply_rewrites(ssa, method, events, rewrites)
}

/// Walks every block and returns the rewrites the alias model proves safe.
fn plan_rewrites<T: Target>(ssa: &SsaFunction<T>, mem_ssa: &MemorySsa<T>) -> Vec<Rewrite> {
    // Index the classified memory operations by position so the block walk can
    // ask "what does this instruction do to memory?" in O(1).
    let mut by_position: HashMap<(usize, usize), &MemoryOp<T>> = HashMap::new();
    for op in mem_ssa.operations() {
        by_position.insert((op.block(), op.instr()), op);
    }

    let mut rewrites = Vec::new();
    for block_idx in 0..ssa.block_count() {
        plan_block(ssa, mem_ssa, &by_position, block_idx, &mut rewrites);
    }
    rewrites
}

/// Plans the rewrites for one block.
fn plan_block<T: Target>(
    ssa: &SsaFunction<T>,
    mem_ssa: &MemorySsa<T>,
    by_position: &HashMap<(usize, usize), &MemoryOp<T>>,
    block_idx: usize,
    rewrites: &mut Vec<Rewrite>,
) {
    let Some(block) = ssa.block(block_idx) else {
        return;
    };

    // Value currently held by each location, and the store that put it there
    // if it has not been read since.
    let mut available: Vec<(MemoryLocation<T>, SsaVarId)> = Vec::new();
    let mut pending_stores: Vec<(MemoryLocation<T>, usize)> = Vec::new();

    seed_from_dominator(mem_ssa, ssa, by_position, block_idx, &mut available);

    for (instr_idx, _) in block.instructions().iter().enumerate() {
        let Some(op) = by_position.get(&(block_idx, instr_idx)) else {
            continue;
        };
        match op {
            MemoryOp::Load { location, dest, .. } => {
                // A load reads whatever may-alias it, so any pending store it
                // could observe is no longer dead.
                pending_stores.retain(|(pending, _)| !pending.may_alias(location));

                if let Some(value) = lookup_available(&available, location) {
                    rewrites.push(Rewrite::Forward {
                        redundant: *dest,
                        available: value,
                        block: block_idx,
                        instr: instr_idx,
                    });
                } else {
                    // Not forwardable, but its result is now the live value for
                    // this cell — that is redundant-load elimination for any
                    // later load of the same cell.
                    set_available(&mut available, location.clone(), *dest);
                }
            }
            MemoryOp::Store {
                location, value, ..
            } => {
                // A store that overwrites an unread must-aliasing store kills
                // it. Only an exact must-alias proves full coverage; a partial
                // overlap leaves some of the earlier store observable.
                if let Some(position) = pending_stores
                    .iter()
                    .position(|(pending, _)| pending.must_alias(location))
                {
                    let (_, dead_instr) = pending_stores.swap_remove(position);
                    rewrites.push(Rewrite::DeadStore {
                        block: block_idx,
                        instr: dead_instr,
                    });
                }
                // Anything this store could touch is no longer known, and any
                // pending store it could partially overlap is no longer
                // provably dead.
                available.retain(|(known, _)| !known.may_alias(location));
                // A store this one only partially overlaps is no longer
                // provably dead: part of it survives. Dropping the candidate is
                // the conservative answer.
                pending_stores.retain(|(pending, _)| !pending.may_alias(location));

                set_available(&mut available, location.clone(), *value);
                pending_stores.push((location.clone(), instr_idx));
            }
            MemoryOp::ReadWrite { location, .. } => {
                // Reads and writes an unmodelled amount of the location.
                pending_stores.retain(|(pending, _)| !pending.may_alias(location));
                available.retain(|(known, _)| !known.may_alias(location));
            }
            MemoryOp::Barrier { .. } => {
                // A call, fence, or opaque native operation: nothing survives.
                available.clear();
                pending_stores.clear();
            }
        }
    }
}

/// Seeds `available` with values reaching this block from a dominating store.
///
/// Safe because a memory definition versions *every location it may alias*, not
/// only the one it names (see `MemorySsa::rename_block`). So a location's
/// block-entry version having a store as its definition really does mean that
/// store's value is intact here: any intervening barrier, call, or overlapping
/// store would have bumped this location's version and made its definition that
/// operation instead, which the `MemoryOp::Store` and `must_alias` checks below
/// then reject.
///
/// Memory SSA renames on the dominator tree, so a location's block-entry
/// version is the version its dominator left. When that version is a store,
/// the stored value dominates this block and can serve loads here.
///
/// A location with a memory phi in this block is skipped: the phi means two or
/// more definitions reach the merge, so no single value is available.
fn seed_from_dominator<T: Target>(
    mem_ssa: &MemorySsa<T>,
    ssa: &SsaFunction<T>,
    by_position: &HashMap<(usize, usize), &MemoryOp<T>>,
    block_idx: usize,
    available: &mut Vec<(MemoryLocation<T>, SsaVarId)>,
) {
    for location in mem_ssa.locations() {
        if mem_ssa
            .memory_phis(block_idx)
            .iter()
            .any(|phi| phi.location == *location)
        {
            continue;
        }
        let Some(version) = mem_ssa.version_at_entry(location, block_idx) else {
            continue;
        };
        let Some(MemoryDefSite::Store {
            block: store_block,
            instr: store_instr,
        }) = mem_ssa.definition(&MemoryVersion::new(location.clone(), version))
        else {
            continue;
        };
        // A store in this very block is handled by the walk itself.
        if store_block == block_idx {
            continue;
        }
        let Some(MemoryOp::Store {
            location: stored_location,
            value,
            ..
        }) = by_position.get(&(store_block, store_instr)).copied()
        else {
            continue;
        };
        // The recorded version identifies the location, but only an exact
        // must-alias proves the store covers a whole read of it.
        if !stored_location.must_alias(location) {
            continue;
        }
        // Guard against a malformed graph handing back a value nothing
        // defines; the editor's dominance check is the real gate, this is
        // cheap insurance.
        if ssa.get_definition(*value).is_none() {
            continue;
        }
        set_available(available, location.clone(), *value);
    }
}

/// Returns the value available for an exact read of `location`.
///
/// Requires [`MemoryLocation::must_alias`], which rejects unknown-width
/// accesses and partial overlaps — a value is only usable when the recorded
/// write covers exactly what is being read.
fn lookup_available<T: Target>(
    available: &[(MemoryLocation<T>, SsaVarId)],
    location: &MemoryLocation<T>,
) -> Option<SsaVarId> {
    available
        .iter()
        .find(|(known, _)| known.must_alias(location))
        .map(|(_, value)| *value)
}

/// Records `value` as the live contents of `location`, replacing any entry for
/// the same cell.
fn set_available<T: Target>(
    available: &mut Vec<(MemoryLocation<T>, SsaVarId)>,
    location: MemoryLocation<T>,
    value: SsaVarId,
) {
    if let Some(slot) = available
        .iter_mut()
        .find(|(known, _)| *known == location)
        .map(|(_, existing)| existing)
    {
        *slot = value;
        return;
    }
    available.push((location, value));
}

/// Applies planned rewrites in one checked editor session.
///
/// A forward is only completed when *every* use of the redundant value could be
/// replaced — the editor's dominance check may refuse some — otherwise the
/// access stays and only the replaced uses benefit.
fn apply_rewrites<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    rewrites: Vec<Rewrite>,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut forwarded = 0usize;
    let mut dead_stores = 0usize;

    let report = ssa.edit(
        SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure),
        |editor| {
            for rewrite in rewrites {
                match rewrite {
                    Rewrite::Forward {
                        redundant,
                        available,
                        block,
                        instr,
                        ..
                    } => {
                        let result = editor.replace_uses_checked(redundant, available);
                        if result.is_complete() {
                            editor.nop_instruction(block, instr)?;
                            forwarded = forwarded.saturating_add(1);
                        }
                    }
                    Rewrite::DeadStore { block, instr, .. } => {
                        editor.nop_instruction(block, instr)?;
                        dead_stores = dead_stores.saturating_add(1);
                    }
                }
            }
            Ok(())
        },
    );

    if report.is_err() {
        return false;
    }
    if forwarded > 0 {
        events.push(Event {
            kind: EventKind::LoadForwarded,
            method: Some(method.clone()),
            location: None,
            message: format!("forwarded {forwarded} load(s) from available memory"),
            pass: None,
        });
    }
    if dead_stores > 0 {
        events.push(Event {
            kind: EventKind::DeadStoreRemoved,
            method: Some(method.clone()),
            location: None,
            message: format!("removed {dead_stores} dead store(s)"),
            pass: None,
        });
    }
    forwarded > 0 || dead_stores > 0
}
