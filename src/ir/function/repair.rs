//! Lightweight SSA repair for passes that do NOT modify the CFG structure.
//!
//! After instruction-only passes (constant propagation, copy propagation, DCE,
//! algebraic simplification), the SSA form needs minor cleanup but NOT full
//! reconstruction. The CFG topology, dominator tree, and phi placement are all
//! still valid - only instruction-level artifacts need attention.
//!
//! [`repair_ssa`](crate::ir::function::SsaFunction::repair_ssa) is the boundary repair for
//! edit sessions classified as `SsaEditScope::InstructionsOnly`.
//!
//! # What it does
//!
//! 1. **Nop stripping** - Removes `Nop` instructions, reindexes variable `DefSite`s
//! 2. **Trivial phi elimination** - Removes phis where all operands resolve to one value
//! 3. **Dead phi elimination** - Removes phis whose result has no consumers
//! 4. **Variable compaction** - Removes orphaned variables and reindexes IDs
//!
//! # What it does NOT do (saving significant overhead)
//!
//! - Recompute dominators or dominance frontiers
//! - Recompute liveness
//! - Re-place phi nodes
//! - Full variable renaming
//! - Orphan origin assignment
//!
//! # When to use vs. `reconstruct_values`
//!
//! | Pass type | Use |
//! |-----------|-----|
//! | Instruction-only (replace opcodes, substitute uses) | `repair_ssa` |
//! | CFG-modifying (add/remove blocks, change branches) | `reconstruct_values` for the disturbed values |

use log::debug;

use crate::{
    ir::function::{
        SsaFunction, TrivialPhiOptions,
        rebuild::SsaRebuilder,
        transforms::{Cleanup, VariableCensus},
    },
    target::Target,
};

impl<T: Target> SsaFunction<T> {
    /// Lightweight SSA repair for passes that don't modify CFG structure.
    ///
    /// The boundary repair for edits classified as `InstructionsOnly`. It
    /// assumes the CFG topology is unchanged and only cleans up
    /// instruction-level artifacts.
    ///
    /// # What this does
    ///
    /// 1. Strips Nop instructions and reindexes variable DefSites
    /// 2. Eliminates trivial phi nodes (all operands resolve to one value)
    /// 3. Eliminates dead phi nodes (result never used)
    /// 4. Compacts orphaned variables and reindexes IDs
    ///
    /// # When to use
    ///
    /// Use this when the pass only:
    /// - Replaces instruction opcodes/operands
    /// - Converts instructions to Nops (for DCE)
    /// - Substitutes variable uses (copy propagation, GVN)
    ///
    /// Do NOT use this if the pass:
    /// - Adds, removes, or reorders blocks
    /// - Changes branch targets (changes predecessor lists)
    /// - Converts branches to jumps (changes CFG edges)
    pub fn repair_ssa(&mut self) {
        if self.blocks.is_empty() {
            return;
        }

        self.strip_nops();
        // One walk both restores the definition sites and says which variables
        // are defined and read; it stays true until a phi is rewritten.
        let mut census = VariableCensus::refresh(self);
        let (_, trivial_changed) =
            self.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });
        let dead_changed = self.eliminate_dead_phis();
        if trivial_changed || dead_changed {
            census = VariableCensus::of(self);
        }
        census.compact(self);
        // An instruction-scope edit can leave a use naming a definition that
        // sits later in the same block. The rebuild path repairs exactly this
        // (Phases 17b/18b); without it here, `repair_ssa` returns IR the
        // transactional guard rejects as `IntraBlockCycle` and the pass is
        // rolled back with its work discarded.
        //
        // A use no dominating value can stand for is left as it is: the
        // verifier reports it as a use above its definition, which is the
        // honest outcome. Substituting a value minted at entry would verify
        // while reading something no caller supplies.
        if let Err(error) = SsaRebuilder::new(self).repair_same_block_future_uses() {
            debug!("repair_ssa left a use unrepaired: {error}");
        }
        // Every phase above rewrites instructions or variables, and none of them
        // maintains the use index. Restoring it here is what lets a caller treat
        // `repair_ssa` as leaving a function it can analyse: the sparse analyses
        // revisit a value's consumers only through this index, so a stale entry
        // is a consumer that never re-runs and a fact that stays narrower than
        // the truth. Callers reach this directly (the scheduler does), not only
        // through an edit session, so the obligation cannot live with the caller.
        self.recompute_uses();
        // An instruction inserted by the pass that just ran arrived without an
        // identity: only the function's allocator issues those, and the raw
        // construction surface has no allocator in hand. Stamping here is what
        // makes "an instruction in a repaired function has an identity" hold
        // for every path into repair, the scheduler's included.
        self.assign_instr_ids();
        // A phi's result is the join of its operands', and an edit that changed
        // which values reach a phi changed that join - its type included, for a
        // phi the edit minted untyped.
        self.settle_phi_types();
        self.refresh_phi_facts();
        // An edit that removed the last address-taking operation makes a slot
        // an ordinary value again, and one that added one makes it memory.
        self.refresh_address_taken();
    }
}
