//! The reconstruction engine behind [`SsaFunction::construct_ssa`].
//!
//! [`SsaRebuilder`] is a structured implementation of the standard Cytron et al.
//! algorithm: it places phis over the iterated dominance frontier and renames
//! every grouped variable along the dominator tree. A pass that changes the
//! control-flow graph does not run it; it repairs the values its edit disturbed
//! with [`SsaFunction::reconstruct_values`], and an instruction-only pass with
//! [`SsaFunction::repair_ssa`].
//!
//! [`SsaFunction::construct_ssa`]: crate::ir::function::SsaFunction::construct_ssa
//! [`SsaFunction::reconstruct_values`]: crate::ir::function::SsaFunction::reconstruct_values
//! [`SsaFunction::repair_ssa`]: crate::ir::function::SsaFunction::repair_ssa
//!
//! # Algorithm Pipeline
//!
//! The reconstruction is split into 18 named phases, each operating on explicit
//! intermediate state stored in `SsaRebuilder`:
//!
//! | Stage | Phases | Description |
//! |-------|--------|-------------|
//! | Cleanup | 1 | Remove unreachable blocks |
//! | Collection | 2-5 | Gather origins, types, assign orphan groups |
//! | Analysis | 6-8 | CFG, dominators, dominance frontiers, liveness |
//! | Phi | 9-11 | Clear old phis, place pruned phis, rename variables |
//! | Finalize | 12-18 | Eliminate trivial phis, compact, reindex, verify |
//!
//! # Key Algorithms
//!
//! - **Dominator computation**: Standard Lengauer-Tarjan algorithm
//! - **Dominance frontiers**: Iterated frontier computation for phi placement
//! - **Phi placement**: Pruned SSA - only place phis for live variables
//! - **Renaming**: DFS traversal of the dominator tree with version stacks
//! - **Group joining**: union-find over the groups each phi ties together
//! - **Exception handler integration**: Local dominator trees for handler roots
//!
//! # Design Decisions
//!
//! - **Explicit phase state**: Each phase writes to named fields on `SsaRebuilder`,
//!   making the pipeline individually testable and easier to debug.
//! - **Rename groups**: Variables are grouped for rename by semantic origin (argument,
//!   local, stack temp), not by variable ID. This correctly handles stack-slot reuse.
//! - **Reachability filtering**: All phases operate only on reachable blocks; unreachable
//!   blocks are cleared early to avoid phantom CFG edges.
//! - **Type preservation**: Variable types are preserved through rebuild via per-variable
//!   and per-group type registries.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    BitSet, Error, Result,
    analysis::{
        cfg::SsaCfg,
        exceptions::FunctionRoots,
        liveness,
        phis::{PhiPlacementConfig, place_pruned_phis},
        verifier::{SsaVerifier, VerifierError, VerifyLevel},
    },
    facts::{Derivation, Substitution},
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominance_frontiers, compute_dominators},
    },
    ir::{
        block::SsaBlock,
        function::{AddressTakenSlots, SsaFunction, TrivialPhiOptions, transforms::Cleanup},
        instruction::SsaInstruction,
        ops::{OperandRole, SsaOp},
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, VariableOrigin},
        varstore::VarSet,
    },
    target::Target,
    vartables::VarTable,
};

/// Immutable context passed through the SSA rename phase.
///
/// Bundles all precomputed data structures needed during variable renaming.
/// These are immutable references passed unchanged through all recursive
/// rename operations, ensuring the rename phase cannot accidentally modify
/// the analysis results it depends on.
struct RenameContext<'a, T: Target> {
    /// Maps each variable ID to its [`VariableOrigin`] (Argument, Local, Phi).
    /// Used during rename to determine which group a variable belongs to.
    var_origins: &'a BTreeMap<SsaVarId, VariableOrigin>,

    /// Maps each rename group to its canonical SSA type.
    /// Used when creating new variable versions during rename to ensure
    /// they inherit the correct type.
    group_types: &'a BTreeMap<u32, T::Type>,

    /// Per-variable type map for stack-derived locals where different
    /// variables at the same origin can have different types.
    /// Preserves precise type information through rebuild.
    var_types: &'a BTreeMap<SsaVarId, T::Type>,

    /// CFG successor map: block index → list of successor block indices.
    /// Used by [`Renamer::fill_successor_phi_operands`] to populate phi node
    /// operands.
    successor_map: &'a BTreeMap<usize, Vec<usize>>,

    /// The terminator relation, for deciding whether a normal operand is owed.
    terminator_successor_map: &'a BTreeMap<usize, Vec<usize>>,

    /// Dominator tree child relationships: parent block → list of child blocks.
    /// The rename phase performs a DFS over this tree structure.
    dom_children: &'a BTreeMap<usize, Vec<usize>>,

    /// Maps each block index to the ordered list of rename groups for its phi nodes.
    /// Built from [`place_pruned_phis`] output so rename can associate each phi
    /// with its group - needed when multiple groups share `Phi` origin.
    phi_groups: &'a BTreeMap<usize, Vec<u32>>,

    /// Number of method arguments (including `this` for instance methods).
    /// Used for computing default group IDs from `VariableOrigin::Argument`.
    num_args: usize,

    /// The phis [`SsaRebuilder::clear_all_phis`] removed, keyed by block and
    /// group. `None` marks a block where two cleared phis shared a group, so
    /// no single placed phi can stand for either.
    original_phis: &'a BTreeMap<(usize, u32), Option<OriginalPhi>>,

    /// For each block, the handler and filter entries a throw in it can reach.
    exceptional_successors: BTreeMap<usize, Vec<usize>>,

    /// The handler and filter entry blocks themselves.
    exceptional_roots: BTreeSet<usize>,

    /// The slots whose address the function takes.
    ///
    /// A load of such a slot reads memory something else may have written
    /// through the pointer, so it keeps its own definition instead of being
    /// resolved to the slot's current value.
    address_taken: AddressTakenSlots,
}

/// A phi the rebuild cleared, remembered so the phi placed for the same group
/// in the same block can stand for it.
#[derive(Debug, Clone)]
struct OriginalPhi {
    /// The cleared phi's result, which uses elsewhere still name.
    result: SsaVarId,
    /// The value it received along each incoming edge, keyed by predecessor.
    operands: BTreeMap<usize, SsaVarId>,
}

/// Where each original variable's renamed definition is in scope.
///
/// Renaming by group alone treats every member of a group as a version of one
/// storage slot and rewrites a use to whichever version is newest. That is only
/// right while no two members are live at once, and values joined by a phi
/// often are: in `return i++` the pre-increment value is read after the
/// increment is defined. A use therefore keeps its own variable - renamed -
/// wherever that variable's definition dominates it, which is exactly where
/// this scope holds a binding, and falls back to the group's reaching
/// definition only where a CFG edit left the definition no longer dominating.
///
/// A phi at an exception root is a barrier for its group: a binding made
/// before the root does not reach the handler, because the throw may have
/// happened before that definition ran. What the handler sees is the phi,
/// which merges every value the region may have produced.
#[derive(Debug, Default)]
struct Identity {
    /// Original variable to its renamed versions in scope, innermost last,
    /// each with the group it was pushed for and its position on that group's
    /// version stack.
    active: BTreeMap<SsaVarId, Vec<(SsaVarId, u32, usize)>>,
    /// Per group, the version-stack positions of exception-root merge values
    /// in scope.
    barriers: BTreeMap<u32, Vec<usize>>,
}

impl Identity {
    /// Binds `original` to `renamed`, the value just pushed onto `group`'s stack.
    fn bind(
        &mut self,
        original: SsaVarId,
        renamed: SsaVarId,
        group: u32,
        version_stacks: &BTreeMap<u32, Vec<SsaVarId>>,
        pushed: &mut Pushed,
    ) {
        let position = version_stacks
            .get(&group)
            .map_or(0, |stack| stack.len().saturating_sub(1));
        self.active
            .entry(original)
            .or_default()
            .push((renamed, group, position));
        pushed.originals.push(original);
    }

    /// Marks the top of `group`'s stack as an exception-root barrier: the phi
    /// placed at a root, past which an earlier binding does not reach.
    fn barrier(
        &mut self,
        group: u32,
        version_stacks: &BTreeMap<u32, Vec<SsaVarId>>,
        pushed: &mut Pushed,
    ) {
        let position = version_stacks
            .get(&group)
            .map_or(0, |stack| stack.len().saturating_sub(1));
        self.barriers.entry(group).or_default().push(position);
        pushed.barriers.push(group);
    }

    /// The renamed value `original` names here, if its definition is in scope.
    fn resolve(&self, original: SsaVarId) -> Option<SsaVarId> {
        let &(renamed, group, position) = self.active.get(&original)?.last()?;
        let barrier = self.barriers.get(&group).and_then(|b| b.last()).copied();
        barrier.is_none_or(|b| position > b).then_some(renamed)
    }

    /// Leaves the scope a block opened.
    fn exit(&mut self, pushed: &Pushed) {
        for original in &pushed.originals {
            if let Some(stack) = self.active.get_mut(original) {
                stack.pop();
            }
        }
        for group in &pushed.barriers {
            if let Some(stack) = self.barriers.get_mut(group) {
                stack.pop();
            }
        }
    }
}

/// The mutable state of one rename walk.
#[derive(Debug, Default)]
struct RenameState {
    /// Per group, the reaching definitions in scope, innermost last.
    version_stacks: BTreeMap<u32, Vec<SsaVarId>>,
    /// Per group, the next version number to hand out.
    next_version: BTreeMap<u32, u32>,
    /// Original variable to the variable renaming replaced it with.
    rename_map: BTreeMap<SsaVarId, SsaVarId>,
    /// Result of each cleared phi to the result of the phi placed for it.
    ///
    /// Kept apart from `rename_map`, whose leftover entries rewrite uses
    /// unconditionally: a use of a cleared phi's result is renamed through
    /// [`Identity`] where the placed phi dominates it and through the group
    /// elsewhere. This map only tells the variable record where the value went.
    replaced_phis: BTreeMap<SsaVarId, SsaVarId>,
    /// Where each original variable's renamed definition is in scope.
    identity: Identity,
    /// Stand-ins minted for a group that has definitions but nothing the
    /// function is entered holding.
    ///
    /// A group's version 0 is the value it holds before its first definition
    /// runs. An argument or a local has one; a stack temporary does not - the
    /// evaluation stack is empty at entry - so the rename mints a variable that
    /// nothing defines, purely so the version stack is never empty while the
    /// walk is in progress.
    ///
    /// Recorded because such a value must never become a *use*. It stands for
    /// "nothing has been assigned yet", and a phi operand naming it is a read
    /// of a value that was never produced - which the verifier reports, and
    /// rightly.
    entry_placeholders: BTreeSet<SsaVarId>,
}

/// What one block pushed during renaming, undone when its dominator subtree is
/// left.
#[derive(Debug, Default)]
struct Pushed {
    /// Versions pushed per group.
    versions: BTreeMap<u32, usize>,
    /// Original variables bound in [`Identity`].
    originals: Vec<SsaVarId>,
    /// Groups given an exception-root barrier.
    barriers: Vec<u32>,
}

impl Pushed {
    /// Records one version pushed onto `group`'s stack.
    fn version(&mut self, group: u32) {
        let count = self.versions.entry(group).or_insert(0);
        *count = count.saturating_add(1);
    }
}

/// What a whole-function SSA reconstruction did.
///
/// The reconstruction renames by *rename group* - the values that are versions
/// of one storage location - and leaves a definition in no group as it was. A
/// function whose definitions are all ungrouped, as a native lift's are, is
/// therefore not reconstructed at all: the rebuild succeeds and places nothing.
/// [`Self::reconstructed_nothing`] says so, so a caller that needed the
/// reconstruction asks for the scoped
/// [`SsaFunction::reconstruct_values`](crate::ir::SsaFunction::reconstruct_values)
/// instead of believing the whole-function form ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebuildReport {
    /// Rename groups the reconstruction placed phis for and renamed reads of.
    pub groups: usize,
    /// Definitions - instruction results and phis - in no rename group, which
    /// were left as they were.
    pub ungrouped_definitions: usize,
}

impl RebuildReport {
    /// Whether the function defines values and none of them was
    /// reconstructed, because none belongs to a rename group.
    #[must_use]
    pub const fn reconstructed_nothing(&self) -> bool {
        self.groups == 0 && self.ungrouped_definitions > 0
    }
}

/// Structured SSA rebuilder with explicit per-phase state.
///
/// Each phase of SSA reconstruction is a named method that reads from and writes to
/// distinct fields. This makes the pipeline individually testable and easier to debug
/// compared to a monolithic implementation.
///
/// # Phase Outputs
///
/// | Field Group | Phases | Contents |
/// |-------------|--------|----------|
/// | `var_origins`, `group_types`, `group_origins`, `var_types` | 3-6 | Variable origins, types, and orphan assignments |
/// | `reachable`, `dominance_frontiers`, `successor_map`, `dom_children` | 7 | CFG analysis results |
/// | `defs` | 8 | Definition sites per rename group |
/// | `live_in` | 9 | Liveness information for phi pruning |
/// | `phi_groups` | 11 | Per-block phi group ordering |
pub struct SsaRebuilder<'a, T: Target> {
    /// Mutable reference to the SSA function being rebuilt.
    ssa: &'a mut SsaFunction<T>,

    // ── Phase 2-5: Origin and type collection ──
    /// Maps each variable ID to its [`VariableOrigin`].
    var_origins: BTreeMap<SsaVarId, VariableOrigin>,

    /// Maps each rename group to its canonical SSA type for type preservation.
    group_types: BTreeMap<u32, T::Type>,

    /// Maps each rename group to its [`VariableOrigin`] for phi node creation.
    group_origins: BTreeMap<u32, VariableOrigin>,

    /// Per-variable type map preserving exact types across rebuild.
    /// Needed because stack-derived locals at the same origin can have
    /// different types at different definition points.
    var_types: BTreeMap<SsaVarId, T::Type>,

    // ── Phase 6: CFG analysis ──
    /// The blocks control can reach ([`FunctionRoots::live_blocks`]).
    reachable: BitSet,

    /// Dominance frontiers for each block, used for phi placement.
    dominance_frontiers: Vec<BitSet>,

    /// CFG successor map: block index → list of successors.
    successor_map: BTreeMap<usize, Vec<usize>>,
    /// The terminator relation of the same graph, for the questions a normal
    /// phi operand asks.
    terminator_successor_map: BTreeMap<usize, Vec<usize>>,

    /// Dominator tree: parent block → list of dominated children.
    ///
    /// A forest when some live block is unreachable from the entry: one tree
    /// per root in [`Self::forest_roots`], over disjoint blocks.
    dom_children: BTreeMap<usize, Vec<usize>>,
    /// The root of each tree in [`Self::dom_children`], the entry block first.
    forest_roots: Vec<usize>,

    // ── Phase 7: Definition sites ──
    /// Definition blocks for each rename group (group ID → set of block indices).
    defs: BTreeMap<u32, BTreeSet<usize>>,
    /// Definitions in reachable blocks that belong to no rename group, and
    /// which the rename therefore leaves as they are.
    ungrouped_definitions: usize,

    // ── Phase 8: Liveness ──
    /// Live-in sets for each rename group (group ID → bitset of blocks).
    live_in: BTreeMap<u32, BitSet>,

    // ── Phase 10: Phi placement ──
    /// Per-block phi group ordering: block index → ordered list of rename groups.
    phi_groups: BTreeMap<usize, Vec<u32>>,

    /// Next auto-incrementing group ID for orphan variables and split groups.
    next_group: u32,

    // ── Phase 9: Phi clearing ──
    /// The phis cleared before placement, by block and group; see
    /// [`RenameContext::original_phis`].
    original_phis: BTreeMap<(usize, u32), Option<OriginalPhi>>,
}

impl<'a, T: Target> SsaRebuilder<'a, T> {
    /// Creates a rebuilder over `ssa` with empty phase state.
    ///
    /// The only non-trivial initialization is `next_group`, the counter that
    /// hands out fresh rename-group IDs to orphan and split variables. Group
    /// IDs `0..num_args` are reserved for arguments and `num_args..num_args +
    /// num_locals` for CIL locals, so the counter starts just past that
    /// reserved range to guarantee minted groups never collide with a slot's
    /// canonical group. `saturating_add` keeps a degenerate function with a
    /// near-`u32::MAX` slot count from wrapping the counter to zero.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to rebuild in place; borrowed for the
    ///   lifetime of the rebuilder.
    ///
    /// # Returns
    ///
    /// A rebuilder ready to run [`Self::rebuild`].
    pub fn new(ssa: &'a mut SsaFunction<T>) -> Self {
        let next_group = (ssa.num_args as u32).saturating_add(ssa.num_locals as u32);
        let block_count = ssa.blocks.len();
        Self {
            ssa,
            var_origins: BTreeMap::new(),
            group_types: BTreeMap::new(),
            group_origins: BTreeMap::new(),
            var_types: BTreeMap::new(),
            reachable: BitSet::new(block_count),
            dominance_frontiers: Vec::new(),
            successor_map: BTreeMap::new(),
            terminator_successor_map: BTreeMap::new(),
            dom_children: BTreeMap::new(),
            forest_roots: Vec::new(),
            defs: BTreeMap::new(),
            ungrouped_definitions: 0,
            live_in: BTreeMap::new(),
            phi_groups: BTreeMap::new(),
            next_group,
            original_phis: BTreeMap::new(),
        }
    }

    /// Runs the full 18-phase reconstruction pipeline.
    ///
    /// Executes all stages in order: pre-cleanup, type/origin collection, CFG analysis,
    /// phi placement and renaming, final cleanup with compaction and verification.
    ///
    /// # Returns
    ///
    /// What the reconstruction did ([`RebuildReport`]).
    ///
    /// # Errors
    ///
    /// Returns an error when a read used before its definition cannot be
    /// given one, or when the reconstructed function fails the closing
    /// verification. The caller has checked that the block table is dense
    /// ([`SsaFunction::construct_ssa`]).
    ///
    /// # Panics
    ///
    /// May panic on internal assertion failures (debug builds) if intermediate
    /// invariants are violated, indicating a bug in a preceding pass.
    pub fn rebuild(&mut self) -> Result<RebuildReport> {
        // An empty function has nothing to reconstruct.
        if self.ssa.blocks.is_empty() {
            return Ok(RebuildReport::default());
        }

        // Stage 1: Pre-clean
        self.pre_clean_unreachable(); // Phase 1

        // Stage 2: Type & origin collection
        self.collect_origins(); // Phase 2
        self.propagate_types(); // Phase 3
        self.propagate_instruction_types(); // Phase 4
        self.assign_orphan_origins(); // Phase 5

        // Stage 3: CFG analysis
        self.compute_cfg(); // Phase 6
        self.collect_defs(); // Phase 7
        let report = RebuildReport {
            groups: self.defs.len(),
            ungrouped_definitions: self.ungrouped_definitions,
        };
        self.collect_uses_and_liveness(); // Phase 8

        // Stage 4: Phi placement & rename
        self.clear_all_phis(); // Phase 9
        self.place_phis(); // Phase 10
        self.rename(); // Phase 11

        // Stage 5: Cleanup & compaction
        // A phi operand names a terminator edge; a block the runtime dispatches
        // to has none, so a phi placed there is unrepresentable. Removing it
        // before trivial-phi elimination also stops that pass folding such a
        // phi onto whichever operand happens to be reachable.
        self.ssa.demote_runtime_entry_phis(); // Phase 11b
        self.eliminate_trivial_phis(); // Phase 12
        self.ssa.strip_nops(); // Phase 13
        self.ssa.compact_variables(); // Phase 14
        self.remove_orphan_pops(); // Phase 15
        self.ssa.reindex_variables(); // Phase 16
        self.repair_same_block_future_uses()?; // Phase 16b
        // reindex can cause stale phi operand refs to collide with new IDs
        self.ssa.demote_runtime_entry_phis(); // Phase 16c
        self.eliminate_trivial_phis(); // Phase 17
        // Phi elimination leaves the eliminated results as orphaned rows in
        // `variables` (it cannot drop them itself without breaking the
        // id-is-index invariant). Phase 14's compaction ran before Phase 17, so
        // without this the orphans would survive to `verify()` as
        // `OrphanVariable` errors.
        self.ssa.compact_variables(); // Phase 17a
        self.repair_same_block_future_uses()?; // Phase 17b
        self.ssa.shrink_num_locals(); // Phase 18
        // The phases above renumber variables and rebuild phis without touching
        // the use index, so it no longer describes the function. Restore it
        // before verifying, both so the verification below sees the finished
        // article and so a caller that never opens an edit session still gets a
        // function whose index is exact.
        self.ssa.refresh_def_sites();
        self.ssa.recompute_uses();
        // The rebuild mints instructions of its own - the rescued definitions,
        // the copies a coalesce leaves behind - and they arrive unidentified.
        self.ssa.assign_instr_ids();
        // Phis are placed and then filled, so the join over their operands -
        // their types, and the host's facts - runs here, with the operands in
        // place.
        self.ssa.settle_phi_types();
        self.ssa.refresh_phi_facts();
        self.ssa.refresh_address_taken();

        // Verification
        self.verify()?;
        Ok(report)
    }

    /// Validates the rebuilt SSA, filtering to reachable blocks only.
    ///
    /// Unreachable blocks (e.g., dead CFF dispatcher remnants) may contain stale
    /// variable references that weren't processed by the rename phase.
    fn verify(&self) -> Result<()> {
        let errors = SsaVerifier::new(self.ssa).verify(VerifyLevel::Standard);
        let reachable_errors: Vec<&VerifierError> = errors
            .iter()
            .filter(|e| {
                let block = match e {
                    VerifierError::UndefinedUse { block, .. }
                    | VerifierError::DuplicatePhiOperand { block, .. }
                    | VerifierError::MissingPhiOperand { block, .. }
                    | VerifierError::ExtraPhiOperand { block, .. }
                    | VerifierError::MissingTerminator { block }
                    | VerifierError::PhiWithoutPredecessors { block, .. }
                    | VerifierError::TerminatorNotLast { block, .. }
                    | VerifierError::IntraBlockCycle { block, .. }
                    | VerifierError::PlaceholderVariable { block, .. }
                    | VerifierError::SelfReferentialInstruction { block, .. }
                    | VerifierError::InvalidVectorOperation { block, .. }
                    | VerifierError::InvalidAtomicOperation { block, .. }
                    | VerifierError::InvalidWideArithmetic { block, .. }
                    | VerifierError::InvalidOperandWidths { block, .. }
                    | VerifierError::InvalidNativeOperation { block, .. }
                    | VerifierError::UnassignedInstructionId { block, .. }
                    | VerifierError::ExceptionalOperandAtNonRoot { block, .. }
                    | VerifierError::ExceptionalOperandFromNonThrowingBlock { block, .. }
                    | VerifierError::MissingExceptionalOperand { block, .. }
                    | VerifierError::DuplicateExceptionalOperand { block, .. }
                    | VerifierError::ExceptionalOperandFromAnotherGroup { block, .. }
                    | VerifierError::MisplacedRuntimeValue { block, .. } => Some(*block),
                    VerifierError::DominanceViolation { use_block, .. } => Some(*use_block),
                    VerifierError::DuplicateDefinition { .. }
                    | VerifierError::UnregisteredVariable { .. }
                    // The rebuild restores the use index just above, so a
                    // mismatch reported here is a phase that damaged it and is
                    // never cosmetic.
                    | VerifierError::StaleUseList { .. }
                    // An edge into the entry is never cosmetic, even from an
                    // unreachable block: every predecessor count includes it.
                    | VerifierError::EntryHasPredecessors { .. }
                    // Block-independent and never cosmetic: `variable(id)` indexes
                    // the table directly, so a hole misdirects every later id.
                    | VerifierError::VariableTableNotDense { .. }
                    // An index that names nothing is never cosmetic either: the
                    // block it sits in may be unreachable, but the analyses that
                    // index by it are not told which blocks those are.
                    | VerifierError::BlockTableNotDense { .. }
                    // Identity is function-wide: a duplicate misdirects a host
                    // table wherever the holders sit, and an unstamped
                    // instruction is one the repair did not reach.
                    | VerifierError::DuplicateInstructionId { .. }
                    | VerifierError::DanglingBranchTarget { .. }
                    | VerifierError::PhiPredecessorOutOfRange { .. }
                    | VerifierError::VariableIdOutOfRange { .. }
                    | VerifierError::DefSiteOutOfRange { .. }
                    | VerifierError::UseSiteOutOfRange { .. } => None,
                    // OrphanVariable is cosmetic - the variable exists without a
                    // definition but nothing reads it. Typically v0 entry
                    // variables from stack temp groups that survive compaction.
                    // A variable that *is* read but has no definition is reported
                    // as `UndefinedUse` instead, and is not filtered here.
                    VerifierError::OrphanVariable { .. } => return false,
                    // A malformed exception clause is not an SSA defect and is
                    // not something reconstruction touches: the rebuild rewrites
                    // definitions and phis, never a clause's block ranges. It is
                    // reported where it can be acted on -- by the verifier a
                    // caller runs -- rather than here, where it would fail a
                    // rebuild that had no way to fix it.
                    VerifierError::MalformedExceptionClause { .. } => return false,
                    // A conversion is the front end's, chosen before its
                    // operand's type was known and settled once it is
                    // (`SsaFunction::settle_conversions`), which may come
                    // after reconstruction. The rebuild neither creates nor
                    // changes one, so it is judged where it is settled.
                    VerifierError::InvalidConversion { .. } => return false,
                };
                // Keep errors for reachable blocks (or block-independent
                // errors). A block index past the set names no block at all,
                // which is never cosmetic, so it is kept rather than filtered.
                block.is_none_or(|b| {
                    b >= self.ssa.blocks.len() || self.reachable.contains_checked(b)
                })
            })
            .collect();

        if !reachable_errors.is_empty() {
            // An undefined use names a variable the rebuild itself numbered, so
            // on its own it cannot be traced back to anything a caller holds.
            // What it was, which group reached it, and what reads it can be.
            let msg = reachable_errors
                .iter()
                .map(|e| match e {
                    VerifierError::UndefinedUse {
                        block,
                        instr_idx,
                        var,
                    } => {
                        let variable = self.ssa.variable(*var);
                        let reader = self
                            .ssa
                            .block(*block)
                            .and_then(|b| b.instruction(*instr_idx))
                            .map(|instr| format!("{:?}", instr.op()));
                        format!(
                            "{e} (origin {:?}, group {}, def site {:?}, read by {})",
                            variable.map(|v| v.origin()),
                            self.ssa.rename_group(*var),
                            variable.map(|v| v.def_site()),
                            reader.unwrap_or_else(|| "a phi".to_string())
                        )
                    }
                    _ => e.to_string(),
                })
                .collect::<Vec<_>>()
                .join("; ");
            return Err(Error::new(format!(
                "SSA rebuild validation failed ({} blocks, {} vars): {}",
                self.ssa.blocks().len(),
                self.ssa.variables.len(),
                msg
            )));
        }

        Ok(())
    }

    /// Removes unreachable blocks and simplifies stale phi operands.
    ///
    /// Runs first, so every later phase sees only the reachable graph. Passes
    /// like anti-debug removal make blocks unreachable without cleaning up the
    /// phis in their successors, and an operand naming such a predecessor is an
    /// edge that no longer exists: group joining would tie the phi to the value
    /// it names, and a phi the reconstruction keeps would carry it through.
    ///
    /// Before clearing, this rescues side-effect-free defs from unreachable
    /// blocks whose result variable is still referenced from reachable code.
    /// CFG-modifying passes (CFF unflattening, jump threading, dead branch
    /// removal, block merging) can redirect control flow past a block that
    /// held a loop-invariant `Const` / `LoadToken` / `LoadStaticField`
    /// hoisted there by an earlier LICM pass. Wiping the block would orphan
    /// the use site and leave codegen reading an uninitialized local slot.
    /// Cloning the def into the entry block keeps it dominating every
    /// reachable use.
    fn pre_clean_unreachable(&mut self) {
        let cfg = SsaCfg::from_ssa(self.ssa);
        let reachable = FunctionRoots::of(self.ssa).live_blocks(&cfg);
        let reachable_preds: Vec<BTreeSet<usize>> = (0..self.ssa.blocks.len())
            .map(|block_idx| {
                cfg.block_predecessors(block_idx)
                    .iter()
                    .copied()
                    .filter(|pred| reachable.contains(*pred))
                    .collect()
            })
            .collect();
        drop(cfg);

        rescue_orphaned_pure_defs(self.ssa, &reachable);

        // Clear unreachable blocks
        for block_idx in 0..self.ssa.blocks.len() {
            if !reachable.contains(block_idx)
                && let Some(b) = self.ssa.blocks.get_mut(block_idx)
            {
                b.clear();
            }
        }

        // Remove phi operands from unreachable predecessors and collect
        // trivial phi replacements (phi with 0 or 1 unique operand)
        let mut replacements: Vec<(SsaVarId, SsaVarId)> = Vec::new();
        for block_idx in 0..self.ssa.blocks.len() {
            if !reachable.contains(block_idx) {
                continue;
            }
            // A phi is trivial only when it names a value for every edge that
            // can reach it. An edge with no operand - a predecessor a CFG edit
            // added - carries a value nobody has recorded yet, and collapsing
            // the phi onto its one named operand would claim that edge carries
            // it too.
            let Some(block_preds) = reachable_preds.get(block_idx) else {
                continue;
            };
            let Some(block) = self.ssa.blocks.get_mut(block_idx) else {
                continue;
            };

            // Remove operands from unreachable predecessors
            for phi in block.phi_nodes_mut().iter_mut() {
                // An operand naming a block that is not reachable describes a
                // path that cannot be taken. An out-of-range block index is by
                // definition unreachable, so the checked read answers it too.
                phi.retain_operands(|operand| reachable.contains_checked(operand.predecessor()));
            }

            // Inline trivial phis (0 or 1 unique operand value)
            block.phi_nodes_mut().retain(|phi| {
                let operands = phi.operands();
                let Some(first_op) = operands.first() else {
                    return false; // Remove empty phi
                };
                let covers_every_edge = block_preds
                    .iter()
                    .all(|pred| operands.iter().any(|op| op.predecessor() == *pred));
                if !covers_every_edge {
                    return true;
                }
                let first = first_op.value();
                // Check if all operands resolve to the same value
                if operands
                    .iter()
                    .all(|op| op.value() == first || op.value() == phi.result())
                {
                    replacements.push((phi.result(), first));
                    return false;
                }
                true
            });
        }

        // Apply replacements: substitute phi result uses with the single operand
        if !replacements.is_empty() {
            let direct: BTreeMap<SsaVarId, SsaVarId> = replacements.into_iter().collect();
            // Resolve chains before substituting. A phi inlined to `first` can
            // name another phi being inlined in the same pass, and applying the
            // map entry-by-entry leaves that intermediate in place - a use then
            // names a phi this loop has already removed, which is `UndefinedUse`.
            // Iterating a `BTreeMap` makes it order-dependent rather than
            // reliably wrong, so it survives wherever key order happens to
            // resolve the chain.
            let replacement_map: BTreeMap<SsaVarId, SsaVarId> = direct
                .iter()
                .filter_map(|(from, to)| {
                    let mut current = *to;
                    let mut visited: BTreeSet<SsaVarId> = BTreeSet::new();
                    visited.insert(*from);
                    while let Some(&next) = direct.get(&current) {
                        if !visited.insert(current) {
                            break;
                        }
                        current = next;
                    }
                    (current != *from).then_some((*from, current))
                })
                .collect();
            for block in self.ssa.blocks.iter_mut() {
                for instr in block.instructions_mut() {
                    for (&old_var, &new_var) in &replacement_map {
                        instr.op_mut().replace_uses(old_var, new_var);
                    }
                }
                for phi in block.phi_nodes_mut() {
                    for (&old_var, &new_var) in &replacement_map {
                        for op in phi.operands_mut() {
                            if op.value() == old_var {
                                // The value changes; which kind of edge the
                                // operand stands for does not.
                                op.set_value(new_var);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Builds the `var_origins`, `group_origins`, `group_types`, and
    /// `var_types` maps that later phases read.
    ///
    /// The subtle step is `group_origins`: when several variables share a rename
    /// group, a `Local` or `Argument` origin is preferred over `Phi`. A group a
    /// host or a pass formed can hold a `Local`-origin variable alongside
    /// `Phi`-origin ones; the group must keep the `Local` origin so codegen
    /// allocates a CIL local slot rather than a stack temporary. Argument and local slots are also registered
    /// unconditionally so a group with no surviving variable still has an origin.
    /// Finally `next_group` is lifted above every existing group ID so freshly
    /// minted groups cannot collide with one already in use.
    fn collect_origins(&mut self) {
        self.var_origins = self
            .ssa
            .variables
            .iter()
            .map(|v| (v.id(), v.origin()))
            .collect();

        // Build group_origins from existing rename_groups.
        // Prefer Local/Argument origins over Phi - a group can hold Local-origin
        // and Phi-origin variables together, and it must retain the Local origin
        // so the codegen allocates a CIL local slot instead of a stack temporary.
        for var in &self.ssa.variables {
            let group = self.ssa.rename_group(var.id());
            if group != u32::MAX {
                self.group_origins
                    .entry(group)
                    .and_modify(|existing| {
                        if matches!(existing, VariableOrigin::Phi)
                            && matches!(
                                var.origin(),
                                VariableOrigin::Local(_) | VariableOrigin::Argument(_)
                            )
                        {
                            *existing = var.origin();
                        }
                    })
                    .or_insert(var.origin());
            }
        }

        // Also register arg/local groups
        let num_args = self.ssa.num_args;
        let slots = (0..num_args)
            .map(|i| VariableOrigin::Argument(i as u16))
            .chain((0..self.ssa.num_locals).map(|i| VariableOrigin::Local(i as u16)));
        for origin in slots {
            if let Some(group) = origin.canonical_rename_group(num_args) {
                self.group_origins.entry(group).or_insert(origin);
            }
        }

        // Track the best type for each group (prefer non-unknown types)
        // and per-variable types
        for var in &self.ssa.variables {
            let var_type = var.var_type();
            if !T::is_unknown(var_type) {
                let group = self.ssa.rename_group(var.id());
                if group != u32::MAX {
                    self.group_types.entry(group).or_insert(var_type.clone());
                }
                self.var_types.insert(var.id(), var_type.clone());
            }
        }

        // Update next_group to be above all existing groups. Asked through
        // [`SsaFunction::max_rename_group`] so an ungrouped variable's sentinel
        // cannot answer for the highest group: reading it as "no groups" skips
        // the lift entirely, and the orphan groups minted later then start at
        // zero and collide with every real group in turn.
        if let Some(max_existing) = self.ssa.max_rename_group() {
            self.next_group = self.next_group.max(max_existing.saturating_add(1));
        }
    }

    /// Propagates a known slot type onto the group of each `LoadLocal` /
    /// `LoadArg` destination that still lacks one.
    ///
    /// Runs after [`Self::collect_origins`] has seeded `group_types` from typed
    /// variables. A load's result variable is often type-unknown on its own, but
    /// the arg or local slot it reads has a canonical type; copying that type to
    /// the destination's group gives the reconstructed variable a precise type
    /// instead of falling through to `unknown`. Existing group types are never
    /// overwritten - the `contains_key` guard makes the first known type win, so
    /// a slot's canonical type is not clobbered by a later load.
    fn propagate_types(&mut self) {
        for block in self.ssa.blocks.iter() {
            for instr in block.instructions() {
                match instr.op() {
                    SsaOp::LoadLocal { dest, local_index } => {
                        let dest_group = self.ssa.rename_group(*dest);
                        if dest_group != u32::MAX && !self.group_types.contains_key(&dest_group) {
                            let local_group = local_group(self.ssa.num_args, *local_index);
                            if let Some(local_type) = self.group_types.get(&local_group).cloned() {
                                self.group_types.insert(dest_group, local_type);
                            }
                        }
                    }
                    SsaOp::LoadArg { dest, arg_index } => {
                        let dest_group = self.ssa.rename_group(*dest);
                        if dest_group != u32::MAX && !self.group_types.contains_key(&dest_group) {
                            let arg_group = argument_group(*arg_index);
                            if let Some(arg_type) = self.group_types.get(&arg_group).cloned() {
                                self.group_types.insert(dest_group, arg_type);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Infers types for instructions whose destination variable has no type in `group_types`.
    ///
    /// Runs after `propagate_types()` and before `assign_orphan_origins()`. For each
    /// instruction with a destination variable, if the variable's origin has no entry
    /// in `group_types`, checks the instruction's `result_type` first (set during SSA
    /// construction with full TypeContext), then falls back to `SsaOp::infer_result_type()`.
    fn propagate_instruction_types(&mut self) {
        for block in self.ssa.blocks.iter() {
            for instr in block.instructions() {
                for dest in instr.op().defs() {
                    let group = self.ssa.rename_group(dest);
                    if group != u32::MAX && !self.group_types.contains_key(&group) {
                        // Priority: instruction result_type (from converter with TypeContext)
                        // > op structural inference
                        if let Some(rt) = instr.result_type()
                            && !T::is_unknown(rt)
                        {
                            self.group_types.insert(group, rt.clone());
                            continue;
                        }
                        if let Some(inferred) = instr.op().infer_result_type() {
                            self.group_types.insert(group, inferred);
                        }
                    }
                }
            }
        }
    }

    /// Puts the names a phi calls equal into one rename group.
    ///
    /// A phi states that its result and its operands are one value seen on
    /// different paths. For a value with a storage location that is already
    /// recorded elsewhere - the slot - but a value without one is held together
    /// by the phi and nothing else. Reconstruction re-derives phis from the
    /// definitions that reach each group, so unless those names share a group
    /// the merge is not expressible and simply does not come back: the operands
    /// lose their only consumer and the next dead-code pass removes the
    /// instructions that produced them.
    ///
    /// Filling in only the names that have no group yet, which is what the
    /// propagation below does, does not achieve this. Every name that went
    /// through an earlier reconstruction already carries a group of its own -
    /// each orphan is given one - so there is no blank left to fill and the
    /// equality is dropped. This is what cost `DemoCharOperations` its
    /// `consonantCount++`: block merging rebuilt the function, the merge at the
    /// increment's join was not re-derived, and the increment died as unread.
    ///
    /// Groups are joined, never overwritten, and a merge that spans two
    /// *storage* locations is left alone: choosing one slot would stop the
    /// other's definitions reaching the merge, and an argument's slot cannot
    /// move at all. Such a merge needs a copy on each edge to become
    /// reconstructible, which is a decision for whoever is rewriting the graph.
    fn unify_phi_groups(&mut self) {
        let num_args = self.ssa.num_args;

        // The groups a phi ties together, one entry per phi.
        let mut ties: Vec<Vec<u32>> = Vec::new();
        for block in self.ssa.blocks.iter() {
            for phi in block.phi_nodes() {
                let mut groups = vec![self.ssa.rename_group(phi.result())];
                groups.extend(
                    phi.operands()
                        .iter()
                        .map(|o| self.ssa.rename_group(o.value())),
                );
                groups.retain(|group| *group != u32::MAX);
                groups.sort_unstable();
                groups.dedup();
                if groups.len() > 1 {
                    ties.push(groups);
                }
            }
        }
        if ties.is_empty() {
            return;
        }

        // Which groups name a storage location, and so cannot be relabeled.
        let mut anchored: BTreeSet<u32> = BTreeSet::new();
        for var in &self.ssa.variables {
            if let Some(group) = var.origin().canonical_rename_group(num_args) {
                anchored.insert(group);
            }
        }

        // Union-find over groups, joining each phi's set onto one representative.
        let mut parent: BTreeMap<u32, u32> = BTreeMap::new();
        for tie in &ties {
            for group in tie {
                parent.entry(*group).or_insert(*group);
            }
        }
        for tie in ties {
            // Two distinct slots in one merge is the case that needs copies.
            if tie.iter().filter(|group| anchored.contains(group)).count() > 1 {
                continue;
            }
            // A slot in the set is the representative, so the storage location
            // survives and the rest join it.
            let representative = tie
                .iter()
                .find(|group| anchored.contains(group))
                .copied()
                .unwrap_or_else(|| tie.first().copied().unwrap_or(u32::MAX));
            if representative == u32::MAX {
                continue;
            }
            for group in tie {
                union_groups(&mut parent, representative, group);
            }
        }

        let relabel: Vec<(SsaVarId, u32)> = self
            .ssa
            .variables
            .iter()
            .filter_map(|var| {
                let group = self.ssa.rename_group(var.id());
                let root = find_group(&parent, group)?;
                (root != group).then_some((var.id(), root))
            })
            .collect();
        for (var, group) in relabel {
            self.ssa.set_rename_group(var, group);
        }
    }

    /// Assigns origins and groups to orphan variables not in self.variables.
    ///
    /// Orphan variables are created by passes. They need origins and groups so they
    /// can be renamed. PHI origins/groups are propagated first, then remaining orphans
    /// get `Phi` origin with unique group IDs.
    ///
    /// A phi whose result has no group yet takes the group its origin names (an
    /// argument or local slot) before propagating. Renaming versions only grouped
    /// variables, so a loop phi left ungrouped would never see its back-edge
    /// value pushed onto a version stack, and the fill would replace that operand
    /// with the slot's entry value - a loop that forgets its own iterations.
    fn assign_orphan_origins(&mut self) {
        self.unify_phi_groups();

        // First pass: propagate PHI origins/groups to ALL operands.
        // This ensures that phi operands use the same origin as the phi during rename,
        // so they end up on the same version stack and properly fill phi operands.
        // Iterated to a fixpoint, because an operand can itself be a phi result
        // whose own operands only receive a group once it has one; each round
        // either assigns a group to a previously ungrouped variable or stops.
        let num_args = self.ssa.num_args;
        for _ in 0..=self.ssa.variables.len() {
            let mut group_propagations: Vec<(SsaVarId, u32)> = Vec::new();
            for block in self.ssa.blocks.iter() {
                for phi in block.phi_nodes() {
                    let phi_origin = self
                        .var_origins
                        .get(&phi.result())
                        .copied()
                        .unwrap_or_else(|| phi.origin());

                    let mut phi_group = self.ssa.rename_group(phi.result());
                    if phi_group == u32::MAX
                        && let Some(group) = phi_origin.canonical_rename_group(num_args)
                    {
                        phi_group = group;
                        group_propagations.push((phi.result(), group));
                    }

                    // Assign the PHI's origin to its result if orphan
                    self.var_origins.entry(phi.result()).or_insert(phi_origin);

                    // Assign the phi's origin and group to ORPHAN operands only.
                    // IMPORTANT: Do NOT overwrite existing origins for non-orphan variables.
                    for operand in phi.operands() {
                        let op_var = operand.value();
                        self.var_origins.entry(op_var).or_insert(phi_origin);
                        // Propagate the phi's group to orphan operands
                        if phi_group != u32::MAX && self.ssa.rename_group(op_var) == u32::MAX {
                            group_propagations.push((op_var, phi_group));
                        }
                    }
                }
            }
            if group_propagations.is_empty() {
                break;
            }
            for (var_id, group) in group_propagations {
                if self.ssa.rename_group(var_id) == u32::MAX {
                    self.ssa.set_rename_group(var_id, group);
                }
            }
        }

        // Second pass: assign Phi origin and unique group IDs to remaining orphan
        // variables. Orphans must participate in rename (with version stacks) so that
        // their defs and uses get proper new IDs. Each orphan variable gets its own
        // group and unique Local(next_idx) origin.
        // Collect orphan var IDs first to avoid borrow conflicts.
        let mut orphan_vars: Vec<(SsaVarId, Option<T::Type>)> = Vec::new();
        for block in self.ssa.blocks.iter() {
            for instr in block.instructions() {
                instr.op().for_each_use(|use_var| {
                    if !self.var_origins.contains_key(&use_var) {
                        orphan_vars.push((use_var, None));
                    }
                });
                for dest in instr.defs() {
                    if !self.var_origins.contains_key(&dest) {
                        // Prefer instruction result_type (from converter), fall back
                        // to structural inference
                        let inferred_type = instr
                            .result_type()
                            .filter(|rt| !T::is_unknown(rt))
                            .cloned()
                            .or_else(|| instr.op().infer_result_type());
                        orphan_vars.push((dest, inferred_type));
                    }
                }
            }
        }

        for (var_id, inferred_type) in orphan_vars {
            self.var_origins
                .entry(var_id)
                .or_insert(VariableOrigin::Phi);
            if self.ssa.rename_group(var_id) == u32::MAX {
                let group = self.next_group;
                self.next_group = self.next_group.saturating_add(1);
                self.ssa.set_rename_group(var_id, group);
                self.group_origins.insert(group, VariableOrigin::Phi);
                if let Some(inferred) = inferred_type {
                    self.group_types.entry(group).or_insert(inferred);
                }
            }
        }

        // No num_locals inflation - orphans use Phi origin
    }

    /// Computes reachability, then the exception-aware dominator tree,
    /// dominance frontiers, successor map, and dominator-tree children.
    ///
    /// Runs in two passes. The first computes reachability from the raw SSA and
    /// clears unreachable blocks, so their outgoing edges cannot pollute the
    /// predecessor graph and skew dominance. The second rebuilds the flow view
    /// from the complete relation, in which a handler or filter entry is
    /// reachable wherever its guard is: one dominator tree and one frontier
    /// pass then cover every block the entry reaches. Per-handler dominator
    /// trees merged together would give contradictory `dominates` answers for a
    /// block reachable from two roots; one tree over the complete relation
    /// cannot.
    ///
    /// A live block the entry does not reach - under a handler whose guard is
    /// itself dead, or a preservation root, which no edge reaches - gets a tree
    /// of its own root, over the blocks no earlier tree holds. Those blocks'
    /// dominators are among them (a dominator an earlier tree held would put
    /// the block in that tree too), so the trees are disjoint and never answer
    /// one block two ways. Before this, such a region had no tree: the rename
    /// never visited its blocks, and a phi placed there kept the placeholder it
    /// was created with.
    ///
    /// The successor lists keep the exceptional edges, because phi placement
    /// prunes on liveness and the terminator relation alone would drop a phi
    /// the exception-aware frontier legitimately demands.
    fn compute_cfg(&mut self) {
        // First pass: compute reachability from the raw SSA
        {
            let cfg = SsaCfg::from_ssa(self.ssa);
            self.reachable = FunctionRoots::of(self.ssa).live_blocks(&cfg);
        }

        // Clear unreachable blocks to remove phantom CFG edges.
        // Without this, unreachable blocks (e.g., dead crash code after anti-debug
        // removal) keep outgoing edges that pollute the predecessor graph. This
        // causes incorrect dominator computation and stale variable references
        // during rename.
        for block_idx in 0..self.ssa.blocks.len() {
            if !self.reachable.contains(block_idx)
                && let Some(b) = self.ssa.blocks.get_mut(block_idx)
            {
                b.clear();
            }
        }

        // Second pass: rebuild the flow view from the cleaned-up SSA.
        //
        // Rooted at the complete relation, not the terminator one: a handler or
        // filter entry is reachable there wherever its guard is, so the entry's
        // tree and frontiers cover it. Rooting at the entry block alone would
        // leave every handler unreachable, and patching that with per-handler
        // dominator trees merged in after the fact gives contradictory
        // `dominates` answers for any block reachable from two roots.
        {
            let eh = SsaCfg::from_ssa(self.ssa);
            let block_count = self.ssa.blocks.len();
            let mut frontiers: Vec<BitSet> = vec![BitSet::new(block_count); block_count];
            let mut claimed = BitSet::new(block_count);
            for root in FunctionRoots::of(self.ssa).entries() {
                if !self.reachable.contains_checked(root) || claimed.contains(root) {
                    continue;
                }
                let dom_tree = compute_dominators(&eh, NodeId::new(root));
                let tree_frontiers = compute_dominance_frontiers(&eh, &dom_tree);
                let region: Vec<usize> = self
                    .reachable
                    .iter()
                    .filter(|&block| {
                        !claimed.contains(block) && dom_tree.is_reachable(NodeId::new(block))
                    })
                    .collect();
                for &block in &region {
                    claimed.insert(block);
                }
                for &block in &region {
                    self.dom_children.insert(
                        block,
                        dom_tree
                            .children(NodeId::new(block))
                            .iter()
                            .map(|n| n.index())
                            .filter(|&child| child < block_count && self.reachable.contains(child))
                            .collect(),
                    );
                    if let (Some(row), Some(tree_row)) =
                        (frontiers.get_mut(block), tree_frontiers.get(block))
                    {
                        row.union_with(tree_row);
                    }
                }
                self.forest_roots.push(root);
            }
            self.dominance_frontiers = frontiers;

            // Successors include the synthesized exceptional edges. Phi
            // placement prunes on liveness, and terminator-only liveness would
            // prune a phi the exception-aware frontier legitimately demands.
            let successor_lists = eh.to_successor_lists();
            for i in self.reachable.iter() {
                self.successor_map
                    .insert(i, successor_lists.get(i).cloned().unwrap_or_default());
                // The narrow relation too. A normal phi operand names a
                // transfer a terminator makes, and which transfers those are is
                // not readable from the terminator alone: a `leave` runs the
                // finallys of every region it exits before it reaches its own
                // target.
                self.terminator_successor_map
                    .insert(i, eh.terminator_successors(i).to_vec());
            }
        }
    }

    /// Collects the per-group definition blocks that drive phi placement,
    /// gathered from reachable blocks before the old phis are cleared.
    ///
    /// Arguments and original CIL locals are seeded as defined at the entry
    /// block, since they hold caller-supplied or default-initialized values
    /// there. The non-obvious part is that a **phi result** is recorded as a
    /// definition of its group in its block, exactly as Cytron's algorithm
    /// treats it: phi placement is a fixpoint over the iterated dominance
    /// frontier of the definition set, so a phi can be what justifies placing
    /// the next one. Omitting phi results means a group whose only reaching
    /// definition in some region was a phi contributes no block there, no phi is
    /// re-placed where one is still needed, and the uses of the dissolved result
    /// are left naming an undefined variable.
    fn collect_defs(&mut self) {
        // Arguments are always defined at entry (block 0) - they have values from the caller.
        for i in 0..self.ssa.num_args {
            self.defs
                .entry(argument_group(i as u16))
                .or_default()
                .insert(0);
        }
        // Locals are default-initialized at entry.
        for i in 0..self.ssa.num_locals {
            let group = local_group(self.ssa.num_args, i as u16);
            self.defs.entry(group).or_default().insert(0);
        }

        // Collect defs from instructions and phi results using group IDs.
        //
        // A phi result is a definition of its group in its block, and it has to
        // seed the def set like any other: `place_phis` derives phi placement
        // from the iterated dominance frontier *of these blocks*, and
        // `clear_all_phis` has already dissolved the phis themselves by then.
        //
        // Omitting them means a group whose only definition reaching some region
        // was a phi contributes no block there, so no phi is re-placed where one
        // is still needed - and the uses of the dissolved result are left naming
        // a variable nothing defines: `UndefinedUse`, on IR that was valid on
        // entry to the rebuild.
        //
        // Cytron places a phi wherever a definition's dominance frontier reaches,
        // and treats a phi as a definition for exactly this reason: phi placement
        // is a fixpoint, so a phi can be what justifies the next one.
        for block in self.ssa.blocks.iter() {
            let block_idx = block.id();
            if !self.reachable.contains_checked(block_idx) {
                continue;
            }
            for phi in block.phi_nodes() {
                let group = self.ssa.rename_group(phi.result());
                if group == u32::MAX {
                    self.ungrouped_definitions = self.ungrouped_definitions.saturating_add(1);
                } else {
                    self.defs.entry(group).or_default().insert(block_idx);
                }
            }
            for instr in block.instructions() {
                for dest in instr.defs() {
                    let group = self.ssa.rename_group(dest);
                    if group == u32::MAX {
                        self.ungrouped_definitions = self.ungrouped_definitions.saturating_add(1);
                    } else {
                        self.defs.entry(group).or_default().insert(block_idx);
                    }
                }
            }
        }
    }

    /// Collects per-group use sites and computes live-in sets for pruned phi
    /// placement.
    ///
    /// Pruned SSA places a phi only where a group is live-in, so the liveness
    /// this phase computes gates every later phi. Two use sources are easy to
    /// miss. First, a phi operand is a use recorded against the **predecessor**
    /// edge it arrives on, not the head of the phi's own block - the value must
    /// be live at the end of that edge - otherwise a group consumed only by phi
    /// operands is live-in nowhere and its phi is never placed, leaving those
    /// operands naming an undefined variable. Second, `LoadLocal` / `LoadArg`
    /// are implicit uses of their slot's group, counted only when the loaded
    /// destination is actually consumed by a non-`Nop` instruction.
    fn collect_uses_and_liveness(&mut self) {
        let block_count = self.ssa.blocks.len();

        // Pre-compute which variables are consumed by non-Nop instructions.
        let mut consumed_vars = VarSet::new(self.ssa.var_id_bound());
        for block in self.ssa.blocks.iter() {
            if !self.reachable.contains_checked(block.id()) {
                continue;
            }
            for instr in block.instructions() {
                if !matches!(instr.op(), SsaOp::Nop) {
                    instr.op().for_each_use(|use_var| {
                        consumed_vars.insert(use_var);
                    });
                }
            }
        }

        let mut use_sites: BTreeMap<u32, BitSet> = BTreeMap::new();
        for block in self.ssa.blocks.iter() {
            let block_idx = block.id();
            if !self.reachable.contains_checked(block_idx) {
                continue;
            }
            // A phi operand is a use, recorded against the **predecessor** it
            // arrives on: the value must be live at the end of that edge, not at
            // the head of this block.
            //
            // Without this, a group consumed only by phi operands is live-in
            // nowhere, so `place_pruned_phis` - which places a phi only where the
            // group is live-in - places none for it. `clear_all_phis` has already
            // dissolved the original by then, leaving the phi-operand uses naming
            // a variable nothing defines: `UndefinedUse`. Recording phi results in
            // `collect_defs` alone does not help, because this liveness gate still
            // suppresses the placement.
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    let group = self.ssa.rename_group(operand.value());
                    if group == u32::MAX {
                        continue;
                    }
                    let pred = operand.predecessor();
                    if !self.reachable.contains_checked(pred) {
                        continue;
                    }
                    use_sites
                        .entry(group)
                        .or_insert_with(|| BitSet::new(block_count))
                        .insert(pred);
                }
            }
            for instr in block.instructions() {
                instr.op().for_each_use(|use_var| {
                    let group = self.ssa.rename_group(use_var);
                    if group != u32::MAX {
                        use_sites
                            .entry(group)
                            .or_insert_with(|| BitSet::new(block_count))
                            .insert(block_idx);
                    }
                });
                // Track implicit uses from LoadLocal/LoadArg
                match instr.op() {
                    SsaOp::LoadLocal { dest, local_index } if consumed_vars.contains(*dest) => {
                        let group = local_group(self.ssa.num_args, *local_index);
                        use_sites
                            .entry(group)
                            .or_insert_with(|| BitSet::new(block_count))
                            .insert(block_idx);
                    }
                    SsaOp::LoadArg { dest, arg_index } if consumed_vars.contains(*dest) => {
                        let group = argument_group(*arg_index);
                        use_sites
                            .entry(group)
                            .or_insert_with(|| BitSet::new(block_count))
                            .insert(block_idx);
                    }
                    _ => {}
                }
            }
        }

        // Build successors list for liveness analysis
        let successors_list: Vec<Vec<usize>> = (0..block_count)
            .map(|i| self.successor_map.get(&i).cloned().unwrap_or_default())
            .collect();

        // The definitions liveness kills on, which are the *instruction* ones
        // only.
        //
        // `self.defs` also counts a phi result as a definition of its group,
        // and placement needs that - Cytron's fixpoint is over the iterated
        // dominance frontier of the definition set, and a phi is what justifies
        // the next one. Liveness must not, because these phis are about to be
        // cleared and re-placed: a phi at a block where a group's only reads
        // are further down would kill the group there, the group would not be
        // live-in, and the pruned placement would put no phi back. The reads
        // below then resolve to whatever reached the *old* phi - for a loop,
        // the value from before it, which is a miscompile rather than a missing
        // optimization.
        let mut defs_for_liveness: BTreeMap<u32, BitSet> = BTreeMap::new();
        for block in self.ssa.blocks.iter() {
            let block_idx = block.id();
            if !self.reachable.contains_checked(block_idx) {
                continue;
            }
            for instr in block.instructions() {
                for dest in instr.op().defs() {
                    let group = self.ssa.rename_group(dest);
                    if group != u32::MAX {
                        defs_for_liveness
                            .entry(group)
                            .or_insert_with(|| BitSet::new(block_count))
                            .insert(block_idx);
                    }
                }
            }
        }

        self.live_in = liveness::compute_live_in_blocks(
            &defs_for_liveness,
            &use_sites,
            &successors_list,
            block_count,
        );

        // A value a handler or filter reads arrives from wherever the throw
        // happened: anywhere in a block that throws to it, before that block's
        // own writes included. So a group live where a handler is entered is
        // live where each block that throws to it is entered - which the uses
        // above cannot show, because the handler's phi and its exceptional
        // operands are only placed after this. Without it a loop that re-enters
        // a protected region had no phi at its head for a variable the region
        // assigns: a throw on the second iteration was answered with the
        // variable's value from before the loop. Repeated until stable, since
        // a newly live block can itself sit in another handler's region.
        let roots = FunctionRoots::of(self.ssa);
        if roots.has_flow_roots() {
            loop {
                let mut changed = false;
                for root in roots.flow_roots() {
                    let sources = roots.exceptional_predecessors(root.block);
                    for (group, live_blocks) in &self.live_in {
                        if !live_blocks.contains_checked(root.block) {
                            continue;
                        }
                        let sites = use_sites
                            .entry(*group)
                            .or_insert_with(|| BitSet::new(block_count));
                        for &source in sources {
                            if source < block_count && !sites.contains(source) {
                                sites.insert(source);
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
                self.live_in = liveness::compute_live_in_blocks(
                    &defs_for_liveness,
                    &use_sites,
                    &successors_list,
                    block_count,
                );
            }
        }
    }

    /// Clears the phi nodes this rebuild will re-place, before fresh placement.
    ///
    /// A phi whose result carries **no rename group** (`u32::MAX`) is kept,
    /// because nothing downstream can restore it:
    ///
    /// - [`Self::assign_orphan_origins`] gives a phi result the group its
    ///   origin names, and [`Self::unify_phi_groups`] only joins groups that
    ///   exist, so a result whose origin names no slot is never given one.
    /// - [`Self::place_phis`] places phis per rename group, so it will not
    ///   re-create a phi belonging to none.
    /// - [`Self::rename`] skips ungrouped variables outright, so no rename-map
    ///   entry is produced and [`Renamer::apply_rename_map`] has nothing to rewrite
    ///   the surviving uses with.
    ///
    /// Clearing it therefore deletes a definition whose uses remain, and the
    /// function reaches the verifier with `UndefinedUse` on each one.
    ///
    /// The reconstruction is SSA *for rename groups*. A phi outside every
    /// group is not part of that model, so it is not this pass's to destroy.
    fn clear_all_phis(&mut self) {
        for (block_idx, block) in self.ssa.blocks.iter().enumerate() {
            for phi in block.phi_nodes() {
                let group = self.ssa.rename_group(phi.result());
                if group == u32::MAX || !self.defs.contains_key(&group) {
                    continue;
                }
                let original = OriginalPhi {
                    result: phi.result(),
                    operands: phi
                        .operands()
                        .iter()
                        .map(|operand| (operand.predecessor(), operand.value()))
                        .collect(),
                };
                self.original_phis
                    .entry((block_idx, group))
                    .and_modify(|slot| *slot = None)
                    .or_insert(Some(original));
            }
        }
        let ungrouped: BTreeSet<SsaVarId> = self
            .ssa
            .blocks
            .iter()
            .flat_map(|block| block.phi_nodes().iter().map(|phi| phi.result()))
            .filter(|result| {
                let group = self.ssa.rename_group(*result);
                group == u32::MAX || !self.defs.contains_key(&group)
            })
            .collect();
        for block in self.ssa.blocks.iter_mut() {
            block
                .phi_nodes_mut()
                .retain(|phi| ungrouped.contains(&phi.result()));
        }
    }

    /// Places phi nodes for every group using iterated dominance frontiers and
    /// records the per-block group ordering in `phi_groups`.
    ///
    /// Delegates the frontier iteration to [`place_pruned_phis`], but first
    /// filters out groups whose only definition is the entry block: such a group
    /// has a single reaching value everywhere and never needs a merge, so
    /// placing a phi for it would only create trivial phis for the finalize
    /// stage to strip. A `leave_target_fn` is supplied so exception-handler phi
    /// placement can follow `Leave` edges. The returned `(block, group)`
    /// placements are accumulated into `phi_groups`, which the rename phase reads
    /// to associate each phi with its group when several groups share `Phi`
    /// origin.
    fn place_phis(&mut self) {
        // Leave target resolver for exception handler phi placement
        let leave_target_fn = |block_idx: usize, blocks: &[SsaBlock<T>]| -> Option<usize> {
            blocks
                .get(block_idx)
                .and_then(|block| match block.control_terminator() {
                    Some(SsaOp::Leave { target }) => Some(*target),
                    _ => None,
                })
        };

        let block_count = self.ssa.blocks.len();

        // Convert defs to BTreeMap<u32, BitSet> and filter to skip single-entry-only groups
        let mut filtered_defs: BTreeMap<u32, BitSet> = self
            .defs
            .iter()
            .filter(|(_, def_blocks)| !(def_blocks.len() == 1 && def_blocks.contains(&0)))
            .map(|(k, v)| {
                let mut bs = BitSet::new(block_count);
                for &b in v {
                    bs.insert(b);
                }
                (*k, bs)
            })
            .collect();

        // A phi at a handler or filter entry defines its group there, exactly as
        // an assignment would: every block the handler `leave`s to, and every
        // block in its dominance frontier, merges the handler's value with what
        // the other paths carry. Decided before the frontier iteration and fed
        // into it, so those merges are placed by the one iteration. Placed after
        // it instead, `try { x = 7; } catch { } return x;` read the try's `x` on
        // the path through the catch.
        let root_placements = self.exception_root_placements();
        for (root, group) in &root_placements {
            filtered_defs
                .entry(*group)
                .or_insert_with(|| BitSet::new(block_count))
                .insert(*root);
        }

        let group_origins = self.group_origins.clone();
        let num_args = self.ssa.num_args;

        let placements = place_pruned_phis(
            &mut self.ssa.blocks,
            PhiPlacementConfig {
                defs: &filtered_defs,
                live_in: &self.live_in,
                dominance_frontiers: &self.dominance_frontiers,
                reachable: Some(&self.reachable),
                group_filter: &|_| true,
                group_to_origin: &|group| {
                    group_origins
                        .get(&group)
                        .copied()
                        .unwrap_or(if (group as usize) < num_args {
                            VariableOrigin::Argument(group as u16)
                        } else {
                            VariableOrigin::Phi
                        })
                },
                leave_target_fn: Some(&leave_target_fn),
            },
        );

        // Build per-block phi group mapping from placement info
        for (block_idx, group) in placements {
            self.phi_groups.entry(block_idx).or_default().push(group);
        }

        self.place_exception_root_phis(&root_placements);
    }

    /// Places a phi at each handler and filter entry for every group its
    /// protected region writes.
    ///
    /// Dominance frontiers cannot answer this. A frontier marks where two
    /// *paths* meet, and a handler whose region is a single block is dominated
    /// by that block - no frontier - while still merging what the storage held
    /// on entry to the region with everything the region assigned before the
    /// throw. The criterion for an exception root is therefore "the region
    /// writes this storage", and the merge is over program points rather than
    /// edges, which is what [`PhiEdge::Exceptional`](crate::ir::phi::PhiEdge::Exceptional) operands express.
    ///
    /// Pruned the same way as ordinary placement: a group that is not live at
    /// the root needs no merge there.
    fn place_exception_root_phis(&mut self, placements: &[(usize, u32)]) {
        let num_args = self.ssa.num_args;
        let group_origins = self.group_origins.clone();
        for (root, group) in placements {
            // The frontier iteration places a phi at a root its writes reach
            // through a frontier too; one phi per group and block.
            let placed = self
                .phi_groups
                .get(root)
                .is_some_and(|groups| groups.contains(group));
            if placed {
                continue;
            }
            let origin =
                group_origins
                    .get(group)
                    .copied()
                    .unwrap_or(if (*group as usize) < num_args {
                        VariableOrigin::Argument(*group as u16)
                    } else {
                        VariableOrigin::Phi
                    });
            if let Some(block) = self.ssa.blocks.get_mut(*root) {
                block.add_phi(PhiNode::new(SsaVarId::PLACEHOLDER, origin));
                self.phi_groups.entry(*root).or_default().push(*group);
            }
        }
    }

    /// The `(root, group)` pairs that need a phi at a handler or filter entry:
    /// the group is written in a block that throws to the root, and is live
    /// there.
    fn exception_root_placements(&self) -> Vec<(usize, u32)> {
        let roots = FunctionRoots::of(self.ssa);
        if !roots.has_flow_roots() {
            return Vec::new();
        }
        let mut placements = Vec::new();
        for root in roots.flow_roots() {
            let sources = roots.exceptional_predecessors(root.block);
            if sources.is_empty() {
                continue;
            }
            for (group, def_blocks) in &self.defs {
                let written_in_region = def_blocks.iter().any(|block| sources.contains(block));
                let live_here = self
                    .live_in
                    .get(group)
                    .is_none_or(|blocks| blocks.contains_checked(root.block));
                if written_in_region && live_here {
                    placements.push((root.block, *group));
                }
            }
        }
        placements
    }

    /// Renames every variable into fresh SSA versions by a dominator-tree walk
    /// with per-group version stacks.
    ///
    /// This is the core Cytron rename. It bundles the precomputed analysis into
    /// an immutable [`RenameContext`], seeds a version-0 reaching definition for
    /// each argument, local, and any other group with definitions, then walks
    /// the dominator tree via [`Renamer::rename_block_recursive`]. Handler roots not
    /// reachable through the entry's dominator tree are renamed as additional
    /// roots so exception bodies are covered.
    ///
    /// Only version-0 variables whose def site is the function entry seed the
    /// initial stacks; a variable defined at a real
    /// instruction must not stand in as the entry reaching definition, or the
    /// entry block would reference a value defined only in a later block. After
    /// the walk, [`Renamer::apply_rename_map`] rewrites remaining uses, missing phi
    /// operands are back-filled with a group's version-0 entry (or a value the
    /// phi already carries when the group has none), and `Pop`s left naming
    /// undefined variables are dropped before a trivial-phi pass runs.
    fn rename(&mut self) {
        let ctx = RenameContext {
            var_origins: &self.var_origins,
            group_types: &self.group_types,
            var_types: &self.var_types,
            successor_map: &self.successor_map,
            terminator_successor_map: &self.terminator_successor_map,
            dom_children: &self.dom_children,
            phi_groups: &self.phi_groups,
            num_args: self.ssa.num_args,
            original_phis: &self.original_phis,
            address_taken: self.ssa.address_taken_slots(),
            exceptional_successors: covering_map(self.ssa),
            exceptional_roots: FunctionRoots::of(self.ssa)
                .flow_roots()
                .iter()
                .map(|root| root.block)
                .collect(),
        };

        // Version stacks: for each group, track the current reaching definition
        let mut version_stacks: BTreeMap<u32, Vec<SsaVarId>> = BTreeMap::new();
        let mut next_version: BTreeMap<u32, u32> = BTreeMap::new();

        // Initialize with arguments and locals version 0 from existing variables.
        // Only use version-0 variables whose def site is the function entry. A
        // version-0 phi result has no instruction either, but it is defined in
        // its own block, and the phi placed for it takes over its uses.
        // Variables with instruction-specific def_sites are actual
        // definitions in specific blocks and must NOT be used as the initial reaching
        // definition for the entry block, as that would create use-before-def when the
        // entry block references a variable defined in a later block.
        for var in &self.ssa.variables {
            let group = self.ssa.rename_group(var.id());
            if group != u32::MAX {
                match var.origin() {
                    // `EntryLiveIn` belongs here with the other two: it is a
                    // value the caller supplies, defined at entry with no
                    // instruction, which is exactly what a group's version 0
                    // is. A machine-code lift names its live-in registers that
                    // way - it has no argument positions to give them until a
                    // convention is recovered - so leaving the origin out means
                    // every register group arrives at entry with an empty
                    // version stack and the loop below invents a variable to
                    // stand in for a value that is already there.
                    VariableOrigin::Argument(_)
                    | VariableOrigin::Local(_)
                    | VariableOrigin::EntryLiveIn
                        if var.version() == 0 && var.def_site() == DefSite::entry() =>
                    {
                        version_stacks.entry(group).or_default().push(var.id());
                        next_version.insert(group, 1);
                    }
                    _ => {}
                }
            }
        }

        // Ensure all groups that have definitions get a version 0 entry.
        let mut entry_placeholders: BTreeSet<SsaVarId> = BTreeSet::new();
        // Without version 0 entries, the rename step will leave stale
        // references when the version stack is empty (which causes
        // apply_rename_map to create use-before-def errors).
        for &group in self.defs.keys() {
            if !version_stacks.contains_key(&group) {
                let origin = self
                    .group_origins
                    .get(&group)
                    .copied()
                    .unwrap_or(VariableOrigin::Phi);
                let var_type = self
                    .group_types
                    .get(&group)
                    .cloned()
                    .unwrap_or_else(T::unknown_type);
                // No parent to derive from: this stands for whatever the slot
                // held on entry, which no value in the function computed.
                let id = self
                    .ssa
                    .create_variable(origin, 0, DefSite::entry(), var_type);
                self.ssa.set_rename_group(id, group);
                version_stacks.entry(group).or_default().push(id);
                next_version.insert(group, 1);
                entry_placeholders.insert(id);
            }
        }

        let state = RenameState {
            version_stacks,
            next_version,
            entry_placeholders,
            ..RenameState::default()
        };

        // Rename each tree of the dominator forest `compute_cfg` built, the
        // entry's first. The entry's tree covers every block it reaches over
        // the complete relation, handlers with a live guard included; a later
        // tree is a live region the entry does not reach, renamed from the
        // values the function was entered with.
        let mut renamer = Renamer {
            ssa: &mut *self.ssa,
            ctx,
            state,
        };
        for root in self.forest_roots.clone() {
            renamer.rename_block_recursive(root);
        }
        let state = renamer.finish();
        let block_count = self.ssa.blocks.len();

        // Fill missing phi operands: if rename didn't provide an operand for a
        // predecessor (e.g., the predecessor was not visited or the version stack
        // was empty), fill it with the version 0 entry variable for that origin.
        //
        // Normal operands only, over the terminator relation. An operand for a
        // throw is not missing merely because the block that could throw was
        // never walked: a block the program cannot reach cannot throw either,
        // and `fill_exceptional_phi_operands` states the ones that can. Filling
        // over the complete relation instead would answer a missing exceptional
        // operand with a normal one, asserting a terminator edge into a handler
        // that no terminator takes.
        {
            // Collect all needed fixups first (to avoid borrow conflicts)
            let fixups: Vec<(usize, usize, usize, SsaVarId)> = {
                let cfg = SsaCfg::from_ssa(self.ssa);
                let mut fixes = Vec::new();
                // One buffer for every phi; each is cleared of exactly the
                // predecessors it set.
                let mut existing = BitSet::new(block_count);
                for block_idx in 0..self.ssa.blocks.len() {
                    let preds: Vec<usize> = cfg.terminator_predecessors(block_idx).to_vec();
                    if preds.is_empty() {
                        continue;
                    }
                    if let Some(block) = self.ssa.block(block_idx) {
                        for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                            for op in phi.normal_operands() {
                                existing.insert_checked(op.predecessor());
                            }
                            let group = self.ssa.rename_group(phi.result());
                            // The version-0 entry for this phi's group is the
                            // natural filler. A phi belonging to no group has no
                            // such entry - [`Self::clear_all_phis`] retains
                            // exactly those, because nothing here can re-create
                            // them - so fall back to a value the phi already
                            // carries on another edge.
                            //
                            // Skipping instead leaves the phi short an operand for
                            // a real predecessor, which is `MissingPhiOperand`;
                            // a CFG edit that adds a predecessor to such a block
                            // is precisely what exposes it. Every operand of a phi
                            // names the same variable, so reusing one is the same
                            // choice `expand_phi_predecessor` makes when it
                            // propagates a value onto newly-redirected edges.
                            let filler = state
                                .version_stacks
                                .get(&group)
                                .and_then(|stack| stack.first().copied())
                                .or_else(|| phi.operands().first().map(PhiOperand::value));
                            for &pred in &preds {
                                if !existing.contains_checked(pred)
                                    && let Some(v0) = filler
                                {
                                    fixes.push((block_idx, phi_idx, pred, v0));
                                }
                            }
                            for op in phi.normal_operands() {
                                if op.predecessor() < block_count {
                                    existing.remove(op.predecessor());
                                }
                            }
                        }
                    }
                }
                fixes
            };

            // Apply fixups
            for (block_idx, phi_idx, pred, v0) in fixups {
                if let Some(block) = self.ssa.block_mut(block_idx)
                    && let Some(phi) = block.phi_nodes_mut().get_mut(phi_idx)
                {
                    phi.set_operand(pred, v0);
                }
            }
        }

        // Final cleanup: Remove Pop instructions that use undefined variables
        {
            let mut defined_vars = VarSet::new(self.ssa.var_id_bound());
            for v in &self.ssa.variables {
                defined_vars.insert(v.id());
            }
            for block in self.ssa.blocks.iter_mut() {
                block.instructions_mut().retain(|instr| {
                    if let SsaOp::Pop { value } = instr.op() {
                        return defined_vars.contains(*value);
                    }
                    true
                });
            }
        }

        // Eliminate trivial PHIs from rename
        self.eliminate_trivial_phis();
    }

    /// Eliminates trivial PHI nodes (all operands resolve to the same value).
    ///
    /// This also handles phis that become trivial when only considering operands
    /// from reachable predecessors. Unreachable predecessors may provide stale
    /// version-0 operands that make a phi look non-trivial when it's actually
    /// trivial for the reachable control flow. Without this, DCE would have to
    /// prune unreachable operands on every iteration, causing a ping-pong cycle.
    fn eliminate_trivial_phis(&mut self) {
        self.ssa.eliminate_trivial_phis(&TrivialPhiOptions {
            reachable: Some(&self.reachable),
        });

        // Also eliminate dead phis: phis whose result is never used.
        // This prevents oscillation with DCE: rebuild places phis for all
        // variables with multiple defs, but some may have no consumers.
        self.ssa.eliminate_dead_phis();
    }

    /// Removes `Pop` instructions whose value was deleted by
    /// `eliminate_trivial_phis` or `compact_variables`.
    ///
    /// After trivial-phi elimination and compaction, a `Pop` can be left naming
    /// a variable that no longer has any definition, which the verifier reports
    /// as an undefined use. This drops exactly those `Pop`s, everywhere and
    /// without exception. A handler entry is no longer a special case: the
    /// exception object it is entered with is defined by
    /// [`SsaOp::RuntimeValue`], so a `Pop` of it names a definition like any
    /// other, and a `Pop` that names nothing is an orphan wherever it sits.
    fn remove_orphan_pops(&mut self) {
        let defined_vars: VarSet = {
            let mut d = VarSet::new(self.ssa.var_id_bound());
            for b in self.ssa.blocks() {
                for phi in b.phi_nodes() {
                    d.insert(phi.result());
                }
                for instr in b.instructions() {
                    for dest in instr.op().defs() {
                        d.insert(dest);
                    }
                }
            }
            d
        };

        for block in self.ssa.blocks.iter_mut() {
            block.instructions_mut().retain(|instr| {
                if let SsaOp::Pop { value } = instr.op() {
                    return defined_vars.contains(*value);
                }
                true
            });
        }
    }

    /// Rewrites instruction operands that reference a definition appearing later
    /// in the same block, restoring intra-block topological order.
    ///
    /// Reindexing and compaction can leave an instruction using a variable whose
    /// only definition sits below it in the same block - an intra-block
    /// use-before-def the verifier rejects. Each such use is redirected to the
    /// nearest prior definition of the same rename group
    /// (`nearest_prior_def_in_group`); if none precedes it, to an entry-scoped
    /// value of the group (`entry_replacement_for_group`) whose definition
    /// dominates the use.
    ///
    /// A variable in no rename group has no other version to read, so a use of
    /// it is left as it is and reported: substituting a value minted at entry
    /// would produce a function that verifies while reading a value no caller
    /// supplies.
    ///
    /// Definition sites come from a single [`InstructionDefSites`] scan, and
    /// the sweep reads each instruction's operands in place, so it allocates
    /// nothing per instruction.
    ///
    /// # Errors
    ///
    /// Returns an error naming the uses it could not repair; every other use is
    /// repaired regardless.
    pub fn repair_same_block_future_uses(&mut self) -> Result<()> {
        let mut future_uses: Vec<(usize, usize, SsaVarId)> = Vec::new();
        let mut repairs: Vec<(usize, usize, SsaVarId, SsaVarId)> = Vec::new();
        let mut unrepaired: Vec<(usize, usize, SsaVarId)> = Vec::new();
        let mut entry_replacements: BTreeMap<u32, SsaVarId> = BTreeMap::new();
        let mut dominators: Option<DominatorTree> = None;
        let def_sites = InstructionDefSites::of(self.ssa);

        for block_idx in 0..self.ssa.block_count() {
            let Some(block) = self.ssa.block(block_idx) else {
                continue;
            };

            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                instr.op().for_each_use(|used| {
                    let Some((def_block, def_instr)) = def_sites.get(used) else {
                        return;
                    };
                    if def_block != block_idx || def_instr < instr_idx {
                        return;
                    }
                    // A variable one instruction reads twice is repaired once.
                    let already = future_uses
                        .iter()
                        .rev()
                        .take_while(|&&(block, instr, _)| block == block_idx && instr == instr_idx)
                        .any(|&(_, _, var)| var == used);
                    if !already {
                        future_uses.push((block_idx, instr_idx, used));
                    }
                });
            }
        }

        for (block_idx, instr_idx, used) in future_uses {
            let replacement = self
                .nearest_prior_def_in_group(block_idx, instr_idx, used)
                .or_else(|| {
                    self.entry_replacement_for_group(
                        used,
                        block_idx,
                        &mut entry_replacements,
                        &def_sites,
                        &mut dominators,
                    )
                });
            match replacement {
                Some(replacement) if replacement != used => {
                    repairs.push((block_idx, instr_idx, used, replacement));
                }
                Some(_) => {}
                None => unrepaired.push((block_idx, instr_idx, used)),
            }
        }

        for (block_idx, instr_idx, old_var, new_var) in repairs {
            if let Some(block) = self.ssa.block_mut(block_idx)
                && let Some(instr) = block.instructions_mut().get_mut(instr_idx)
            {
                instr.op_mut().replace_uses(old_var, new_var);
            }
        }

        if unrepaired.is_empty() {
            return Ok(());
        }
        let sites = unrepaired
            .iter()
            .map(|(block, instr, var)| format!("{var} at B{block}:{instr}"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::new(format!(
            "no dominating value can stand for a use above its definition: {sites}"
        )))
    }

    /// Finds the closest definition of `var`'s rename group that precedes a
    /// given instruction within the same block.
    ///
    /// Used by [`Self::repair_same_block_future_uses`] to redirect a use that
    /// names a later definition to a value that actually dominates it. The scan
    /// walks instructions above `instr_idx` in reverse and, within each, checks
    /// definitions in reverse so the textually nearest reaching definition of
    /// the group wins. Returns `None` for an ungrouped variable or when no
    /// prior definition of the group exists in the block.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the use being repaired.
    /// * `instr_idx` - Position of the using instruction; only definitions
    ///   strictly before it are considered.
    /// * `var` - The used variable whose rename group is matched.
    ///
    /// # Returns
    ///
    /// The nearest prior same-group definition, or `None` if there is none.
    fn nearest_prior_def_in_group(
        &self,
        block_idx: usize,
        instr_idx: usize,
        var: SsaVarId,
    ) -> Option<SsaVarId> {
        let group = self.ssa.rename_group(var);
        if group == u32::MAX {
            return None;
        }
        let block = self.ssa.block(block_idx)?;
        for instr in block.instructions().iter().take(instr_idx).rev() {
            let defs = instr.defs().collect::<Vec<_>>();
            for candidate in defs.into_iter().rev() {
                if self.ssa.rename_group(candidate) == group {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Returns a value of `var`'s rename group that no instruction defines and
    /// whose definition dominates `use_block`, minting an entry live-in if the
    /// group has none.
    ///
    /// The fallback when [`Self::nearest_prior_def_in_group`] finds nothing:
    /// the used value has no definition that dominates it, so a group member
    /// defined at a block boundary - the function entry, or a phi - stands in.
    /// Preference order is: a replacement already chosen for this group
    /// (cached in `replacements`) that dominates this use, then an existing
    /// member with no instruction def site whose block dominates `use_block`,
    /// and only otherwise a new variable (`VariableOrigin::EntryLiveIn`,
    /// `DefSite::entry()`), the slot's value on entry.
    ///
    /// A phi result elsewhere in the function also has no instruction def site,
    /// but it holds its value only below its block: adopting one that does not
    /// dominate the use would read a value from a block that has not run.
    ///
    /// # Arguments
    ///
    /// * `var` - The undefined use whose group (and type) drives the replacement.
    /// * `use_block` - The block of the use being repaired.
    /// * `replacements` - Per-group cache of chosen replacements, read and
    ///   updated so one group yields one entry variable.
    /// * `def_sites` - Instruction definition sites, used to confirm a candidate
    ///   is not itself defined at some instruction.
    /// * `dominators` - The function's dominator tree, built on first need.
    ///
    /// # Returns
    ///
    /// The replacement variable, or `None` if `var` is in no rename group or is
    /// not a known variable.
    fn entry_replacement_for_group(
        &mut self,
        var: SsaVarId,
        use_block: usize,
        replacements: &mut BTreeMap<u32, SsaVarId>,
        def_sites: &InstructionDefSites,
        dominators: &mut Option<DominatorTree>,
    ) -> Option<SsaVarId> {
        let group = self.ssa.rename_group(var);
        if group == u32::MAX {
            return None;
        }
        if let Some(replacement) = replacements.get(&group).copied()
            && let Some(site) = self.ssa.variable(replacement).map(|v| v.def_site())
            && self.definition_dominates(site.block(), use_block, dominators)
        {
            return Some(replacement);
        }

        let candidates: Vec<(SsaVarId, usize)> = self
            .ssa
            .variables()
            .iter()
            .filter(|candidate| {
                self.ssa.rename_group(candidate.id()) == group
                    && candidate.def_site().index().is_none()
                    && def_sites.get(candidate.id()).is_none()
            })
            .map(|candidate| (candidate.id(), candidate.def_site().block()))
            .collect();
        for (candidate, def_block) in candidates {
            if self.definition_dominates(def_block, use_block, dominators) {
                replacements.insert(group, candidate);
                return Some(candidate);
            }
        }

        let source = self.ssa.variable(var)?;
        let var_type = source.var_type().clone();
        let replacement = self.ssa.create_variable_derived(
            var,
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            var_type,
            Derivation::EntryValue,
        );
        self.ssa.set_rename_group(replacement, group);
        replacements.insert(group, replacement);
        Some(replacement)
    }

    /// Returns whether a value defined at the top of `def_block` is available
    /// throughout `use_block`.
    ///
    /// The entry block and the use's own block need no tree; anything else asks
    /// the exception-aware dominator tree, built once on first need.
    fn definition_dominates(
        &self,
        def_block: usize,
        use_block: usize,
        dominators: &mut Option<DominatorTree>,
    ) -> bool {
        if def_block == 0 || def_block == use_block {
            return true;
        }
        let block_count = self.ssa.block_count();
        if def_block >= block_count || use_block >= block_count {
            return false;
        }
        let tree = dominators
            .get_or_insert_with(|| compute_dominators(&SsaCfg::from_ssa(self.ssa), NodeId::new(0)));
        tree.dominates(NodeId::new(def_block), NodeId::new(use_block))
    }
}

/// Where each instruction-defined variable's first definition stands, built in
/// a single linear pass.
///
/// One scan here gives constant-time lookups where a scan of the function per
/// query would be `O(blocks × instructions)`, which would make the
/// same-block-future-use repair and rename-map application quadratic on large
/// functions. The table is indexed by variable id, so a lookup is a load rather
/// than a tree descent. Its dense part follows the function's variable count; an
/// id the instructions name past it costs one map entry, so a definition of a
/// huge id does not size the table.
///
/// Block and instruction indices are positional (matching enumeration over
/// `ssa.blocks()`), and the first definition wins for any variable defined
/// more than once. [`SsaVarId::PLACEHOLDER`] is not a variable and has no
/// entry.
struct InstructionDefSites {
    /// The `(block, instruction)` of each variable's first definition.
    sites: VarTable<Option<(usize, usize)>>,
}

impl InstructionDefSites {
    /// Scans `ssa` for its instruction definitions.
    fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let mut sites = VarTable::new(ssa.variables().len());
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                instr.op().visit_operands(|role, var| {
                    if matches!(role, OperandRole::Use) {
                        return;
                    }
                    if let Some(slot @ None) = sites.entry(var) {
                        *slot = Some((block_idx, instr_idx));
                    }
                });
            }
        }
        Self { sites }
    }

    /// Returns where `var` is first defined by an instruction, if anywhere.
    fn get(&self, var: SsaVarId) -> Option<(usize, usize)> {
        self.sites.get(var).copied().flatten()
    }
}

/// The rename group of argument slot `index`.
fn argument_group(index: u16) -> u32 {
    VariableOrigin::Argument(index)
        .canonical_rename_group(0)
        .unwrap_or(u32::MAX)
}

/// The rename group of local slot `index` in a function with `num_args`
/// arguments.
fn local_group(num_args: usize, index: u16) -> u32 {
    VariableOrigin::Local(index)
        .canonical_rename_group(num_args)
        .unwrap_or(u32::MAX)
}

/// The representative of `group`, following the union chain.
///
/// # Arguments
///
/// * `parent` - The union-find forest over rename groups.
/// * `group` - The group to resolve.
fn find_group(parent: &BTreeMap<u32, u32>, group: u32) -> Option<u32> {
    let mut current = *parent.get(&group)?;
    for _ in 0..parent.len() {
        let next = *parent.get(&current)?;
        if next == current {
            return Some(current);
        }
        current = next;
    }
    Some(current)
}

/// Joins `group` onto `representative`.
///
/// # Arguments
///
/// * `parent` - The union-find forest over rename groups.
/// * `representative` - The group that survives the join.
/// * `group` - The group being joined.
fn union_groups(parent: &mut BTreeMap<u32, u32>, representative: u32, group: u32) {
    let (Some(root), Some(other)) = (
        find_group(parent, representative),
        find_group(parent, group),
    ) else {
        return;
    };
    if root != other {
        parent.insert(other, root);
    }
}

/// One rename walk: the function renamed, the analysis it is renamed by, and
/// the version stacks and maps the walk keeps.
struct Renamer<'r, T: Target> {
    /// The function being renamed.
    ssa: &'r mut SsaFunction<T>,
    /// What the walk reads, taken before it starts.
    ctx: RenameContext<'r, T>,
    /// The reaching definitions and renames so far.
    state: RenameState,
}

impl<T: Target> Renamer<'_, T> {
    /// Rewrites the uses the walk left to their renamed variables and records
    /// the renames on the variables, handing back what the walk kept.
    fn finish(mut self) -> RenameState {
        self.apply_rename_map();
        self.record_rename();
        self.state
    }

    /// Renames a block and its dominated subtree with an explicit work stack.
    ///
    /// A recursive dominator-tree walk would risk stack overflow on the deep
    /// trees obfuscated functions produce, so the traversal is reified as a
    /// work stack of `Enter`/`Exit` actions. Entering a block processes it and
    /// pushes an `Exit` marker recording how many versions were pushed per
    /// group; popping that marker pops exactly those versions, which is what
    /// scopes each group's reaching definition to the dominator subtree. A
    /// visited bitset guards against cycles in `dom_children`, which can arise
    /// when exception handlers share blocks outside the main dominator tree.
    fn rename_block_recursive(&mut self, entry_block_idx: usize) {
        enum RenameAction {
            Enter(usize),
            Exit(Pushed),
        }

        let mut work_stack = vec![RenameAction::Enter(entry_block_idx)];
        let mut visited = BitSet::new(self.ssa.blocks.len());

        while let Some(action) = work_stack.pop() {
            match action {
                RenameAction::Exit(pushed) => {
                    self.state.identity.exit(&pushed);
                    for (group, count) in pushed.versions {
                        if let Some(stack) = self.state.version_stacks.get_mut(&group) {
                            for _ in 0..count {
                                stack.pop();
                            }
                        }
                    }
                }
                RenameAction::Enter(block_idx) => {
                    // Guard against cycles in dom_children (can occur when
                    // multiple exception handlers share blocks outside the
                    // main dominator tree).
                    if !visited.insert(block_idx) {
                        continue;
                    }

                    let pushed = self.rename_block_process(block_idx);

                    let children = self
                        .ctx
                        .dom_children
                        .get(&block_idx)
                        .cloned()
                        .unwrap_or_default();

                    work_stack.push(RenameAction::Exit(pushed));

                    for child in children.into_iter().rev() {
                        work_stack.push(RenameAction::Enter(child));
                    }
                }
            }
        }
    }
    /// Renames one block, returning how many versions each group pushed.
    ///
    /// Runs the per-block rename steps in the order the algorithm requires:
    /// rename this block's phi results, rename instruction uses and
    /// definitions, fill the phi operands of successor blocks with the reaching
    /// definitions now on top of the version stacks, and fill the phis of every
    /// handler this block can throw to with the values the block may hold while
    /// it runs. The returned `pushed_counts` is handed to the caller's `Exit`
    /// marker so the same number of versions is popped when the subtree is left.
    ///
    /// # Returns
    ///
    /// Everything pushed while processing this block, for the `Exit` marker to
    /// undo.
    fn rename_block_process(&mut self, block_idx: usize) -> Pushed {
        let mut pushed = Pushed::default();

        self.rename_phis(block_idx, &mut pushed);
        self.rename_instructions(block_idx, &mut pushed);
        self.fill_successor_phi_operands(block_idx);
        self.fill_exceptional_phi_operands(block_idx, &pushed);

        pushed
    }
    /// Assigns a fresh SSA version to each phi result in the block and pushes it
    /// as the group's current reaching definition.
    ///
    /// Each phi's group comes from the `phi_groups` mapping built during
    /// placement; when a phi predates placement (created by rename itself), the
    /// group is derived from its origin, and `Phi` / `EntryLiveIn` origins - which
    /// name no argument or local slot - fall back to `u32::MAX`. A new variable
    /// is minted at the phi's def site with the group's type, the phi's result is
    /// rewritten to it, the version is pushed onto the group's stack (and counted
    /// in `pushed_counts` so it is popped on subtree exit), and the old-to-new
    /// mapping is recorded so downstream uses are rewritten later.
    fn rename_phis(&mut self, block_idx: usize, pushed: &mut Pushed) {
        let RenameState {
            version_stacks,
            next_version,
            rename_map,
            identity,
            replaced_phis,
            entry_placeholders: _,
        } = &mut self.state;
        // Look up the group for each phi from the phi_groups mapping built during placement.
        let phi_info: Vec<(u32, VariableOrigin, SsaVarId)> = {
            let block_phi_groups = self.ctx.phi_groups.get(&block_idx);
            self.ssa
                .block(block_idx)
                .map(|b| {
                    b.phi_nodes()
                        .iter()
                        .enumerate()
                        .map(|(i, phi)| {
                            let origin = phi.origin();
                            let group = block_phi_groups
                                .and_then(|groups| groups.get(i).copied())
                                .unwrap_or_else(|| {
                                    // Fallback: the group an argument or local
                                    // slot names; a phi or entry live-in names
                                    // no slot, so there is none to derive.
                                    origin
                                        .canonical_rename_group(self.ctx.num_args)
                                        .unwrap_or(u32::MAX)
                                });
                            (group, origin, phi.result())
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        for (i, (group, origin, old_result)) in phi_info.iter().enumerate() {
            let version = *next_version.get(group).unwrap_or(&0);
            let entry = next_version.entry(*group).or_insert(0);
            *entry = entry.saturating_add(1);

            // A phi result is a joined value, but not yet: reconstruction
            // places the phis and fills their operands afterwards, so there is
            // nothing to join here. `SsaFunction::refresh_phi_facts` performs
            // the join at the repair boundary, once the operands are in place.
            //
            // Unless it is a phi that already existed - one cleared from the
            // same place, or one this renaming mints a new version of - which
            // it goes on being, so it starts with what the host knew about it.
            let original = self
                .ctx
                .original_phis
                .get(&(block_idx, *group))
                .and_then(Option::as_ref);
            let standing_for = original
                .map(|original| original.result)
                .or_else(|| (!old_result.is_placeholder()).then_some(*old_result))
                .filter(|parent| self.ssa.variable(*parent).is_some());
            // The group's type where the group is an argument or local slot,
            // every version of which has the slot's type; otherwise the type
            // of the phi this one goes on being. A stack depth holds unrelated
            // values, and its group "type" is merely the first one seen there:
            // a merge of two `int32` constants at a depth that earlier held a
            // `bool` argument was typed `bool`, and a slot given that type
            // truncates what it stores. Such a phi starts untyped and
            // `settle_phi_types` types it from its operands. An ungrouped phi
            // left unknown for good stopped every conversion chain that reads
            // it from collapsing, since a chain is walked by its sources'
            // widths.
            let slot_group_limit =
                u32::try_from(self.ssa.num_args.saturating_add(self.ssa.num_locals))
                    .unwrap_or(u32::MAX);
            let var_type = (*group < slot_group_limit)
                .then(|| self.ctx.group_types.get(group).cloned())
                .flatten()
                .or_else(|| {
                    standing_for
                        .and_then(|parent| self.ssa.variable(parent))
                        .map(|parent| parent.var_type().clone())
                })
                .unwrap_or_else(T::unknown_type);
            let new_var_id = match standing_for {
                Some(parent) => self.ssa.create_variable_derived(
                    parent,
                    *origin,
                    version,
                    DefSite::phi(block_idx),
                    var_type,
                    Derivation::Rename,
                ),
                None => {
                    self.ssa
                        .create_variable(*origin, version, DefSite::phi(block_idx), var_type)
                }
            };
            self.ssa.set_rename_group(new_var_id, *group);

            if let Some(block) = self.ssa.block_mut(block_idx)
                && let Some(phi) = block.phi_nodes_mut().get_mut(i)
            {
                phi.set_result(new_var_id);
            }

            version_stacks.entry(*group).or_default().push(new_var_id);
            pushed.version(*group);

            // A phi at a handler or filter entry is a barrier for its group: a
            // definition that dominates the root still does not reach it, since
            // the throw may have happened before that definition ran. Without
            // this, a use in the handler resolves to its own dominating
            // version and reads straight past the merge this phi exists to
            // express.
            if self.ctx.exceptional_roots.contains(&block_idx) {
                identity.barrier(*group, version_stacks, pushed);
            }

            // The phi placed here for this group stands for the one cleared
            // from the same place, so uses of that phi's result keep naming it.
            if let Some(original) = original {
                identity.bind(original.result, new_var_id, *group, version_stacks, pushed);
                replaced_phis.insert(original.result, new_var_id);
            }

            if *old_result != new_var_id {
                rename_map.insert(*old_result, new_var_id);
            }
        }
    }
    /// Rewrites each instruction's uses to the current reaching definition and
    /// mints fresh versions for its definitions.
    ///
    /// Uses are redirected to the top of their group's version stack; new
    /// definitions push a fresh version onto that stack, preferring a
    /// per-variable type over the group type so stack-derived local types
    /// survive. `LoadArg` / `LoadLocal` are the exception: rather than defining a
    /// new version, their destination is resolved to the slot group's current
    /// reaching definition and the load is turned into a `Nop`, so repeated loads
    /// of one slot collapse to the same SSA value (enabling identities like
    /// `x - x = 0`). The reaching definition is also pushed onto the destination
    /// group's stack so within-block uses - which resolve via the version stacks
    /// rather than the rename map - see it too.
    fn rename_instructions(&mut self, block_idx: usize, pushed: &mut Pushed) {
        let RenameState {
            version_stacks,
            next_version,
            rename_map,
            identity,
            ..
        } = &mut self.state;
        // Groups below this are the canonical groups of the arguments and
        // locals (see `VariableOrigin::canonical_rename_group`).
        let slot_group_limit = u32::try_from(self.ssa.num_args.saturating_add(self.ssa.num_locals))
            .unwrap_or(u32::MAX);
        // Collect instruction info including load targets for LoadArg/LoadLocal.
        // A load_target of Some(group) means the instruction is a LoadArg/LoadLocal
        // that reads from the given arg/local group. During rename, these are
        // resolved to the current reaching definition instead of creating new versions,
        // ensuring multiple loads of the same arg/local produce the same SSA variable.
        type InstrRenameInfo = (usize, Vec<SsaVarId>, Vec<SsaVarId>, Option<u32>);
        let instr_info: Vec<InstrRenameInfo> = self
            .ssa
            .block(block_idx)
            .map(|b| {
                b.instructions()
                    .iter()
                    .enumerate()
                    .map(|(i, instr)| {
                        // A load of a slot whose address escaped is a read of
                        // memory: a store through the pointer changed what it
                        // holds, and no value in the graph says so. Resolving
                        // it to the slot's current SSA value would read the
                        // value the slot had *before* that store.
                        let load_target_group = match instr.op() {
                            SsaOp::LoadArg { arg_index, .. } => {
                                (!self.ctx.address_taken.argument(*arg_index))
                                    .then(|| argument_group(*arg_index))
                            }
                            SsaOp::LoadLocal { local_index, .. } => {
                                (!self.ctx.address_taken.local(*local_index))
                                    .then(|| local_group(self.ctx.num_args, *local_index))
                            }
                            _ => None,
                        };
                        (i, instr.uses(), instr.defs().collect(), load_target_group)
                    })
                    .collect()
            })
            .unwrap_or_default();

        for (instr_idx, old_uses, old_defs, load_target_group) in &instr_info {
            // Apply use renames directly to the instruction
            let mut use_renames: Vec<(SsaVarId, SsaVarId)> = Vec::new();
            for &old_use in old_uses {
                // Its own definition where that is in scope; the group's
                // reaching definition only where it is not.
                let group = self.ssa.rename_group(old_use);
                let reaching_def = identity.resolve(old_use).or_else(|| {
                    (group != u32::MAX)
                        .then(|| version_stacks.get(&group).and_then(|s| s.last().copied()))
                        .flatten()
                });
                if let Some(reaching_def) = reaching_def
                    && reaching_def != old_use
                {
                    use_renames.push((old_use, reaching_def));
                }
            }

            if !use_renames.is_empty()
                && let Some(block) = self.ssa.block_mut(block_idx)
                && let Some(instr) = block.instructions_mut().get_mut(*instr_idx)
            {
                let op = instr.op_mut();
                for (old_use, new_use) in &use_renames {
                    op.replace_uses(*old_use, *new_use);
                }
            }

            if !old_defs.is_empty() {
                // LoadArg/LoadLocal: resolve dest to the current reaching definition
                // for the arg/local instead of creating a new version. This ensures
                // that multiple loads of the same arg/local produce the same SSA
                // variable, enabling patterns like `x - x = 0` to be recognized.
                if let (Some(target_group), Some(old_dest)) = (load_target_group, old_defs.first())
                    && let Some(reaching_def) = version_stacks
                        .get(target_group)
                        .and_then(|stack| stack.last().copied())
                {
                    rename_map.insert(*old_dest, reaching_def);
                    // Also push the reaching def onto the dest's group stack
                    // so that within-block uses (which resolve via version_stacks, not
                    // rename_map) also see the correct reaching definition.
                    let dest_group = self.ssa.rename_group(*old_dest);
                    let bound_group = if dest_group == u32::MAX {
                        *target_group
                    } else {
                        version_stacks
                            .entry(dest_group)
                            .or_default()
                            .push(reaching_def);
                        pushed.version(dest_group);
                        dest_group
                    };
                    identity.bind(*old_dest, reaching_def, bound_group, version_stacks, pushed);
                    // The load's destination is gone and the slot's reaching
                    // definition stands for it, so what the host knew about the
                    // loaded value folds into the value it resolved to.
                    self.ssa.note_substitution(
                        *old_dest,
                        reaching_def,
                        Substitution::SlotLoadCollapse,
                    );
                    // Convert to Nop since the value is the reaching definition
                    if let Some(block) = self.ssa.block_mut(block_idx)
                        && let Some(instr) = block.instructions_mut().get_mut(*instr_idx)
                    {
                        instr.set_op(SsaOp::Nop);
                    }
                    continue;
                }

                let mut def_renames = Vec::new();
                for old_dest in old_defs {
                    let group = self.ssa.rename_group(*old_dest);
                    let origin = self.ctx.var_origins.get(old_dest).copied();
                    if group != u32::MAX
                        && let Some(origin) = origin
                    {
                        let version = *next_version.get(&group).unwrap_or(&0);
                        let nv = next_version.entry(group).or_insert(0);
                        *nv = nv.saturating_add(1);

                        // Use per-variable type first (preserves stack-derived local types),
                        // fall back to the group's type only where the group is
                        // an argument or local slot: every version of a slot
                        // has its type. Any other group - a stack depth, a
                        // minted orphan group - holds unrelated values, and its
                        // "type" is merely the first one seen there.
                        let slot_group = group < slot_group_limit;
                        let var_type = self
                            .ctx
                            .var_types
                            .get(old_dest)
                            .or_else(|| {
                                slot_group
                                    .then(|| self.ctx.group_types.get(&group))
                                    .flatten()
                            })
                            .cloned()
                            .unwrap_or_else(T::unknown_type);
                        // The renamed definition is the same value under a new
                        // id, so the host's facts about it - and the role it
                        // was declared with - come along.
                        let new_var_id = self.ssa.create_variable_derived(
                            *old_dest,
                            origin,
                            version,
                            DefSite::instruction(block_idx, *instr_idx),
                            var_type,
                            Derivation::Rename,
                        );
                        self.ssa.set_rename_group(new_var_id, group);

                        version_stacks.entry(group).or_default().push(new_var_id);
                        pushed.version(group);
                        identity.bind(*old_dest, new_var_id, group, version_stacks, pushed);

                        if *old_dest != new_var_id {
                            rename_map.insert(*old_dest, new_var_id);
                            def_renames.push((*old_dest, new_var_id));
                        }
                    }
                }

                if !def_renames.is_empty()
                    && let Some(block) = self.ssa.block_mut(block_idx)
                    && let Some(instr) = block.instructions_mut().get_mut(*instr_idx)
                {
                    for (old_dest, new_dest) in def_renames {
                        instr.op_mut().replace_def(old_dest, new_dest);
                    }
                }
            }
        }
    }
    /// Fills each successor block's phi operands for this edge with the current
    /// reaching definition of the phi's group.
    ///
    /// This is the standard "fill successor phis on the way down the dominator
    /// tree" step: an edge carries the group's newest reaching definition. On a
    /// back edge whose loop does not redefine the group that is the successor's
    /// own phi - the value goes round unchanged - and the operand names it.
    /// Where the phi cleared from this place carried a value on this edge that
    /// is still in scope, the edge keeps that value instead.
    fn fill_successor_phi_operands(&mut self, block_idx: usize) {
        let RenameState {
            version_stacks,
            identity,
            ..
        } = &mut self.state;
        let successors = self
            .ctx
            .successor_map
            .get(&block_idx)
            .cloned()
            .unwrap_or_default();
        for succ_idx in successors {
            // The successor map carries the synthesized edge from a region's
            // guard to its handler, which is not a terminator edge: control
            // gets there by throwing, from anywhere in the region. Its operands
            // are the exceptional ones `fill_exceptional_phi_operands` adds, so
            // filling a normal operand here would assert an edge the block's
            // terminator does not take.
            if self.ctx.exceptional_roots.contains(&succ_idx)
                && !self
                    .ctx
                    .terminator_successor_map
                    .get(&block_idx)
                    .is_some_and(|successors| successors.contains(&succ_idx))
            {
                continue;
            }
            // Collect each successor phi's group from the phi_groups mapping
            let phi_updates: Vec<(usize, u32)> = {
                let succ_phi_groups = self.ctx.phi_groups.get(&succ_idx);
                self.ssa
                    .block(succ_idx)
                    .map(|b| {
                        b.phi_nodes()
                            .iter()
                            .enumerate()
                            .map(|(i, phi)| {
                                let group = succ_phi_groups
                                    .and_then(|groups| groups.get(i).copied())
                                    .unwrap_or_else(|| {
                                        // Fallback for phis created outside place_pruned_phis
                                        // (e.g., by rename itself). Use the phi result's group.
                                        let result = phi.result();
                                        if self.ssa.rename_groups.records(result) {
                                            self.ssa.rename_group(result)
                                        } else {
                                            phi.origin()
                                                .canonical_rename_group(self.ctx.num_args)
                                                .unwrap_or(u32::MAX)
                                        }
                                    });
                                (i, group)
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };

            for (phi_idx, group) in phi_updates {
                if group == u32::MAX {
                    continue;
                }
                // The newest definition of the group dominating this edge. On a
                // back edge the loop does not redefine the group through, that
                // is the successor's own phi: the variable goes round holding
                // the value it came in with. A definition further down the
                // stack is the one the loop was *entered* with, and naming it
                // resets the variable to that on every iteration - a dispatcher
                // re-entered through a forwarding block handed each case the
                // `null` a local held before the first case assigned it.
                //
                // Where the phi cleared from this place received a value along
                // this edge and that value's definition is still in scope, the
                // edge keeps it: the group's newest version may be a different
                // value live at the same time. That is also what keeps a value
                // LICM hoisted to a preheader and left on the edge.
                let original_operand = self
                    .ctx
                    .original_phis
                    .get(&(succ_idx, group))
                    .and_then(Option::as_ref)
                    .and_then(|original| original.operands.get(&block_idx))
                    .and_then(|value| identity.resolve(*value));
                let reaching_def = original_operand.or_else(|| {
                    version_stacks
                        .get(&group)
                        .and_then(|stack| stack.last().copied())
                });
                if let Some(reaching_def) = reaching_def
                    && let Some(succ_block) = self.ssa.block_mut(succ_idx)
                    && let Some(phi) = succ_block.phi_nodes_mut().get_mut(phi_idx)
                {
                    phi.set_operand(block_idx, reaching_def);
                }
            }
        }
    }
    /// Fills the phis of every handler this block can throw to with the values
    /// the block may hold while it runs.
    ///
    /// A throw can happen at any point in a protected block, so the handler may
    /// see the merged storage holding what it held on entry to the block, or
    /// any value the block defined before the throw. Each is recorded as an
    /// exceptional operand naming this block; the handler's phi therefore merges
    /// over program points rather than over edges, which is what an exception
    /// edge actually is.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block just renamed.
    /// * `pushed` - What this block pushed, which is what it defined.
    fn fill_exceptional_phi_operands(&mut self, block_idx: usize, pushed: &Pushed) {
        let roots = self.ctx.exceptional_successors.get(&block_idx);
        let Some(roots) = roots.filter(|roots| !roots.is_empty()) else {
            return;
        };
        let placeholders = &self.state.entry_placeholders;

        // The groups this block carries a phi for. A phi holds from the very
        // start of the block, so for those groups the phi's result *is* the
        // value on entry - the value the predecessor supplied never holds here,
        // and listing it would have the handler merge a value no throw in this
        // block can see.
        let phi_groups_here: BTreeSet<u32> = self
            .ctx
            .phi_groups
            .get(&block_idx)
            .map(|groups| groups.iter().copied().collect())
            .unwrap_or_default();

        for root in roots.clone() {
            // The phis at the root, paired with the group each merges.
            let phi_groups: Vec<(usize, u32)> = {
                let root_phi_groups = self.ctx.phi_groups.get(&root);
                self.ssa
                    .block(root)
                    .map(|block| {
                        block
                            .phi_nodes()
                            .iter()
                            .enumerate()
                            .map(|(index, phi)| {
                                let group = root_phi_groups
                                    .and_then(|groups| groups.get(index).copied())
                                    .unwrap_or_else(|| self.ssa.rename_group(phi.result()));
                                (index, group)
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };

            for (phi_idx, group) in phi_groups {
                if group == u32::MAX {
                    continue;
                }
                let Some(stack) = self.state.version_stacks.get(&group) else {
                    continue;
                };
                // Everything this block pushed for the group is a value it
                // defined, and the entry below them is what the block started
                // with. A throw can land before, between or after any of them -
                // except where a phi here supplies the entry value, in which
                // case the pushes are the whole story.
                let defined_here = pushed.versions.get(&group).copied().unwrap_or(0);
                let entry_position = stack.len().saturating_sub(defined_here);
                let first = if phi_groups_here.contains(&group) {
                    entry_position
                } else {
                    entry_position.saturating_sub(1)
                };
                let values: Vec<SsaVarId> = stack.iter().skip(first).copied().collect();
                let Some(root_block) = self.ssa.block_mut(root) else {
                    continue;
                };
                let Some(phi) = root_block.phi_nodes_mut().get_mut(phi_idx) else {
                    continue;
                };
                // A block holding the handler's own merge - the handler itself,
                // or a block it reaches before the group is redefined, when the
                // clause may be thrown to from anywhere - throws that merge
                // back, and the operand names the phi's own result, as a loop's
                // back edge does for a normal phi.
                for value in values {
                    // A stand-in is not a value a throw can carry: it means the
                    // group had nothing yet, and an operand naming it would be
                    // a read of something never produced. A throw before the
                    // group's first definition simply contributes nothing to
                    // the merge.
                    if placeholders.contains(&value) {
                        continue;
                    }
                    phi.add_exceptional_operand(block_idx, value);
                }
            }
        }
    }
    /// Rewrites every remaining variable use in the function through the
    /// old-to-new rename map, following chains transitively.
    ///
    /// The map can contain chains (a variable renamed to another that is itself
    /// renamed), so each old value is resolved to the end of its chain, with a
    /// visited set breaking any accidental cycle. Instruction uses are rewritten
    /// only when the resolved value is not defined later in the same block: an
    /// [`InstructionDefSites`] index makes that check constant-time rather than
    /// re-scanning the function per use, which would make this pass quadratic.
    /// Instructions are deliberately not re-sorted afterward - rename already
    /// emits definitions before their uses, and sorting would move pure
    /// instructions ahead of their original position and corrupt reaching-def
    /// assignment on a subsequent rebuild.
    ///
    /// # Arguments
    ///
    fn apply_rename_map(&mut self) {
        if self.state.rename_map.is_empty() {
            return;
        }

        let resolve = |var: SsaVarId| -> SsaVarId {
            let mut current = var;
            let mut visited = BTreeSet::new();
            while let Some(&new_var) = self.state.rename_map.get(&current) {
                if !visited.insert(current) {
                    break;
                }
                current = new_var;
            }
            current
        };

        // Collect all phi operand updates first, by operand index rather than by
        // predecessor. A rename changes what an operand carries and nothing
        // else, so the operand that carried the old value is the one to rewrite
        // - and one predecessor can supply a normal operand and several
        // exceptional ones. Addressing by predecessor would rewrite the normal
        // one, or, where there is none, add a normal operand asserting a
        // terminator edge that does not exist.
        let mut phi_updates: Vec<(usize, usize, usize, SsaVarId)> = Vec::new();
        for block in self.ssa.blocks.iter() {
            let block_idx = block.id();
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                for (operand_idx, op) in phi.operands().iter().enumerate() {
                    let old_val = op.value();
                    let new_val = resolve(old_val);
                    if new_val != old_val {
                        phi_updates.push((block_idx, phi_idx, operand_idx, new_val));
                    }
                }
            }
        }

        for (block_idx, phi_idx, operand_idx, new_val) in phi_updates {
            if let Some(block) = self.ssa.block_mut(block_idx)
                && let Some(phi) = block.phi_nodes_mut().get_mut(phi_idx)
                && let Some(operand) = phi.operands_mut().get_mut(operand_idx)
            {
                operand.set_value(new_val);
            }
        }

        // Collect all instruction use updates. A single def-site index avoids
        // re-scanning the whole function for every renamed use, which would make
        // this pass quadratic in instruction count on large functions.
        let def_sites = InstructionDefSites::of(self.ssa);
        let mut instr_updates: Vec<(usize, usize, SsaVarId, SsaVarId)> = Vec::new();
        // Scratch buffers reused across every instruction.
        let mut use_buf: Vec<SsaVarId> = Vec::new();
        let mut seen: BTreeSet<SsaVarId> = BTreeSet::new();
        for block in self.ssa.blocks.iter() {
            let block_idx = block.id();
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                seen.clear();
                use_buf.clear();
                instr.op().for_each_use(|var| use_buf.push(var));
                for &old_use in &use_buf {
                    if seen.insert(old_use) {
                        let new_use = resolve(old_use);
                        if new_use != old_use {
                            let is_future_def =
                                def_sites
                                    .get(new_use)
                                    .is_some_and(|(def_block, def_instr)| {
                                        def_block == block_idx && def_instr >= instr_idx
                                    });
                            if is_future_def {
                                continue;
                            }
                            instr_updates.push((block_idx, instr_idx, old_use, new_use));
                        }
                    }
                }
            }
        }

        for (block_idx, instr_idx, old_var, new_var) in instr_updates {
            if let Some(block) = self.ssa.block_mut(block_idx)
                && let Some(instr) = block.instructions_mut().get_mut(instr_idx)
            {
                instr.op_mut().replace_uses(old_var, new_var);
            }
        }

        // NOTE: We intentionally do NOT sort instructions topologically here.
        // The rename phase processes instructions in their current order and
        // produces definitions before their uses within each block, so the
        // result is already in valid topological order.
        //
        // Sorting here would reorder pure instructions (like Const) earlier
        // than their original program position. When construction runs
        // again (e.g., after normalization passes), the rename phase would
        // then process the sorted order, causing incorrect reaching-definition
        // assignments for variables sharing the same origin (stack slot).
    }
    /// Folds the rename walk into the function's variable record.
    ///
    /// Renaming replaces every definition it visits with a fresh variable and
    /// every re-placed phi with a new result, leaving the originals as orphans
    /// the next compaction removes. Without this, the record would report each
    /// renamed variable as removed. The map is resolved transitively, as
    /// [`Renamer::apply_rename_map`] resolves it: a `LoadLocal` destination names
    /// the slot's reaching definition, which may itself have been renamed.
    fn record_rename(&mut self) {
        let resolve = |var: SsaVarId| -> Option<SsaVarId> {
            let mut current = self.state.replaced_phis.get(&var).copied().unwrap_or(var);
            let mut visited = BTreeSet::new();
            while let Some(&next) = self.state.rename_map.get(&current) {
                if !visited.insert(current) {
                    break;
                }
                current = next;
            }
            Some(current)
        };
        let bound = self.ssa.var_id_bound();
        self.ssa.record_renumbering(bound, resolve);
    }
}

/// The handler and filter entries each block can throw to.
///
/// Materialized once per rebuild because the rename walk asks it per block,
/// and [`FunctionRoots`] is rebuilt from the exception table each time it is
/// constructed.
fn covering_map<T: Target>(ssa: &SsaFunction<T>) -> BTreeMap<usize, Vec<usize>> {
    let roots = FunctionRoots::of(ssa);
    if !roots.has_flow_roots() {
        return BTreeMap::new();
    }
    (0..ssa.block_count())
        .filter_map(|block| {
            let covering = roots.exceptional_successors(block);
            (!covering.is_empty()).then(|| (block, covering.to_vec()))
        })
        .collect()
}

/// Clones side-effect-free defs from unreachable blocks into the entry
/// block so their downstream uses survive block clearing.
///
/// Called from `pre_clean_unreachable` before the clear step. Only
/// `Const`, `LoadToken`, and `LoadStaticField` are eligible - these
/// have no operands (or only compile-time-constant operands) and no
/// side effects, so relocating them to the entry is always safe.
///
/// # Arguments
///
/// * `ssa` - The function whose unreachable blocks are scanned and whose
///   entry block receives the rescued clones.
/// * `reachable` - Block indices still reachable; only defs outside this set
///   whose result is read from inside it are rescued.
fn rescue_orphaned_pure_defs<T: Target>(ssa: &mut SsaFunction<T>, reachable: &BitSet) {
    if ssa.blocks.is_empty() {
        return;
    }

    // Collect every var used from reachable code - instruction operands
    // and phi operand values.
    let mut reachable_uses: BTreeSet<SsaVarId> = BTreeSet::new();
    for bi in reachable.iter() {
        let Some(block) = ssa.block(bi) else {
            continue;
        };
        for phi in block.phi_nodes() {
            for op in phi.operands() {
                reachable_uses.insert(op.value());
            }
        }
        for instr in block.instructions() {
            instr.op().for_each_use(|u| {
                reachable_uses.insert(u);
            });
        }
    }

    if reachable_uses.is_empty() {
        return;
    }

    // Collect rescuable instructions from unreachable blocks.
    let mut to_rescue: Vec<SsaInstruction<T>> = Vec::new();
    for bi in 0..ssa.block_count() {
        if reachable.contains(bi) {
            continue;
        }
        let Some(block) = ssa.block(bi) else {
            continue;
        };
        for instr in block.instructions() {
            if !instr.defs().any(|dest| reachable_uses.contains(&dest)) {
                continue;
            }
            if !matches!(
                instr.op(),
                SsaOp::Const { .. } | SsaOp::LoadToken { .. } | SsaOp::LoadStaticField { .. }
            ) {
                continue;
            }
            to_rescue.push(instr.clone());
        }
    }

    if to_rescue.is_empty() {
        return;
    }

    let Some(entry) = ssa.block_mut(0) else {
        return;
    };
    let term_idx = entry
        .instructions()
        .iter()
        .position(|i| i.is_terminator())
        .unwrap_or(entry.instructions().len());
    for (i, instr) in to_rescue.into_iter().enumerate() {
        let pos = term_idx.saturating_add(i);
        entry.instructions_mut().insert(pos, instr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            block::SsaBlock, function::SsaConstructOptions, instruction::SsaInstruction,
            value::ConstValue,
        },
        testing::{MockTarget, MockType},
    };

    /// A block nothing reaches is dead code: clearing it forgets the host
    /// block it stood for, and the live one keeps its own.
    /// **A report says when the rebuild reconstructed nothing.** Definitions
    /// in no rename group, and no group, is a rebuild that renamed nothing; a
    /// group renamed, or no definition at all, is not.
    #[test]
    fn a_report_says_when_nothing_was_reconstructed() {
        let report = |groups, ungrouped_definitions| RebuildReport {
            groups,
            ungrouped_definitions,
        };
        assert!(report(0, 3).reconstructed_nothing());
        assert!(!report(1, 3).reconstructed_nothing());
        assert!(!report(0, 0).reconstructed_nothing());
    }

    #[test]
    fn a_rebuild_forgets_what_an_unreached_block_stood_for() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        for (id, host) in [(0, 0x10), (1, 0x20)] {
            let mut block = SsaBlock::new(id);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            block.provenance_mut().carry(host);
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        ssa.construct_ssa(&SsaConstructOptions::renaming_only())
            .unwrap_or_else(|error| panic!("construction: {error}"));
        assert_eq!(
            ssa.block(0).map(|block| block.provenance().carried()),
            Some(&[0x10][..])
        );
        assert!(
            ssa.block(1)
                .is_some_and(|block| block.provenance().is_empty())
        );
    }

    /// The first definition of an id wins, the placeholder has no entry, and an
    /// id far past the variable table is found without sizing a table for it.
    #[test]
    fn def_sites_cover_a_far_id_beside_the_dense_ones() {
        let far = SsaVarId::from_index(3_000_000_000);
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let near = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut block = SsaBlock::new(0);
        for dest in [near, far, far, SsaVarId::PLACEHOLDER] {
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(1),
            }));
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);

        let sites = InstructionDefSites::of(&ssa);
        assert_eq!(sites.get(near), Some((0, 0)));
        assert_eq!(sites.get(far), Some((0, 1)));
        assert_eq!(sites.get(SsaVarId::PLACEHOLDER), None);
        assert_eq!(sites.get(SsaVarId::from_index(7)), None);
    }
}
