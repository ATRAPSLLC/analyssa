//! Complete SSA function representation: blocks, variables, and maintenance operations.
//!
//! [`SsaFunction`] is the top-level container for a method's SSA form. It holds all
//! basic blocks, SSA variables, signature metadata (argument/local counts), exception
//! handlers, and the rename group system used by SSA construction and rebuild.
//!
//! # Sub-modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `canonical` | Final cleanup: strip nops, remove empty blocks, compact indices |
//! | `queries` | Read-only analysis: return info, purity, variable tracing |
//! | `rebuild` | Full SSA reconstruction (Cytron et al. algorithm) after CFG changes |
//! | `repair` | Lightweight SSA repair for instruction-only passes |
//! | `transforms` | Mutation operations: replace uses, eliminate phis, compact, fold |
//!
//! # Structure
//!
//! ```text
//! SsaFunction<T>
//! ├── blocks: Vec<SsaBlock<T>>           // Basic blocks indexed by ID
//! ├── variables: Vec<SsaVariable<T>>      // All SSA variables (dense indexing)
//! ├── var_allocator: FunctionVarAllocator  // Dense ID allocation
//! ├── origin_versions: BTreeMap            // Origin → version list mappings
//! ├── origin_types: BTreeMap              // Origin → canonical type
//! ├── num_args / num_locals              // Method signature info
//! ├── exception_handlers: Vec<..>        // Preserved exception handler metadata
//! └── rename_groups: Vec<u32>            // Per-variable rename group IDs
//! ```
//!
//! # Construction
//!
//! Built by the `SsaConverter` (in the host crate) which:
//! 1. Simulates the CIL evaluation stack to create explicit SSA variables
//! 2. Places phi nodes at dominance frontiers (Cytron et al.)
//! 3. Renames variables to achieve single-assignment form
//! 4. Records exception handler block ranges
//!
//! # Maintenance Operations
//!
//! | Operation | Scope | Cost |
//! |-----------|-------|------|
//! | `repair_ssa()` | Instruction-only passes | Lightweight |
//! | `reconstruct_values()` | CFG-modifying passes | The disturbed values |
//! | `canonicalize()` | Pre-codegen cleanup | Moderate |
//!
//! # Thread Safety
//!
//! `SsaFunction` is `Send` and `Sync` once constructed.

mod blocklist;
mod boundary;
mod builder;
mod canonical;
mod construct;
mod editor;
mod groups;
mod kind;
mod machine;
mod queries;
mod rebuild;
mod refutations;
mod repair;
mod transforms;
mod update;

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, mem,
};

use blocklist::BlockList;
pub use boundary::{
    BlockBoundary, BoundaryConstruction, BoundaryRecorder, RecordedConstruction, RecordedRead,
    RecordedWrite,
};
pub use builder::{
    AtomicCmpXchgSpec, AtomicLockRmwSpec, SsaBlockBuilder, SsaDefSpec, SsaFunctionBuilder,
    VectorFaultingLoadSpec, VectorGatherSpec,
};
pub use construct::{ConstructionReport, EntryRule, SsaConstructOptions};
pub use editor::{
    CheckedReplaceResult, ReplacementSkipReason, SkippedReplacement, SsaEditOptions, SsaEditReport,
    SsaEditScope, SsaEditor, SsaRollbackPolicy,
};
pub use groups::RenameGroups;
pub use kind::FunctionKind;
pub use machine::{MachinePoint, MachineStates, StatedRegister, Widening};
pub use queries::{
    AddressTakenSlots, ExceptionalWeb, MethodPurity, RecordedDefinition, ReturnInfo, StorageSlots,
};
pub use rebuild::RebuildReport;
pub use refutations::{Again, Fingerprint, Refutation, Refutations, Unsettled};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use transforms::{CopyPropagationResult, TrivialPhiOptions};
pub use update::{Reconstruction, ValueUpdate};

use crate::{
    Error, Result,
    analysis::{
        pointsto::SolvedPointsTo,
        verifier::{SsaVerifier, VerifierError, VerifyLevel},
    },
    facts::{Derivation, Join, Rewrite, Substitution, VariableFacts},
    ir::{
        block::SsaBlock,
        exception::{ClausePart, ExceptionBlocks, SsaExceptionHandler},
        identity::{FunctionInstrAllocator, Identity},
        instruction::{InstrId, SsaInstruction},
        ops::{SsaOp, convert::ConversionIntent},
        phi::{PhiNode, PhiOperand},
        variable::{
            DefSite, FunctionVarAllocator, SsaVarId, SsaVariable, VariableOrigin, VariableRole,
        },
        varstore::VarMap,
    },
    target::{Target, VectorShape},
};

/// Every instruction identity in a function, indexed to its current position.
///
/// Built by [`SsaFunction::instruction_locations`] for a caller resolving many
/// ids at once; a single lookup is [`SsaFunction::locate_instruction`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstrLocations {
    /// Indexed by [`InstrId::get`]; `None` where that id names no instruction
    /// in the function any more.
    entries: Vec<Option<(usize, usize)>>,
}

impl InstrLocations {
    /// Returns where `id` sits, or `None` when nothing in the function has it.
    #[must_use]
    pub fn get(&self, id: InstrId) -> Option<(usize, usize)> {
        self.entries.get(id.get() as usize).copied().flatten()
    }

    /// Returns how many identities the index covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len().saturating_sub(1)
    }

    /// Returns `true` when the function holds no identified instruction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(Option::is_none)
    }
}

/// Where each variable that existed when tracking began has ended up.
///
/// Renumbering is invisible from outside the function: `compact_variables`,
/// [`SsaFunction::reindex_variables`] and [`SsaFunction::construct_ssa`] rewrite
/// every `SsaVarId` the function itself holds, but a caller keeping its own
/// tables keyed by `SsaVarId` - a per-value type lane, a map from value to the
/// machine storage it occupies, a set of values to treat specially - has no way
/// to learn that the ids moved. Its entries then describe whichever variable
/// inherited each index, which is not an error any assertion downstream can
/// catch, because the ids are all still valid.
///
/// This is the record that makes those tables repairable. It composes every
/// renumbering applied since [`SsaFunction::reset_variable_remap`], so one pass
/// over a caller's table with [`Self::get`] restores it.
///
/// The record's domain is exactly the ids that existed at the reset. A variable
/// created afterwards is in it only while nothing has moved: until then every
/// id, old or new, still names the variable it was given to, but once a
/// renumbering has moved or removed one, a new id is drawn from a space that
/// overlaps the reset-time one, and no lookup could tell the two apart.
///
/// # Examples
///
/// ```
/// # use analyssa::ir::{SsaFunction, SsaVarId, variable::{DefSite, VariableOrigin}};
/// # use analyssa::testing::{MockTarget, MockType};
/// # let mut function: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
/// # let kept = function.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
/// function.reset_variable_remap();
/// // ... passes run, variables are removed and the survivors renumbered ...
/// let remap = function.variable_remap();
/// let still_valid: Option<SsaVarId> = remap.get(kept);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VariableRemap {
    /// Indexed by the variable's id when tracking began; `None` once that
    /// variable has been removed.
    entries: Vec<Option<SsaVarId>>,
    /// Whether a renumbering has moved or removed any of them.
    moved: bool,
}

impl VariableRemap {
    /// Starts a record over the ids `0..count`, each mapped to itself.
    fn identity(count: usize) -> Self {
        Self {
            entries: (0..count)
                .map(|index| Some(SsaVarId::from_index(index)))
                .collect(),
            moved: false,
        }
    }

    /// Returns where `original` has ended up, or `None` if it was removed.
    ///
    /// `original` is an id as it was at [`SsaFunction::reset_variable_remap`],
    /// or at the function's creation when it was never reset. While nothing
    /// has moved, every id - one created since, included - answers itself,
    /// which is where its variable still is. Once something has moved, an id
    /// outside the record's domain answers `None`: nothing about it was
    /// recorded, and an id minted after the move may coincide with an earlier
    /// id that names a different variable.
    #[must_use]
    pub fn get(&self, original: SsaVarId) -> Option<SsaVarId> {
        match self.entries.get(original.index()) {
            Some(entry) => *entry,
            None => (!self.moved).then_some(original),
        }
    }

    /// Re-keys a table keyed by value id: each entry moves to where its
    /// variable ended up, and an entry whose variable was removed is dropped.
    ///
    /// Hosts key their facts about values by the dense id, as a `u32` or as an
    /// [`SsaVarId`], and a normalization that removes a value compacts the ids
    /// after it: a fact left under its old id would describe whichever value
    /// took that id. When nothing moved the table is returned whole, ids minted
    /// since the reset included; otherwise an id outside the record's domain is
    /// dropped, since it may coincide with a reset-time id that names another
    /// variable.
    ///
    /// The table is any container of `(key, fact)` pairs that collects from
    /// them: a `BTreeMap<u32, _>`, a [`VarMap`], or
    /// another.
    ///
    /// # Arguments
    ///
    /// * `table` - The facts, keyed by the ids they were computed against.
    ///
    /// # Returns
    ///
    /// The same facts keyed by the ids their variables hold now.
    #[must_use]
    pub fn remap_keys<K, V, C>(&self, table: C) -> C
    where
        K: Into<SsaVarId> + From<SsaVarId>,
        C: IntoIterator<Item = (K, V)> + FromIterator<(K, V)>,
    {
        if self.is_identity() {
            return table;
        }
        table
            .into_iter()
            .filter_map(|(id, value)| Some((K::from(self.get(id.into())?), value)))
            .collect()
    }

    /// Returns `true` when nothing has moved or been removed, so a caller's
    /// tables need no repair.
    ///
    /// A record no renumbering has been folded into is trivially the
    /// identity.
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        !self.moved
    }

    /// Number of variables in the record's domain: the function's variable
    /// count at the reset, grown to the ids a renumbering found while nothing
    /// had moved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the record's domain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Folds one renumbering into the record.
    ///
    /// `bound` covers every id the function held before the renumbering, and
    /// `survivor` answers where a variable's *current* id has moved to, or
    /// `None` if it is gone. While nothing has moved, a current id is still the
    /// id the variable was given, so the domain first grows to `bound`: a
    /// variable created since the reset is followed like the others.
    /// Composition is what keeps a single `get` correct across any number of
    /// passes.
    fn compose(&mut self, bound: usize, survivor: impl Fn(SsaVarId) -> Option<SsaVarId>) {
        if !self.moved && self.entries.len() < bound {
            let start = self.entries.len();
            self.entries
                .extend((start..bound).map(|index| Some(SsaVarId::from_index(index))));
        }
        for entry in &mut self.entries {
            let next = entry.and_then(&survivor);
            self.moved |= next != *entry;
            *entry = next;
        }
    }
}

/// One dense renumbering, as `reassign_dense_ids` applied it.
struct IdRenumbering {
    /// A variable's id after the renumbering, by its id before; no entry for a
    /// variable that was removed.
    new_ids: VarMap<SsaVarId>,
    /// How many surviving variables received a different id.
    moved: usize,
}

impl IdRenumbering {
    /// The id the variable that held `old` now has, or `None` if it is gone.
    fn survivor(&self, old: SsaVarId) -> Option<SsaVarId> {
        self.new_ids.get(old).copied()
    }

    /// The new id of `old` when the renumbering moved it; `None` when it kept
    /// its id or names no surviving variable.
    fn moved_to(&self, old: SsaVarId) -> Option<SsaVarId> {
        self.survivor(old).filter(|&new| new != old)
    }
}

/// Serializes a `BTreeMap` as a sequence of `(key, value)` pairs.
///
/// [`VariableOrigin`] is an enum, and several data formats - JSON among them -
/// only accept string map keys, so deriving would produce a type that compiles
/// but fails at runtime with "key must be a string". Encoding as a pair sequence
/// keeps [`SsaFunction`] serializable in *every* format rather than only the
/// self-describing ones with non-string key support.
#[cfg(feature = "serde")]
mod map_as_pairs {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Writes the map as a sequence of `(key, value)` pairs.
    pub fn serialize<K, V, S>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: Serialize,
        V: Serialize,
        S: Serializer,
    {
        serializer.collect_seq(map.iter())
    }

    /// Rebuilds the map from a sequence of `(key, value)` pairs.
    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        K: Deserialize<'de> + Ord,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let pairs = Vec::<(K, V)>::deserialize(deserializer)?;
        Ok(pairs.into_iter().collect())
    }
}

/// A complete method in SSA (Static Single Assignment) form.
///
/// The top-level container holding all SSA state for a single method:
/// basic blocks with phi nodes and instructions, all SSA variables with
/// full metadata (origins, types, def/use sites), and method signature info.
///
/// # Examples
///
/// ```rust
/// use analyssa::{ir::{function::SsaFunction, block::SsaBlock, variable::SsaVarId}, MockTarget};
///
/// let mut func: SsaFunction<MockTarget> = SsaFunction::new(2, 1);
/// func.add_block(SsaBlock::new(0));
/// func.add_block(SsaBlock::new(1));
/// func.add_block(SsaBlock::new(2));
///
/// for var in func.variables() {
///     println!("Variable: {}", var);
/// }
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::Type: serde::Serialize, T::TypeRef: serde::Serialize, \
                     T::MethodRef: serde::Serialize, T::FieldRef: serde::Serialize, T::SymbolRef: serde::Serialize, \
                     T::SigRef: serde::Serialize, T::OriginalInstruction: serde::Serialize, \
                     T::ExceptionKind: serde::Serialize, T::LocalSignature: serde::Serialize, \
                     T::VariableFacts: serde::Serialize, T::InstructionFacts: serde::Serialize, \
                     T::FunctionFacts: serde::Serialize",
        deserialize = "T::Type: serde::Deserialize<'de>, T::TypeRef: serde::Deserialize<'de>, \
                       T::MethodRef: serde::Deserialize<'de>, T::FieldRef: serde::Deserialize<'de>, T::SymbolRef: serde::Deserialize<'de>, \
                       T::SigRef: serde::Deserialize<'de>, \
                       T::OriginalInstruction: serde::Deserialize<'de>, \
                       T::ExceptionKind: serde::Deserialize<'de>, \
                       T::LocalSignature: serde::Deserialize<'de>, \
                       T::VariableFacts: serde::Deserialize<'de>, \
                       T::InstructionFacts: serde::Deserialize<'de>, \
                       T::FunctionFacts: serde::Deserialize<'de>"
    ))
)]
pub struct SsaFunction<T: Target> {
    /// Basic blocks, indexed by block ID (0..block_count()-1).
    /// Maintained as a dense vector; blocks can be removed by `canonicalize`.
    blocks: BlockList<T>,

    /// All SSA variables, densely indexed by `SsaVarId` (0..len()).
    ///
    /// Invariant: `variables[i].id().index() == i` for all entries.
    /// Maintained by `create_variable()` (dense allocation) and
    /// `compact_variables()` / `reindex_variables()` (post-removal repair).
    variables: Vec<SsaVariable<T>>,

    /// Per-function allocator producing dense sequential variable IDs.
    var_allocator: FunctionVarAllocator,

    /// Per-function allocator producing instruction identities.
    ///
    /// Counts up and never reissues, so an id names at most one instruction
    /// over the function's whole life. See [`InstrId`].
    #[cfg_attr(feature = "serde", serde(default))]
    instr_allocator: FunctionInstrAllocator,

    /// What the host knows about this function as a whole.
    ///
    /// Cloned, rolled back and serialized with it, which is what distinguishes
    /// it from a table the host keeps beside the function.
    #[cfg_attr(feature = "serde", serde(default))]
    facts: T::FunctionFacts,

    /// Where a duplicated instruction came from, keyed by the copy's identity.
    ///
    /// Out of line rather than a field on [`SsaInstruction`]: duplication is
    /// rare - most functions never hold a copy at all - and an `Option<InstrId>`
    /// on every instruction costs eight bytes each once padding is counted,
    /// against an empty map here. Keyed by identity, so nothing in it goes
    /// stale when instructions move; entries for retired ids are dropped by
    /// [`Self::assign_instr_ids`].
    #[cfg_attr(feature = "serde", serde(default, with = "map_as_pairs"))]
    instr_provenance: BTreeMap<InstrId, InstrId>,

    /// What the function's machine registers hold where execution crosses a
    /// boundary its SSA does not state ([`MachineStates`]): kept current
    /// through every edit, and keeping nothing alive.
    #[cfg_attr(feature = "serde", serde(default))]
    machine: MachineStates,

    /// Maps each [`VariableOrigin`] to its SSA variable IDs, ordered by version.
    ///
    /// Enables O(1) lookup of all versions of a given origin (e.g., all
    /// SSA versions of `Local(3)`). Populated by `create_variable()` and
    /// rebuilt by `rebuild_origin_versions()` after compaction.
    #[cfg_attr(feature = "serde", serde(with = "map_as_pairs"))]
    origin_versions: BTreeMap<VariableOrigin, Vec<SsaVarId>>,

    /// Maps each variable origin to its canonical type.
    ///
    /// Populated during SSA construction from method signatures and type
    /// inference. Read through [`origin_type`](Self::origin_type), and by
    /// [`derive_variable_for_origin`](Self::derive_variable_for_origin), so new
    /// versions inherit the correct type. First registration wins.
    #[cfg_attr(feature = "serde", serde(with = "map_as_pairs"))]
    origin_types: BTreeMap<VariableOrigin, T::Type>,

    /// Number of method arguments (including `this` for instance methods).
    num_args: usize,

    /// Number of local variables declared in the current SSA state.
    /// May be larger than `original_num_locals` if passes create temporaries.
    num_locals: usize,

    /// Number of local variables in the original method signature.
    /// Used as a floor for `shrink_num_locals()` since these locals
    /// have default-initialization semantics.
    original_num_locals: usize,

    /// Original local variable types from the method signature.
    ///
    /// Preserved during SSA construction so code generation can emit correct
    /// type information in the output assembly. Set via
    /// [`set_original_local_types`](Self::set_original_local_types).
    original_local_types: Option<Vec<T::LocalSignature>>,

    /// Exception handlers preserved from the original method body.
    ///
    /// Each handler stores both original IL offsets and SSA block index
    /// ranges. Remapped during canonicalization; used during code generation
    /// to emit correct exception handler metadata.
    exception_handlers: Vec<SsaExceptionHandler<T>>,

    /// Per-variable rename group IDs, indexed by `SsaVarId::index()`.
    ///
    /// During SSA construction and rebuild, variables sharing the same rename
    /// group share a version stack for phi placement and renaming. See the
    /// [module docs](self) for group assignment rules.
    ///
    /// Group assignment:
    /// - `Argument(i)` → group `i`
    /// - `Local(i)` → group `num_args + i`
    /// - Stack temp at depth D → group `num_args + num_locals + D`
    /// - Orphan/pass-created → auto-incrementing from max group + 1
    /// - `u32::MAX` means no group assigned
    rename_groups: RenameGroups,

    /// Function kind: normal, interrupt handler, etc.
    ///
    /// Defaults to [`FunctionKind::Normal`]. Set during SSA construction by
    /// frontends that need to mark functions as interrupt service routines.
    kind: FunctionKind,

    /// Whether a checked edit has mutated this function since the flag was last
    /// taken.
    ///
    /// [`Self::edit`] sets this the moment an edit session reports a change,
    /// *before* boundary repair can fail. A session running under
    /// [`SsaRollbackPolicy::Never`]
    /// leaves its edits applied when repair fails, so the caller's own
    /// "did anything change" return cannot be trusted to say so - and a pass
    /// group that believes nothing changed skips both verification and
    /// rollback. Consumers read this instead of trusting that return; see
    /// [`Self::take_edit_dirty`].
    ///
    /// A rollback replaces the whole function with its pre-edit clone, which
    /// restores this to whatever it was before the edit, so a restored failure
    /// correctly reports clean.
    edit_dirty: bool,

    /// Where every variable has moved since [`Self::reset_variable_remap`], or
    /// since the function was created.
    ///
    /// Bookkeeping for the benefit of callers, not state the function reads:
    /// renumbering already rewrites every id the function holds, and this is how
    /// a caller holding its own `SsaVarId`-keyed tables learns to rewrite
    /// theirs. See [`VariableRemap`]. It is kept from the start, so a caller
    /// that never reset it still reads where its values went; folding a
    /// renumbering in costs one step per variable, as the renumbering does.
    ///
    /// Not serialized. It describes a span of edits, and a function that has
    /// been through a round trip has no edits behind it.
    #[cfg_attr(feature = "serde", serde(skip))]
    variable_remap: VariableRemap,

    /// Why the last edit session that failed without restoring the function
    /// failed, until [`Self::take_edit_failure`] takes it.
    ///
    /// A session under [`SsaRollbackPolicy::Never`] that fails after mutating
    /// leaves those edits in place: moved instructions, a phi still naming a
    /// predecessor it no longer has, whatever the closure had done when the
    /// step failed. Nothing about that is guaranteed to fail verification - a
    /// phi reading a slot's entry value where the edit had a constant reach it
    /// is well-formed - so a pass that reports the change "so it is verified"
    /// can keep a miscompile. The code that runs a pass reads this and
    /// fails the pass, and the transaction around it restores the function.
    ///
    /// Not serialized, for the same reason as `variable_remap`.
    #[cfg_attr(feature = "serde", serde(skip))]
    edit_failure: Option<String>,

    /// The searches passes ran over this function to a refutation, so a later
    /// run answers them without searching again while what they read stands
    /// ([`Refutations`]).
    ///
    /// Not state the function reads, and not serialized: it is a cache of
    /// answers, and a function without it answers the same, only later.
    #[cfg_attr(feature = "serde", serde(skip))]
    refutations: Refutations,

    /// The relations the function's points-to constraints were last solved
    /// to, so the same constraints are not solved again
    /// ([`SolvedPointsTo`]).
    ///
    /// A cache of answers like [`Self::refutations`], and not serialized for
    /// the same reason.
    #[cfg_attr(feature = "serde", serde(skip))]
    points_to: SolvedPointsTo,
}

/// The function as it reads: every field but its [`Refutations`] and its
/// [`SolvedPointsTo`], which are caches of answers about it and differ with
/// how often a pass asked, not with what the function is. A rendering is what a host compares to tell whether
/// a function changed.
impl<T: Target + fmt::Debug> fmt::Debug for SsaFunction<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            blocks,
            variables,
            var_allocator,
            instr_allocator,
            facts,
            instr_provenance,
            machine,
            origin_versions,
            origin_types,
            num_args,
            num_locals,
            original_num_locals,
            original_local_types,
            exception_handlers,
            rename_groups,
            kind,
            edit_dirty,
            variable_remap,
            edit_failure,
            refutations: _,
            points_to: _,
        } = self;
        f.debug_struct("SsaFunction")
            .field("blocks", blocks)
            .field("variables", variables)
            .field("var_allocator", var_allocator)
            .field("instr_allocator", instr_allocator)
            .field("facts", facts)
            .field("instr_provenance", instr_provenance)
            .field("machine", machine)
            .field("origin_versions", origin_versions)
            .field("origin_types", origin_types)
            .field("num_args", num_args)
            .field("num_locals", num_locals)
            .field("original_num_locals", original_num_locals)
            .field("original_local_types", original_local_types)
            .field("exception_handlers", exception_handlers)
            .field("rename_groups", rename_groups)
            .field("kind", kind)
            .field("edit_dirty", edit_dirty)
            .field("variable_remap", variable_remap)
            .field("edit_failure", edit_failure)
            .finish()
    }
}

impl<T: Target> Clone for SsaFunction<T> {
    fn clone(&self) -> Self {
        let Self {
            blocks,
            variables,
            var_allocator,
            instr_allocator,
            instr_provenance,
            machine,
            facts,
            origin_versions,
            origin_types,
            num_args,
            num_locals,
            original_num_locals,
            original_local_types,
            exception_handlers,
            rename_groups,
            kind,
            edit_dirty,
            variable_remap,
            edit_failure,
            refutations,
            points_to,
        } = self;
        Self {
            blocks: blocks.clone(),
            variables: variables.clone(),
            var_allocator: var_allocator.clone(),
            instr_allocator: instr_allocator.clone(),
            instr_provenance: instr_provenance.clone(),
            machine: machine.clone(),
            facts: facts.clone(),
            origin_versions: origin_versions.clone(),
            origin_types: origin_types.clone(),
            num_args: *num_args,
            num_locals: *num_locals,
            original_num_locals: *original_num_locals,
            original_local_types: original_local_types.clone(),
            exception_handlers: exception_handlers.clone(),
            rename_groups: rename_groups.clone(),
            kind: *kind,
            edit_dirty: *edit_dirty,
            variable_remap: variable_remap.clone(),
            edit_failure: edit_failure.clone(),
            refutations: refutations.clone(),
            points_to: points_to.clone(),
        }
    }

    /// Overwrites `self` from `source`, **reusing existing allocations**.
    ///
    /// The derived `Clone` provides only the default `clone_from`
    /// (`*self = source.clone()`), which frees and reallocates every `Vec` and
    /// `BTreeMap` on every call. Callers that snapshot repeatedly into the same
    /// buffer - the optimization pipeline's per-pass-group rollback point being
    /// the motivating one - reuse capacity instead.
    ///
    /// The exhaustive destructure is deliberate: adding a field to
    /// [`SsaFunction`] makes this fail to compile rather than silently
    /// producing a snapshot that is stale in exactly one field, which would be
    /// a rollback that quietly loses data.
    fn clone_from(&mut self, source: &Self) {
        let Self {
            blocks,
            variables,
            var_allocator,
            instr_allocator,
            instr_provenance,
            machine,
            facts,
            origin_versions,
            origin_types,
            num_args,
            num_locals,
            original_num_locals,
            original_local_types,
            exception_handlers,
            rename_groups,
            kind,
            edit_dirty,
            variable_remap,
            edit_failure,
            refutations,
            points_to,
        } = source;
        self.blocks.clone_from(blocks);
        self.variables.clone_from(variables);
        self.var_allocator.clone_from(var_allocator);
        self.instr_allocator.clone_from(instr_allocator);
        self.instr_provenance.clone_from(instr_provenance);
        self.facts.clone_from(facts);
        self.origin_versions.clone_from(origin_versions);
        self.origin_types.clone_from(origin_types);
        self.num_args = *num_args;
        self.num_locals = *num_locals;
        self.original_num_locals = *original_num_locals;
        self.original_local_types.clone_from(original_local_types);
        self.exception_handlers.clone_from(exception_handlers);
        self.rename_groups.clone_from(rename_groups);
        self.kind = *kind;
        self.edit_dirty = *edit_dirty;
        self.variable_remap.clone_from(variable_remap);
        self.edit_failure.clone_from(edit_failure);
        self.refutations.clone_from(refutations);
        self.points_to.clone_from(points_to);
        self.machine.clone_from(machine);
    }
}

impl<T: Target> SsaFunction<T> {
    /// Creates a new empty SSA function.
    ///
    /// # Arguments
    ///
    /// * `num_args` - Number of method arguments (including `this` for instance methods)
    /// * `num_locals` - Number of local variables declared in the method
    ///
    /// # Returns
    ///
    /// A new empty [`SsaFunction`] with no blocks or variables.
    #[must_use]
    pub fn new(num_args: usize, num_locals: usize) -> Self {
        Self {
            blocks: BlockList::new(),
            variables: Vec::new(),
            var_allocator: FunctionVarAllocator::new(),
            instr_allocator: FunctionInstrAllocator::new(),
            instr_provenance: BTreeMap::new(),
            machine: MachineStates::default(),
            facts: T::FunctionFacts::default(),
            origin_versions: BTreeMap::new(),
            origin_types: BTreeMap::new(),
            num_args,
            num_locals,
            original_num_locals: num_locals,
            variable_remap: VariableRemap::default(),
            original_local_types: None,
            exception_handlers: Vec::new(),
            rename_groups: RenameGroups::default(),
            kind: FunctionKind::Normal,
            edit_dirty: false,
            edit_failure: None,
            refutations: Refutations::default(),
            points_to: SolvedPointsTo::default(),
        }
    }

    /// Creates a new SSA function with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `num_args` - Number of method arguments
    /// * `num_locals` - Number of local variables
    /// * `block_capacity` - Expected number of blocks
    /// * `var_capacity` - Expected number of SSA variables
    ///
    /// # Returns
    ///
    /// A new empty [`SsaFunction`] with pre-allocated storage.
    #[must_use]
    pub fn with_capacity(
        num_args: usize,
        num_locals: usize,
        block_capacity: usize,
        var_capacity: usize,
    ) -> Self {
        Self {
            blocks: BlockList::with_capacity(block_capacity),
            variables: Vec::with_capacity(var_capacity),
            var_allocator: FunctionVarAllocator::new(),
            instr_allocator: FunctionInstrAllocator::new(),
            instr_provenance: BTreeMap::new(),
            machine: MachineStates::default(),
            facts: T::FunctionFacts::default(),
            origin_versions: BTreeMap::new(),
            origin_types: BTreeMap::new(),
            num_args,
            num_locals,
            original_num_locals: num_locals,
            variable_remap: VariableRemap::default(),
            original_local_types: None,
            exception_handlers: Vec::new(),
            rename_groups: RenameGroups::with_capacity(var_capacity),
            kind: FunctionKind::Normal,
            edit_dirty: false,
            edit_failure: None,
            refutations: Refutations::default(),
            points_to: SolvedPointsTo::default(),
        }
    }

    /// Returns the SSA blocks.
    ///
    /// # Returns
    ///
    /// A slice of all [`SsaBlock<T>`]s in this function.
    #[must_use]
    pub fn blocks(&self) -> &[SsaBlock<T>] {
        &self.blocks
    }

    /// Returns an iterator over blocks with their indices.
    ///
    /// This is a convenience method that pairs each block with its index,
    /// avoiding the common `for block_idx in 0..ssa.block_count()` pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{SsaBlock, SsaFunction}};
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// ssa.add_block(SsaBlock::new(0));
    ///
    /// for (block_idx, block) in ssa.iter_blocks() {
    ///     println!("Block {}: {} instructions", block_idx, block.instruction_count());
    /// }
    /// ```
    pub fn iter_blocks(&self) -> impl Iterator<Item = (usize, &SsaBlock<T>)> {
        self.blocks.iter().enumerate()
    }

    /// Returns an iterator over all instructions with their block and instruction indices.
    ///
    /// This flattens the nested block/instruction structure into a single iterator,
    /// which is useful for passes that need to scan all instructions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{ConstValue, SsaBlock, SsaFunction, SsaInstruction, SsaOp, SsaVarId}};
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut block = SsaBlock::new(0);
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
    ///     dest: SsaVarId::from_index(0),
    ///     value: ConstValue::I32(1),
    /// }));
    /// ssa.add_block(block);
    ///
    /// for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
    ///     let op = instr.op();
    ///     assert_eq!((block_idx, instr_idx), (0, 0));
    ///     assert!(matches!(op, SsaOp::Const { .. }));
    /// }
    /// ```
    pub fn iter_instructions(&self) -> impl Iterator<Item = (usize, usize, &SsaInstruction<T>)> {
        self.blocks
            .iter()
            .enumerate()
            .flat_map(|(block_idx, block)| {
                block
                    .instructions()
                    .iter()
                    .enumerate()
                    .map(move |(instr_idx, instr)| (block_idx, instr_idx, instr))
            })
    }

    /// Returns a mutable iterator over all instructions with their block and instruction indices.
    ///
    /// This is the mutable counterpart to [`iter_instructions`], allowing passes to
    /// modify instructions while iterating. Note that structural changes (adding/removing
    /// instructions) require collecting the modifications and applying them separately.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{ConstValue, SsaBlock, SsaFunction, SsaInstruction, SsaOp, SsaVarId}};
    /// let old_var = SsaVarId::from_index(0);
    /// let new_var = SsaVarId::from_index(1);
    /// let dest = SsaVarId::from_index(2);
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut block = SsaBlock::new(0);
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
    ///     dest: new_var,
    ///     value: ConstValue::I32(1),
    /// }));
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy { dest, src: old_var }));
    /// ssa.add_block(block);
    ///
    /// for (block_idx, instr_idx, instr) in ssa.iter_instructions_mut() {
    ///     let _ = (block_idx, instr_idx);
    ///     instr.op_mut().replace_uses(old_var, new_var);
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// For passes that need to add or remove instructions, use [`blocks_mut`] to access
    /// the blocks directly, as the iterator cannot handle structural modifications.
    ///
    /// [`iter_instructions`]: Self::iter_instructions
    /// [`blocks_mut`]: Self::blocks_mut
    pub fn iter_instructions_mut(
        &mut self,
    ) -> impl Iterator<Item = (usize, usize, &mut SsaInstruction<T>)> {
        self.blocks
            .iter_mut()
            .enumerate()
            .flat_map(|(block_idx, block)| {
                block
                    .instructions_mut()
                    .iter_mut()
                    .enumerate()
                    .map(move |(instr_idx, instr)| (block_idx, instr_idx, instr))
            })
    }

    /// Returns an iterator over all phi nodes with their block and phi indices.
    ///
    /// This flattens the nested block/phi structure into a single iterator,
    /// which is useful for passes that need to analyze all phi nodes.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use analyssa::{MockTarget, ir::{PhiNode, SsaBlock, SsaFunction, SsaVarId, VariableOrigin}};
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut block = SsaBlock::new(0);
    /// block.add_phi(PhiNode::new(SsaVarId::from_index(0), VariableOrigin::Local(0)));
    /// ssa.add_block(block);
    ///
    /// for (block_idx, phi_idx, phi) in ssa.iter_phis() {
    ///     println!("Phi {} in block {} defines {}", phi_idx, block_idx, phi.result());
    /// }
    /// ```
    pub fn iter_phis(&self) -> impl Iterator<Item = (usize, usize, &PhiNode)> {
        self.blocks
            .iter()
            .enumerate()
            .flat_map(|(block_idx, block)| {
                block
                    .phi_nodes()
                    .iter()
                    .enumerate()
                    .map(move |(phi_idx, phi)| (block_idx, phi_idx, phi))
            })
    }

    /// Returns a mutable reference to the blocks.
    ///
    /// # Returns
    ///
    /// A mutable reference to the vector of [`SsaBlock<T>`]s.
    pub fn blocks_mut(&mut self) -> &mut Vec<SsaBlock<T>> {
        &mut self.blocks
    }

    /// Returns the SSA variables.
    ///
    /// # Returns
    ///
    /// A slice of all [`SsaVariable<T>`]s in this function.
    #[must_use]
    pub fn variables(&self) -> &[SsaVariable<T>] {
        &self.variables
    }

    /// Returns a mutable reference to the variables.
    ///
    /// # Returns
    ///
    /// A mutable reference to the vector of [`SsaVariable<T>`]s.
    pub fn variables_mut(&mut self) -> &mut Vec<SsaVariable<T>> {
        &mut self.variables
    }

    /// Releases the capacity the function's storage grew past its contents.
    ///
    /// A function built from a capacity estimate and edited by appending keeps
    /// the room it did not use, and a pass that removes most of what the
    /// function held leaves the room behind. A finished function can hold a
    /// large share of its memory that way: variables, each block's
    /// instructions and phi nodes, each variable's use list, the rename groups
    /// and the per-origin version lists. This returns all of it. The function
    /// is unchanged; only a later edit that grows a vector again pays to
    /// reallocate it, so call this when the function has stopped changing for
    /// good, or when an edit has shrunk it a great deal.
    pub fn shrink_to_fit(&mut self) {
        self.blocks.shrink_to_fit();
        self.variables.shrink_to_fit();
        for variable in &mut self.variables {
            variable.shrink_to_fit();
        }
        self.rename_groups.shrink_to_fit();
        for versions in self.origin_versions.values_mut() {
            versions.shrink_to_fit();
        }
    }

    /// Returns the value the host declared to be the stack pointer on entry.
    ///
    /// The declaration is carried by the variable
    /// ([`VariableRole::EntryStackPointer`]), so it follows the value through
    /// every renumbering rather than naming an id that compaction invalidates.
    ///
    /// # Returns
    ///
    /// The entry stack pointer's id, or `None` when the host declared none.
    #[must_use]
    pub fn entry_stack_pointer(&self) -> Option<SsaVarId> {
        self.variables
            .iter()
            .find(|variable| variable.role() == Some(VariableRole::EntryStackPointer))
            .map(SsaVariable::id)
    }

    /// Returns the number of method arguments.
    ///
    /// # Returns
    ///
    /// The count of method arguments, including `this` for instance methods.
    #[must_use]
    pub const fn num_args(&self) -> usize {
        self.num_args
    }

    /// Returns the number of local variables.
    ///
    /// # Returns
    ///
    /// The count of local variables declared in the method.
    #[must_use]
    pub const fn num_locals(&self) -> usize {
        self.num_locals
    }

    /// Returns the number of locals from the original method signature.
    ///
    /// With the group-based rename system, this is always equal to `num_locals`
    /// since stack temporaries use `Phi` origin instead of inflated local indices.
    #[must_use]
    pub const fn original_num_locals(&self) -> usize {
        self.original_num_locals
    }

    /// Sets both the current and original local variable counts.
    ///
    /// The two counts differ once passes create temporaries: `num_locals` grows
    /// while `original_num_locals` stays fixed as the floor that
    /// `shrink_num_locals` will not go below, since those locals carry
    /// default-initialization semantics.
    ///
    /// # Arguments
    ///
    /// * `num_locals` - The current total number of local variables.
    /// * `original_num_locals` - The count from the original method signature.
    pub fn set_num_locals(&mut self, num_locals: usize, original_num_locals: usize) {
        self.num_locals = num_locals;
        self.original_num_locals = original_num_locals;
    }

    /// Returns the number of blocks.
    ///
    /// # Returns
    ///
    /// The count of basic blocks in this function.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the number of variables.
    ///
    /// # Returns
    ///
    /// The count of SSA variables in this function.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Returns an O(1) upper bound on every variable id this function has
    /// allocated - the size to use for [`VarSet`](crate::ir::VarSet) and
    /// [`VarMap`].
    ///
    /// This is the allocator's high-water mark, so it covers ids that were
    /// allocated but never registered in `variables` (or were removed from it
    /// without compaction) as well as every registered id.
    ///
    /// Prefer this over [`Self::var_id_capacity`], which walks every block,
    /// instruction, phi and operand to find the exact maximum. That scan is
    /// O(function) and several passes size a set per loop, which on a
    /// 27,384-block function with 3,633 loops meant thousands of whole-function
    /// walks just to pick a capacity.
    ///
    /// The only ids this does not cover are ones built directly with
    /// [`SsaVarId::from_index`] outside the allocator's range. A `VarMap` and
    /// a `VarSet` hold such an id in an ordered part beside their dense one, at
    /// the cost of one entry.
    #[must_use]
    pub fn var_id_bound(&self) -> usize {
        self.variables.len().max(self.var_allocator.count())
    }

    /// Returns the exact minimum capacity needed to index all variable IDs that
    /// appear in this function (in the variables vec, block instructions, and
    /// phi nodes).
    ///
    /// Walks the whole function, so it is O(function). Never size storage by
    /// it: a malformed id of three billion makes it three billion, and a table
    /// that large is gigabytes. Use [`Self::var_id_bound`] with a
    /// [`VarSet`](crate::ir::VarSet) or [`VarMap`], which
    /// hold a far id as one entry. This exists for the cases that need the true
    /// maximum, including ids built with [`SsaVarId::from_index`] outside the
    /// allocator's range.
    #[must_use]
    pub fn var_id_capacity(&self) -> usize {
        // Single allocation-free fold over every referenced variable id. Floored
        // at `variables.len()` since IDs are normally dense.
        //
        // `SsaVarId::PLACEHOLDER` is excluded: it is a sentinel for "no id
        // assigned yet", not a variable, and its index is `u32::MAX - 1`.
        // Folding it in would make this return ~4.29 billion, so every caller
        // that sizes storage from it - eleven `BitSet`s across the passes, plus
        // the verifier's maps - would allocate 512 MB or more the moment a
        // function held an unrenamed phi. Placeholders are legitimately present
        // between phi placement and renaming, so this is reachable rather than
        // hypothetical. `VarSet` and `VarMap` apply the same rule on the
        // storage side.
        let mut max_idx = self.variables.len();
        let widen = |id: SsaVarId, max_idx: &mut usize| {
            if !id.is_placeholder() {
                *max_idx = (*max_idx).max(id.index().saturating_add(1));
            }
        };
        for v in &self.variables {
            widen(v.id(), &mut max_idx);
        }
        for b in self.blocks.iter() {
            for p in b.phi_nodes() {
                widen(p.result(), &mut max_idx);
                for op in p.operands() {
                    widen(op.value(), &mut max_idx);
                }
            }
            for i in b.instructions() {
                i.op().visit_operands(|_, var| widen(var, &mut max_idx));
            }
        }
        max_idx
    }

    /// Returns all variable IDs for a given origin, ordered by creation.
    ///
    /// This is O(1) via the version registry. For example,
    /// `versions_of(VariableOrigin::Local(3))` returns all SSA versions
    /// of local variable 3.
    #[must_use]
    pub fn versions_of(&self, origin: VariableOrigin) -> &[SsaVarId] {
        self.origin_versions
            .get(&origin)
            .map_or(&[], |v| v.as_slice())
    }

    /// Returns the most recently created variable ID for a given origin.
    #[must_use]
    pub fn latest_version(&self, origin: VariableOrigin) -> Option<SsaVarId> {
        self.origin_versions
            .get(&origin)
            .and_then(|v| v.last().copied())
    }

    /// Gets the local index for a variable ID.
    ///
    /// With dense IDs, this is always O(1) - the index equals `id.index()`.
    ///
    /// # Arguments
    ///
    /// * `id` - The variable ID to look up
    ///
    /// # Returns
    ///
    /// The local index (0-based), or `None` if the variable is not in this function.
    #[must_use]
    pub fn var_index(&self, id: SsaVarId) -> Option<usize> {
        let idx = id.index();
        if idx < self.variables.len() {
            Some(idx)
        } else {
            None
        }
    }

    /// Returns `true` if this function has no blocks.
    ///
    /// # Returns
    ///
    /// `true` if the function contains no blocks, `false` otherwise.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Gets a block by index.
    ///
    /// # Arguments
    ///
    /// * `index` - The block index to retrieve
    ///
    /// # Returns
    ///
    /// A reference to the block, or `None` if the index is out of bounds.
    #[must_use]
    pub fn block(&self, index: usize) -> Option<&SsaBlock<T>> {
        self.blocks.get(index)
    }

    /// Gets a mutable block by index.
    ///
    /// # Arguments
    ///
    /// * `index` - The block index to retrieve
    ///
    /// # Returns
    ///
    /// A mutable reference to the block, or `None` if the index is out of bounds.
    pub fn block_mut(&mut self, index: usize) -> Option<&mut SsaBlock<T>> {
        self.blocks.get_mut(index)
    }

    /// Gets a variable by ID. O(1) via dense indexing.
    ///
    /// # Arguments
    ///
    /// * `id` - The variable ID to look up
    ///
    /// # Returns
    ///
    /// A reference to the variable, or `None` if the ID is out of bounds.
    #[must_use]
    pub fn variable(&self, id: SsaVarId) -> Option<&SsaVariable<T>> {
        self.variables.get(id.index())
    }

    /// Gets a mutable variable by ID. O(1) via dense indexing.
    ///
    /// # Arguments
    ///
    /// * `id` - The variable ID to look up
    ///
    /// # Returns
    ///
    /// A mutable reference to the variable, or `None` if the ID is out of bounds.
    pub fn variable_mut(&mut self, id: SsaVarId) -> Option<&mut SsaVariable<T>> {
        self.variables.get_mut(id.index())
    }

    /// Returns the bit width of variable `id`'s type, when the type has a
    /// fixed one ([`Target::bit_width`]).
    #[must_use]
    pub fn bit_width(&self, id: SsaVarId) -> Option<u32> {
        T::bit_width(self.variable(id)?.var_type())
    }

    /// Returns whether variable `id` read as `bits` bits is read whole: its
    /// type is that wide, or it states none, and is then the bits the
    /// operation reading it reads (a register a move copies, a stack slot an
    /// argument arrives in).
    #[must_use]
    pub fn is_read_whole_at(&self, id: SsaVarId, bits: u32) -> bool {
        self.variable(id).is_some_and(|variable| {
            let ty = variable.var_type();
            T::is_unknown(ty) || T::bit_width(ty) == Some(bits)
        })
    }

    /// Returns the lanes of variable `id`'s type, when it is a vector of a
    /// fixed lane count ([`Target::vector_shape`]).
    #[must_use]
    pub fn vector_shape(&self, id: SsaVarId) -> Option<VectorShape> {
        T::vector_shape(self.variable(id)?.var_type())
    }

    /// Adds a block to this function.
    ///
    /// Instructions the block carries are identified as they join the function:
    /// an [`InstrId`] comes from the function's allocator, so a block built on
    /// its own has none to give, and joining is the first moment one can be
    /// issued. An instruction added to a block that is *already* in the
    /// function is identified by the next repair boundary, or by
    /// [`Self::assign_instr_ids`].
    ///
    /// # Arguments
    ///
    /// * `block` - The block to add
    pub fn add_block(&mut self, mut block: SsaBlock<T>) {
        for instr in block.instructions_mut() {
            if instr.id().is_none() {
                let id = self.instr_allocator.alloc();
                instr.set_id(id);
            }
        }
        self.blocks.push(block);
    }

    /// Creates a new variable with a dense ID allocated by this function.
    ///
    /// This is the **only** way to create variables. The ID is guaranteed to be
    /// dense (equal to the variable's index in the variables Vec), enabling
    /// O(1) lookup via direct indexing.
    ///
    /// If `var_type` is not `Unknown` and `origin` names a slot
    /// ([`VariableOrigin::names_a_slot`]), it is registered in the origin type
    /// registry for future lookups.
    ///
    /// # Arguments
    ///
    /// * `origin` - Where this variable came from (argument, local, stack temp, phi).
    /// * `version` - The SSA version number for this origin.
    /// * `def_site` - The location where this variable is defined.
    /// * `var_type` - The variable's type; if concrete, it seeds the origin type
    ///   registry for a slot's origin.
    ///
    /// # Returns
    ///
    /// The dense [`SsaVarId`] allocated for the new variable.
    pub fn create_variable(
        &mut self,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        var_type: T::Type,
    ) -> SsaVarId {
        self.var_allocator.advance_past(self.variables.len());
        let id = self.var_allocator.alloc();
        let var = SsaVariable::new(id, origin, version, def_site, var_type.clone());
        debug_assert_eq!(id.index(), self.variables.len());
        self.origin_versions.entry(origin).or_default().push(id);
        self.register_origin_type(origin, var_type);
        self.variables.push(var);
        // A fresh variable belongs to no rename group, and saying so takes a
        // write rather than a resize: the vector can be longer than the
        // variable table after a renumbering, and then the new variable's slot
        // already holds whatever group the variable that last occupied that
        // index carried. Inheriting it silently makes two unrelated storages
        // one variable - the same collapse the sentinel bug produced, arriving
        // by a different route.
        self.set_rename_group(id, RenameGroups::NONE);
        id
    }

    /// Creates a new variable that stands for `parent`, typed as `parent` is,
    /// or from the origin type registry where `parent` has no type.
    ///
    /// The host's facts come from `parent` through
    /// [`VariableFacts::derive`], as [`Self::create_variable_derived`] carries
    /// them: a value minted to hold what another held on some path is that
    /// value there, not a fresh one - and so is its type. The registry is the
    /// fallback only, and answers only for a slot's origin: a value of an
    /// origin every temporary shares stays untyped when `parent` is.
    ///
    /// # Arguments
    ///
    /// * `parent` - The value the new one stands for.
    /// * `origin` - Where the new value comes from.
    /// * `version` - The SSA version number for the new variable.
    /// * `def_site` - The location where the new variable is defined.
    /// * `cause` - What derived it.
    ///
    /// # Returns
    ///
    /// The dense [`SsaVarId`] allocated for the new variable.
    pub fn derive_variable_for_origin(
        &mut self,
        parent: SsaVarId,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        cause: Derivation,
    ) -> SsaVarId {
        let var_type = self
            .variable(parent)
            .map(|parent| parent.var_type().clone())
            .filter(|var_type| !T::is_unknown(var_type))
            .unwrap_or_else(|| self.origin_type(origin));
        self.create_variable_derived(parent, origin, version, def_site, var_type, cause)
    }

    /// Registers the canonical type for a variable origin.
    ///
    /// Only registers if the type is not the host's "unknown" type and `origin`
    /// names a slot ([`VariableOrigin::names_a_slot`]): an origin every
    /// temporary shares has no type of its own. If a type is already registered
    /// for this origin, it is not overwritten (first wins).
    ///
    /// # Arguments
    ///
    /// * `origin` - The origin to associate a canonical type with.
    /// * `var_type` - The type to register; ignored if it is the host's "unknown" type.
    pub fn register_origin_type(&mut self, origin: VariableOrigin, var_type: T::Type) {
        if origin.names_a_slot()
            && !T::is_unknown(&var_type)
            && !self.origin_types.contains_key(&origin)
        {
            self.origin_types.insert(origin, var_type);
        }
    }

    /// Returns the registered type for a variable origin, or the host's
    /// "unknown" type.
    ///
    /// Only a slot's origin is ever registered
    /// ([`Self::register_origin_type`]), so an origin every temporary shares
    /// answers "unknown".
    #[must_use]
    pub fn origin_type(&self, origin: VariableOrigin) -> T::Type {
        self.origin_types
            .get(&origin)
            .cloned()
            .unwrap_or_else(T::unknown_type)
    }

    /// Returns the origin type registry.
    #[must_use]
    pub fn origin_types(&self) -> &BTreeMap<VariableOrigin, T::Type> {
        &self.origin_types
    }

    /// Returns what the host knows about this function.
    #[must_use]
    pub const fn facts(&self) -> &T::FunctionFacts {
        &self.facts
    }

    /// Returns what the host knows about this function, for updating.
    pub const fn facts_mut(&mut self) -> &mut T::FunctionFacts {
        &mut self.facts
    }

    /// Creates a variable that stands for an existing one.
    ///
    /// The host's facts for `parent` are carried onto the new value through
    /// [`VariableFacts::derive`], and so is the parent's
    /// [`VariableRole`] - a value minted to stand for the stack pointer is
    /// still the stack pointer. Everything else matches
    /// [`create_variable`](Self::create_variable).
    ///
    /// # Arguments
    ///
    /// * `parent` - The value the new one stands for.
    /// * `origin` - Where the new value comes from.
    /// * `version` - Its SSA version.
    /// * `def_site` - Where it is defined.
    /// * `var_type` - Its type.
    /// * `cause` - What derived it.
    ///
    /// # Returns
    ///
    /// The new variable's id.
    pub fn create_variable_derived(
        &mut self,
        parent: SsaVarId,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        var_type: T::Type,
        cause: Derivation,
    ) -> SsaVarId {
        let inherited = self
            .variable(parent)
            .map(|var| (T::VariableFacts::derive(var.facts(), cause), var.role()));
        let id = self.create_variable(origin, version, def_site, var_type);
        if let Some((facts, role)) = inherited
            && let Some(var) = self.variable_mut(id)
        {
            var.set_facts(facts);
            if let Some(role) = role {
                var.set_role(role);
            }
        }
        id
    }

    /// Creates a variable that merges several existing ones.
    ///
    /// The host's facts for the operands are folded into the new value through
    /// [`VariableFacts::join`]. Used for a phi result, where the value is
    /// whichever operand the path supplied.
    ///
    /// # Arguments
    ///
    /// * `parents` - The values being merged, in operand order.
    /// * `origin` - Where the new value comes from.
    /// * `version` - Its SSA version.
    /// * `def_site` - Where it is defined.
    /// * `var_type` - Its type.
    /// * `cause` - What kind of merge this is.
    ///
    /// # Returns
    ///
    /// The new variable's id.
    pub fn create_variable_joined(
        &mut self,
        parents: &[SsaVarId],
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        var_type: T::Type,
        cause: Join,
    ) -> SsaVarId {
        let operands: Vec<T::VariableFacts> = parents
            .iter()
            .filter_map(|parent| self.variable(*parent).map(|var| var.facts().clone()))
            .collect();
        let joined = T::VariableFacts::join(&mut operands.iter(), cause);
        let id = self.create_variable(origin, version, def_site, var_type);
        if let Some(var) = self.variable_mut(id) {
            var.set_facts(joined);
        }
        id
    }

    /// Recomputes every phi result's host facts from its operands'.
    ///
    /// A phi's result is created before its operands are known - reconstruction
    /// places phis, then fills them - so the join runs here, once the operands
    /// are in place. An operand that is the phi's own result is left out of
    /// the join: it carries the value around a loop and says nothing of it. Called by the repair boundaries, which is what makes
    /// "a phi result's facts are the join of its operands'" hold for any
    /// function a host reads back. Each result is asked through
    /// [`VariableFacts::rejoin`], so a host whose facts know more about a phi
    /// than its operands do keeps it.
    ///
    /// Costs nothing for a host that carries no per-variable facts: the whole
    /// walk is skipped when the fact type is zero-sized.
    ///
    /// # Returns
    ///
    /// The number of phi results whose facts were recomputed.
    pub fn refresh_phi_facts(&mut self) -> usize {
        if size_of::<T::VariableFacts>() == 0 {
            return 0;
        }
        let phis: Vec<(SsaVarId, Vec<SsaVarId>)> = self
            .blocks
            .iter()
            .flat_map(|block| {
                block.phi_nodes().iter().map(|phi| {
                    let result = phi.result();
                    // An operand that is the phi itself says only that the
                    // value is kept around a loop: `x = phi(7, x)` is 7.
                    let operands = phi
                        .operands()
                        .iter()
                        .map(PhiOperand::value)
                        .filter(|value| *value != result)
                        .collect();
                    (result, operands)
                })
            })
            .collect();
        let mut refreshed = 0usize;
        for (result, operands) in phis {
            let facts: Vec<T::VariableFacts> = operands
                .iter()
                .filter_map(|operand| self.variable(*operand).map(|var| var.facts().clone()))
                .collect();
            let Some(var) = self.variable_mut(result) else {
                continue;
            };
            let joined = var.facts().rejoin(&mut facts.iter(), Join::PlacedPhi);
            var.set_facts(joined);
            refreshed = refreshed.saturating_add(1);
        }
        refreshed
    }

    /// Types every untyped phi result as the values it merges, and returns how
    /// many it typed.
    ///
    /// A phi result is minted before its operands are known - construction,
    /// reconstruction and value repair all place a phi and fill it afterwards -
    /// so the minting cannot say what type it is. Once the operands are in
    /// place it is theirs, when they agree; operands that disagree leave it
    /// untyped rather than choose. Left untyped, a phi was the one value of its
    /// storage with no width, and every analysis that walks a value by its
    /// width - a conversion chain collapsing through it, a reshape - stopped
    /// there.
    ///
    /// Integers of one width agree whatever sign their types spell: a value's
    /// IR type states its width, and its sign is whatever the instruction that
    /// made it happened to name - a constant `0` and an `add` merged at a loop
    /// head are one 32-bit integer. Such a phi is typed as the target's unsigned
    /// integer of that width, the spelling that claims no sign.
    ///
    /// Iterated until nothing changes, because phis feed phis: a loop header's
    /// back-edge operand is often another join, typed only once this reaches
    /// it. A result that already has a type keeps it.
    pub fn settle_phi_types(&mut self) -> usize {
        let mut settled_total = 0usize;
        loop {
            let mut settled: Vec<(SsaVarId, T::Type)> = Vec::new();
            for block in self.blocks.iter() {
                for phi in block.phi_nodes() {
                    let result = phi.result();
                    if self
                        .variable(result)
                        .is_none_or(|variable| !T::is_unknown(variable.var_type()))
                    {
                        continue;
                    }
                    let mut agreed: Option<T::Type> = None;
                    let mut conflict = false;
                    for operand in phi.operands() {
                        let Some(ty) = self
                            .variable(operand.value())
                            .map(SsaVariable::var_type)
                            .filter(|ty| !T::is_unknown(ty))
                        else {
                            continue;
                        };
                        match agreed.as_ref() {
                            None => agreed = Some(ty.clone()),
                            Some(seen) if seen == ty => {}
                            Some(seen) => match Self::same_width_integer(seen, ty) {
                                Some(unsigned) => agreed = Some(unsigned),
                                None => {
                                    conflict = true;
                                    break;
                                }
                            },
                        }
                    }
                    if let (false, Some(ty)) = (conflict, agreed) {
                        settled.push((result, ty));
                    }
                }
            }
            if settled.is_empty() {
                return settled_total;
            }
            settled_total = settled_total.saturating_add(settled.len());
            for (result, ty) in settled {
                if let Some(variable) = self.variable_mut(result) {
                    variable.set_type(ty);
                }
            }
        }
    }

    /// Restates every conversion for the type its operand has, as `intent`
    /// says the front end meant it, and returns how many it restated.
    ///
    /// A front end chooses each conversion from its operand's type at the time
    /// it lifts the instruction, and a read construction answers afterwards
    /// has only a guess then - the definition put in its place holds its bits
    /// as that definition says. Called once the types are final (every read
    /// answered, every phi typed), each conversion becomes the one
    /// [`SsaOp::restated_for_operand`] gives: a reinterpretation the
    /// conversion reads through is placed before it, attributed to the same
    /// original instruction, and the function's identities, definition sites
    /// and use lists are brought up to date.
    ///
    /// # Arguments
    ///
    /// * `intent` - What the front end meant its conversions to do.
    pub fn settle_conversions(&mut self, intent: ConversionIntent) -> usize {
        let mut settled = 0usize;
        for block_idx in 0..self.blocks.len() {
            let count = self
                .blocks
                .get(block_idx)
                .map_or(0, |block| block.instructions().len());
            // Backwards, so a reinterpretation placed before one conversion
            // moves none of the ones still to visit.
            for instr_idx in (0..count).rev() {
                let Some(instruction) = self
                    .blocks
                    .get(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                else {
                    continue;
                };
                let Some(operand) = instruction.op().converted_operand() else {
                    continue;
                };
                let Some(restated) = self.variable(operand).and_then(|variable| {
                    instruction
                        .op()
                        .restated_for_operand(variable.var_type(), intent)
                }) else {
                    continue;
                };
                let original = instruction.original().clone();
                let mut op = restated.op;
                let mut at = instr_idx;
                if let Some(through) = restated.through {
                    let bits = self.create_variable(
                        VariableOrigin::Phi,
                        0,
                        DefSite::instruction(block_idx, instr_idx),
                        through.clone(),
                    );
                    op.replace_uses(operand, bits);
                    let Some(block) = self.blocks.get_mut(block_idx) else {
                        continue;
                    };
                    block.instructions_mut().insert(
                        instr_idx,
                        SsaInstruction::new(
                            original,
                            SsaOp::Bitcast {
                                dest: bits,
                                operand,
                                target: through,
                            },
                        ),
                    );
                    at = instr_idx.saturating_add(1);
                }
                if let Some(instruction) = self
                    .blocks
                    .get_mut(block_idx)
                    .and_then(|block| block.instructions_mut().get_mut(at))
                {
                    instruction.set_op_preserving_type(op, Rewrite::Settled);
                    settled = settled.saturating_add(1);
                }
            }
        }
        if settled > 0 {
            self.assign_instr_ids();
            self.refresh_def_sites();
            self.recompute_uses();
        }
        settled
    }

    /// The unsigned integer type of `left`'s width, when `left` and `right` are
    /// integers of one width.
    fn same_width_integer(left: &T::Type, right: &T::Type) -> Option<T::Type> {
        if !(T::is_integer(left) && T::is_integer(right)) {
            return None;
        }
        let bits = T::bit_width(left)?;
        (T::bit_width(right)? == bits)
            .then(|| T::integer_type(bits, false))
            .flatten()
    }

    /// Records that `new_var` now stands for `old_var`, folding the host's
    /// facts for the replaced value into the survivor.
    ///
    /// Called where one value's uses become another's - value numbering, copy
    /// propagation, a phi folded away - which is the moment the replaced
    /// value's facts would otherwise be lost.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The value being replaced.
    /// * `new_var` - The value that now stands for it.
    /// * `cause` - What replaced it.
    pub fn note_substitution(&mut self, old_var: SsaVarId, new_var: SsaVarId, cause: Substitution) {
        if old_var == new_var {
            return;
        }
        let Some(replaced) = self.variable(old_var).map(|var| var.facts().clone()) else {
            return;
        };
        if let Some(survivor) = self.variable_mut(new_var) {
            survivor.facts_mut().absorb(&replaced, cause);
        }
    }

    /// Issues a fresh instruction identity from this function's allocator.
    ///
    /// The identity is attached to nothing: a caller building an instruction to
    /// insert normally lets the editor or [`Self::assign_instr_ids`] stamp it,
    /// and reaches for this only when it needs the id before the instruction is
    /// in place.
    pub fn mint_instr_id(&mut self) -> InstrId {
        self.instr_allocator.alloc()
    }

    /// Gives every instruction without an identity one, and returns how many it
    /// issued.
    ///
    /// Idempotent: an instruction that already has an id keeps it, so running
    /// this after an edit costs one walk and changes nothing else. Every repair
    /// boundary calls it, which is what makes "an instruction in a repaired
    /// function has an identity" a property a host can rely on.
    pub fn assign_instr_ids(&mut self) -> usize {
        if !self.instr_provenance.is_empty() || !self.machine.is_empty() {
            let live: BTreeSet<InstrId> = self
                .blocks
                .iter()
                .flat_map(|block| block.instructions().iter().filter_map(SsaInstruction::id))
                .collect();
            self.instr_provenance.retain(|copy, _| live.contains(copy));
            self.machine.retain_instructions(&live);
        }
        let mut assigned = 0usize;
        for block_idx in 0..self.blocks.len() {
            let count = self
                .blocks
                .get(block_idx)
                .map_or(0, |block| block.instructions().len());
            for instr_idx in 0..count {
                let unassigned = self
                    .blocks
                    .get(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                    .is_some_and(|instr| instr.id().is_none());
                if !unassigned {
                    continue;
                }
                let id = self.instr_allocator.alloc();
                if let Some(instr) = self
                    .blocks
                    .get_mut(block_idx)
                    .and_then(|block| block.instructions_mut().get_mut(instr_idx))
                {
                    instr.set_id(id);
                    assigned = assigned.saturating_add(1);
                }
            }
        }
        assigned
    }

    /// Finds where `id` sits now, as a `(block, instruction)` pair.
    ///
    /// A linear scan. A caller resolving more than a couple of ids builds
    /// [`Self::instruction_locations`] once instead.
    #[must_use]
    pub fn locate_instruction(&self, id: InstrId) -> Option<(usize, usize)> {
        self.blocks
            .iter()
            .enumerate()
            .find_map(|(block_idx, block)| {
                block
                    .instructions()
                    .iter()
                    .position(|instr| instr.id() == Some(id))
                    .map(|instr_idx| (block_idx, instr_idx))
            })
    }

    /// Indexes every identity in the function to its current position.
    ///
    /// One walk, answering [`InstrLocations::get`] in constant time, for a
    /// caller resolving a table of ids at once.
    #[must_use]
    pub fn instruction_locations(&self) -> InstrLocations {
        let mut entries = vec![None; self.instr_allocator.count().saturating_add(2)];
        for (block_idx, block) in self.blocks.iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if let Some(id) = instr.id()
                    && let Some(slot) = entries.get_mut(id.get() as usize)
                {
                    *slot = Some((block_idx, instr_idx));
                }
            }
        }
        InstrLocations { entries }
    }

    /// Returns the instruction a copy was made from, if this one is a copy.
    ///
    /// Recorded by
    /// [`SsaEditor::duplicate_instruction`](crate::ir::function::SsaEditor::duplicate_instruction).
    /// The answer stays meaningful only while the source instruction is still
    /// in the function; once it is removed, its id names nothing and
    /// [`Self::locate_instruction`] says so.
    #[must_use]
    pub fn instruction_provenance(&self, copy: InstrId) -> Option<InstrId> {
        self.instr_provenance.get(&copy).copied()
    }

    /// Records that `copy` was duplicated from `source`.
    fn record_instruction_provenance(&mut self, copy: InstrId, source: InstrId) {
        self.instr_provenance.insert(copy, source);
    }

    /// Adopts instructions built against another function, re-stamping them.
    ///
    /// An instruction carries an id from the function that minted it, and two
    /// functions allocate independently, so splicing a body in - inlining a
    /// callee, say - would import ids this function has already issued. This
    /// gives each one a fresh identity here and records where it came from, so
    /// every id still names one instruction.
    ///
    /// # Arguments
    ///
    /// * `instructions` - The instructions to adopt, rewritten in place.
    pub fn adopt_instructions(&mut self, instructions: &mut [SsaInstruction<T>]) {
        for instr in instructions {
            *instr = instr.duplicated();
            let id = self.instr_allocator.alloc();
            instr.set_id(id);
        }
    }

    /// Where each variable has ended up since [`Self::reset_variable_remap`].
    ///
    /// Renumbering is invisible from outside: `compact_variables`,
    /// [`Self::reindex_variables`] and [`Self::construct_ssa`] rewrite every id the
    /// function holds, and a caller's own `SsaVarId`-keyed tables keep pointing
    /// at indices that now belong to other variables. Reading this after a pass -
    /// or after a whole pipeline, since the record composes - is what lets those
    /// tables be repaired.
    ///
    /// Once anything has moved, the record answers only for ids the function
    /// held at the last reset (see [`VariableRemap::get`]). Without a reset it
    /// is measured from the function's creation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use analyssa::ir::{SsaFunction, variable::{DefSite, VariableOrigin}};
    /// # use analyssa::testing::{MockTarget, MockType};
    /// # let mut function: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
    /// # let value = function.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
    /// # let mut side_table = std::collections::BTreeMap::from([(value, "eax")]);
    /// function.reset_variable_remap();
    /// // ... passes run ...
    /// let remap = function.variable_remap().clone();
    /// side_table = side_table
    ///     .into_iter()
    ///     .filter_map(|(value, storage)| Some((remap.get(value)?, storage)))
    ///     .collect();
    /// ```
    #[must_use]
    pub fn variable_remap(&self) -> &VariableRemap {
        &self.variable_remap
    }

    /// Starts the record again from the ids as they are now.
    ///
    /// Call this at the point a caller's tables are built, so what
    /// [`Self::variable_remap`] later reports is measured from ids the caller
    /// actually holds. Every later renumbering is folded in.
    pub fn reset_variable_remap(&mut self) {
        self.variable_remap = VariableRemap::identity(self.variables.len());
    }

    /// The searches passes ran over this function to a refutation
    /// ([`Refutations`]).
    #[must_use]
    pub const fn refutations(&self) -> &Refutations {
        &self.refutations
    }

    /// The searches passes ran over this function to a refutation, to record
    /// another in.
    pub const fn refutations_mut(&mut self) -> &mut Refutations {
        &mut self.refutations
    }

    /// The relations the function's points-to constraints were last solved
    /// to ([`SolvedPointsTo`]).
    #[must_use]
    pub const fn solved_points_to(&self) -> &SolvedPointsTo {
        &self.points_to
    }

    /// Folds one renumbering into the variable record.
    ///
    /// `bound` covers every id the function held before the renumbering;
    /// `survivor` maps a current id to the id it now has, or `None` if that
    /// variable is gone.
    fn record_renumbering(
        &mut self,
        bound: usize,
        survivor: impl Fn(SsaVarId) -> Option<SsaVarId>,
    ) {
        self.variable_remap.compose(bound, survivor);
    }

    /// Rebuilds the origin_versions registry from the current variables list.
    ///
    /// Called after operations that modify the variables list (compact, reindex).
    fn rebuild_origin_versions(&mut self) {
        // Emptied and refilled in place, so the lists keep their allocations: a
        // repair that removed nothing rebuilds the same registry it started with.
        for versions in self.origin_versions.values_mut() {
            versions.clear();
        }
        for var in &self.variables {
            self.origin_versions
                .entry(var.origin())
                .or_default()
                .push(var.id());
        }
        self.origin_versions
            .retain(|_, versions| !versions.is_empty());
    }

    /// Reassigns dense variable IDs after variable removal.
    ///
    /// This must be called after removing variables from `self.variables` to restore
    /// the dense indexing invariant (`variables[i].id().index() == i`).
    ///
    /// Alongside renumbering, it carries each survivor's rename group across to
    /// its new slot and folds the move into the variable record, so a caller reading the record after any number of passes sees one
    /// composed answer. The returned table is what the caller then uses to
    /// rewrite id references it does not own (see `remap_var_ids_in_blocks`).
    fn reassign_dense_ids(&mut self) -> IdRenumbering {
        // Keyed by the id a variable holds on entry, which is below the bound
        // unless the table was malformed: a row holding `SsaVarId::PLACEHOLDER`
        // or an id past every one the allocator issued costs no more than an
        // entry. An id with no entry is a removed variable, and a caller's entry
        // for it describes nothing - which is the answer, not a gap.
        let bound = self.var_id_bound();
        let mut new_ids: VarMap<SsaVarId> = VarMap::new(bound);
        let mut moved = 0usize;
        let old_groups = mem::take(&mut self.rename_groups);
        let count = self.variables.len();
        self.var_allocator = FunctionVarAllocator::starting_from(count);
        let mut new_groups = RenameGroups::ungrouped(count);
        for (index, var) in self.variables.iter_mut().enumerate() {
            let old_id = var.id();
            let new_id = SsaVarId::from_index(index);
            // Carry over the rename group from the old position
            new_groups.set(new_id, old_groups.get(old_id), count);
            new_ids.insert(old_id, new_id);
            if old_id != new_id {
                moved = moved.saturating_add(1);
                var.set_id(new_id);
            }
        }
        self.rename_groups = new_groups;
        let renumbering = IdRenumbering { new_ids, moved };
        self.machine.renumbered(|old| renumbering.survivor(old));
        self.record_renumbering(bound, |current| renumbering.survivor(current));
        renumbering
    }

    /// Remaps all variable ID references in blocks (instructions, phi nodes, terminators)
    /// using the given renumbering.
    ///
    /// Instructions are rewritten through `remap_variables` so only ids change and
    /// each resolved result type is preserved rather than cleared. Ids the
    /// renumbering does not move are left unchanged. Returns early when
    /// nothing moved.
    ///
    /// # Arguments
    ///
    /// * `renumbering` - The renumbering `reassign_dense_ids` applied.
    fn remap_var_ids_in_blocks(&mut self, renumbering: &IdRenumbering) {
        if renumbering.moved == 0 {
            return;
        }
        let lookup = |id: SsaVarId| -> Option<SsaVarId> { renumbering.moved_to(id) };
        let resolve = |id: SsaVarId| -> SsaVarId { renumbering.moved_to(id).unwrap_or(id) };

        for block in self.blocks.iter_mut() {
            // Remap phi nodes
            for phi in block.phi_nodes_mut() {
                let old_result = phi.result();
                phi.set_result(resolve(old_result));
                for operand in phi.operands_mut() {
                    // Renumbering changes which variable an operand names, not
                    // the edge it arrived on.
                    operand.set_value(resolve(operand.value()));
                }
            }
            // Remap instructions using existing remap_variables. The op shape is
            // unchanged (only variable ids are rewritten), so the resolved
            // result type is preserved rather than cleared.
            for instr in block.instructions_mut() {
                instr.remap_variables_preserving_type(lookup);
            }
        }
    }

    /// Sets the original local variable types from the method signature.
    ///
    /// These types are preserved so they can be used during code generation
    /// to maintain correct type information in the output assembly.
    ///
    /// # Arguments
    ///
    /// * `types` - The original local variable types from the method signature.
    pub fn set_original_local_types(&mut self, types: Vec<T::LocalSignature>) {
        self.original_local_types = Some(types);
    }

    /// Returns the original local variable types if set.
    ///
    /// # Returns
    ///
    /// The original local types, or `None` if not set.
    #[must_use]
    pub fn original_local_types(&self) -> Option<&[T::LocalSignature]> {
        self.original_local_types.as_deref()
    }

    /// Sets the exception handlers for this function.
    ///
    /// These are preserved from the original method body and will be
    /// remapped during code generation based on the new instruction layout.
    ///
    /// # Arguments
    ///
    /// * `handlers` - The exception handlers from the original method body.
    pub fn set_exception_handlers(&mut self, handlers: Vec<SsaExceptionHandler<T>>) {
        self.exception_handlers = handlers;
    }

    /// Returns the exception handlers for this function.
    ///
    /// # Returns
    ///
    /// A slice of exception handlers, or an empty slice if none are set.
    #[must_use]
    pub fn exception_handlers(&self) -> &[SsaExceptionHandler<T>] {
        &self.exception_handlers
    }

    /// Indexes the block roles this function's exception table assigns.
    ///
    /// Prefer this over reading the five `Option<usize>` block fields of each
    /// [`SsaExceptionHandler`] by hand: it answers "can control enter this
    /// block without a terminator naming it" and "is this block a protected
    /// region boundary" once, totally, and for any index.
    ///
    /// The result borrows the function, so it cannot outlive an edit to the
    /// table it was built from.
    #[must_use]
    pub fn exception_blocks(&self) -> ExceptionBlocks<'_, T> {
        ExceptionBlocks::from_handlers(&self.exception_handlers, self.block_count())
    }

    /// Whether `from`'s edge into `old_target` can go to `new_target` instead
    /// without breaking the function's exception structure.
    ///
    /// A branch enters a protected region only at its first block, and never
    /// enters a handler or a filter: the runtime does. Moving an edge past a
    /// region's entry - a forwarding block that starts the `try`, a dispatch at
    /// its head that a constant decides - would land it in the region's middle.
    /// Leaving a region is the other half: an edge that stayed inside one must
    /// not start leaving it, since the transfer out goes through the clause's
    /// exit and the original edge was not one.
    ///
    /// # Arguments
    ///
    /// * `from` - The block whose terminator would be rewritten.
    /// * `old_target` - Where the edge goes now.
    /// * `new_target` - Where it would go.
    ///
    /// # Returns
    ///
    /// `true` when every clause part the new edge enters, it enters at a
    /// protected region's start, and every part it leaves, the old edge left too.
    #[must_use]
    pub fn retarget_respects_regions(
        &self,
        from: usize,
        old_target: usize,
        new_target: usize,
    ) -> bool {
        self.exception_handlers.iter().all(|handler| {
            handler.parts().all(|(part, range)| {
                if range.contains(from) {
                    range.contains(new_target) || !range.contains(old_target)
                } else {
                    !range.contains(new_target)
                        || (part == ClausePart::Protected && range.start() == new_target)
                }
            })
        })
    }

    /// Returns whether this function has any exception handlers.
    ///
    /// # Returns
    ///
    /// `true` if the function has at least one exception handler.
    #[must_use]
    pub fn has_exception_handlers(&self) -> bool {
        !self.exception_handlers.is_empty()
    }

    /// Returns `true` if any block in this function contains an
    /// [`InterruptReturn`](SsaOp::InterruptReturn) op.
    ///
    /// This is useful for validation: a function using `InterruptReturn`
    /// should typically be marked as [`FunctionKind::InterruptHandler`].
    ///
    /// # Returns
    ///
    /// `true` if at least one `InterruptReturn` op exists in the function.
    #[must_use]
    pub fn has_interrupt_return(&self) -> bool {
        self.blocks.iter().any(|block| {
            block
                .instructions()
                .iter()
                .any(|instr| matches!(instr.op(), SsaOp::InterruptReturn))
        })
    }

    /// Returns the function kind.
    ///
    /// # Returns
    ///
    /// The [`FunctionKind`] of this function (defaults to [`FunctionKind::Normal`]).
    #[must_use]
    pub fn kind(&self) -> FunctionKind {
        self.kind
    }

    /// Returns whether a checked edit has mutated this function, clearing the
    /// flag.
    ///
    /// A pass group must treat a `true` result as "changed" regardless of what
    /// the pass itself reported: under
    /// [`SsaRollbackPolicy::Never`]
    /// a failed boundary repair leaves the edits applied, and a pass that
    /// returns "unchanged" would otherwise have its damaged IR kept without
    /// verification.
    pub fn take_edit_dirty(&mut self) -> bool {
        mem::replace(&mut self.edit_dirty, false)
    }

    /// Marks this function as mutated by a checked edit.
    fn mark_edit_dirty(&mut self) {
        self.edit_dirty = true;
    }

    /// Returns why an edit session failed and left its edits applied, if one
    /// did since the last call, clearing the record.
    ///
    /// Whoever runs a pass must fail it when this is `Some`: the function holds
    /// a half-applied edit, and only a restore from before the pass is known to
    /// be sound. See the field's documentation for why verification alone does
    /// not catch it.
    pub fn take_edit_failure(&mut self) -> Option<String> {
        self.edit_failure.take()
    }

    /// Records that an edit session failed with `error` and left its edits
    /// applied.
    fn mark_edit_failed(&mut self, error: &Error) {
        self.edit_failure = Some(error.to_string());
    }

    /// Sets the function kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - The [`FunctionKind`] to assign (e.g., [`FunctionKind::InterruptHandler`]).
    pub fn set_kind(&mut self, kind: FunctionKind) {
        self.kind = kind;
    }

    /// Returns the rename group for a variable.
    ///
    /// Returns `u32::MAX` if no group has been assigned (the variable was
    /// created without a rename group, e.g. by a compiler pass).
    #[must_use]
    pub fn rename_group(&self, var_id: SsaVarId) -> u32 {
        self.rename_groups.get(var_id)
    }

    /// The highest rename group any variable actually carries.
    ///
    /// `None` when no variable is grouped. The distinction matters because
    /// [`u32::MAX`] is the in-band sentinel for "this variable has no group",
    /// so a plain `max()` over the group vector returns it as soon as *one*
    /// variable is ungrouped - which reads as "the groups go all the way up" to
    /// one caller and "there are no groups" to another, and both are wrong.
    /// Every caller that mints a fresh group id needs this question answered
    /// the same way, or the new id collides with one already in use and two
    /// unrelated variables become versions of one.
    ///
    /// IR whose frontend groups every variable never exposes the difference; a
    /// machine-code lift, whose temporaries are deliberately ungrouped, exposes
    /// it always.
    #[must_use]
    pub fn max_rename_group(&self) -> Option<u32> {
        self.rename_groups.max_group()
    }

    /// Sets the rename group for a variable.
    ///
    /// The groups of the ids below [`Self::var_id_bound`] are held in a vector
    /// extended as far as the id needs; any other id costs one entry beside it.
    /// [`SsaVarId::PLACEHOLDER`] is not a variable, so a group set for it is
    /// ignored: its index is `u32::MAX - 1`, and a table that grew to hold it
    /// would allocate gigabytes.
    pub fn set_rename_group(&mut self, var_id: SsaVarId, group: u32) {
        let bound = self.var_id_bound();
        self.rename_groups.set(var_id, group, bound);
    }

    /// Sorts instructions in all blocks in topological order.
    ///
    /// This ensures that within each block, if instruction A uses a value defined
    /// by instruction B, then B appears before A.
    ///
    /// For passes that may have disrupted instruction order. Construction does
    /// not sort: its rename keeps program order, which already defines every
    /// value before its uses in a block.
    ///
    /// # Returns
    ///
    /// `true` if all blocks were successfully sorted, `false` if any block has
    /// cyclic dependencies (which indicates invalid SSA).
    pub fn sort_all_blocks_topologically(&mut self) -> bool {
        let mut all_sorted = true;
        for block in self.blocks.iter_mut() {
            if !block.sort_instructions_topologically() {
                all_sorted = false;
            }
        }
        all_sorted
    }

    /// Validates that no meaningfully-used variable has `SsaType::Unknown`.
    ///
    /// This ensures that all variables whose values are actually consumed have a
    /// concrete type. Variables are considered NOT meaningfully used if:
    /// - They have no uses at all (dead variables, stripped by DCE)
    /// - Their only uses are in `Pop` instructions (value is discarded)
    /// - Their only uses are as phi operands where the phi result is also unused
    ///
    /// # Errors
    ///
    /// Returns `Err` with a description listing the first Unknown-typed
    /// variable that has meaningful uses.
    pub fn validate_types(&self) -> Result<(), String> {
        for var in &self.variables {
            if !T::is_unknown(var.var_type()) || var.uses().is_empty() {
                continue;
            }

            // Check if all uses are in Pop instructions (value is discarded)
            let has_meaningful_use = var.uses().iter().any(|use_site| {
                if use_site.is_phi_operand() {
                    // Phi operand - only meaningful if the phi result has a known type.
                    // If the phi result is also Unknown, this is just Unknown feeding
                    // Unknown (e.g., uninitialized locals in a loop), not a real error.
                    if let Some(block) = self.block(use_site.block())
                        && let Some(phi) = block.phi(use_site.index())
                        && let Some(result_var) = self.variable(phi.result())
                    {
                        return !T::is_unknown(result_var.var_type());
                    }
                    return false;
                }
                if let Some(block) = self.block(use_site.block())
                    && let Some(instr) = block.instruction(use_site.index())
                {
                    return !matches!(instr.op(), SsaOp::Pop { .. });
                }
                true // Conservative: assume meaningful if we can't check
            });

            if has_meaningful_use {
                // Collect details about the meaningful uses for debugging
                let use_details: Vec<String> = var
                    .uses()
                    .iter()
                    .map(|use_site| {
                        if use_site.is_phi_operand() {
                            return format!("phi in block {}", use_site.block());
                        }
                        if let Some(block) = self.block(use_site.block())
                            && let Some(instr) = block.instruction(use_site.index())
                        {
                            return format!(
                                "block {} instr {}: {:?}",
                                use_site.block(),
                                use_site.index(),
                                instr.op()
                            );
                        }
                        format!(
                            "block {} instr {}: <unknown>",
                            use_site.block(),
                            use_site.index()
                        )
                    })
                    .collect();
                return Err(format!(
                    "Variable {} (origin={:?}) has Unknown type but is used ({} uses): [{}]",
                    var.id(),
                    var.origin(),
                    var.uses().len(),
                    use_details.join(", ")
                ));
            }
        }
        Ok(())
    }

    /// Validates that the SSA function is well-formed.
    ///
    /// This checks several SSA invariants:
    ///
    /// 1. **No cyclic dependencies within a block** - Operations must have a valid
    ///    topological order. If operation A uses the result of operation B, then B
    ///    must come before A in the instruction list.
    ///
    /// 2. **Single definition** - Each variable should be defined at most once
    ///    (the defining property of SSA form).
    ///
    /// 3. **Phi nodes at block start** - Phi nodes should only appear at the
    ///    beginning of blocks, not mixed with regular instructions.
    ///
    /// # Errors
    ///
    /// Returns `Err` with structured verifier diagnostics if any SSA
    /// invariant is violated, such as cyclic dependencies, duplicate
    /// definitions, or misplaced terminators.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{ir::SsaOp, testing};
    ///
    /// let mut ssa = testing::const_i32_return(42);
    ///
    /// // A well-formed SSA function validates cleanly.
    /// assert!(ssa.validate().is_ok());
    /// assert!(ssa.is_valid());
    ///
    /// // Breaking an invariant (here: a terminator that returns an undefined
    /// // variable) is reported instead of silently accepted.
    /// let bogus = analyssa::ir::SsaVarId::from_index(999);
    /// ssa.block_mut(0).unwrap().instructions_mut()[1]
    ///     .set_op(SsaOp::returns(Some(bogus)));
    ///
    /// let errors = ssa.validate().unwrap_err();
    /// assert!(!errors.is_empty());
    /// ```
    pub fn validate(&self) -> Result<(), Vec<VerifierError>> {
        let errors = SsaVerifier::new(self).verify(VerifyLevel::Standard);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validates that every index this function names exists.
    ///
    /// The structural floor beneath [`validate`](Self::validate): terminator
    /// targets, phi predecessors, variable ids, and the def and use sites on
    /// the variable table all name blocks, instructions and variables by index,
    /// and every analysis indexes a fixed-size structure with what it finds.
    /// Running this first is what turns malformed IR into a diagnostic instead
    /// of a panic inside a pass.
    ///
    /// A function that passes this may still be invalid SSA; it is merely
    /// safe to analyse.
    ///
    /// # Errors
    ///
    /// Returns every out-of-range index, in discovery order.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{ir::SsaOp, testing};
    ///
    /// let mut ssa = testing::const_i32_return(42);
    /// assert!(ssa.validate_indices().is_ok());
    ///
    /// // A terminator naming a block the function does not hold.
    /// ssa.block_mut(0).unwrap().instructions_mut()[1].set_op(SsaOp::Jump { target: 7 });
    /// assert!(ssa.validate_indices().is_err());
    /// ```
    pub fn validate_indices(&self) -> Result<(), Vec<VerifierError>> {
        let errors = SsaVerifier::new(self).index_bounds_only();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Checks if the SSA function is valid without returning detailed errors.
    ///
    /// This is a convenience method that returns `true` if [`validate`](Self::validate)
    /// would return `Ok(())`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

impl<T: Target> fmt::Display for SsaFunction<T>
where
    SsaBlock<T>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "SSA Function ({}, {} args, {} locals):",
            self.kind, self.num_args, self.num_locals
        )?;
        writeln!(f, "  Variables: {}", self.variables.len())?;
        writeln!(f, "  Blocks: {}", self.blocks.len())?;
        writeln!(f)?;

        for block in self.blocks.iter() {
            write!(f, "{block}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        analysis::verifier::VerifierError,
        facts::Derivation,
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            function::{SsaConstructOptions, SsaFunction, transforms::Cleanup},
            instruction::SsaInstruction,
            ops::{SsaOp, convert::ConversionIntent},
            phi::PhiNode,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
            varstore::VarMap,
        },
        testing::{self, MockTarget, MockType},
    };

    /// The diamond fixture's phi left untyped, with its right arm typed `right`.
    fn untyped_diamond(right: MockType) -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa = testing::diamond_phi_fixture();
        let (result, operand) = ssa
            .block(3)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| (phi.result(), phi.operands()[1].value()))
            .expect("the fixture merges at B3");
        if let Some(variable) = ssa.variable_mut(result) {
            variable.set_type(MockType::Unknown);
        }
        if let Some(variable) = ssa.variable_mut(operand) {
            variable.set_type(right);
        }
        (ssa, result)
    }

    /// A phi takes the type its operands agree on; integers of one width agree
    /// whatever sign they spell, as the unsigned spelling; anything else leaves
    /// it untyped.
    #[test]
    fn a_phi_is_typed_as_what_it_merges() {
        let settled = |right: MockType| {
            let (mut ssa, result) = untyped_diamond(right);
            ssa.settle_phi_types();
            ssa.variable(result).map(|variable| *variable.var_type())
        };
        assert_eq!(settled(MockType::I32), Some(MockType::I32));
        assert_eq!(settled(MockType::U32), Some(MockType::U32));
        assert_eq!(settled(MockType::I64), Some(MockType::Unknown));
        assert_eq!(settled(MockType::Ptr), Some(MockType::Unknown));
    }

    /// Shrinking releases the room the storage grew past its contents and
    /// changes nothing the function says.
    #[test]
    fn shrinking_releases_capacity_and_keeps_the_function() {
        let mut ssa = constants(5);
        ssa.variables_mut().reserve(64);
        ssa.blocks_mut().reserve(16);
        for block in ssa.blocks_mut().iter_mut() {
            block.instructions_mut().reserve(64);
            block.phi_nodes_mut().reserve(8);
        }
        for versions in ssa.origin_versions.values_mut() {
            versions.reserve(8);
        }
        let before = ssa.to_string();
        assert!(ssa.variables_mut().capacity() > ssa.variables().len());
        assert!(!ssa.origin_versions.is_empty());
        assert!(
            ssa.origin_versions
                .values()
                .all(|versions| versions.capacity() > versions.len())
        );

        ssa.shrink_to_fit();

        assert_eq!(ssa.variables_mut().capacity(), ssa.variables().len());
        assert_eq!(ssa.blocks_mut().capacity(), ssa.blocks().len());
        for block in ssa.blocks_mut().iter_mut() {
            assert_eq!(
                block.instructions_mut().capacity(),
                block.instructions().len()
            );
            assert_eq!(block.phi_nodes_mut().capacity(), block.phi_nodes().len());
        }
        assert!(
            ssa.origin_versions
                .values()
                .all(|versions| versions.capacity() == versions.len())
        );
        assert_eq!(ssa.to_string(), before);
    }

    /// A value minted to stand for another has that value's type, not the one
    /// the origin registry happened to record first.
    #[test]
    fn a_derived_variable_takes_its_parents_type() {
        let mut ssa = constants(1);
        let parent = SsaVarId::from_index(0);
        if let Some(variable) = ssa.variable_mut(parent) {
            variable.set_type(MockType::I64);
        }
        assert_eq!(ssa.origin_type(VariableOrigin::Local(0)), MockType::I32);
        let derived = ssa.derive_variable_for_origin(
            parent,
            VariableOrigin::Local(0),
            0,
            DefSite::phi(0),
            Derivation::LoopSplit,
        );
        assert_eq!(
            ssa.variable(derived).map(|variable| *variable.var_type()),
            Some(MockType::I64)
        );
    }

    /// A value minted to stand for an untyped one stays untyped: the origin
    /// every temporary shares has no type of its own, whatever type the first
    /// temporary of that origin had. A slot's origin keeps the type its first
    /// typed version stated.
    #[test]
    fn a_derived_variable_takes_no_type_from_a_shared_origin() {
        let mut ssa = constants(1);
        ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(0), MockType::I32);
        let parent =
            ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(0), MockType::Unknown);
        ssa.register_origin_type(VariableOrigin::EntryLiveIn, MockType::Ptr);
        for origin in [VariableOrigin::Phi, VariableOrigin::EntryLiveIn] {
            let derived = ssa.derive_variable_for_origin(
                parent,
                origin,
                0,
                DefSite::phi(0),
                Derivation::LoopSplit,
            );
            assert_eq!(
                ssa.variable(derived).map(|variable| *variable.var_type()),
                Some(MockType::Unknown),
                "{origin:?}"
            );
        }
        ssa.create_variable(VariableOrigin::Local(7), 0, DefSite::phi(0), MockType::I64);
        assert_eq!(ssa.origin_type(VariableOrigin::Local(7)), MockType::I64);
    }

    /// A function of `count` constants in one block, returning the last.
    ///
    /// Variable `i` is `Local(i)`, so a test can tell where each one went by
    /// its origin whatever id it holds.
    fn constants(count: u16) -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, usize::from(count));
        let mut block = SsaBlock::new(0);
        let mut last = None;
        for index in 0..count {
            let dest = ssa.create_variable(
                VariableOrigin::Local(index),
                1,
                DefSite::instruction(0, usize::from(index)),
                MockType::I32,
            );
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(i32::from(index)),
            }));
            last = Some(dest);
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(last)));
        ssa.add_block(block);
        ssa.recompute_uses();
        ssa
    }

    /// Turns the instruction defining `var` into a `Nop`, so the next
    /// compaction removes it.
    fn kill(ssa: &mut SsaFunction<MockTarget>, var: SsaVarId) {
        let block = ssa.block_mut(0).expect("the entry block");
        for instr in block.instructions_mut() {
            if instr.op().dest() == Some(var) {
                instr.set_op(SsaOp::Nop);
            }
        }
    }

    /// The origin of the variable `original` has ended up at.
    fn origin_of(ssa: &SsaFunction<MockTarget>, original: SsaVarId) -> Option<VariableOrigin> {
        let current = ssa.variable_remap().get(original)?;
        ssa.variable(current).map(|variable| variable.origin())
    }

    /// Compacting away a dead variable reports where every survivor moved.
    ///
    /// Without the record a caller's `SsaVarId`-keyed table silently re-points:
    /// the ids stay valid, so nothing downstream can catch it, and each entry
    /// simply describes whichever variable inherited the index.
    #[test]
    fn compaction_reports_where_the_survivors_moved() {
        let mut ssa = constants(2);
        let dead = SsaVarId::from_index(0);
        let live = SsaVarId::from_index(1);
        ssa.reset_variable_remap();

        kill(&mut ssa, dead);
        ssa.compact_variables();

        assert_eq!(
            ssa.variable_remap().get(dead),
            None,
            "a removed variable has nowhere to be"
        );
        assert_eq!(
            ssa.variable_remap().get(live),
            Some(SsaVarId::from_index(0))
        );
        assert_eq!(origin_of(&ssa, live), Some(VariableOrigin::Local(1)));
    }

    /// A record no renumbering has moved answers every id as itself, one of a
    /// function never reset included. A host settling its tables against it
    /// keeps them whole; answering `None` would read every value as removed.
    #[test]
    fn an_unmoved_record_answers_every_id_as_itself() {
        let ssa = constants(2);
        let kept = SsaVarId::from_index(1);

        assert!(ssa.variable_remap().is_identity());
        assert_eq!(ssa.variable_remap().get(kept), Some(kept));
    }

    /// A variable created after the reset, before anything moved, is followed
    /// through the renumbering that then moves it: its id was still the one
    /// it was given, so the record's domain grows to take it in.
    #[test]
    fn a_variable_created_before_any_move_is_followed() {
        let mut ssa = constants(2);
        ssa.reset_variable_remap();
        let fresh = ssa.create_variable(
            VariableOrigin::Local(2),
            1,
            DefSite::instruction(0, 2),
            MockType::I32,
        );
        let block = ssa.block_mut(0).expect("the entry block");
        block.instructions_mut().insert(
            2,
            SsaInstruction::synthetic(SsaOp::Const {
                dest: fresh,
                value: ConstValue::I32(2),
            }),
        );
        assert_eq!(ssa.variable_remap().get(fresh), Some(fresh));

        kill(&mut ssa, SsaVarId::from_index(0));
        ssa.compact_variables();

        assert_eq!(
            ssa.variable_remap().get(fresh),
            Some(SsaVarId::from_index(1)),
            "{ssa}"
        );
        assert_eq!(origin_of(&ssa, fresh), Some(VariableOrigin::Local(2)));
    }

    /// A table keyed by value id follows its values: a fact about a removed
    /// value goes with it, and the others move to their new ids.
    #[test]
    fn a_value_keyed_table_follows_the_renumbering() {
        let mut ssa = constants(3);
        ssa.reset_variable_remap();
        let table: BTreeMap<u32, &str> = [(0, "dead"), (1, "one"), (2, "two")].into();
        assert_eq!(ssa.variable_remap().remap_keys(table.clone()), table);

        kill(&mut ssa, SsaVarId::from_index(0));
        ssa.compact_variables();
        let moved = ssa.variable_remap().remap_keys(table);
        assert_eq!(moved, BTreeMap::from([(0, "one"), (1, "two")]));
    }

    /// The renumbering moves a dense table as it moves an ordered one: the same
    /// entries under the same ids, whichever container holds them. The control
    /// is the ordered table over the same renumbering.
    #[test]
    fn a_dense_table_follows_the_renumbering_as_an_ordered_one_does() {
        let mut ssa = constants(3);
        ssa.reset_variable_remap();
        let ordered: BTreeMap<u32, &str> = [(0, "dead"), (1, "one"), (2, "two")].into();
        let dense: VarMap<&str> = ordered
            .iter()
            .map(|(id, entry)| (SsaVarId::from(*id), *entry))
            .collect();
        assert_eq!(ssa.variable_remap().remap_keys(dense.clone()), dense);

        kill(&mut ssa, SsaVarId::from_index(0));
        ssa.compact_variables();
        let moved = ssa.variable_remap().remap_keys(dense);
        let expected = ssa.variable_remap().remap_keys(ordered);
        assert_eq!(expected, BTreeMap::from([(0, "one"), (1, "two")]));
        assert_eq!(
            moved
                .iter()
                .map(|(var, entry)| (u32::from(var), *entry))
                .collect::<BTreeMap<_, _>>(),
            expected
        );
    }

    /// Two renumberings that each move the survivors compose into one answer.
    ///
    /// A caller repairs its tables once, at the end of a pipeline, so the
    /// second renumbering must be applied to where the first left each
    /// variable, not to its reset-time id.
    #[test]
    fn two_moving_renumberings_compose() {
        let mut ssa = constants(4);
        let [first, second, third, fourth] = [0, 1, 2, 3].map(SsaVarId::from_index);
        ssa.reset_variable_remap();

        kill(&mut ssa, first);
        ssa.compact_variables();
        // `second` now holds id 0, `third` 1, `fourth` 2.
        let second_now = ssa.variable_remap().get(second).expect("second survives");
        kill(&mut ssa, second_now);
        ssa.compact_variables();

        let remap = ssa.variable_remap();
        assert_eq!(remap.get(first), None);
        assert_eq!(remap.get(second), None);
        assert_eq!(remap.get(third), Some(SsaVarId::from_index(0)));
        assert_eq!(remap.get(fourth), Some(SsaVarId::from_index(1)));
        assert_eq!(origin_of(&ssa, third), Some(VariableOrigin::Local(2)));
        assert_eq!(origin_of(&ssa, fourth), Some(VariableOrigin::Local(3)));
    }

    /// A renaming construction renames every value it visits, and the record
    /// follows it.
    ///
    /// Construction replaces each grouped definition with a fresh variable and
    /// each phi with a newly placed one. The originals become orphans the next
    /// compaction removes, so a record that ignored the rename would report
    /// every one of them as gone - or, for a phi result kept as though it were
    /// an entry value, as a variable nothing defines. The report carries the
    /// record, since construction leaves the function's own measured from the
    /// finished ids.
    #[test]
    fn a_renaming_construction_is_folded_into_the_record() {
        let mut ssa = testing::diamond_phi_fixture();
        let merged = ssa
            .blocks()
            .get(3)
            .and_then(|block| block.phi_nodes().first())
            .map(PhiNode::result)
            .expect("the fixture's phi");
        let left = SsaVarId::from_index(1);

        let report = ssa
            .construct_ssa(&SsaConstructOptions::renaming_only())
            .expect("the diamond is constructed");

        let returned = match ssa.block(3).and_then(|block| block.control_terminator()) {
            Some(SsaOp::Return {
                value: Some(value), ..
            }) => *value,
            other => panic!("the join returns a value: {other:?}"),
        };
        assert_eq!(
            report.variables.get(merged),
            Some(returned),
            "the phi's value is what the join returns\n{ssa}"
        );
        let left_now = report
            .variables
            .get(left)
            .expect("the left constant survives");
        assert!(
            matches!(
                ssa.block(1).and_then(|block| block.instructions().first()).map(|i| i.op()),
                Some(SsaOp::Const { dest, value: ConstValue::I32(10) }) if *dest == left_now
            ),
            "the left constant is still the one defined in B1\n{ssa}"
        );
    }

    /// Compaction removes a phi result no phi defines and nothing reads.
    ///
    /// Such a variable has no instruction def site, like a value the caller
    /// supplies at entry, but its def site names the phi's block and its origin
    /// can be the local the phi merged. Kept, it is an orphan the verifier
    /// reports and a caller's variable record can still resolve to.
    #[test]
    fn compaction_removes_a_superseded_phi_result() {
        let mut ssa = constants(1);
        let parameter =
            ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
        let superseded =
            ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::phi(1), MockType::I32);
        ssa.reset_variable_remap();

        ssa.compact_variables();

        assert_eq!(ssa.variable_remap().get(superseded), None, "{ssa}");
        assert!(
            ssa.variable_remap().get(parameter).is_some(),
            "a default-initialized local is defined at entry"
        );
    }

    /// The record answers only for ids the function held at the reset.
    ///
    /// A variable created afterwards takes an id from the current space, which
    /// after a renumbering overlaps the reset-time one. Folding it in at its
    /// current id would make one entry answer for two different variables.
    #[test]
    fn a_variable_created_after_the_reset_is_outside_the_record() {
        let mut ssa = constants(3);
        ssa.reset_variable_remap();
        kill(&mut ssa, SsaVarId::from_index(0));
        ssa.compact_variables();

        let created = ssa.create_variable(
            VariableOrigin::Local(9),
            1,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let later = ssa.create_variable(
            VariableOrigin::Local(10),
            1,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let block = ssa.block_mut(0).expect("the entry block");
        block.instructions_mut().insert(
            0,
            SsaInstruction::synthetic(SsaOp::Const {
                dest: created,
                value: ConstValue::I32(9),
            }),
        );
        block.instructions_mut().insert(
            0,
            SsaInstruction::synthetic(SsaOp::Const {
                dest: later,
                value: ConstValue::I32(10),
            }),
        );
        ssa.reindex_variables();

        assert_eq!(
            ssa.variable_remap().len(),
            3,
            "the domain is fixed at the reset"
        );
        assert_eq!(
            ssa.variable_remap().get(later),
            None,
            "id {later} did not exist at the reset"
        );
        assert_eq!(
            origin_of(&ssa, SsaVarId::from_index(2)),
            Some(VariableOrigin::Local(2))
        );
    }

    /// A function is tracked from its creation: a renumbering before any
    /// reset is recorded, so a caller that never reset still learns where its
    /// values went.
    ///
    /// The removed variable is the last one, so no survivor moves; the record
    /// must still have taken it into its domain, or it would answer the
    /// removed id as itself.
    #[test]
    fn a_renumbering_before_any_reset_is_recorded() {
        let mut ssa = constants(2);
        let [kept, removed] = [0, 1].map(SsaVarId::from_index);
        let block = ssa.block_mut(0).expect("the entry block");
        if let Some(ret) = block.instructions_mut().last_mut() {
            ret.set_op(SsaOp::returns(Some(kept)));
        }
        kill(&mut ssa, removed);
        ssa.compact_variables();

        assert!(!ssa.variable_remap().is_identity());
        assert_eq!(ssa.variable_remap().get(removed), None);
        assert_eq!(ssa.variable_remap().get(kept), Some(kept));

        let mut moving = constants(2);
        kill(&mut moving, kept);
        moving.compact_variables();
        assert_eq!(moving.variable_remap().get(kept), None);
        assert_eq!(moving.variable_remap().get(removed), Some(kept));
    }

    /// A function nothing has renumbered reports the identity, so a caller can
    /// skip the repair entirely.
    #[test]
    fn an_untouched_function_reports_the_identity() {
        let mut ssa = constants(1);
        ssa.reset_variable_remap();

        assert!(ssa.variable_remap().is_identity());
    }

    /// The record survives a clone, which is how a rollback restores it.
    #[test]
    fn a_clone_carries_the_record() {
        let mut ssa = constants(2);
        ssa.reset_variable_remap();
        kill(&mut ssa, SsaVarId::from_index(0));
        ssa.compact_variables();

        let mut restored = constants(1);
        restored.clone_from(&ssa);
        assert_eq!(restored.variable_remap(), ssa.variable_remap());
        assert_eq!(ssa.clone().variable_remap(), ssa.variable_remap());
    }

    /// A function holding an unrenamed placeholder phi must still report a
    /// capacity proportional to its real variables.
    ///
    /// `SsaVarId::PLACEHOLDER` encodes index `u32::MAX - 1`. Folded into the
    /// capacity, it would make this return about 4.29 billion, and every caller
    /// that sizes storage from it - the passes' `BitSet`s and the verifier's
    /// maps - would allocate 512 MB or more. Placeholder phis exist between phi
    /// placement and renaming, so this is reachable on real input, not just in
    /// hand-built IR.
    #[test]
    fn placeholder_does_not_inflate_capacity() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let mut block = SsaBlock::new(0);
        block.add_phi(PhiNode::new(
            SsaVarId::PLACEHOLDER,
            VariableOrigin::Local(0),
        ));
        ssa.add_block(block);

        let capacity = ssa.var_id_capacity();
        assert!(
            capacity < 1024,
            "placeholder inflated var_id_capacity to {capacity}"
        );
    }

    /// The capacity still covers a real id that appears only inside a block,
    /// which is the case the fold exists for.
    #[test]
    fn capacity_covers_block_only_ids() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let mut block = SsaBlock::new(0);
        block.add_phi(PhiNode::new(
            SsaVarId::from_index(42),
            VariableOrigin::Local(0),
        ));
        ssa.add_block(block);

        assert!(ssa.var_id_capacity() > 42, "must cover id 42");
    }

    /// The highest group in use is the highest group in use, not the sentinel
    /// an ungrouped variable carries.
    ///
    /// `u32::MAX` means "no group", and it sits in the same vector as the real
    /// groups. Every caller that mints a fresh group id asks this question, and
    /// a plain `max()` answers `u32::MAX` the moment one variable is ungrouped:
    /// read as "the groups go to the top" it makes the next id saturate, read
    /// as "there are no groups" it makes the next id restart at zero and
    /// collide with every group already in use. Partially-grouped IR is the
    /// normal case for a machine-code lift, whose temporaries carry no group.
    #[test]
    fn the_highest_group_ignores_the_ungrouped_sentinel() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        assert_eq!(ssa.max_rename_group(), None, "nothing is grouped yet");

        let grouped = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);
        let ungrouped =
            ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);
        ssa.set_rename_group(grouped, 3);

        assert_eq!(
            ssa.rename_group(ungrouped),
            u32::MAX,
            "an ungrouped variable carries the sentinel"
        );
        assert_eq!(
            ssa.max_rename_group(),
            Some(3),
            "and the sentinel must not answer for the highest group"
        );
    }

    /// A `double` argument read by an integer conversion and by an
    /// integer-to-float one, as a lift that chose both before it knew what the
    /// argument held writes them.
    fn conversions_of_a_double() -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa = SsaFunction::<MockTarget>::new(1, 0);
        let double = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::F64,
        );
        let word = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 0),
            MockType::U32,
        );
        let converted = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 1),
            MockType::F64,
        );
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::IntConv {
            dest: word,
            operand: double,
            target: MockType::U32,
            overflow_check: false,
            unsigned: true,
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::IntToFloat {
            dest: converted,
            operand: double,
            target: MockType::F64,
            unsigned: false,
            rounding: None,
            environment: None,
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(word))));
        ssa.add_block(entry);
        ssa.recompute_uses();
        (ssa, double)
    }

    /// **Settled as bits, each conversion reads what its operand holds.** The
    /// integer conversion of a `double` reads its bits through the integer of
    /// its width, and so does the integer-to-float conversion, as a signed
    /// integer; each reinterpretation is a new instruction with an identity,
    /// and the function verifies.
    #[test]
    fn settled_as_bits_a_converted_double_is_read_through_its_bits() {
        let (mut ssa, double) = conversions_of_a_double();
        ssa.assign_instr_ids();
        assert_eq!(ssa.settle_conversions(ConversionIntent::Bits), 2);
        assert!(ssa.validate().is_ok(), "{ssa}");
        assert_eq!(
            ssa.settle_conversions(ConversionIntent::Bits),
            0,
            "settled conversions are settled: {ssa}"
        );
        let ops: Vec<&SsaOp<MockTarget>> = ssa
            .blocks()
            .iter()
            .flat_map(SsaBlock::instructions)
            .map(SsaInstruction::op)
            .collect();
        let read_through = |at: usize, ty: MockType| match (ops.get(at), ops.get(at + 1)) {
            (
                Some(SsaOp::Bitcast {
                    dest,
                    operand,
                    target,
                }),
                Some(conversion),
            ) => {
                *operand == double && *target == ty && conversion.converted_operand() == Some(*dest)
            }
            _ => false,
        };
        assert!(
            read_through(0, MockType::U64) && matches!(ops.get(1), Some(SsaOp::IntConv { .. })),
            "{ssa}"
        );
        assert!(
            read_through(2, MockType::I64) && matches!(ops.get(3), Some(SsaOp::IntToFloat { .. })),
            "{ssa}"
        );
        assert!(
            ssa.blocks()
                .iter()
                .flat_map(SsaBlock::instructions)
                .all(|instruction| instruction.id().is_some())
        );
    }

    /// **Construction judges the conversions it settles.** Asked to settle
    /// them, finishing restates the integer conversion of a `double` before
    /// the verdict, which finds nothing; finishing that leaves them as the
    /// front end chose them, the control, reports the conversion it judged.
    #[test]
    fn construction_settles_conversions_before_its_verdict() {
        let finish = |settle: Option<ConversionIntent>| {
            let (mut ssa, _) = conversions_of_a_double();
            let report = ssa
                .finish_construction(&SsaConstructOptions {
                    settle_conversions: settle,
                    ..SsaConstructOptions::default()
                })
                .expect("one block with no predecessors finishes");
            let refused = report
                .findings
                .iter()
                .any(|finding| matches!(finding, VerifierError::InvalidConversion { .. }));
            (report.conversions_settled, refused)
        };
        assert_eq!(finish(Some(ConversionIntent::Bits)), (2, false));
        assert_eq!(finish(None), (0, true));
    }

    /// **Reconstruction leaves conversions to the front end that settles
    /// them.** A front end that lifts a conversion from its destination alone
    /// (CIL `conv.i8` of a `double`) constructs SSA first and settles its
    /// conversions after: construction of the unsettled integer conversion
    /// succeeds, and settled as values it becomes the truncation the verifier
    /// admits.
    #[test]
    fn reconstruction_leaves_conversions_to_their_settling() {
        let (mut ssa, _) = conversions_of_a_double();
        ssa.construct_ssa(&SsaConstructOptions::unverified())
            .expect("an unsettled conversion is no reconstruction defect");
        ssa.settle_conversions(ConversionIntent::Values);
        assert!(ssa.validate().is_ok(), "{ssa}");
    }

    /// Settled as values, the integer conversion of a `double` truncates it and
    /// the integer-to-float one widens it, with nothing placed before either.
    #[test]
    fn settled_as_values_a_converted_double_is_converted() {
        let (mut ssa, _) = conversions_of_a_double();
        assert_eq!(ssa.settle_conversions(ConversionIntent::Values), 2);
        assert_eq!(ssa.settle_conversions(ConversionIntent::Values), 0);
        let ops: Vec<&SsaOp<MockTarget>> = ssa
            .blocks()
            .iter()
            .flat_map(SsaBlock::instructions)
            .map(SsaInstruction::op)
            .collect();
        assert!(
            matches!(
                ops.as_slice(),
                [
                    SsaOp::FloatToInt { .. },
                    SsaOp::FloatConv { .. },
                    SsaOp::Return { .. }
                ]
            ),
            "{ssa}"
        );
    }

    /// B0 outside, a `try` over B1..B3 with its finally in B3, B4 outside.
    #[test]
    fn an_edge_enters_a_region_only_at_its_start_and_leaves_only_where_it_did() {
        const PROTECTED: (usize, usize) = (1, 3);
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        for index in 0..5 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(PROTECTED.0, PROTECTED.1),
            handler_range: BlockRange::new(PROTECTED.1, PROTECTED.1 + 1),
            filter_range: None,
            selector: None,
        }]);

        assert!(
            ssa.retarget_respects_regions(0, 4, 1),
            "into the try at its start"
        );
        assert!(
            !ssa.retarget_respects_regions(0, 1, 2),
            "into the try's middle"
        );
        assert!(!ssa.retarget_respects_regions(0, 4, 3), "into the handler");
        assert!(ssa.retarget_respects_regions(1, 2, 2), "within the try");
        assert!(
            !ssa.retarget_respects_regions(1, 2, 4),
            "out of the try where the edge stayed inside"
        );
        assert!(
            ssa.retarget_respects_regions(2, 4, 0),
            "out of the try where the edge already left it"
        );
    }
}
