//! Mutation and transform methods for SSA functions.
//!
//! Provides all operations that modify an [`SsaFunction`] - replacing uses,
//! eliminating phi nodes, folding constants, compacting variables, and
//! optimizing local variable layout.
//!
//! # Variable Replacement Architecture
//!
//! Two primitives with different safety profiles:
//!
//! | Primitive | Scope | Safety |
//! |-----------|-------|--------|
//! | [`replace_uses(old, new)`] | Instructions only | Safe for passes |
//! | [`replace_uses_including_phis(old, new)`] | Instructions + phi operands | Internal only |
//!
//! **`replace_uses`** (instruction uses only) is the safe default for compiler passes.
//! It never makes a phi read a value of another rename group: a renaming
//! construction fills each phi operand from the phi's own group, so an operand
//! naming another group's value does not survive it.
//!
//! **`replace_uses_including_phis`** also replaces phi operands.
//! Needed for infrastructure operations like trivial phi elimination where the
//! eliminated phi and its forwarding target share the same origin context.
//!
//! ## Self-Referential Guard
//!
//! Both methods skip replacements where the instruction's destination equals `new_var`,
//! preventing self-referential instructions like `v0 = add(v0, v1)`. The
//! [`ReplaceResult`] reports both successful replacements and skips.
//!
//! ## High-Level Operations
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | `propagate_copies` | Batch copy propagation with completion tracking |
//! | `eliminate_trivial_phis` | Iterative trivial phi removal to fixpoint |
//! | `prune_phi_operands` | Remove stale operands after CFG changes |
//! | `fold_constant` | Replace an instruction with its constant result |
//! | `compact_variables` | Remove orphaned variables and reindex |
//! | `strip_nops` | Remove Nop instructions and fix DefSites |
//! | `remove_unread_instructions` | Remove listed instructions whose results nothing reads |
//! | `recompute_uses` | Rebuild use-site tracking from scratch |

use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};

use crate::{
    BitSet, Result,
    analysis::{
        cfg::SsaCfg, context::AnalysisContext, defuse::DefUseIndex, exceptions::FunctionRoots,
        memory::MemorySsa,
    },
    facts::{InstructionFacts, Rewrite, Substitution},
    graph::{
        DirectedGraph, NodeId,
        algorithms::{compute_dominators, strongly_connected_components},
    },
    ir::{
        block::ReplaceResult,
        exception::{HandlerKind, SsaExceptionHandler},
        function::{
            MachinePoint, SsaEditOptions, SsaFunction, SsaRollbackPolicy, StatedRegister, Widening,
            editor::Rewiring,
        },
        identity::InstrId,
        instruction::SsaInstruction,
        ops::{
            OperandRole, RuntimeValueKind, SsaOp,
            calls::{CallInput, CallInputKey},
        },
        phi::PhiOperand,
        value::ConstValue,
        variable::{DefSite, SsaVarId, SsaVariable, UseSite, VariableOrigin},
        varstore::VarSet,
    },
    passes::{
        memory::{dead_frame_stores, unread_loads},
        utils::resolve_chains,
    },
    target::Target,
};

/// Options for trivial phi elimination.
pub struct TrivialPhiOptions<'a> {
    /// If set, only consider phis in reachable blocks and use reachability-aware
    /// self-referential checks. Unreachable predecessor operands are filtered out
    /// as a second-pass check. All trivial phis are removed unconditionally.
    ///
    /// If `None`, all blocks are considered. Chain resolution is applied, and only
    /// fully propagated phis (no skipped uses from the self-ref guard) are removed.
    pub reachable: Option<&'a BitSet>,
}

/// What one walk over a function finds about the variables it uses.
///
/// Repair asks the same two questions of the function several times - which
/// variables does something define, and which does something read - and a
/// walk that answers both is the only walk it needs while the phis and the
/// instructions stay as they are.
pub struct VariableCensus {
    /// Variables some phi or non-`Nop` instruction defines.
    defined: VarSet,
    /// Variables some phi operand or non-`Nop` instruction reads.
    read: VarSet,
}

impl VariableCensus {
    /// Walks `ssa` once and reports which variables it defines and reads.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to walk.
    pub fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let bound = ssa.var_id_bound();
        let mut defined = VarSet::new(bound);
        let mut read = VarSet::new(bound);
        for block in ssa.blocks.iter() {
            for phi in block.phi_nodes() {
                defined.insert(phi.result());
                for operand in phi.operands() {
                    read.insert(operand.value());
                }
            }
            for instr in block.instructions() {
                if matches!(instr.op(), SsaOp::Nop) {
                    continue;
                }
                instr.op().visit_operands(|role, var| match role {
                    OperandRole::Use => {
                        read.insert(var);
                    }
                    OperandRole::Def | OperandRole::FlagsDef => {
                        defined.insert(var);
                    }
                });
            }
        }
        Self { defined, read }
    }

    /// Restores every variable's definition site as
    /// [`SsaFunction::refresh_def_sites`] does, and returns what the same walk
    /// found about which variables are defined and read.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function whose definition sites to restore.
    pub fn refresh<T: Target>(ssa: &mut SsaFunction<T>) -> Self {
        let bound = ssa.var_id_bound();
        let mut defined = VarSet::new(bound);
        let mut read = VarSet::new(bound);

        for (block_idx, block) in ssa.blocks.iter().enumerate() {
            for phi in block.phi_nodes() {
                let result = phi.result();
                defined.insert(result);
                if !result.is_placeholder()
                    && let Some(row) = ssa.variables.get_mut(result.index())
                {
                    row.set_def_site(DefSite::phi(block_idx));
                }
                for operand in phi.operands() {
                    read.insert(operand.value());
                }
            }

            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if matches!(instr.op(), SsaOp::Nop) {
                    continue;
                }
                instr.op().visit_operands(|role, var| match role {
                    OperandRole::Use => {
                        read.insert(var);
                    }
                    OperandRole::Def | OperandRole::FlagsDef => {
                        defined.insert(var);
                        if !var.is_placeholder()
                            && let Some(row) = ssa.variables.get_mut(var.index())
                        {
                            row.set_def_site(DefSite::instruction(block_idx, instr_idx));
                        }
                    }
                });
            }
        }

        for var in &mut ssa.variables {
            let id = var.id();
            if !id.is_placeholder()
                && !defined.contains(id)
                && !read.contains(id)
                && var.def_site().block() != 0
            {
                var.set_def_site(DefSite::entry());
            }
        }
        Self { defined, read }
    }

    /// Compacts the variable table, given what the function defines and reads.
    ///
    /// A variable is considered orphaned if:
    /// - It's not defined by any instruction in any block
    /// - It's not defined by any phi node in any block
    /// - No instruction or phi operand reads it
    /// - It is not a caller-supplied entry value: a version-0 `Argument`,
    ///   `Local` or `EntryLiveIn` whose def site is [`DefSite::entry`]
    ///
    /// A variable that something still reads is kept even when nothing defines
    /// it. This can happen when `replace_uses` skips replacements due to the
    /// self-referential guard (dest == new_var), leaving uses behind after the
    /// definition was Nop'd or eliminated, and when a phi names a variable whose
    /// defining instruction was Nop'd - without it, compact would remove the
    /// variable and the phi operand would become a dangling reference.
    ///
    /// Also keeps the values the caller supplies. These have no definition in
    /// the body but are implicitly defined at function entry: method
    /// parameters, default-initialized locals, and live-in storage. Any other
    /// variable with no definition and no reader is gone - notably the result
    /// of a phi that was eliminated or replaced (its def site still names the
    /// phi's block), which kept here would be an orphan a caller's variable
    /// record could still resolve to.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function this census was taken of, unchanged since.
    ///
    /// # Returns
    ///
    /// The number of variables that were removed.
    pub fn compact<T: Target>(&self, ssa: &mut SsaFunction<T>) -> usize {
        let count = ssa.variables.len();
        let supplies_entry_value = |var: &SsaVariable<T>| {
            var.version() == 0
                && var.def_site() == DefSite::entry()
                && matches!(
                    var.origin(),
                    VariableOrigin::Argument(_)
                        | VariableOrigin::Local(_)
                        | VariableOrigin::EntryLiveIn
                )
        };

        // Most repairs find nothing to remove, and then renumbering is the
        // identity: the table is dense, so every id stays, and the groups and
        // the allocator already describe exactly the table. Only the
        // bookkeeping that follows any compaction is left to do. In a dense
        // table every id belongs to one row, so a row is kept by its own facts.
        let nothing_to_remove = ssa.rename_groups.describes(count)
            && ssa.var_allocator.count() == count
            && ssa.variables.iter().enumerate().all(|(slot, var)| {
                var.id().index() == slot
                    && (self.defined.contains(var.id())
                        || self.read.contains(var.id())
                        || supplies_entry_value(var))
            });
        if nothing_to_remove {
            ssa.record_renumbering(count, |id| (id.index() < count).then_some(id));
            ssa.rebuild_origin_versions();
            return 0;
        }

        // The values the caller supplies, by id: a row that supplies its id
        // keeps every row that shares it.
        let mut supplied = VarSet::new(ssa.var_id_bound());
        for var in &ssa.variables {
            if supplies_entry_value(var) {
                supplied.insert(var.id());
            }
        }
        let keeps = |var: &SsaVariable<T>| {
            let id = var.id();
            self.defined.contains(id) || self.read.contains(id) || supplied.contains(id)
        };

        let original_count = count;
        ssa.variables.retain(keeps);
        // Reassign dense IDs and rebuild registries
        let renumbering = ssa.reassign_dense_ids();
        ssa.remap_var_ids_in_blocks(&renumbering);
        ssa.rebuild_origin_versions();
        original_count.saturating_sub(ssa.variables.len())
    }
}

/// The `Nop`s one block lost to [`SsaFunction::strip_nops`].
struct StrippedNops {
    /// Where they stood, ascending.
    positions: Vec<usize>,
    /// How many instructions the block held before.
    original_len: usize,
}

/// What a phi's operands, other than references to its own result, name.
enum SoleSource {
    /// Nothing: the phi has no operands, or only ones naming its own result.
    None,
    /// Exactly one other variable.
    One(SsaVarId),
    /// More than one.
    Many,
}

impl SoleSource {
    /// Classifies `values` against `result`.
    fn of(values: impl Iterator<Item = SsaVarId>, result: SsaVarId) -> Self {
        let mut sole = Self::None;
        for value in values.filter(|&value| value != result) {
            match sole {
                Self::None => sole = Self::One(value),
                Self::One(first) if first == value => {}
                Self::One(_) | Self::Many => return Self::Many,
            }
        }
        sole
    }
}

/// Result of batch copy propagation.
pub struct CopyPropagationResult {
    /// Total number of uses replaced across all copies.
    pub total_replaced: usize,
    /// Set of copy destinations that were fully propagated (all uses replaced).
    /// These copies can safely be Nop'd by the caller.
    pub fully_propagated: VarSet,
    /// Set of copy destinations that still have remaining instruction uses
    /// (due to self-referential guard). These copies must be kept alive.
    pub partially_propagated: VarSet,
}

/// Reports whether `var` is still read anywhere in the function, counting phi
/// operands as reads.
///
/// [`simplify_phi_to_copy`](SsaFunction::simplify_phi_to_copy) removes a phi
/// only once nothing outside that phi still reads its result, and a surviving
/// phi operand counts: dropping the definition while another phi names it would
/// strand that operand. `skip_phi` excludes the phi under consideration so its
/// own operands cannot keep it alive.
///
/// # Arguments
///
/// * `ssa` - The function to scan.
/// * `var` - The variable whose remaining reads are being counted.
/// * `skip_phi` - `(block id, phi index)` of a phi to ignore, or `None` to
///   count every phi.
///
/// # Returns
///
/// `true` if any instruction use or non-skipped phi operand still reads `var`.
fn has_remaining_uses_including_phis<T: Target>(
    ssa: &SsaFunction<T>,
    var: SsaVarId,
    skip_phi: Option<(usize, usize)>,
) -> bool {
    ssa.blocks().iter().any(|block| {
        block.instructions().iter().any(|instr| {
            let mut found = false;
            instr.op().for_each_use(|used| found |= used == var);
            found
        }) || block
            .phi_nodes()
            .iter()
            .enumerate()
            .filter(|(phi_idx, _)| skip_phi != Some((block.id(), *phi_idx)))
            .any(|(_, phi)| phi.operands().iter().any(|operand| operand.value() == var))
    })
}

impl<T: Target> SsaFunction<T> {
    /// Records the registers the function's code names and its entry value
    /// of each ([`MachineStates`](crate::ir::function::MachineStates)).
    pub fn set_machine_registers(&mut self, named: Box<[CallInputKey]>, entries: Box<[CallInput]>) {
        self.machine.set_registers(named, entries);
    }

    /// Records the register map of `point` at the instruction `at`
    /// ([`MachineStates`](crate::ir::function::MachineStates)).
    pub fn set_machine_state(&mut self, at: InstrId, point: MachinePoint, state: Box<[CallInput]>) {
        self.machine.insert(at, point, state);
    }

    /// Records the register map control enters the host block `host` with
    /// ([`MachineStates`](crate::ir::function::MachineStates)).
    pub fn set_host_entry_state(&mut self, host: u64, state: Box<[CallInput]>) {
        self.machine.insert_host(host, state);
    }

    /// The keys of every register the function's code reads or writes.
    #[must_use]
    pub fn named_registers(&self) -> &[CallInputKey] {
        self.machine.named()
    }

    /// The function's entry value of each register it reads before writing,
    /// under the register's key.
    #[must_use]
    pub fn register_entries(&self) -> &[CallInput] {
        self.machine.entries()
    }

    /// The machine state execution goes on with after the instruction `at`,
    /// where a write it made changed the code: each register's value its
    /// record stated, followed through the substitutions since, where
    /// something still defines it. Empty where `at` is no resume point.
    #[must_use]
    pub fn resume_state(&self, at: InstrId) -> Vec<StatedRegister> {
        self.machine
            .state(at, MachinePoint::After, |value| self.is_defined(value))
    }

    /// The registers the instruction `at` hands on as it transfers: what a
    /// call's callee enters with, what a return leaves its caller.
    #[must_use]
    pub fn handover_state(&self, at: InstrId) -> Vec<StatedRegister> {
        self.machine
            .state(at, MachinePoint::Handover, |value| self.is_defined(value))
    }

    /// The machine state execution goes on with where control enters the
    /// host block `host` ([`BlockProvenance`](crate::ir::block::BlockProvenance))
    /// after a write changed its code: each register's value as control
    /// brings it there, read as [`Self::resume_state`] reads a resume point's.
    /// Empty where the record names no entry of `host`.
    #[must_use]
    pub fn host_entry_state(&self, host: u64) -> Vec<StatedRegister> {
        self.machine
            .host_state(host, |value| self.is_defined(value))
    }

    /// Every recorded host block entry map, in order, each as it is read.
    pub fn host_entry_states(&self) -> impl Iterator<Item = (u64, Vec<StatedRegister>)> + '_ {
        self.machine
            .hosts()
            .map(|host| (host, self.host_entry_state(host)))
    }

    /// Whether execution can go on in a lift of changed code after `at`.
    #[must_use]
    pub fn is_resume_point(&self, at: InstrId) -> bool {
        self.machine.contains(at, MachinePoint::After)
    }

    /// Every recorded register map, in order, each as its point states it.
    pub fn machine_states(
        &self,
    ) -> impl Iterator<Item = (InstrId, MachinePoint, Vec<StatedRegister>)> + '_ {
        self.machine.points().map(|(at, point)| {
            (
                at,
                point,
                self.machine
                    .state(at, point, |value| self.is_defined(value)),
            )
        })
    }

    /// Drops the operands only static execution reads, and what only they
    /// kept: each return's `target`, the value the executor checks a return
    /// goes back through, and its `handover`, the registers it hands its
    /// caller; each unresolved branch's `state`, the registers execution
    /// hands to the code it lifts there; and each call input
    /// `execution_input` names as one only execution reads. A host that is
    /// done executing the function releases them before it renders it.
    ///
    /// The values they read go with them where nothing else reads them, and
    /// so, in turn, what only those values read: the reload of a saved link
    /// register (a load that cannot fault, [`unread_loads`]), the address it
    /// was loaded from. So does a store into the
    /// frame that only such a reload observed: the save of that register,
    /// dead at the frame's exit ([`dead_frame_stores`]) once the reload is
    /// gone and live before. A definition anything else reads stays, as does
    /// every write to a slot the function reads as storage, and nothing the
    /// released operands never reached moves: dead code they did not keep is
    /// not theirs to remove. A value a phi defines stops the walk, the phi
    /// kept. The machine states ([`Self::resume_state`]) go too; they keep
    /// nothing alive.
    ///
    /// # Arguments
    ///
    /// * `context` - The analysis context the frame's stores are judged
    ///   under.
    /// * `execution_input` - Whether a call input's key names one only
    ///   execution reads.
    ///
    /// # Errors
    ///
    /// Returns an error where repairing the function after the removals
    /// fails; the function is then as it was before them, the operands
    /// released.
    ///
    /// # Returns
    ///
    /// How many definitions went with the operands, or `None` where the
    /// function held no operand only execution reads. Even where none went,
    /// the values the operands read are read less, which a rewrite that
    /// counts a value's reads may now act on.
    pub fn release_execution_operands(
        &mut self,
        context: &AnalysisContext<'_>,
        execution_input: impl Fn(CallInputKey) -> bool,
    ) -> Result<Option<usize>> {
        self.machine.clear();
        let dead_before = dead_frame_stores(self, &MemorySsa::build(self, context), context);
        let mut released: Vec<SsaVarId> = Vec::new();
        for block in self.blocks_mut() {
            for instruction in block.instructions_mut() {
                match instruction.op_mut() {
                    SsaOp::Return { transfer, .. } => {
                        if let Some(transfer) = transfer.take() {
                            released.extend(transfer.target);
                            released.extend(transfer.handover.iter().map(|input| input.value));
                        }
                    }
                    SsaOp::IndirectBranch { state, .. } => {
                        released.extend(mem::take(state).iter().map(|input| input.value));
                    }
                    SsaOp::Call(call) => {
                        call.implicit_inputs.retain(|input| {
                            let only_execution = execution_input(input.key);
                            if only_execution {
                                released.push(input.value);
                            }
                            !only_execution
                        });
                    }
                    _ => {}
                }
            }
        }
        if released.is_empty() {
            return Ok(None);
        }
        self.recompute_uses();
        let options = SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure);
        // Every definition the released values reach through what defines
        // them, and through what a save that goes with them read: each a
        // candidate the removal settles, which one still read elsewhere
        // survives, keeping what it reads. Held by the defining instruction's
        // identity: each removal's repair renumbers the values, never the
        // identities. A value no instruction defines (a phi's, an entry
        // value) is no candidate.
        let mut reached: BTreeSet<InstrId> = BTreeSet::new();
        let mut removed = 0usize;
        loop {
            let mut walked: BTreeSet<SsaVarId> = BTreeSet::new();
            while let Some(value) = released.pop() {
                if !walked.insert(value) {
                    continue;
                }
                let Some(site) = self.variable(value).map(SsaVariable::def_site) else {
                    continue;
                };
                let Some(instruction) = site
                    .index()
                    .and_then(|index| self.block(site.block())?.instructions().get(index))
                else {
                    continue;
                };
                if let Some(id) = instruction.id() {
                    reached.insert(id);
                }
                instruction.op().for_each_use(|read| released.push(read));
            }
            // A pure definition goes once nothing reads it, and an unread load
            // that cannot fault ([`unread_loads`]); each removal may leave
            // another unread, so the two alternate until neither removes one.
            // The memory the last search for unread loads read, while nothing
            // was removed after it: what the saves are read from next.
            let mut unchanged: Option<MemorySsa<T>> = None;
            loop {
                let memory = MemorySsa::build(self, context);
                let loads = unread_loads(self, &memory, context);
                let mut sites: Vec<(usize, usize)> = Vec::new();
                for (block_idx, block) in self.blocks().iter().enumerate() {
                    for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                        let op = instruction.op();
                        let removable = op.dest().is_some_and(|dest| {
                            (op.effects().removable_when_unused()
                                && !matches!(op, SsaOp::Pop { .. }))
                                || loads.contains(&dest)
                        });
                        if removable && instruction.id().is_some_and(|id| reached.contains(&id)) {
                            sites.push((block_idx, instr_idx));
                        }
                    }
                }
                if sites.is_empty() {
                    unchanged = Some(memory);
                    break;
                }
                let mut round = 0;
                self.edit(options, |editor| {
                    round = editor.remove_unread_instructions(sites).len();
                    Ok(())
                })?;
                if round == 0 {
                    break;
                }
                removed = removed.saturating_add(round);
            }
            if removed == 0 {
                return Ok(Some(0));
            }
            // The saves only the removed reloads observed, and what they
            // read in turn.
            let memory = unchanged.unwrap_or_else(|| MemorySsa::build(self, context));
            let saves: BTreeSet<InstrId> = dead_frame_stores(self, &memory, context)
                .difference(&dead_before)
                .copied()
                .collect();
            let mut sites: Vec<(usize, usize)> = Vec::new();
            for (block_idx, block) in self.blocks().iter().enumerate() {
                for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                    if instruction.id().is_some_and(|id| saves.contains(&id)) {
                        sites.push((block_idx, instr_idx));
                        instruction.op().for_each_use(|read| released.push(read));
                    }
                }
            }
            if sites.is_empty() {
                return Ok(Some(removed));
            }
            let mut round = 0;
            self.edit(options, |editor| {
                round = editor.remove_unread_instructions(sites).len();
                Ok(())
            })?;
            if round == 0 {
                return Ok(Some(removed));
            }
            removed = removed.saturating_add(round);
        }
    }

    /// Makes the values each handler phi receives on a throw fit one place.
    ///
    /// No copy runs on an exceptional edge, so a lowering keeps a handler
    /// phi's result and every value a throw hands it in one place
    /// ([`Self::exceptional_webs`]). That is sound only while those values
    /// are never needed at once, which holds for the versions of one variable
    /// as a lift produces them, and stops holding once a rewrite reuses a
    /// value. Two steps restore it:
    ///
    /// 1. A value also read outside the phi, or held by another exceptional
    ///    phi, is replaced in the phi's exceptional operands by a copy defined
    ///    right after it. Global value numbering finds that two locals both
    ///    start as zero and hands one zero to both: without the copy the
    ///    version of one local the handler reads is another local's value
    ///    too, still needed after the first local was overwritten.
    /// 2. Where one of the phi's values is written while another is still
    ///    needed - a block the other reaches the handler from comes after the
    ///    write, around a loop - each exceptional operand not defined in its
    ///    own block becomes a copy made at that block's start. A local the
    ///    original code reset every iteration to a value it held before the
    ///    loop is one value in SSA once the resets are folded, and a place
    ///    shared with the version written mid-iteration would hold that
    ///    version when the next iteration throws; the copies re-establish the
    ///    value where the original stored it.
    ///
    /// # Returns
    ///
    /// The number of copies inserted.
    ///
    /// # Errors
    ///
    /// Returns an error when an insertion fails; the function is then as it
    /// was when the failing step began.
    pub fn isolate_exceptional_operands(&mut self) -> Result<usize> {
        let shared = self.copy_shared_exceptional_operands()?;
        let conflicting = self.copy_conflicting_exceptional_operands()?;
        Ok(shared.saturating_add(conflicting))
    }

    /// Step 1 of [`Self::isolate_exceptional_operands`]: copies, right after
    /// their definitions, the exceptional operands read anywhere else or held
    /// by another exceptional phi.
    fn copy_shared_exceptional_operands(&mut self) -> Result<usize> {
        let webs = self.exceptional_webs();
        if webs.is_empty() {
            return Ok(0);
        }
        let mut membership: BTreeMap<SsaVarId, usize> = BTreeMap::new();
        for web in &webs {
            for value in &web.values {
                let count = membership.entry(*value).or_insert(0);
                *count = count.saturating_add(1);
            }
        }
        let defuse = DefUseIndex::build(self);
        // (block, phi, value) for every operand to isolate, in phi order.
        let mut plan: Vec<(usize, usize, SsaVarId)> = Vec::new();
        for web in &webs {
            for &value in web.values.iter().skip(1) {
                let shared = membership.get(&value).is_some_and(|count| *count > 1);
                let read_elsewhere = defuse.uses_of(value).is_some_and(|uses| {
                    uses.iter().any(|site| {
                        !(site.is_phi_operand()
                            && site.block() == web.block
                            && site.index() == web.phi_idx)
                    })
                });
                if shared || read_elsewhere {
                    plan.push((web.block, web.phi_idx, value));
                }
            }
        }
        if plan.is_empty() {
            return Ok(0);
        }

        let mut copies: Vec<(usize, usize, SsaVarId, SsaVarId)> = Vec::with_capacity(plan.len());
        let options = SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure);
        self.edit(options, |editor| {
            for &(block, phi_idx, value) in &plan {
                let Some(variable) = editor.function().variable(value) else {
                    continue;
                };
                let site = variable.def_site();
                let var_type = variable.var_type().clone();
                // After the instruction defining it, or - for a phi result or
                // an entry value - first in its block, behind the exception
                // object a handler block defines before anything else.
                let at = site.index().map_or_else(
                    || {
                        editor.function().block(site.block()).map_or(0, |block| {
                            block
                                .instructions()
                                .iter()
                                .take_while(|instruction| {
                                    matches!(
                                        instruction.op(),
                                        SsaOp::RuntimeValue {
                                            kind: RuntimeValueKind::ExceptionObject,
                                            ..
                                        }
                                    )
                                })
                                .count()
                        })
                    },
                    |index| index.saturating_add(1),
                );
                let copy = editor.create_fresh_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(site.block(), at),
                    var_type,
                );
                editor.insert_instruction(
                    site.block(),
                    at,
                    SsaInstruction::synthetic(SsaOp::Copy {
                        dest: copy,
                        src: value,
                    }),
                )?;
                copies.push((block, phi_idx, value, copy));
            }
            Ok(())
        })?;
        for &(block, phi_idx, value, copy) in &copies {
            let Some(phi) = self
                .block_mut(block)
                .and_then(|block| block.phi_nodes_mut().get_mut(phi_idx))
            else {
                continue;
            };
            for operand in phi.operands_mut() {
                if operand.is_exceptional() && operand.value() == value {
                    *operand = PhiOperand::exceptional(copy, operand.predecessor());
                }
            }
        }
        self.recompute_uses();
        Ok(copies.len())
    }

    /// Step 2 of [`Self::isolate_exceptional_operands`]: in each exceptional
    /// phi one of whose values is written while another is still needed,
    /// copies each exceptional operand not defined in its own block at that
    /// block's start.
    fn copy_conflicting_exceptional_operands(&mut self) -> Result<usize> {
        let webs = self.exceptional_webs();
        if webs.is_empty() {
            return Ok(0);
        }
        let cfg = SsaCfg::from_ssa(self);
        let block_count = self.block_count();
        // The blocks reachable from `block` over at least one edge.
        let after = |block: usize| {
            let mut seen = BitSet::new(block_count);
            let mut pending: Vec<usize> = cfg.block_successors(block).to_vec();
            while let Some(next) = pending.pop() {
                if next < block_count && seen.insert(next) {
                    pending.extend(cfg.block_successors(next).iter().copied());
                }
            }
            seen
        };
        let defining_block = |value: SsaVarId| {
            self.variable(value)
                .map(|variable| variable.def_site().block())
        };

        // (handler block, phi, value, throwing block) for every operand to copy.
        let mut plan: Vec<(usize, usize, SsaVarId, usize)> = Vec::new();
        for web in &webs {
            let Some(phi) = self
                .block(web.block)
                .and_then(|block| block.phi_nodes().get(web.phi_idx))
            else {
                continue;
            };
            let mut throws_from: BTreeMap<SsaVarId, BTreeSet<usize>> = BTreeMap::new();
            for operand in phi.exceptional_operands() {
                throws_from
                    .entry(operand.value())
                    .or_default()
                    .insert(operand.predecessor());
            }
            let conflict = throws_from.keys().any(|&written| {
                // An entry value - neither an instruction nor a phi defines it -
                // is written before anything runs.
                let entry_value = self.get_definition(written).is_none()
                    && self.find_phi_defining(written).is_none();
                let Some(block) = defining_block(written).filter(|_| !entry_value) else {
                    return false;
                };
                let reached = after(block);
                throws_from.iter().any(|(&needed, blocks)| {
                    needed != written && blocks.iter().any(|&from| reached.contains(from))
                })
            });
            if !conflict {
                continue;
            }
            for (&value, blocks) in &throws_from {
                for &from in blocks {
                    if defining_block(value) != Some(from) {
                        plan.push((web.block, web.phi_idx, value, from));
                    }
                }
            }
        }
        if plan.is_empty() {
            return Ok(0);
        }

        let mut copies: Vec<(usize, usize, SsaVarId, usize, SsaVarId)> =
            Vec::with_capacity(plan.len());
        let options = SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure);
        self.edit(options, |editor| {
            for &(block, phi_idx, value, from) in &plan {
                let Some(var_type) = editor
                    .function()
                    .variable(value)
                    .map(|variable| variable.var_type().clone())
                else {
                    continue;
                };
                // First in the block, behind the exception object a handler
                // block defines before anything else.
                let at = editor.function().block(from).map_or(0, |block| {
                    block
                        .instructions()
                        .iter()
                        .take_while(|instruction| {
                            matches!(
                                instruction.op(),
                                SsaOp::RuntimeValue {
                                    kind: RuntimeValueKind::ExceptionObject,
                                    ..
                                }
                            )
                        })
                        .count()
                });
                let copy = editor.create_fresh_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(from, at),
                    var_type,
                );
                editor.insert_instruction(
                    from,
                    at,
                    SsaInstruction::synthetic(SsaOp::Copy {
                        dest: copy,
                        src: value,
                    }),
                )?;
                copies.push((block, phi_idx, value, from, copy));
            }
            Ok(())
        })?;
        for &(block, phi_idx, value, from, copy) in &copies {
            let Some(phi) = self
                .block_mut(block)
                .and_then(|block| block.phi_nodes_mut().get_mut(phi_idx))
            else {
                continue;
            };
            for operand in phi.operands_mut() {
                if operand.is_exceptional()
                    && operand.value() == value
                    && operand.predecessor() == from
                {
                    *operand = PhiOperand::exceptional(copy, from);
                }
            }
        }
        self.recompute_uses();
        Ok(copies.len())
    }

    /// Gives the exception object a definition wherever a handler is entered
    /// holding one.
    ///
    /// The migration for IR built against the older model, in which the object
    /// the runtime pushes had no definition at all and a leading
    /// [`SsaOp::Pop`] at the handler entry stood for it. Nothing produced the
    /// value, so every check that asks where a value comes from had to carve
    /// the shape out by position, and every pass that rewrote the entry block
    /// could silently take the carve-out away.
    ///
    /// The rule is narrow on purpose. Only a catch or filter entry is entered
    /// with an object; only a *leading* `Pop` can be the one consuming it,
    /// since anything before it would already have run; and only a `Pop` whose
    /// value nothing defines is the legacy shape rather than an ordinary
    /// discard. Each one found becomes an
    /// [`SsaOp::RuntimeValue`] defining
    /// the same variable, which says what the `Pop` only implied. The discard
    /// itself needs no instruction: a value nothing reads is removed by dead
    /// code elimination like any other.
    ///
    /// A host that emits `RuntimeValue` directly never needs this; calling it
    /// on such a function finds nothing and changes nothing.
    ///
    /// # Returns
    ///
    /// The number of definitions adopted.
    pub fn adopt_handler_entry_values(&mut self) -> usize {
        let entries: Vec<usize> = self
            .exception_handlers()
            .iter()
            .filter(|handler| matches!(handler.kind(), HandlerKind::Catch | HandlerKind::Filter))
            .flat_map(SsaExceptionHandler::entry_blocks)
            .filter(|block| *block < self.blocks().len())
            .collect();
        if entries.is_empty() {
            return 0;
        }

        let mut defined = VarSet::new(self.var_id_bound());
        for block in self.blocks() {
            for phi in block.phi_nodes() {
                defined.insert(phi.result());
            }
            for instr in block.instructions() {
                for dest in instr.op().defs() {
                    defined.insert(dest);
                }
            }
        }

        let mut adopted = 0usize;
        for entry in entries {
            let Some(block) = self.block_mut(entry) else {
                continue;
            };
            let Some(instr) = block.instructions_mut().first_mut() else {
                continue;
            };
            let SsaOp::Pop { value } = *instr.op() else {
                continue;
            };
            if defined.contains(value) {
                continue;
            }
            instr.set_op_preserving_type(
                SsaOp::RuntimeValue {
                    dest: value,
                    kind: RuntimeValueKind::ExceptionObject,
                },
                Rewrite::Host,
            );
            defined.insert(value);
            if let Some(variable) = self.variable_mut(value) {
                variable.set_def_site(DefSite::instruction(entry, 0));
            }
            adopted = adopted.saturating_add(1);
        }

        if adopted > 0 {
            self.recompute_uses();
        }
        adopted
    }

    /// Removes every phi from a block nothing can reach, declaring its result
    /// runtime-supplied instead.
    ///
    /// A phi merges the values that reach a block, and control reaches one two
    /// ways: a terminator transfers to it, or a throw inside a protected region
    /// dispatches to the handler guarding it. A handler with either kind of
    /// predecessor keeps its phis - the exceptional ones name the blocks a throw
    /// may leave, which is what [`PhiEdge::Exceptional`] expresses. Only a root
    /// with *neither* has nothing for an operand to name, which is a handler
    /// whose region no longer exists.
    ///
    /// When a removed phi's result is still used, a
    /// [`RuntimeValueKind::HandlerEntry`] definition takes its place: the
    /// handler does read the value, and entering it is what makes it hold, so
    /// the value stays defined without asserting a merge over edges that cannot
    /// exist. Nothing reads it and the definition goes, like any other whose
    /// result is dead.
    ///
    /// [`PhiEdge::Exceptional`]: crate::ir::phi::PhiEdge::Exceptional
    ///
    /// # Returns
    ///
    /// The number of phis removed.
    pub fn demote_runtime_entry_phis(&mut self) -> usize {
        let exception_entries: Vec<usize> = {
            let blocks = self.exception_blocks();
            if blocks.is_empty() {
                return 0;
            }
            blocks.runtime_entries().to_vec()
        };

        let unreachable_entries: Vec<usize> = {
            let cfg = SsaCfg::from_ssa(self);
            let roots = FunctionRoots::of(self);
            exception_entries
                .into_iter()
                .filter(|entry| {
                    cfg.block_predecessors(*entry).is_empty()
                        && roots.exceptional_predecessors(*entry).is_empty()
                })
                .collect()
        };

        let mut demoted: usize = 0;
        for entry in unreachable_entries {
            let results: Vec<SsaVarId> = self
                .block(entry)
                .map(|block| block.phi_nodes().iter().map(|phi| phi.result()).collect())
                .unwrap_or_default();
            if results.is_empty() {
                continue;
            }

            if let Some(block) = self.block_mut(entry) {
                block.phi_nodes_mut().clear();
            }
            demoted = demoted.saturating_add(results.len());

            // One use census per entry: `count_uses` is a whole-function walk.
            let use_counts = self.count_uses();
            for (offset, result) in results.into_iter().enumerate() {
                if use_counts.get(&result).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let Some(block) = self.block_mut(entry) else {
                    continue;
                };
                // Leading, and in the order the phis stood in: the definition
                // has to precede every read of it in the handler, and the
                // verifier requires nothing else to run first.
                let at = offset.min(block.instructions().len());
                block.instructions_mut().insert(
                    at,
                    SsaInstruction::synthetic(SsaOp::RuntimeValue {
                        dest: result,
                        kind: RuntimeValueKind::HandlerEntry,
                    }),
                );
                if let Some(variable) = self.variable_mut(result) {
                    variable.set_def_site(DefSite::instruction(entry, at));
                }
            }
        }

        demoted
    }

    /// Replaces all uses of `old_var` with `new_var` throughout the function.
    ///
    /// This is the core operation for copy propagation - when we know that
    /// `v1 = v0` (a copy), we can replace all uses of `v1` with `v0`.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose uses are rewritten.
    /// * `new_var` - The variable substituted in for each use.
    ///
    /// # Returns
    ///
    /// A [`ReplaceResult`] summing, across all blocks, the uses replaced and
    /// the uses skipped by the self-referential guard.
    ///
    /// # Note
    ///
    /// This method only replaces uses in instructions, not in PHI operands.
    /// For internal operations that need to also replace PHI operands, use
    /// `replace_uses_including_phis`.
    pub fn replace_uses(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> ReplaceResult {
        let carried = self.substituted_facts(old_var, new_var);
        self.machine.substituted(old_var, new_var);
        self.blocks
            .iter_mut()
            .map(|block| block.replace_uses(old_var, new_var, &carried))
            .fold(ReplaceResult::default(), |acc, r| ReplaceResult {
                replaced: acc.replaced.saturating_add(r.replaced),
                skipped: acc.skipped.saturating_add(r.skipped),
            })
    }

    /// Replaces all uses of `old_var` with `new_var`, including in PHI operands.
    ///
    /// Unlike [`replace_uses`](Self::replace_uses), this method also replaces uses
    /// in PHI node operands across all blocks. This is necessary for internal SSA
    /// operations that eliminate PHI nodes and need to forward their values through
    /// other PHIs.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose uses are rewritten, in instructions and
    ///   in PHI operands alike.
    /// * `new_var` - The variable substituted in for each use.
    ///
    /// # Returns
    ///
    /// A [`ReplaceResult`] summing, across all blocks, the uses replaced and
    /// the uses skipped by the self-referential guard.
    ///
    /// # Phi groups
    ///
    /// Misused, it makes a phi read a value of another rename group, which a
    /// renaming construction does not keep: it fills each phi operand from the
    /// phi's own group.
    ///
    /// # When to Use
    ///
    /// Only use this method for:
    /// - **Trivial PHI elimination**: When removing a PHI like `v10 = phi(v5, v5)`,
    ///   we need to replace uses of `v10` with `v5` everywhere, including in other
    ///   PHI operands.
    /// - **Copy propagation within PHIs**: When a copy's destination is a PHI result
    ///   and we're eliminating that PHI.
    pub fn replace_uses_including_phis(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> ReplaceResult {
        let carried = self.substituted_facts(old_var, new_var);
        let result = self
            .blocks
            .iter_mut()
            .map(|block| block.replace_uses_including_phis(old_var, new_var, &carried))
            .fold(ReplaceResult::default(), |acc, r| ReplaceResult {
                replaced: acc.replaced.saturating_add(r.replaced),
                skipped: acc.skipped.saturating_add(r.skipped),
            });
        self.inherit_rename_group(
            old_var,
            new_var,
            result.replaced > 0 || result.skipped == 0,
            Substitution::TrivialPhi,
        );
        result
    }

    /// Applies a set of substitutions at once, in instructions and phi
    /// operands alike, resolving chains first.
    ///
    /// The batch form of [`Self::replace_uses_including_phis`], with its
    /// self-referential guard and its rename-group bookkeeping per pair, in one
    /// pass over the function rather than one per pair. The pairs are resolved
    /// to a fixed point first ([`resolve_chains`]): applied one at a time,
    /// `(q -> p)` after `(p -> r)` would write `p` back into use position after
    /// its own uses were rewritten.
    ///
    /// # Arguments
    ///
    /// * `pairs` - `(old, new)` substitutions, possibly chained.
    ///
    /// # Returns
    ///
    /// A [`ReplaceResult`] summing the uses replaced and the uses the
    /// self-referential guard skipped.
    pub fn replace_uses_resolved(&mut self, pairs: &[(SsaVarId, SsaVarId)]) -> ReplaceResult {
        let resolved = resolve_chains(pairs);
        if resolved.is_empty() {
            return ReplaceResult::default();
        }
        // What a reader of each replaced value takes in where it then computes
        // in that value's place ([`Self::substituted_facts`]).
        let carried: BTreeMap<SsaVarId, Vec<T::InstructionFacts>> = resolved
            .iter()
            .map(|(&old, &new)| (old, self.substituted_facts(old, new)))
            .collect();
        let mut replaced_of: BTreeMap<SsaVarId, usize> = BTreeMap::new();
        let mut refused: BTreeSet<SsaVarId> = BTreeSet::new();
        let mut total = ReplaceResult::default();
        for block in self.blocks.iter_mut() {
            for instruction in block.instructions_mut() {
                let op = instruction.op_mut();
                // The guard of `replace_uses`: a use is left alone where the
                // instruction defines its replacement, which would read its own
                // result.
                let defs: Vec<SsaVarId> = op.defs().collect();
                let mut replaced_here: Vec<SsaVarId> = Vec::new();
                op.replace_uses_with(|var| {
                    let new = *resolved.get(&var)?;
                    if defs.contains(&new) {
                        total.skipped = total.skipped.saturating_add(1);
                        refused.insert(var);
                        return None;
                    }
                    let count = replaced_of.entry(var).or_default();
                    *count = count.saturating_add(1);
                    replaced_here.push(var);
                    Some(new)
                });
                replaced_here.sort_unstable();
                replaced_here.dedup();
                for old in replaced_here {
                    for facts in carried.get(&old).into_iter().flatten() {
                        instruction.facts_mut().absorb_computation(facts);
                    }
                }
            }
            for phi in block.phi_nodes_mut() {
                for operand in phi.operands_mut() {
                    if let Some(&new) = resolved.get(&operand.value()) {
                        let count = replaced_of.entry(operand.value()).or_default();
                        *count = count.saturating_add(1);
                        // The value changes; the edge it arrived on does not.
                        operand.set_value(new);
                    }
                }
            }
        }
        for (&old, &new) in &resolved {
            let count = replaced_of.get(&old).copied().unwrap_or(0);
            total.replaced = total.replaced.saturating_add(count);
            self.inherit_rename_group(
                old,
                new,
                count > 0 || !refused.contains(&old),
                Substitution::TrivialPhi,
            );
        }
        total
    }

    /// Replaces all uses of `old_var` with `new_var` within a specific block.
    ///
    /// This is a targeted version of `replace_uses` that only affects instructions
    /// within the specified block (not PHI operands).
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block whose instructions are rewritten.
    /// * `old_var` - The variable whose uses are rewritten.
    /// * `new_var` - The variable substituted in for each use.
    ///
    /// # Returns
    ///
    /// A [`ReplaceResult`] for that block, or the default (zero) result if the
    /// index names no block.
    pub fn replace_uses_in_block(
        &mut self,
        block_idx: usize,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> ReplaceResult {
        let carried = self.substituted_facts(old_var, new_var);
        self.block_mut(block_idx)
            .map_or(ReplaceResult::default(), |block| {
                block.replace_uses(old_var, new_var, &carried)
            })
    }

    /// Prunes phi operands from non-existent or unreachable predecessors.
    ///
    /// After block removal or CFG changes, phi nodes may reference predecessors
    /// that no longer exist or are unreachable. This method removes those stale
    /// operands, ensuring phi nodes reference exactly the block's real
    /// predecessors.
    ///
    /// Pruning is deliberately **structural only**: an operand is dropped solely
    /// because its predecessor edge is gone, never because its *value* looks
    /// undefined. A phi's operand list is a per-incoming-edge mapping, so
    /// dropping the operand for a live edge breaks the phi/CFG invariant that
    /// [`MissingPhiOperand`](crate::analysis::VerifierError::MissingPhiOperand)
    /// guards - and, worse, can leave a
    /// two-operand phi with a single operand, which trivial-phi simplification
    /// then collapses into a plain copy of the surviving value. That silently
    /// discards one of the merged values. An operand naming a variable with no
    /// reachable definition is an upstream defect to fix at its source; keeping
    /// it is strictly safer than deleting a live edge to hide it.
    ///
    /// # Arguments
    ///
    /// * `reachable` - The blocks to consider; predecessors are recomputed from
    ///   only these, and an operand naming a block outside the set is stale.
    ///
    /// # Returns
    ///
    /// The number of operands pruned.
    pub fn prune_phi_operands(&mut self, reachable: &BitSet) -> usize {
        // Actual predecessors, from the exception-aware CFG: a handler's
        // predecessors are the blocks a throw in which dispatches to it, and
        // its operands for throws name exactly those. Reading predecessors from
        // terminators alone found none for a handler, so every operand for a
        // throw was pruned whenever the phi had one other operand to keep - a
        // `finally` entered by `leave` then saw only the value the `leave`
        // brought, whatever the throw left.
        let block_count = self.blocks.len();
        let cfg = SsaCfg::from_ssa(self);
        let mut actual_predecessors: BTreeMap<usize, BitSet> = BTreeMap::new();
        for block_idx in reachable.iter() {
            let mut preds = BitSet::new(block_count);
            for &pred in cfg.block_predecessors(block_idx) {
                if reachable.contains_checked(pred) {
                    preds.insert(pred);
                }
            }
            actual_predecessors.insert(block_idx, preds);
        }
        drop(cfg);

        let mut pruned: usize = 0;

        for block_idx in reachable.iter() {
            if let Some(block) = self.block_mut(block_idx) {
                let preds = actual_predecessors.get(&block_idx);

                for phi in block.phi_nodes_mut() {
                    let operands = phi.operands_mut();
                    let original_len = operands.len();

                    if original_len == 0 {
                        continue;
                    }

                    // Predicate for operands worth keeping; evaluated without
                    // materializing a per-phi `Vec<bool>`.
                    let keeps = |op: &PhiOperand| -> bool {
                        let pred = op.predecessor();
                        pred < block_count && preds.is_some_and(|p| p.contains(pred))
                    };

                    // Never leave a PHI completely empty.
                    let keep_count = operands.iter().filter(|op| keeps(op)).count();
                    if keep_count == 0 {
                        continue;
                    }

                    operands.retain(keeps);

                    pruned = pruned.saturating_add(original_len.saturating_sub(operands.len()));
                }
            }
        }

        pruned
    }

    /// Recomputes all use information from scratch.
    ///
    /// This should be called after SSA transformations that may have invalidated
    /// the use tracking.
    pub fn recompute_uses(&mut self) {
        let variables = &mut self.variables;

        // Step 1: Clear all existing uses
        for var in variables.iter_mut() {
            var.clear_uses();
        }

        // Step 2: Scan instructions to record uses
        for (block_idx, block) in self.blocks.iter().enumerate() {
            // Record uses from instructions
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                instr.op().for_each_use(|use_var| {
                    let var = use_var.index();
                    if let Some(slot) = variables.get_mut(var) {
                        let use_site = UseSite::instruction(block_idx, instr_idx);
                        slot.add_use(use_site);
                    }
                });
            }

            // Record uses from phi nodes
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                for operand in phi.operands() {
                    let var = operand.value().index();
                    if let Some(slot) = variables.get_mut(var) {
                        let use_site = UseSite::phi_operand(block_idx, phi_idx);
                        slot.add_use(use_site);
                    }
                }
            }
        }
    }

    /// Replaces the operation of an instruction at a specific location.
    ///
    /// Uses `set_op_preserving_type`, so the destination's recorded type
    /// survives the swap even when the new operation would otherwise imply a
    /// different one; a pass substituting one computation for another relies on
    /// that to keep the variable's type stable.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block containing the instruction.
    /// * `instr_idx` - Index of the instruction within that block.
    /// * `new_op` - The operation to install in its place.
    ///
    /// The instruction computes in place of every operand it no longer reads:
    /// those operands' definitions' facts are folded into its own
    /// ([`InstructionFacts::absorb_computation`]), so a fold to a constant keeps where the
    /// operands came from.
    ///
    /// # Returns
    ///
    /// `true` if the location existed and the operation was replaced.
    pub fn replace_instruction_op(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        new_op: SsaOp<T>,
    ) -> bool {
        let Some(old) = self
            .blocks
            .get(block_idx)
            .and_then(|block| block.instructions().get(instr_idx))
        else {
            return false;
        };
        let absorbed = self.facts_dropped((block_idx, instr_idx), old.op(), &new_op);
        let retired = old.op().clone();
        let kept: Vec<SsaVarId> = new_op.defs().collect();
        self.retire_definitions(&retired, &kept);
        let Some(instr) = self
            .blocks
            .get_mut(block_idx)
            .and_then(|block| block.instructions_mut().get_mut(instr_idx))
        else {
            return false;
        };
        instr.set_op_preserving_type(new_op, Rewrite::Host);
        for facts in &absorbed {
            instr.facts_mut().absorb_computation(facts);
        }
        true
    }

    /// Notes that `op` stops defining what it defined, but for `kept`. The
    /// machine states keep nothing alive, so a value one names that
    /// normalization stops defining would leave its register unknown. Where
    /// `op` computed the value as a view of one as wide (a copy, a
    /// reinterpretation, a pointer and integer of one width), the states read
    /// that value from then on; where it is an integer conversion that widens
    /// one, they read that value widened as the conversion did
    /// ([`Widening`]), which a crossing applies before it reads the register.
    /// A truncation, which would claim bits the register never held, and
    /// every other operation leave the register unknown.
    fn retire_definitions(&mut self, op: &SsaOp<T>, kept: &[SsaVarId]) {
        if self.machine.is_empty() {
            return;
        }
        let source = match op {
            SsaOp::IntConv { operand, .. }
            | SsaOp::IntToPtr { operand, .. }
            | SsaOp::PtrToInt { operand, .. }
            | SsaOp::Bitcast { operand, .. }
            | SsaOp::VectorReinterpret { value: operand, .. }
            | SsaOp::Copy { src: operand, .. } => *operand,
            _ => return,
        };
        let Some(dest) = op.dest().filter(|dest| !kept.contains(dest)) else {
            return;
        };
        // A view as wide as its operand is the operand; an integer
        // conversion that widens it is the operand extended. A narrowing, or
        // a widening no extension states, is no view a register's value can
        // be read back through.
        match (self.bit_width(source), self.bit_width(dest), op) {
            (Some(from), Some(to), _) if from == to => self.machine.substituted(dest, source),
            (Some(from), Some(to), SsaOp::IntConv { unsigned, .. }) if from < to => {
                self.machine.widened(
                    dest,
                    source,
                    Widening {
                        bits: to,
                        signed: !*unsigned,
                    },
                );
            }
            _ => {}
        }
    }

    /// The facts of the computations the instruction at `site`, rewritten
    /// from `old` to `new`, now holds: those of every operand `old` reads,
    /// `new` does not, and nothing but `site` reads, which go with the rewrite
    /// (`computation_facts`). An operand something else still reads
    /// keeps its computation where it is.
    #[must_use]
    fn facts_dropped(
        &self,
        site: (usize, usize),
        old: &SsaOp<T>,
        new: &SsaOp<T>,
    ) -> Vec<T::InstructionFacts> {
        let kept: BTreeSet<SsaVarId> = new.uses().into_iter().collect();
        let dropped: BTreeSet<SsaVarId> = old
            .uses()
            .into_iter()
            .filter(|var| !kept.contains(var))
            .collect();
        dropped
            .into_iter()
            .filter(|&var| self.read_only_at(var, site))
            .flat_map(|var| self.computation_facts(var, &kept))
            .collect()
    }

    /// The facts a reader of `old` takes in when its uses become `new`'s: the
    /// computation that goes with `old` once nothing reads it
    /// (`computation_facts`), where the reader then computes in its
    /// place, which it does where `new` is a constant or is computed apart
    /// from `old`. Where `new`'s definition reads `old`, `old`'s computation
    /// stays where it was, and nothing is carried.
    #[must_use]
    pub fn substituted_facts(&self, old: SsaVarId, new: SsaVarId) -> Vec<T::InstructionFacts> {
        let new_site = self.variable(new).map(SsaVariable::def_site);
        let reads_old = new_site
            .and_then(|site| {
                self.blocks
                    .get(site.block())?
                    .instructions()
                    .get(site.index()?)
            })
            .is_some_and(|definition| {
                !matches!(definition.op(), SsaOp::Const { .. }) && definition.op().uses_var(old)
            });
        if reads_old {
            return Vec::new();
        }
        self.computation_facts(old, &BTreeSet::from([new]))
    }

    /// The facts of the computation of `var` that goes with it once nothing
    /// reads it: its definition's, and, through every operand that definition
    /// alone reads, those operands' computations in turn. A value in `kept`
    /// is still read where the computation's result is, and keeps its own; a
    /// phi, or a value nothing defines, ends a computation; so does the bound
    /// on how many definitions one computation spans.
    #[must_use]
    fn computation_facts(
        &self,
        var: SsaVarId,
        kept: &BTreeSet<SsaVarId>,
    ) -> Vec<T::InstructionFacts> {
        // As many definitions as a folded address chain or a forwarded load's
        // address computation spans, with room to spare.
        const MAX_DEFINITIONS: usize = 64;
        let mut facts = Vec::new();
        let mut seen: BTreeSet<SsaVarId> = BTreeSet::new();
        let mut pending = vec![var];
        while let Some(var) = pending.pop() {
            if facts.len() >= MAX_DEFINITIONS || kept.contains(&var) || !seen.insert(var) {
                continue;
            }
            let Some(site) = self.variable(var).map(SsaVariable::def_site) else {
                continue;
            };
            let Some(at) = site.index() else {
                continue;
            };
            let Some(definition) = self
                .blocks
                .get(site.block())
                .and_then(|block| block.instructions().get(at))
            else {
                continue;
            };
            facts.push(definition.facts().clone());
            pending.extend(
                definition
                    .op()
                    .uses()
                    .into_iter()
                    .filter(|&operand| self.read_only_at(operand, (site.block(), at))),
            );
        }
        facts
    }

    /// Whether every read of `var` is by the instruction at `site`.
    ///
    /// Within an edit session a use list can still name an instruction an
    /// earlier rewrite stopped reading `var` at
    /// ([`SsaEditor`](crate::ir::function::SsaEditor)); such an
    /// entry reads nothing, and is passed over rather than taken for a
    /// second reader.
    fn read_only_at(&self, var: SsaVarId, site: (usize, usize)) -> bool {
        let reads = |used: &&UseSite| {
            used.is_phi_operand()
                || self
                    .blocks
                    .get(used.block())
                    .and_then(|block| block.instructions().get(used.index()))
                    .is_some_and(|instruction| instruction.op().uses_var(var))
        };
        self.variable(var).is_some_and(|variable| {
            let mut live = variable.uses().iter().filter(reads).peekable();
            live.peek().is_some()
                && live.all(|used| !used.is_phi_operand() && (used.block(), used.index()) == site)
        })
    }

    /// Simplifies a phi node by converting it to a copy operation.
    ///
    /// When a phi node has all identical operands (excluding self-references),
    /// instruction uses of the phi result can be replaced with `source`. The
    /// phi is removed only when no remaining use of its result exists outside
    /// the phi being removed.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block holding the phi.
    /// * `phi_idx` - Index of the phi within that block.
    /// * `source` - The single value the phi collapses to; uses of the phi
    ///   result are rewritten to it.
    ///
    /// # Returns
    ///
    /// `true` if the phi was removed; `false` if the location was invalid or a
    /// use of the result survived the self-referential guard.
    pub fn simplify_phi_to_copy(
        &mut self,
        block_idx: usize,
        phi_idx: usize,
        source: SsaVarId,
    ) -> bool {
        let Some(block) = self.blocks.get(block_idx) else {
            return false;
        };

        let Some(phi) = block.phi_nodes().get(phi_idx) else {
            return false;
        };

        let dest = phi.result();

        if dest != source {
            let _ = self.replace_uses_checked(dest, source);
        }

        if has_remaining_uses_including_phis(self, dest, Some((block_idx, phi_idx))) {
            return false;
        }

        let Some(block) = self.blocks.get_mut(block_idx) else {
            return false;
        };
        if phi_idx >= block.phi_nodes().len() {
            return false;
        }
        block.phi_nodes_mut().remove(phi_idx);

        true
    }

    /// Removes a phi node by index without any validation.
    ///
    /// Unlike [`simplify_phi_to_copy`](Self::simplify_phi_to_copy), no uses of
    /// the phi result are rewritten first, so the caller is responsible for
    /// ensuring nothing still reads it.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block holding the phi.
    /// * `phi_idx` - Index of the phi to remove within that block.
    ///
    /// # Returns
    ///
    /// `true` if the location was valid and the phi was removed.
    pub fn remove_phi_unchecked(&mut self, block_idx: usize, phi_idx: usize) -> bool {
        if let Some(block) = self.blocks.get_mut(block_idx)
            && phi_idx < block.phi_nodes().len()
        {
            block.phi_nodes_mut().remove(phi_idx);
            return true;
        }
        false
    }

    /// Returns each phi of a set of phis that, reading one another, reads one
    /// value from outside the set, with that value (Braun et al., "Simple and
    /// Efficient Construction of Static Single Assignment Form", 2013, §3.2).
    ///
    /// The phis that read other phis form a graph; each strongly connected
    /// set of two or more whose operands from outside it are one value `v` is
    /// `v` throughout: every value flowing around the set came from `v`.
    /// Where a set reads several values from outside, the phis of it that
    /// read only the set may still be one value, so they are examined alone.
    /// Sets are taken operands first, so a set found to be `v` reads as `v`
    /// in the sets that read it. Only operands `keep` admits count, given the
    /// block the phi is in: the operands of reachable predecessors where a
    /// rebuild knows which those are. A set of one phi is not returned: the
    /// single-phi rule handles it.
    fn redundant_phi_cycles(
        &self,
        keep: impl Fn(usize, &PhiOperand) -> bool,
    ) -> Vec<(SsaVarId, SsaVarId)> {
        let mut operands: BTreeMap<SsaVarId, Vec<SsaVarId>> = BTreeMap::new();
        for block in self.blocks.iter() {
            let block_idx = block.id();
            for phi in block.phi_nodes() {
                operands.insert(
                    phi.result(),
                    phi.operands()
                        .iter()
                        .filter(|operand| keep(block_idx, operand))
                        .map(PhiOperand::value)
                        .collect(),
                );
            }
        }
        let mut found: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
        let mut pending: Vec<BTreeSet<SsaVarId>> = vec![operands.keys().copied().collect()];
        while let Some(phis) = pending.pop() {
            let mut graph: DirectedGraph<'_, SsaVarId, ()> = DirectedGraph::new();
            let nodes: BTreeMap<SsaVarId, NodeId> =
                phis.iter().map(|&phi| (phi, graph.add_node(phi))).collect();
            for (&phi, &node) in &nodes {
                for operand in operands.get(&phi).into_iter().flatten() {
                    if let Some(&target) = nodes.get(operand)
                        && target != node
                    {
                        let _ = graph.add_edge(node, target, ());
                    }
                }
            }
            for component in strongly_connected_components(&graph) {
                let members: BTreeSet<SsaVarId> = component
                    .iter()
                    .filter_map(|node| graph.node(*node).copied())
                    .collect();
                if members.len() < 2 {
                    continue;
                }
                // What each member reads from outside the set, through the
                // sets already found to be one value.
                let mut outside: BTreeSet<SsaVarId> = BTreeSet::new();
                let mut inner: BTreeSet<SsaVarId> = BTreeSet::new();
                for &phi in &members {
                    let mut reads_outside = false;
                    for &operand in operands.get(&phi).into_iter().flatten() {
                        let value = found.get(&operand).copied().unwrap_or(operand);
                        if !members.contains(&value) {
                            outside.insert(value);
                            reads_outside = true;
                        }
                    }
                    if !reads_outside {
                        inner.insert(phi);
                    }
                }
                match outside.len() {
                    1 => {
                        if let Some(&value) = outside.first() {
                            for &phi in &members {
                                found.insert(phi, value);
                            }
                        }
                    }
                    0 => {}
                    _ if !inner.is_empty() && inner.len() < members.len() => pending.push(inner),
                    _ => {}
                }
            }
        }
        found.into_iter().collect()
    }

    /// Gives the trivial phi `phi` the type of its source `source` where the
    /// two are integers of one width that differ only in sign.
    ///
    /// A trivial phi is its source. A sign the source does not have is an
    /// annotation the value does not have either: a phi a pruned edge left
    /// one operand, typed apart from it by what it joined before. Left as it
    /// was, its uses refuse the source as a value of another type, and the
    /// phi stays a copy of what it is. Integers of two widths are a join that
    /// extends its operand, and keep their types.
    fn take_sign_of(&mut self, phi: SsaVarId, source: SsaVarId) {
        let (Some(phi_type), Some(source_type)) = (
            self.variable(phi)
                .map(|variable| variable.var_type().clone()),
            self.variable(source)
                .map(|variable| variable.var_type().clone()),
        ) else {
            return;
        };
        if phi_type != source_type
            && T::is_integer(&phi_type)
            && T::is_integer(&source_type)
            && T::operand_width(&phi_type).is_some()
            && T::operand_width(&phi_type) == T::operand_width(&source_type)
            && let Some(variable) = self.variable_mut(phi)
        {
            variable.set_type(source_type);
        }
    }

    /// Eliminates trivial phi nodes where all non-self operands resolve to a
    /// single value. Iterates to fixpoint (cascading simplification).
    ///
    /// A phi is trivial when, excluding self-references, all operands provide
    /// the same value. The phi result is replaced by that value everywhere
    /// (including other phi operands). So is each phi of a set that reads
    /// only its own members and one value from outside them, however the set
    /// is joined (Braun et al.'s redundant phi cycles): a loop's stack pointer
    /// carried around two nested back edges is the value it entered with.
    ///
    /// # Arguments
    ///
    /// * `options` - Selects the elimination mode; see the modes below.
    ///
    /// # Modes
    ///
    /// When `options.reachable` is `Some`:
    /// - Uses reachability-aware self-referential checks (definitions in unreachable
    ///   blocks don't count as creating cycles).
    /// - Performs a second-pass check filtering operands from unreachable predecessors.
    /// - All trivial phis are removed unconditionally, as SSA reconstruction
    ///   needs.
    ///
    /// When `options.reachable` is `None`:
    /// - Uses basic self-referential checks.
    /// - Resolves chains among trivial phis to avoid stale references.
    /// - Only fully propagated phis (no skipped uses) are removed (suitable for repair_ssa).
    ///
    /// # Returns
    ///
    /// The number of phis eliminated, and whether the function changed: an
    /// operand rewritten or a phi removed. Repair keeps what it learned about
    /// the function's variables for as long as nothing is rewritten, so it
    /// needs to know the difference between a pass that found no trivial phi
    /// and one that found some and could not remove them.
    pub fn eliminate_trivial_phis(&mut self, options: &TrivialPhiOptions) -> (usize, bool) {
        let mut total_eliminated: usize = 0;
        let mut changed = false;
        let block_count = self.blocks.len();

        // Precompute reachability data if in reachable mode. Build the full
        // predecessor relation in one O(V+E) pass instead of calling
        // `block_predecessors` per block (which is O(V) each → O(V²)).
        let reachable_preds: Option<BTreeMap<usize, BitSet>> = options.reachable.map(|reachable| {
            let all_preds = SsaCfg::from_ssa(self).to_predecessor_sets();
            let mut map = BTreeMap::new();
            for block in self.blocks.iter() {
                let block_idx = block.id();
                if !reachable.contains(block_idx) {
                    continue;
                }
                let mut preds = BitSet::new(block_count);
                if let Some(plist) = all_preds.get(block_idx) {
                    for &p in plist {
                        if reachable.contains(p) {
                            preds.insert(p);
                        }
                    }
                }
                map.insert(block_idx, preds);
            }
            map
        });

        let var_def_block: Option<BTreeMap<SsaVarId, usize>> = options.reachable.map(|_| {
            let mut map = BTreeMap::new();
            for block in self.blocks.iter() {
                let block_idx = block.id();
                for instr in block.instructions() {
                    for dest in instr.op().defs() {
                        map.insert(dest, block_idx);
                    }
                }
            }
            map
        });

        loop {
            let mut trivial_phis: Vec<(SsaVarId, SsaVarId)> = Vec::new();

            for block in self.blocks.iter() {
                let block_idx = block.id();
                let block_reachable_preds =
                    reachable_preds.as_ref().and_then(|rp| rp.get(&block_idx));

                for phi in block.phi_nodes() {
                    let result = phi.result();

                    // The sole non-self operand, if there is exactly one
                    let sources =
                        SoleSource::of(phi.operands().iter().map(PhiOperand::value), result);

                    if let SoleSource::One(source) = sources {
                        let is_self_ref = match (&var_def_block, options.reachable) {
                            (Some(vdb), Some(reachable)) => self
                                .would_create_self_reference_reachable(
                                    source, result, vdb, reachable,
                                ),
                            _ => self.would_create_self_reference(source, result),
                        };

                        if !is_self_ref {
                            trivial_phis.push((result, source));
                            continue;
                        }
                    } else if matches!(sources, SoleSource::None) && !phi.operands().is_empty() {
                        // Fully self-referential phi
                        trivial_phis.push((result, result));
                        continue;
                    }

                    // Reachable-only second pass: filter out operands from
                    // unreachable predecessors and check triviality again
                    if matches!(sources, SoleSource::Many)
                        && let Some(rpreds) = block_reachable_preds
                    {
                        let reachable_sources = SoleSource::of(
                            phi.operands()
                                .iter()
                                .filter(|op| {
                                    let pred = op.predecessor();
                                    pred < block_count && rpreds.contains(pred)
                                })
                                .map(PhiOperand::value),
                            result,
                        );

                        if let SoleSource::One(source) = reachable_sources {
                            let is_self_ref = match (&var_def_block, options.reachable) {
                                (Some(vdb), Some(reachable)) => self
                                    .would_create_self_reference_reachable(
                                        source, result, vdb, reachable,
                                    ),
                                _ => self.would_create_self_reference(source, result),
                            };
                            if !is_self_ref {
                                trivial_phis.push((result, source));
                            }
                        } else if matches!(reachable_sources, SoleSource::None)
                            && phi.operands().iter().any(|op| {
                                let pred = op.predecessor();
                                pred < block_count && rpreds.contains(pred)
                            })
                        {
                            trivial_phis.push((result, result));
                        }
                    }
                }
            }

            // A set of phis that read only each other and one value from
            // outside them is that value. Asked every round, not only when no
            // phi is trivial alone: a trivial phi whose uses cannot take its
            // source keeps a round from retiring anything, and the cycles
            // would then never be asked. A phi both rules name is the value
            // its cycle carries, which is no other phi.
            let cycles: BTreeMap<SsaVarId, SsaVarId> = self
                .redundant_phi_cycles(|block_idx, operand| {
                    reachable_preds.as_ref().is_none_or(|preds| {
                        preds.get(&block_idx).is_some_and(|preds| {
                            let pred = operand.predecessor();
                            pred < block_count && preds.contains(pred)
                        })
                    })
                })
                .into_iter()
                .collect();
            for (result, source) in &mut trivial_phis {
                if let Some(&value) = cycles.get(result) {
                    *source = value;
                }
            }
            let named: BTreeSet<SsaVarId> =
                trivial_phis.iter().map(|(result, _)| *result).collect();
            trivial_phis.extend(
                cycles
                    .into_iter()
                    .filter(|(result, _)| !named.contains(result)),
            );
            if trivial_phis.is_empty() {
                break;
            }
            changed = true;

            let bound = self.var_id_bound();

            if options.reachable.is_none() {
                // Repair mode: resolve chains among trivial phis.
                let trivial_map: BTreeMap<SsaVarId, SsaVarId> =
                    trivial_phis.iter().copied().collect();
                for entry in &mut trivial_phis {
                    if entry.0 == entry.1 {
                        continue;
                    }
                    let mut current = entry.1;
                    let mut visited = BTreeSet::new();
                    while let Some(&next) = trivial_map.get(&current) {
                        if next == current || !visited.insert(current) {
                            break;
                        }
                        current = next;
                    }
                    entry.1 = current;
                }

                // Replace instruction uses through the checked path and only
                // remove phis whose result is completely unused afterward. The
                // dominator tree is built once for the whole batch - use
                // replacement leaves the CFG (and therefore dominance) unchanged.
                let dominators = if self.block_count() > 0 {
                    let eh = SsaCfg::from_ssa(self);
                    Some(compute_dominators(&eh, NodeId::new(0)))
                } else {
                    None
                };
                let mut trivial_set = VarSet::new(bound);
                // All replacements first, then *one* scan for what is still
                // read. Interleaving them meant a whole-function scan per
                // trivial phi - O(trivial_phis × function) per fixpoint round,
                // and a rebuild produces trivial phis in proportion to the
                // function, so the round was quadratic.
                //
                // Batching is equivalent: `replace_uses_checked_indexed(p, s)`
                // only removes uses of `p` and adds uses of `s`, and `s` is
                // never another phi result being tested in this round (chains
                // were already resolved above), so no replacement can resurrect
                // a use of a phi an earlier one retired.
                //
                // Each replacement costs the uses of its own phi: the index is
                // built once for the round and every replacement keeps it
                // exact.
                self.recompute_uses();
                for (phi_result, source) in &trivial_phis {
                    if *phi_result != *source {
                        self.take_sign_of(*phi_result, *source);
                        let checked = self.replace_uses_checked_indexed(
                            *phi_result,
                            *source,
                            dominators.as_ref(),
                            None,
                        );
                        self.inherit_rename_group(
                            *phi_result,
                            *source,
                            checked.took(),
                            Substitution::TrivialPhi,
                        );
                    }
                }
                // Collapse the phi-operand references the checked replacement
                // declined too.
                //
                // Otherwise a trivial phi whose result is another phi's operand
                // stays "read" and survives the round. In a chain
                // `p1 = phi(x); p2 = phi(p1); ...` that retires exactly one phi
                // per round - the fixpoint runs as many rounds as there are
                // phis, and since each round is already O(phis x function),
                // the whole thing would be cubic - seconds on a chain of a few
                // hundred phis.
                //
                // Rewriting the operand is the sanctioned case for touching phi
                // operands directly (see the module header on
                // `replace_uses_including_phis`): a trivial phi's value *is* its
                // source, so an operand naming it and an operand naming the
                // source denote the same value on the same edge. Chains were
                // resolved into `trivial_phis` above, so `source` is already the
                // end of the chain.
                let resolved: BTreeMap<SsaVarId, SsaVarId> = trivial_phis
                    .iter()
                    .filter(|(result, source)| result != source)
                    .copied()
                    .collect();
                if !resolved.is_empty() {
                    for block in self.blocks.iter_mut() {
                        for phi in block.phi_nodes_mut() {
                            let result = phi.result();
                            for operand in phi.operands_mut() {
                                if let Some(&target) = resolved.get(&operand.value())
                                    // Never rewrite an operand into a reference
                                    // to its own phi - that would manufacture the
                                    // self-referential shape this pass removes.
                                    && target != result
                                {
                                    operand.set_value(target);
                                }
                            }
                        }
                    }
                }

                let still_read = self.collect_read_variables();
                for (phi_result, _) in &trivial_phis {
                    if !still_read.contains(*phi_result) {
                        trivial_set.insert(*phi_result);
                    }
                }
                if trivial_set.is_empty() {
                    break;
                }

                total_eliminated = total_eliminated.saturating_add(trivial_set.len());
                for block in self.blocks.iter_mut() {
                    block
                        .phi_nodes_mut()
                        .retain(|phi| !trivial_set.contains(phi.result()));
                }
                // The variable rows are deliberately left in place. `variables`
                // is indexed by id (`variables[i].id().index() == i`), and
                // dropping a row without renumbering makes every higher id
                // resolve to the wrong variable. Renumbering here is not an
                // option either: `var_def_block` above is built once and keyed
                // by id, and the next fixpoint iteration reads it. Removing a
                // phi removes a *definition*; deleting the now-orphaned row is
                // `compact_variables`' job, and it already recognises variables
                // with no remaining definition.
            } else {
                // Rebuild mode: replace uses and remove unconditionally.
                //
                // The substitutions are composed into one map and applied in a
                // single pass. Doing them one at a time meant a whole-function
                // scan per trivial phi, and a rebuild produces trivial phis in
                // proportion to the function, so every fixpoint round was
                // quadratic - the same defect the repair branch above was
                // already corrected for.
                //
                // Sequential application is order-sensitive: each substitution
                // only sees the values left by the ones before it, so `p2 -> p1`
                // applied after `p1 -> x` leaves `p1`, not `x`. Composing the
                // map back-to-front reproduces that exactly - when an entry is
                // resolved every later entry is already final, and earlier ones
                // must not be followed.
                let resolved = {
                    // Every substitution target must be a variable that survives
                    // this round. Composing back-to-front only resolves a source
                    // through entries already inserted, so a chain
                    // `p1 -> p2 -> x` can leave `p1 -> p2` while `p2` is retired
                    // in the same round - the rewrite then points uses of `p1` at
                    // `p2`, which is deleted moments later, and those uses are
                    // stranded on a variable nothing defines. Walk each chain to
                    // its end instead, exactly as the repair branch above does.
                    let direct: BTreeMap<SsaVarId, SsaVarId> = trivial_phis
                        .iter()
                        .filter(|(result, source)| result != source)
                        .copied()
                        .collect();
                    let mut resolved: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
                    for (result, source) in &direct {
                        let mut current = *source;
                        let mut visited: BTreeSet<SsaVarId> = BTreeSet::new();
                        visited.insert(*result);
                        while let Some(&next) = direct.get(&current) {
                            if !visited.insert(current) {
                                break;
                            }
                            current = next;
                        }
                        if current != *result {
                            resolved.insert(*result, current);
                        }
                    }
                    resolved
                };
                if !resolved.is_empty() {
                    for block in self.blocks.iter_mut() {
                        for instr in block.instructions_mut() {
                            instr
                                .op_mut()
                                .replace_uses_with(|var| resolved.get(&var).copied());
                        }
                        for phi in block.phi_nodes_mut() {
                            for operand in phi.operands_mut() {
                                if let Some(&target) = resolved.get(&operand.value()) {
                                    operand.set_value(target);
                                }
                            }
                        }
                    }
                }

                // A self-referential phi is recorded as `(result, result)` and is
                // deliberately absent from `resolved` - there is no other value
                // to rewrite its uses to. Removing it regardless strands every
                // one of those uses on a variable nothing defines. Retire it
                // only once nothing reads it; a later fixpoint round collects it
                // after its readers go away. The repair branch already applies
                // exactly this condition.
                let still_read = self.collect_read_variables();
                let mut trivial_set = VarSet::new(bound);
                for (result, source) in &trivial_phis {
                    if result == source && still_read.contains(*result) {
                        continue;
                    }
                    trivial_set.insert(*result);
                }
                if trivial_set.is_empty() {
                    break;
                }
                total_eliminated = total_eliminated.saturating_add(trivial_set.len());
                for block in self.blocks.iter_mut() {
                    block
                        .phi_nodes_mut()
                        .retain(|phi| !trivial_set.contains(phi.result()));
                }
                // Left in place for the same reason as the repair-mode branch
                // above: `compact_variables` owns row removal, because it is the
                // only path that renumbers afterwards.
            }
        }

        (total_eliminated, changed)
    }

    /// Rewrites the instruction at a location into a `Const` defining the same
    /// destination.
    ///
    /// Constant folding does not touch use sites: the destination variable keeps
    /// its id, so every existing read resolves to the folded value once the
    /// producing instruction becomes a `Const`. The rewrite goes through
    /// `set_op_preserving_type`, so the destination's recorded type is retained.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block containing the instruction.
    /// * `instr_idx` - Index of the instruction within that block.
    /// * `value` - The constant value the instruction is folded to.
    ///
    /// # Returns
    ///
    /// `true` if the location existed and the instruction defined a destination
    /// that was folded.
    pub fn fold_constant(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        value: ConstValue<T>,
    ) -> bool {
        if let Some(block) = self.blocks.get_mut(block_idx)
            && let Some(instr) = block.instructions_mut().get_mut(instr_idx)
            && let Some(dest) = instr.op().dest()
        {
            instr.set_op_preserving_type(SsaOp::Const { dest, value }, Rewrite::Host);
            return true;
        }
        false
    }

    /// Recomputes every variable's definition site from its current position.
    ///
    /// A definition site names the block and the index of the phi or
    /// instruction that produces the variable. Any edit that inserts, removes,
    /// or reorders instructions invalidates those indices, and a stale index
    /// that runs past the end of its block is rejected by index-bounds
    /// verification. This restores them from the IR as it actually stands.
    ///
    /// Variables with no remaining definition are reset to an entry site unless
    /// something still reads them, so a destroyed definition is not disguised as
    /// a legitimate entry value.
    pub fn refresh_def_sites(&mut self) {
        VariableCensus::refresh(self);
    }

    /// Reassigns all variable IDs to dense contiguous indices (0..N-1) and
    /// remaps all references in blocks.
    ///
    /// **Warning**: This invalidates any externally-held `SsaVarId` references;
    /// a caller holding them repairs them through
    /// [`Self::variable_remap`].
    ///
    /// # Returns
    ///
    /// The number of variables that received a new dense id.
    pub fn reindex_variables(&mut self) -> usize {
        let renumbering = self.reassign_dense_ids();
        let remapped = renumbering.moved;
        self.remap_var_ids_in_blocks(&renumbering);
        self.rebuild_origin_versions();
        remapped
    }

    /// Eliminates dead phi nodes whose result is never used.
    ///
    /// A phi is dead if its result variable has no consumers (no instruction
    /// or other phi uses it). Handles dead phi cycles (A uses B, B uses A,
    /// neither used elsewhere) via liveness propagation.
    ///
    /// Also bridges implicit uses from `LoadLocal`/`LoadArg` instructions
    /// to the corresponding phi nodes for that local/arg origin, ensuring
    /// phis that are read by index-based loads are not incorrectly eliminated.
    ///
    /// # Returns
    ///
    /// Whether any phi was removed.
    pub fn eliminate_dead_phis(&mut self) -> bool {
        let bound = self.var_id_bound();
        let mut all_phi_results = VarSet::new(bound);
        for block in self.blocks.iter() {
            for phi in block.phi_nodes() {
                all_phi_results.insert(phi.result());
            }
        }

        if all_phi_results.is_empty() {
            return false;
        }

        // Map from phi origin to phi result IDs for LoadLocal/LoadArg bridging,
        // built when the first such load asks for it.
        let mut origin_to_phi_results: Option<BTreeMap<VariableOrigin, Vec<SsaVarId>>> = None;

        // Phase 1: Mark phis as live if used by any non-phi instruction
        let mut live_phis = VarSet::new(bound);
        for block in self.blocks.iter() {
            for instr in block.instructions() {
                // Direct SSA uses
                instr.for_each_use(|u| {
                    if all_phi_results.contains(u) {
                        live_phis.insert(u);
                    }
                });

                // Implicit uses via LoadLocal/LoadArg (index-based reads).
                // These don't appear in uses() but create a dependency on
                // the corresponding PHI node for that local/arg origin.
                match instr.op() {
                    SsaOp::LoadLocal { local_index, .. } => {
                        let origin = VariableOrigin::Local(*local_index);
                        if let Some(phi_results) = self
                            .phi_results_by_origin(&mut origin_to_phi_results)
                            .get(&origin)
                        {
                            for &phi_result in phi_results {
                                live_phis.insert(phi_result);
                            }
                        }
                    }
                    SsaOp::LoadArg { arg_index, .. } => {
                        let origin = VariableOrigin::Argument(*arg_index);
                        if let Some(phi_results) = self
                            .phi_results_by_origin(&mut origin_to_phi_results)
                            .get(&origin)
                        {
                            for &phi_result in phi_results {
                                live_phis.insert(phi_result);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Phase 2: Propagate liveness through phi operands
        let mut changed = true;
        while changed {
            changed = false;
            for block in self.blocks.iter() {
                for phi in block.phi_nodes() {
                    if live_phis.contains(phi.result()) {
                        for op in phi.operands() {
                            let value = op.value();
                            if all_phi_results.contains(value) && live_phis.insert(value) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Phase 3: Remove dead phis (all_phi_results - live_phis). The
        // placeholder is no variable, so a phi holding it stays.
        if all_phi_results
            .iter()
            .all(|result| live_phis.contains(result))
        {
            return false;
        }

        for block in self.blocks.iter_mut() {
            block
                .phi_nodes_mut()
                .retain(|phi| phi.result().is_placeholder() || live_phis.contains(phi.result()));
        }

        // As in `eliminate_trivial_phis`: the orphaned rows stay until
        // `compact_variables` removes them, because that is the only path that
        // renumbers `variables` and remaps the ids in blocks afterwards.
        // Dropping them here would break `variables[i].id().index() == i`.
        true
    }

    /// Returns the phi results of each origin, building the map on first use.
    ///
    /// # Arguments
    ///
    /// * `cache` - Where the map is kept between calls; empty until built.
    fn phi_results_by_origin<'c>(
        &self,
        cache: &'c mut Option<BTreeMap<VariableOrigin, Vec<SsaVarId>>>,
    ) -> &'c BTreeMap<VariableOrigin, Vec<SsaVarId>> {
        cache.get_or_insert_with(|| {
            let mut map: BTreeMap<VariableOrigin, Vec<SsaVarId>> = BTreeMap::new();
            for block in self.blocks.iter() {
                for phi in block.phi_nodes() {
                    map.entry(phi.origin()).or_default().push(phi.result());
                }
            }
            map
        })
    }

    /// Shrinks `num_locals` to the actual maximum local index in use.
    ///
    /// After `compact_variables()` removes unused variables, `num_locals` may
    /// exceed the actual maximum local index referenced. This scans all
    /// `VariableOrigin::Local(idx)` references (variables, phi nodes, and
    /// `LoadLocal`/`LoadLocalAddr` instructions) to find the true maximum, then
    /// sets `num_locals = max(max_used + 1, original_num_locals)`.
    ///
    /// The `original_num_locals` floor ensures we never drop below the method's
    /// declared local count (those locals have default-initialization semantics).
    pub fn shrink_num_locals(&mut self) {
        let mut max_local_idx: Option<u16> = None;

        // From variables
        for var in &self.variables {
            if let VariableOrigin::Local(idx) = var.origin() {
                max_local_idx = Some(max_local_idx.map_or(idx, |cur| cur.max(idx)));
            }
        }

        // From phi nodes
        for block in self.blocks.iter() {
            for phi in block.phi_nodes() {
                if let VariableOrigin::Local(idx) = phi.origin() {
                    max_local_idx = Some(max_local_idx.map_or(idx, |cur| cur.max(idx)));
                }
            }
        }

        // From LoadLocal and LoadLocalAddr instructions
        for block in self.blocks.iter() {
            for instr in block.instructions() {
                match instr.op() {
                    SsaOp::LoadLocal { local_index, .. }
                    | SsaOp::LoadLocalAddr { local_index, .. } => {
                        max_local_idx =
                            Some(max_local_idx.map_or(*local_index, |cur| cur.max(*local_index)));
                    }
                    _ => {}
                }
            }
        }

        let needed = max_local_idx.map_or(0, |idx| (idx as usize).saturating_add(1));
        self.num_locals = needed.max(self.original_num_locals);
    }

    /// Shrinks a candidate removal set until nothing it removes is still read.
    ///
    /// Removing an instruction (rewriting it to [`SsaOp::Nop`]) or a phi
    /// destroys the values it defines. An instruction or phi operand outside
    /// the set that still reads one of them would read a value nothing defines,
    /// which the verifier rejects. So a candidate whose result has a reader
    /// outside the set is dropped from it, and once dropped it survives, so the
    /// values it reads stay too: the sets shrink to a fixpoint. What is left is
    /// the largest subset whose definitions are read only inside the subset, so
    /// a cycle of candidates through phis that nothing else reads goes as a
    /// whole.
    ///
    /// The sets must say what the caller actually removes. A candidate the
    /// caller leaves in place, such as a `Return` it cannot rewrite, does not
    /// belong in `instrs`: counted as going, it no longer protects the values
    /// it reads. A candidate that defines nothing, such as a branch the caller
    /// folds to a jump, destroys nothing and is never dropped.
    ///
    /// # Arguments
    ///
    /// * `instrs` - Candidate `(block, instruction)` locations. Shrunk in place.
    /// * `phis` - Candidate `(block, phi)` locations. Shrunk in place.
    pub fn retain_unread_removals(
        &self,
        instrs: &mut BTreeSet<(usize, usize)>,
        phis: &mut BTreeSet<(usize, usize)>,
    ) {
        loop {
            // Every value the candidates would destroy, mapped to the candidate
            // that has to stay if the value turns out to be read.
            let mut by_instr: BTreeMap<SsaVarId, (usize, usize)> = BTreeMap::new();
            for &(block_idx, instr_idx) in instrs.iter() {
                if let Some(instr) = self
                    .blocks
                    .get(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                {
                    for def in instr.op().defs() {
                        by_instr.insert(def, (block_idx, instr_idx));
                    }
                }
            }
            let mut by_phi: BTreeMap<SsaVarId, (usize, usize)> = BTreeMap::new();
            for &(block_idx, phi_idx) in phis.iter() {
                if let Some(phi) = self
                    .blocks
                    .get(block_idx)
                    .and_then(|block| block.phi_nodes().get(phi_idx))
                {
                    by_phi.insert(phi.result(), (block_idx, phi_idx));
                }
            }
            if by_instr.is_empty() && by_phi.is_empty() {
                return;
            }

            // Whatever is not being removed survives, and its reads keep the
            // definitions behind them.
            let mut kept_instrs: BTreeSet<(usize, usize)> = BTreeSet::new();
            let mut kept_phis: BTreeSet<(usize, usize)> = BTreeSet::new();
            let mut rescue = |var: SsaVarId| {
                if let Some(&site) = by_instr.get(&var) {
                    kept_instrs.insert(site);
                }
                if let Some(&site) = by_phi.get(&var) {
                    kept_phis.insert(site);
                }
            };
            for (block_idx, block) in self.blocks.iter().enumerate() {
                for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                    if phis.contains(&(block_idx, phi_idx)) {
                        continue;
                    }
                    for operand in phi.operands() {
                        rescue(operand.value());
                    }
                }
                for (instr_idx, instr) in block.instructions().iter().enumerate() {
                    if instrs.contains(&(block_idx, instr_idx)) {
                        continue;
                    }
                    instr.op().for_each_use(&mut rescue);
                }
            }
            if kept_instrs.is_empty() && kept_phis.is_empty() {
                return;
            }
            instrs.retain(|site| !kept_instrs.contains(site));
            phis.retain(|site| !kept_phis.contains(site));
        }
    }

    /// Removes the listed instructions whose results nothing else reads.
    ///
    /// Each listed instruction is rewritten to [`SsaOp::Nop`] unless a value
    /// it defines is still read by an instruction or phi operand outside the
    /// list. One that stays keeps what it reads in turn, so the list is
    /// settled to a fixpoint first ([`Self::retain_unread_removals`]); what
    /// stays is still correct code, and an instruction that defines nothing
    /// always goes. A terminator never goes, since its block needs one, nor
    /// does a write to a slot the function reads as storage
    /// ([`StorageSlots`]), which a read through the slot sees; a location
    /// naming no instruction, or a `Nop`, is ignored.
    ///
    /// [`StorageSlots`]: crate::ir::function::StorageSlots
    ///
    /// The caller decides which instructions are worth removing and answers
    /// for their other effects: a call listed here goes when its result is
    /// unread, whatever else it does.
    ///
    /// # Arguments
    ///
    /// * `sites` - `(block, instruction)` locations of the instructions to
    ///   remove.
    ///
    /// # Returns
    ///
    /// The removed instructions in location order, each with the operation it
    /// held.
    pub fn remove_unread_instructions(
        &mut self,
        sites: impl IntoIterator<Item = (usize, usize)>,
    ) -> Vec<(usize, usize, SsaOp<T>)> {
        let mut instrs: BTreeSet<(usize, usize)> = sites
            .into_iter()
            .filter(|&(block_idx, instr_idx)| {
                self.blocks
                    .get(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                    .is_some_and(|instr| {
                        !instr.is_terminator() && !matches!(instr.op(), SsaOp::Nop)
                    })
            })
            .collect();
        // A write to a slot something still reads as storage stays, which
        // keeps what it reads; a slot whose only reads go in the same removal
        // is no storage once they do. Both only ever shrink the set, so the
        // rounds end.
        let slots = self.storage_slots();
        loop {
            self.retain_unread_removals(&mut instrs, &mut BTreeSet::new());
            let remaining = slots.excluding(self, &instrs);
            let before = instrs.len();
            instrs.retain(|&(block_idx, instr_idx)| {
                !self
                    .blocks
                    .get(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                    .is_some_and(|instr| remaining.written_by(self, instr.op()))
            });
            if instrs.len() == before {
                break;
            }
        }

        let mut removed = Vec::with_capacity(instrs.len());
        for (block_idx, instr_idx) in instrs {
            if let Some(instr) = self
                .blocks
                .get_mut(block_idx)
                .and_then(|block| block.instructions_mut().get_mut(instr_idx))
            {
                let op = instr.op().clone();
                instr.set_op(SsaOp::Nop);
                removed.push((block_idx, instr_idx, op));
            }
        }
        for (_, _, op) in &removed {
            self.retire_definitions(op, &[]);
        }
        removed
    }
}

/// The structural cleanups an SSA function's own construction, repair and edits run.
///
/// Each leaves the function between consistent states only as part of the work
/// that calls it - variables compacted under live references, nops stripped
/// before their def sites are repaired - so they are this module tree's to
/// call, not a caller's: a pass reaches them through an edit session.
pub trait Cleanup<T: Target> {
    /// Propagates a batch of copy mappings (dest → src) through all instructions.
    ///
    /// For each mapping, replaces all uses of `dest` with `src` in instructions
    /// (NOT in phi operands - this is the safe default that avoids cross-origin
    /// phi references). Reports which copies were fully propagated vs. which
    /// still have remaining uses due to the self-referential guard.
    ///
    /// # Arguments
    ///
    /// * `copies` - Map of copy destination to source; each `dest` has its uses
    ///   rewritten to `src`, and any mapping where `dest == src` is skipped.
    ///
    /// # Returns
    ///
    /// A [`CopyPropagationResult`] recording the total uses replaced and which
    /// copy destinations were fully vs. partially propagated.
    ///
    /// # Usage
    ///
    /// This is a crate-internal method used by the copy-propagation pass. It is
    /// reachable from outside the crate through the checked edit session on
    /// [`SsaEditor::propagate_copies`](crate::ir::function::SsaEditor::propagate_copies):
    ///
    /// ```rust
    /// use std::collections::BTreeMap;
    /// use analyssa::{
    ///     ir::{function::SsaEditOptions, SsaVarId},
    ///     testing,
    /// };
    ///
    /// // The fixture ends block 0 with `v9 = copy v8`, and block 1 returns v9.
    /// let mut ssa = testing::scalar_rewrite_fixture();
    /// let dest = SsaVarId::from_index(9);
    /// let src = SsaVarId::from_index(8);
    ///
    /// let copies = BTreeMap::from([(dest, src)]);
    /// ssa.edit(SsaEditOptions::new(), |editor| {
    ///     let result = editor.propagate_copies(&copies);
    ///
    ///     // The single use of `dest` (the return in block 1) now reads `src`.
    ///     assert_eq!(result.total_replaced, 1);
    ///     assert!(result.fully_propagated.contains(dest));
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    fn propagate_copies(&mut self, copies: &BTreeMap<SsaVarId, SsaVarId>) -> CopyPropagationResult;

    /// Neutralizes Copy instructions that define the given variable by
    /// replacing them with Nop.
    ///
    /// This is used after copy propagation to eliminate dead copy instructions
    /// whose destination has been fully propagated to all use sites. Without
    /// this, a renaming construction would re-create versions for the Copy's
    /// origin, shadowing the source variable and undoing the propagation.
    ///
    /// # Arguments
    ///
    /// * `dest` - The copy destination whose defining `Copy` is neutralized.
    ///
    /// # Returns
    ///
    /// `true` if a defining `Copy` was found and replaced with `Nop`.
    fn nop_copy_defining(&mut self, dest: SsaVarId) -> bool;

    /// Returns the variables read by any live instruction or phi operand.
    ///
    /// Used by every path that would otherwise rewrite a variable's definition
    /// site to [`DefSite::entry`] after its defining instruction disappeared.
    /// That stamp is how an argument or a default-initialized local says "the
    /// caller supplies this", so applying it to a variable something still
    /// *reads* turns a dangling read into IR that looks legitimate - and
    /// [`SsaVerifier`](crate::analysis::verifier::SsaVerifier), whose job is to
    /// catch the pass that destroyed the definition, is left with nothing to
    /// see. Leaving the stale site in place is the honest answer: it still names
    /// an instruction, so the read is reported as an `UndefinedUse`.
    ///
    /// `Nop` instructions are skipped: a nopped instruction reads nothing.
    fn collect_read_variables(&self) -> VarSet;

    /// Compacts the variable table by removing orphaned variables.
    ///
    /// # Returns
    ///
    /// The number of variables that were removed.
    fn compact_variables(&mut self) -> usize;

    /// Strips Nop instructions from all blocks and reindexes variable DefSites.
    ///
    /// This is the shared implementation used by both `repair_ssa` and SSA
    /// reconstruction. After stripping Nops:
    ///
    /// 1. Non-Nop instructions that shifted get their DefSites remapped
    /// 2. Variables whose defining instruction was a Nop get reset to entry DefSite
    /// 3. Any remaining out-of-bounds DefSites are reset to entry DefSite
    fn strip_nops(&mut self);
}

impl<T: Target> Cleanup<T> for SsaFunction<T> {
    fn propagate_copies(&mut self, copies: &BTreeMap<SsaVarId, SsaVarId>) -> CopyPropagationResult {
        let bound = self.var_id_bound();
        let mut total_replaced: usize = 0;
        let mut fully_propagated = VarSet::new(bound);
        let mut partially_propagated = VarSet::new(bound);

        // Build the dominator tree once: rewriting instruction uses never
        // changes any terminator, so it stays valid across every replacement
        // below; rebuilding it per copy would repeat the whole CFG walk.
        let dominators = if self.block_count() > 0 {
            let eh = SsaCfg::from_ssa(self);
            Some(compute_dominators(&eh, NodeId::new(0)))
        } else {
            None
        };

        // Each replacement costs the uses of its own copy: the index is built
        // once for the batch and every replacement keeps it exact.
        self.recompute_uses();
        for (dest, src) in copies {
            if dest == src {
                continue;
            }

            let checked = self.replace_uses_checked_indexed(*dest, *src, dominators.as_ref(), None);
            self.inherit_rename_group(*dest, *src, checked.took(), Substitution::CopyPropagation);
            let result = checked.as_replace_result();

            if result.replaced > 0 {
                if result.is_complete() {
                    fully_propagated.insert(*dest);
                } else {
                    partially_propagated.insert(*dest);
                }
                total_replaced = total_replaced.saturating_add(result.replaced);
            }
        }

        CopyPropagationResult {
            total_replaced,
            fully_propagated,
            partially_propagated,
        }
    }

    fn nop_copy_defining(&mut self, dest: SsaVarId) -> bool {
        let mut retired = None;
        'blocks: for block in self.blocks.iter_mut() {
            for instr in block.instructions_mut() {
                if let SsaOp::Copy { dest: d, .. } = instr.op()
                    && *d == dest
                {
                    retired = Some(instr.op().clone());
                    instr.set_op(SsaOp::Nop);
                    break 'blocks;
                }
            }
        }
        let Some(op) = retired else {
            return false;
        };
        self.retire_definitions(&op, &[]);
        true
    }

    fn collect_read_variables(&self) -> VarSet {
        let mut read = VarSet::new(self.var_id_bound());
        for block in self.blocks.iter() {
            for instr in block.instructions() {
                if matches!(instr.op(), SsaOp::Nop) {
                    continue;
                }
                instr.op().for_each_use(|used| {
                    read.insert(used);
                });
            }
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    read.insert(operand.value());
                }
            }
        }
        read
    }

    fn compact_variables(&mut self) -> usize {
        VariableCensus::of(self).compact(self)
    }

    fn strip_nops(&mut self) {
        // Per block that held a Nop: where they were and how long the block
        // was, which is all a position needs to find where it moved to.
        let mut stripped: Vec<Option<StrippedNops>> = Vec::new();

        for (block_idx, block) in self.blocks.iter_mut().enumerate() {
            let instructions = block.instructions_mut();

            let positions: Vec<usize> = instructions
                .iter()
                .enumerate()
                .filter(|(_, instr)| matches!(instr.op(), SsaOp::Nop))
                .map(|(position, _)| position)
                .collect();
            if positions.is_empty() {
                continue;
            }
            let original_len = instructions.len();
            instructions.retain(|instr| !matches!(instr.op(), SsaOp::Nop));

            if stripped.len() <= block_idx {
                stripped.resize_with(block_idx.saturating_add(1), || None);
            }
            if let Some(slot) = stripped.get_mut(block_idx) {
                *slot = Some(StrippedNops {
                    positions,
                    original_len,
                });
            }
        }

        // Update variable DefSites to reflect new instruction positions.
        // A variable whose defining instruction was a Nop is reset to entry -
        // unless something still reads it, in which case the reset would
        // disguise a dangling read as a legitimate entry definition. See
        // `collect_read_variables`.
        if !stripped.is_empty() {
            let still_read = self.collect_read_variables();
            for var in &mut self.variables {
                let site = var.def_site();
                let Some(old_instr) = site.index() else {
                    continue;
                };
                let Some(Some(nops)) = stripped.get(site.block()) else {
                    continue;
                };
                if nops.positions.binary_search(&old_instr).is_ok() {
                    if !still_read.contains(var.id()) {
                        var.set_def_site(DefSite::entry());
                    }
                } else if old_instr < nops.original_len {
                    let before = nops
                        .positions
                        .partition_point(|&position| position < old_instr);
                    var.set_def_site(DefSite::instruction(
                        site.block(),
                        old_instr.saturating_sub(before),
                    ));
                }
            }
        }

        // Validate remaining DefSites are in-bounds. Catches stale DefSites
        // that existed before strip_nops was called (e.g., from passes that
        // modified instructions without updating DefSites).
        for var in &mut self.variables {
            let site = var.def_site();
            if let Some(instr_idx) = site.index() {
                let out_of_bounds = match self.blocks.get(site.block()) {
                    Some(block) => instr_idx >= block.instructions().len(),
                    None => true,
                };
                if out_of_bounds {
                    var.set_def_site(DefSite::entry());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            CallData, Callee, block::SsaBlock, exception::BlockRange, instruction::SsaInstruction,
            phi::PhiNode,
        },
        testing::{MockTarget, MockType},
    };

    /// A machine state naming a view of a value reads the value once the
    /// view is retired: as it is where the view is as wide, extended where
    /// an integer conversion widened it, and not at all through a narrowing.
    #[test]
    fn retired_views_leave_machine_states_their_operands() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let narrow = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let mut defined = |ty, index| {
            ssa.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(0, index), ty)
        };
        let zero = defined(MockType::I64, 0);
        let sign = defined(MockType::I64, 1);
        let low = defined(MockType::I16, 2);
        let same = defined(MockType::I32, 3);
        let conversion = |dest, target, unsigned| SsaOp::IntConv {
            dest,
            operand: narrow,
            target,
            overflow_check: false,
            unsigned,
        };
        let mut block = SsaBlock::new(0);
        for op in [
            conversion(zero, MockType::I64, true),
            conversion(sign, MockType::I64, false),
            conversion(low, MockType::I16, true),
            SsaOp::Copy {
                dest: same,
                src: narrow,
            },
            SsaOp::returns(None),
        ] {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        ssa.add_block(block);
        let at = InstrId::from_raw(9).expect("nonzero");
        let key = |raw| CallInputKey::new(raw);
        ssa.set_machine_state(
            at,
            MachinePoint::After,
            [(1, zero), (2, sign), (3, low), (4, same)]
                .into_iter()
                .map(|(raw, value)| CallInput {
                    key: key(raw),
                    value,
                })
                .collect(),
        );
        for index in 0..4 {
            assert!(ssa.replace_instruction_op(0, index, SsaOp::Nop));
        }
        let stated = |raw, widened| StatedRegister {
            key: key(raw),
            value: narrow,
            widened,
        };
        assert_eq!(
            ssa.resume_state(at),
            vec![
                stated(
                    1,
                    Some(Widening {
                        bits: 64,
                        signed: false
                    })
                ),
                stated(
                    2,
                    Some(Widening {
                        bits: 64,
                        signed: true
                    })
                ),
                stated(4, None),
            ]
        );
    }

    /// `B0: z = 1. B1 (try): x = arg * 2; leave B3. B2 (finally): p = phi(z
    /// thrown before the multiply, x after it). B3: return z + x` - `z` is in
    /// the finally's exceptional web and read after `x`, so it is isolated;
    /// `x`, read elsewhere too, is isolated as well, and `p` reads only copies.
    #[test]
    fn exceptional_operands_read_elsewhere_get_copies_of_their_own() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let var = |ssa: &mut SsaFunction<MockTarget>, origin, site| {
            ssa.create_variable(origin, 0, site, MockType::I32)
        };
        let arg = var(&mut ssa, VariableOrigin::Argument(0), DefSite::entry());
        let z = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(0, 0));
        let two = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(1, 0));
        let x = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(1, 1));
        let p = var(&mut ssa, VariableOrigin::Phi, DefSite::phi(2));
        let r = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(3, 0));
        let block = |index: usize, ops: Vec<SsaOp<MockTarget>>| {
            let mut block = SsaBlock::new(index);
            for op in ops {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            block
        };
        ssa.add_block(block(
            0,
            vec![
                SsaOp::Const {
                    dest: z,
                    value: ConstValue::I32(1),
                },
                SsaOp::Jump { target: 1 },
            ],
        ));
        ssa.add_block(block(
            1,
            vec![
                SsaOp::Const {
                    dest: two,
                    value: ConstValue::I32(2),
                },
                SsaOp::Mul {
                    dest: x,
                    left: arg,
                    right: two,
                    flags: None,
                },
                SsaOp::Leave { target: 3 },
            ],
        ));
        let mut finally = block(2, vec![SsaOp::EndFinally]);
        let mut handed = PhiNode::new(p, VariableOrigin::Phi);
        handed.add_operand(PhiOperand::exceptional(z, 1));
        handed.add_operand(PhiOperand::exceptional(x, 1));
        finally.add_phi(handed);
        ssa.add_block(finally);
        ssa.add_block(block(
            3,
            vec![
                SsaOp::Add {
                    dest: r,
                    left: z,
                    right: x,
                    flags: None,
                },
                SsaOp::returns(Some(r)),
            ],
        ));
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, 2),
            handler_range: BlockRange::new(2, 3),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        assert_eq!(ssa.isolate_exceptional_operands().ok(), Some(2));
        let operands: Vec<SsaVarId> = ssa
            .block(2)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| {
                phi.operands()
                    .iter()
                    .map(|operand| operand.value())
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            !operands.contains(&z) && !operands.contains(&x),
            "{operands:?}"
        );
        // Each copy follows the value it copies.
        assert!(matches!(
            ssa.block(0).and_then(|block| block.instructions().get(1)).map(SsaInstruction::op),
            Some(SsaOp::Copy { src, .. }) if *src == z
        ));
        assert!(matches!(
            ssa.block(1).and_then(|block| block.instructions().get(2)).map(SsaInstruction::op),
            Some(SsaOp::Copy { src, .. }) if *src == x
        ));
        // A second call has nothing left to isolate.
        assert_eq!(ssa.isolate_exceptional_operands().ok(), Some(0));
    }

    /// A loop's try writes `l` on one path and leaves `a`, from before the
    /// loop, on the other; the finally's phi takes both. `l` is written while
    /// `a` is still needed - the next iteration reaches the finally holding
    /// `a` - so every exceptional operand is copied in the block it is thrown
    /// from, where it alone holds the place.
    #[test]
    fn conflicting_exceptional_operands_are_copied_where_they_are_thrown() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let var = |ssa: &mut SsaFunction<MockTarget>, origin, site| {
            ssa.create_variable(origin, 0, site, MockType::I32)
        };
        let n = var(&mut ssa, VariableOrigin::Argument(0), DefSite::entry());
        let a = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(0, 0));
        let i = var(&mut ssa, VariableOrigin::Phi, DefSite::phi(1));
        let more = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(1, 0));
        let l = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(3, 0));
        let merged = var(&mut ssa, VariableOrigin::Phi, DefSite::phi(5));
        let next = var(&mut ssa, VariableOrigin::Phi, DefSite::instruction(6, 0));
        let block = |index: usize, ops: Vec<SsaOp<MockTarget>>| {
            let mut block = SsaBlock::new(index);
            for op in ops {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            block
        };
        ssa.add_block(block(
            0,
            vec![
                SsaOp::Const {
                    dest: a,
                    value: ConstValue::I32(7),
                },
                SsaOp::Jump { target: 1 },
            ],
        ));
        let mut header = block(
            1,
            vec![
                SsaOp::Clt {
                    dest: more,
                    left: i,
                    right: n,
                    unsigned: false,
                    signaling: false,
                },
                SsaOp::Branch {
                    condition: more,
                    true_target: 2,
                    false_target: 7,
                },
            ],
        );
        let mut counter = PhiNode::new(i, VariableOrigin::Phi);
        counter.add_operand(PhiOperand::new(a, 0));
        counter.add_operand(PhiOperand::new(next, 6));
        header.add_phi(counter);
        ssa.add_block(header);
        ssa.add_block(block(
            2,
            vec![SsaOp::Branch {
                condition: i,
                true_target: 4,
                false_target: 3,
            }],
        ));
        ssa.add_block(block(
            3,
            vec![
                SsaOp::Add {
                    dest: l,
                    left: i,
                    right: a,
                    flags: None,
                },
                SsaOp::Leave { target: 6 },
            ],
        ));
        ssa.add_block(block(4, vec![SsaOp::Leave { target: 6 }]));
        let mut finally = block(5, vec![SsaOp::EndFinally]);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(l, 3));
        phi.add_operand(PhiOperand::new(a, 4));
        phi.add_operand(PhiOperand::exceptional(a, 2));
        phi.add_operand(PhiOperand::exceptional(a, 3));
        phi.add_operand(PhiOperand::exceptional(l, 3));
        phi.add_operand(PhiOperand::exceptional(a, 4));
        finally.add_phi(phi);
        ssa.add_block(finally);
        ssa.add_block(block(
            6,
            vec![
                SsaOp::Add {
                    dest: next,
                    left: i,
                    right: merged,
                    flags: None,
                },
                SsaOp::Jump { target: 1 },
            ],
        ));
        ssa.add_block(block(7, vec![SsaOp::returns(Some(i))]));
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(2, 5),
            handler_range: BlockRange::new(5, 6),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        assert!(
            ssa.isolate_exceptional_operands()
                .is_ok_and(|copies| copies > 0)
        );
        let finally = ssa.block(5).and_then(|block| block.phi_nodes().first());
        for operand in finally
            .into_iter()
            .flat_map(|phi| phi.exceptional_operands())
        {
            let defined_in = ssa
                .variable(operand.value())
                .map(|variable| variable.def_site().block());
            assert_eq!(
                defined_in,
                Some(operand.predecessor()),
                "{:?} is thrown from B{} but defined elsewhere",
                operand.value(),
                operand.predecessor()
            );
        }
    }

    /// Builds `B0 -> B2`, `B1 -> B2`, where `B2` carries a phi merging one value
    /// defined in `B0` with a second operand whose value has no definition in any
    /// block. Both incoming edges are live.
    fn merge_with_one_undefined_operand() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);

        let phi_res = SsaVarId::from_index(0);
        ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::phi(2), MockType::I32);
        let defined = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        // Registered but never defined by any instruction or phi.
        let undefined = SsaVarId::from_index(2);
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(9, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: defined,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_res, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(defined, 0));
        phi.add_operand(PhiOperand::new(undefined, 1));
        b2.add_phi(phi);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(phi_res))));
        ssa.add_block(b2);

        ssa.recompute_uses();
        ssa
    }

    /// Builds `B0 -> {B1, B2} -> B3` where `B3` carries a phi whose result is
    /// never used, and where a *later* variable (`kept`) is allocated after the
    /// dead phi's result. Removing the phi's row from `variables` without
    /// renumbering shifts `kept` down a slot, so `variables[i].id().index() == i`
    /// no longer holds.
    fn dead_phi_before_a_live_variable() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);

        // Slot 0: the branch condition, defined in B0.
        let cond = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        // Slot 1: the dead phi's result, defined by the phi in B3.
        let dead = SsaVarId::from_index(1);
        ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(3), MockType::I32);
        // Slot 2: allocated *after* the dead phi - this is the one that shifts.
        let kept = SsaVarId::from_index(2);
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(3, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        for id in [1usize, 2] {
            let mut block = SsaBlock::new(id);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(block);
        }

        let mut b3 = SsaBlock::new(3);
        // Dead: `dead` is never read by any instruction or phi operand.
        let mut phi = PhiNode::new(dead, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(cond, 1));
        phi.add_operand(PhiOperand::new(cond, 2));
        b3.add_phi(phi);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: kept,
            value: ConstValue::I32(7),
        }));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(kept))));
        ssa.add_block(b3);

        ssa.recompute_uses();
        ssa
    }

    /// A chain `p1 = phi(x); p2 = phi(p1); ...` is entirely trivial and must
    /// collapse in a bounded number of fixpoint rounds.
    ///
    /// `replace_uses_checked_with` rewrites only instruction uses, so without the
    /// operand collapse each phi in the chain stays "read" by the next phi's
    /// operand and exactly one phi retires per round. The fixpoint then runs once
    /// per phi, and each round is already O(phis x function) - cubic, and
    /// measurably so: this shape reaches seconds at a few hundred phis and tens
    /// of seconds at 1,600.
    #[test]
    fn a_chain_of_trivial_phis_collapses_completely() {
        const CHAIN: usize = 512;

        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, CHAIN + 1);
        let seed = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: seed,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut prev = seed;
        for id in 1..=CHAIN {
            let origin = VariableOrigin::Local(u16::try_from(id % 1000).unwrap_or(0));
            let result = ssa.create_variable(origin, 0, DefSite::phi(id), MockType::I32);
            let mut block = SsaBlock::new(id);
            let mut phi = PhiNode::new(result, origin);
            phi.add_operand(PhiOperand::new(prev, id.saturating_sub(1)));
            block.add_phi(phi);
            if id == CHAIN {
                block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(result))));
            } else {
                block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
                    target: id.saturating_add(1),
                }));
            }
            ssa.add_block(block);
            prev = result;
        }
        ssa.recompute_uses();

        ssa.repair_ssa();

        let remaining: usize = ssa.blocks().iter().map(|b| b.phi_nodes().len()).sum();
        assert_eq!(
            remaining, 0,
            "every phi in the chain is trivial and must be eliminated"
        );
        assert_dense_variable_table(&ssa, "after collapsing a trivial phi chain");
        // The return must now read the value the chain forwarded.
        let returned = ssa
            .blocks()
            .iter()
            .find_map(|block| match block.control_terminator() {
                Some(SsaOp::Return { value: Some(v), .. }) => Some(*v),
                _ => None,
            });
        assert_eq!(
            returned,
            Some(seed),
            "the chain forwarded the seed, so the return must read it directly"
        );
    }

    /// `SsaFunction::variable(id)` is a raw index into `variables`, so the whole
    /// IR depends on `variables[i].id().index() == i`. Any method that drops a
    /// row must renumber, or every id above the hole silently resolves to a
    /// *different* variable's `def_site`/`var_type`/use list - which
    /// `can_replace_instruction_use_with_dominators` then reads to approve
    /// replacements that violate dominance.
    fn assert_dense_variable_table(ssa: &SsaFunction<MockTarget>, label: &str) {
        for (slot, var) in ssa.variables().iter().enumerate() {
            assert_eq!(
                var.id().index(),
                slot,
                "{label}: variable table density broken at slot {slot} \
                 (found id {}), so variable(id) now resolves to the wrong row",
                var.id().index()
            );
        }
    }

    /// Removing a dead phi must not shift the variable rows below it out of
    /// index alignment.
    #[test]
    fn eliminate_dead_phis_preserves_dense_variable_table() {
        let mut ssa = dead_phi_before_a_live_variable();
        assert_dense_variable_table(&ssa, "before");

        ssa.eliminate_dead_phis();

        assert!(
            ssa.block(3).unwrap().phi_nodes().is_empty(),
            "the dead phi should have been removed"
        );
        assert_dense_variable_table(&ssa, "after eliminate_dead_phis");
    }

    /// Collapsing a trivial phi in repair mode must keep the variable table
    /// dense.
    #[test]
    fn eliminate_trivial_phis_preserves_dense_variable_table() {
        let mut ssa = dead_phi_before_a_live_variable();

        // The phi merges the same value on both edges, so it is trivial as well
        // as dead; repair mode removes it through the trivial-phi path.
        ssa.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });

        assert_dense_variable_table(&ssa, "after eliminate_trivial_phis");
    }

    /// `b0: x = 7; goto b1`, `b1: p = phi(x); return p`, with no use list
    /// computed: `create_variable` and `add_instruction` record none.
    fn trivial_phi_with_no_use_lists() -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let p = ssa.create_variable(VariableOrigin::Local(0), 1, DefSite::phi(1), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(p, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(x, 0));
        b1.add_phi(phi);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(p))));
        ssa.add_block(b1);
        (ssa, x, p)
    }

    /// Whether the last instruction of block 1 reads `var`.
    fn return_reads(ssa: &SsaFunction<MockTarget>, var: SsaVarId) -> bool {
        ssa.block(1)
            .and_then(|block| block.instructions().last())
            .is_some_and(|instruction| instruction.op().uses_var(var))
    }

    /// **Collapsing a trivial phi needs no use list from its caller.** The
    /// round builds the index it replaces through, so a function nothing has
    /// indexed still has every read of the phi rewritten.
    #[test]
    fn a_trivial_phi_is_collapsed_without_a_use_list() {
        let (mut ssa, x, p) = trivial_phi_with_no_use_lists();
        assert!(return_reads(&ssa, p), "the fixture reads the phi");
        ssa.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });
        assert!(return_reads(&ssa, x));
        assert!(!return_reads(&ssa, p));
    }

    /// **Propagating copies needs no use list from its caller**, for the
    /// reason above.
    #[test]
    fn copies_propagate_without_a_use_list() {
        let (mut ssa, x, p) = trivial_phi_with_no_use_lists();
        let result = ssa.propagate_copies(&BTreeMap::from([(p, x)]));
        assert_eq!(result.total_replaced, 1);
        assert!(return_reads(&ssa, x));
        assert!(!return_reads(&ssa, p));
    }

    /// Builds two loops that carry one value `x` around three phis, none
    /// trivial alone: `p1 = phi(x, p2)` in B1, `p2 = phi(p1, p3)` in B2 and
    /// `p3 = phi(x, p2)` in B3, with B4 returning `p2`. B0 enters B1 or B3; B2
    /// goes back to B1 or on to B5, which goes back to B3 or out to B4.
    fn phis_carrying_one_value_around_two_loops() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Bool,
        );
        let x = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let [p1, p2, p3] = [1usize, 2, 3].map(|block| {
            ssa.create_variable(
                VariableOrigin::Local(0),
                1,
                DefSite::phi(block),
                MockType::I32,
            )
        });
        let phi = |result, operands: &[(SsaVarId, usize)]| {
            let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
            for &(value, predecessor) in operands {
                phi.add_operand(PhiOperand::new(value, predecessor));
            }
            phi
        };
        let branch = |true_target, false_target| {
            SsaInstruction::synthetic(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            })
        };
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(branch(1, 3));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_phi(phi(p1, &[(x, 0), (p2, 2)]));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_phi(phi(p2, &[(p1, 1), (p3, 3)]));
        b2.add_instruction(branch(1, 5));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_phi(phi(p3, &[(x, 0), (p2, 5)]));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b3);
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(p2))));
        ssa.add_block(b4);
        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(branch(3, 4));
        ssa.add_block(b5);
        ssa.recompute_uses();
        ssa
    }

    /// **Phis that read only each other and one value are that value.** In
    /// both modes (repair, and a rebuild that knows every block reachable),
    /// the three phis carrying `x` around two loops are removed and the
    /// return reads `x`, the variable table kept dense. A cycle reading two
    /// values from outside is no one value and stays.
    #[test]
    fn phis_that_read_only_each_other_and_one_value_are_that_value() {
        let x = SsaVarId::from_index(1);
        let mut reachable = BitSet::new(6);
        for block in 0..6 {
            reachable.insert(block);
        }
        for rebuild in [false, true] {
            let mut ssa = phis_carrying_one_value_around_two_loops();
            let (eliminated, changed) = ssa.eliminate_trivial_phis(&TrivialPhiOptions {
                reachable: rebuild.then_some(&reachable),
            });
            assert!(changed, "rebuild={rebuild}");
            assert_eq!(eliminated, 3, "rebuild={rebuild}: {ssa}");
            assert!(
                ssa.blocks()
                    .iter()
                    .all(|block| block.phi_nodes().is_empty()),
                "rebuild={rebuild}: {ssa}"
            );
            let returned = ssa
                .block(4)
                .and_then(|block| block.instructions().last())
                .map(|i| i.op().clone());
            assert_eq!(
                returned,
                Some(SsaOp::returns(Some(x))),
                "rebuild={rebuild}: {ssa}"
            );
            assert_dense_variable_table(&ssa, "after the cycle's elimination");
        }
        // A second value from outside: B3 enters with `y` instead of `x`.
        let mut ssa = phis_carrying_one_value_around_two_loops();
        let y = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        if let Some(b0) = ssa.block_mut(0) {
            b0.instructions_mut().insert(
                1,
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: y,
                    value: ConstValue::I32(8),
                }),
            );
        }
        if let Some(phi) = ssa
            .block_mut(3)
            .and_then(|block| block.phi_nodes_mut().first_mut())
        {
            for operand in phi.operands_mut() {
                if operand.predecessor() == 0 {
                    operand.set_value(y);
                }
            }
        }
        ssa.refresh_def_sites();
        ssa.recompute_uses();
        let (eliminated, _) = ssa.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });
        assert_eq!(eliminated, 0, "{ssa}");
    }

    /// **A trivial phi that cannot be retired does not keep a cycle.** Beside
    /// the three phis carrying `x` around two loops, B4 holds `q = phi(z)`,
    /// typed `u64` over its `i32` source, which the copy reading it cannot
    /// take: `q` stays, every round, and the cycle is still `x`.
    #[test]
    fn a_trivial_phi_that_stays_does_not_keep_a_cycle() {
        let x = SsaVarId::from_index(1);
        let mut ssa = phis_carrying_one_value_around_two_loops();
        let z = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let q = ssa.create_variable(VariableOrigin::Local(2), 1, DefSite::phi(4), MockType::U64);
        let w = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(4, 0),
            MockType::U64,
        );
        if let Some(b0) = ssa.block_mut(0) {
            b0.instructions_mut().insert(
                1,
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: z,
                    value: ConstValue::I32(9),
                }),
            );
        }
        if let Some(b4) = ssa.block_mut(4) {
            let mut phi = PhiNode::new(q, VariableOrigin::Local(2));
            phi.add_operand(PhiOperand::new(z, 5));
            b4.add_phi(phi);
            b4.instructions_mut().insert(
                0,
                SsaInstruction::synthetic(SsaOp::Copy { dest: w, src: q }),
            );
        }
        ssa.refresh_def_sites();
        ssa.recompute_uses();
        let (eliminated, _) = ssa.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });
        assert_eq!(eliminated, 3, "{ssa}");
        let phis: Vec<SsaVarId> = ssa
            .blocks()
            .iter()
            .flat_map(|block| block.phi_nodes().iter().map(PhiNode::result))
            .collect();
        assert_eq!(phis, vec![q], "{ssa}");
        let returned = ssa
            .block(4)
            .and_then(|block| block.instructions().last())
            .map(|i| i.op().clone());
        assert_eq!(returned, Some(SsaOp::returns(Some(x))), "{ssa}");
    }

    /// **A trivial phi differing from its source in sign alone is its
    /// source.** `q = phi(z)`, typed `u32` over its `i32` source as a phi a
    /// pruned edge left one operand can be, is retired and the copy reading
    /// it reads `z`; typed `u64`, a join that extends its operand, it stays
    /// (the control).
    #[test]
    fn a_trivial_phi_differing_from_its_source_in_sign_alone_is_its_source() {
        let run = |phi_type: MockType| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
            let z = ssa.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            );
            let q = ssa.create_variable(VariableOrigin::Local(0), 1, DefSite::phi(1), phi_type);
            let w = ssa.create_variable(
                VariableOrigin::Local(1),
                0,
                DefSite::instruction(1, 0),
                phi_type,
            );
            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: z,
                value: ConstValue::I32(9),
            }));
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
            ssa.add_block(b0);
            let mut b1 = SsaBlock::new(1);
            let mut phi = PhiNode::new(q, VariableOrigin::Local(0));
            phi.add_operand(PhiOperand::new(z, 0));
            b1.add_phi(phi);
            b1.add_instruction(SsaInstruction::synthetic(SsaOp::Copy { dest: w, src: q }));
            b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(w))));
            ssa.add_block(b1);
            ssa.recompute_uses();
            let (eliminated, _) =
                ssa.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });
            let copied = ssa
                .block(1)
                .and_then(|block| block.instructions().first())
                .map(|i| i.op().clone());
            (eliminated, copied, z, q, w)
        };
        let (eliminated, copied, z, _, w) = run(MockType::U32);
        assert_eq!(eliminated, 1);
        assert_eq!(copied, Some(SsaOp::Copy { dest: w, src: z }));
        let (eliminated, copied, _, q, w) = run(MockType::U64);
        assert_eq!(eliminated, 0, "a wider join stays");
        assert_eq!(copied, Some(SsaOp::Copy { dest: w, src: q }));
    }

    /// An operand on a live incoming edge must survive pruning even when its
    /// value has no reachable definition. Dropping it leaves the phi without an
    /// operand for a real predecessor (`MissingPhiOperand`), and shrinks the phi
    /// to a single operand that trivial-phi simplification then collapses into a
    /// copy - silently discarding the other merged value.
    #[test]
    fn prune_phi_operands_keeps_live_edges_with_undefined_values() {
        let mut ssa = merge_with_one_undefined_operand();
        let mut reachable = BitSet::new(3);
        for block in 0..3 {
            reachable.insert(block);
        }

        let pruned = ssa.prune_phi_operands(&reachable);

        assert_eq!(pruned, 0, "no live-edge operand may be pruned");
        let operands = ssa.block(2).unwrap().phi_nodes()[0].operands();
        assert_eq!(
            operands.len(),
            2,
            "both incoming edges must retain an operand"
        );
        let mut preds: Vec<usize> = operands.iter().map(PhiOperand::predecessor).collect();
        preds.sort_unstable();
        assert_eq!(preds, vec![0, 1]);
    }

    /// Control: an operand naming a block that is not a predecessor is still
    /// pruned. Structural staleness remains the sole removal criterion.
    #[test]
    fn prune_phi_operands_drops_operands_from_non_predecessors() {
        let mut ssa = merge_with_one_undefined_operand();
        let defined = SsaVarId::from_index(1);
        // B1 no longer reaches B2, so its operand is stale.
        ssa.block_mut(1)
            .unwrap()
            .instructions_mut()
            .last_mut()
            .unwrap()
            .set_op(SsaOp::returns(None));

        let mut reachable = BitSet::new(3);
        for block in 0..3 {
            reachable.insert(block);
        }

        let pruned = ssa.prune_phi_operands(&reachable);

        assert_eq!(pruned, 1, "the stale operand must be pruned");
        let operands = ssa.block(2).unwrap().phi_nodes()[0].operands();
        assert_eq!(operands.len(), 1);
        assert_eq!(operands[0].predecessor(), 0);
        assert_eq!(operands[0].value(), defined);
    }

    /// **A chain of substitutions lands every use on its end**, in
    /// instructions and phi operands alike. Applied one pair at a time,
    /// `(v2 -> v1)` after `(v1 -> v0)` would put `v1` - the value being
    /// retired - back into use.
    #[test]
    fn resolved_substitutions_land_on_the_end_of_their_chain() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let values: Vec<SsaVarId> = (0..3)
            .map(|index| {
                ssa.create_variable(
                    VariableOrigin::Local(index),
                    0,
                    DefSite::instruction(0, index as usize),
                    MockType::I32,
                )
            })
            .collect();
        let merged =
            ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(1), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        for (index, value) in values.iter().enumerate() {
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: *value,
                value: ConstValue::I32(i32::try_from(index).unwrap_or(0)),
            }));
        }
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(values[2], 0));
        b1.add_phi(phi);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(values[2]))));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let result = ssa.replace_uses_resolved(&[(values[1], values[0]), (values[2], values[1])]);

        assert_eq!(result.replaced, 2, "the return and the phi operand");
        let block = &ssa.blocks()[1];
        assert_eq!(
            block.phi_nodes()[0].operands()[0].value(),
            values[0],
            "the phi operand reaches the end of the chain"
        );
        assert!(
            matches!(block.instructions()[0].op(), SsaOp::Return { value: Some(v), .. } if *v == values[0]),
            "the return reaches the end of the chain"
        );
    }

    /// A block of `count` calls, each reading the one before, ending in a
    /// return of nothing. Returns the function and the call results in order.
    fn call_chain(count: u32) -> (SsaFunction<MockTarget>, Vec<SsaVarId>) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut block = SsaBlock::new(0);
        let mut values = Vec::new();
        for index in 0..count {
            let dest = SsaVarId::from_index(index as usize);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::from(CallData::new(
                Some(dest),
                Callee::Direct(index),
                values.last().copied().into_iter().collect(),
            ))));
            values.push(dest);
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        (ssa, values)
    }

    /// An unread chain goes as a whole, though only the removal of its last
    /// member frees the others.
    #[test]
    fn unread_chain_is_removed_to_a_fixpoint() {
        let (mut ssa, _) = call_chain(3);
        let removed = ssa.remove_unread_instructions([(0, 0), (0, 1), (0, 2)]);
        let sites: Vec<(usize, usize)> = removed.iter().map(|(b, i, _)| (*b, *i)).collect();
        assert_eq!(sites, [(0, 0), (0, 1), (0, 2)]);
        assert!(
            matches!(&removed[1].2, SsaOp::Call(call) if call.callee == Callee::Direct(1)),
            "each removal carries the operation it held"
        );
        assert!(
            ssa.blocks()[0].instructions()[..3]
                .iter()
                .all(|instr| matches!(instr.op(), SsaOp::Nop))
        );
    }

    /// A result read outside the list keeps its definition, and the kept
    /// instruction keeps what it reads; the rest of the list still goes.
    #[test]
    fn a_value_read_elsewhere_keeps_its_definition_and_its_inputs() {
        let (mut ssa, values) = call_chain(4);
        // The last call reads the second; only the first two are listed.
        let removed = ssa.remove_unread_instructions([(0, 0), (0, 1)]);
        assert!(
            removed.is_empty(),
            "the second call is read, so the first is too"
        );
        assert!(matches!(
            ssa.blocks()[0].instructions()[1].op(),
            SsaOp::Call(call) if call.dest == Some(values[1])
        ));

        // A terminator listed with the definition it reads stays, and keeps it.
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let value = SsaVarId::from_index(0);
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(value))));
        ssa.add_block(block);
        assert!(ssa.remove_unread_instructions([(0, 0), (0, 1)]).is_empty());
        assert!(matches!(
            ssa.blocks()[0].instructions()[1].op(),
            SsaOp::Return { value: Some(v), .. } if *v == value
        ));
    }

    /// A phi cycle nothing else reads is removable as a whole; one read by a
    /// surviving instruction keeps every member of the cycle and the
    /// definitions feeding it.
    #[test]
    fn retained_removals_take_phi_cycles_whole() {
        // B0: v0 = const; jump B1. B1: v1 = phi(v0 from B0, v2 from B1);
        // v2 = neg v1; branch to B1 or B2. B2: return v1 when `read`.
        let build = |read: bool| {
            let (v0, v1, v2, cond) = (
                SsaVarId::from_index(0),
                SsaVarId::from_index(1),
                SsaVarId::from_index(2),
                SsaVarId::from_index(3),
            );
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: v0,
                value: ConstValue::I32(1),
            }));
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: cond,
                value: ConstValue::I32(0),
            }));
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
            ssa.add_block(b0);
            let mut b1 = SsaBlock::new(1);
            let mut phi = PhiNode::new(v1, VariableOrigin::Local(0));
            phi.add_operand(PhiOperand::new(v0, 0));
            phi.add_operand(PhiOperand::new(v2, 1));
            b1.add_phi(phi);
            b1.add_instruction(SsaInstruction::synthetic(SsaOp::Neg {
                dest: v2,
                operand: v1,
                flags: None,
            }));
            b1.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
                condition: cond,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b1);
            let mut b2 = SsaBlock::new(2);
            b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(
                read.then_some(v1),
            )));
            ssa.add_block(b2);
            ssa
        };

        let candidates = || (BTreeSet::from([(0, 0), (1, 0)]), BTreeSet::from([(1, 0)]));

        let ssa = build(false);
        let (mut instrs, mut phis) = candidates();
        ssa.retain_unread_removals(&mut instrs, &mut phis);
        assert_eq!(
            instrs,
            BTreeSet::from([(0, 0), (1, 0)]),
            "the cycle goes whole"
        );
        assert_eq!(phis, BTreeSet::from([(1, 0)]));

        let ssa = build(true);
        let (mut instrs, mut phis) = candidates();
        ssa.retain_unread_removals(&mut instrs, &mut phis);
        assert!(
            instrs.is_empty(),
            "a read phi keeps its operands' definitions"
        );
        assert!(phis.is_empty());
    }

    /// A repair of a function that is already as repair leaves it removes
    /// nothing and renumbers nothing: the function comes out exactly as it went
    /// in, the registry of origins included, and a caller's record of where
    /// variables moved stays the identity.
    #[test]
    fn a_repair_with_nothing_to_remove_changes_nothing() {
        let mut ssa = dead_phi_before_a_live_variable();
        ssa.repair_ssa();
        ssa.reset_variable_remap();
        let repaired = format!("{ssa:?}");

        ssa.repair_ssa();

        assert_eq!(format!("{ssa:?}"), repaired);
        for var in ssa.variables() {
            assert_eq!(
                ssa.variable_remap().get(var.id()),
                Some(var.id()),
                "{:?} did not move",
                var.id()
            );
        }
    }

    /// A phi nothing reads goes, and its row goes with it: what compaction
    /// knows about the function is taken after the phi is gone.
    #[test]
    fn a_dead_phis_row_leaves_the_table_with_the_phi() {
        let mut ssa = dead_phi_before_a_live_variable();
        assert_eq!(ssa.variables().len(), 3);

        ssa.repair_ssa();

        assert!(
            ssa.blocks()
                .iter()
                .all(|block| block.phi_nodes().is_empty())
        );
        assert_eq!(ssa.variables().len(), 2);
        assert_dense_variable_table(&ssa, "after removing a dead phi");
    }

    /// A refresh restores the definition sites and takes the same census a
    /// separate walk takes, over every id the function names, including ones
    /// past the variable table.
    #[test]
    fn a_refresh_takes_the_census_of_every_named_id() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let defined = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 5),
            MockType::I32,
        );
        let beyond = SsaVarId::from_index(40);
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: defined,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
            dest: beyond,
            src: defined,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(beyond))));
        ssa.add_block(block);

        let refreshed = VariableCensus::refresh(&mut ssa);
        let walked = VariableCensus::of(&ssa);
        assert_eq!(
            refreshed.defined.iter().collect::<Vec<_>>(),
            walked.defined.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            refreshed.read.iter().collect::<Vec<_>>(),
            walked.read.iter().collect::<Vec<_>>()
        );
        assert!(refreshed.defined.contains(beyond));
        assert_eq!(
            ssa.variable(defined).map(|row| row.def_site()),
            Some(DefSite::instruction(0, 0)),
            "the stale site is restored"
        );
    }

    /// Rows that share an id are kept or dropped together: the id is what
    /// something defines or reads, and a row that supplies the value at entry
    /// answers for every row of that id.
    #[test]
    fn rows_sharing_an_id_are_kept_whole_when_one_supplies_it() {
        let build = |supplier_version: u32| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
            let defined = ssa.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            );
            let supplier = ssa.create_variable(
                VariableOrigin::Local(1),
                supplier_version,
                DefSite::entry(),
                MockType::I32,
            );
            let twin = ssa.create_variable(
                VariableOrigin::Local(1),
                supplier_version.saturating_add(1),
                DefSite::entry(),
                MockType::I32,
            );
            if let Some(row) = ssa.variable_mut(twin) {
                row.set_id(supplier);
            }
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: defined,
                value: ConstValue::I32(1),
            }));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(defined))));
            ssa.add_block(block);
            ssa.recompute_uses();
            ssa
        };

        // Version 0 of a local with an entry site is a value the caller
        // supplies, so both rows of its id stay.
        let mut supplied = build(0);
        supplied.compact_variables();
        assert_eq!(supplied.variables().len(), 3);

        // The control: at version 1 nothing supplies the id, and with no
        // definition and no reader both rows go.
        let mut unsupplied = build(1);
        unsupplied.compact_variables();
        assert_eq!(unsupplied.variables().len(), 1);
    }

    /// Stripping `Nop`s moves each surviving definition's site with its
    /// instruction; a definition that was a `Nop` is reset to entry unless
    /// something still reads it, in which case the site is left naming an
    /// instruction so the read stays a reported dangling read.
    #[test]
    fn stripping_nops_moves_definition_sites_with_their_instructions() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 6);
        let var = |ssa: &mut SsaFunction<MockTarget>, slot: u16, instr: usize| {
            ssa.create_variable(
                VariableOrigin::Local(slot),
                0,
                DefSite::instruction(0, instr),
                MockType::I32,
            )
        };
        let a = var(&mut ssa, 0, 0);
        let gone_read = var(&mut ssa, 1, 1);
        let b = var(&mut ssa, 2, 2);
        let gone_unread = var(&mut ssa, 3, 3);
        let c = var(&mut ssa, 4, 6);
        let sum = var(&mut ssa, 5, 7);

        let constant = |dest| {
            SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(1),
            })
        };
        let nop = || SsaInstruction::synthetic(SsaOp::Nop);
        let mut block = SsaBlock::new(0);
        for instruction in [
            constant(a),
            nop(),
            constant(b),
            nop(),
            nop(),
            nop(),
            constant(c),
            SsaInstruction::synthetic(SsaOp::Add {
                dest: sum,
                left: gone_read,
                right: c,
                flags: None,
            }),
            SsaInstruction::synthetic(SsaOp::returns(Some(sum))),
        ] {
            block.add_instruction(instruction);
        }
        ssa.add_block(block);
        ssa.recompute_uses();

        ssa.strip_nops();

        let site = |ssa: &SsaFunction<MockTarget>, var| {
            ssa.variable(var).map(|variable| variable.def_site())
        };
        assert_eq!(ssa.block(0).map(|b| b.instructions().len()), Some(5));
        assert_eq!(site(&ssa, a), Some(DefSite::instruction(0, 0)));
        assert_eq!(site(&ssa, b), Some(DefSite::instruction(0, 1)));
        assert_eq!(site(&ssa, c), Some(DefSite::instruction(0, 2)));
        assert_eq!(site(&ssa, sum), Some(DefSite::instruction(0, 3)));
        assert_eq!(site(&ssa, gone_unread), Some(DefSite::entry()));
        assert!(
            site(&ssa, gone_read).is_some_and(|site| site.index().is_some()),
            "a nopped definition something still reads is not disguised as an entry value"
        );
    }
}
