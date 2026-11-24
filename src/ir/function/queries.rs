//! Read-only query and analysis methods for SSA functions.
//!
//! These methods analyze an [`SsaFunction`] without modifying it,
//! providing information about variables, control flow, return behavior, and purity.
//!
//! # Query Categories
//!
//! | Category | Methods |
//! |----------|---------|
//! | Variable queries | `argument_variables`, `local_variables`, `variables_from_*`, `versions_of` |
//! | Counting | `phi_count`, `instruction_count`, `dead_variable_count`, `count_uses` |
//! | Definition lookup | `recorded_definition`, `get_definition`, `get_definition_instruction`, `find_phi_defining`, `try_constant_value` |
//! | Control flow | `block_predecessors`, `block_successors`, `block_has_successor` |
//! | Analysis | `return_info`, `purity`, `find_trampoline_blocks`, `find_constants`, `trace_to_phi` |
//! | Checks | `is_var_constant`, `is_parameter_variable`, `is_void_return`, `has_xor_operations` |
//!
//! # Performance Notes
//!
//! Most query methods use O(1) lookup via dense indexing or the `origin_versions` registry.
//! `recorded_definition` is the guarded, scan-free def-site dereference every other
//! definition lookup is built on: it is O(1) whatever the answer, and refuses a stale
//! site rather than reporting a neighbouring instruction. `get_definition` and
//! `get_definition_instruction` take that path first, then fall back to an O(n) scan
//! that recovers from a stale site.
//! Methods like `block_predecessors` and `count_uses` require O(n) scans of all blocks.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    BitSet,
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{Callee, SsaOp},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{SsaVarId, SsaVariable, VariableOrigin},
        varstore::VarSet,
    },
    target::Target,
};

/// How many conversions and copies [`SsaFunction::symbol_of`] follows to the
/// symbol a value holds: a lifted address converts to an integer and back,
/// and a copy or two moves it, no more.
const MAX_SYMBOL_CONVERSIONS: usize = 8;

/// One handler phi's values, which a host must place in a single storage slot.
///
/// See [`SsaFunction::exceptional_webs`] for why the obligation exists: a phi
/// with no incoming edges cannot be lowered with moves, so its operands have to
/// be in the right place already.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionalWeb {
    /// The handler or filter entry holding the phi.
    pub block: usize,
    /// Index of the phi within that block's phi list.
    pub phi_idx: usize,
    /// The rename group the phi merges, or `u32::MAX` when it has none.
    pub group: u32,
    /// The phi's result, followed by every distinct value a throw may arrive
    /// with. All of them share one storage location.
    pub values: Vec<SsaVarId>,
}

/// The argument and local slots whose address a function takes.
///
/// A slot whose address escapes is **memory**: a store through the pointer
/// changes what a later load of that slot reads, and nothing in the value graph
/// says so. Every rule that treats a slot's loads as reads of its current SSA
/// value - resolving a load to the value the slot holds, numbering two loads as
/// equal, dropping a store nothing reads - is sound only for the slots that do
/// *not* appear here.
///
/// Built by [`SsaFunction::address_taken_slots`] from the address-taking
/// operations themselves, so arguments and locals are covered alike.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AddressTakenSlots {
    /// Argument slots, indexed by argument number.
    args: BitSet,
    /// Local slots, indexed by local number.
    locals: BitSet,
}

impl AddressTakenSlots {
    /// Returns whether argument slot `index` has its address taken.
    #[must_use]
    pub fn argument(&self, index: u16) -> bool {
        self.args.contains_checked(usize::from(index))
    }

    /// Returns whether local slot `index` has its address taken.
    #[must_use]
    pub fn local(&self, index: u16) -> bool {
        self.locals.contains_checked(usize::from(index))
    }

    /// Returns whether the slot a rename group stands for has its address
    /// taken.
    ///
    /// Groups number the arguments first and the locals after them, which is
    /// the numbering [`VariableOrigin::canonical_rename_group`] produces.
    ///
    /// # Arguments
    ///
    /// * `group` - The rename group to ask about.
    /// * `num_args` - How many arguments the function has.
    #[must_use]
    pub fn group(&self, group: u32, num_args: usize) -> bool {
        let group = group as usize;
        if group < num_args {
            u16::try_from(group).is_ok_and(|index| self.argument(index))
        } else {
            u16::try_from(group.saturating_sub(num_args)).is_ok_and(|index| self.local(index))
        }
    }

    /// Returns whether any slot has its address taken.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty() && self.locals.is_empty()
    }
}

/// The argument and local slots a function reads as storage.
///
/// A slot is read as storage when something reads the slot itself rather than
/// one of its SSA versions: a `LoadLocal` or `LoadArg` of it, or an address of
/// it taken with `LoadLocalAddr` or `LoadArgAddr` (the [`AddressTakenSlots`]).
/// Such a read sees whatever was last written to the slot, so a definition of
/// one of the slot's versions is a **write** the read sees, whether or not the
/// value graph says anything reads that version. A pass that removes an
/// instruction because its results go unread - a redundant value, a
/// propagated copy, a forwarded or unused load - keeps one that writes such a
/// slot: removing it drops the write.
///
/// Built by [`SsaFunction::storage_slots`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StorageSlots {
    /// How many instructions read each slot as storage, by origin.
    reads: BTreeMap<VariableOrigin, usize>,
}

impl StorageSlots {
    /// Returns whether `origin` is one of the slots.
    ///
    /// # Arguments
    ///
    /// * `origin` - The origin of a variable.
    #[must_use]
    pub fn contains(&self, origin: VariableOrigin) -> bool {
        self.reads.get(&origin).is_some_and(|count| *count > 0)
    }

    /// Returns whether `op` writes one of the slots: whether it defines a
    /// version of one, other than by reading the slot itself.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function `op` belongs to, for its variables' origins.
    /// * `op` - The operation.
    #[must_use]
    pub fn written_by<T: Target>(&self, ssa: &SsaFunction<T>, op: &SsaOp<T>) -> bool {
        if self.is_empty() || storage_read(op).is_some() {
            return false;
        }
        op.defs().any(|def| {
            ssa.variable(def)
                .is_some_and(|variable| self.contains(variable.origin()))
        })
    }

    /// The slots still read as storage once the instructions at `sites` are
    /// gone.
    ///
    /// A slot read only by instructions that are being removed with its writes
    /// is no storage after the removal: nothing reads it any more, so its
    /// writes are as dead as any other unread value.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function the sites are in.
    /// * `sites` - `(block, instruction)` locations of the instructions
    ///   removed.
    #[must_use]
    pub fn excluding<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        sites: &BTreeSet<(usize, usize)>,
    ) -> Self {
        let mut reads = self.reads.clone();
        for &(block_idx, instr_idx) in sites {
            let read = ssa
                .block(block_idx)
                .and_then(|block| block.instructions().get(instr_idx))
                .and_then(|instr| storage_read(instr.op()));
            if let Some(count) = read.and_then(|origin| reads.get_mut(&origin)) {
                *count = count.saturating_sub(1);
            }
        }
        reads.retain(|_, count| *count > 0);
        Self { reads }
    }

    /// Returns whether no slot is read as storage.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
    }
}

/// The slot `op` reads as storage, if it reads one: by value or by address.
fn storage_read<T: Target>(op: &SsaOp<T>) -> Option<VariableOrigin> {
    match op {
        SsaOp::LoadLocal { local_index, .. } | SsaOp::LoadLocalAddr { local_index, .. } => {
            Some(VariableOrigin::Local(*local_index))
        }
        SsaOp::LoadArg { arg_index, .. } | SsaOp::LoadArgAddr { arg_index, .. } => {
            Some(VariableOrigin::Argument(*arg_index))
        }
        _ => None,
    }
}

/// A variable's definition exactly as its own `DefSite` records it.
///
/// Produced by [`SsaFunction::recorded_definition`], which is the crate's only
/// guarded, scan-free way to turn a variable into the instruction that defines
/// it. Holding it keeps the block and instruction indices alongside the
/// instruction, so a caller that needs to edit or locate the definition does
/// not recompute them.
///
/// # Correctness
///
/// The handle exists only when the recorded site really does define the
/// variable asked for: [`SsaFunction::recorded_definition`] returns `None`
/// rather than a neighbouring instruction when the site is stale. Everything
/// reachable from it therefore describes `var` and nothing else.
#[derive(Debug)]
pub struct RecordedDefinition<'a, T: Target> {
    /// Index of the block holding the defining instruction.
    block_index: usize,
    /// Index of the defining instruction within that block.
    instruction_index: usize,
    /// The defining instruction itself.
    instruction: &'a SsaInstruction<T>,
}

impl<'a, T: Target> RecordedDefinition<'a, T> {
    /// Returns the index of the block holding the defining instruction.
    #[must_use]
    pub const fn block_index(&self) -> usize {
        self.block_index
    }

    /// Returns the index of the defining instruction within its block.
    #[must_use]
    pub const fn instruction_index(&self) -> usize {
        self.instruction_index
    }

    /// Returns the defining instruction.
    #[must_use]
    pub const fn instruction(&self) -> &'a SsaInstruction<T> {
        self.instruction
    }

    /// Returns the operation performed by the defining instruction.
    #[must_use]
    pub fn op(&self) -> &'a SsaOp<T> {
        self.instruction.op()
    }
}

/// Classification of a method's return behavior.
///
/// Determined by analyzing all `Return` instructions in the SSA function
/// and tracing their values back through definitions. Used by inlining,
/// constant propagation, and other inter-procedural optimizations.
#[derive(Debug, Clone, PartialEq)]
pub enum ReturnInfo<T: Target> {
    /// Always returns this exact compile-time constant value.
    Constant(ConstValue<T>),

    /// Returns a method parameter unchanged (pass-through).
    /// The `usize` is the parameter index.
    PassThrough(usize),

    /// Returns a pure computation of parameters.
    /// Potentially foldable at call sites if all arguments are constants.
    PureComputation,

    /// Return value depends on runtime state, input, or I/O.
    /// Cannot be determined at compile time.
    Dynamic,

    /// Method returns void (no return value).
    Void,

    /// Return behavior could not be determined.
    Unknown,
}

impl<T: Target> ReturnInfo<T> {
    /// Checks if the return value is known at compile time.
    ///
    /// # Returns
    ///
    /// `true` if the return value is a constant or void.
    #[must_use]
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Constant(_) | Self::Void)
    }

    /// Checks if the return value might be foldable with known inputs.
    ///
    /// # Returns
    ///
    /// `true` if the return could be computed at compile time given known inputs.
    #[must_use]
    pub fn is_potentially_foldable(&self) -> bool {
        matches!(
            self,
            Self::Constant(_) | Self::PassThrough(_) | Self::PureComputation
        )
    }
}

/// Purity classification of a method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodPurity {
    /// Method has no side effects - safe to inline, eliminate, or reorder.
    Pure,

    /// Method only reads fields but doesn't modify state.
    ReadOnly,

    /// Method modifies no state, but an instruction of it may raise an
    /// exception (a division, a checked conversion, a cast, a load through a
    /// reference that may be null). Inlining or evaluating it keeps the
    /// exception; eliminating or reordering a call to it does not.
    MayThrow,

    /// Method modifies local state only (fields of `this` object).
    LocalMutation,

    /// Method has global side effects (I/O, static fields, exceptions, etc.).
    Impure,

    /// Purity is unknown (calls external methods, uses reflection, etc.).
    Unknown,
}

impl MethodPurity {
    /// The purity of a method doing what both `self` and `other` describe:
    /// the stronger of the two, by the order `Pure`, `ReadOnly`, `MayThrow`,
    /// `LocalMutation`, `Impure`, `Unknown`.
    #[must_use]
    pub fn join(self, other: Self) -> Self {
        let rank = |purity: Self| match purity {
            Self::Pure => 0,
            Self::ReadOnly => 1,
            Self::MayThrow => 2,
            Self::LocalMutation => 3,
            Self::Impure => 4,
            Self::Unknown => 5,
        };
        if rank(other) > rank(self) {
            other
        } else {
            self
        }
    }

    /// Checks if the method can be safely eliminated if its result is unused.
    ///
    /// # Returns
    ///
    /// `true` if the method has no observable side effects.
    #[must_use]
    pub fn can_eliminate_if_unused(&self) -> bool {
        matches!(self, Self::Pure | Self::ReadOnly)
    }

    /// Checks if the method can be safely inlined.
    ///
    /// Pure and ReadOnly methods can always be inlined. LocalMutation can
    /// be inlined but requires care with the `this` reference.
    ///
    /// # Returns
    ///
    /// `true` if the method is safe to inline.
    #[must_use]
    pub fn can_inline(&self) -> bool {
        // Pure, ReadOnly and MayThrow methods can always be inlined: the
        // inlined body raises what the call would have.
        // LocalMutation can be inlined but requires care with `this`
        matches!(
            self,
            Self::Pure | Self::ReadOnly | Self::MayThrow | Self::LocalMutation
        )
    }

    /// Checks if calls to this method can be safely reordered.
    ///
    /// # Returns
    ///
    /// `true` if calls to this method can be reordered with respect to other calls.
    #[must_use]
    pub fn can_reorder(&self) -> bool {
        matches!(self, Self::Pure)
    }
}

impl<T: Target> SsaFunction<T> {
    /// Returns the argument and local slots this function reads as storage
    /// ([`StorageSlots`]): those a `LoadLocal`, `LoadArg` or an address of
    /// either reads.
    #[must_use]
    pub fn storage_slots(&self) -> StorageSlots {
        let mut reads: BTreeMap<VariableOrigin, usize> = BTreeMap::new();
        for block in self.blocks() {
            for instr in block.instructions() {
                if let Some(origin) = storage_read(instr.op()) {
                    let count = reads.entry(origin).or_insert(0);
                    *count = count.saturating_add(1);
                }
            }
        }
        StorageSlots { reads }
    }

    /// Returns the argument and local slots whose address this function takes.
    ///
    /// One scan of the instructions for the address-taking operations. The
    /// result is the authority every consumer shares, rather than each pass
    /// re-deriving it and disagreeing about which operations count - which is
    /// how argument slots came to be left out of one.
    #[must_use]
    pub fn address_taken_slots(&self) -> AddressTakenSlots {
        let mut slots = AddressTakenSlots {
            args: BitSet::new(self.num_args().saturating_add(1)),
            locals: BitSet::new(self.num_locals().saturating_add(1)),
        };
        for block in self.blocks() {
            for instr in block.instructions() {
                match instr.op() {
                    SsaOp::LoadLocalAddr { local_index, .. } => {
                        slots.locals.insert_checked(usize::from(*local_index));
                    }
                    SsaOp::LoadArgAddr { arg_index, .. } => {
                        slots.args.insert_checked(usize::from(*arg_index));
                    }
                    _ => {}
                }
            }
        }
        slots
    }

    /// The values a handler phi merges, grouped into the sets that must share
    /// one storage location.
    ///
    /// A phi is normally lowered by putting a move on each incoming edge. A phi
    /// at a handler or filter entry has no edges to put one on: control arrives
    /// by a throw, from part-way through a block, and nothing runs in between.
    /// The only way the handler reads the right value is therefore that every
    /// operand is *already* in the storage the handler reads - so a host
    /// assigning storage has to place the phi's result and all of its
    /// exceptional operands together, whatever its register allocator would
    /// otherwise prefer.
    ///
    /// Each returned set is one such obligation: the phi result first, then the
    /// values a throw may arrive with. Sets from different phis of the same
    /// block are separate, because they merge different storage.
    ///
    /// # Returns
    ///
    /// One [`ExceptionalWeb`] per phi that carries an exceptional operand, in
    /// block order.
    #[must_use]
    pub fn exceptional_webs(&self) -> Vec<ExceptionalWeb> {
        let mut webs = Vec::new();
        for (block_idx, block) in self.blocks().iter().enumerate() {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                let mut values: Vec<SsaVarId> = vec![phi.result()];
                for operand in phi.exceptional_operands() {
                    if !values.contains(&operand.value()) {
                        values.push(operand.value());
                    }
                }
                if values.len() > 1 {
                    webs.push(ExceptionalWeb {
                        block: block_idx,
                        phi_idx,
                        group: self.rename_group(phi.result()),
                        values,
                    });
                }
            }
        }
        webs
    }

    /// Marks every variable of an address-taken slot, and returns how many.
    ///
    /// [`SsaVariable::is_address_taken`] is the per-value form of
    /// [`Self::address_taken_slots`], stamped here so a consumer holding a
    /// variable can ask it directly. The repair boundaries refresh it, because
    /// an edit that removes the last address-taking operation makes a slot an
    /// ordinary value again.
    pub fn refresh_address_taken(&mut self) -> usize {
        let slots = self.address_taken_slots();
        let num_args = self.num_args();
        let mut marked = 0usize;
        for variable in self.variables_mut() {
            let taken = match variable.origin() {
                VariableOrigin::Argument(index) => slots.argument(index),
                VariableOrigin::Local(index) => slots.local(index),
                VariableOrigin::Phi | VariableOrigin::EntryLiveIn => false,
            };
            if taken {
                variable.set_address_taken();
                marked = marked.saturating_add(1);
            } else {
                variable.clear_address_taken();
            }
        }
        let _ = num_args;
        marked
    }

    /// Returns an iterator over argument variables (version 0).
    ///
    /// These are the initial SSA versions of arguments at method entry.
    /// Uses the version registry for O(1) lookup per argument.
    ///
    /// # Returns
    ///
    /// An iterator over argument variables with version 0.
    pub fn argument_variables(&self) -> impl Iterator<Item = &SsaVariable<T>> {
        (0..self.num_args() as u16).filter_map(|idx| {
            let origin = VariableOrigin::Argument(idx);
            self.versions_of(origin)
                .first()
                .and_then(|&id| self.variable(id))
                .filter(|v| v.version() == 0)
        })
    }

    /// Returns an iterator over local variables (version 0).
    ///
    /// These are the initial SSA versions of locals at method entry.
    /// Uses the version registry for O(1) lookup per local.
    ///
    /// # Returns
    ///
    /// An iterator over local variables with version 0.
    pub fn local_variables(&self) -> impl Iterator<Item = &SsaVariable<T>> {
        (0..self.num_locals() as u16).filter_map(|idx| {
            let origin = VariableOrigin::Local(idx);
            self.versions_of(origin)
                .first()
                .and_then(|&id| self.variable(id))
                .filter(|v| v.version() == 0)
        })
    }

    /// Finds all variables originating from a specific argument.
    ///
    /// Uses the version registry for O(1) lookup.
    ///
    /// # Arguments
    ///
    /// * `arg_index` - The argument index to filter by
    ///
    /// # Returns
    ///
    /// An iterator over all SSA versions of the specified argument.
    pub fn variables_from_argument(&self, arg_index: u16) -> impl Iterator<Item = &SsaVariable<T>> {
        let origin = VariableOrigin::Argument(arg_index);
        self.versions_of(origin)
            .iter()
            .filter_map(|&id| self.variable(id))
    }

    /// Finds all variables originating from a specific local.
    ///
    /// Uses the version registry for O(1) lookup.
    ///
    /// # Arguments
    ///
    /// * `local_index` - The local variable index to filter by
    ///
    /// # Returns
    ///
    /// An iterator over all SSA versions of the specified local variable.
    pub fn variables_from_local(&self, local_index: u16) -> impl Iterator<Item = &SsaVariable<T>> {
        let origin = VariableOrigin::Local(local_index);
        self.versions_of(origin)
            .iter()
            .filter_map(|&id| self.variable(id))
    }

    /// Returns the total number of phi nodes across all blocks.
    ///
    /// # Returns
    ///
    /// The sum of phi node counts in all blocks.
    pub fn phi_count(&self) -> usize {
        self.blocks().iter().map(SsaBlock::phi_count).sum()
    }

    /// Returns the total number of instructions across all blocks.
    ///
    /// # Returns
    ///
    /// The sum of instruction counts in all blocks.
    pub fn instruction_count(&self) -> usize {
        self.blocks().iter().map(SsaBlock::instruction_count).sum()
    }

    /// Returns an iterator over all phi nodes in the function.
    ///
    /// # Returns
    ///
    /// An iterator yielding references to all [`PhiNode`]s across all blocks.
    pub fn all_phi_nodes(&self) -> impl Iterator<Item = &PhiNode> {
        self.blocks().iter().flat_map(SsaBlock::phi_nodes)
    }

    /// Returns an iterator over all instructions in the function.
    ///
    /// # Returns
    ///
    /// An iterator yielding references to all [`SsaInstruction`]s across all blocks.
    pub fn all_instructions(&self) -> impl Iterator<Item = &SsaInstruction<T>> {
        self.blocks().iter().flat_map(SsaBlock::instructions)
    }

    /// Finds dead variables (variables with no uses).
    ///
    /// # Returns
    ///
    /// An iterator over variables that have no uses recorded.
    pub fn dead_variables(&self) -> impl Iterator<Item = &SsaVariable<T>> {
        self.variables().iter().filter(|v| v.is_dead())
    }

    /// Counts dead variables.
    ///
    /// # Returns
    ///
    /// The number of variables with no uses.
    #[must_use]
    pub fn dead_variable_count(&self) -> usize {
        self.variables().iter().filter(|v| v.is_dead()).count()
    }

    /// Reports whether any SSA version of a parameter is read.
    ///
    /// A source parameter fans out into many SSA versions as it is reassigned
    /// and merged, so a use of the parameter can appear under any of them. This
    /// scans every version sharing the argument origin and short-circuits on the
    /// first with a recorded use, rather than looking only at version 0.
    ///
    /// # Arguments
    ///
    /// * `param_index` - Zero-based parameter (argument) index to inspect.
    ///
    /// # Returns
    ///
    /// `true` if at least one SSA version of the parameter has a recorded use.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn is_parameter_used(&self, param_index: usize) -> bool {
        // Parameter indices > u16::MAX are not possible in practice
        self.variables_from_argument(param_index as u16)
            .any(|v| v.use_count() > 0)
    }

    /// Sums the recorded uses of a parameter across all of its SSA versions.
    ///
    /// Because a parameter appears under one SSA variable per reassignment, its
    /// total consumption is the sum over every version sharing the argument
    /// origin. Unlike [`is_parameter_used`](Self::is_parameter_used), this
    /// visits all versions rather than stopping at the first used one.
    ///
    /// # Arguments
    ///
    /// * `param_index` - Zero-based parameter (argument) index to inspect.
    ///
    /// # Returns
    ///
    /// The combined use count over every SSA version of the parameter.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn parameter_use_count(&self, param_index: usize) -> usize {
        // Parameter indices > u16::MAX are not possible in practice
        self.variables_from_argument(param_index as u16)
            .map(SsaVariable::use_count)
            .sum()
    }

    /// Checks if the function has any XOR operations.
    #[must_use]
    pub fn has_xor_operations(&self) -> bool {
        self.all_instructions()
            .any(|instr| matches!(instr.op(), SsaOp::Xor { .. }))
    }

    /// Checks if the function has any array element access operations.
    #[must_use]
    pub fn has_array_element_access(&self) -> bool {
        self.all_instructions().any(|instr| {
            matches!(
                instr.op(),
                SsaOp::LoadElement { .. } | SsaOp::StoreElement { .. }
            )
        })
    }

    /// Checks if the function has any field store operations.
    #[must_use]
    pub fn has_field_stores(&self) -> bool {
        self.all_instructions().any(|instr| {
            matches!(
                instr.op(),
                SsaOp::StoreField { .. } | SsaOp::StoreStaticField { .. }
            )
        })
    }

    /// Checks if the function accesses any static fields.
    #[must_use]
    pub fn has_static_field_access(&self) -> bool {
        self.all_instructions().any(|instr| {
            matches!(
                instr.op(),
                SsaOp::LoadStaticField { .. }
                    | SsaOp::StoreStaticField { .. }
                    | SsaOp::LoadStaticFieldAddr { .. }
            )
        })
    }

    /// Checks if the function has any field load operations.
    #[must_use]
    pub fn has_field_loads(&self) -> bool {
        self.all_instructions().any(|instr| {
            matches!(
                instr.op(),
                SsaOp::LoadField { .. } | SsaOp::LoadStaticField { .. }
            )
        })
    }

    /// Returns the target count of the largest switch in the function, if any.
    #[must_use]
    pub fn largest_switch_target_count(&self) -> Option<usize> {
        self.all_instructions()
            .filter_map(|instr| {
                if let SsaOp::Switch { targets, .. } = instr.op() {
                    Some(targets.len())
                } else {
                    None
                }
            })
            .max()
    }

    /// Checks if the function returns void (no return value).
    #[must_use]
    pub fn is_void_return(&self) -> bool {
        self.all_instructions()
            .any(|instr| matches!(instr.op(), SsaOp::Return { value: None, .. }))
    }

    /// Returns the definition `var`'s own def site records, when that site is
    /// still exact.
    ///
    /// This is the crate's single guarded, scan-free def-site dereference, and
    /// the primitive [`get_definition`](Self::get_definition),
    /// [`get_definition_instruction`](Self::get_definition_instruction) and
    /// [`try_constant_value`](Self::try_constant_value) are all built on. A def
    /// site is variable *metadata*, so an edit can leave it naming an
    /// instruction that defines something else; dereferencing it without
    /// checking reads a foreign instruction's operands and reports them as this
    /// variable's definition. The check is one `defs()` comparison and it is
    /// what makes the answer trustworthy.
    ///
    /// # Returns
    ///
    /// `Some` when the recorded block and instruction exist *and* that
    /// instruction defines `var`. `None` when the variable is unknown, is
    /// recorded as phi-defined (a phi has no defining instruction), or the site
    /// is stale. `None` therefore means "not answerable from the recorded
    /// site", never "answerable, and the answer is elsewhere" - callers that
    /// must survive a stale site use
    /// [`get_definition`](Self::get_definition), which falls back to a scan.
    ///
    /// # Complexity
    ///
    /// O(1). Never scans, whatever the answer.
    #[must_use]
    pub fn recorded_definition(&self, var: SsaVarId) -> Option<RecordedDefinition<'_, T>> {
        let site = self.variable(var)?.def_site();
        let instruction_index = site.index()?;
        let instruction = self.block(site.block())?.instruction(instruction_index)?;
        instruction
            .op()
            .defs()
            .any(|def| def == var)
            .then_some(RecordedDefinition {
                block_index: site.block(),
                instruction_index,
                instruction,
            })
    }

    /// The value whose computation set `flags`.
    ///
    /// A flags variable is a *secondary* definition of the operation that produced
    /// it, so it names an operation rather than a value and its own range says
    /// nothing (the range analysis records every secondary definition as `Top`,
    /// [`crate::passes::ranges`]). The fact a flag test wants is one variable
    /// over, on the same instruction's primary destination.
    ///
    /// One method for every reader of a flag test - the range fixpoint pruning
    /// an edge, the range pass folding the terminator, the guard facts - since
    /// answering in two places is how they come to disagree.
    ///
    /// # Arguments
    ///
    /// * `flags` - The flags variable a terminator tests.
    ///
    /// # Returns
    ///
    /// The producing instruction's primary destination, or `None` when no
    /// instruction defines `flags` or it defines no primary value.
    #[must_use]
    pub fn flag_source(&self, flags: SsaVarId) -> Option<SsaVarId> {
        self.recorded_definition(flags)?.instruction().op().dest()
    }

    /// Every value control transfers to: an indirect call's function
    /// pointer, an indirect branch's target, and each value one of those
    /// copies, reinterprets, converts, merges at a phi or selects.
    ///
    /// A load whose result is among them reads a code pointer the program
    /// enters: the cell a `call [g]` or `jmp [g]` goes through. Every operand a
    /// phi or a select may yield is entered on the path that yields it; a
    /// select's condition is not.
    #[must_use]
    pub fn entered_values(&self) -> BTreeSet<SsaVarId> {
        let mut pending: Vec<SsaVarId> = self
            .blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .filter_map(|instruction| match instruction.op() {
                SsaOp::Call(call) => call.callee.fptr(),
                SsaOp::IndirectBranch { target, .. } => Some(*target),
                _ => None,
            })
            .collect();
        let mut entered = BTreeSet::new();
        while let Some(var) = pending.pop() {
            if !entered.insert(var) {
                continue;
            }
            if let Some((_, phi)) = self.find_phi_defining(var) {
                pending.extend(phi.operands().iter().map(PhiOperand::value));
                continue;
            }
            match self.get_definition(var) {
                Some(SsaOp::Copy { src, .. }) => pending.push(*src),
                Some(SsaOp::Bitcast { operand, .. } | SsaOp::IntConv { operand, .. }) => {
                    pending.push(*operand);
                }
                Some(SsaOp::Select {
                    true_val,
                    false_val,
                    ..
                }) => pending.extend([*true_val, *false_val]),
                _ => {}
            }
        }
        entered
    }

    /// Returns `true` when `var` is defined at function entry rather than by any
    /// instruction or phi: an argument, a default-initialized local, or a
    /// machine-code front end's entry live-in.
    ///
    /// This test is deliberately narrow, and the narrowness is the point. The
    /// obvious formulation - "version 0 and no defining instruction" - also
    /// matches a variable whose defining instruction was *destroyed*:
    /// [`refresh_def_sites`](Self::repair_ssa) resets such a variable to
    /// [`DefSite::entry`](crate::ir::variable::DefSite::entry), leaving it
    /// indistinguishable from a real argument. A
    /// pass that nops a definition while a use survives would then produce IR
    /// that verifies clean, which is exactly how a dangling `flags` use escaped
    /// detection.
    ///
    /// Requiring the origin to name a real argument or local closes that: a
    /// temporary whose definition was removed does not.
    #[must_use]
    pub fn is_entry_defined(&self, var: &SsaVariable<T>) -> bool {
        if var.version() != 0 || var.def_site().index().is_some() {
            return false;
        }
        match var.origin() {
            // An argument origin is an explicit statement that the caller
            // supplies this value, so it needs no definition in the body. (The
            // index is not bounds-checked against `num_args`: a mismatch there is
            // a signature inconsistency, not a dangling read, and reporting it as
            // an orphan variable would be misleading.)
            VariableOrigin::Argument(_) => true,
            // A declared local is default-initialized at entry. An index beyond
            // the declared count is not a real local - it is a temporary, and a
            // temporary with no definition left is the case worth catching.
            VariableOrigin::Local(index) => usize::from(index) < self.num_locals(),
            // A phi-origin variable is defined by a phi node, full stop. The
            // rebuilder does create placeholder reaching-defs with this origin
            // for stack temp groups, and those legitimately have no definition
            // *while it is running* - but they are resolved before it verifies,
            // so exempting them here would only widen the hole. Nothing in the
            // suite depends on the exemption.
            VariableOrigin::Phi => false,
            // A machine-code front end's function-entry live-in: the caller
            // supplies it, so like an argument it needs no definition in the
            // body. This is the origin such a lifter must use - `Phi` above is
            // rejected precisely so a destroyed definition cannot masquerade as
            // an entry value, and `Argument` would assert a signature position
            // the lifter has not recovered.
            VariableOrigin::EntryLiveIn => true,
        }
    }

    /// Returns whether something defines `var`: an instruction, a phi, or the
    /// function's entry ([`Self::is_entry_defined`]).
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to look up.
    #[must_use]
    pub fn is_defined(&self, var: SsaVarId) -> bool {
        self.get_definition(var).is_some()
            || self.find_phi_defining(var).is_some()
            || self
                .variable(var)
                .is_some_and(|variable| self.is_entry_defined(variable))
    }

    /// Returns the integer constant `var` is defined as: the value of the
    /// [`SsaOp::Const`] defining it, as an `i64`, where it has one
    /// ([`ConstValue::resolved_i64`], a placed symbol its address).
    ///
    /// # Arguments
    ///
    /// * `var` - The value to read.
    #[must_use]
    pub fn const_i64(&self, var: SsaVarId) -> Option<i64> {
        match self.get_definition(var)? {
            SsaOp::Const { value, .. } => value.resolved_i64(),
            _ => None,
        }
    }

    /// The symbol `var` holds, through the conversions that keep a value's
    /// bits: a pointer converted to an integer of its width and back, a
    /// reinterpretation, a copy, each between two widths known equal. `None`
    /// where it holds no symbol constant, or a conversion may drop bits: two
    /// addresses truncated alike would compare equal though they differ.
    ///
    /// A lifted address reaches an integer comparison converted:
    /// `ptrtoint.u32 sym(0x250ec)` is the address `__TMC_END__`, as the
    /// constant it converts is.
    #[must_use]
    pub fn symbol_of(&self, var: SsaVarId) -> Option<&ConstValue<T>> {
        let mut value = var;
        for _ in 0..MAX_SYMBOL_CONVERSIONS {
            match self.get_definition(value)? {
                symbol @ SsaOp::Const {
                    value: ConstValue::Symbol(_),
                    ..
                } => {
                    let SsaOp::Const { value, .. } = symbol else {
                        return None;
                    };
                    return Some(value);
                }
                SsaOp::PtrToInt { operand, .. }
                | SsaOp::IntToPtr { operand, .. }
                | SsaOp::Bitcast { operand, .. }
                | SsaOp::Copy { src: operand, .. } => {
                    if self.bit_width(value)? != self.bit_width(*operand)? {
                        return None;
                    }
                    value = *operand;
                }
                _ => return None,
            }
        }
        None
    }

    /// Returns [`Self::const_i64`] where it is not negative, as a `u64`.
    ///
    /// # Arguments
    ///
    /// * `var` - The value to read.
    #[must_use]
    pub fn const_u64(&self, var: SsaVarId) -> Option<u64> {
        u64::try_from(self.const_i64(var)?).ok()
    }

    /// Gets the instruction operation that defines a variable.
    ///
    /// Searches through all blocks and instructions to find where the given
    /// variable is defined (appears as a destination).
    ///
    /// **Note**: This only returns definitions from instructions, not phi nodes.
    /// For phi node definitions, use [`find_phi_defining()`](Self::find_phi_defining).
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to look up.
    ///
    /// # Returns
    ///
    /// The defining `SsaOp` if found in an instruction, or `None` if the variable
    /// is defined by a phi node or not found.
    #[must_use]
    pub fn get_definition(&self, var: SsaVarId) -> Option<&SsaOp<T>> {
        // Fast path: O(1) through the crate's single guarded def-site dereference.
        if let Some(recorded) = self.recorded_definition(var) {
            return Some(recorded.op());
        }

        // A variable whose recorded definition really is a phi has no defining
        // *instruction*, so there is nothing for the scan below to find. Confirm
        // the phi actually exists before short-circuiting - an absent
        // instruction index can also mean a stale def site, which is exactly
        // what the scan recovers from. Without this, every lookup of a
        // phi-defined variable walked the whole function to return `None`, and
        // callers such as the memory pass do that per value in a loop.
        if let Some(def) = self.variable(var).map(SsaVariable::def_site)
            && def.is_phi()
            && self
                .block(def.block())
                .is_some_and(|block| block.phi_nodes().iter().any(|phi| phi.result() == var))
        {
            return None;
        }

        // No instruction defines it: an incoming value, or a variable only a
        // phi defines. There is nothing for the scan to find.
        if !self.blocks.defines_variable(var, self.var_id_bound()) {
            return None;
        }

        // Slow path: O(n) scan (DefSite may be stale after transforms or from builder)
        for block in self.blocks() {
            for instr in block.instructions() {
                let op = instr.op();
                if op.defs().any(|def| def == var) {
                    return Some(op);
                }
            }
        }
        None
    }

    /// Gets the instruction that defines a variable.
    ///
    /// Like [`get_definition()`](Self::get_definition) but returns the full
    /// `SsaInstruction` instead of just the `SsaOp`. This is needed by codegen
    /// to access `instr.result_type()`.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to look up.
    ///
    /// # Returns
    ///
    /// The defining `SsaInstruction` if found, or `None` if the variable
    /// is defined by a phi node or not found.
    #[must_use]
    pub fn get_definition_instruction(&self, var: SsaVarId) -> Option<&SsaInstruction<T>> {
        // Fast path: O(1) through the crate's single guarded def-site dereference.
        if let Some(recorded) = self.recorded_definition(var) {
            return Some(recorded.instruction());
        }

        // See `get_definition`: a confirmed phi definition has no defining
        // instruction, so the scan cannot find one and is pure cost.
        if let Some(def) = self.variable(var).map(SsaVariable::def_site)
            && def.is_phi()
            && self
                .block(def.block())
                .is_some_and(|block| block.phi_nodes().iter().any(|phi| phi.result() == var))
        {
            return None;
        }

        // See `get_definition`: with no defining instruction the scan is pure
        // cost.
        if !self.blocks.defines_variable(var, self.var_id_bound()) {
            return None;
        }

        // Slow path: O(n) scan (DefSite may be stale after transforms or from builder)
        for block in self.blocks() {
            for instr in block.instructions() {
                if instr.op().defs().any(|def| def == var) {
                    return Some(instr);
                }
            }
        }
        None
    }

    /// Checks whether replacing `result` with `source` in all uses would
    /// create a self-referential instruction (i.e., `source = f(..., source, ...)`).
    ///
    /// This happens when `source` is defined by an instruction that uses `result`.
    /// In such cases, eliminating a trivial phi `result = phi(source, result)` by
    /// replacing `result → source` would create a self-referential cycle.
    ///
    /// # Arguments
    ///
    /// * `source` - The variable that would become the replacement.
    /// * `result` - The variable being replaced (e.g., a trivial phi result).
    ///
    /// # Returns
    ///
    /// `true` if the replacement would create a self-referential instruction.
    #[must_use]
    pub fn would_create_self_reference(&self, source: SsaVarId, result: SsaVarId) -> bool {
        self.get_definition(source)
            .is_some_and(|op| op.uses_var(result))
    }

    /// Like [`would_create_self_reference`](Self::would_create_self_reference), but only
    /// considers definitions in reachable blocks. Definitions in unreachable blocks will
    /// be cleared by DCE, so they don't create real self-referential cycles.
    ///
    /// # Arguments
    ///
    /// * `source` - The variable that would become the replacement.
    /// * `result` - The variable being replaced.
    /// * `var_def_block` - Map from variable to the block that defines it.
    /// * `reachable` - Set of reachable block indices.
    ///
    /// # Returns
    ///
    /// `true` if `source` is defined in a reachable block and substituting it
    /// for `result` would form a self-reference; `false` otherwise, including
    /// when `source` is defined only in unreachable code.
    #[must_use]
    pub fn would_create_self_reference_reachable(
        &self,
        source: SsaVarId,
        result: SsaVarId,
        var_def_block: &BTreeMap<SsaVarId, usize>,
        reachable: &BitSet,
    ) -> bool {
        if let Some(&def_block) = var_def_block.get(&source)
            && reachable.contains(def_block)
        {
            return self.would_create_self_reference(source, result);
        }
        false
    }

    /// Checks if a variable is defined by a constant instruction.
    ///
    /// This is useful for analysis passes that need to identify compile-time
    /// constant values vs. runtime-computed values.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to check.
    ///
    /// # Returns
    ///
    /// `true` if the variable is defined by a `Const` instruction.
    #[must_use]
    pub fn is_var_constant(&self, var: SsaVarId) -> bool {
        self.get_definition(var)
            .is_some_and(|op| matches!(op, SsaOp::Const { .. }))
    }

    /// Gets the constant value if a variable is defined by a constant instruction.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to check.
    ///
    /// # Returns
    ///
    /// The constant value if the variable is defined by a `Const` instruction,
    /// `None` otherwise.
    #[must_use]
    pub fn get_var_constant(&self, var: SsaVarId) -> Option<&ConstValue<T>> {
        match self.get_definition(var) {
            Some(SsaOp::Const { value, .. }) => Some(value),
            _ => None,
        }
    }

    /// Returns the constant value of a variable if it was defined by a `Const` operation.
    ///
    /// Resolves the variable through [`recorded_definition`](Self::recorded_definition),
    /// so the lookup is O(1) with no fallback scan and a stale def site answers
    /// `None` rather than an unrelated `Const`'s value. Also `None` for
    /// phi-defined variables and for non-constant definitions.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to check.
    ///
    /// # Returns
    ///
    /// The constant value if the variable is defined by a `Const` instruction,
    /// `None` otherwise.
    #[must_use]
    pub fn try_constant_value(&self, var: SsaVarId) -> Option<ConstValue<T>> {
        // Scan-free by contract, so the guard is the whole defence: a stale site
        // that happens to name a `Const` belonging to another variable would
        // otherwise be reported as this variable's value.
        match self.recorded_definition(var)?.op() {
            SsaOp::Const { value, .. } => Some(value.clone()),
            _ => None,
        }
    }

    /// Finds the PHI node that defines a variable.
    ///
    /// Uses O(1) lookup via the variable's definition site when available,
    /// falling back to O(n) scan across all blocks otherwise.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable ID to find the defining PHI for.
    ///
    /// # Returns
    ///
    /// `Some((block_idx, &PhiNode))` if the variable is defined by a PHI node,
    /// `None` if the variable is not defined by a PHI or doesn't exist.
    #[must_use]
    pub fn find_phi_defining(&self, var: SsaVarId) -> Option<(usize, &PhiNode)> {
        // Try O(1) lookup via the variable's definition site
        if let Some(variable) = self.variable(var) {
            let def_site = variable.def_site();
            if def_site.is_phi() {
                // Variable is defined by a phi - look in that block
                if let Some(block) = self.block(def_site.block()) {
                    for phi in block.phi_nodes() {
                        if phi.result() == var {
                            return Some((def_site.block(), phi));
                        }
                    }
                }
            }
            // Variable exists but is not defined by a phi
            return None;
        }

        // Fallback: O(n) scan if variable not defined by a phi in its block
        for (block_idx, block) in self.iter_blocks() {
            for phi in block.phi_nodes() {
                if phi.result() == var {
                    return Some((block_idx, phi));
                }
            }
        }

        None
    }

    /// Traces a variable backward through arithmetic operations to find a PHI source.
    ///
    /// This is useful for control flow unflattening where a switch variable may be
    /// computed from a state PHI through operations like `(state ^ key) % N`.
    ///
    /// The tracing follows these operations backward:
    /// - `Rem` (remainder) and `And`: the left operand
    /// - `Xor`, `Mul`, `Add`, `Sub`: the left operand, then the right
    /// - `Shr`/`Shl` (shifts): the value operand
    /// - negation, complement and conversions: the operand
    /// - `Copy`: the source
    ///
    /// The walk goes depth first, the left operand before the right, so the phi
    /// returned is the first one that order reaches, in whatever block it is
    /// defined. It stops twenty definitions deep. A variable explored once and
    /// found to reach no phi is not explored again: the binary operations each
    /// lead into both operands, so without that a chain of `n` of them costs
    /// `2^n` steps, and a variable's answer does not depend on the path that
    /// reached it.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to trace backward from.
    ///
    /// # Returns
    ///
    /// The PHI variable that is the ultimate source, or `None` if no PHI is found.
    #[must_use]
    pub fn trace_to_phi(&self, var: SsaVarId) -> Option<SsaVarId> {
        const MAX_DEPTH: usize = 20;
        let mut visited = VarSet::new(self.var_id_bound());
        // The operands still to explore, with their depth, the next one last.
        let mut pending = vec![(var, 0usize)];
        while let Some((var, depth)) = pending.pop() {
            if depth > MAX_DEPTH || !visited.insert(var) {
                continue;
            }
            if let Some((_, phi)) = self.find_phi_defining(var) {
                return Some(phi.result());
            }
            let Some(def) = self.get_definition(var) else {
                continue;
            };
            let next = depth.saturating_add(1);
            match def {
                SsaOp::Phi { dest, .. } => return Some(*dest),
                SsaOp::Rem { left, .. } | SsaOp::And { left, .. } => pending.push((*left, next)),
                // The right operand is explored once the left one reached no
                // phi: it goes below the left on the stack.
                SsaOp::Xor { left, right, .. }
                | SsaOp::Mul { left, right, .. }
                | SsaOp::Add { left, right, .. }
                | SsaOp::Sub { left, right, .. } => {
                    pending.push((*right, next));
                    pending.push((*left, next));
                }
                SsaOp::Shr { value, .. } | SsaOp::Shl { value, .. } => {
                    pending.push((*value, next));
                }
                SsaOp::Neg { operand, .. }
                | SsaOp::Not { operand, .. }
                | SsaOp::IntConv { operand, .. }
                | SsaOp::IntToPtr { operand, .. }
                | SsaOp::PtrToInt { operand, .. }
                | SsaOp::IntToFloat { operand, .. }
                | SsaOp::FloatToInt { operand, .. }
                | SsaOp::FloatConv { operand, .. } => pending.push((*operand, next)),
                SsaOp::Copy { src, .. } => pending.push((*src, next)),
                _ => {}
            }
        }
        None
    }

    /// Returns the parameter `var` is, directly or through copies.
    ///
    /// A variable in the entry argument range (index below `num_args`) is a
    /// parameter directly. Otherwise the variable may be a copy of one, so the
    /// walk follows `Copy` sources back toward the argument range; any other
    /// definition, and a cycle of copies, ends it.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to check.
    ///
    /// # Returns
    ///
    /// The parameter index if this is a parameter variable, `None` otherwise.
    #[must_use]
    pub fn is_parameter_variable(&self, var: SsaVarId) -> Option<usize> {
        let mut visited = VarSet::new(self.var_id_bound());
        let mut var = var;
        loop {
            if !visited.insert(var) {
                return None;
            }
            if var.index() < self.num_args() {
                return Some(var.index());
            }
            match self.get_definition(var)? {
                SsaOp::Copy { src, .. } => var = *src,
                _ => return None,
            }
        }
    }

    /// Counts how many times each variable is used across all blocks.
    ///
    /// This scans all phi node operands and instruction operands to build
    /// a map of variable use counts. This is useful for optimization passes
    /// that need to know whether a variable has multiple uses (e.g., for
    /// deciding whether to inline an expression).
    ///
    /// # Returns
    ///
    /// A map from each used variable ID to its use count.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{ir::SsaVarId, testing};
    ///
    /// let ssa = testing::const_i32_return(42);
    /// let var_id = SsaVarId::from_index(0);
    /// let use_counts = ssa.count_uses();
    /// assert_eq!(use_counts.get(&var_id).copied(), Some(1));
    /// ```
    #[must_use]
    pub fn count_uses(&self) -> BTreeMap<SsaVarId, usize> {
        let mut counts = BTreeMap::new();

        for block in self.blocks() {
            // Count phi node operands
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    let entry = counts.entry(operand.value()).or_insert(0_usize);
                    *entry = entry.saturating_add(1);
                }
            }

            // Count instruction operands
            for instr in block.instructions() {
                instr.op().for_each_use(|var| {
                    let entry = counts.entry(var).or_insert(0_usize);
                    *entry = entry.saturating_add(1);
                });
            }
        }

        counts
    }

    /// Finds all trampoline blocks in this SSA function.
    ///
    /// A trampoline block is one that has no phi nodes and contains only a single
    /// unconditional control transfer (`Jump` or `Leave`). These blocks can be
    /// bypassed by redirecting predecessors directly to their targets.
    ///
    /// # Arguments
    ///
    /// * `skip_entry` - If true, skips block 0 (entry block).
    ///
    /// # Returns
    ///
    /// A map from trampoline block index to its target block index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaBlock, SsaFunction, SsaInstruction, SsaOp}};
    ///
    /// let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut entry = SsaBlock::new(0);
    /// entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
    /// ssa.add_block(entry);
    /// let mut exit = SsaBlock::new(1);
    /// exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
    /// ssa.add_block(exit);
    ///
    /// let trampolines = ssa.find_trampoline_blocks(true);
    /// assert!(trampolines.is_empty());
    /// assert_eq!(ssa.find_trampoline_blocks(false).get(&0), Some(&1));
    /// ```
    #[must_use]
    pub fn find_trampoline_blocks(&self, skip_entry: bool) -> BTreeMap<usize, usize> {
        self.iter_blocks()
            .filter(|&(block_idx, _)| !skip_entry || block_idx != 0)
            .filter_map(|(block_idx, block)| {
                block.is_trampoline().map(|target| (block_idx, target))
            })
            .collect()
    }

    /// Finds all constant definitions in this SSA function.
    ///
    /// Scans all blocks for `Const` instructions and returns a mapping from
    /// the destination variable to its constant value.
    ///
    /// # Returns
    ///
    /// A map from variable ID to its constant value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{ir::SsaVarId, testing};
    ///
    /// let ssa = testing::const_i32_return(42);
    /// let var_id = SsaVarId::from_index(0);
    /// let constants = ssa.find_constants();
    /// assert!(constants.contains_key(&var_id));
    /// ```
    #[must_use]
    pub fn find_constants(&self) -> BTreeMap<SsaVarId, ConstValue<T>> {
        let mut constants = BTreeMap::new();

        for block in self.blocks() {
            for instr in block.instructions() {
                if let SsaOp::Const { dest, value } = instr.op() {
                    constants.insert(*dest, value.clone());
                }
            }
        }

        constants
    }

    /// Finds all blocks that use a given variable.
    ///
    /// Scans instructions and phi nodes across all blocks to find blocks
    /// that reference the specified variable.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to search for.
    /// * `exclude_block` - Optional block to exclude from results.
    ///
    /// # Returns
    ///
    /// A vector of block indices where the variable is used.
    #[must_use]
    pub fn find_var_user_blocks(&self, var: SsaVarId, exclude_block: Option<usize>) -> Vec<usize> {
        self.iter_blocks()
            .filter(|&(block_idx, _)| exclude_block != Some(block_idx))
            .filter(|(_, block)| {
                // Check instructions
                block.instructions().iter().any(|instr| {
                    let mut found = false;
                    instr.for_each_use(|used| found |= used == var);
                    found
                })
                    // Check phi operands
                    || block.phi_nodes().iter().any(|phi| {
                        phi.operands().iter().any(|op| op.value() == var)
                    })
            })
            .map(|(block_idx, _)| block_idx)
            .collect()
    }

    /// Classifies what every `Return` in the function yields.
    ///
    /// Collects the value of each `Return` instruction and looks for a property
    /// they all share, resolving each returned variable through its definition.
    /// Returns carrying no value (or the absence of any `Return`) make the
    /// function void; a single constant shared by every return - null included,
    /// since null is a `Const` - is reported as that constant; a lone return of
    /// a parameter is a pass-through; returns computed purely from parameters
    /// are foldable; anything else is dynamic. The classification drives
    /// inlining and inter-procedural constant folding at call sites.
    ///
    /// # Returns
    ///
    /// The [`ReturnInfo`] classification of the function's return behavior.
    #[must_use]
    pub fn return_info(&self) -> ReturnInfo<T> {
        let mut return_values: Vec<Option<SsaVarId>> = Vec::new();

        for block in self.blocks() {
            for instr in block.instructions() {
                if let SsaOp::Return { value, .. } = instr.op() {
                    return_values.push(*value);
                }
            }
        }

        // If no returns found, assume void
        if return_values.is_empty() {
            return ReturnInfo::Void;
        }

        // Check if all returns are void (None)
        if return_values.iter().all(Option::is_none) {
            return ReturnInfo::Void;
        }

        // If there's only one return with a value, trace what it is
        let non_void_returns: Vec<_> = return_values.iter().filter_map(|v| *v).collect();

        if non_void_returns.is_empty() {
            return ReturnInfo::Void;
        }

        // Try to determine what all returns have in common
        // Check if they all return the same constant
        let mut constants_found: Vec<Option<ConstValue<T>>> = Vec::new();

        for &ret_var in &non_void_returns {
            // Find the definition of this variable
            let def = self.get_definition(ret_var);

            match def {
                Some(SsaOp::Const { value, .. }) => {
                    // Const includes null values (ConstValue::Null)
                    constants_found.push(Some(value.clone()));
                }
                _ => {
                    constants_found.push(None);
                }
            }
        }

        // If all returns are the same constant
        if constants_found.iter().all(Option::is_some)
            && let Some(first) = constants_found.first()
            && constants_found.iter().all(|c| c == first)
            && let Some(const_val) = first
        {
            return ReturnInfo::Constant(const_val.clone());
        }

        // Check if returns a specific parameter (pass-through)
        for &ret_var in &non_void_returns {
            if let Some(param_idx) = self.is_parameter_variable(ret_var)
                && non_void_returns.len() == 1
            {
                return ReturnInfo::PassThrough(param_idx);
            }
        }

        // Check if all returns come from pure computations
        let all_pure = non_void_returns.iter().all(|&var| {
            if let Some(def) = self.get_definition(var) {
                def.is_pure()
            } else {
                false
            }
        });

        if all_pure {
            return ReturnInfo::PureComputation;
        }

        // Returns depend on state or have complex control flow
        ReturnInfo::Dynamic
    }

    /// Classifies the method's observable side effects, with what each call
    /// does answered by `callee`.
    ///
    /// Makes a single pass over every instruction, noting whether it writes
    /// external state, reads it, takes an address (which could enable an
    /// untracked write), calls an opaque method, throws, or may raise an
    /// exception. The findings are then collapsed by a fixed precedence - any
    /// definite write or throw is impure; an opaque call or an address-of
    /// leaves purity unknown; a method an instruction of which may raise is
    /// may-throw; a method that only reads is read-only; one that does none of
    /// these is pure - so the strongest effect present decides the result.
    ///
    /// A call to a method of known purity has that method's effects; every
    /// other call is opaque. A virtual call may also raise on its own, before
    /// the callee runs, since its receiver may be null.
    ///
    /// # Arguments
    ///
    /// * `callee` - The purity of the method a call, virtual call, indirect
    ///   call or `newobj` runs, or `None` when it is not known; `|_| None`
    ///   knows no callee.
    ///
    /// # Returns
    ///
    /// The [`MethodPurity`] level - `Impure`, `Unknown`, `MayThrow`,
    /// `ReadOnly`, or `Pure`, in decreasing order of side effects.
    #[must_use]
    pub fn purity(&self, callee: impl Fn(&SsaOp<T>) -> Option<MethodPurity>) -> MethodPurity {
        let mut has_writes = false;
        let mut has_reads = false;
        let mut has_unknown_calls = false;
        let mut has_indirect_access = false;
        let mut has_throws = false;
        let mut may_raise = false;

        for block in self.blocks() {
            for instr in block.instructions() {
                let op = instr.op();
                let is_call = matches!(op, SsaOp::Call(_) | SsaOp::NewObj { .. });
                if !is_call {
                    may_raise |= op.may_throw();
                }
                match op {
                    // Definite writes - impure
                    SsaOp::StoreField { .. }
                    | SsaOp::StoreStaticField { .. }
                    | SsaOp::StoreElement { .. }
                    | SsaOp::StoreIndirect { .. }
                    | SsaOp::InitObj { .. }
                    | SsaOp::InitBlk { .. }
                    | SsaOp::CopyBlk { .. }
                    | SsaOp::MoveBlk { .. } => {
                        has_writes = true;
                    }

                    // Reads from external state
                    SsaOp::LoadField { .. }
                    | SsaOp::LoadStaticField { .. }
                    | SsaOp::LoadElement { .. }
                    | SsaOp::LoadIndirect { .. }
                    | SsaOp::LoadObj { .. } => {
                        has_reads = true;
                    }

                    // Address-of operations might lead to indirect access
                    SsaOp::LoadFieldAddr { .. }
                    | SsaOp::LoadStaticFieldAddr { .. }
                    | SsaOp::LoadElementAddr { .. } => {
                        has_indirect_access = true;
                    }

                    // Calls do what their callee does, when that is known
                    SsaOp::Call(_) | SsaOp::NewObj { .. } => {
                        // A call through a receiver checks it is there.
                        if op.as_call().is_some_and(|call| {
                            matches!(call.callee, Callee::Virtual(_) | Callee::Constrained { .. })
                        }) {
                            may_raise = true;
                        }
                        match callee(op) {
                            None | Some(MethodPurity::Unknown) => has_unknown_calls = true,
                            Some(MethodPurity::Pure) => {}
                            Some(MethodPurity::ReadOnly) => has_reads = true,
                            Some(MethodPurity::MayThrow) => may_raise = true,
                            Some(MethodPurity::LocalMutation | MethodPurity::Impure) => {
                                has_writes = true;
                            }
                        }
                    }

                    // Throws and interrupt return are side effects (control flow)
                    SsaOp::Throw { .. }
                    | SsaOp::Rethrow
                    | SsaOp::InterruptReturn
                    | SsaOp::EndFinally
                    | SsaOp::EndFilter { .. } => {
                        has_throws = true;
                    }

                    // Everything else is either pure or doesn't affect state
                    _ => {}
                }
            }
        }

        // Determine purity level based on what we found
        if has_writes {
            return MethodPurity::Impure;
        }

        if has_unknown_calls {
            // Calls to unknown methods could be impure
            return MethodPurity::Unknown;
        }

        if has_throws {
            // Throwing exceptions is a side effect (abnormal control flow)
            return MethodPurity::Impure;
        }

        if has_indirect_access {
            // Address-of operations could enable writes we can't track
            return MethodPurity::Unknown;
        }

        if may_raise {
            // Deleting a call to it would delete the exception
            return MethodPurity::MayThrow;
        }

        if has_reads {
            return MethodPurity::ReadOnly;
        }

        MethodPurity::Pure
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::{
            CallData, Callee,
            block::SsaBlock,
            function::{MethodPurity, SsaFunction},
            instruction::SsaInstruction,
            ops::SsaOp,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// A phi-defined variable has no defining instruction, so both lookups must
    /// answer `None` - and must do so without walking the function.
    ///
    /// Before the short-circuit these fell through to a whole-function scan that
    /// could only ever return `None`, and callers such as the memory pass run
    /// that per value in a loop.
    #[test]
    fn phi_defined_variable_resolves_without_scanning() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::phi(1), MockType::I32);

        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::new((), SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);

        let mut merge = SsaBlock::new(1);
        merge.add_phi(PhiNode::new(phi_var, VariableOrigin::Local(0)));
        merge.add_instruction(SsaInstruction::new((), SsaOp::returns(None)));
        ssa.add_block(merge);

        assert!(ssa.get_definition(phi_var).is_none());
        assert!(ssa.get_definition_instruction(phi_var).is_none());
    }

    /// A variable whose recorded def site is stale must still be found by the
    /// recovery scan.
    ///
    /// The short-circuit above keys on a *confirmed* phi, not merely on an
    /// absent instruction index - an absent index also describes a stale def
    /// site, and skipping the scan for those would silently lose definitions.
    #[test]
    fn stale_def_site_still_recovers_by_scanning() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        // Claims a phi definition in block 0, but is actually defined by a real
        // instruction there and block 0 has no phi at all.
        let var = ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::phi(0), MockType::I32);

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::new(
            (),
            SsaOp::Const {
                dest: var,
                value: ConstValue::I32(5),
            },
        ));
        block.add_instruction(SsaInstruction::new((), SsaOp::returns(None)));
        ssa.add_block(block);

        assert!(
            ssa.get_definition(var).is_some(),
            "stale def site must fall through to the recovery scan"
        );
        assert!(ssa.get_definition_instruction(var).is_some());
    }

    /// Builds a block whose two `Const` instructions define `first` and
    /// `second`, with `second`'s def site left stale by one - naming `first`'s
    /// instruction.
    fn stale_by_one() -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let first = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let second = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::new(
            (),
            SsaOp::Const {
                dest: first,
                value: ConstValue::I32(11),
            },
        ));
        block.add_instruction(SsaInstruction::new(
            (),
            SsaOp::Const {
                dest: second,
                value: ConstValue::I32(22),
            },
        ));
        block.add_instruction(SsaInstruction::new((), SsaOp::returns(None)));
        ssa.add_block(block);
        (ssa, first, second)
    }

    /// The guard is the whole point of the handle: a site stale by one index
    /// names a real instruction defining a *different* variable, and reporting
    /// it would hand the caller a foreign operation as this variable's
    /// definition.
    #[test]
    fn recorded_definition_refuses_a_stale_def_site() {
        let (ssa, first, second) = stale_by_one();

        assert!(
            ssa.recorded_definition(first).is_some(),
            "an exact site resolves"
        );
        assert!(
            ssa.recorded_definition(second).is_none(),
            "a site naming another variable's instruction must be refused"
        );
    }

    /// The scan-free handle refuses; the recovering lookup built on it must
    /// still find the definition, and find the *right* one.
    #[test]
    fn get_definition_still_recovers_a_stale_def_site_by_scan() {
        let (ssa, _first, second) = stale_by_one();

        let recovered = ssa
            .get_definition(second)
            .expect("the scan recovers a stale def site");
        assert!(
            matches!(recovered, SsaOp::Const { dest, .. } if *dest == second),
            "the scan must find the instruction that really defines the variable: {recovered:?}"
        );
        let instruction = ssa
            .get_definition_instruction(second)
            .expect("the scan recovers a stale def site");
        assert!(matches!(instruction.op(), SsaOp::Const { dest, .. } if *dest == second));
    }

    /// `try_constant_value` is scan-free by contract, so it inherits the guard
    /// and nothing else. Before the guard, a site stale by one reported the
    /// neighbouring `Const`'s value as this variable's.
    #[test]
    fn try_constant_value_refuses_a_stale_def_site() {
        let (ssa, first, second) = stale_by_one();

        assert_eq!(ssa.try_constant_value(first), Some(ConstValue::I32(11)));
        assert_eq!(
            ssa.try_constant_value(second),
            None,
            "an unknown stride is the safe answer; 11 is the wrong one"
        );
    }

    /// An unknown variable is absent from both lookups.
    #[test]
    fn unknown_variable_has_no_definition() {
        let ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        assert!(ssa.get_definition(SsaVarId::from_index(7)).is_none());
        assert!(
            ssa.get_definition_instruction(SsaVarId::from_index(7))
                .is_none()
        );
    }

    /// A variable no instruction defines has no definition to find, and that
    /// answer must not outlive an instruction that later defines it.
    #[test]
    fn a_definition_added_after_a_miss_is_found() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let late =
            ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::new((), SsaOp::returns(None)));
        ssa.add_block(block);

        assert!(ssa.get_definition(late).is_none());
        assert!(ssa.get_definition_instruction(late).is_none());

        // The site still says "entry": only a scan can find this definition.
        let Some(block) = ssa.block_mut(0) else {
            panic!("block 0 exists");
        };
        block.instructions_mut().insert(
            0,
            SsaInstruction::new(
                (),
                SsaOp::Const {
                    dest: late,
                    value: ConstValue::I32(5),
                },
            ),
        );

        assert!(matches!(
            ssa.get_definition(late),
            Some(SsaOp::Const { dest, .. }) if *dest == late
        ));
        assert!(ssa.get_definition_instruction(late).is_some());
    }

    /// `return a op b` over two arguments.
    fn binary(
        make: impl FnOnce(SsaVarId, SsaVarId, SsaVarId) -> SsaOp<MockTarget>,
    ) -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 0);
        let a = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let b = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let result = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::new((), make(result, a, b)));
        block.add_instruction(SsaInstruction::new((), SsaOp::returns(Some(result))));
        ssa.add_block(block);
        ssa
    }

    /// A division may raise: the method can be inlined, which keeps the
    /// exception, but a call to it cannot be deleted, which would lose it.
    #[test]
    fn a_method_that_may_raise_is_not_eliminable() {
        let divides = binary(|dest, left, right| SsaOp::Div {
            dest,
            left,
            right,
            unsigned: false,
            flags: None,
        });
        let purity = divides.purity(|_| None);
        assert_eq!(purity, MethodPurity::MayThrow);
        assert!(!purity.can_eliminate_if_unused());
        assert!(!purity.can_reorder());
        assert!(purity.can_inline());

        let adds = binary(|dest, left, right| SsaOp::Add {
            dest,
            left,
            right,
            flags: None,
        });
        assert_eq!(adds.purity(|_| None), MethodPurity::Pure);
    }

    /// An operation of the traced function, given the variables made so far and
    /// its destination.
    type TracedOp<'o> = &'o dyn Fn(&[SsaVarId], SsaVarId) -> SsaOp<MockTarget>;

    /// A function of two parameters whose block 1 joins `phis` phis and then
    /// runs `body`, each operation given the variables made so far; returns
    /// the function and every variable, the parameters first, then the phis,
    /// then one per operation.
    fn traced(phis: usize, body: &[TracedOp<'_>]) -> (SsaFunction<MockTarget>, Vec<SsaVarId>) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, phis + body.len());
        let mut vars = Vec::new();
        for index in 0..2u16 {
            vars.push(ssa.create_variable(
                VariableOrigin::Argument(index),
                0,
                DefSite::entry(),
                MockType::I32,
            ));
        }
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::new((), SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);
        let mut block = SsaBlock::new(1);
        for index in 0..phis {
            let origin = VariableOrigin::Local(u16::try_from(index).unwrap());
            let phi = ssa.create_variable(origin, 0, DefSite::phi(1), MockType::I32);
            block.add_phi(PhiNode::new(phi, origin));
            vars.push(phi);
        }
        for (index, op) in body.iter().enumerate() {
            let origin = VariableOrigin::Local(u16::try_from(phis + index).unwrap());
            let dest =
                ssa.create_variable(origin, 0, DefSite::instruction(1, index), MockType::I32);
            block.add_instruction(SsaInstruction::new((), op(&vars, dest)));
            vars.push(dest);
        }
        block.add_instruction(SsaInstruction::new((), SsaOp::returns(None)));
        ssa.add_block(block);
        (ssa, vars)
    }

    /// **A trace takes the left operand first and passes through what it
    /// reads.** `(p ^ q) + 7` reaches `p`, `(q ^ p)` reaches `q`, a copy of
    /// either reaches it, and a constant reaches nothing.
    #[test]
    fn a_trace_reaches_the_first_phi_left_to_right() {
        let xor = |left: usize, right: usize| {
            move |vars: &[SsaVarId], dest| SsaOp::Xor {
                dest,
                left: vars[left],
                right: vars[right],
                flags: None,
            }
        };
        let seven = |_: &[SsaVarId], dest| SsaOp::Const {
            dest,
            value: ConstValue::I32(7),
        };
        let pq = xor(2, 3);
        let qp = xor(3, 2);
        let sum = |vars: &[SsaVarId], dest| SsaOp::Add {
            dest,
            left: vars[4],
            right: vars[6],
            flags: None,
        };
        let copy = |vars: &[SsaVarId], dest| SsaOp::Copy { dest, src: vars[5] };
        let (ssa, vars) = traced(2, &[&pq, &qp, &seven, &sum, &copy]);
        let (p, q) = (vars[2], vars[3]);
        assert_eq!(ssa.trace_to_phi(vars[7]), Some(p));
        assert_eq!(ssa.trace_to_phi(vars[5]), Some(q));
        assert_eq!(ssa.trace_to_phi(vars[8]), Some(q));
        assert_eq!(ssa.trace_to_phi(vars[6]), None);
    }

    /// **A trace stops twenty definitions deep.** A phi under twenty copies
    /// is reached; one under twenty-one is not.
    #[test]
    fn a_trace_stops_twenty_definitions_deep() {
        let copy = |vars: &[SsaVarId], dest| SsaOp::Copy {
            dest,
            src: *vars.last().unwrap(),
        };
        let chain: Vec<TracedOp<'_>> = vec![&copy; 21];
        let (ssa, vars) = traced(1, &chain);
        let phi = vars[2];
        assert_eq!(ssa.trace_to_phi(vars[2 + 20]), Some(phi));
        assert_eq!(ssa.trace_to_phi(vars[2 + 21]), None);
    }

    /// **A parameter is itself or what a chain of copies copies.** A copy of a
    /// copy of the second parameter is the second parameter; a constant and a
    /// copy of a phi are none.
    #[test]
    fn a_parameter_is_reached_through_copies_only() {
        let copy_of = |index: usize| {
            move |vars: &[SsaVarId], dest| SsaOp::Copy {
                dest,
                src: vars[index],
            }
        };
        let seven = |_: &[SsaVarId], dest| SsaOp::Const {
            dest,
            value: ConstValue::I32(7),
        };
        let (first, again, of_phi) = (copy_of(1), copy_of(3), copy_of(2));
        let (ssa, vars) = traced(1, &[&first, &again, &seven, &of_phi]);
        assert_eq!(ssa.is_parameter_variable(vars[0]), Some(0));
        assert_eq!(ssa.is_parameter_variable(vars[4]), Some(1));
        assert_eq!(ssa.is_parameter_variable(vars[5]), None);
        assert_eq!(ssa.is_parameter_variable(vars[6]), None);
    }

    /// **What a call enters is every value its pointer may be.** Two loaded
    /// pointers merge at a phi; the phi and a third pointer are selected by a
    /// condition, copied, and called through. Every pointer is entered on the
    /// path that yields it, the condition is not, and a value the function
    /// only returns is not.
    #[test]
    fn a_call_enters_every_value_its_pointer_merges_or_selects() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
        let local = |ssa: &mut SsaFunction<MockTarget>, index: u16, site: DefSite| {
            ssa.create_variable(VariableOrigin::Local(index), 0, site, MockType::U64)
        };
        let left = local(&mut ssa, 0, DefSite::instruction(0, 0));
        let right = local(&mut ssa, 1, DefSite::instruction(0, 1));
        let other = local(&mut ssa, 2, DefSite::instruction(0, 2));
        let condition = local(&mut ssa, 3, DefSite::instruction(0, 3));
        let returned = local(&mut ssa, 4, DefSite::instruction(0, 4));
        let merged = local(&mut ssa, 5, DefSite::phi(1));
        let selected = local(&mut ssa, 6, DefSite::instruction(1, 0));
        let copied = local(&mut ssa, 7, DefSite::instruction(1, 1));
        let constant = |dest, value| SsaInstruction::new((), SsaOp::Const { dest, value });
        let mut entry = SsaBlock::new(0);
        for (index, dest) in [left, right, other, condition, returned]
            .into_iter()
            .enumerate()
        {
            entry.add_instruction(constant(
                dest,
                ConstValue::I64(i64::try_from(index).unwrap()),
            ));
        }
        entry.add_instruction(SsaInstruction::new((), SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);
        let mut call = SsaBlock::new(1);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(5));
        phi.add_operand(PhiOperand::new(left, 0));
        phi.add_operand(PhiOperand::new(right, 1));
        call.add_phi(phi);
        call.add_instruction(SsaInstruction::new(
            (),
            SsaOp::Select {
                dest: selected,
                condition,
                true_val: merged,
                false_val: other,
            },
        ));
        call.add_instruction(SsaInstruction::new(
            (),
            SsaOp::Copy {
                dest: copied,
                src: selected,
            },
        ));
        call.add_instruction(SsaInstruction::new(
            (),
            SsaOp::from(CallData::new(
                None,
                Callee::Indirect {
                    fptr: copied,
                    signature: 0,
                },
                Vec::new(),
            )),
        ));
        call.add_instruction(SsaInstruction::new((), SsaOp::returns(Some(returned))));
        ssa.add_block(call);
        let entered = ssa.entered_values();
        for value in [left, right, other, merged, selected, copied] {
            assert!(entered.contains(&value), "{value:?} is entered");
        }
        assert!(
            !entered.contains(&condition),
            "a select's condition is not entered"
        );
        assert!(
            !entered.contains(&returned),
            "a returned value is not entered"
        );
    }
}
