//! Def-Use Index for efficient SSA variable lookup.
//!
//! This module provides [`DefUseIndex`], a shared index structure that enables
//! efficient queries about variable definitions and uses across an SSA function.
//!
//! # Purpose
//!
//! While each `SsaVariable` tracks its own definition site and use sites, this
//! index provides additional views for O(1) or O(k) access:
//!
//! - **Definitions by location**: What variables are defined in block B at instruction I?
//! - **Uses by location**: What variables are used in block B at instruction I?
//! - **All definitions in block**: All variables defined in block B
//! - **Unused variables**: Variables with no uses (candidates for elimination)
//! - **Defining operations**: What operation defines variable V? (with `build_with_ops`)
//!
//! # Algorithm
//!
//! Building the index is an O(n) scan where n is the total number of
//! instructions and phi nodes:
//!
//! 1. Seed all registered `SsaVariable` entries so entry and unused variables
//!    remain queryable.
//! 2. Count uses per variable and per location, prefix-sum those counts into
//!    offsets, then scan again to fill the flattened lists (see below).
//! 3. Scan active phi nodes and instructions to collect definitions from the
//!    actual IR.
//! 4. Identify phi-defined variables and unused variables.
//! 5. Optionally collect defining operations while scanning instructions
//!    (needed for `build_with_ops`).
//!
//! # Data Structures
//!
//! The per-variable and per-location lists are **flattened** (CSR: an offsets
//! array plus one concatenated values array) rather than a vector per variable
//! or a `BTreeMap<Location, Vec<_>>`. This index is rebuilt from scratch by
//! nearly every pass, and the nested form allocated once per variable and once
//! per instruction on every rebuild — measured as the largest single source of
//! allocations in a native lift. The flat form costs one extra counting walk and
//! a fixed handful of buffers, and keeps every query O(1)/O(k) on a slice.
//!
//! - `Vec<Option<DefSite>>`, dense by `SsaVarId::index()`: O(1) definition lookup
//! - Flattened per-variable use lists: O(1) to reach a variable's use slice
//! - Flattened per-location def/use lists, block-major: O(1) per location query
//! - `BitSet`: O(1) for phi-defined and unused queries
//!
//! # Basic Usage
//!
//! ```rust
//! use analyssa::{analysis::DefUseIndex, ir::variable::SsaVarId, testing};
//!
//! let ssa = testing::const_i32_return(42);
//! let index = DefUseIndex::build(&ssa);
//! let v0 = SsaVarId::from_index(0);
//!
//! // Find all uses of variable v0
//! if let Some(uses) = index.uses_of(v0) {
//!     for use_site in uses {
//!         println!("v0 used at block {}, instr {}", use_site.block, use_site.instruction);
//!     }
//! }
//!
//! // Find variables defined at a specific instruction
//! for var_id in index.defs_at(0, 0) {
//!     println!("Variable {} defined here", var_id);
//! }
//!
//! // Check if a variable is dead (unused)
//! if index.is_unused(v0) {
//!     println!("Variable {} can be eliminated", v0);
//! }
//! ```
//!
//! # Building with Operations
//!
//! For passes that need to analyze the defining operation (e.g., constant folding,
//! pattern matching), use [`DefUseIndex::build_with_ops`]:
//!
//! ```rust
//! use analyssa::{analysis::DefUseIndex, ir::{ops::SsaOp, variable::SsaVarId}, testing};
//!
//! let ssa = testing::const_i32_return(42);
//! let index = DefUseIndex::build_with_ops(&ssa);
//! let var_id = SsaVarId::from_index(0);
//!
//! // Get the defining operation for a variable
//! if let Some(op) = index.def_op(var_id) {
//!     match op {
//!         SsaOp::Add { left, right, .. } => { /* analyze operands */ }
//!         SsaOp::Const { value, .. } => { /* it's a constant */ }
//!         _ => {}
//!     }
//! }
//!
//! // Or get everything at once: (block, instruction, operation)
//! if let Some((block, instr, op)) = index.full_definition(var_id) {
//!     println!("Defined at B{}:{} by {:?}", block, instr, op);
//! }
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    bitset::BitSet,
    ir::{
        function::SsaFunction,
        ops::SsaOp,
        variable::{DefSite, SsaVarId, UseSite},
    },
    target::Target,
};

/// Location in the SSA function (block + instruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Location {
    /// Block index.
    pub block: usize,
    /// Instruction index within the block.
    pub instruction: usize,
}

impl Location {
    /// Creates a new location.
    #[must_use]
    pub const fn new(block: usize, instruction: usize) -> Self {
        Self { block, instruction }
    }
}

/// Index for efficient def-use queries on an SSA function.
///
/// This structure is built once from an `SsaFunction` and provides O(1) or O(k)
/// access to various def-use relationships (where k is the result size).
///
/// # Building with Operations
///
/// Use [`build_with_ops`](Self::build_with_ops) to also index the defining operations,
/// enabling efficient lookups via [`def_op`](Self::def_op) and
/// [`full_definition`](Self::full_definition).
///
/// ```rust
/// use analyssa::{analysis::DefUseIndex, ir::variable::SsaVarId, testing};
///
/// let ssa = testing::const_i32_return(42);
/// let index = DefUseIndex::build_with_ops(&ssa);
/// let var_id = SsaVarId::from_index(0);
///
/// // Get block, instruction, and operation in one call
/// if let Some((block, instr, op)) = index.full_definition(var_id) {
///     println!("Defined at B{}:{} by {:?}", block, instr, op);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct DefUseIndex<T: Target> {
    /// Definition site for each variable, indexed densely by `SsaVarId::index()`.
    /// `None` marks an index outside the registered variable range.
    definitions: Vec<Option<DefSite>>,

    /// Start offset into [`Self::use_values`] for each variable's use list,
    /// indexed densely by `SsaVarId::index()`. Length is `capacity + 1`, so
    /// variable `i` owns `use_values[use_offsets[i]..use_offsets[i + 1]]`.
    ///
    /// Flattened (CSR) rather than a `Vec<Vec<UseSite>>` because this index is
    /// rebuilt from scratch by nearly every pass, and a vector-per-variable
    /// allocates once for every variable that has any use — measured as the
    /// single largest source of allocations in the whole lift. The flattened
    /// form costs one extra counting walk and a fixed two allocations.
    use_offsets: Vec<u32>,

    /// Concatenated per-variable use lists; see [`Self::use_offsets`].
    use_values: Vec<UseSite>,

    /// Row base for each block: block `b` owns rows
    /// `block_row_base[b]..block_row_base[b + 1]`, and row `block_row_base[b] + i`
    /// is `Location(b, i)`. Length is `block_count + 1`.
    ///
    /// A block owns `max(phi_nodes, instructions)` rows because phi-operand uses
    /// and instruction uses **share** a row index — preserving the behaviour of
    /// the `Location`-keyed maps this replaces, where `Location(b, 0)` held both
    /// phi 0's operands and instruction 0's uses.
    block_row_base: Vec<u32>,

    /// Variables defined at each row, flattened: row `r` owns
    /// `loc_defs_values[loc_defs_offsets[r]..loc_defs_offsets[r + 1]]`.
    ///
    /// Flattened for the same reason as [`Self::use_offsets`]: the previous
    /// `BTreeMap<Location, Vec<SsaVarId>>` cost a tree node *and* a vector for
    /// every instruction, on an index that is rebuilt by nearly every pass.
    loc_defs_offsets: Vec<u32>,
    /// Concatenated per-row definition lists; see [`Self::loc_defs_offsets`].
    loc_defs_values: Vec<SsaVarId>,

    /// Variables used at each row; see [`Self::loc_defs_offsets`] for the layout.
    loc_uses_offsets: Vec<u32>,
    /// Concatenated per-row use lists; see [`Self::loc_uses_offsets`].
    loc_uses_values: Vec<SsaVarId>,

    /// Variables defined in each block (including phi nodes), indexed by block.
    defs_in_block: Vec<Vec<SsaVarId>>,

    /// Variables defined by phi nodes.
    phi_defs: BitSet,

    /// Variables with no uses (dead variables).
    unused_vars: BitSet,

    /// Total variable count.
    var_count: usize,

    /// Optional: defining operations for each variable, indexed densely by
    /// `SsaVarId::index()`. Populated when built with
    /// [`build_with_ops`](Self::build_with_ops).
    def_ops: Option<Vec<Option<SsaOp<T>>>>,
}

impl<T: Target> Default for DefUseIndex<T> {
    fn default() -> Self {
        Self {
            definitions: Vec::new(),
            use_offsets: Vec::new(),
            use_values: Vec::new(),
            block_row_base: Vec::new(),
            loc_defs_offsets: Vec::new(),
            loc_defs_values: Vec::new(),
            loc_uses_offsets: Vec::new(),
            loc_uses_values: Vec::new(),
            defs_in_block: Vec::new(),
            phi_defs: BitSet::default(),
            unused_vars: BitSet::default(),
            var_count: 0,
            def_ops: None,
        }
    }
}

/// Appends `var` to `row`'s region of a CSR pair, advancing that row's cursor.
///
/// The cursor array starts as a copy of the offsets, so each row fills forward
/// from its own start; a row can never overrun because the offsets were sized by
/// a counting walk over the same data.
fn csr_push(cursors: &mut [u32], values: &mut [SsaVarId], row: usize, var: SsaVarId) {
    if let Some(cursor) = cursors.get_mut(row) {
        if let Some(slot) = values.get_mut(*cursor as usize) {
            *slot = var;
        }
        *cursor = cursor.saturating_add(1);
    }
}

/// Visits every `(variable, use site)` pair in the function, in the order the
/// def-use index records them: for each block, phi operands first, then
/// instruction operands.
///
/// Factored out so the counting walk and the filling walk that build the
/// flattened use lists cannot drift apart — they must agree exactly, or a
/// variable's uses land at the wrong offsets.
fn visit_uses<T: Target>(ssa: &SsaFunction<T>, mut visit: impl FnMut(SsaVarId, UseSite)) {
    for (block_idx, block) in ssa.blocks().iter().enumerate() {
        for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
            for operand in phi.operands() {
                visit(operand.value(), UseSite::phi_operand(block_idx, phi_idx));
            }
        }
        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            instr
                .op()
                .for_each_use(|var| visit(var, UseSite::instruction(block_idx, instr_idx)));
        }
    }
}

impl<T: Target> DefUseIndex<T> {
    /// Returns the recorded use sites for a dense variable index, or `None`
    /// when the index falls outside the registered variable range.
    fn uses_slice(&self, index: usize) -> Option<&[UseSite]> {
        let start = *self.use_offsets.get(index)? as usize;
        let end = *self.use_offsets.get(index.checked_add(1)?)? as usize;
        self.use_values.get(start..end)
    }

    /// Returns the flat row index for `Location(block, index)`, or `None` when
    /// the block does not exist or the index is past the block's row count.
    fn row_of(&self, block: usize, index: usize) -> Option<usize> {
        let base = *self.block_row_base.get(block)? as usize;
        let end = *self.block_row_base.get(block.checked_add(1)?)? as usize;
        let row = base.checked_add(index)?;
        (row < end).then_some(row)
    }

    /// Returns one row's slice out of a CSR pair, or an empty slice when the row
    /// does not exist.
    fn csr_row<'s>(offsets: &[u32], values: &'s [SsaVarId], row: usize) -> &'s [SsaVarId] {
        let Some(start) = offsets.get(row).map(|offset| *offset as usize) else {
            return &[];
        };
        let Some(end) = row
            .checked_add(1)
            .and_then(|next| offsets.get(next))
            .map(|offset| *offset as usize)
        else {
            return &[];
        };
        values.get(start..end).unwrap_or(&[])
    }

    /// Builds a def-use index from an SSA function.
    ///
    /// This is an O(n) operation where n is the total number of instructions
    /// and phi nodes in the function.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to index.
    ///
    /// # Returns
    ///
    /// A new `DefUseIndex` with all relationships computed.
    #[must_use]
    pub fn build(ssa: &SsaFunction<T>) -> Self {
        // Row layout for the location-indexed lists: block-major, with
        // `max(phis, instructions)` rows per block so a phi index and an
        // instruction index share a row exactly as the `Location` key did.
        let mut block_row_base: Vec<u32> = Vec::with_capacity(ssa.blocks().len().saturating_add(1));
        let mut row_total: u32 = 0;
        block_row_base.push(0);
        for block in ssa.blocks() {
            let rows = block.phi_nodes().len().max(block.instructions().len());
            row_total = row_total.saturating_add(u32::try_from(rows).unwrap_or(u32::MAX));
            block_row_base.push(row_total);
        }
        let row_count = row_total as usize;

        // Count defs/uses per row (staged one slot high, as for `use_offsets`),
        // prefix-sum into offsets, then fill through per-row cursors.
        let mut loc_defs_offsets: Vec<u32> = vec![0; row_count.saturating_add(1)];
        let mut loc_uses_offsets: Vec<u32> = vec![0; row_count.saturating_add(1)];
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            let base = block_row_base.get(block_idx).map_or(0, |b| *b as usize);
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                if let Some(slot) = base
                    .checked_add(phi_idx)
                    .and_then(|row| row.checked_add(1))
                    .and_then(|next| loc_uses_offsets.get_mut(next))
                {
                    *slot = slot
                        .saturating_add(u32::try_from(phi.operands().len()).unwrap_or(u32::MAX));
                }
            }
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let Some(next) = base
                    .checked_add(instr_idx)
                    .and_then(|row| row.checked_add(1))
                else {
                    continue;
                };
                let op = instr.op();
                if let Some(slot) = loc_defs_offsets.get_mut(next) {
                    *slot =
                        slot.saturating_add(u32::try_from(op.defs().count()).unwrap_or(u32::MAX));
                }
                if let Some(slot) = loc_uses_offsets.get_mut(next) {
                    *slot = slot.saturating_add(u32::try_from(op.use_count()).unwrap_or(u32::MAX));
                }
            }
        }
        let mut running_defs: u32 = 0;
        for slot in &mut loc_defs_offsets {
            running_defs = running_defs.saturating_add(*slot);
            *slot = running_defs;
        }
        let mut running_uses: u32 = 0;
        for slot in &mut loc_uses_offsets {
            running_uses = running_uses.saturating_add(*slot);
            *slot = running_uses;
        }
        let mut loc_defs_values: Vec<SsaVarId> = vec![SsaVarId::PLACEHOLDER; running_defs as usize];
        let mut loc_uses_values: Vec<SsaVarId> = vec![SsaVarId::PLACEHOLDER; running_uses as usize];
        let mut defs_cursors: Vec<u32> = loc_defs_offsets.clone();
        let mut uses_cursors: Vec<u32> = loc_uses_offsets.clone();
        let variable_count = ssa.variable_count();
        let max_var_idx = ssa
            .variables()
            .iter()
            .map(|v| v.id().index().saturating_add(1))
            .max()
            .unwrap_or(0);
        let bitset_capacity = max_var_idx.max(variable_count);
        let block_count = ssa.blocks().len();

        // Variables are densely indexed by `SsaVarId::index()`, so back the
        // per-variable relationships with `Vec`s for O(1) cache-friendly lookup
        // instead of `BTreeMap`'s O(log n) probes.
        let mut definitions: Vec<Option<DefSite>> = vec![None; bitset_capacity];
        let mut defs_in_block: Vec<Vec<SsaVarId>> = vec![Vec::new(); block_count];
        let mut phi_defs = BitSet::new(bitset_capacity);

        // Flattened per-variable use lists: count first, prefix-sum into
        // offsets, then fill through per-variable cursors. Both walks go through
        // `visit_uses`, so the recorded order matches a per-variable `push`.
        let mut use_offsets: Vec<u32> = vec![0; bitset_capacity.saturating_add(1)];
        visit_uses(ssa, |var, _site| {
            // Counts are staged one slot high so the prefix sum below turns this
            // same vector into the offsets array without a second allocation.
            if let Some(slot) = var
                .index()
                .checked_add(1)
                .and_then(|next| use_offsets.get_mut(next))
            {
                *slot = slot.saturating_add(1);
            }
        });
        let mut running: u32 = 0;
        for slot in &mut use_offsets {
            running = running.saturating_add(*slot);
            *slot = running;
        }
        let mut use_values: Vec<UseSite> = vec![UseSite::instruction(0, 0); running as usize];
        // Cursors start at each variable's region start, i.e. the *previous*
        // entry of the offsets array.
        let mut cursors: Vec<u32> = use_offsets.clone();
        visit_uses(ssa, |var, site| {
            if let Some(cursor) = cursors.get_mut(var.index()) {
                if let Some(slot) = use_values.get_mut(*cursor as usize) {
                    *slot = site;
                }
                *cursor = cursor.saturating_add(1);
            }
        });

        // Helper: write a definition site at a variable's dense index.
        let set_def = |definitions: &mut Vec<Option<DefSite>>, var: SsaVarId, site: DefSite| {
            if let Some(slot) = definitions.get_mut(var.index()) {
                *slot = Some(site);
            }
        };
        // Seed every registered variable so entry values and unused variables
        // remain queryable even when they have no active instruction or phi
        // definition. Active definitions and uses are then scanned from the IR
        // itself so stale variable metadata cannot mislead optimization passes.
        for var in ssa.variables() {
            set_def(&mut definitions, var.id(), var.def_site());
        }

        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            let row_base = block_row_base.get(block_idx).map_or(0, |b| *b as usize);
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                let result = phi.result();
                set_def(&mut definitions, result, DefSite::phi(block_idx));
                // `insert_checked`: the capacity comes from the *registered*
                // variables, and malformed IR can name a phi result the
                // allocator never issued. `set_def` above is already tolerant of
                // that (it uses `get_mut`), so this matches — an unregistered
                // variable simply has no entry, which is the same answer every
                // query here gives for one.
                phi_defs.insert_checked(result.index());
                if let Some(slot) = defs_in_block.get_mut(block_idx) {
                    slot.push(result);
                }

                let Some(row) = row_base.checked_add(phi_idx) else {
                    continue;
                };
                for operand in phi.operands() {
                    csr_push(
                        &mut uses_cursors,
                        &mut loc_uses_values,
                        row,
                        operand.value(),
                    );
                }
            }

            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let Some(row) = row_base.checked_add(instr_idx) else {
                    continue;
                };
                let op = instr.op();
                for dest in op.defs() {
                    set_def(
                        &mut definitions,
                        dest,
                        DefSite::instruction(block_idx, instr_idx),
                    );
                    csr_push(&mut defs_cursors, &mut loc_defs_values, row, dest);
                    if let Some(slot) = defs_in_block.get_mut(block_idx) {
                        slot.push(dest);
                    }
                }
                op.for_each_use(|var| {
                    csr_push(&mut uses_cursors, &mut loc_uses_values, row, var);
                });
            }
        }

        // Identify unused variables. Only registered variables are considered,
        // matching the previous seeded-map behavior.
        let mut unused_vars = BitSet::new(bitset_capacity);
        for var in ssa.variables() {
            let i = var.id().index();
            if use_offsets
                .get(i)
                .zip(i.checked_add(1).and_then(|next| use_offsets.get(next)))
                .is_some_and(|(start, end)| start == end)
            {
                unused_vars.insert(i);
            }
        }

        Self {
            definitions,
            use_offsets,
            use_values,
            block_row_base,
            loc_defs_offsets,
            loc_defs_values,
            loc_uses_offsets,
            loc_uses_values,
            defs_in_block,
            phi_defs,
            unused_vars,
            var_count: variable_count,
            def_ops: None,
        }
    }

    /// Builds a def-use index with defining operations stored internally.
    ///
    /// This version indexes the defining operation for each variable, enabling
    /// efficient lookups via [`def_op`](Self::def_op) and
    /// [`full_definition`](Self::full_definition).
    ///
    /// Use this when passes need to analyze the defining operation alongside
    /// the definition site.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to index.
    ///
    /// # Returns
    ///
    /// A `DefUseIndex` with operations indexed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{analysis::DefUseIndex, ir::{SsaOp, SsaVarId}, testing};
    ///
    /// let ssa = testing::const_i32_return(1);
    /// let index = DefUseIndex::build_with_ops(&ssa);
    /// let var_id = SsaVarId::from_index(0);
    ///
    /// // Get the defining operation
    /// if let Some(op) = index.def_op(var_id) {
    ///     match op {
    ///         SsaOp::Add { left, right, .. } => { /* analyze add */ }
    ///         _ => {}
    ///     }
    /// }
    ///
    /// // Or get everything at once
    /// if let Some((block, instr, op)) = index.full_definition(var_id) {
    ///     println!("B{}:{} {:?}", block, instr, op);
    /// }
    /// ```
    #[must_use]
    pub fn build_with_ops(ssa: &SsaFunction<T>) -> Self {
        let mut index = Self::build(ssa);

        // Collect defining operations, densely indexed by variable.
        let mut def_ops: Vec<Option<SsaOp<T>>> = vec![None; index.definitions.len()];
        for (_block_idx, _instr_idx, instr) in ssa.iter_instructions() {
            let op = instr.op();
            for dest in op.defs() {
                if let Some(slot) = def_ops.get_mut(dest.index()) {
                    *slot = Some(op.clone());
                }
            }
        }
        index.def_ops = Some(def_ops);

        index
    }

    /// Builds a def-use index with operations, also returning a separate map.
    ///
    /// This is a compatibility method for code that needs both the index
    /// and a separate operation map. Prefer [`build_with_ops`](Self::build_with_ops)
    /// for new code.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to index.
    ///
    /// # Returns
    ///
    /// A tuple of (`DefUseIndex`, operation map).
    #[must_use]
    pub fn build_with_ops_map(ssa: &SsaFunction<T>) -> (Self, BTreeMap<SsaVarId, SsaOp<T>>) {
        let index = Self::build_with_ops(ssa);
        // Materialize the compatibility map once from the dense op table.
        let mut ops = BTreeMap::new();
        if let Some(def_ops) = index.def_ops.as_ref() {
            for (i, op) in def_ops.iter().enumerate() {
                if let Some(op) = op {
                    ops.insert(SsaVarId::from_index(i), op.clone());
                }
            }
        }
        (index, ops)
    }

    /// Returns whether this index has operation information.
    ///
    /// Returns `true` if built with [`build_with_ops`](Self::build_with_ops).
    #[must_use]
    pub fn has_ops(&self) -> bool {
        self.def_ops.is_some()
    }

    /// Returns the defining operation for a variable.
    ///
    /// This method requires the index to be built with
    /// [`build_with_ops`](Self::build_with_ops).
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to look up.
    ///
    /// # Returns
    ///
    /// The defining operation, or `None` if:
    /// - The variable is unknown
    /// - The variable is defined by a phi node (no operation)
    /// - The index was not built with operations
    #[must_use]
    pub fn def_op(&self, var: SsaVarId) -> Option<&SsaOp<T>> {
        self.def_ops.as_ref()?.get(var.index())?.as_ref()
    }

    /// Returns full definition information: block, instruction index, and operation.
    ///
    /// This is a convenience method for passes that need all three pieces of
    /// information together. Requires the index to be built with
    /// [`build_with_ops`](Self::build_with_ops).
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to look up.
    ///
    /// # Returns
    ///
    /// A tuple of `(block_index, instruction_index, operation)`, or `None` if:
    /// - The variable is unknown
    /// - The variable is defined by a phi node (no instruction index)
    /// - The index was not built with operations
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{analysis::DefUseIndex, ir::{SsaOp, SsaVarId}, testing};
    ///
    /// let ssa = testing::const_i32_return(1);
    /// let index = DefUseIndex::build_with_ops(&ssa);
    /// let var_id = SsaVarId::from_index(0);
    ///
    /// if let Some((block, instr, op)) = index.full_definition(var_id) {
    ///     // Check if this is an add of two constants
    ///     if let SsaOp::Add { left, right, .. } = op {
    ///         let left_const = index.def_op(*left);
    ///         let right_const = index.def_op(*right);
    ///         // ...
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn full_definition(&self, var: SsaVarId) -> Option<(usize, usize, &SsaOp<T>)> {
        let site = self.def_site(var)?;
        let instr = site.instruction?; // None for phi nodes
        let op = self.def_op(var)?;
        Some((site.block, instr, op))
    }

    /// Returns the definition site for a variable.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to look up.
    ///
    /// # Returns
    ///
    /// The definition site, or `None` if the variable is unknown.
    #[must_use]
    pub fn def_site(&self, var: SsaVarId) -> Option<DefSite> {
        self.definitions.get(var.index()).copied().flatten()
    }

    /// Returns all use sites for a variable.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to look up.
    ///
    /// # Returns
    ///
    /// A slice of use sites, or `None` if the variable is unknown.
    #[must_use]
    pub fn uses_of(&self, var: SsaVarId) -> Option<&[UseSite]> {
        self.uses_slice(var.index())
    }

    /// Returns the number of uses for a variable.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to count uses for.
    ///
    /// # Returns
    ///
    /// The use count, or 0 if the variable is unknown.
    #[must_use]
    pub fn use_count(&self, var: SsaVarId) -> usize {
        self.uses_slice(var.index()).map_or(0, <[UseSite]>::len)
    }

    /// Checks if a variable has any uses.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if the variable has at least one use.
    #[must_use]
    pub fn has_uses(&self, var: SsaVarId) -> bool {
        self.uses_slice(var.index()).is_some_and(|u| !u.is_empty())
    }

    /// Checks if a variable is unused (dead).
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if the variable has no uses.
    #[must_use]
    pub fn is_unused(&self, var: SsaVarId) -> bool {
        var.index() < self.unused_vars.len() && self.unused_vars.contains(var.index())
    }

    /// Checks if a variable is defined by a phi node.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if the variable is defined by a phi node.
    #[must_use]
    pub fn is_phi_def(&self, var: SsaVarId) -> bool {
        var.index() < self.phi_defs.len() && self.phi_defs.contains(var.index())
    }

    /// Returns variables defined at a specific location.
    ///
    /// # Arguments
    ///
    /// * `block` - The block index.
    /// * `instruction` - The instruction index within the block.
    ///
    /// # Returns
    ///
    /// A slice of variable IDs defined at that location.
    #[must_use]
    pub fn defs_at(&self, block: usize, instruction: usize) -> &[SsaVarId] {
        self.row_of(block, instruction).map_or(&[], |row| {
            Self::csr_row(&self.loc_defs_offsets, &self.loc_defs_values, row)
        })
    }

    /// Returns variables used at a specific location.
    ///
    /// # Arguments
    ///
    /// * `block` - The block index.
    /// * `instruction` - The instruction index within the block.
    ///
    /// # Returns
    ///
    /// A slice of variable IDs used at that location.
    #[must_use]
    pub fn uses_at(&self, block: usize, instruction: usize) -> &[SsaVarId] {
        self.row_of(block, instruction).map_or(&[], |row| {
            Self::csr_row(&self.loc_uses_offsets, &self.loc_uses_values, row)
        })
    }

    /// Returns all variables defined in a block.
    ///
    /// This includes both phi node definitions and instruction definitions.
    ///
    /// # Arguments
    ///
    /// * `block` - The block index.
    ///
    /// # Returns
    ///
    /// A slice of variable IDs defined in the block.
    #[must_use]
    pub fn defs_in_block(&self, block: usize) -> &[SsaVarId] {
        self.defs_in_block.get(block).map_or(&[], Vec::as_slice)
    }

    /// Returns all unused (dead) variables.
    ///
    /// These are candidates for dead code elimination.
    ///
    /// # Returns
    ///
    /// A reference to the set of unused variable IDs.
    #[must_use]
    pub fn unused_variables(&self) -> &BitSet {
        &self.unused_vars
    }

    /// Returns all phi-defined variables.
    ///
    /// # Returns
    ///
    /// A reference to the set of phi-defined variable IDs.
    #[must_use]
    pub fn phi_definitions(&self) -> &BitSet {
        &self.phi_defs
    }

    /// Returns the total number of variables indexed.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.var_count
    }

    /// Returns the number of unused variables.
    #[must_use]
    pub fn unused_count(&self) -> usize {
        self.unused_vars.count()
    }

    /// Checks if a variable has a single use.
    ///
    /// Single-use variables are good candidates for inlining.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if the variable has exactly one use.
    #[must_use]
    pub fn is_single_use(&self, var: SsaVarId) -> bool {
        self.use_count(var) == 1
    }

    /// Checks if a variable is only used in phi nodes.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if all uses are phi node operands.
    #[must_use]
    pub fn only_used_in_phis(&self, var: SsaVarId) -> bool {
        self.uses_slice(var.index())
            .is_some_and(|uses| !uses.is_empty() && uses.iter().all(|u| u.is_phi_operand))
    }

    /// Returns all variables used in a block.
    ///
    /// This is computed by scanning all use locations in the block.
    ///
    /// # Arguments
    ///
    /// * `block` - The block index.
    ///
    /// # Returns
    ///
    /// A set of variable IDs used anywhere in the block.
    #[must_use]
    pub fn uses_in_block(&self, block: usize) -> BTreeSet<SsaVarId> {
        let mut result = BTreeSet::new();
        let (Some(base), Some(end)) = (
            self.block_row_base.get(block).map(|b| *b as usize),
            block
                .checked_add(1)
                .and_then(|next| self.block_row_base.get(next))
                .map(|b| *b as usize),
        ) else {
            return result;
        };
        for row in base..end {
            result.extend(
                Self::csr_row(&self.loc_uses_offsets, &self.loc_uses_values, row)
                    .iter()
                    .copied(),
            );
        }
        result
    }

    /// Finds the unique use site if a variable has exactly one use.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// The single use site, or `None` if the variable has zero or multiple uses.
    #[must_use]
    pub fn single_use_site(&self, var: SsaVarId) -> Option<UseSite> {
        self.uses_slice(var.index()).and_then(|uses| {
            if uses.len() == 1 {
                uses.first().copied()
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::DefUseIndex;
    use crate::{
        ir::{
            block::SsaBlock,
            function::SsaFunction,
            instruction::SsaInstruction,
            ops::SsaOp,
            phi::PhiNode,
            value::ConstValue,
            variable::{DefSite, SsaVarId, SsaVariable, UseSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    type SsaType = MockType;

    /// Helper to create test SSA and return variable IDs for assertions
    fn make_test_ssa() -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        // Create a simple SSA function:
        // Block 0:
        //   v0 = const 42
        //   v1 = add v0, v0
        //   ret v1
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);

        // Create variables first to get their IDs
        let mut v0 = SsaVariable::new(
            SsaVarId::from_index(0),
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            SsaType::Unknown,
        );
        let id0 = v0.id();
        v0.add_use(UseSite::instruction(0, 1));
        v0.add_use(UseSite::instruction(0, 1)); // Used twice in add
        ssa.variables_mut().push(v0);

        let mut v1 = SsaVariable::new(
            SsaVarId::from_index(1),
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            SsaType::Unknown,
        );
        let id1 = v1.id();
        v1.add_use(UseSite::instruction(0, 2));
        ssa.variables_mut().push(v1);

        // Now create block with instructions using the auto-allocated IDs
        let mut block = SsaBlock::new(0);

        // v0 = const 42
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: id0,
            value: ConstValue::I32(42),
        }));

        // v1 = add v0, v0
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: id1,
            left: id0,
            right: id0,
            flags: None,
        }));

        // ret v1
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(id1),
        }));

        ssa.add_block(block);

        (ssa, id0, id1)
    }

    #[test]
    fn test_build_index() {
        let (ssa, _id0, _id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        assert_eq!(index.variable_count(), 2);
    }

    #[test]
    fn test_def_site_lookup() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        let def0 = index.def_site(id0).unwrap();
        assert_eq!(def0.block, 0);
        assert_eq!(def0.instruction, Some(0));

        let def1 = index.def_site(id1).unwrap();
        assert_eq!(def1.block, 0);
        assert_eq!(def1.instruction, Some(1));
    }

    #[test]
    fn test_uses_of() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        // v0 is used twice
        let uses0 = index.uses_of(id0).unwrap();
        assert_eq!(uses0.len(), 2);

        // v1 is used once
        let uses1 = index.uses_of(id1).unwrap();
        assert_eq!(uses1.len(), 1);
    }

    #[test]
    fn test_use_count() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        assert_eq!(index.use_count(id0), 2);
        assert_eq!(index.use_count(id1), 1);
        assert_eq!(index.use_count(SsaVarId::from_index(999999)), 0); // Unknown var
    }

    #[test]
    fn test_defs_at_location() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        let defs_0_0 = index.defs_at(0, 0);
        assert_eq!(defs_0_0.len(), 1);
        assert!(defs_0_0.contains(&id0));

        let defs_0_1 = index.defs_at(0, 1);
        assert_eq!(defs_0_1.len(), 1);
        assert!(defs_0_1.contains(&id1));

        // No defs at ret instruction
        let defs_0_2 = index.defs_at(0, 2);
        assert!(defs_0_2.is_empty());
    }

    #[test]
    fn test_build_with_ops_indexes_secondary_defs() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let value = SsaVarId::from_index(2);
        let flags = SsaVarId::from_index(3);

        ssa.variables_mut().push(SsaVariable::new(
            left,
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            SsaType::I32,
        ));
        ssa.variables_mut().push(SsaVariable::new(
            right,
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            SsaType::I32,
        ));
        ssa.variables_mut().push(SsaVariable::new(
            value,
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            SsaType::I32,
        ));
        ssa.variables_mut().push(SsaVariable::new(
            flags,
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(0, 2),
            SsaType::I32,
        ));

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: left,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: right,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: value,
            left,
            right,
            flags: Some(flags),
        }));
        ssa.add_block(block);

        let index = DefUseIndex::build_with_ops(&ssa);
        assert!(matches!(
            index.def_op(value),
            Some(SsaOp::Add {
                flags: Some(f), ..
            }) if *f == flags
        ));
        assert!(matches!(
            index.def_op(flags),
            Some(SsaOp::Add {
                dest,
                flags: Some(f),
                ..
            }) if *dest == value && *f == flags
        ));
    }

    #[test]
    fn test_uses_at_location() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        // v0 used at instruction 1
        let uses_0_1 = index.uses_at(0, 1);
        assert!(uses_0_1.contains(&id0));

        // v1 used at instruction 2
        let uses_0_2 = index.uses_at(0, 2);
        assert!(uses_0_2.contains(&id1));
    }

    #[test]
    fn test_defs_in_block() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        let defs = index.defs_in_block(0);
        assert_eq!(defs.len(), 2);
        assert!(defs.contains(&id0));
        assert!(defs.contains(&id1));
    }

    #[test]
    fn test_unused_variables() {
        // Create SSA with an unused variable
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut block = SsaBlock::new(0);

        let dest0 = SsaVarId::from_index(0);
        let dest1 = SsaVarId::from_index(1);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: dest0,
            value: ConstValue::I32(42),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: dest1,
            value: ConstValue::I32(0),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(dest1),
        }));

        ssa.add_block(block);

        // v0: defined but never used
        let v0 = SsaVariable::new(
            dest0,
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            SsaType::Unknown,
        );
        let v0_id = v0.id();
        ssa.variables_mut().push(v0);

        // v1: defined and used
        let mut v1 = SsaVariable::new(
            dest1,
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            SsaType::Unknown,
        );
        let v1_id = v1.id();
        v1.add_use(UseSite::instruction(0, 2));
        ssa.variables_mut().push(v1);

        let index = DefUseIndex::build(&ssa);

        assert!(index.is_unused(v0_id));
        assert!(!index.is_unused(v1_id));
        assert_eq!(index.unused_count(), 1);
        assert!(index.unused_variables().contains(v0_id.index()));
    }

    #[test]
    fn test_single_use() {
        let (ssa, v0_id, v1_id) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        // v0 has 2 uses -> not single use
        assert!(!index.is_single_use(v0_id));

        // v1 has 1 use -> single use
        assert!(index.is_single_use(v1_id));

        // Get the single use site
        let use_site = index.single_use_site(v1_id).unwrap();
        assert_eq!(use_site.block, 0);
        assert_eq!(use_site.instruction, 2);

        // v0 doesn't have single use site
        assert!(index.single_use_site(v0_id).is_none());
    }

    #[test]
    fn test_phi_definitions() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut block = SsaBlock::new(0);
        block.add_phi(PhiNode::new(SsaVarId::from_index(0), VariableOrigin::Phi));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: SsaVarId::from_index(1),
            value: ConstValue::I32(0),
        }));
        ssa.add_block(block);

        // v0: phi definition
        let v0 = SsaVariable::new(
            SsaVarId::from_index(0),
            VariableOrigin::Phi,
            0,
            DefSite::phi(0),
            SsaType::Unknown,
        );
        let v0_id = v0.id();
        ssa.variables_mut().push(v0);

        // v1: instruction definition
        let v1 = SsaVariable::new(
            SsaVarId::from_index(1),
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            SsaType::Unknown,
        );
        let v1_id = v1.id();
        ssa.variables_mut().push(v1);

        let index = DefUseIndex::build(&ssa);

        assert!(index.is_phi_def(v0_id));
        assert!(!index.is_phi_def(v1_id));
        assert!(index.phi_definitions().contains(v0_id.index()));
    }

    #[test]
    fn test_uses_in_block() {
        let (ssa, v0_id, v1_id) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        let uses = index.uses_in_block(0);
        assert!(uses.contains(&v0_id));
        assert!(uses.contains(&v1_id));
    }

    #[test]
    fn test_default() {
        let index: DefUseIndex<MockTarget> = DefUseIndex::default();
        assert_eq!(index.variable_count(), 0);
        assert_eq!(index.unused_count(), 0);
    }

    #[test]
    fn test_build_without_ops() {
        let (ssa, id0, _id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        // Index built without ops should not have operations
        assert!(!index.has_ops());
        assert!(index.def_op(id0).is_none());
        assert!(index.full_definition(id0).is_none());
    }

    #[test]
    fn test_build_with_ops() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build_with_ops(&ssa);

        // Index built with ops should have operations
        assert!(index.has_ops());

        // Check v0's defining operation (const 42)
        let op0 = index.def_op(id0).unwrap();
        assert!(matches!(op0, SsaOp::Const { value, .. } if value.as_i32() == Some(42)));

        // Check v1's defining operation (add)
        let op1 = index.def_op(id1).unwrap();
        assert!(matches!(op1, SsaOp::Add { .. }));
    }

    #[test]
    fn test_full_definition() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build_with_ops(&ssa);

        // v0 = const 42 at block 0, instruction 0
        let (block0, instr0, op0) = index.full_definition(id0).unwrap();
        assert_eq!(block0, 0);
        assert_eq!(instr0, 0);
        assert!(matches!(op0, SsaOp::Const { .. }));

        // v1 = add at block 0, instruction 1
        let (block1, instr1, op1) = index.full_definition(id1).unwrap();
        assert_eq!(block1, 0);
        assert_eq!(instr1, 1);
        assert!(matches!(op1, SsaOp::Add { .. }));
    }

    #[test]
    fn test_full_definition_phi_returns_none() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let block = SsaBlock::new(0);
        ssa.add_block(block);

        // v0: phi definition (no instruction index)
        let v0 = SsaVariable::new(
            SsaVarId::from_index(0),
            VariableOrigin::Phi,
            0,
            DefSite::phi(0),
            SsaType::Unknown,
        );
        let v0_id = v0.id();
        ssa.variables_mut().push(v0);

        let index = DefUseIndex::build_with_ops(&ssa);

        // Phi definitions should return None from full_definition
        // (because there's no instruction index)
        assert!(index.full_definition(v0_id).is_none());

        // But def_site still works
        let site = index.def_site(v0_id).unwrap();
        assert_eq!(site.block, 0);
        assert!(site.instruction.is_none());
    }

    #[test]
    fn test_build_with_ops_map_compatibility() {
        let (ssa, id0, id1) = make_test_ssa();
        let (index, ops) = DefUseIndex::build_with_ops_map(&ssa);

        // The index should have ops internally
        assert!(index.has_ops());

        // The returned map should have the same ops
        assert!(ops.contains_key(&id0));
        assert!(ops.contains_key(&id1));

        // Both should match
        let op0_from_index = index.def_op(id0).unwrap();
        let op0_from_map = ops.get(&id0).unwrap();
        assert!(matches!(op0_from_index, SsaOp::Const { .. }));
        assert!(matches!(op0_from_map, SsaOp::Const { .. }));
    }
}
