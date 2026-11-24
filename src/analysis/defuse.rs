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
//! nearly every pass, and the nested form would allocate once per variable and
//! once per instruction on every rebuild. The flat form costs one extra counting
//! walk and a fixed handful of buffers, and keeps every query O(1)/O(k) on a
//! slice.
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
//!         println!("v0 used at block {}, instr {}", use_site.block(), use_site.index());
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
    graph::Adjacency,
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

    /// The use sites of each variable, indexed densely by `SsaVarId::index()`
    /// over `0..capacity`.
    ///
    /// Compressed rather than a `Vec<Vec<UseSite>>` because this index is
    /// rebuilt from scratch by nearly every pass, and a vector-per-variable
    /// allocates once for every variable that has any use, on every rebuild.
    uses: Adjacency<UseSite>,

    /// Row base for each block: block `b` owns rows
    /// `block_row_base[b]..block_row_base[b + 1]`, and row `block_row_base[b] + i`
    /// is `Location(b, i)`. Length is `block_count + 1`.
    ///
    /// A block owns `max(phi_nodes, instructions)` rows because phi-operand uses
    /// and instruction uses **share** a row index - preserving the behaviour of
    /// the `Location`-keyed maps this replaces, where `Location(b, 0)` held both
    /// phi 0's operands and instruction 0's uses.
    block_row_base: Vec<usize>,

    /// Variables defined at each row.
    ///
    /// Compressed for the same reason as [`Self::uses`]: the previous
    /// `BTreeMap<Location, Vec<SsaVarId>>` cost a tree node *and* a vector for
    /// every instruction, on an index that is rebuilt by nearly every pass.
    loc_defs: Adjacency<SsaVarId>,

    /// Variables used at each row.
    loc_uses: Adjacency<SsaVarId>,

    /// Variables defined in each block (including phi nodes), indexed by block.
    defs_in_block: Vec<Vec<SsaVarId>>,

    /// Variables some phi or instruction in the function defines, found by
    /// scanning the IR.
    ///
    /// Not the same question as [`Self::def_site`], which answers from the
    /// variable table first and so still names the instruction that defined a
    /// value after a pass removed that instruction.
    active_defs: BitSet,

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
            uses: Adjacency::default(),
            block_row_base: Vec::new(),
            loc_defs: Adjacency::default(),
            loc_uses: Adjacency::default(),
            defs_in_block: Vec::new(),
            phi_defs: BitSet::default(),
            active_defs: BitSet::default(),
            unused_vars: BitSet::default(),
            var_count: 0,
            def_ops: None,
        }
    }
}

impl<T: Target> DefUseIndex<T> {
    /// Returns the recorded use sites for a dense variable index, or `None`
    /// when the index falls outside the registered variable range.
    fn uses_slice(&self, index: usize) -> Option<&[UseSite]> {
        (index < self.uses.node_count()).then(|| self.uses.row(index))
    }

    /// Returns the flat row index for `Location(block, index)`, or `None` when
    /// the block does not exist or the index is past the block's row count.
    fn row_of(&self, block: usize, index: usize) -> Option<usize> {
        let base = *self.block_row_base.get(block)?;
        let end = *self.block_row_base.get(block.checked_add(1)?)?;
        let row = base.checked_add(index)?;
        (row < end).then_some(row)
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
        let mut block_row_base = Vec::with_capacity(ssa.block_count().saturating_add(1));
        let mut row_total = 0usize;
        block_row_base.push(0);
        for block in ssa.blocks() {
            row_total =
                row_total.saturating_add(block.phi_nodes().len().max(block.instructions().len()));
            block_row_base.push(row_total);
        }
        let loc_defs = Adjacency::from_walk(row_total, SsaVarId::PLACEHOLDER, |visit| {
            for (block_index, block) in ssa.blocks().iter().enumerate() {
                let base = block_row_base.get(block_index).copied().unwrap_or(0);
                for (instruction_index, instruction) in block.instructions().iter().enumerate() {
                    if let Some(row) = base.checked_add(instruction_index) {
                        instruction.op().for_each_def(|dest| visit(row, dest));
                    }
                }
            }
        });
        let loc_uses = Adjacency::from_walk(row_total, SsaVarId::PLACEHOLDER, |visit| {
            for (block_index, block) in ssa.blocks().iter().enumerate() {
                let base = block_row_base.get(block_index).copied().unwrap_or(0);
                for (phi_index, phi) in block.phi_nodes().iter().enumerate() {
                    if let Some(row) = base.checked_add(phi_index) {
                        for operand in phi.operands() {
                            visit(row, operand.value());
                        }
                    }
                }
                for (instruction_index, instruction) in block.instructions().iter().enumerate() {
                    if let Some(row) = base.checked_add(instruction_index) {
                        instruction.op().for_each_use(|var| visit(row, var));
                    }
                }
            }
        });
        let capacity = ssa.var_id_bound();
        let uses = Adjacency::from_walk(capacity, UseSite::instruction(0, 0), |visit| {
            for (block_index, block) in ssa.blocks().iter().enumerate() {
                for (phi_index, phi) in block.phi_nodes().iter().enumerate() {
                    for operand in phi.operands() {
                        visit(
                            operand.value().index(),
                            UseSite::phi_operand(block_index, phi_index),
                        );
                    }
                }
                for (instruction_index, instruction) in block.instructions().iter().enumerate() {
                    instruction.op().for_each_use(|var| {
                        visit(
                            var.index(),
                            UseSite::instruction(block_index, instruction_index),
                        );
                    });
                }
            }
        });
        let mut definitions = vec![None; capacity];
        let mut defs_in_block = vec![Vec::new(); ssa.block_count()];
        let mut phi_defs = BitSet::new(capacity);
        let mut active_defs = BitSet::new(capacity);
        for variable in ssa.variables() {
            if let Some(slot) = definitions.get_mut(variable.id().index()) {
                *slot = Some(variable.def_site());
            }
        }
        for (block_index, block) in ssa.blocks().iter().enumerate() {
            for phi in block.phi_nodes() {
                let dest = phi.result();
                if let Some(slot) = definitions.get_mut(dest.index()) {
                    *slot = Some(DefSite::phi(block_index));
                }
                phi_defs.insert_checked(dest.index());
                active_defs.insert_checked(dest.index());
                if let Some(slot) = defs_in_block.get_mut(block_index) {
                    slot.push(dest);
                }
            }
            for (instruction_index, instruction) in block.instructions().iter().enumerate() {
                instruction.op().for_each_def(|dest| {
                    if let Some(slot) = definitions.get_mut(dest.index()) {
                        *slot = Some(DefSite::instruction(block_index, instruction_index));
                    }
                    active_defs.insert_checked(dest.index());
                    if let Some(slot) = defs_in_block.get_mut(block_index) {
                        slot.push(dest);
                    }
                });
            }
        }
        let mut unused_vars = BitSet::new(capacity);
        for variable in ssa.variables() {
            let index = variable.id().index();
            if index < uses.node_count() && uses.row(index).is_empty() {
                unused_vars.insert_checked(index);
            }
        }
        Self {
            definitions,
            uses,
            block_row_base,
            loc_defs,
            loc_uses,
            defs_in_block,
            active_defs,
            phi_defs,
            unused_vars,
            var_count: ssa.variable_count(),
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
        let instr = site.index()?; // None for phi nodes
        let op = self.def_op(var)?;
        Some((site.block(), instr, op))
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

    /// Checks whether a variable is absent from the program: no instruction or
    /// phi in the function defines it, and nothing reads it.
    ///
    /// Such a value keeps its entry in the variable table - an entry
    /// placeholder SSA construction rewrote to another value, a definition a
    /// pass removed - but describes nothing the program computes or consumes.
    /// A value the function is entered with is present exactly while something
    /// reads it.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable ID to check.
    ///
    /// # Returns
    ///
    /// `true` if nothing in the function defines or reads the variable.
    #[must_use]
    pub fn is_absent(&self, var: SsaVarId) -> bool {
        let defined =
            var.index() < self.active_defs.len() && self.active_defs.contains(var.index());
        !defined && self.is_unused(var)
    }

    /// Returns a site only when the current IR actually defines the value.
    ///
    /// Registered historical sites remain available through [`Self::def_site`],
    /// but are not evidence that a removed definition still produces a value.
    pub fn active_def_site(&self, var: SsaVarId) -> Option<DefSite> {
        self.active_defs
            .contains_checked(var.index())
            .then(|| self.def_site(var))
            .flatten()
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
        self.row_of(block, instruction)
            .map_or(&[], |row| self.loc_defs.row(row))
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
        self.row_of(block, instruction)
            .map_or(&[], |row| self.loc_uses.row(row))
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
            .is_some_and(|uses| !uses.is_empty() && uses.iter().all(|u| u.is_phi_operand()))
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
        let (Some(&base), Some(&end)) = (
            self.block_row_base.get(block),
            block
                .checked_add(1)
                .and_then(|next| self.block_row_base.get(next)),
        ) else {
            return result;
        };
        for row in base..end {
            result.extend(self.loc_uses.row(row).iter().copied());
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
            float::{FloatExceptionLayout, FloatExceptions},
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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(id1))));

        ssa.add_block(block);

        (ssa, id0, id1)
    }

    /// A value is absent when the program neither defines nor reads it, whatever
    /// the variable table still records for it.
    ///
    /// `stale` names an instruction in its table entry that the function does
    /// not contain, and nothing reads it. `arrives` is defined at entry and read.
    /// `unread` is defined by an instruction and never read. Only `stale` is
    /// absent: a read makes an entry value present, and so does a definition.
    #[test]
    fn a_value_neither_defined_nor_read_is_absent() {
        let (mut ssa, id0, id1) = make_test_ssa();
        let stale = SsaVarId::from_index(2);
        ssa.variables_mut().push(SsaVariable::new(
            stale,
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 7),
            SsaType::Unknown,
        ));
        let arrives = SsaVarId::from_index(3);
        ssa.variables_mut().push(SsaVariable::new(
            arrives,
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            SsaType::Unknown,
        ));
        let unread = SsaVarId::from_index(4);
        ssa.variables_mut().push(SsaVariable::new(
            unread,
            VariableOrigin::Local(4),
            0,
            DefSite::instruction(0, 0),
            SsaType::Unknown,
        ));
        let mut block = SsaBlock::new(1);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: unread,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(arrives))));
        ssa.add_block(block);

        let index = DefUseIndex::build(&ssa);
        assert!(index.is_absent(stale));
        assert!(!index.is_absent(arrives));
        assert!(!index.is_absent(unread));
        assert!(!index.is_absent(id0));
        assert!(!index.is_absent(id1));
    }

    /// Verifies a built index reports the number of registered variables.
    #[test]
    fn test_build_index() {
        let (ssa, _id0, _id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        assert_eq!(index.variable_count(), 2);
    }

    /// Verifies `def_site` returns each variable's defining block and instruction.
    #[test]
    fn test_def_site_lookup() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        let def0 = index.def_site(id0).unwrap();
        assert_eq!(def0.block(), 0);
        assert_eq!(def0.index(), Some(0));

        let def1 = index.def_site(id1).unwrap();
        assert_eq!(def1.block(), 0);
        assert_eq!(def1.index(), Some(1));
    }

    /// Verifies `uses_of` returns every recorded use site of a variable.
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

    /// Verifies `use_count` counts uses and returns zero for an unknown variable.
    #[test]
    fn test_use_count() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        assert_eq!(index.use_count(id0), 2);
        assert_eq!(index.use_count(id1), 1);
        assert_eq!(index.use_count(SsaVarId::from_index(999999)), 0); // Unknown var
    }

    /// Verifies `defs_at` reports the variable defined at each instruction and none at a non-defining one.
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

    /// Verifies `build_with_ops` indexes secondary definitions so both an op's primary result and its flags result resolve to that op.
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

    /// Verifies `uses_at` reports the variables used at a given instruction.
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

    /// Verifies `defs_in_block` lists every variable defined in a block.
    #[test]
    fn test_defs_in_block() {
        let (ssa, id0, id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        let defs = index.defs_in_block(0);
        assert_eq!(defs.len(), 2);
        assert!(defs.contains(&id0));
        assert!(defs.contains(&id1));
    }

    /// Verifies unused-variable detection flags a defined-but-never-used variable and not a used one.
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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest1))));

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

    /// Verifies `is_single_use` and `single_use_site` identify variables with exactly one use.
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
        assert_eq!(use_site.block(), 0);
        assert_eq!(use_site.index(), 2);

        // v0 doesn't have single use site
        assert!(index.single_use_site(v0_id).is_none());
    }

    /// Verifies phi-defined variables are distinguished from instruction-defined ones.
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

    /// Verifies `uses_in_block` gathers all variables used anywhere in a block.
    #[test]
    fn test_uses_in_block() {
        let (ssa, v0_id, v1_id) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        let uses = index.uses_in_block(0);
        assert!(uses.contains(&v0_id));
        assert!(uses.contains(&v1_id));
    }

    /// Verifies a default-constructed index is empty.
    #[test]
    fn test_default() {
        let index: DefUseIndex<MockTarget> = DefUseIndex::default();
        assert_eq!(index.variable_count(), 0);
        assert_eq!(index.unused_count(), 0);
    }

    /// Verifies an index built without ops reports no operation information.
    #[test]
    fn test_build_without_ops() {
        let (ssa, id0, _id1) = make_test_ssa();
        let index = DefUseIndex::build(&ssa);

        // Index built without ops should not have operations
        assert!(!index.has_ops());
        assert!(index.def_op(id0).is_none());
        assert!(index.full_definition(id0).is_none());
    }

    /// Verifies `build_with_ops` records each variable's defining operation.
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

    /// Verifies `full_definition` returns block, instruction, and defining op together.
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

    /// Verifies `full_definition` returns `None` for phi-defined variables while `def_site` still resolves them.
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
        assert_eq!(site.block(), 0);
        assert!(site.index().is_none());
    }

    /// Verifies `build_with_ops_map` returns a standalone op map matching the index's own ops.
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

    /// A loop with a phi, flags, duplicate uses, stale sites and one unregistered operand.
    fn canonical_defuse_fixture() -> (SsaFunction<MockTarget>, [SsaVarId; 7]) {
        let mut ssa = SsaFunction::<MockTarget>::new(2, 5);
        let ids = [
            VariableOrigin::Argument(0),
            VariableOrigin::Local(0),
            VariableOrigin::Local(1),
            VariableOrigin::Local(2),
            VariableOrigin::Local(3),
            VariableOrigin::Argument(1),
            VariableOrigin::Local(4),
        ]
        .map(|origin| ssa.create_variable(origin, 0, DefSite::entry(), MockType::I32));
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: ids[1],
            value: ConstValue::I32(7),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);
        let mut body = SsaBlock::new(1);
        let mut phi = PhiNode::new(ids[2], VariableOrigin::Local(1));
        phi.set_operand(0, ids[0]);
        phi.set_operand(1, ids[3]);
        body.add_phi(phi);
        body.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: ids[3],
            left: ids[2],
            right: ids[0],
            flags: Some(ids[6]),
        }));
        body.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: ids[3],
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(body);
        let mut exit = SsaBlock::new(2);
        exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns_with(
            Some(ids[3]),
            vec![ids[2], ids[6], SsaVarId::from_index(900)],
        )));
        ssa.add_block(exit);
        for index in [1, 2, 3, 6] {
            ssa.variable_mut(ids[index])
                .unwrap()
                .set_def_site(DefSite::instruction(99, 99));
        }
        ssa.variable_mut(ids[4])
            .unwrap()
            .set_def_site(DefSite::instruction(1, 0));
        (ssa, ids)
    }

    /// Actual IR overrides stale metadata while preserving phi/instruction row overlap and use order.
    #[test]
    fn canonical_defuse_matches_independent_literal_site_and_use_oracles() {
        let (ssa, v) = canonical_defuse_fixture();
        let index = DefUseIndex::build(&ssa);
        assert_eq!(index.variable_count(), 7);
        assert_eq!(index.def_site(v[0]), Some(DefSite::entry()));
        assert_eq!(index.active_def_site(v[0]), None);
        assert_eq!(index.def_site(v[1]), Some(DefSite::instruction(0, 0)));
        assert_eq!(index.active_def_site(v[2]), Some(DefSite::phi(1)));
        assert_eq!(
            index.active_def_site(v[3]),
            Some(DefSite::instruction(1, 0))
        );
        assert_eq!(
            index.active_def_site(v[6]),
            Some(DefSite::instruction(1, 0))
        );
        assert_eq!(index.def_site(v[4]), Some(DefSite::instruction(1, 0)));
        assert_eq!(index.active_def_site(v[4]), None);
        assert_eq!(index.defs_at(0, 0), [v[1]]);
        assert_eq!(index.defs_at(1, 0), [v[3], v[6]]);
        assert_eq!(index.defs_in_block(1), [v[2], v[3], v[6]]);
        assert_eq!(index.uses_at(1, 0), [v[0], v[3], v[2], v[0]]);
        assert_eq!(
            index.uses_at(2, 0),
            [v[3], v[2], v[6], SsaVarId::from_index(900)]
        );
        assert_eq!(
            index.uses_of(v[0]),
            Some(&[UseSite::phi_operand(1, 0), UseSite::instruction(1, 0)][..])
        );
        assert_eq!(
            index.uses_of(v[3]),
            Some(
                &[
                    UseSite::phi_operand(1, 0),
                    UseSite::instruction(1, 1),
                    UseSite::instruction(2, 0)
                ][..]
            )
        );
        assert_eq!(index.uses_of(SsaVarId::from_index(900)), None);
        assert!(index.is_unused(v[1]));
        assert!(!index.is_absent(v[1]));
        assert!(index.is_absent(v[4]));
        assert!(index.is_absent(v[5]));
    }

    /// Nested raises retain only their effective definitions and every mask use.
    #[test]
    fn canonical_defuse_preserves_nested_raise_definitions_and_mask_uses() {
        for masked in [false, true] {
            let mut ssa = SsaFunction::<MockTarget>::new(2, 1);
            let left = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let right = ssa.create_variable(
                VariableOrigin::Argument(1),
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let dest = ssa.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            );
            let mut operation = SsaOp::Add {
                dest: SsaVarId::from_index(900),
                left,
                right,
                flags: Some(SsaVarId::from_index(901)),
            };
            for _ in 0..1024 {
                operation = SsaOp::FloatRaise {
                    dest,
                    op: Box::new(operation),
                    lanes: masked.then_some(left),
                    raises: FloatExceptions::NONE,
                    layout: FloatExceptionLayout::IEEE,
                };
            }
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(operation));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest))));
            ssa.add_block(block);
            let index = DefUseIndex::build(&ssa);
            assert_eq!(index.defs_at(0, 0), [dest]);
            assert_eq!(index.use_count(left), if masked { 1025 } else { 1 });
            assert_eq!(index.use_count(right), 1);
        }
    }
}
