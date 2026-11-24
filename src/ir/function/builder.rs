//! Fluent SSA function construction.
//!
//! This module provides [`SsaFunctionBuilder`], an ergonomic construction API for
//! tests, fixtures, and frontends. It owns a fresh [`SsaFunction`] while it is
//! being assembled, registers every builder-created definition, records correct
//! definition sites, and can recompute uses plus run the verifier before handing
//! the function to callers.

use std::collections::BTreeMap;

use crate::{
    analysis::verifier::{SsaVerifier, VerifyLevel},
    error::{Error, Result},
    ir::{
        block::SsaBlock,
        float::FloatClassLayout,
        function::{EntryRule, SsaConstructOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::{
            AccessQualifiers, AtomicAccessWidth, AtomicOrdering, AtomicRmwOp, BinaryOpKind,
            BreakpointOp, CallData, Callee, CmpKind, FenceKind, FlagCondition, FlagsMask, SsaOp,
            UnaryOpKind, VectorBinaryKind, VectorBitmaskKind, VectorCastKind, VectorCompareKind,
            VectorElement, VectorFaultMode, VectorMaskMode, VectorReduceKind, VectorSegmentLayout,
            VectorShuffleSelector, VectorTernaryKind, VectorUnaryKind,
        },
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

/// Describes a definition that a builder-created instruction or phi node will produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaDefSpec<T: Target> {
    /// Variable origin used for version tracking and rename groups.
    pub origin: VariableOrigin,
    /// Type assigned to the new SSA variable.
    pub var_type: T::Type,
}

impl<T: Target> SsaDefSpec<T> {
    /// Creates a definition specification.
    #[must_use]
    pub const fn new(origin: VariableOrigin, var_type: T::Type) -> Self {
        Self { origin, var_type }
    }

    /// Creates a temporary definition specification using [`VariableOrigin::Phi`].
    #[must_use]
    pub const fn tmp(var_type: T::Type) -> Self {
        Self::new(VariableOrigin::Phi, var_type)
    }

    /// Creates an argument-origin definition specification.
    #[must_use]
    pub const fn argument(index: u16, var_type: T::Type) -> Self {
        Self::new(VariableOrigin::Argument(index), var_type)
    }

    /// Creates a local-origin definition specification.
    #[must_use]
    pub const fn local(index: u16, var_type: T::Type) -> Self {
        Self::new(VariableOrigin::Local(index), var_type)
    }
}

/// Describes an atomic compare-exchange operation emitted by [`SsaBlockBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicCmpXchgSpec<T: Target> {
    /// Definition for the loaded old value.
    pub old_def: SsaDefSpec<T>,
    /// Optional definition for the boolean success result.
    pub success_def: Option<SsaDefSpec<T>>,
    /// Address being compared and exchanged.
    pub addr: SsaVarId,
    /// Expected value compared against memory.
    pub expected: SsaVarId,
    /// Desired value written on success.
    pub desired: SsaVarId,
    /// Memory ordering used on a successful exchange.
    pub success_ordering: AtomicOrdering,
    /// Memory ordering used on a failed exchange.
    pub failure_ordering: AtomicOrdering,
    /// Access width used by the atomic operation.
    pub width: AtomicAccessWidth,
    /// Whether the compare-exchange may fail spuriously.
    pub weak: bool,
    /// How the access is made, volatility included ([`AccessQualifiers`]).
    pub access: AccessQualifiers,
}

/// Describes a vector gather operation emitted by [`SsaBlockBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorGatherSpec<T: Target> {
    /// Definition for the gathered vector result.
    pub def: SsaDefSpec<T>,
    /// Base address vector or scalar base used by the gather.
    pub base: SsaVarId,
    /// Vector of indices used by the gather.
    pub indices: SsaVarId,
    /// Mask controlling which lanes are loaded.
    pub mask: SsaVarId,
    /// Optional passthrough vector for inactive lanes.
    pub passthrough: Option<SsaVarId>,
    /// Type of the gathered vector.
    pub vector_type: T::Type,
    /// Masking policy used by inactive lanes.
    pub mode: VectorMaskMode,
}

/// Describes a faulting vector load operation emitted by [`SsaBlockBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorFaultingLoadSpec<T: Target> {
    /// Definition for the loaded vector result.
    pub dest_def: SsaDefSpec<T>,
    /// Optional definition for the fault-status mask.
    pub fault_def: Option<SsaDefSpec<T>>,
    /// Address loaded by the operation.
    pub addr: SsaVarId,
    /// Optional lane mask.
    pub mask: Option<SsaVarId>,
    /// Optional passthrough vector for inactive lanes.
    pub passthrough: Option<SsaVarId>,
    /// Type of the loaded vector.
    pub vector_type: T::Type,
    /// Faulting behavior used by the load.
    pub fault_mode: VectorFaultMode,
    /// Masking policy used by inactive lanes.
    pub mask_mode: VectorMaskMode,
}

/// Describes a lock-prefixed atomic read-modify-write operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicLockRmwSpec<T: Target> {
    /// Definition for the previous memory value.
    pub def: SsaDefSpec<T>,
    /// Address modified by the operation.
    pub addr: SsaVarId,
    /// Value combined with memory.
    pub value: SsaVarId,
    /// Read-modify-write operator.
    pub op: AtomicRmwOp,
    /// Memory ordering used by the operation.
    pub ordering: AtomicOrdering,
    /// Access width used by the atomic operation.
    pub width: AtomicAccessWidth,
    /// How the access is made, volatility included ([`AccessQualifiers`]).
    pub access: AccessQualifiers,
}

/// Fluent builder for constructing a fresh [`SsaFunction`].
#[derive(Debug, Clone)]
pub struct SsaFunctionBuilder<T: Target> {
    function: SsaFunction<T>,
    next_versions: BTreeMap<VariableOrigin, u32>,
}

impl<T: Target> SsaFunctionBuilder<T> {
    /// Creates a builder for a new empty SSA function.
    #[must_use]
    pub fn new(num_args: usize, num_locals: usize) -> Self {
        Self {
            function: SsaFunction::new(num_args, num_locals),
            next_versions: BTreeMap::new(),
        }
    }

    /// Creates a builder with pre-allocated function storage.
    #[must_use]
    pub fn with_capacity(
        num_args: usize,
        num_locals: usize,
        block_capacity: usize,
        var_capacity: usize,
    ) -> Self {
        Self {
            function: SsaFunction::with_capacity(
                num_args,
                num_locals,
                block_capacity,
                var_capacity,
            ),
            next_versions: BTreeMap::new(),
        }
    }

    /// Returns the function currently being built.
    #[must_use]
    pub const fn function(&self) -> &SsaFunction<T> {
        &self.function
    }

    /// Returns a mutable reference to the function currently being built.
    pub fn function_mut(&mut self) -> &mut SsaFunction<T> {
        &mut self.function
    }

    /// Registers the canonical type for a variable origin.
    pub fn register_origin_type(&mut self, origin: VariableOrigin, var_type: T::Type) {
        self.function.register_origin_type(origin, var_type);
    }

    /// Returns the registered type for a variable origin.
    #[must_use]
    pub fn origin_type(&self, origin: VariableOrigin) -> T::Type {
        self.function.origin_type(origin)
    }

    /// Ensures that a dense block with `block_idx` exists and returns it.
    ///
    /// Blocks are stored in a dense vector indexed by id, so reaching
    /// `block_idx` requires that every lower index already exist. This appends
    /// fresh empty blocks one at a time until the count exceeds `block_idx`,
    /// back-filling any gap rather than requiring callers to create
    /// intermediate blocks in order.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block to guarantee exists.
    ///
    /// # Returns
    ///
    /// `block_idx`, now known to be a valid block index.
    pub fn ensure_block(&mut self, block_idx: usize) -> usize {
        while self.function.block_count() <= block_idx {
            let id = self.function.block_count();
            self.function.add_block(SsaBlock::new(id));
        }
        block_idx
    }

    /// Appends a new empty block and returns its index.
    pub fn append_block(&mut self) -> usize {
        let block_idx = self.function.block_count();
        self.function.add_block(SsaBlock::new(block_idx));
        block_idx
    }

    /// Builds instructions or phi nodes inside `block_idx`.
    ///
    /// Scopes construction to a single block: [`ensure_block`](Self::ensure_block)
    /// guarantees the target exists, then a short-lived [`SsaBlockBuilder`]
    /// borrows this builder so the closure can append phi nodes and
    /// instructions. Threading the closure through a borrowed sub-builder is
    /// what lets each emitted definition record a def site pointing at the
    /// block currently under construction without the caller passing the block
    /// index into every call.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block to build into; created on demand if absent.
    /// * `build` - Closure that appends to the block via the [`SsaBlockBuilder`].
    ///
    /// # Returns
    ///
    /// The value the closure returns.
    ///
    /// # Errors
    ///
    /// Returns any error produced by `build`.
    pub fn in_block<R>(
        &mut self,
        block_idx: usize,
        build: impl FnOnce(&mut SsaBlockBuilder<'_, T>) -> Result<R>,
    ) -> Result<R> {
        self.ensure_block(block_idx);
        let mut block = SsaBlockBuilder {
            builder: self,
            block_idx,
        };
        build(&mut block)
    }

    /// Adds an operand to an existing phi node identified by its result variable.
    ///
    /// Completes a phi that was created without all of its operands, the shape
    /// [`empty_phi`](SsaBlockBuilder::empty_phi) leaves behind for loop
    /// backedges: the incoming value from a predecessor is not yet defined when
    /// the header phi is emitted, so its operand is filled in here once that
    /// block has been built. The phi is located by matching its result variable
    /// rather than by position, so operands may be added in any order.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block holding the phi node.
    /// * `phi_result` - Result variable that identifies the phi node.
    /// * `predecessor` - Predecessor block the incoming value arrives from.
    /// * `value` - Variable flowing in along that predecessor edge.
    ///
    /// # Errors
    ///
    /// Returns an error when the block or phi result cannot be found.
    pub fn add_phi_operand(
        &mut self,
        block_idx: usize,
        phi_result: SsaVarId,
        predecessor: usize,
        value: SsaVarId,
    ) -> Result<()> {
        let block = self
            .function
            .block_mut(block_idx)
            .ok_or_else(|| Error::new(format!("block {block_idx} does not exist")))?;
        let phi = block
            .phi_nodes_mut()
            .iter_mut()
            .find(|phi| phi.result() == phi_result)
            .ok_or_else(|| Error::new(format!("phi result {phi_result} not found")))?;
        phi.add_operand(PhiOperand::new(value, predecessor));
        Ok(())
    }

    /// Finishes construction and recomputes use lists.
    ///
    /// The builder tracks definitions as it emits them but never maintains the
    /// reverse use lists, so those are stale until this final pass rebuilds
    /// them from scratch. Before doing so it checks that each block sits at the
    /// array index equal to its own id: the rest of the crate indexes blocks by
    /// id and assumes that invariant, so a builder that produced a
    /// non-dense or misordered set of blocks is rejected here rather than
    /// handed on as a subtly corrupt function.
    ///
    /// Block numbering is the caller's, so a function whose entry block is a
    /// branch target is finished as built: the verifier reports it
    /// ([`EntryHasPredecessors`](crate::analysis::VerifierError::EntryHasPredecessors)),
    /// and [`SsaFunction::ensure_entry_has_no_predecessors`] is the repair,
    /// called by a caller that can account for every block moving up by one.
    ///
    /// # Returns
    ///
    /// The completed function with fully populated use lists.
    ///
    /// # Errors
    ///
    /// Returns an error if block identifiers are not dense and ordered.
    pub fn finish(mut self) -> Result<SsaFunction<T>> {
        for (idx, block) in self.function.blocks().iter().enumerate() {
            if block.id() != idx {
                return Err(Error::new(format!(
                    "block id {} is stored at index {idx}",
                    block.id()
                )));
            }
        }
        // The shared finishing sequence, with everything optional turned off: a
        // builder states its own phis, its own block set and its own numbering,
        // so pruning, folding or renumbering any of it would answer a question
        // the caller did not ask. What is left - identities, address-taken
        // slots, definition sites, use lists - is the part no caller can state
        // and every consumer assumes.
        self.function
            .finish_construction(&SsaConstructOptions {
                entry_rule: EntryRule::Keep,
                prune_unreachable: false,
                simplify_phis: false,
                compact_variables: false,
                verify: None,
                settle_conversions: None,
            })
            .map(|_| self.function)
    }

    /// Finishes construction, recomputes uses, and verifies the function.
    ///
    /// Runs [`finish`](Self::finish) and then the [`SsaVerifier`] at the
    /// requested level, folding any reported errors into a single failure. This
    /// is the entry point tests and fixtures use to assert that a hand-built
    /// function is well-formed SSA, so a non-empty error list is surfaced as an
    /// `Err` rather than left for a caller to inspect.
    ///
    /// # Arguments
    ///
    /// * `level` - Verification depth applied to the finished function.
    ///
    /// # Returns
    ///
    /// The completed, verified function.
    ///
    /// # Errors
    ///
    /// Returns an error if construction finalization fails or verification reports
    /// one or more SSA errors.
    pub fn finish_verified(self, level: VerifyLevel) -> Result<SsaFunction<T>> {
        let function = self.finish()?;
        let errors = SsaVerifier::new(&function).verify(level);
        if errors.is_empty() {
            Ok(function)
        } else {
            Err(Error::new(format!(
                "SSA builder verification failed: {errors:?}"
            )))
        }
    }

    /// Creates a new SSA variable for `spec` and advances its origin version.
    ///
    /// Each [`VariableOrigin`] carries an independent monotonic version counter,
    /// so successive definitions of the same source variable get distinct
    /// versions - the numbering SSA rename groups rely on. The counter is looked
    /// up (defaulting to zero for a first definition), stamped onto the new
    /// variable, then saturated-incremented so the next allocation for the same
    /// origin is one higher.
    ///
    /// # Arguments
    ///
    /// * `spec` - Origin and type of the variable to create.
    /// * `site` - Definition site recorded for the new variable.
    ///
    /// # Returns
    ///
    /// The id of the newly created variable.
    fn allocate_var(&mut self, spec: SsaDefSpec<T>, site: DefSite) -> SsaVarId {
        let version = self.next_versions.entry(spec.origin).or_insert(0);
        let id = self
            .function
            .create_variable(spec.origin, *version, site, spec.var_type);
        *version = version.saturating_add(1);
        id
    }
}

/// Fluent builder for appending phi nodes and instructions to one SSA block.
pub struct SsaBlockBuilder<'a, T: Target> {
    builder: &'a mut SsaFunctionBuilder<T>,
    block_idx: usize,
}

impl<'a, T: Target> SsaBlockBuilder<'a, T> {
    /// Returns the index of the block being built.
    #[must_use]
    pub const fn block_idx(&self) -> usize {
        self.block_idx
    }

    /// Emits an instruction that must not define any variables.
    ///
    /// The value-producing `emit_*` entry points allocate a variable and record
    /// its def site; this is the counterpart for effect-only operations (stores,
    /// terminators, prefixes). Rejecting an `op` that carries a definition guards
    /// against routing a result-producing operation through the path that would
    /// leave its result unallocated and without a def site. The instruction is
    /// tagged with a synthetic original, since a builder-created op has no
    /// backing machine instruction.
    ///
    /// # Arguments
    ///
    /// * `op` - Effect-only operation to append.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation has definitions.
    pub fn emit_no_defs(&mut self, op: SsaOp<T>) -> Result<()> {
        if let Some(def) = op.defs().next() {
            return Err(Error::new(format!(
                "emit_no_defs received operation defining {def}"
            )));
        }
        self.push_instruction(T::synthetic_instruction(), op, None)
    }

    /// Emits an original-instruction-backed instruction that must not define variables.
    ///
    /// Like [`emit_no_defs`](Self::emit_no_defs), but preserves a caller-supplied
    /// original machine instruction and result type instead of substituting a
    /// synthetic one. Frontends use this to keep the provenance of a lifted
    /// effect-only instruction attached to its SSA form.
    ///
    /// # Arguments
    ///
    /// * `original` - Backing machine instruction to record as provenance.
    /// * `result_type` - Result type carried by the instruction, if any.
    /// * `op` - Effect-only operation to append.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation has definitions.
    pub fn emit_no_defs_with_original(
        &mut self,
        original: T::OriginalInstruction,
        result_type: Option<T::Type>,
        op: SsaOp<T>,
    ) -> Result<()> {
        if let Some(def) = op.defs().next() {
            return Err(Error::new(format!(
                "emit_no_defs_with_original received operation defining {def}"
            )));
        }
        self.push_instruction(original, op, result_type)
    }

    /// Emits an instruction with an already-registered set of definitions.
    ///
    /// Unlike the allocating `emit_*` methods, the variables this operation
    /// defines were created earlier (for example by another builder pass), so no
    /// fresh allocation happens here. What is still missing is their def site:
    /// this rebinds each defined variable's [`DefSite`] to the position the
    /// instruction is about to occupy, keeping the def site accurate now that the
    /// definition has a concrete home. A definition the function does not know
    /// about is rejected, since its def site could not be updated.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation whose definitions are already registered variables.
    ///
    /// # Errors
    ///
    /// Returns an error if any definition is not registered in the function.
    pub fn emit_existing_defs(&mut self, op: SsaOp<T>) -> Result<()> {
        let instr_idx = self.next_instruction_index()?;
        for def in op.defs() {
            let var = self
                .builder
                .function
                .variable_mut(def)
                .ok_or_else(|| Error::new(format!("definition {def} is not registered")))?;
            var.set_def_site(DefSite::instruction(self.block_idx, instr_idx));
        }
        self.push_instruction(T::synthetic_instruction(), op, None)
    }

    /// Emits a one-result instruction.
    ///
    /// This is the core value-producing primitive the scalar `emit_*` wrappers
    /// funnel through. The def site is computed from the block's current
    /// instruction count before anything is pushed, so the freshly allocated
    /// variable records a [`DefSite`] pointing at the very instruction it will
    /// name. The `build` closure then receives that variable id and returns the
    /// operation, which is cross-checked against the allocation so a closure that
    /// wires up the wrong destination is caught here rather than surfacing as a
    /// mismatched def later.
    ///
    /// # Arguments
    ///
    /// * `def` - Origin and type of the variable to define.
    /// * `build` - Closure that builds the operation from the allocated variable.
    ///
    /// # Returns
    ///
    /// The id of the newly defined variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the constructed operation does not define exactly the
    /// variable allocated for `def`.
    pub fn emit_def(
        &mut self,
        def: SsaDefSpec<T>,
        build: impl FnOnce(SsaVarId) -> SsaOp<T>,
    ) -> Result<SsaVarId> {
        let instr_idx = self.next_instruction_index()?;
        let dest = self
            .builder
            .allocate_var(def, DefSite::instruction(self.block_idx, instr_idx));
        let op = build(dest);
        self.require_defs(&op, &[dest])?;
        self.push_instruction(T::synthetic_instruction(), op, None)?;
        Ok(dest)
    }

    /// Emits a two-result instruction.
    ///
    /// The two-output counterpart to [`emit_def`](Self::emit_def), used by
    /// operations that produce a value plus a secondary output such as a flags
    /// register. Both variables are allocated against the same pending
    /// instruction index, so they share a def site, and the `build` closure must
    /// return an operation defining exactly those two in the order they were
    /// allocated.
    ///
    /// # Arguments
    ///
    /// * `first` - Origin and type of the first (primary) result.
    /// * `second` - Origin and type of the second result.
    /// * `build` - Closure that builds the operation from both allocated variables.
    ///
    /// # Returns
    ///
    /// The ids of the two newly defined variables in allocation order.
    ///
    /// # Errors
    ///
    /// Returns an error if the constructed operation does not define both
    /// allocated variables in order.
    pub fn emit_two_defs(
        &mut self,
        first: SsaDefSpec<T>,
        second: SsaDefSpec<T>,
        build: impl FnOnce(SsaVarId, SsaVarId) -> SsaOp<T>,
    ) -> Result<(SsaVarId, SsaVarId)> {
        let instr_idx = self.next_instruction_index()?;
        let first = self
            .builder
            .allocate_var(first, DefSite::instruction(self.block_idx, instr_idx));
        let second = self
            .builder
            .allocate_var(second, DefSite::instruction(self.block_idx, instr_idx));
        let op = build(first, second);
        self.require_defs(&op, &[first, second])?;
        self.push_instruction(T::synthetic_instruction(), op, None)?;
        Ok((first, second))
    }

    /// Emits an instruction with any number of results.
    ///
    /// The variadic form of [`emit_def`](Self::emit_def), for operations such as
    /// native-opaque or segmented vector loads that define an arbitrary tuple of
    /// results. Every entry of `defs` is allocated against the same pending
    /// instruction index and passed to `build` as a slice, and the returned
    /// operation must define exactly that slice in order.
    ///
    /// # Arguments
    ///
    /// * `defs` - Origin and type of each result, in definition order.
    /// * `build` - Closure that builds the operation from the allocated variables.
    ///
    /// # Returns
    ///
    /// The ids of the newly defined variables, in the order of `defs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the constructed operation definitions do not exactly
    /// match the variables allocated from `defs`.
    pub fn emit_many_defs(
        &mut self,
        defs: &[SsaDefSpec<T>],
        build: impl FnOnce(&[SsaVarId]) -> SsaOp<T>,
    ) -> Result<Vec<SsaVarId>> {
        let instr_idx = self.next_instruction_index()?;
        let ids = defs
            .iter()
            .cloned()
            .map(|def| {
                self.builder
                    .allocate_var(def, DefSite::instruction(self.block_idx, instr_idx))
            })
            .collect::<Vec<_>>();
        let op = build(&ids);
        self.require_defs(&op, &ids)?;
        self.push_instruction(T::synthetic_instruction(), op, None)?;
        Ok(ids)
    }

    /// Emits a one-result instruction with original-instruction metadata.
    ///
    /// Behaves like [`emit_def`](Self::emit_def) but attaches a caller-supplied
    /// backing machine instruction and result type in place of the synthetic
    /// defaults, so a frontend lifting real code keeps each SSA definition tied
    /// to the instruction it came from.
    ///
    /// # Arguments
    ///
    /// * `def` - Origin and type of the variable to define.
    /// * `original` - Backing machine instruction to record as provenance.
    /// * `result_type` - Result type carried by the instruction, if any.
    /// * `build` - Closure that builds the operation from the allocated variable.
    ///
    /// # Returns
    ///
    /// The id of the newly defined variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the constructed operation does not define exactly the
    /// variable allocated for `def`.
    pub fn emit_def_with_original(
        &mut self,
        def: SsaDefSpec<T>,
        original: T::OriginalInstruction,
        result_type: Option<T::Type>,
        build: impl FnOnce(SsaVarId) -> SsaOp<T>,
    ) -> Result<SsaVarId> {
        let instr_idx = self.next_instruction_index()?;
        let dest = self
            .builder
            .allocate_var(def, DefSite::instruction(self.block_idx, instr_idx));
        let op = build(dest);
        self.require_defs(&op, &[dest])?;
        self.push_instruction(original, op, result_type)?;
        Ok(dest)
    }

    /// Emits a two-result instruction with original-instruction metadata.
    ///
    /// Behaves like [`emit_two_defs`](Self::emit_two_defs) but attaches a
    /// caller-supplied backing machine instruction and result type in place of
    /// the synthetic defaults, preserving provenance for a lifted two-output
    /// instruction.
    ///
    /// # Arguments
    ///
    /// * `first` - Origin and type of the first (primary) result.
    /// * `second` - Origin and type of the second result.
    /// * `original` - Backing machine instruction to record as provenance.
    /// * `result_type` - Result type carried by the instruction, if any.
    /// * `build` - Closure that builds the operation from both allocated variables.
    ///
    /// # Returns
    ///
    /// The ids of the two newly defined variables in allocation order.
    ///
    /// # Errors
    ///
    /// Returns an error if the constructed operation does not define both
    /// allocated variables in order.
    pub fn emit_two_defs_with_original(
        &mut self,
        first: SsaDefSpec<T>,
        second: SsaDefSpec<T>,
        original: T::OriginalInstruction,
        result_type: Option<T::Type>,
        build: impl FnOnce(SsaVarId, SsaVarId) -> SsaOp<T>,
    ) -> Result<(SsaVarId, SsaVarId)> {
        let instr_idx = self.next_instruction_index()?;
        let first = self
            .builder
            .allocate_var(first, DefSite::instruction(self.block_idx, instr_idx));
        let second = self
            .builder
            .allocate_var(second, DefSite::instruction(self.block_idx, instr_idx));
        let op = build(first, second);
        self.require_defs(&op, &[first, second])?;
        self.push_instruction(original, op, result_type)?;
        Ok((first, second))
    }

    /// Emits an instruction with any number of results and original-instruction metadata.
    ///
    /// Behaves like [`emit_many_defs`](Self::emit_many_defs) but attaches a
    /// caller-supplied backing machine instruction and result type in place of
    /// the synthetic defaults, preserving provenance for a lifted multi-output
    /// instruction.
    ///
    /// # Arguments
    ///
    /// * `defs` - Origin and type of each result, in definition order.
    /// * `original` - Backing machine instruction to record as provenance.
    /// * `result_type` - Result type carried by the instruction, if any.
    /// * `build` - Closure that builds the operation from the allocated variables.
    ///
    /// # Returns
    ///
    /// The ids of the newly defined variables, in the order of `defs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the constructed operation definitions do not exactly
    /// match the variables allocated from `defs`.
    pub fn emit_many_defs_with_original(
        &mut self,
        defs: &[SsaDefSpec<T>],
        original: T::OriginalInstruction,
        result_type: Option<T::Type>,
        build: impl FnOnce(&[SsaVarId]) -> SsaOp<T>,
    ) -> Result<Vec<SsaVarId>> {
        let instr_idx = self.next_instruction_index()?;
        let ids = defs
            .iter()
            .cloned()
            .map(|def| {
                self.builder
                    .allocate_var(def, DefSite::instruction(self.block_idx, instr_idx))
            })
            .collect::<Vec<_>>();
        let op = build(&ids);
        self.require_defs(&op, &ids)?;
        self.push_instruction(original, op, result_type)?;
        Ok(ids)
    }

    /// Creates a phi node with incoming operands.
    ///
    /// Convenience over [`empty_phi`](Self::empty_phi) for the common case where
    /// every incoming value is already defined: it creates the phi and then
    /// attaches each `(predecessor, value)` operand. Use `empty_phi` plus
    /// [`add_phi_operand`](SsaFunctionBuilder::add_phi_operand) when an operand
    /// arrives on a backedge and is not yet available at phi-creation time.
    ///
    /// # Arguments
    ///
    /// * `def` - Origin and type of the phi result variable.
    /// * `operands` - `(predecessor block, incoming value)` pairs to attach.
    ///
    /// # Returns
    ///
    /// The id of the phi result variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn phi(
        &mut self,
        def: SsaDefSpec<T>,
        operands: impl IntoIterator<Item = (usize, SsaVarId)>,
    ) -> Result<SsaVarId> {
        let result = self.empty_phi(def)?;
        for (predecessor, value) in operands {
            self.builder
                .add_phi_operand(self.block_idx, result, predecessor, value)?;
        }
        Ok(result)
    }

    /// Creates a phi node without operands for later backedge completion.
    ///
    /// A loop header's phi needs an operand for the value flowing in along the
    /// backedge, but that value is defined inside the loop body and so does not
    /// exist when the header is first built. This allocates the phi result now -
    /// with a phi def site rather than an instruction one - so the value can be
    /// referenced immediately, leaving the operands to be filled in with
    /// [`add_phi_operand`](SsaFunctionBuilder::add_phi_operand) once the
    /// predecessors have been built.
    ///
    /// # Arguments
    ///
    /// * `def` - Origin and type of the phi result variable.
    ///
    /// # Returns
    ///
    /// The id of the operand-less phi result variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn empty_phi(&mut self, def: SsaDefSpec<T>) -> Result<SsaVarId> {
        let result = self
            .builder
            .allocate_var(def.clone(), DefSite::phi(self.block_idx));
        let phi = PhiNode::new(result, def.origin);
        self.block_mut()?.add_phi(phi);
        Ok(result)
    }

    /// Emits a constant operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn const_value(&mut self, def: SsaDefSpec<T>, value: ConstValue<T>) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Const { dest, value })
    }

    /// Emits an `i32` constant.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn const_i32(&mut self, def: SsaDefSpec<T>, value: i32) -> Result<SsaVarId> {
        self.const_value(def, ConstValue::I32(value))
    }

    /// Emits an `i64` constant.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn const_i64(&mut self, def: SsaDefSpec<T>, value: i64) -> Result<SsaVarId> {
        self.const_value(def, ConstValue::I64(value))
    }

    /// Emits a boolean constant.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn const_bool(&mut self, def: SsaDefSpec<T>, value: bool) -> Result<SsaVarId> {
        self.const_value(def, ConstValue::from_bool(value))
    }

    /// Emits a copy operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn copy(&mut self, def: SsaDefSpec<T>, src: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Copy { dest, src })
    }

    /// Emits a scalar binary operation selected by [`BinaryOpKind`].
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn binary_op(
        &mut self,
        def: SsaDefSpec<T>,
        kind: BinaryOpKind,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| match kind {
            BinaryOpKind::Add => SsaOp::Add {
                dest,
                left,
                right,
                flags: None,
            },
            BinaryOpKind::AddOvf => SsaOp::AddOvf {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            BinaryOpKind::Sub => SsaOp::Sub {
                dest,
                left,
                right,
                flags: None,
            },
            BinaryOpKind::SubOvf => SsaOp::SubOvf {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            BinaryOpKind::Mul => SsaOp::Mul {
                dest,
                left,
                right,
                flags: None,
            },
            BinaryOpKind::MulOvf => SsaOp::MulOvf {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            BinaryOpKind::Div => SsaOp::Div {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            BinaryOpKind::Rem => SsaOp::Rem {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            BinaryOpKind::And => SsaOp::And {
                dest,
                left,
                right,
                flags: None,
            },
            BinaryOpKind::Or => SsaOp::Or {
                dest,
                left,
                right,
                flags: None,
            },
            BinaryOpKind::Xor => SsaOp::Xor {
                dest,
                left,
                right,
                flags: None,
            },
            BinaryOpKind::Shl => SsaOp::Shl {
                dest,
                value: left,
                amount: right,
                flags: None,
            },
            BinaryOpKind::Shr => SsaOp::Shr {
                dest,
                value: left,
                amount: right,
                unsigned,
                flags: None,
            },
            BinaryOpKind::Ceq => SsaOp::Ceq {
                dest,
                left,
                right,
                signaling: false,
            },
            BinaryOpKind::Clt => SsaOp::Clt {
                dest,
                left,
                right,
                unsigned,
                signaling: false,
            },
            BinaryOpKind::Cgt => SsaOp::Cgt {
                dest,
                left,
                right,
                unsigned,
                signaling: false,
            },
            BinaryOpKind::Rol => SsaOp::Rol {
                dest,
                value: left,
                amount: right,
            },
            BinaryOpKind::Ror => SsaOp::Ror {
                dest,
                value: left,
                amount: right,
            },
        })
    }

    /// Emits a scalar binary operation with a secondary flags output.
    ///
    /// # Errors
    ///
    /// Returns an error when `kind` does not have a flags-producing SSA form or
    /// when instruction insertion fails.
    pub fn binary_op_with_flags(
        &mut self,
        value_def: SsaDefSpec<T>,
        flags_def: SsaDefSpec<T>,
        kind: BinaryOpKind,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<(SsaVarId, SsaVarId)> {
        match kind {
            BinaryOpKind::Add
            | BinaryOpKind::AddOvf
            | BinaryOpKind::Sub
            | BinaryOpKind::SubOvf
            | BinaryOpKind::Mul
            | BinaryOpKind::MulOvf
            | BinaryOpKind::Div
            | BinaryOpKind::Rem
            | BinaryOpKind::And
            | BinaryOpKind::Or
            | BinaryOpKind::Xor
            | BinaryOpKind::Shl
            | BinaryOpKind::Shr => self.emit_two_defs(value_def, flags_def, |dest, flags| {
                let flags = Some(flags);
                match kind {
                    BinaryOpKind::Add => SsaOp::Add {
                        dest,
                        left,
                        right,
                        flags,
                    },
                    BinaryOpKind::AddOvf => SsaOp::AddOvf {
                        dest,
                        left,
                        right,
                        unsigned,
                        flags,
                    },
                    BinaryOpKind::Sub => SsaOp::Sub {
                        dest,
                        left,
                        right,
                        flags,
                    },
                    BinaryOpKind::SubOvf => SsaOp::SubOvf {
                        dest,
                        left,
                        right,
                        unsigned,
                        flags,
                    },
                    BinaryOpKind::Mul => SsaOp::Mul {
                        dest,
                        left,
                        right,
                        flags,
                    },
                    BinaryOpKind::MulOvf => SsaOp::MulOvf {
                        dest,
                        left,
                        right,
                        unsigned,
                        flags,
                    },
                    BinaryOpKind::Div => SsaOp::Div {
                        dest,
                        left,
                        right,
                        unsigned,
                        flags,
                    },
                    BinaryOpKind::Rem => SsaOp::Rem {
                        dest,
                        left,
                        right,
                        unsigned,
                        flags,
                    },
                    BinaryOpKind::And => SsaOp::And {
                        dest,
                        left,
                        right,
                        flags,
                    },
                    BinaryOpKind::Or => SsaOp::Or {
                        dest,
                        left,
                        right,
                        flags,
                    },
                    BinaryOpKind::Xor => SsaOp::Xor {
                        dest,
                        left,
                        right,
                        flags,
                    },
                    BinaryOpKind::Shl => SsaOp::Shl {
                        dest,
                        value: left,
                        amount: right,
                        flags,
                    },
                    BinaryOpKind::Shr => SsaOp::Shr {
                        dest,
                        value: left,
                        amount: right,
                        unsigned,
                        flags,
                    },
                    BinaryOpKind::Ceq
                    | BinaryOpKind::Clt
                    | BinaryOpKind::Cgt
                    | BinaryOpKind::Rol
                    | BinaryOpKind::Ror => unreachable!("filtered above"),
                }
            }),
            BinaryOpKind::Ceq
            | BinaryOpKind::Clt
            | BinaryOpKind::Cgt
            | BinaryOpKind::Rol
            | BinaryOpKind::Ror => Err(Error::new(format!(
                "{kind} does not have a flags-producing SSA form"
            ))),
        }
    }

    /// Emits a scalar unary operation selected by [`UnaryOpKind`].
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn unary_op(
        &mut self,
        def: SsaDefSpec<T>,
        kind: UnaryOpKind,
        operand: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| match kind {
            UnaryOpKind::Neg => SsaOp::Neg {
                dest,
                operand,
                flags: None,
            },
            UnaryOpKind::Not => SsaOp::Not {
                dest,
                operand,
                flags: None,
            },
            UnaryOpKind::Ckfinite => SsaOp::Ckfinite { dest, operand },
            UnaryOpKind::BSwap => SsaOp::BSwap { dest, src: operand },
            UnaryOpKind::BRev => SsaOp::BRev { dest, src: operand },
            UnaryOpKind::BitScanForward => SsaOp::BitScanForward { dest, src: operand },
            UnaryOpKind::BitScanReverse => SsaOp::BitScanReverse { dest, src: operand },
            UnaryOpKind::Popcount => SsaOp::Popcount { dest, src: operand },
            UnaryOpKind::Parity => SsaOp::Parity { dest, src: operand },
        })
    }

    /// Emits a scalar unary operation with a secondary flags output.
    ///
    /// # Errors
    ///
    /// Returns an error when `kind` does not have a flags-producing SSA form or
    /// when instruction insertion fails.
    pub fn unary_op_with_flags(
        &mut self,
        value_def: SsaDefSpec<T>,
        flags_def: SsaDefSpec<T>,
        kind: UnaryOpKind,
        operand: SsaVarId,
    ) -> Result<(SsaVarId, SsaVarId)> {
        match kind {
            UnaryOpKind::Neg | UnaryOpKind::Not => {
                self.emit_two_defs(value_def, flags_def, |dest, flags| match kind {
                    UnaryOpKind::Neg => SsaOp::Neg {
                        dest,
                        operand,
                        flags: Some(flags),
                    },
                    UnaryOpKind::Not => SsaOp::Not {
                        dest,
                        operand,
                        flags: Some(flags),
                    },
                    UnaryOpKind::Ckfinite
                    | UnaryOpKind::BSwap
                    | UnaryOpKind::BRev
                    | UnaryOpKind::BitScanForward
                    | UnaryOpKind::BitScanReverse
                    | UnaryOpKind::Popcount
                    | UnaryOpKind::Parity => unreachable!("filtered above"),
                })
            }
            UnaryOpKind::Ckfinite
            | UnaryOpKind::BSwap
            | UnaryOpKind::BRev
            | UnaryOpKind::BitScanForward
            | UnaryOpKind::BitScanReverse
            | UnaryOpKind::Popcount
            | UnaryOpKind::Parity => Err(Error::new(format!(
                "{kind} does not have a flags-producing SSA form"
            ))),
        }
    }

    /// Emits an addition operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn add(&mut self, def: SsaDefSpec<T>, left: SsaVarId, right: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Add {
            dest,
            left,
            right,
            flags: None,
        })
    }

    /// Emits an addition operation with a flags output.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn add_with_flags(
        &mut self,
        value_def: SsaDefSpec<T>,
        flags_def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
    ) -> Result<(SsaVarId, SsaVarId)> {
        self.emit_two_defs(value_def, flags_def, |dest, flags| SsaOp::Add {
            dest,
            left,
            right,
            flags: Some(flags),
        })
    }

    /// Emits a checked addition operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn add_ovf(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::AddOvf {
            dest,
            left,
            right,
            unsigned,
            flags: None,
        })
    }

    /// Emits a subtraction operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn sub(&mut self, def: SsaDefSpec<T>, left: SsaVarId, right: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Sub {
            dest,
            left,
            right,
            flags: None,
        })
    }

    /// Emits a checked subtraction operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn sub_ovf(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::SubOvf {
            dest,
            left,
            right,
            unsigned,
            flags: None,
        })
    }

    /// Emits a multiplication operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn mul(&mut self, def: SsaDefSpec<T>, left: SsaVarId, right: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Mul {
            dest,
            left,
            right,
            flags: None,
        })
    }

    /// Emits a checked multiplication operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn mul_ovf(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::MulOvf {
            dest,
            left,
            right,
            unsigned,
            flags: None,
        })
    }

    /// Emits a division operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn div(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Div {
            dest,
            left,
            right,
            unsigned,
            flags: None,
        })
    }

    /// Emits a remainder operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn rem(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Rem {
            dest,
            left,
            right,
            unsigned,
            flags: None,
        })
    }

    /// Emits a bitwise-and operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn and(&mut self, def: SsaDefSpec<T>, left: SsaVarId, right: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::And {
            dest,
            left,
            right,
            flags: None,
        })
    }

    /// Emits a bitwise-or operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn or(&mut self, def: SsaDefSpec<T>, left: SsaVarId, right: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Or {
            dest,
            left,
            right,
            flags: None,
        })
    }

    /// Emits a bitwise-xor operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn xor(&mut self, def: SsaDefSpec<T>, left: SsaVarId, right: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Xor {
            dest,
            left,
            right,
            flags: None,
        })
    }

    /// Emits a negation operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn neg(&mut self, def: SsaDefSpec<T>, operand: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Neg {
            dest,
            operand,
            flags: None,
        })
    }

    /// Emits a bitwise-not operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn not(&mut self, def: SsaDefSpec<T>, operand: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Not {
            dest,
            operand,
            flags: None,
        })
    }

    /// Emits a shift-left operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn shl(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        amount: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Shl {
            dest,
            value,
            amount,
            flags: None,
        })
    }

    /// Emits a shift-right operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn shr(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        amount: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Shr {
            dest,
            value,
            amount,
            unsigned,
            flags: None,
        })
    }

    /// Emits a rotate-left operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn rol(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        amount: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Rol {
            dest,
            value,
            amount,
        })
    }

    /// Emits a rotate-right operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn ror(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        amount: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Ror {
            dest,
            value,
            amount,
        })
    }

    /// Emits a byte-swap operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn bswap(&mut self, def: SsaDefSpec<T>, src: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BSwap { dest, src })
    }

    /// Emits a bit-reverse operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn brev(&mut self, def: SsaDefSpec<T>, src: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BRev { dest, src })
    }

    /// Emits a bit-scan-forward operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn bit_scan_forward(&mut self, def: SsaDefSpec<T>, src: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BitScanForward { dest, src })
    }

    /// Emits a bit-scan-reverse operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn bit_scan_reverse(&mut self, def: SsaDefSpec<T>, src: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BitScanReverse { dest, src })
    }

    /// Emits a population-count operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn popcount(&mut self, def: SsaDefSpec<T>, src: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Popcount { dest, src })
    }

    /// Emits a parity operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn parity(&mut self, def: SsaDefSpec<T>, src: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Parity { dest, src })
    }

    /// Emits an equality comparison.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn ceq(&mut self, def: SsaDefSpec<T>, left: SsaVarId, right: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Ceq {
            dest,
            left,
            right,
            signaling: false,
        })
    }

    /// Emits a less-than comparison.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn clt(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Clt {
            dest,
            left,
            right,
            unsigned,
            signaling: false,
        })
    }

    /// Emits a greater-than comparison.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn cgt(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Cgt {
            dest,
            left,
            right,
            unsigned,
            signaling: false,
        })
    }

    /// Emits a boolean-and operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn bool_and(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BoolAnd { dest, left, right })
    }

    /// Emits a boolean-or operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn bool_or(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BoolOr { dest, left, right })
    }

    /// Emits a boolean-xor operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn bool_xor(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BoolXor { dest, left, right })
    }

    /// Emits a boolean-not operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn bool_not(&mut self, def: SsaDefSpec<T>, value: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::BoolNot { dest, value })
    }

    /// Emits an integer→integer conversion operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn conv(
        &mut self,
        def: SsaDefSpec<T>,
        operand: SsaVarId,
        target: T::Type,
        overflow_check: bool,
        unsigned: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::IntConv {
            dest,
            operand,
            target,
            overflow_check,
            unsigned,
        })
    }

    /// Emits a select operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn select(
        &mut self,
        def: SsaDefSpec<T>,
        condition: SsaVarId,
        true_val: SsaVarId,
        false_val: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Select {
            dest,
            condition,
            true_val,
            false_val,
        })
    }

    /// Emits a read-flags operation.
    ///
    /// The mask must name only flags [`FlagsMask::NAMED`] defines. A bit
    /// outside that table selects a flag nothing downstream can interpret, and
    /// refusing it here is what keeps such a mask out of the IR in the first
    /// place - the verifier's matching check exists for the masks that arrive
    /// from a persisted blob rather than from this builder.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask` names an undefined flag bit, or if
    /// instruction insertion fails.
    pub fn read_flags(
        &mut self,
        def: SsaDefSpec<T>,
        flags: SsaVarId,
        mask: FlagsMask,
    ) -> Result<SsaVarId> {
        if !mask.is_valid() {
            return Err(Error::new(format!(
                "flag mask {mask} selects bits that name no defined flag"
            )));
        }
        self.emit_def(def, |dest| SsaOp::ReadFlags { dest, flags, mask })
    }

    /// Emits a load-argument operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_arg(&mut self, def: SsaDefSpec<T>, arg_index: u16) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadArg { dest, arg_index })
    }

    /// Emits a load-local operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_local(&mut self, def: SsaDefSpec<T>, local_index: u16) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadLocal { dest, local_index })
    }

    /// Emits a load-argument-address operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_arg_addr(&mut self, def: SsaDefSpec<T>, arg_index: u16) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadArgAddr { dest, arg_index })
    }

    /// Emits a load-local-address operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_local_addr(&mut self, def: SsaDefSpec<T>, local_index: u16) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadLocalAddr { dest, local_index })
    }

    /// Emits a store-local operation as a tracked pop-like use.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn pop(&mut self, value: SsaVarId) -> Result<()> {
        self.emit_no_defs(SsaOp::Pop { value })
    }

    /// Emits a call of `callee` with a result.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn call(
        &mut self,
        def: SsaDefSpec<T>,
        callee: Callee<T>,
        args: Vec<SsaVarId>,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| CallData::new(Some(dest), callee, args).into())
    }

    /// Emits a call of `callee` without a result.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn call_void(&mut self, callee: Callee<T>, args: Vec<SsaVarId>) -> Result<()> {
        self.emit_no_defs(CallData::new(None, callee, args).into())
    }

    /// Emits a load-field operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_field(
        &mut self,
        def: SsaDefSpec<T>,
        object: SsaVarId,
        field: T::FieldRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadField {
            access: AccessQualifiers::FLAT,
            dest,
            object,
            field,
        })
    }

    /// Emits a store-field operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn store_field(
        &mut self,
        object: SsaVarId,
        field: T::FieldRef,
        value: SsaVarId,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object,
            field,
            value,
        })
    }

    /// Emits a load-static-field operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_static_field(
        &mut self,
        def: SsaDefSpec<T>,
        field: T::FieldRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadStaticField {
            access: AccessQualifiers::FLAT,
            dest,
            field,
        })
    }

    /// Emits a store-static-field operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn store_static_field(&mut self, field: T::FieldRef, value: SsaVarId) -> Result<()> {
        self.emit_no_defs(SsaOp::StoreStaticField {
            access: AccessQualifiers::FLAT,
            field,
            value,
        })
    }

    /// Emits a load-field-address operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_field_addr(
        &mut self,
        def: SsaDefSpec<T>,
        object: SsaVarId,
        field: T::FieldRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadFieldAddr {
            dest,
            object,
            field,
        })
    }

    /// Emits a load-static-field-address operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_static_field_addr(
        &mut self,
        def: SsaDefSpec<T>,
        field: T::FieldRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadStaticFieldAddr { dest, field })
    }

    /// Emits a load-element operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_element(
        &mut self,
        def: SsaDefSpec<T>,
        array: SsaVarId,
        index: SsaVarId,
        elem_type: T::Type,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadElement {
            dest,
            array,
            index,
            elem_type,
        })
    }

    /// Emits a store-element operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn store_element(
        &mut self,
        array: SsaVarId,
        index: SsaVarId,
        value: SsaVarId,
        elem_type: T::Type,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::StoreElement {
            array,
            index,
            value,
            elem_type,
        })
    }

    /// Emits a load-element-address operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_element_addr(
        &mut self,
        def: SsaDefSpec<T>,
        array: SsaVarId,
        index: SsaVarId,
        elem_type: T::TypeRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadElementAddr {
            readonly: false,
            dest,
            array,
            index,
            elem_type,
        })
    }

    /// Emits an array-length operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn array_length(&mut self, def: SsaDefSpec<T>, array: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::ArrayLength { dest, array })
    }

    /// Emits an indirect load operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_indirect(
        &mut self,
        def: SsaDefSpec<T>,
        addr: SsaVarId,
        value_type: T::Type,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadIndirect {
            dest,
            addr,
            value_type,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits an indirect store operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn store_indirect(
        &mut self,
        addr: SsaVarId,
        value: SsaVarId,
        value_type: T::Type,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::StoreIndirect {
            addr,
            value,
            value_type,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a new-object operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn new_obj(
        &mut self,
        def: SsaDefSpec<T>,
        ctor: T::MethodRef,
        args: Vec<SsaVarId>,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::NewObj { dest, ctor, args })
    }

    /// Emits a new-array operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn new_arr(
        &mut self,
        def: SsaDefSpec<T>,
        elem_type: T::TypeRef,
        length: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::NewArr {
            dest,
            elem_type,
            length,
        })
    }

    /// Emits a load-function-pointer operation: of `method`, or of
    /// `constraint`'s implementation of a static virtual `method`.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_function_ptr(
        &mut self,
        def: SsaDefSpec<T>,
        method: T::MethodRef,
        constraint: Option<T::TypeRef>,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadFunctionPtr {
            dest,
            method,
            constraint,
        })
    }

    /// Emits a load-virtual-function-pointer operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_virt_function_ptr(
        &mut self,
        def: SsaDefSpec<T>,
        object: SsaVarId,
        method: T::MethodRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadVirtFunctionPtr {
            dest,
            object,
            method,
        })
    }

    /// Emits a cast-class operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn cast_class(
        &mut self,
        def: SsaDefSpec<T>,
        object: SsaVarId,
        target_type: T::TypeRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::CastClass {
            dest,
            object,
            target_type,
        })
    }

    /// Emits an is-instance operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn is_inst(
        &mut self,
        def: SsaDefSpec<T>,
        object: SsaVarId,
        target_type: T::TypeRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::IsInst {
            dest,
            object,
            target_type,
        })
    }

    /// Emits a box operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn box_value(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        value_type: T::TypeRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Box {
            dest,
            value,
            value_type,
        })
    }

    /// Emits an unbox operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn unbox(
        &mut self,
        def: SsaDefSpec<T>,
        object: SsaVarId,
        value_type: T::TypeRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Unbox {
            dest,
            object,
            value_type,
        })
    }

    /// Emits an unbox-any operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn unbox_any(
        &mut self,
        def: SsaDefSpec<T>,
        object: SsaVarId,
        value_type: T::TypeRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::UnboxAny {
            dest,
            object,
            value_type,
        })
    }

    /// Emits a size-of operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn size_of(&mut self, def: SsaDefSpec<T>, value_type: T::TypeRef) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::SizeOf { dest, value_type })
    }

    /// Emits a load-token operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_token(&mut self, def: SsaDefSpec<T>, token: T::TypeRef) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadToken { dest, token })
    }

    /// Emits an init-block operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn init_blk(
        &mut self,
        dest_addr: SsaVarId,
        value: SsaVarId,
        size: SsaVarId,
        backward: Option<SsaVarId>,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::InitBlk {
            dest_addr,
            value,
            size,
            backward,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a copy-block operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn copy_blk(
        &mut self,
        dest_addr: SsaVarId,
        src_addr: SsaVarId,
        size: SsaVarId,
        backward: Option<SsaVarId>,
    ) -> Result<()> {
        self.copy_blk_elements(dest_addr, src_addr, size, 1, backward)
    }

    /// Emits a streamed block copy with an explicit complete-element read width.
    ///
    /// Addresses name the first elements. `size` is a byte count divisible by
    /// the nonzero `element_bytes`; each element is read before any of its writes.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn copy_blk_elements(
        &mut self,
        dest_addr: SsaVarId,
        src_addr: SsaVarId,
        size: SsaVarId,
        element_bytes: u32,
        backward: Option<SsaVarId>,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::CopyBlk {
            dest_addr,
            src_addr,
            size,
            element_bytes,
            backward,
            dest_access: AccessQualifiers::FLAT,
            src_access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a move-block operation: a copy whose regions may overlap.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn move_blk(
        &mut self,
        dest_addr: SsaVarId,
        src_addr: SsaVarId,
        size: SsaVarId,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::MoveBlk {
            dest_addr,
            src_addr,
            size,
            dest_access: AccessQualifiers::FLAT,
            src_access: AccessQualifiers::FLAT,
        })
    }

    /// Emits an init-object operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn init_obj(&mut self, dest_addr: SsaVarId, value_type: T::TypeRef) -> Result<()> {
        self.emit_no_defs(SsaOp::InitObj {
            dest_addr,
            value_type,
        })
    }

    /// Emits a copy-object operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn copy_obj(
        &mut self,
        dest_addr: SsaVarId,
        src_addr: SsaVarId,
        value_type: T::TypeRef,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::CopyObj {
            dest_addr,
            src_addr,
            value_type,
        })
    }

    /// Emits a load-object operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn load_obj(
        &mut self,
        def: SsaDefSpec<T>,
        src_addr: SsaVarId,
        value_type: T::TypeRef,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LoadObj {
            access: AccessQualifiers::FLAT,
            dest,
            src_addr,
            value_type,
        })
    }

    /// Emits a store-object operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn store_obj(
        &mut self,
        dest_addr: SsaVarId,
        value: SsaVarId,
        value_type: T::TypeRef,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::StoreObj {
            access: AccessQualifiers::FLAT,
            dest_addr,
            value,
            value_type,
        })
    }

    /// Emits an allocation of `size` bytes of the frame, holding zero where
    /// `zeroed`.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn local_alloc(
        &mut self,
        def: SsaDefSpec<T>,
        size: SsaVarId,
        zeroed: bool,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::LocalAlloc { dest, size, zeroed })
    }

    /// Emits a check-finite operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn ckfinite(&mut self, def: SsaDefSpec<T>, operand: SsaVarId) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::Ckfinite { dest, operand })
    }

    /// Emits a floating-point classification operation (`fclass`/`class.fmt`).
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn fp_classify(
        &mut self,
        def: SsaDefSpec<T>,
        operand: SsaVarId,
        layout: FloatClassLayout,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::FpClassify {
            dest,
            operand,
            layout,
        })
    }

    /// Emits an atomic exchange operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn atomic_exchange(
        &mut self,
        def: SsaDefSpec<T>,
        addr: SsaVarId,
        value: SsaVarId,
        ordering: AtomicOrdering,
        width: AtomicAccessWidth,
        access: AccessQualifiers,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::AtomicExchange {
            dest,
            addr,
            value,
            ordering,
            width,
            access,
        })
    }

    /// Emits an atomic compare-exchange operation with old and success results.
    ///
    /// A compare-exchange always yields the previous memory value and may
    /// additionally yield a boolean success flag. Whether that second result
    /// exists changes the operation's definition count, so the spec's
    /// `success_def` selects the arity: present routes through
    /// [`emit_two_defs`](Self::emit_two_defs) and returns `Some` success id,
    /// absent routes through [`emit_def`](Self::emit_def) and returns `None`.
    ///
    /// # Arguments
    ///
    /// * `spec` - Operands, orderings, width, and result definitions for the
    ///   exchange; its `success_def` decides whether a success flag is produced.
    ///
    /// # Returns
    ///
    /// The old-value variable and, when a success definition was requested, the
    /// success-flag variable.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn atomic_cmp_xchg(
        &mut self,
        spec: AtomicCmpXchgSpec<T>,
    ) -> Result<(SsaVarId, Option<SsaVarId>)> {
        let AtomicCmpXchgSpec {
            old_def,
            success_def,
            addr,
            expected,
            desired,
            success_ordering,
            failure_ordering,
            width,
            weak,
            access,
        } = spec;
        if let Some(success_def) = success_def {
            let (old, success) =
                self.emit_two_defs(old_def, success_def, |old, success| SsaOp::AtomicCmpXchg {
                    old,
                    success: Some(success),
                    addr,
                    expected,
                    desired,
                    success_ordering,
                    failure_ordering,
                    width,
                    weak,
                    access,
                })?;
            Ok((old, Some(success)))
        } else {
            let old = self.emit_def(old_def, |old| SsaOp::AtomicCmpXchg {
                old,
                success: None,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
                width,
                weak,
                access,
            })?;
            Ok((old, None))
        }
    }

    /// Emits a wide multiply operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn wide_mul(
        &mut self,
        low_def: SsaDefSpec<T>,
        high_def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        unsigned: bool,
    ) -> Result<(SsaVarId, SsaVarId)> {
        self.emit_two_defs(low_def, high_def, |low, high| SsaOp::WideMul {
            low,
            high,
            left,
            right,
            unsigned,
        })
    }

    /// Emits a wide divide operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn wide_div(
        &mut self,
        quotient_def: SsaDefSpec<T>,
        remainder_def: SsaDefSpec<T>,
        high: SsaVarId,
        low: SsaVarId,
        divisor: SsaVarId,
        unsigned: bool,
    ) -> Result<(SsaVarId, SsaVarId)> {
        self.emit_two_defs(quotient_def, remainder_def, |quotient, remainder| {
            SsaOp::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                unsigned,
            }
        })
    }

    /// Emits a vector unary operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_unary(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        kind: VectorUnaryKind,
        element: VectorElement,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorUnary {
            dest,
            value,
            kind,
            element,
            environment: None,
        })
    }

    /// Emits a vector binary operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_binary(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        kind: VectorBinaryKind,
        element: VectorElement,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorBinary {
            dest,
            left,
            right,
            kind,
            element,
            environment: None,
        })
    }

    /// Emits a vector ternary operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_ternary(
        &mut self,
        def: SsaDefSpec<T>,
        first: SsaVarId,
        second: SsaVarId,
        third: SsaVarId,
        kind: VectorTernaryKind,
        element: VectorElement,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorTernary {
            dest,
            first,
            second,
            third,
            kind,
            element,
            environment: None,
        })
    }

    /// Emits a vector compare operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_compare(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: SsaVarId,
        kind: VectorCompareKind,
        element: VectorElement,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorCompare {
            dest,
            left,
            right,
            kind,
            element,
            signaling: false,
        })
    }

    /// Emits a vector load operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_load(
        &mut self,
        def: SsaDefSpec<T>,
        addr: SsaVarId,
        vector_type: T::Type,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorLoad {
            dest,
            addr,
            vector_type,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a masked vector load operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_masked_load(
        &mut self,
        def: SsaDefSpec<T>,
        addr: SsaVarId,
        mask: SsaVarId,
        passthrough: Option<SsaVarId>,
        vector_type: T::Type,
        mode: VectorMaskMode,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorMaskedLoad {
            dest,
            addr,
            mask,
            passthrough,
            vector_type,
            mode,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a vector store operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_store(
        &mut self,
        addr: SsaVarId,
        value: SsaVarId,
        vector_type: T::Type,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::VectorStore {
            addr,
            value,
            vector_type,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a masked vector store operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_masked_store(
        &mut self,
        addr: SsaVarId,
        value: SsaVarId,
        mask: SsaVarId,
        vector_type: T::Type,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::VectorMaskedStore {
            addr,
            value,
            mask,
            vector_type,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a vector broadcast-load operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_broadcast_load(
        &mut self,
        def: SsaDefSpec<T>,
        addr: SsaVarId,
        vector_type: T::Type,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorBroadcastLoad {
            dest,
            addr,
            vector_type,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a vector gather operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_gather(&mut self, spec: VectorGatherSpec<T>) -> Result<SsaVarId> {
        let VectorGatherSpec {
            def,
            base,
            indices,
            mask,
            passthrough,
            vector_type,
            mode,
        } = spec;
        self.emit_def(def, |dest| SsaOp::VectorGather {
            dest,
            base,
            indices,
            mask,
            passthrough,
            vector_type,
            mode,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a vector scatter operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_scatter(
        &mut self,
        base: SsaVarId,
        indices: SsaVarId,
        value: SsaVarId,
        mask: SsaVarId,
        vector_type: T::Type,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::VectorScatter {
            base,
            indices,
            value,
            mask,
            vector_type,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a vector splat operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_splat(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        vector_type: T::Type,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorSplat {
            dest,
            value,
            vector_type,
        })
    }

    /// Emits a vector extract operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_extract(
        &mut self,
        def: SsaDefSpec<T>,
        vector: SsaVarId,
        lane: u32,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorExtract { dest, vector, lane })
    }

    /// Emits a vector insert operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_insert(
        &mut self,
        def: SsaDefSpec<T>,
        vector: SsaVarId,
        lane: u32,
        value: SsaVarId,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorInsert {
            dest,
            vector,
            lane,
            value,
        })
    }

    /// Emits a vector shuffle selecting lanes of `left` and `right` by
    /// `selector`.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_shuffle(
        &mut self,
        def: SsaDefSpec<T>,
        left: SsaVarId,
        right: Option<SsaVarId>,
        selector: VectorShuffleSelector,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorShuffle {
            dest,
            left,
            right,
            selector,
        })
    }

    /// Emits a vector cast operation reading its source lanes as `from` and
    /// writing its destination lanes as `to`.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_cast(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        target_type: T::Type,
        from: VectorCastKind,
        to: VectorCastKind,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorCast {
            dest,
            value,
            target_type,
            from,
            to,
            rounding: None,
            saturate: false,
            environment: None,
            format: None,
        })
    }

    /// Emits a vector reinterpret operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_reinterpret(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        target_type: T::Type,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorReinterpret {
            dest,
            value,
            target_type,
        })
    }

    /// Emits a faulting vector load with an optional fault-status output.
    ///
    /// A first-faulting load always yields the loaded vector and may also yield
    /// a mask recording which lanes faulted. As with
    /// [`atomic_cmp_xchg`](Self::atomic_cmp_xchg), the optional second result
    /// changes the operation's definition count, so the spec's `fault_def`
    /// selects the arity: present routes through
    /// [`emit_two_defs`](Self::emit_two_defs) and returns `Some` fault mask,
    /// absent routes through [`emit_def`](Self::emit_def) and returns `None`.
    ///
    /// # Arguments
    ///
    /// * `spec` - Address, mask, passthrough, vector type, and modes for the
    ///   load; its `fault_def` decides whether a fault-status mask is produced.
    ///
    /// # Returns
    ///
    /// The loaded-vector variable and, when a fault definition was requested, the
    /// fault-status mask variable.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_faulting_load(
        &mut self,
        spec: VectorFaultingLoadSpec<T>,
    ) -> Result<(SsaVarId, Option<SsaVarId>)> {
        let VectorFaultingLoadSpec {
            dest_def,
            fault_def,
            addr,
            mask,
            passthrough,
            vector_type,
            fault_mode,
            mask_mode,
        } = spec;
        if let Some(fault_def) = fault_def {
            let (dest, fault) = self.emit_two_defs(dest_def, fault_def, |dest, fault| {
                SsaOp::VectorFaultingLoad {
                    dest,
                    fault: Some(fault),
                    addr,
                    mask,
                    passthrough,
                    vector_type,
                    fault_mode,
                    mask_mode,
                    access: AccessQualifiers::FLAT,
                }
            })?;
            Ok((dest, Some(fault)))
        } else {
            let dest = self.emit_def(dest_def, |dest| SsaOp::VectorFaultingLoad {
                dest,
                fault: None,
                addr,
                mask,
                passthrough,
                vector_type,
                fault_mode,
                mask_mode,
                access: AccessQualifiers::FLAT,
            })?;
            Ok((dest, None))
        }
    }

    /// Emits a vector segment load operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_segment_load(
        &mut self,
        defs: &[SsaDefSpec<T>],
        base: SsaVarId,
        mask: Option<SsaVarId>,
        vector_type: T::Type,
        segments: u32,
        layout: VectorSegmentLayout,
    ) -> Result<Vec<SsaVarId>> {
        self.emit_many_defs(defs, |dests| SsaOp::VectorSegmentLoad {
            dests: dests.to_vec(),
            base,
            mask,
            vector_type,
            segments,
            layout,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a vector segment store operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_segment_store(
        &mut self,
        base: SsaVarId,
        values: Vec<SsaVarId>,
        mask: Option<SsaVarId>,
        vector_type: T::Type,
        segments: u32,
        layout: VectorSegmentLayout,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::VectorSegmentStore {
            base,
            values,
            mask,
            vector_type,
            segments,
            layout,
            access: AccessQualifiers::FLAT,
        })
    }

    /// Emits a vector reduce operation over lanes described by `element`: of
    /// the lanes alone, in an order it does not state.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_reduce(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        kind: VectorReduceKind,
        element: VectorElement,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorReduce {
            dest,
            value,
            kind,
            element,
            initial: None,
            ordered: false,
            groups: None,
            environment: None,
        })
    }

    /// Emits a vector bitmask operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn vector_bitmask(
        &mut self,
        def: SsaDefSpec<T>,
        value: SsaVarId,
        kind: VectorBitmaskKind,
    ) -> Result<SsaVarId> {
        self.emit_def(def, |dest| SsaOp::VectorBitmask { dest, value, kind })
    }

    /// Emits an unconditional jump terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn jump(&mut self, target: usize) -> Result<()> {
        self.emit_no_defs(SsaOp::Jump { target })
    }

    /// Emits a conditional branch terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn branch(
        &mut self,
        condition: SsaVarId,
        true_target: usize,
        false_target: usize,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::Branch {
            condition,
            true_target,
            false_target,
        })
    }

    /// Emits a compare-and-branch terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn branch_cmp(
        &mut self,
        left: SsaVarId,
        right: SsaVarId,
        cmp: CmpKind,
        unsigned: bool,
        true_target: usize,
        false_target: usize,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned,
            true_target,
            false_target,
        })
    }

    /// Emits a flag-based branch terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn branch_flags(
        &mut self,
        flags: SsaVarId,
        condition: FlagCondition,
        true_target: usize,
        false_target: usize,
    ) -> Result<()> {
        self.emit_no_defs(SsaOp::BranchFlags {
            flags,
            condition,
            true_target,
            false_target,
        })
    }

    /// Emits a switch terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn switch(&mut self, value: SsaVarId, targets: Vec<usize>, default: usize) -> Result<()> {
        self.emit_no_defs(SsaOp::Switch {
            value,
            targets,
            default,
        })
    }

    /// Emits a return terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn ret(&mut self, value: Option<SsaVarId>) -> Result<()> {
        self.emit_no_defs(SsaOp::returns(value))
    }

    /// Emits a throw terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn throw(&mut self, exception: SsaVarId) -> Result<()> {
        self.emit_no_defs(SsaOp::Throw { exception })
    }

    /// Emits a rethrow terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn rethrow(&mut self) -> Result<()> {
        self.emit_no_defs(SsaOp::Rethrow)
    }

    /// Emits an end-finally terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn end_finally(&mut self) -> Result<()> {
        self.emit_no_defs(SsaOp::EndFinally)
    }

    /// Emits an end-filter terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn end_filter(&mut self, result: SsaVarId) -> Result<()> {
        self.emit_no_defs(SsaOp::EndFilter { result })
    }

    /// Emits an interrupt-return terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn interrupt_return(&mut self) -> Result<()> {
        self.emit_no_defs(SsaOp::InterruptReturn)
    }

    /// Emits a leave terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn leave(&mut self, target: usize) -> Result<()> {
        self.emit_no_defs(SsaOp::Leave { target })
    }

    /// Emits an unreachable terminator.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn unreachable(&mut self) -> Result<()> {
        self.emit_no_defs(SsaOp::Unreachable)
    }

    /// Emits a memory fence.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn fence(&mut self, kind: FenceKind) -> Result<()> {
        self.emit_no_defs(SsaOp::Fence { kind })
    }

    /// Emits a no-op instruction.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn nop(&mut self) -> Result<()> {
        self.emit_no_defs(SsaOp::Nop)
    }

    /// Emits a breakpoint instruction.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn break_(&mut self, op: BreakpointOp) -> Result<()> {
        self.emit_no_defs(SsaOp::Break(op))
    }

    /// Emits a lock-prefixed atomic read-modify-write operation.
    ///
    /// # Errors
    ///
    /// Returns an error if instruction insertion fails.
    pub fn atomic_lock_rmw(&mut self, spec: AtomicLockRmwSpec<T>) -> Result<SsaVarId> {
        let AtomicLockRmwSpec {
            def,
            addr,
            value,
            op,
            ordering,
            width,
            access,
        } = spec;
        self.emit_def(def, |dest| SsaOp::AtomicLockRmw {
            dest,
            addr,
            value,
            op,
            ordering,
            width,
            access,
        })
    }

    /// Returns the block being built, or an error if it no longer exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the block index is out of range.
    fn block_mut(&mut self) -> Result<&mut SsaBlock<T>> {
        self.builder
            .function
            .block_mut(self.block_idx)
            .ok_or_else(|| Error::new(format!("block {} does not exist", self.block_idx)))
    }

    /// Returns the index the next appended instruction will occupy.
    ///
    /// This is the block's current instruction count, taken before an emit
    /// pushes anything, so an `emit_*` method can stamp a variable's def site
    /// with the position its defining instruction is about to land at.
    ///
    /// # Errors
    ///
    /// Returns an error if the block index is out of range.
    fn next_instruction_index(&self) -> Result<usize> {
        self.builder
            .function
            .block(self.block_idx)
            .map(SsaBlock::instruction_count)
            .ok_or_else(|| Error::new(format!("block {} does not exist", self.block_idx)))
    }

    /// Wraps an operation in an instruction and appends it to the block.
    ///
    /// The shared tail of every emit path: it pairs the operation with its
    /// backing original (synthetic for builder-created ops) and result type,
    /// then appends the instruction. Def-site bookkeeping has already happened in
    /// the caller, so this only performs the append.
    ///
    /// # Arguments
    ///
    /// * `original` - Backing machine instruction recorded as provenance.
    /// * `op` - Operation to wrap.
    /// * `result_type` - Result type carried by the instruction, if any.
    ///
    /// # Errors
    ///
    /// Returns an error if the block index is out of range.
    fn push_instruction(
        &mut self,
        original: T::OriginalInstruction,
        op: SsaOp<T>,
        result_type: Option<T::Type>,
    ) -> Result<()> {
        let mut instruction = SsaInstruction::new(original, op);
        instruction.set_result_type(result_type);
        self.block_mut()?.add_instruction(instruction);
        Ok(())
    }

    /// Checks that an operation defines exactly the expected variables in order.
    ///
    /// The allocating emit paths hand a `build` closure the ids they reserved and
    /// trust it to wire them in as the operation's destinations. This verifies
    /// that trust: the operation's actual definitions must equal `expected`
    /// element for element, catching a closure that returns the wrong `SsaOp`
    /// shape or transposes its outputs before the mismatch can corrupt the
    /// function's def/use structure.
    ///
    /// # Arguments
    ///
    /// * `op` - Operation whose definitions are inspected.
    /// * `expected` - Variables the caller allocated, in order.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation's definitions do not match `expected`.
    fn require_defs(&self, op: &SsaOp<T>, expected: &[SsaVarId]) -> Result<()> {
        let actual = op.defs().collect::<Vec<_>>();
        if actual == expected {
            Ok(())
        } else {
            Err(Error::new(format!(
                "operation definitions {actual:?} do not match allocated definitions {expected:?}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        analysis::verifier::{SsaVerifier, VerifierError, VerifyLevel},
        ir::{
            BinaryOpKind, ConstValue, SsaDefSpec, SsaFunctionBuilder, UnaryOpKind,
            function::VectorFaultingLoadSpec,
            ops::{
                AccessQualifiers, AtomicAccessWidth, AtomicOrdering, FlagCondition, FlagsMask,
                NativeKindedData, SsaOp, SystemOpKind, VectorBinaryKind, VectorCompareKind,
                VectorElement, VectorFaultMode, VectorMaskMode, VectorSegmentLayout,
            },
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    fn i32_tmp() -> SsaDefSpec<MockTarget> {
        SsaDefSpec::tmp(MockType::I32)
    }

    fn ptr_tmp() -> SsaDefSpec<MockTarget> {
        SsaDefSpec::tmp(MockType::Ptr)
    }

    fn v4i32_tmp() -> SsaDefSpec<MockTarget> {
        SsaDefSpec::tmp(MockType::V4I32)
    }

    fn mask4_tmp() -> SsaDefSpec<MockTarget> {
        SsaDefSpec::tmp(MockType::Mask4)
    }

    fn v2f64_tmp() -> SsaDefSpec<MockTarget> {
        SsaDefSpec::tmp(MockType::V2F64)
    }

    /// A masked vector op whose lane count the target does not model as a
    /// distinct mask type must still verify. `MockTarget` maps only 4-lane
    /// masks (`Mask4`), so a 2-lane (`V2F64`) masked load's
    /// `vector_mask_descriptor_type` is `None` - meaning "target models no
    /// mask type for this shape", exactly like `VisusTarget` (which returns
    /// `None` for every shape and carries an AVX `vmaskmov` mask in a plain
    /// vector register). The verifier must skip the mask-shape check in that
    /// case rather than reject a concretely-typed mask operand. Regression for
    /// the ~127-fn x86_64 "mask operand is not compatible with vector lane
    /// count" false-positive class.
    #[test]
    fn masked_op_verifies_when_target_models_no_mask_type_for_lane_count() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |block| {
                let addr = block.const_value(ptr_tmp(), ConstValue::NativeUInt(0x2000))?;
                // The mask register carries a concrete 2-lane vector type - the
                // shape `MockTarget` does not model as a distinct mask type.
                let mask = block.vector_load(v2f64_tmp(), addr, MockType::V2F64)?;
                let loaded = block.vector_masked_load(
                    v2f64_tmp(),
                    addr,
                    mask,
                    None,
                    MockType::V2F64,
                    VectorMaskMode::Zero,
                )?;
                block.ret(Some(loaded))?;
                Ok(())
            })
            .unwrap();
        let function = builder.finish().unwrap();
        let errors = SsaVerifier::new(&function).verify(VerifyLevel::Standard);
        assert!(
            !errors.iter().any(|error| matches!(
                error,
                VerifierError::InvalidVectorOperation { reason, .. }
                    if reason.contains("mask operand is not compatible")
            )),
            "masked op must verify when the target models no mask type for the \
             lane count: {errors:?}"
        );
    }

    /// A straight-line block records each definition's site and recomputes its
    /// use count on finish.
    #[test]
    fn builds_scalar_block_with_registered_defs_and_uses() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |block| {
                let left = block.const_i32(i32_tmp(), 2)?;
                let right = block.const_i32(i32_tmp(), 40)?;
                let sum = block.add(i32_tmp(), left, right)?;
                block.ret(Some(sum))?;
                Ok(())
            })
            .unwrap();

        let ssa = builder.finish_verified(VerifyLevel::Full).unwrap();
        assert_eq!(ssa.block_count(), 1);
        assert_eq!(ssa.variable_count(), 3);
        assert_eq!(
            ssa.variable(SsaVarId::from_index(2)).unwrap().def_site(),
            DefSite::instruction(0, 2)
        );
        assert_eq!(
            ssa.variable(SsaVarId::from_index(2)).unwrap().uses().len(),
            1
        );
    }

    /// A diamond CFG with a merge-block phi verifies and tracks its uses.
    #[test]
    fn builds_diamond_with_phi_operands() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.ensure_block(3);

        let condition = builder
            .in_block(0, |block| {
                let condition = block.const_bool(i32_tmp(), true)?;
                block.branch(condition, 1, 2)?;
                Ok(condition)
            })
            .unwrap();
        let left = builder
            .in_block(1, |block| {
                let value = block.const_i32(i32_tmp(), 10)?;
                block.jump(3)?;
                Ok(value)
            })
            .unwrap();
        let right = builder
            .in_block(2, |block| {
                let value = block.const_i32(i32_tmp(), 20)?;
                block.jump(3)?;
                Ok(value)
            })
            .unwrap();

        builder
            .in_block(3, |block| {
                let merged = block.phi(i32_tmp(), [(1, left), (2, right)])?;
                block.ret(Some(merged))?;
                Ok(())
            })
            .unwrap();

        let ssa = builder.finish_verified(VerifyLevel::Standard).unwrap();
        assert_eq!(ssa.variable(condition).unwrap().uses().len(), 1);
        assert_eq!(ssa.block(3).unwrap().phi_count(), 1);
    }

    /// An empty phi at a loop header can be completed with a backedge operand
    /// after the body is built.
    #[test]
    fn supports_empty_phi_and_later_backedge_operand() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.ensure_block(2);

        let zero = builder
            .in_block(0, |block| {
                let zero = block.const_i32(i32_tmp(), 0)?;
                block.jump(1)?;
                Ok(zero)
            })
            .unwrap();

        let (counter, next) = builder
            .in_block(1, |block| {
                let counter =
                    block.empty_phi(SsaDefSpec::new(VariableOrigin::Local(0), MockType::I32))?;
                let one = block.const_i32(i32_tmp(), 1)?;
                let next = block.add(i32_tmp(), counter, one)?;
                let limit = block.const_i32(i32_tmp(), 3)?;
                let keep_going = block.clt(i32_tmp(), next, limit, false)?;
                block.branch(keep_going, 1, 2)?;
                Ok((counter, next))
            })
            .unwrap();
        builder.add_phi_operand(1, counter, 0, zero).unwrap();
        builder.add_phi_operand(1, counter, 1, next).unwrap();
        builder
            .in_block(2, |block| {
                block.ret(Some(counter))?;
                Ok(())
            })
            .unwrap();

        let ssa = builder.finish_verified(VerifyLevel::Standard).unwrap();
        assert_eq!(ssa.block(1).unwrap().phi_nodes()[0].operand_count(), 2);
    }

    /// Multi-output operations - flag-producing arithmetic, atomics, and
    /// native-opaque ops - build and verify with all their definitions tracked.
    #[test]
    fn supports_multi_output_flags_atomics_and_system_ops() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |block| {
                let addr = block.const_value(ptr_tmp(), ConstValue::NativeUInt(0x1000))?;
                let left = block.const_i32(i32_tmp(), 7)?;
                let right = block.const_i32(i32_tmp(), 5)?;
                let (sum, flags) = block.binary_op_with_flags(
                    i32_tmp(),
                    i32_tmp(),
                    BinaryOpKind::Add,
                    left,
                    right,
                    false,
                )?;
                let _negated = block.unary_op(i32_tmp(), UnaryOpKind::Neg, sum)?;
                let zero = block.read_flags(i32_tmp(), flags, FlagsMask::ZERO)?;
                let _old = block.atomic_exchange(
                    i32_tmp(),
                    addr,
                    sum,
                    AtomicOrdering::SeqCst,
                    AtomicAccessWidth::Bits32,
                    AccessQualifiers::FLAT,
                )?;
                let _ = block.emit_many_defs(&[i32_tmp(), i32_tmp()], |outputs| {
                    SsaOp::SystemOp(Box::new(NativeKindedData {
                        kind: SystemOpKind::CpuId,
                        mnemonic: "cpuid".into(),
                        metadata: None,
                        outputs: outputs.to_vec(),
                        inputs: vec![zero],
                    }))
                })?;
                block.branch_flags(flags, FlagCondition::Zero, 1, 1)?;
                Ok(())
            })
            .unwrap();
        builder
            .in_block(1, |block| {
                block.ret(None)?;
                Ok(())
            })
            .unwrap();

        let ssa = builder.finish_verified(VerifyLevel::Standard).unwrap();
        assert_eq!(ssa.variable_count(), 10);
    }

    /// The builder refuses a mask naming a flag the crate does not define.
    ///
    /// Enforcement at the point of construction is what keeps such a mask from
    /// entering the IR at all; the verifier's matching check covers the masks
    /// that arrive from a persisted blob rather than from here.
    #[test]
    fn read_flags_builder_rejects_an_invalid_mask() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let outcome = builder.in_block(0, |block| {
            let left = block.const_i32(i32_tmp(), 1)?;
            let right = block.const_i32(i32_tmp(), 2)?;
            let (_sum, flags) = block.binary_op_with_flags(
                i32_tmp(),
                i32_tmp(),
                BinaryOpKind::Add,
                left,
                right,
                false,
            )?;
            // Every defined bit is accepted.
            let _all = block.read_flags(i32_tmp(), flags, FlagsMask::ALL)?;
            // One bit past them is not.
            let _bad = block.read_flags(i32_tmp(), flags, FlagsMask::from_bits(1 << 15))?;
            block.ret(None)?;
            Ok(())
        });
        assert!(
            outcome.is_err(),
            "a mask with an undefined bit must not reach the IR"
        );
    }

    /// The vector emit surface - splat, load, binary, compare, faulting load,
    /// and segmented load/store - builds and verifies as a fixture.
    #[test]
    fn supports_vector_fixture_construction() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |block| {
                let addr = block.const_value(ptr_tmp(), ConstValue::NativeUInt(0x2000))?;
                let scalar = block.const_i32(i32_tmp(), 9)?;
                let vector = block.vector_splat(v4i32_tmp(), scalar, MockType::V4I32)?;
                let loaded = block.vector_load(v4i32_tmp(), addr, MockType::V4I32)?;
                let added = block.vector_binary(
                    v4i32_tmp(),
                    vector,
                    loaded,
                    VectorBinaryKind::Add,
                    VectorElement::default(),
                )?;
                let mask = block.vector_compare(
                    mask4_tmp(),
                    added,
                    loaded,
                    VectorCompareKind::Eq,
                    VectorElement::default(),
                )?;
                let (_faulting, fault) = block.vector_faulting_load(VectorFaultingLoadSpec {
                    dest_def: v4i32_tmp(),
                    fault_def: Some(mask4_tmp()),
                    addr,
                    mask: Some(mask),
                    passthrough: Some(added),
                    vector_type: MockType::V4I32,
                    fault_mode: VectorFaultMode::FirstFault,
                    mask_mode: VectorMaskMode::Merge,
                })?;
                let segments = block.vector_segment_load(
                    &[v4i32_tmp(), v4i32_tmp()],
                    addr,
                    Some(mask),
                    MockType::V4I32,
                    2,
                    VectorSegmentLayout::Interleaved,
                )?;
                block.vector_segment_store(
                    addr,
                    segments,
                    Some(mask),
                    MockType::V4I32,
                    2,
                    VectorSegmentLayout::Interleaved,
                )?;
                block.ret(fault)?;
                Ok(())
            })
            .unwrap();

        let ssa = builder.finish_verified(VerifyLevel::Standard).unwrap();
        assert_eq!(ssa.variable_count(), 10);
    }
}
