//! SSA instruction wrapper with explicit def/use metadata.
//!
//! Unlike stack-based CIL where operands are implicit on the evaluation stack,
//! SSA instructions have explicit operands (uses) and results (defs). Each
//! [`SsaInstruction`] wraps a decomposed [`SsaOp`] along with the original
//! host instruction and optional resolved result type.
//!
//! # Dual Representation
//!
//! Each SSA instruction stores two representations:
//!
//! 1. **Original instruction** (`T::OriginalInstruction`): The raw host instruction
//!    (e.g., a CIL opcode with its operands). Retained for debugging, source mapping,
//!    and code generation to produce faithful output.
//!
//! 2. **Decomposed SSA operation** ([`SsaOp`]): The clean `result = op(operands)` form.
//!    This is the primary representation for analysis and optimization passes.
//!
//! This dual representation enables:
//! - Direct def-use chain construction from `SsaOp::defs()` / `SsaOp::uses()`
//! - Dead code elimination (variable with no uses)
//! - Pattern matching on decomposed operations for optimization
//! - Faithful code generation from the original instruction where needed
//!
//! # Result Type Resolution
//!
//! The `result_type` field captures the precise type available during initial SSA
//! construction (when full assembly metadata is available). This type survives
//! through deobfuscation transforms and is used by rebuild and codegen to recover
//! types that cannot be inferred structurally from the op alone (e.g., call return
//! types, field types, argument/local types).

use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use crate::ir::identity::InstrId;
use crate::{
    facts::{InstructionFacts, Rewrite},
    ir::{identity::Identity, ops::SsaOp, variable::SsaVarId},
    target::Target,
};

/// A host-declared fact about what an instruction *does* beyond its operation.
///
/// A role is a claim analyses may act on, and only the front end that lifted
/// the function can make it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InstructionRole {
    /// A store that keeps the caller's state, to give it back: the save of a
    /// register the convention obliges the function to preserve, of the
    /// return address, or of the caller's stack pointer a realigning prologue
    /// restores the stack pointer from.
    ///
    /// A frame cell every store to which is such a save is the caller's state
    /// and no object of the function's, so nothing reaches it but the restore
    /// that reads it back at its fixed place: a callee reads its stack
    /// arguments, which a caller stages below its locals, never the saves
    /// above them, and an address of a local reaches that local. A call the
    /// host did not describe, or memory an address escaped to, therefore does
    /// not keep such a cell's store alive, and a save nothing restores is
    /// removed even in a function that calls.
    ///
    /// The role is the store's, not the stored value's: the caller's stack
    /// pointer `lea ecx, [esp + 4]` keeps for the restore is also the base
    /// `main` reads `argv` through, and the same value stored into a local
    /// whose address is passed on is data.
    ///
    /// # Host obligations
    ///
    /// - **The cell is no object.** Declare the role only on a store into
    ///   the function's frame whose cell no access names but the saves and
    ///   their restores.
    /// - **A save is no argument.** A store into the area a call reads its
    ///   stack arguments from would be read, and the role says it is not.
    CallerStateSave,
}

/// An SSA instruction wrapping a decomposed operation and original host instruction.
///
/// Combines the decomposed [`SsaOp`] (the canonical representation for analysis)
/// with the original host instruction (for debugging and faithful code generation)
/// and an optional resolved result type.
///
/// # Examples
///
/// ```rust
/// use analyssa::{ir::{SsaInstruction, SsaOp, SsaVarId}, MockTarget};
///
/// let left = SsaVarId::from_index(0);
/// let right = SsaVarId::from_index(1);
/// let result = SsaVarId::from_index(2);
/// let instr: SsaInstruction<MockTarget> = SsaInstruction::new(
///     <MockTarget as analyssa::Target>::synthetic_instruction(),
///     SsaOp::Add { dest: result, left, right, flags: None },
/// );
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::Type: serde::Serialize, T::TypeRef: serde::Serialize, \
                     T::MethodRef: serde::Serialize, T::FieldRef: serde::Serialize, T::SymbolRef: serde::Serialize, \
                     T::SigRef: serde::Serialize, T::OriginalInstruction: serde::Serialize, \
                     T::InstructionFacts: serde::Serialize",
        deserialize = "T::Type: serde::Deserialize<'de>, T::TypeRef: serde::Deserialize<'de>, \
                       T::MethodRef: serde::Deserialize<'de>, T::FieldRef: serde::Deserialize<'de>, T::SymbolRef: serde::Deserialize<'de>, \
                       T::SigRef: serde::Deserialize<'de>, T::OriginalInstruction: serde::Deserialize<'de>, \
                       T::InstructionFacts: serde::Deserialize<'de>"
    ))
)]
pub struct SsaInstruction<T: Target> {
    /// This instruction's identity, or `None` until a boundary assigns one.
    ///
    /// Construction through the raw surface leaves it unset - a host building a
    /// function block by block has no allocator in hand - and
    /// [`SsaFunction::assign_instr_ids`](crate::ir::function::SsaFunction::assign_instr_ids)
    /// stamps whatever is unstamped. Every repair boundary calls it, so an
    /// instruction is unidentified only between construction and the first
    /// repair.
    #[cfg_attr(feature = "serde", serde(default))]
    id: Option<InstrId>,

    /// The original host instruction retained for debugging, source mapping,
    /// and code generation. Hosts can store a decoded instruction, source
    /// location, RVA, or `()` for synthetic instructions.
    original: T::OriginalInstruction,

    /// The decomposed SSA operation in `result = op(operands)` form.
    ///
    /// This is the authoritative representation used by all analysis and
    /// optimization passes. All data dependencies are explicit in the
    /// operation's `defs()` and `uses()`.
    op: SsaOp<T>,

    /// What the host knows about this instruction.
    ///
    /// Carried here rather than in a table keyed by position, so it travels
    /// with the instruction when a pass moves it and is restored with it when
    /// an edit rolls back. [`InstructionFacts`] states what happens to it when
    /// the instruction is copied or its operation rewritten.
    #[cfg_attr(feature = "serde", serde(default))]
    facts: T::InstructionFacts,

    /// What the host declares this instruction does, beyond its operation;
    /// `None` for one it declared nothing about.
    #[cfg_attr(feature = "serde", serde(default))]
    role: Option<InstructionRole>,

    /// Optional resolved result type captured during SSA construction.
    ///
    /// Set when the SSA converter has access to full assembly metadata
    /// (TypeContext). Survives deobfuscation transforms. Used by rebuild
    /// and codegen to recover types that cannot be inferred structurally
    /// from the SSA op alone: call return types, field types, argument
    /// and local types, etc.
    result_type: Option<T::Type>,
}

impl<T: Target> SsaInstruction<T> {
    /// Creates a new SSA instruction with a decomposed operation.
    ///
    /// # Arguments
    ///
    /// * `original` - The original instruction
    /// * `op` - The decomposed SSA operation
    #[must_use]
    pub fn new(original: T::OriginalInstruction, op: SsaOp<T>) -> Self {
        Self {
            id: None,
            facts: T::InstructionFacts::default(),
            role: None,
            original,
            op,
            result_type: None,
        }
    }

    /// Creates an SSA instruction with only a decomposed operation (no original
    /// instruction breadcrumb).
    ///
    /// This is useful for synthetic instructions like phi nodes that don't
    /// correspond to any source-level instruction. The placeholder original
    /// is supplied by `Target::synthetic_instruction()`.
    #[must_use]
    pub fn synthetic(op: SsaOp<T>) -> Self {
        Self {
            id: None,
            facts: T::InstructionFacts::default(),
            role: None,
            original: T::synthetic_instruction(),
            op,
            result_type: None,
        }
    }

    /// Returns this instruction's identity, or `None` before one is assigned.
    #[must_use]
    pub const fn id(&self) -> Option<InstrId> {
        self.id
    }

    /// Returns a copy of this instruction as a *new* instruction.
    ///
    /// The copy has no identity of its own yet; the boundary that inserts it
    /// stamps one, and
    /// [`SsaEditor::duplicate_instruction`](crate::ir::function::SsaEditor::duplicate_instruction)
    /// records where the copy came from. This is what a pass uses to clone an
    /// instruction, as opposed to moving one:
    /// [`SsaEditor::detach_instruction`](crate::ir::function::SsaEditor::detach_instruction)
    /// takes an instruction out *with* its identity, because a moved
    /// instruction is the same instruction somewhere else.
    #[must_use]
    pub fn duplicated(&self) -> Self {
        Self {
            id: None,
            facts: self.facts.duplicate(),
            role: self.role,
            original: self.original.clone(),
            op: self.op.clone(),
            result_type: self.result_type.clone(),
        }
    }

    /// Returns what the host knows about this instruction.
    #[must_use]
    pub const fn facts(&self) -> &T::InstructionFacts {
        &self.facts
    }

    /// Returns what the host knows about this instruction, for updating.
    pub const fn facts_mut(&mut self) -> &mut T::InstructionFacts {
        &mut self.facts
    }

    /// Replaces what the host knows about this instruction.
    pub fn set_facts(&mut self, facts: T::InstructionFacts) {
        self.facts = facts;
    }

    /// Returns what the host declares this instruction does, if anything.
    #[must_use]
    pub const fn role(&self) -> Option<InstructionRole> {
        self.role
    }

    /// Records what the host declares this instruction does.
    ///
    /// # Arguments
    ///
    /// * `role` - The fact the front end states about the instruction.
    pub fn set_role(&mut self, role: InstructionRole) {
        self.role = Some(role);
    }

    /// Withdraws what the host declared this instruction does: the pair to
    /// [`Self::set_role`], for a host that learns the declaration no longer
    /// holds (a store it called a save turns out to stage an argument).
    pub fn clear_role(&mut self) {
        self.role = None;
    }

    /// Returns a reference to the original instruction.
    #[must_use]
    pub const fn original(&self) -> &T::OriginalInstruction {
        &self.original
    }

    /// Returns the decomposed SSA operation.
    #[must_use]
    pub const fn op(&self) -> &SsaOp<T> {
        &self.op
    }

    /// Returns a mutable reference to the decomposed SSA operation.
    pub fn op_mut(&mut self) -> &mut SsaOp<T> {
        &mut self.op
    }

    /// Replaces every use of `old_var` in the operation with `new_var`, and
    /// where it replaced any, folds in `carried`: the facts of the replaced
    /// value's computation, which this instruction now computes in its place
    /// ([`SsaFunction::substituted_facts`]).
    ///
    /// # Returns
    ///
    /// How many uses were replaced.
    ///
    /// [`SsaFunction::substituted_facts`]: crate::ir::function::SsaFunction::substituted_facts
    pub fn replace_uses(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        carried: &[T::InstructionFacts],
    ) -> usize {
        let replaced = self.op.replace_uses(old_var, new_var);
        if replaced > 0 {
            for facts in carried {
                self.facts.absorb_computation(facts);
            }
        }
        replaced
    }

    /// Sets the decomposed SSA operation.
    ///
    /// Clears `result_type` because the new op may have a different result type.
    /// Callers that know the type should call `set_result_type()` afterwards.
    ///
    /// The instruction is the same instruction computing something else, so the
    /// host's facts are told ([`InstructionFacts::rewritten`]) rather than
    /// discarded: a fact about the operation may no longer hold, while one about
    /// where the instruction came from still does.
    pub fn set_op(&mut self, op: SsaOp<T>) {
        self.op = op;
        self.result_type = None;
        self.facts.rewritten(Rewrite::Host);
    }

    /// Sets the decomposed SSA operation while preserving the resolved result
    /// type.
    ///
    /// Unlike [`Self::set_op`] (which clears `result_type`), this keeps the
    /// type the instruction records, and reads one from the new op's
    /// structural inference ([`SsaOp::infer_result_type`]) only where it
    /// records none. Rewriting passes that replace an op without changing the
    /// value its destination produces - variable remapping, constant folding,
    /// in-place op replacement - use this, and that value's type is the one
    /// its producer stated: a constant a pointer folds to is still the
    /// pointer, though its value names only an integer's width, and a sum of
    /// two 32-bit words is still 32 bits wide, whatever a host answers for
    /// arithmetic it cannot see the operands of. The resolved types the
    /// rebuild cannot recover (call return types, field types) do not vanish
    /// either.
    ///
    /// An op with no destination ([`SsaOp::Nop`], stores, terminators) produces
    /// no value, so there is nothing to preserve and `result_type` is cleared.
    /// This keeps the removal path (`SsaEditor::nop_instruction`) from leaving a
    /// dead type stamped on an instruction that no longer defines anything.
    ///
    /// The cause reaches the host's facts, which is how a fact that survives a
    /// constant fold but not a strength reduction tells the two apart.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to install.
    /// * `cause` - What rewrote it; [`Rewrite::Host`] for a host's own edit.
    pub fn set_op_preserving_type(&mut self, op: SsaOp<T>, cause: Rewrite) {
        let prev = self.result_type.take();
        self.op = op;
        self.result_type = if self.op.dest().is_some() {
            prev.or_else(|| self.op.infer_result_type())
        } else {
            None
        };
        self.facts.rewritten(cause);
    }

    /// Renames the variables the operation names, in place, preserving the
    /// resolved result type as [`Self::set_op_preserving_type`] does.
    ///
    /// The same instruction as `set_op_preserving_type(op.remap_variables(..))`
    /// leaves, without building the second operation.
    ///
    /// # Arguments
    ///
    /// * `remap` - The new id of a variable, or `None` to keep it.
    pub fn remap_variables_preserving_type<F>(&mut self, remap: F)
    where
        F: Fn(SsaVarId) -> Option<SsaVarId>,
    {
        let prev = self.result_type.take();
        self.op.visit_operands_mut(|_, var| {
            if let Some(new) = remap(*var) {
                *var = new;
            }
        });
        self.result_type = if self.op.dest().is_some() {
            prev.or_else(|| self.op.infer_result_type())
        } else {
            None
        };
        self.facts.rewritten(Rewrite::Host);
    }

    /// Returns the resolved result type, if set during SSA construction.
    #[must_use]
    pub fn result_type(&self) -> Option<&T::Type> {
        self.result_type.as_ref()
    }

    /// Sets the resolved result type.
    pub fn set_result_type(&mut self, ty: Option<T::Type>) {
        self.result_type = ty;
    }

    /// Builder pattern: sets the result type and returns self.
    #[must_use]
    pub fn with_result_type(mut self, ty: T::Type) -> Self {
        self.result_type = Some(ty);
        self
    }

    /// Builder pattern: gives the instruction `id`, one its function minted
    /// ([`SsaFunction::mint_instr_id`](crate::ir::function::SsaFunction::mint_instr_id)),
    /// for a host that keys a record of its own by the instruction before
    /// the instruction joins the function.
    #[must_use]
    pub const fn with_id(mut self, id: InstrId) -> Self {
        self.id = Some(id);
        self
    }

    /// Returns `true` if this instruction is a terminator.
    ///
    /// Terminators are instructions that end a basic block (jumps, branches, returns, throws).
    #[must_use]
    pub fn is_terminator(&self) -> bool {
        self.op.is_terminator()
    }

    /// Returns `true` if this instruction may throw an exception.
    #[must_use]
    pub fn may_throw(&self) -> bool {
        self.op.may_throw()
    }

    /// Returns `true` if this instruction is pure (has no side effects).
    ///
    /// Pure instructions can be eliminated if their result is unused.
    #[must_use]
    pub fn is_pure(&self) -> bool {
        self.op.is_pure()
    }

    /// Returns the SSA variables used (read) by this instruction.
    #[must_use]
    pub fn uses(&self) -> Vec<SsaVarId> {
        self.op.uses()
    }

    /// Calls `f` for every SSA variable used by this instruction.
    pub fn for_each_use<F>(&self, f: F)
    where
        F: FnMut(SsaVarId),
    {
        self.op.for_each_use(f);
    }

    /// Returns `true` if this instruction uses (reads) the given variable.
    ///
    /// Allocation-free; prefer over `self.uses().contains(&var)` in hot paths.
    #[must_use]
    pub fn uses_var(&self, var: SsaVarId) -> bool {
        self.op.uses_var(var)
    }

    /// Returns the SSA variable defined by this instruction, if any.
    #[must_use]
    pub fn def(&self) -> Option<SsaVarId> {
        self.op.dest()
    }

    /// Returns all SSA variables defined by this instruction.
    pub fn defs(&self) -> impl Iterator<Item = SsaVarId> {
        self.op.defs()
    }

    /// Returns `true` if this instruction defines a value.
    #[must_use]
    pub fn has_def(&self) -> bool {
        self.op.defs().next().is_some()
    }

    /// Returns `true` if this instruction has no uses.
    #[must_use]
    pub fn has_no_uses(&self) -> bool {
        let mut has_use = false;
        self.op.for_each_use(|_| has_use = true);
        !has_use
    }

    /// Returns all SSA variables referenced by this instruction.
    ///
    /// This includes both uses and the def (if present).
    #[must_use]
    pub fn all_variables(&self) -> Vec<SsaVarId> {
        let mut vars = Vec::new();
        self.op.for_each_use(|var| vars.push(var));
        vars.extend(self.op.defs());
        vars
    }

    /// Calls `f` for every SSA variable referenced (used or defined) by this
    /// instruction, without allocating.
    pub fn for_each_variable<F>(&self, mut f: F)
    where
        F: FnMut(SsaVarId),
    {
        self.op.for_each_use(&mut f);
        for def in self.op.defs() {
            f(def);
        }
    }

    // Original-instruction accessors. Implementation routes through
    // `Target::instruction_mnemonic` / `instruction_rva`, so this works for any
    // host that retains a meaningful `OriginalInstruction`.
    /// Returns the instruction's mnemonic via `Target::instruction_mnemonic`.
    #[must_use]
    pub fn mnemonic(&self) -> &'static str {
        T::instruction_mnemonic(&self.original)
    }

    /// Returns the instruction's RVA via `Target::instruction_rva`.
    #[must_use]
    pub fn rva(&self) -> u64 {
        T::instruction_rva(&self.original)
    }
}

impl<T: Target> fmt::Display for SsaInstruction<T>
where
    SsaOp<T>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.op)
    }
}

impl<T: Target> Identity for SsaInstruction<T> {
    fn set_id(&mut self, id: InstrId) {
        self.id = Some(id);
    }

    fn clear_identity(&mut self) {
        self.id = None;
        self.facts = T::InstructionFacts::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{exception::HandlerKind, ops::SsaOp, value::ConstValue},
        target::Target,
        testing::{MockTarget, MockType},
    };

    /// A host that answers the result-type queries as a target of one
    /// width would: a 32-bit constant is an `int32`, and arithmetic is
    /// 64 bits wide whatever its operands are.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct AnsweringTarget;

    impl Target for AnsweringTarget {
        type TypeRef = u32;
        type MethodRef = u32;
        type FieldRef = u32;
        type SigRef = u32;
        type SymbolRef = u32;
        type ExceptionKind = ();
        type Type = MockType;
        type OriginalInstruction = ();
        type LocalSignature = ();
        type VariableFacts = ();
        type InstructionFacts = ();
        type FunctionFacts = ();
        type Capability = ();
        fn ptr_bytes(&self) -> u32 {
            4
        }
        fn synthetic_instruction() {}
        fn unknown_type() -> MockType {
            MockType::Unknown
        }
        fn instruction_mnemonic(_: &()) -> &'static str {
            "fixture"
        }
        fn instruction_rva(_: &()) -> u64 {
            0
        }
        fn handler_kind(_: &()) -> HandlerKind {
            HandlerKind::Catch
        }
        fn is_integer(ty: &MockType) -> bool {
            MockTarget::is_integer(ty)
        }
        fn is_floating(ty: &MockType) -> bool {
            MockTarget::is_floating(ty)
        }
        fn is_signed(ty: &MockType) -> bool {
            MockTarget::is_signed(ty)
        }
        fn is_pointer(ty: &MockType) -> bool {
            MockTarget::is_pointer(ty)
        }
        fn is_reference(ty: &MockType) -> bool {
            MockTarget::is_reference(ty)
        }
        fn is_unknown(ty: &MockType) -> bool {
            MockTarget::is_unknown(ty)
        }
        fn bit_width(ty: &MockType) -> Option<u32> {
            MockTarget::bit_width(ty)
        }
        fn result_type_for_const(value: &ConstValue<Self>) -> Option<MockType> {
            matches!(value, ConstValue::I32(_)).then_some(MockType::I32)
        }
        fn arithmetic_result_type() -> Option<MockType> {
            Some(MockType::I64)
        }
    }

    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    /// The delegating accessors report the wrapped op's shape: uses, def,
    /// purity, terminator/throw flags, and the host mnemonic/RVA breadcrumbs.
    #[test]
    fn instruction_accessors_reflect_operation_shape() {
        let instr = SsaInstruction::<MockTarget>::synthetic(SsaOp::Add {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        });

        assert_eq!(instr.original(), &());
        assert!(matches!(instr.op(), SsaOp::Add { .. }));
        assert!(instr.is_pure());
        assert!(!instr.is_terminator());
        assert!(!instr.may_throw());
        assert_eq!(instr.uses(), vec![var(0), var(1)]);
        assert_eq!(instr.def(), Some(var(2)));
        assert!(instr.has_def());
        assert!(!instr.has_no_uses());
        assert_eq!(instr.all_variables(), vec![var(0), var(1), var(2)]);
        assert_eq!(instr.mnemonic(), "<mock>");
        assert_eq!(instr.rva(), 0);
    }

    /// Mutating through `op_mut` leaves the resolved type intact, but `set_op`
    /// clears it (the replacement may produce a different type), and it can then
    /// be restored explicitly via `set_result_type`.
    #[test]
    fn mutating_op_clears_result_type_until_reset() {
        let mut instr = SsaInstruction::<MockTarget>::synthetic(SsaOp::Const {
            dest: var(0),
            value: ConstValue::I32(1),
        })
        .with_result_type(MockType::I32);

        assert_eq!(instr.result_type(), Some(&MockType::I32));

        if let SsaOp::Const { value, .. } = instr.op_mut() {
            *value = ConstValue::I32(2);
        }
        assert_eq!(instr.result_type(), Some(&MockType::I32));

        instr.set_op(SsaOp::returns(Some(var(0))));
        assert_eq!(instr.result_type(), None);
        assert!(instr.is_terminator());
        assert_eq!(instr.def(), None);
        assert!(!instr.has_def());
        assert_eq!(instr.uses(), vec![var(0)]);

        instr.set_result_type(Some(MockType::Unknown));
        assert_eq!(instr.result_type(), Some(&MockType::Unknown));
        instr.set_result_type(None);
        assert_eq!(instr.result_type(), None);
    }

    /// `set_op_preserving_type` carries a non-structurally-inferrable resolved
    /// type forward across an op replacement, whereas `set_op` still clears it.
    #[test]
    fn set_op_preserving_type_keeps_non_inferrable_type() {
        // `LoadArg`'s type is not structurally inferrable (the same class as
        // call returns and field loads). A rewriting pass that replaces the op
        // via `set_op_preserving_type` must carry the resolved type forward;
        // clearing it (as `set_op` does) would make the type silently vanish
        // before the next rebuild.
        let mut instr = SsaInstruction::<MockTarget>::synthetic(SsaOp::LoadArg {
            dest: var(0),
            arg_index: 0,
        })
        .with_result_type(MockType::I32);
        assert_eq!(instr.result_type(), Some(&MockType::I32));

        instr.set_op_preserving_type(
            SsaOp::LoadArg {
                dest: var(0),
                arg_index: 1,
            },
            Rewrite::Host,
        );
        assert_eq!(
            instr.result_type(),
            Some(&MockType::I32),
            "a non-inferrable replacement carries the prior type forward",
        );

        // The clearing primitive still clears.
        instr.set_op(SsaOp::LoadArg {
            dest: var(0),
            arg_index: 2,
        });
        assert_eq!(instr.result_type(), None);
    }

    /// Replacing with a destination-less op (here `Nop`) via
    /// `set_op_preserving_type` clears the resolved type, since no value remains
    /// whose type could be preserved.
    #[test]
    fn set_op_preserving_type_clears_type_for_destless_op() {
        // `SsaOp::Nop` defines nothing, so there is no value whose type could be
        // preserved. This is the removal primitive DCE and GVN drive through
        // `SsaEditor::nop_instruction`; carrying `I32` onto a `Nop` leaves a
        // dead type stamped on an instruction with no destination.
        let mut instr = SsaInstruction::<MockTarget>::synthetic(SsaOp::LoadArg {
            dest: var(0),
            arg_index: 0,
        })
        .with_result_type(MockType::I32);

        instr.set_op_preserving_type(SsaOp::Nop, Rewrite::Host);

        assert_eq!(
            instr.result_type(),
            None,
            "an op with no destination must not retain the replaced value's type",
        );
    }

    /// **A rewrite keeps the type its instruction records, and a host's
    /// answer fills only one it lacks.** A pointer folded to a constant whose
    /// value names an `int32`'s width is still the pointer, and a 32-bit sum
    /// renamed is still 32 bits, though the host answers 64 for arithmetic.
    /// The control is the same rewrites of instructions that record no type:
    /// the host's answers are theirs.
    #[test]
    fn a_rewrite_keeps_the_recorded_type_over_the_hosts_answer() {
        let sum = || SsaOp::Add {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        let constant = || SsaOp::Const {
            dest: var(2),
            value: ConstValue::I32(0x1000),
        };

        let mut folded =
            SsaInstruction::<AnsweringTarget>::synthetic(sum()).with_result_type(MockType::Ptr);
        folded.set_op_preserving_type(constant(), Rewrite::Host);
        assert_eq!(folded.result_type(), Some(&MockType::Ptr));

        let mut renamed =
            SsaInstruction::<AnsweringTarget>::synthetic(sum()).with_result_type(MockType::I32);
        renamed.remap_variables_preserving_type(|id| (id == var(0)).then(|| var(5)));
        assert_eq!(renamed.uses(), vec![var(5), var(1)]);
        assert_eq!(renamed.result_type(), Some(&MockType::I32));

        let mut untyped_fold = SsaInstruction::<AnsweringTarget>::synthetic(sum());
        untyped_fold.set_op_preserving_type(constant(), Rewrite::Host);
        assert_eq!(
            untyped_fold.result_type(),
            Some(&MockType::I32),
            "control: an untyped fold takes the host's answer"
        );
        let mut untyped_rename = SsaInstruction::<AnsweringTarget>::synthetic(sum());
        untyped_rename.remap_variables_preserving_type(|id| (id == var(0)).then(|| var(5)));
        assert_eq!(
            untyped_rename.result_type(),
            Some(&MockType::I64),
            "control: an untyped rename takes the host's answer"
        );
    }

    /// An operand-less instruction (a bare `Return`) reports no uses and no
    /// referenced variables.
    #[test]
    fn no_operand_instructions_report_no_uses() {
        let instr = SsaInstruction::<MockTarget>::new((), SsaOp::returns(None));

        assert!(instr.has_no_uses());
        assert_eq!(instr.all_variables(), Vec::<SsaVarId>::new());
    }
}
