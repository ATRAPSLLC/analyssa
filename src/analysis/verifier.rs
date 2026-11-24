//! SSA verifier for validating SSA invariants.
//!
//! Provides comprehensive verification of SSA form at three levels to catch
//! invariant violations introduced by transformations, preventing silent
//! corruption that would manifest as broken codegen or incorrect deobfuscation.
//!
//! # Verification Levels
//!
//! ## Quick (O(n))
//!
//! Checks the fundamental SSA invariants:
//! - **Single definition**: Every variable is defined at most once (scan all
//!   phi nodes and instructions, noting the first definition of each variable).
//! - **Block structure**: Every non-empty block ends in a terminator, and
//!   terminators are the last instruction in their block.
//! - **Entry block**: No terminator transfers control to block 0.
//! - **No intra-block cycles**: A variable is not used before its definition
//!   within the same block (each definition stamps its variable with its block
//!   and position, and a use that finds a stamp at or after it is a cycle).
//! - **No placeholders**: No `SsaVarId::PLACEHOLDER` remains in finalized SSA.
//! - **No self-referential instructions**: An instruction's destination does
//!   not appear in its own operands.
//!
//! ## Standard (O(n * m))
//!
//! Adds:
//! - **Def-use chain integrity**: Every variable used in an instruction or phi
//!   operand has a definition somewhere (either in the variables vec or a block).
//! - **Phi operand coverage**: Each phi node has exactly one operand per CFG
//!   predecessor, no operands from non-predecessor blocks, and no phis in the
//!   entry block (which has no predecessors).
//! - **Variable registration**: Every variable used in a block is registered in
//!   the function's variables vec, and vice-versa (no orphan or unregistered vars).
//!
//! ## Full (O(n^2) worst case)
//!
//! Adds:
//! - **Dominance verification**: Every use of a variable must be dominated by its
//!   definition (standard SSA requirement). For phi operands, the use is considered
//!   to be at the end of the predecessor block, following standard SSA semantics.
//!   Only verified for reachable blocks.
//!
//! # Error Types
//!
//! Various [`VerifierError`] variants describe each invariant violation with
//! precise locations (block, instruction, variable) for debugging.
//!
//! # Complexity
//!
//! | Level | Time | Description |
//! |-------|------|-------------|
//! | Quick | O(n) | Single pass over blocks and instructions |
//! | Standard | O(n*m) | n = blocks, m = avg predecessors per block |
//! | Full | O(b^2) | Dominator computation + O(b * u) queries |
//!
//! # One scan
//!
//! The checks do not each walk the function. One walk over the blocks reads an
//! operation's operands once and hands them to every check that wants them;
//! checks that need a fact about the whole function (who defines what, who reads
//! what, which use lists are stale) note what they saw and conclude when the
//! walk ends. Findings are kept per check, so the report reads as if each check
//! had made its own pass: check by check, each in the order it found them.

use std::{collections::BTreeSet, error::Error, fmt, mem};

use rustc_hash::FxHashMap;

use crate::{
    BitSet,
    analysis::{cfg::SsaCfg, exceptions::EhDominance},
    graph::NodeId,
    ir::{
        block::SsaBlock,
        exception::ExceptionTableError,
        function::SsaFunction,
        instruction::{InstrId, SsaInstruction},
        ops::{OperandRole, RuntimeValueKind, SsaOp},
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, SsaVariable},
    },
    target::{Target, VectorDescriptor, VectorMaskDescriptor},
    vartables::{VarBuckets, VarTable},
};

mod operations;
mod vectors;

use self::operations::OperationScan;

/// Bit extent as a fixed term plus a rational multiple of runtime vscale.
///
/// Keeping both terms makes a fixed prefix distinguishable from scalable
/// storage; a minimum vector width is never treated as a concrete width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegionExtent {
    /// Bits independent of vscale.
    fixed: u128,
    /// Numerator of the coefficient multiplying vscale.
    scaled: u128,
    /// Nonzero denominator of the scalable coefficient, in reduced form.
    divisor: u128,
}

impl RegionExtent {
    /// Returns the normalized extent when its divisor is valid.
    fn new(fixed: u128, scaled: u128, divisor: u128) -> Option<Self> {
        if divisor == 0 {
            return None;
        }
        let gcd = greatest_common_divisor(scaled, divisor);
        Some(Self {
            fixed,
            scaled: scaled.checked_div(gcd)?,
            divisor: divisor.checked_div(gcd)?,
        })
    }

    /// Returns the sum when the rational arithmetic fits the representation.
    fn plus(self, other: Self) -> Option<Self> {
        let gcd = greatest_common_divisor(self.divisor, other.divisor);
        let left = other.divisor.checked_div(gcd)?;
        let right = self.divisor.checked_div(gcd)?;
        Self::new(
            self.fixed.checked_add(other.fixed)?,
            self.scaled
                .checked_mul(left)?
                .checked_add(other.scaled.checked_mul(right)?)?,
            self.divisor.checked_mul(left)?,
        )
    }

    /// Returns the normalized count of elements in a fixed or scalable vector.
    fn lanes(shape: VectorDescriptor) -> Option<Self> {
        match shape {
            VectorDescriptor::Fixed(shape) => Self::new(u128::from(shape.lane_count), 0, 1),
            VectorDescriptor::Scalable(shape) => Self::new(
                0,
                u128::from(shape.min_lane_count)
                    .checked_mul(u128::from(shape.length_multiplier.numerator))?,
                u128::from(shape.length_multiplier.denominator),
            ),
        }
    }

    /// Returns whether the fixed and scalable terms each cover another extent.
    fn covers(self, other: Self) -> bool {
        self.fixed >= other.fixed
            && self
                .scaled
                .checked_mul(other.divisor)
                .zip(other.scaled.checked_mul(self.divisor))
                .is_some_and(|(left, right)| left >= right)
    }

    /// Returns the physical bit extent the target type states, when known.
    fn of<T: Target>(ty: &T::Type) -> Option<Self> {
        if let Some(shape) = T::vector_descriptor(ty) {
            return match shape {
                VectorDescriptor::Fixed(shape) => Self::new(u128::from(shape.total_bits), 0, 1),
                VectorDescriptor::Scalable(shape) => Self::new(
                    0,
                    u128::from(shape.min_lane_count)
                        .checked_mul(u128::from(shape.lane_bits))?
                        .checked_mul(u128::from(shape.length_multiplier.numerator))?,
                    u128::from(shape.length_multiplier.denominator),
                ),
            };
        }
        if let Some(shape) = T::vector_mask_descriptor(ty) {
            return match shape {
                VectorMaskDescriptor::Fixed(shape) => Self::new(
                    u128::from(shape.lane_count).checked_mul(u128::from(shape.lane_bits))?,
                    0,
                    1,
                ),
                VectorMaskDescriptor::Scalable(shape) => Self::new(
                    0,
                    u128::from(shape.min_lane_count)
                        .checked_mul(u128::from(shape.lane_bits))?
                        .checked_mul(u128::from(shape.length_multiplier.numerator))?,
                    u128::from(shape.length_multiplier.denominator),
                ),
            };
        }
        Self::new(u128::from(T::bit_width(ty)?), 0, 1)
    }
}

/// Definition site for verifier error reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierDefSite {
    /// Index of the block containing the definition.
    pub block: usize,
    /// Location class of the definition within the block.
    pub kind: DefKind,
}

/// What kind of definition produced a variable within a block.
///
/// Distinguishes between phi node definitions (from SSA control flow merges)
/// and instruction definitions (from regular operations). Used by the verifier
/// to provide precise error locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefKind {
    /// Variable is defined by a phi node at the given phi index within the block.
    Phi(usize),
    /// Variable is defined by an instruction at the given instruction index within the block.
    Instruction(usize),
}

/// Where a function refers to a variable id.
///
/// An id that names no row of the variable table is reported with the place
/// that named it, because the repair differs: a phi operand out of range is a
/// phi the rename never filled, while an instruction operand out of range is an
/// edit that outlived the variable it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarRefSite {
    /// The result of the phi at `phi_idx` in `block`.
    PhiResult {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
    },
    /// An operand of the phi at `phi_idx` in `block`, arriving from `pred`.
    PhiOperand {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
        /// Predecessor the operand names.
        pred: usize,
    },
    /// A destination of the instruction at `instr_idx` in `block`.
    Definition {
        /// Block holding the instruction.
        block: usize,
        /// Index of the instruction within the block.
        instr_idx: usize,
    },
    /// An operand of the instruction at `instr_idx` in `block`.
    Operand {
        /// Block holding the instruction.
        block: usize,
        /// Index of the instruction within the block.
        instr_idx: usize,
    },
}

/// The bound a def site or use site ran past.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteBound {
    /// The function holds `block_count` blocks.
    Block {
        /// Number of blocks the function holds.
        block_count: usize,
    },
    /// Block `block` holds `instruction_count` instructions.
    Instruction {
        /// Block the site names.
        block: usize,
        /// Number of instructions that block holds.
        instruction_count: usize,
    },
    /// Block `block` holds `phi_count` phis.
    Phi {
        /// Block the site names.
        block: usize,
        /// Number of phis that block holds.
        phi_count: usize,
    },
}

/// Why a runtime-supplied definition stands where it cannot.
///
/// The two are different defects with different repairs: a block that is not a
/// runtime entry is a definition in the wrong place entirely, while a
/// non-leading position is the right block with an edit that slipped an
/// instruction above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeValuePlacement {
    /// Control cannot enter this block the way the value's kind requires.
    NotARuntimeEntry,
    /// The block is right, but something else in it runs first.
    NotLeading,
}

impl fmt::Display for RuntimeValuePlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotARuntimeEntry => {
                write!(f, "control cannot enter this block from the runtime")
            }
            Self::NotLeading => write!(f, "another instruction runs before it"),
        }
    }
}

impl fmt::Display for VarRefSite {
    /// Names the place, as a phrase that reads inside a sentence about the id.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhiResult { block, phi_idx } => {
                write!(f, "the result of phi {phi_idx} in block {block}")
            }
            Self::PhiOperand {
                block,
                phi_idx,
                pred,
            } => write!(
                f,
                "the operand from block {pred} of phi {phi_idx} in block {block}"
            ),
            Self::Definition { block, instr_idx } => {
                write!(
                    f,
                    "a destination of instruction {instr_idx} in block {block}"
                )
            }
            Self::Operand { block, instr_idx } => {
                write!(f, "an operand of instruction {instr_idx} in block {block}")
            }
        }
    }
}

impl fmt::Display for SiteBound {
    /// States the bound the site ran past.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Block { block_count } => write!(f, "the function holds {block_count} block(s)"),
            Self::Instruction {
                block,
                instruction_count,
            } => write!(f, "block {block} holds {instruction_count} instruction(s)"),
            Self::Phi { block, phi_count } => {
                write!(f, "block {block} holds {phi_count} phi(s)")
            }
        }
    }
}

/// Errors detected by the SSA verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierError {
    /// A variable is used but never defined.
    UndefinedUse {
        /// Index of the block containing the use.
        block: usize,
        /// Index of the instruction containing the use.
        instr_idx: usize,
        /// Variable read before any verifier-visible definition.
        var: SsaVarId,
    },
    /// A phi node has more than one operand naming the same predecessor edge.
    ///
    /// One edge carries one value, so duplicates have no defined meaning and
    /// consumers disagree about them: [`PhiNode::operand_from`] returns the first
    /// and discards the rest, while SCCP meets all of them and yields Bottom.
    ///
    /// [`PhiNode::operand_from`]: crate::ir::phi::PhiNode::operand_from
    DuplicatePhiOperand {
        /// Index of the block containing the phi node.
        block: usize,
        /// Index of the phi node in the block's phi list.
        phi_idx: usize,
        /// Predecessor named by more than one operand.
        pred: usize,
    },
    /// A phi node is missing an operand for a CFG predecessor.
    MissingPhiOperand {
        /// Index of the block containing the phi node.
        block: usize,
        /// Index of the phi node in the block's phi list.
        phi_idx: usize,
        /// Predecessor block that lacks a corresponding phi operand.
        missing_pred: usize,
    },
    /// A phi node has an operand for a non-predecessor block.
    ExtraPhiOperand {
        /// Index of the block containing the phi node.
        block: usize,
        /// Index of the phi node in the block's phi list.
        phi_idx: usize,
        /// Block referenced by the phi operand that is not a CFG predecessor.
        extra_pred: usize,
    },
    /// A variable is defined more than once.
    DuplicateDefinition {
        /// Variable that has more than one definition site.
        var: SsaVarId,
        /// First definition site found by the verifier.
        def1: VerifierDefSite,
        /// Later conflicting definition site found by the verifier.
        def2: VerifierDefSite,
    },
    /// A variable exists in the variables vec but has no definition in any block.
    OrphanVariable {
        /// Registered variable that has no block-local definition.
        var: SsaVarId,
    },
    /// A variable appears in an instruction but is not in the variables vec.
    UnregisteredVariable {
        /// Variable referenced by IR but absent from the function's registry.
        var: SsaVarId,
    },
    /// A variable's recorded use list disagrees with the uses in the IR, in
    /// count or in the sites it names.
    ///
    /// The use index is not metadata: the sparse analyses re-evaluate a
    /// definition's consumers by walking it, so a use missing from the list is a
    /// consumer that is never revisited when the value changes, leaving a stale
    /// result that is narrower than the truth. Editing the IR without
    /// [`SsaFunction::recompute_uses`] is what produces it.
    ///
    /// [`SsaFunction::recompute_uses`]: crate::ir::function::SsaFunction::recompute_uses
    StaleUseList {
        /// Variable whose use list is out of date.
        var: SsaVarId,
        /// Number of uses the variable currently records. It can equal
        /// `actual` when the recorded sites are stale rather than missing.
        recorded: usize,
        /// Number of uses the IR actually contains.
        actual: usize,
    },
    /// A non-empty block whose last instruction does not transfer control, so
    /// control falls off the end of it.
    ///
    /// Every successor model in this crate derives a block's edges from its
    /// terminator alone - [`SsaBlock::successors`](crate::ir::block::SsaBlock::successors),
    /// [`SsaCfg`] and the exception CFG all read the
    /// last instruction and stop. A block that ends in something else therefore
    /// contributes no edge, and whatever it falls through to loses a
    /// predecessor. Nothing downstream can tell that apart from a block control
    /// genuinely leaves: a pass asking "is `A` `B`'s only predecessor" gets
    /// `true` from an under-count exactly as it would from the truth.
    ///
    /// An *empty* block is not this. It carries no instruction to fall off the
    /// end of, and passes leave them behind as placeholders with no
    /// predecessor, so no edge is lost by their having no terminator.
    MissingTerminator {
        /// Index of the unterminated block.
        block: usize,
    },
    /// A terminator transfers control to the entry block.
    ///
    /// The entry block is where a call enters the function, and nothing else
    /// may enter it: every forward analysis seeds its boundary state there, and
    /// a correlated-condition rule reads "one incoming edge" as "entered along
    /// that edge". A back edge into block 0 breaks both - the boundary would
    /// hide the loop's own state, and the entry would be taken to know a guard
    /// the call never tested. A lifted loop that starts at the function's first
    /// instruction gets a fresh empty entry in front of it instead; see
    /// [`SsaFunction::ensure_entry_has_no_predecessors`].
    EntryHasPredecessors {
        /// Block whose terminator names the entry block.
        predecessor: usize,
    },
    /// A phi node appears in a block no terminator transfers to.
    ///
    /// A phi operand names the edge it arrived along, so a block with no
    /// incoming terminator edges has no edges for its operands to name. The
    /// entry block is one such block; a handler entry reached only by runtime
    /// dispatch is another, and `SsaFunction::demote_runtime_entry_phis` is
    /// what removes a phi from one.
    PhiWithoutPredecessors {
        /// Block containing the phi.
        block: usize,
        /// Index of the phi node in that block's phi list.
        phi_idx: usize,
    },
    /// A variable is used in a block not dominated by its definition block.
    DominanceViolation {
        /// Variable whose definition does not dominate its use.
        var: SsaVarId,
        /// Index of the block containing the variable definition.
        def_block: usize,
        /// Index of the block containing the variable use.
        use_block: usize,
    },
    /// A terminator instruction is not the last instruction in its block.
    TerminatorNotLast {
        /// Index of the block with a misplaced terminator.
        block: usize,
        /// Index of the terminator instruction.
        instr_idx: usize,
        /// Total number of instructions in the block.
        instr_count: usize,
    },
    /// An instruction uses a variable defined later in the same block (cycle).
    IntraBlockCycle {
        /// Index of the block containing both instructions.
        block: usize,
        /// Index of the instruction that reads too early.
        use_instr: usize,
        /// Index of the later instruction that defines the variable.
        def_instr: usize,
        /// Variable read before its in-block definition.
        var: SsaVarId,
    },
    /// The variables table is not densely indexed by variable id.
    ///
    /// [`SsaFunction::variable`] is a raw index into the table, so the whole IR
    /// depends on `variables[i].id().index() == i`. When a transform drops a row
    /// without renumbering, every id above the hole silently resolves to a
    /// *different* variable's def site, type, and use list - corruption that no
    /// other check detects, because the resulting IR is structurally well-formed.
    VariableTableNotDense {
        /// Position in the variables table.
        slot: usize,
        /// Id actually stored at that position.
        found: SsaVarId,
    },
    /// A placeholder variable ID (usize::MAX) remains in finalized SSA.
    PlaceholderVariable {
        /// Index of the block containing the placeholder.
        block: usize,
        /// Human-readable location within the block, such as a phi or instruction operand.
        location: String,
    },
    /// An instruction's destination appears in its own operands (self-referential).
    SelfReferentialInstruction {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the self-referential instruction.
        instr_idx: usize,
        /// Destination variable that also appears as an operand.
        var: SsaVarId,
    },
    /// A vector operation has incompatible or unsupported vector shapes.
    InvalidVectorOperation {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid vector instruction.
        instr_idx: usize,
        /// Human-readable description of the shape violation.
        reason: String,
    },
    /// A native atomic operation has illegal ordering, width, or outputs.
    InvalidAtomicOperation {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid atomic instruction.
        instr_idx: usize,
        /// Human-readable description of the atomic violation.
        reason: String,
    },
    /// A wide arithmetic operation has incompatible operand or result widths.
    InvalidWideArithmetic {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid wide arithmetic instruction.
        instr_idx: usize,
        /// Human-readable description of the width violation.
        reason: String,
    },
    /// A scalar operation computes over values of different widths: an
    /// addition, logical operation, shift, comparison or select whose
    /// operands, or operands and result, are integers, pointers or floats
    /// computing at different widths ([`Target::operand_width`]).
    InvalidOperandWidths {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid instruction.
        instr_idx: usize,
        /// Human-readable description of the width violation.
        reason: String,
    },
    /// A conversion states two things that disagree: a bitcast that changes
    /// the width it reads, a float-to-integer conversion whose `unsigned` flag
    /// contradicts its integer target's sign, or an integer conversion of or
    /// to a float or a vector.
    InvalidConversion {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid conversion.
        instr_idx: usize,
        /// Human-readable description of the disagreement.
        reason: String,
    },
    /// A native opaque operation or effect summary is malformed.
    InvalidNativeOperation {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid native instruction.
        instr_idx: usize,
        /// Human-readable description of the native/effect violation.
        reason: String,
    },
    /// Two instructions carry the same identity.
    ///
    /// An [`InstrId`] names one instruction for the whole life of the function,
    /// which is what lets a host key a table by it. Two holders means one of
    /// them was copied rather than moved - [`SsaInstruction::duplicated`] is
    /// what a copy goes through - or that instructions from another function
    /// were spliced in without
    /// [`SsaFunction::adopt_instructions`](crate::ir::function::SsaFunction::adopt_instructions).
    ///
    /// [`InstrId`]: crate::ir::instruction::InstrId
    /// [`SsaInstruction::duplicated`]: crate::ir::instruction::SsaInstruction::duplicated
    DuplicateInstructionId {
        /// The id two instructions hold.
        id: InstrId,
        /// Where the first one sits, as `(block, instruction)`.
        first: (usize, usize),
        /// Where the second one sits.
        second: (usize, usize),
    },
    /// An instruction in a finished function has no identity.
    ///
    /// Construction leaves instructions unidentified until they join a function
    /// and every repair boundary stamps the rest, so this reports a function
    /// edited through the raw surface and never repaired.
    /// `SsaFunction::assign_instr_ids` is what finishes it.
    UnassignedInstructionId {
        /// Block holding the instruction.
        block: usize,
        /// Index of the instruction within that block.
        instr_idx: usize,
    },
    /// A phi outside a handler or filter entry carries an operand for a throw.
    ///
    /// Only a block a throw can dispatch to merges over program points; a phi
    /// anywhere else merges over terminator edges alone.
    ExceptionalOperandAtNonRoot {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
    },
    /// An exceptional operand names a block that cannot throw here.
    ExceptionalOperandFromNonThrowingBlock {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
        /// The block the operand names.
        source: usize,
    },
    /// A block that can throw here contributes no operand.
    ///
    /// The merge is then missing a value the handler can actually see, and
    /// every analysis that meets the operands reads a narrower answer than the
    /// program allows.
    MissingExceptionalOperand {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
        /// The block whose values are unaccounted for.
        source: usize,
    },
    /// One `(block, value)` pairing appears twice among a phi's exceptional
    /// operands.
    DuplicateExceptionalOperand {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
        /// The block the operands name.
        source: usize,
        /// The value they both carry.
        value: SsaVarId,
    },
    /// An exceptional operand names a value from another rename group.
    ///
    /// A phi at a handler entry is never materialized by moves: there is no
    /// edge to put one on, so the only way the handler reads the right value is
    /// if every operand already lives in the storage the handler reads. That
    /// holds exactly while the operands are versions of one storage location.
    /// An operand from another group names a value living somewhere else, which
    /// no lowering can repair.
    ExceptionalOperandFromAnotherGroup {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
        /// The value the operand carries.
        value: SsaVarId,
        /// The rename group that value belongs to.
        group: u32,
        /// The rename group the phi merges.
        expected: u32,
    },
    /// A runtime-supplied value is defined where the runtime supplies nothing.
    ///
    /// [`SsaOp::RuntimeValue`] states that entering *this* block is what makes
    /// the value hold, which is true only at a block control enters that way
    /// and only before anything else in it runs. Anywhere else the definition
    /// is a claim nothing backs, and every use of it reads a value that was
    /// never produced.
    ///
    /// [`SsaOp::RuntimeValue`]: crate::ir::ops::SsaOp::RuntimeValue
    MisplacedRuntimeValue {
        /// Block holding the instruction.
        block: usize,
        /// Index of the instruction within that block.
        instr_idx: usize,
        /// Which runtime value it claims to supply.
        kind: RuntimeValueKind,
        /// Why the position is wrong.
        reason: RuntimeValuePlacement,
    },
    /// A block's recorded id is not its position in the block list.
    ///
    /// Blocks are addressed by position everywhere: a terminator target, a phi
    /// predecessor, a def site and every block-sized set index the block list
    /// directly. [`SsaBlock::id`] is the block's own record of that position,
    /// read by the passes that walk blocks rather than indices, so the two
    /// disagreeing means those passes and those indices describe different
    /// blocks - corruption no other check detects, because each half is
    /// internally consistent.
    ///
    /// [`SsaBlock::id`]: crate::ir::block::SsaBlock::id
    BlockTableNotDense {
        /// Position in the block list.
        slot: usize,
        /// Id the block at that position records.
        found: usize,
    },
    /// A terminator names a block the function does not hold.
    ///
    /// Every graph in this crate is built by reading terminators, and the
    /// analyses index their block-sized sets with what they find there. A
    /// target past the last block is therefore not a wrong answer waiting to be
    /// noticed downstream but an index out of range, so it is reported before
    /// any check that walks the graph.
    DanglingBranchTarget {
        /// Block whose terminator names the missing target.
        block: usize,
        /// Index of the terminator within that block.
        instr_idx: usize,
        /// Target the terminator names.
        target: usize,
        /// Number of blocks the function holds.
        block_count: usize,
    },
    /// A phi operand names a predecessor block the function does not hold.
    PhiPredecessorOutOfRange {
        /// Block holding the phi.
        block: usize,
        /// Index of the phi in that block's phi list.
        phi_idx: usize,
        /// Predecessor the operand names.
        pred: usize,
        /// Number of blocks the function holds.
        block_count: usize,
    },
    /// A variable id names no row of the variable table.
    ///
    /// The table is dense (see [`VariableTableNotDense`]), so an id at or above
    /// its length resolves to nothing at all: every structure a pass sizes by
    /// the variable count is indexed by ids read out of the IR, and one past
    /// the end is a panic rather than a wrong answer.
    ///
    /// [`VariableTableNotDense`]: Self::VariableTableNotDense
    VariableIdOutOfRange {
        /// The id that names no variable.
        var: SsaVarId,
        /// Where the function names it.
        site: VarRefSite,
        /// Number of variables the function holds.
        var_count: usize,
    },
    /// A variable's definition site names a block or instruction that is not
    /// there.
    DefSiteOutOfRange {
        /// Variable whose definition site is out of range.
        var: SsaVarId,
        /// The site as recorded.
        site: DefSite,
        /// The bound it runs past.
        bound: SiteBound,
    },
    /// A variable's use list names a block, instruction or phi that is not
    /// there.
    UseSiteOutOfRange {
        /// Variable whose use site is out of range.
        var: SsaVarId,
        /// Block the site names.
        block: usize,
        /// Instruction or phi index the site names.
        index: usize,
        /// Whether the site names a phi rather than an instruction.
        is_phi_operand: bool,
        /// The bound it runs past.
        bound: SiteBound,
    },
    /// An exception clause cannot be laid out over the function's blocks.
    MalformedExceptionClause {
        /// Index of the clause in `SsaFunction::exception_handlers`.
        clause: usize,
        /// What is wrong with it.
        reason: ExceptionTableError,
    },
}

impl VerifierError {
    /// Whether the finding leaves the function's meaning intact: a variable
    /// registered with no definition and no reader, a row compaction collects.
    ///
    /// Pruning an unreachable block unregisters nothing, so a host that
    /// constructs with compaction off is left one for each value the block
    /// defined. A read with no definition is [`Self::UndefinedUse`], which is
    /// not cosmetic.
    #[must_use]
    pub fn is_cosmetic(&self) -> bool {
        matches!(self, Self::OrphanVariable { .. })
    }
}

impl fmt::Display for VerifierError {
    /// Formats the error as a human-readable diagnostic naming the offending
    /// block, instruction, and variable where applicable.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndefinedUse {
                block,
                instr_idx,
                var,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} uses undefined variable {var:?}"
            ),
            Self::MissingPhiOperand {
                block,
                phi_idx,
                missing_pred,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} missing operand for predecessor {missing_pred}"
            ),
            Self::ExtraPhiOperand {
                block,
                phi_idx,
                extra_pred,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} has operand for non-predecessor {extra_pred}"
            ),
            Self::DuplicateDefinition { var, def1, def2 } => write!(
                f,
                "Variable {var:?} defined twice: at block {} ({:?}) and block {} ({:?})",
                def1.block, def1.kind, def2.block, def2.kind
            ),
            Self::OrphanVariable { var } => {
                write!(
                    f,
                    "Variable {var:?} in variables vec but not defined in any block"
                )
            }
            Self::UnregisteredVariable { var } => write!(
                f,
                "Variable {var:?} used in instruction but not in variables vec"
            ),
            Self::StaleUseList {
                var,
                recorded,
                actual,
            } => write!(
                f,
                "Variable {var:?}: use list records {recorded} use(s) that do not match the {actual} in the IR"
            ),
            Self::MissingTerminator { block } => {
                write!(
                    f,
                    "Block {block}: control falls off the end - last instruction is not a terminator"
                )
            }
            Self::EntryHasPredecessors { predecessor } => {
                write!(
                    f,
                    "Block {predecessor}: terminator transfers control to the entry block"
                )
            }
            Self::PhiWithoutPredecessors { block, phi_idx } => {
                write!(
                    f,
                    "Block {block}: phi {phi_idx} in a block no terminator transfers to"
                )
            }
            Self::DominanceViolation {
                var,
                def_block,
                use_block,
            } => write!(
                f,
                "Variable {var:?}: def in block {def_block} does not dominate use in block {use_block}"
            ),
            Self::TerminatorNotLast {
                block,
                instr_idx,
                instr_count,
            } => write!(
                f,
                "Block {block}: terminator at position {instr_idx}/{instr_count} is not last"
            ),
            Self::IntraBlockCycle {
                block,
                use_instr,
                def_instr,
                var,
            } => write!(
                f,
                "Block {block}: instruction {use_instr} uses {var:?} defined at instruction {def_instr}"
            ),
            Self::DuplicatePhiOperand {
                block,
                phi_idx,
                pred,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} has multiple operands for predecessor {pred}"
            ),
            Self::VariableTableNotDense { slot, found } => write!(
                f,
                "Variable table slot {slot} holds {found:?}; `variable(id)` indexes this table \
                 directly, so every id at or above {slot} now resolves to the wrong variable"
            ),
            Self::PlaceholderVariable { block, location } => write!(
                f,
                "Block {block}: placeholder variable ID (usize::MAX) at {location}"
            ),
            Self::SelfReferentialInstruction {
                block,
                instr_idx,
                var,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has self-referential use of {var:?}"
            ),
            Self::InvalidVectorOperation {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid vector op: {reason}"
            ),
            Self::InvalidAtomicOperation {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid atomic op: {reason}"
            ),
            Self::InvalidWideArithmetic {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid wide arithmetic op: {reason}"
            ),
            Self::InvalidOperandWidths {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} computes over mismatched widths: {reason}"
            ),
            Self::InvalidConversion {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid conversion: {reason}"
            ),
            Self::InvalidNativeOperation {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid native op: {reason}"
            ),
            Self::MalformedExceptionClause { clause, reason } => {
                write!(f, "Exception clause {clause}: {reason}")
            }
            Self::DuplicateInstructionId { id, first, second } => write!(
                f,
                "Instruction identity {id} is held by both block {}, instruction {} and block {}, \
                 instruction {}",
                first.0, first.1, second.0, second.1
            ),
            Self::UnassignedInstructionId { block, instr_idx } => write!(
                f,
                "Block {block}: instruction {instr_idx} has no identity; a repair boundary assigns \
                 one"
            ),
            Self::ExceptionalOperandAtNonRoot { block, phi_idx } => write!(
                f,
                "Block {block}: phi {phi_idx} carries an operand for a throw, but nothing can \
                 throw here"
            ),
            Self::ExceptionalOperandFromNonThrowingBlock {
                block,
                phi_idx,
                source,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} names a throw in block {source}, which cannot \
                 dispatch here"
            ),
            Self::MissingExceptionalOperand {
                block,
                phi_idx,
                source,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} accounts for no value from block {source}, which \
                 can throw here"
            ),
            Self::DuplicateExceptionalOperand {
                block,
                phi_idx,
                source,
                value,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} names {value:?} from a throw in block {source} \
                 twice"
            ),
            Self::ExceptionalOperandFromAnotherGroup {
                block,
                phi_idx,
                value,
                group,
                expected,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} merges rename group {expected}, but names \
                 {value:?} from group {group} for a throw; a handler's operands must share one \
                 storage location"
            ),
            Self::MisplacedRuntimeValue {
                block,
                instr_idx,
                kind,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} defines the runtime-supplied {kind} here, \
                 but {reason}"
            ),
            Self::BlockTableNotDense { slot, found } => write!(
                f,
                "Block list slot {slot} holds a block recording id {found}; blocks are addressed \
                 by position, so the two must agree"
            ),
            Self::DanglingBranchTarget {
                block,
                instr_idx,
                target,
                block_count,
            } => write!(
                f,
                "Block {block}: terminator at instruction {instr_idx} names block {target}, but \
                 the function holds {block_count} block(s)"
            ),
            Self::PhiPredecessorOutOfRange {
                block,
                phi_idx,
                pred,
                block_count,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} names predecessor {pred}, but the function holds \
                 {block_count} block(s)"
            ),
            Self::VariableIdOutOfRange {
                var,
                site,
                var_count,
            } => write!(
                f,
                "{var:?} names no variable ({site}); the function holds {var_count} variable(s)"
            ),
            Self::DefSiteOutOfRange { var, site, bound } => write!(
                f,
                "{var:?} is defined at block {}, instruction {:?}, but {bound}",
                site.block(),
                site.index()
            ),
            Self::UseSiteOutOfRange {
                var,
                block,
                index,
                is_phi_operand,
                bound,
            } => write!(
                f,
                "{var:?} records a use at block {block}, {} {index}, but {bound}",
                if *is_phi_operand {
                    "phi"
                } else {
                    "instruction"
                }
            ),
        }
    }
}

impl Error for VerifierError {}

/// Verification depth levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerifyLevel {
    /// Single-definition + block structure checks (O(n)).
    Quick,
    /// + def-use chains + phi operand coverage (O(n*m)).
    Standard,
    /// + dominance checking (O(n^2) worst case).
    Full,
}

/// The checks, in the order their findings are reported.
///
/// One scan of the function feeds every check at once, so findings arrive
/// interleaved; each check collects its own and [`Findings::into_errors`] lays
/// them out check by check, which is the order a caller reads them in.
#[derive(Debug, Clone, Copy)]
enum Check {
    /// Every index the function names exists.
    IndexBounds,
    /// Instruction identities are unique.
    InstructionIdentities,
    /// Every block records the position it occupies.
    BlockTable,
    /// The variable table is densely indexed by variable id.
    VariableTable,
    /// Every variable is defined at most once.
    SingleDefinition,
    /// Terminators close their blocks and nothing reads a value before its
    /// definition inside a block.
    BlockStructure,
    /// No terminator names the entry block.
    EntryPredecessors,
    /// No placeholder id or self-referential instruction remains.
    Placeholders,
    /// Every exception clause can be laid out over the blocks.
    ExceptionTable,
    /// Every runtime-supplied value stands where the runtime supplies it.
    RuntimeValues,
    /// Every value something reads is defined.
    ReadsHaveDefinitions,
    /// Every phi has the operands its block's predecessors call for.
    PhiOperands,
    /// Every instruction operand has a definition.
    DefinedBeforeUse,
    /// Registered variables and defined variables agree.
    RegisteredVariables,
    /// Recorded use lists match the IR.
    UseLists,
    /// Vector operations have consistent shapes.
    VectorOperations,
    /// Atomic operations have consistent ordering, width and outputs.
    AtomicOperations,
    /// Wide arithmetic has matching half widths.
    WideArithmetic,
    /// Scalar arithmetic, logic, comparisons and selects compute at one width.
    OperandWidths,
    /// Conversions keep their widths, state their sign once and convert the
    /// kinds of value they name.
    Conversions,
    /// Effect summaries are consistent.
    NativeEffects,
    /// Flag-selecting operations name defined flags, and a float raise is of
    /// a computation and sets only flags its layout keeps.
    FlagMasks,
    /// Every use is dominated by its definition.
    Dominance,
    /// Every instruction carries an identity.
    IdentitiesAssigned,
}

/// How many checks there are; sizes [`Findings`].
const CHECK_COUNT: usize = 24;

// A check beyond the last slot would have its findings dropped.
const _: () = assert!(Check::IdentitiesAssigned as usize + 1 == CHECK_COUNT);

/// What each check found, kept apart until the scan ends.
struct Findings {
    /// One list per [`Check`], indexed by its position in the enum.
    by_check: [Vec<VerifierError>; CHECK_COUNT],
}

impl Findings {
    /// Creates an empty record.
    fn new() -> Self {
        Self {
            by_check: std::array::from_fn(|_| Vec::new()),
        }
    }

    /// Records `error` under `check`.
    fn push(&mut self, check: Check, error: VerifierError) {
        if let Some(found) = self.by_check.get_mut(check as usize) {
            found.push(error);
        }
    }

    /// Lays the findings out check by check, each in discovery order.
    fn into_errors(self) -> Vec<VerifierError> {
        let total: usize = self.by_check.iter().map(Vec::len).sum();
        let mut errors = Vec::with_capacity(total);
        for found in self.by_check {
            errors.extend(found);
        }
        errors
    }
}

/// Marks a block or index that [`VarFacts`] has not been given.
const ABSENT: u32 = u32::MAX;

/// What the scan has learned about one variable id.
///
/// Held per id in a table the scan touches for every operand, so the fields are
/// narrow: the table is the largest thing the scan allocates.
#[derive(Clone, Copy)]
struct VarFacts {
    /// The block of the first phi or instruction, in block order, that defines
    /// it. Phis come before instructions within a block.
    def_block: u32,
    /// Its index among that block's phis or instructions.
    def_index: u32,
    /// Whether it is a phi.
    def_is_phi: bool,
    /// The block of the first read of it. A phi operand read is recorded at
    /// instruction `0`.
    read_block: u32,
    /// The instruction of that read.
    read_index: u32,
    /// The block of its latest instruction definition seen so far. A block that
    /// is not the one being scanned marks the entry stale, which is what lets
    /// one table serve every block without being cleared.
    stamp_block: u32,
    /// The instruction of that definition.
    stamp_index: u32,
    /// The block its variable-table row names as its definition, when the table
    /// has a row for the id. Present is what "registered" means.
    row_block: u32,
}

impl Default for VarFacts {
    fn default() -> Self {
        Self {
            def_block: ABSENT,
            def_index: 0,
            def_is_phi: false,
            read_block: ABSENT,
            read_index: 0,
            stamp_block: ABSENT,
            stamp_index: 0,
            row_block: ABSENT,
        }
    }
}

impl VarFacts {
    /// Narrows a block or instruction index to the width the fields store.
    ///
    /// The indices of a function's own blocks and instructions are far below
    /// the limit. One a malformed variable row names saturates just short of
    /// [`ABSENT`], which is as far past the end of the function as it was.
    fn narrow(index: usize) -> u32 {
        u32::try_from(index).map_or(ABSENT.saturating_sub(1), |index| {
            index.min(ABSENT.saturating_sub(1))
        })
    }

    /// Whether the variable table has a row for the id.
    const fn is_registered(&self) -> bool {
        self.row_block != ABSENT
    }

    /// Whether some phi or instruction defines the id.
    const fn is_defined(&self) -> bool {
        self.def_block != ABSENT
    }

    /// Whether anything reads the id.
    const fn is_read(&self) -> bool {
        self.read_block != ABSENT
    }

    /// The first `(block, instruction)` reading the id.
    fn first_read(&self) -> Option<(usize, usize)> {
        self.is_read()
            .then_some((self.read_block as usize, self.read_index as usize))
    }

    /// The block where the id is defined: the block its variable-table row
    /// records, or failing that the first block that defines it.
    fn definition_block(&self) -> Option<usize> {
        match (self.is_registered(), self.is_defined()) {
            (true, _) => Some(self.row_block as usize),
            (false, true) => Some(self.def_block as usize),
            (false, false) => None,
        }
    }

    /// Where the id is first defined.
    fn first_def(&self) -> Option<VerifierDefSite> {
        self.is_defined().then_some(VerifierDefSite {
            block: self.def_block as usize,
            kind: if self.def_is_phi {
                DefKind::Phi(self.def_index as usize)
            } else {
                DefKind::Instruction(self.def_index as usize)
            },
        })
    }
}

/// One occurrence of a variable in an operand, for the use-list comparison.
#[derive(Clone, Copy)]
struct UseRecord {
    /// The variable read.
    var: SsaVarId,
    /// The block holding the reader.
    block: usize,
    /// The instruction or phi index of the reader.
    index: usize,
    /// Whether the reader is a phi operand.
    is_phi: bool,
}

/// The operands of one block's instructions, read out of the operations once.
///
/// Every check that wants an instruction's definitions or uses takes them from
/// here, so a block's operations are walked once however many checks look at
/// them. The buffers are reused from block to block.
#[derive(Default)]
struct BlockOperands {
    /// Every definition of every instruction, in instruction order.
    defs: Vec<SsaVarId>,
    /// Every use of every instruction, in instruction order.
    uses: Vec<SsaVarId>,
    /// Where each instruction's definitions and uses end in the two lists.
    ends: Vec<(usize, usize)>,
}

impl BlockOperands {
    /// Reads the operands of `block`'s instructions into the buffers.
    fn fill<T: Target>(&mut self, block: &SsaBlock<T>) {
        self.defs.clear();
        self.uses.clear();
        self.ends.clear();
        for instr in block.instructions() {
            instr.op().visit_operands(|role, var| match role {
                OperandRole::Use => self.uses.push(var),
                OperandRole::Def | OperandRole::FlagsDef => self.defs.push(var),
            });
            self.ends.push((self.defs.len(), self.uses.len()));
        }
    }

    /// Returns the definitions and uses of instruction `instr_idx`, given where
    /// the previous instruction's ended.
    ///
    /// # Arguments
    ///
    /// * `instr_idx` - The instruction asked about.
    /// * `start` - Where the previous instruction's definitions and uses ended;
    ///   updated to where this one's end.
    fn of(&self, instr_idx: usize, start: &mut (usize, usize)) -> (&[SsaVarId], &[SsaVarId]) {
        let end = self.ends.get(instr_idx).copied().unwrap_or(*start);
        let defs = self.defs.get(start.0..end.0).unwrap_or_default();
        let uses = self.uses.get(start.1..end.1).unwrap_or_default();
        *start = end;
        (defs, uses)
    }
}

/// The instruction identities seen so far, and where.
///
/// Identities are issued by counting up and never reissued, so a function that
/// has been edited for a long time holds ids far above its instruction count.
/// A bit per id answers "seen before" without a table that size; the site of an
/// earlier occurrence is only wanted when there is one to report, and a
/// well-formed function has none.
#[derive(Default)]
struct InstructionIdentities {
    /// The ids below [`Self::DENSE_LIMIT`] seen so far.
    seen: BitSet,
    /// The latest site of each dense id seen more than once.
    repeated: FxHashMap<u32, (usize, usize)>,
    /// The latest site of each id at or above [`Self::DENSE_LIMIT`].
    sparse: FxHashMap<u32, (usize, usize)>,
}

impl InstructionIdentities {
    /// Ids below this are tracked by a bit; a malformed id above it must not
    /// size an allocation.
    const DENSE_LIMIT: usize = 1 << 22;
}

/// Everything one scan of a function checks.
///
/// The scan walks the blocks once. Each phi and each instruction is handed to
/// every check that looks at it, and the checks that need a fact about the
/// whole function - which variables are defined anywhere, who reads what - note
/// what they saw and conclude when the walk is over. Findings are kept per
/// check, so the report reads as if each check had made its own pass.
struct Scan<'a, T: Target> {
    /// The function being checked.
    ssa: &'a SsaFunction<T>,
    /// Whether the Quick checks run; false for the index-bounds-only scan.
    quick: bool,
    /// Whether the Standard checks run.
    standard: bool,
    /// Whether the Full checks run.
    full: bool,
    /// What each check found.
    found: Findings,
    /// What the scan knows of each variable id.
    vars: VarTable<VarFacts>,
    /// Number of rows in the variable table.
    var_count: usize,
    /// Number of blocks.
    block_count: usize,
    /// Whether the variable table has a row that carries the placeholder id.
    ///
    /// The placeholder is not a variable and no table is indexed by it, but a
    /// row can be given its id, and a definition of the placeholder is then
    /// registered as much as any other definition that has a row.
    placeholder_has_row: bool,
    /// Instruction identities seen so far.
    identities: InstructionIdentities,
    /// Every operand occurrence of a registered variable, for the use-list
    /// comparison.
    use_records: Vec<UseRecord>,
    /// Instruction operands naming an id the variable table has no row for, as
    /// `(block, instruction, variable)`. Whether a block defines one is known
    /// only once every block has been seen.
    unregistered_uses: Vec<(usize, usize, SsaVarId)>,
    /// The current block's operands.
    operands: BlockOperands,
    /// The control-flow graph, built for the Standard checks.
    cfg: Option<SsaCfg<'a, T>>,
    /// Blocks that start a catch or filter clause.
    exception_entries: BTreeSet<usize>,
    /// Blocks that start any handler clause.
    handler_entries: BTreeSet<usize>,
    /// Scratch for the phi-operand check: the block's predecessors.
    preds: Vec<usize>,
    /// Scratch for the phi-operand check: one phi's normal operand predecessors
    /// in operand order.
    operand_preds: Vec<usize>,
    /// Scratch for the phi-operand check: the same, sorted and unique.
    operand_pred_set: Vec<usize>,
    /// Scratch for the phi-operand check: `(predecessor, position)` pairs.
    operand_positions: Vec<(usize, usize)>,
}

impl<'a, T: Target> Scan<'a, T> {
    /// Prepares a scan of `ssa` that runs the checks `depth` calls for.
    fn new(ssa: &'a SsaFunction<T>, depth: Depth) -> Self {
        let (quick, standard, full) = match depth {
            Depth::IndexBounds => (false, false, false),
            Depth::Level(level) => (
                true,
                level >= VerifyLevel::Standard,
                level >= VerifyLevel::Full,
            ),
        };
        let var_count = ssa.variables().len();
        Self {
            ssa,
            quick,
            standard,
            full,
            found: Findings::new(),
            vars: VarTable::new(var_count),
            var_count,
            block_count: ssa.block_count(),
            placeholder_has_row: false,
            identities: InstructionIdentities::default(),
            use_records: Vec::new(),
            unregistered_uses: Vec::new(),
            operands: BlockOperands::default(),
            cfg: None,
            exception_entries: BTreeSet::new(),
            handler_entries: BTreeSet::new(),
            preds: Vec::new(),
            operand_preds: Vec::new(),
            operand_pred_set: Vec::new(),
            operand_positions: Vec::new(),
        }
    }

    /// Runs the scan and returns every finding, check by check.
    fn run(mut self) -> Vec<VerifierError> {
        self.prepare();
        let ssa = self.ssa;
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            self.scan_block(block_idx, block);
        }
        self.conclude();
        self.found.into_errors()
    }

    /// Records what the checks need to know before the first block is read.
    fn prepare(&mut self) {
        for var in self.ssa.variables() {
            if let Some(facts) = self.vars.entry(var.id()) {
                facts.row_block = VarFacts::narrow(var.def_site().block());
            }
            self.placeholder_has_row |= var.id().is_placeholder();
        }
        if self.standard {
            self.cfg = Some(SsaCfg::from_ssa(self.ssa));
        }
        if self.quick {
            for handler in self.ssa.exception_handlers() {
                self.handler_entries.extend(handler.entry_blocks());
                if handler.receives_exception() {
                    self.exception_entries.extend(handler.entry_blocks());
                }
            }
        }
    }

    /// Hands one block to every check.
    fn scan_block(&mut self, block_idx: usize, block: &SsaBlock<T>) {
        let mut operands = mem::take(&mut self.operands);
        operands.fill(block);
        self.scan_edges(block_idx, block);
        self.scan_phis(block_idx, block);
        self.scan_instructions(block_idx, block, &operands);
        self.operands = operands;
    }

    /// Checks the block's record of its position and the blocks it transfers
    /// to.
    fn scan_edges(&mut self, block_idx: usize, block: &SsaBlock<T>) {
        if self.quick && block.id() != block_idx {
            self.found.push(
                Check::BlockTable,
                VerifierError::BlockTableNotDense {
                    slot: block_idx,
                    found: block.id(),
                },
            );
        }
        let block_count = self.block_count;
        let terminator_idx = block.instruction_count().saturating_sub(1);
        let mut names_entry = false;
        let found = &mut self.found;
        block.for_each_successor(|target| {
            if target >= block_count {
                found.push(
                    Check::IndexBounds,
                    VerifierError::DanglingBranchTarget {
                        block: block_idx,
                        instr_idx: terminator_idx,
                        target,
                        block_count,
                    },
                );
            }
            names_entry |= target == 0;
        });
        if self.quick && names_entry {
            self.found.push(
                Check::EntryPredecessors,
                VerifierError::EntryHasPredecessors {
                    predecessor: block_idx,
                },
            );
        }
    }

    /// Reports `var` if it is an id the variable table cannot hold.
    fn note_var(&mut self, var: SsaVarId, site: VarRefSite) {
        if var != SsaVarId::PLACEHOLDER && var.index() >= self.var_count {
            self.found.push(
                Check::IndexBounds,
                VerifierError::VariableIdOutOfRange {
                    var,
                    site,
                    var_count: self.var_count,
                },
            );
        }
    }

    /// Hands one definition to the checks that look at definitions.
    ///
    /// Records where `var` is first defined and reports a second definition,
    /// reports a definition of an id the variable table has no row for, and
    /// stamps an instruction definition for the use-before-definition check.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable defined.
    /// * `block_idx` - The block of the definition.
    /// * `index` - Its index among the block's phis, or among its instructions.
    /// * `is_phi` - Whether a phi defines it.
    fn scan_def(&mut self, var: SsaVarId, block_idx: usize, index: usize, is_phi: bool) {
        let Some(facts) = self.vars.entry(var) else {
            // The placeholder is no variable, so nothing is recorded about it;
            // a definition of it is registered only if a table row carries it.
            if self.standard && !self.placeholder_has_row {
                self.found.push(
                    Check::RegisteredVariables,
                    VerifierError::UnregisteredVariable { var },
                );
            }
            return;
        };
        if facts.is_defined() {
            let site = VerifierDefSite {
                block: block_idx,
                kind: if is_phi {
                    DefKind::Phi(index)
                } else {
                    DefKind::Instruction(index)
                },
            };
            if let Some(first) = facts.first_def() {
                self.found.push(
                    Check::SingleDefinition,
                    VerifierError::DuplicateDefinition {
                        var,
                        def1: first,
                        def2: site,
                    },
                );
            }
        } else {
            facts.def_block = VarFacts::narrow(block_idx);
            facts.def_index = VarFacts::narrow(index);
            facts.def_is_phi = is_phi;
        }
        if !is_phi {
            facts.stamp_block = VarFacts::narrow(block_idx);
            facts.stamp_index = VarFacts::narrow(index);
        }
        if self.standard && !facts.is_registered() {
            self.found.push(
                Check::RegisteredVariables,
                VerifierError::UnregisteredVariable { var },
            );
        }
    }

    /// Hands one read to the checks that look at reads.
    ///
    /// Records where `var` is first read, notes the occurrence for the use-list
    /// comparison, and sets aside an instruction operand naming an id the
    /// variable table has no row for.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable read.
    /// * `block_idx` - The block of the reader.
    /// * `index` - The reader's index among the block's phis, or instructions.
    /// * `is_phi` - Whether the reader is a phi.
    fn scan_use(&mut self, var: SsaVarId, block_idx: usize, index: usize, is_phi: bool) {
        let Some(facts) = self.vars.entry(var) else {
            if self.standard && !is_phi {
                self.unregistered_uses.push((block_idx, index, var));
            }
            return;
        };
        if !facts.is_read() {
            facts.read_block = VarFacts::narrow(block_idx);
            facts.read_index = VarFacts::narrow(if is_phi { 0 } else { index });
        }
        if self.standard {
            if facts.is_registered() {
                self.use_records.push(UseRecord {
                    var,
                    block: block_idx,
                    index,
                    is_phi,
                });
            } else if !is_phi {
                self.unregistered_uses.push((block_idx, index, var));
            }
        }
    }

    /// Hands the block's phis to every check.
    fn scan_phis(&mut self, block_idx: usize, block: &SsaBlock<T>) {
        for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
            let result = phi.result();
            self.note_var(
                result,
                VarRefSite::PhiResult {
                    block: block_idx,
                    phi_idx,
                },
            );
            if self.quick && result.is_placeholder() {
                self.found.push(
                    Check::Placeholders,
                    VerifierError::PlaceholderVariable {
                        block: block_idx,
                        location: format!("phi {phi_idx} result"),
                    },
                );
            }
            for operand in phi.operands() {
                let pred = operand.predecessor();
                let value = operand.value();
                if pred >= self.block_count {
                    self.found.push(
                        Check::IndexBounds,
                        VerifierError::PhiPredecessorOutOfRange {
                            block: block_idx,
                            phi_idx,
                            pred,
                            block_count: self.block_count,
                        },
                    );
                }
                self.note_var(
                    value,
                    VarRefSite::PhiOperand {
                        block: block_idx,
                        phi_idx,
                        pred,
                    },
                );
                if self.quick {
                    if value.is_placeholder() {
                        self.found.push(
                            Check::Placeholders,
                            VerifierError::PlaceholderVariable {
                                block: block_idx,
                                location: format!("phi {phi_idx} operand from B{pred}"),
                            },
                        );
                    }
                    self.scan_use(value, block_idx, phi_idx, true);
                }
            }
            if self.quick {
                self.scan_def(result, block_idx, phi_idx, true);
            }
        }
        if self.standard {
            self.check_phi_operands(block_idx, block);
        }
    }

    /// Hands the block's instructions to every check.
    fn scan_instructions(
        &mut self,
        block_idx: usize,
        block: &SsaBlock<T>,
        operands: &BlockOperands,
    ) {
        let instrs = block.instructions();
        let instr_count = instrs.len();
        let mut cursor = (0, 0);
        let mut any_terminator = false;
        // Whether only runtime values have been seen so far in this block.
        let mut leading = true;

        for (instr_idx, instr) in instrs.iter().enumerate() {
            let op = instr.op();
            let (defs, uses) = operands.of(instr_idx, &mut cursor);

            if self.quick {
                let is_terminator = op.is_terminator();
                any_terminator |= is_terminator;
                self.check_instruction_identity(block_idx, instr_idx, instr);
                if is_terminator && instr_idx < instr_count.saturating_sub(1) {
                    self.found.push(
                        Check::BlockStructure,
                        VerifierError::TerminatorNotLast {
                            block: block_idx,
                            instr_idx,
                            instr_count,
                        },
                    );
                }
                self.check_runtime_value(block_idx, instr_idx, op, &mut leading);
            }

            for &dest in defs {
                self.note_var(
                    dest,
                    VarRefSite::Definition {
                        block: block_idx,
                        instr_idx,
                    },
                );
                if self.quick {
                    if dest.is_placeholder() {
                        self.found.push(
                            Check::Placeholders,
                            VerifierError::PlaceholderVariable {
                                block: block_idx,
                                location: format!("instruction {instr_idx} dest"),
                            },
                        );
                    }
                    // A destination among the operands is a self-referential
                    // instruction.
                    if uses.contains(&dest) {
                        self.found.push(
                            Check::Placeholders,
                            VerifierError::SelfReferentialInstruction {
                                block: block_idx,
                                instr_idx,
                                var: dest,
                            },
                        );
                    }
                    self.scan_def(dest, block_idx, instr_idx, false);
                }
            }
            for &used in uses {
                self.note_var(
                    used,
                    VarRefSite::Operand {
                        block: block_idx,
                        instr_idx,
                    },
                );
                if self.quick {
                    if used.is_placeholder() {
                        self.found.push(
                            Check::Placeholders,
                            VerifierError::PlaceholderVariable {
                                block: block_idx,
                                location: format!("instruction {instr_idx} operand"),
                            },
                        );
                    }
                    self.scan_use(used, block_idx, instr_idx, false);
                }
            }

            if self.standard {
                OperationScan {
                    ssa: self.ssa,
                    found: &mut self.found,
                }
                .run(block_idx, instr_idx, op);
            }
        }

        if !self.quick {
            return;
        }

        // Control must leave a non-empty block the way the successor models
        // read it leaving: through the last instruction.
        if block.control_terminator().is_none() && !instrs.is_empty() && !any_terminator {
            self.found.push(
                Check::BlockStructure,
                VerifierError::MissingTerminator { block: block_idx },
            );
        }
        self.check_intra_block_order(block_idx, operands);
    }

    /// Verifies instruction identities are unique, and present once the
    /// function has been through a repair.
    ///
    /// Uniqueness is Quick: a host table keyed by identity is misdirected by a
    /// duplicate exactly as a variable table with a hole is. Presence is
    /// [`VerifyLevel::Full`], because an unidentified instruction is a function
    /// that is not finished rather than one that is wrong - identity comes from
    /// the function's allocator, so a block built through the raw surface has
    /// none to give until it joins one, and every repair boundary stamps
    /// whatever is still unstamped.
    fn check_instruction_identity(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        instr: &SsaInstruction<T>,
    ) {
        let Some(id) = instr.id() else {
            if self.full {
                self.found.push(
                    Check::IdentitiesAssigned,
                    VerifierError::UnassignedInstructionId {
                        block: block_idx,
                        instr_idx,
                    },
                );
            }
            return;
        };
        let site = (block_idx, instr_idx);
        let raw = id.get();
        let previous = match usize::try_from(raw) {
            Ok(index) if index < InstructionIdentities::DENSE_LIMIT => {
                self.identities.seen.ensure_capacity(index);
                if self.identities.seen.insert(index) {
                    None
                } else {
                    // The earlier occurrence is the latest one recorded, or the
                    // only one so far, which a second look at the function finds.
                    self.identities
                        .repeated
                        .insert(raw, site)
                        .or_else(|| self.first_site_of(id))
                }
            }
            _ => self.identities.sparse.insert(raw, site),
        };
        if let Some(first) = previous {
            self.found.push(
                Check::InstructionIdentities,
                VerifierError::DuplicateInstructionId {
                    id,
                    first,
                    second: site,
                },
            );
        }
    }

    /// Returns where the first instruction carrying `id` stands.
    ///
    /// Only asked once an id has turned up twice, which is a defect, so the
    /// search is paid by the broken function and no other.
    fn first_site_of(&self, id: InstrId) -> Option<(usize, usize)> {
        self.ssa
            .blocks()
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

    /// Verifies a runtime-supplied value is defined where the runtime supplies
    /// it.
    ///
    /// [`SsaOp::RuntimeValue`] is the one definition whose justification is its
    /// position: the value holds because control arrived at *this* block from
    /// the runtime, before anything in the block ran. Two positions break that,
    /// and the error says which, because the repairs differ - a block that
    /// cannot be entered that way is a definition in the wrong place, while a
    /// non-leading one is an edit that slipped an instruction above it.
    ///
    /// The kind decides which blocks qualify. An exception object is a value
    /// where its handler is entered holding it
    /// ([`crate::Target::handler_receives_exception`]):
    /// in the CLR a catch handler or a filter, and not a finally or a fault,
    /// which run for an exit and have no object to read; an Itanium landing
    /// pad for a cleanup too. Restored handler state, by contrast, is a fact
    /// about entering any handler, so every clause's entry qualifies for it.
    ///
    /// Quick, not Standard: a misplaced definition misleads every later check
    /// that reads it as reaching its uses.
    ///
    /// [`SsaOp::RuntimeValue`]: crate::ir::ops::SsaOp::RuntimeValue
    ///
    /// # Arguments
    ///
    /// * `leading` - Whether every instruction before this one in the block was
    ///   a runtime value; cleared by the first that is not.
    fn check_runtime_value(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        op: &SsaOp<T>,
        leading: &mut bool,
    ) {
        let SsaOp::RuntimeValue { kind, .. } = op else {
            *leading = false;
            return;
        };
        let entries = match kind {
            RuntimeValueKind::ExceptionObject => &self.exception_entries,
            RuntimeValueKind::HandlerEntry => &self.handler_entries,
            // A choice the implementation makes where control reaches it
            // stands anywhere, and ends the values a block is entered with.
            RuntimeValueKind::Unspecified => {
                *leading = false;
                return;
            }
        };
        let reason = if !entries.contains(&block_idx) {
            Some(RuntimeValuePlacement::NotARuntimeEntry)
        } else if *leading {
            None
        } else {
            Some(RuntimeValuePlacement::NotLeading)
        };
        if let Some(reason) = reason {
            self.found.push(
                Check::RuntimeValues,
                VerifierError::MisplacedRuntimeValue {
                    block: block_idx,
                    instr_idx,
                    kind: *kind,
                    reason,
                },
            );
        }
    }

    /// Checks a block for use-before-def cycles.
    ///
    /// Every instruction definition of the block has been stamped on its
    /// variable's facts by the time this runs, with the latest definition
    /// winning. The stamp carries the block index rather than being cleared: an
    /// entry whose stamp is not the current block is stale. Clearing a
    /// variable-sized table per block would be O(variables x blocks), and a
    /// fresh map per block costs one allocation per block on every verify.
    fn check_intra_block_order(&mut self, block_idx: usize, operands: &BlockOperands) {
        let stamp = VarFacts::narrow(block_idx);
        let mut cursor = (0, 0);
        for instr_idx in 0..operands.ends.len() {
            let (_, uses) = operands.of(instr_idx, &mut cursor);
            for &used in uses {
                if let Some(facts) = self.vars.get(used)
                    && facts.stamp_block == stamp
                    && facts.stamp_index as usize >= instr_idx
                {
                    self.found.push(
                        Check::BlockStructure,
                        VerifierError::IntraBlockCycle {
                            block: block_idx,
                            use_instr: instr_idx,
                            def_instr: facts.stamp_index as usize,
                            var: used,
                        },
                    );
                }
            }
        }
    }

    /// Checks that every phi node has the correct operand set: one operand per
    /// CFG predecessor, no operands from non-predecessor blocks, and no phis in
    /// the entry block (which has no predecessors).
    ///
    /// Predecessors come from the CFG while operand predecessors come from the
    /// IR, and the two are compared as sorted lists: a malformed phi may name a
    /// block that does not exist, so nothing here is indexed by a block id.
    /// Duplicates are counted rather than set-tested, because two operands
    /// naming one predecessor collapse to a single entry in a set and a plain
    /// set comparison would miss the "exactly one operand per predecessor"
    /// violation that edge redirection during block merging can introduce. A
    /// block with no predecessors at all (the entry, or a handler entry reached
    /// only by runtime dispatch) has no edge for any operand to name, so each
    /// of its phis is reported directly.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block whose phis are checked.
    /// * `block` - The block itself.
    fn check_phi_operands(&mut self, block_idx: usize, block: &SsaBlock<T>) {
        if block.phi_nodes().is_empty() {
            return;
        }
        let Some(cfg) = self.cfg.take() else {
            return;
        };
        // Normal operands name terminator edges, so that is the relation they
        // are checked against; the exceptional ones are checked against the
        // blocks that can throw here, just below.
        let throwing = cfg.roots().exceptional_predecessors(block_idx);
        self.preds.clear();
        self.preds
            .extend_from_slice(cfg.terminator_predecessors(block_idx));
        self.preds.sort_unstable();
        self.preds.dedup();

        for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
            // A phi operand names an incoming edge, of which there are two
            // kinds: a terminator transfer, and a throw inside a protected
            // region dispatching here. With neither, there is nothing for an
            // operand to name - and a zero-operand phi in such a block, which
            // the operand checks below cannot see, is reported here too.
            if self.preds.is_empty() && throwing.is_empty() {
                self.found.push(
                    Check::PhiOperands,
                    VerifierError::PhiWithoutPredecessors {
                        block: block_idx,
                        phi_idx,
                    },
                );
                continue;
            }

            self.check_exceptional_operands(block_idx, phi_idx, phi, throwing);

            // Counted, not set-tested: two operands naming one predecessor
            // collapse to a single entry in a set, so a set comparison cannot
            // see the duplicate and the advertised "exactly one operand per
            // predecessor" invariant goes unchecked. `PhiNode::add_operand`
            // does not dedup and `PhiOperand::set_predecessor` allows a
            // retarget onto an edge that already has an operand, which is what
            // edge redirection during block merging does.
            self.operand_preds.clear();
            self.operand_preds
                .extend(phi.normal_operands().map(PhiOperand::predecessor));
            self.report_duplicate_operands(block_idx, phi_idx);

            self.operand_pred_set.clear();
            self.operand_pred_set.extend_from_slice(&self.operand_preds);
            self.operand_pred_set.sort_unstable();
            self.operand_pred_set.dedup();

            // Check for missing predecessors
            for &pred in &self.preds {
                if self.operand_pred_set.binary_search(&pred).is_err() {
                    self.found.push(
                        Check::PhiOperands,
                        VerifierError::MissingPhiOperand {
                            block: block_idx,
                            phi_idx,
                            missing_pred: pred,
                        },
                    );
                }
            }

            // Check for extra (non-predecessor) operands
            for &op_pred in &self.operand_pred_set {
                if self.preds.binary_search(&op_pred).is_err() {
                    self.found.push(
                        Check::PhiOperands,
                        VerifierError::ExtraPhiOperand {
                            block: block_idx,
                            phi_idx,
                            extra_pred: op_pred,
                        },
                    );
                }
            }
        }
        self.cfg = Some(cfg);
    }

    /// Reports each predecessor `operand_preds` names more than once, at the
    /// position of its second occurrence, so the findings follow operand order.
    fn report_duplicate_operands(&mut self, block_idx: usize, phi_idx: usize) {
        if self.operand_preds.len() < 2 {
            return;
        }
        self.operand_positions.clear();
        self.operand_positions.extend(
            self.operand_preds
                .iter()
                .copied()
                .enumerate()
                .map(|(position, pred)| (pred, position)),
        );
        // Sorted by predecessor and then by position, so the second entry of a
        // run of one predecessor is its second occurrence.
        self.operand_positions.sort_unstable();
        let mut second_occurrences: Vec<(usize, usize)> = Vec::new();
        let mut run_of: Option<usize> = None;
        let mut run_length: usize = 0;
        for &(pred, position) in &self.operand_positions {
            if run_of == Some(pred) {
                run_length = run_length.saturating_add(1);
            } else {
                run_of = Some(pred);
                run_length = 1;
            }
            if run_length == 2 {
                second_occurrences.push((position, pred));
            }
        }
        second_occurrences.sort_unstable();
        for (_, pred) in second_occurrences {
            self.found.push(
                Check::PhiOperands,
                VerifierError::DuplicatePhiOperand {
                    block: block_idx,
                    phi_idx,
                    pred,
                },
            );
        }
    }

    /// Verifies one phi's exceptional operands against the blocks that can
    /// throw to its block.
    ///
    /// Three ways to be wrong, and each is a different defect: an exceptional
    /// operand outside a handler or filter entry (an edit that mislabelled an
    /// operand), one naming a block that cannot throw here (a stale operand
    /// left by a CFG change), and a covering block that contributes none (a
    /// merge that is missing a value, which every meet-based analysis would
    /// then read as narrower than the truth).
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block holding the phi.
    /// * `phi_idx` - Index of the phi within that block.
    /// * `phi` - The phi itself.
    /// * `throwing` - The blocks a throw in which dispatches to `block_idx`.
    fn check_exceptional_operands(
        &mut self,
        block_idx: usize,
        phi_idx: usize,
        phi: &PhiNode,
        throwing: &[usize],
    ) {
        let mut contributed: BTreeSet<usize> = BTreeSet::new();
        let mut seen: BTreeSet<(usize, SsaVarId)> = BTreeSet::new();
        for operand in phi.exceptional_operands() {
            let source = operand.predecessor();
            if throwing.is_empty() {
                self.found.push(
                    Check::PhiOperands,
                    VerifierError::ExceptionalOperandAtNonRoot {
                        block: block_idx,
                        phi_idx,
                    },
                );
                continue;
            }
            if !throwing.contains(&source) {
                self.found.push(
                    Check::PhiOperands,
                    VerifierError::ExceptionalOperandFromNonThrowingBlock {
                        block: block_idx,
                        phi_idx,
                        source,
                    },
                );
                continue;
            }
            // Pinned to one storage: see `ExceptionalOperandFromAnotherGroup`.
            // Only when both sides are grouped - an ungrouped value is a
            // temporary the host never coalesces, and the question does not
            // arise for it.
            let expected = self.ssa.rename_group(phi.result());
            let group = self.ssa.rename_group(operand.value());
            if expected != u32::MAX && group != u32::MAX && group != expected {
                self.found.push(
                    Check::PhiOperands,
                    VerifierError::ExceptionalOperandFromAnotherGroup {
                        block: block_idx,
                        phi_idx,
                        value: operand.value(),
                        group,
                        expected,
                    },
                );
            }
            if !seen.insert((source, operand.value())) {
                self.found.push(
                    Check::PhiOperands,
                    VerifierError::DuplicateExceptionalOperand {
                        block: block_idx,
                        phi_idx,
                        source,
                        value: operand.value(),
                    },
                );
            }
            contributed.insert(source);
        }

        // Every block that can throw here has to say what the merged storage
        // may hold when it does. A block with no instructions cannot throw, and
        // an unreachable one cannot run at all, so neither owes an operand.
        for source in throwing {
            if contributed.contains(source) {
                continue;
            }
            let runs = self
                .ssa
                .block(*source)
                .is_some_and(|block| !block.instructions().is_empty());
            if runs {
                self.found.push(
                    Check::PhiOperands,
                    VerifierError::MissingExceptionalOperand {
                        block: block_idx,
                        phi_idx,
                        source: *source,
                    },
                );
            }
        }
    }

    /// Verifies one variable's definition and use sites against the blocks they
    /// name.
    ///
    /// Concluded after the walk, not during it: a variable's sites name blocks
    /// and instructions anywhere in the function.
    fn check_variable_sites(&mut self, var: &SsaVariable<T>, block_count: usize) {
        let def_site = var.def_site();
        match self.ssa.block(def_site.block()) {
            None => self.found.push(
                Check::IndexBounds,
                VerifierError::DefSiteOutOfRange {
                    var: var.id(),
                    site: def_site,
                    bound: SiteBound::Block { block_count },
                },
            ),
            Some(block) => {
                if let Some(instruction) = def_site.index()
                    && instruction >= block.instruction_count()
                {
                    self.found.push(
                        Check::IndexBounds,
                        VerifierError::DefSiteOutOfRange {
                            var: var.id(),
                            site: def_site,
                            bound: SiteBound::Instruction {
                                block: def_site.block(),
                                instruction_count: block.instruction_count(),
                            },
                        },
                    );
                }
            }
        }

        for use_site in var.uses() {
            let bound = match self.ssa.block(use_site.block()) {
                None => Some(SiteBound::Block { block_count }),
                Some(block) if use_site.is_phi_operand() => (use_site.index() >= block.phi_count())
                    .then(|| SiteBound::Phi {
                        block: use_site.block(),
                        phi_count: block.phi_count(),
                    }),
                Some(block) => (use_site.index() >= block.instruction_count()).then(|| {
                    SiteBound::Instruction {
                        block: use_site.block(),
                        instruction_count: block.instruction_count(),
                    }
                }),
            };
            if let Some(bound) = bound {
                self.found.push(
                    Check::IndexBounds,
                    VerifierError::UseSiteOutOfRange {
                        var: var.id(),
                        block: use_site.block(),
                        index: use_site.index(),
                        is_phi_operand: use_site.is_phi_operand(),
                        bound,
                    },
                );
            }
        }
    }

    /// Verifies the variables table is densely indexed by variable id.
    ///
    /// [`SsaFunction::variable`] and `variable_mut` are raw
    /// `variables.get(id.index())`, so `variables[i].id().index() == i` is a
    /// load-bearing invariant rather than a tidiness one. Removing a row without
    /// renumbering shifts every id above it, and the reads that go wrong -
    /// `def_site()` for dominance decisions, `var_type()` for type
    /// compatibility - are exactly the ones passes consult before rewriting.
    ///
    /// This is a Quick check because it is a single linear scan and because
    /// nothing else detects the violation: the resulting IR is well-formed in
    /// every other respect, merely describing different variables than intended.
    fn check_variable_table_density(&mut self) {
        for (slot, var) in self.ssa.variables().iter().enumerate() {
            if var.id().index() != slot {
                self.found.push(
                    Check::VariableTable,
                    VerifierError::VariableTableNotDense {
                        slot,
                        found: var.id(),
                    },
                );
                // One report is enough: everything after the first hole is
                // shifted too, and listing all of it buries the cause.
                break;
            }
        }
    }

    /// Checks that every exception clause can be laid out over the function's
    /// blocks.
    ///
    /// Runs at [`VerifyLevel::Quick`]: the check is one pass over the table,
    /// which is a handful of rows, and a clause that names blocks the function
    /// does not have is the kind of corruption every later check then reasons
    /// from.
    ///
    /// This is a question about the IR, not about pass robustness. Passes read
    /// the table through the total accessors and tolerate a stale index by
    /// losing a root or a barrier; the verifier is where "tolerated" stops
    /// meaning "valid". A clause whose blocks lie outside this function
    /// altogether - the funclet handler - is legal and reported by nothing.
    fn check_exception_table(&mut self) {
        let block_count = self.ssa.blocks().len();
        for (clause, handler) in self.ssa.exception_handlers().iter().enumerate() {
            if let Err(reason) = handler.layout(block_count) {
                self.found.push(
                    Check::ExceptionTable,
                    VerifierError::MalformedExceptionClause { clause, reason },
                );
            }
        }
    }

    /// Concludes the checks that need the whole function, now that every block
    /// has been read.
    fn conclude(&mut self) {
        let block_count = self.block_count;
        for var in self.ssa.variables() {
            self.check_variable_sites(var, block_count);
        }
        if !self.quick {
            return;
        }
        self.check_variable_table_density();
        self.check_exception_table();
        self.check_reads_have_definitions();
        if !self.standard {
            return;
        }
        self.check_defined_before_use();
        self.check_orphan_variables();
        self.check_use_lists();
        if self.full {
            self.check_dominance();
        }
    }

    /// Checks that every variable something reads is defined by the function's
    /// blocks, by function entry, or by the runtime.
    ///
    /// A dangling read - a variable whose defining instruction a pass removed
    /// while an instruction or phi operand still names it - is reported as
    /// [`VerifierError::UndefinedUse`] at the first site that reads it.
    ///
    /// A Quick check: it is linear, and it is the corruption the post-pass
    /// tripwire exists to catch. Release builds verify at Quick and debug
    /// builds at a higher level, so a check left out of Quick would let the two
    /// profiles recover the same bytes differently: a pass that deletes a
    /// definition under a live phi operand would be accepted in one and undone
    /// in the other.
    fn check_reads_have_definitions(&mut self) {
        for var in self.ssa.variables() {
            let facts = self.vars.get(var.id());
            if facts.is_some_and(VarFacts::is_defined) {
                continue;
            }
            if let Some((block, instr_idx)) = facts.and_then(VarFacts::first_read)
                && !self.ssa.is_entry_defined(var)
            {
                self.found.push(
                    Check::ReadsHaveDefinitions,
                    VerifierError::UndefinedUse {
                        block,
                        instr_idx,
                        var: var.id(),
                    },
                );
            }
        }
    }

    /// Checks that every variable used in an instruction is defined somewhere,
    /// either in the variables vec or in a block.
    ///
    /// A use whose variable is absent from both is reported as a
    /// [`VerifierError::UndefinedUse`], with no exemptions: a value the runtime
    /// supplies is defined by [`SsaOp::RuntimeValue`], which is a definition
    /// like any other. A variable the table has a row for is defined for this
    /// purpose, so only the uses of ids the table lacks were set aside during
    /// the scan, and what is left to learn about them is whether a block
    /// defines them. Phi operand values are not checked here; their coverage
    /// falls to the phi-operand and registration checks.
    ///
    /// [`SsaOp::RuntimeValue`]: crate::ir::ops::SsaOp::RuntimeValue
    fn check_defined_before_use(&mut self) {
        for (block, instr_idx, var) in mem::take(&mut self.unregistered_uses) {
            if !self.vars.get(var).is_some_and(VarFacts::is_defined) {
                self.found.push(
                    Check::DefinedBeforeUse,
                    VerifierError::UndefinedUse {
                        block,
                        instr_idx,
                        var,
                    },
                );
            }
        }
    }

    /// Checks that every registered variable is actually defined.
    ///
    /// The check runs in both directions with the registration check made as
    /// each definition was read. Backward, here: a registered variable with no
    /// block-local definition is classified by whether anything reads it. A
    /// dangling read is reported by `check_reads_have_definitions` (unless the
    /// variable is genuinely entry-defined, see `is_entry_defined`), while a
    /// variable nothing reads is a leftover row reported as an
    /// [`VerifierError::OrphanVariable`]. The two are split because consumers
    /// treat them differently: `SsaRebuilder::verify` discards the orphan class
    /// as cosmetic but not the dangling read. The weaker entry test is right
    /// here - tightening it would report every temporary a pass legitimately
    /// killed.
    fn check_orphan_variables(&mut self) {
        for var in self.ssa.variables() {
            let facts = self.vars.get(var.id());
            if facts.is_some_and(VarFacts::is_defined)
                || facts.is_some_and(VarFacts::is_read)
                || (var.version() == 0 && var.def_site().index().is_none())
            {
                continue;
            }
            self.found.push(
                Check::RegisteredVariables,
                VerifierError::OrphanVariable { var: var.id() },
            );
        }
    }

    /// Checks that every variable's recorded use list matches the IR.
    ///
    /// Takes each variable's uses straight from the blocks - instruction
    /// operands and phi operands alike, one entry per occurrence, noted during
    /// the scan - and compares them, as sorted multisets of `(block, index,
    /// is_phi)` sites, against what the variable table records. Comparing
    /// counts alone would miss the commonest staleness: an instruction inserted
    /// or removed above a use shifts its index while every count stays the
    /// same.
    ///
    /// This exists because use-list validity is otherwise an unenforced caller
    /// obligation. [`SsaEditor`] restores the index at the end of an edit
    /// session, but IR built or rewritten outside one keeps whatever the
    /// construction left behind, and nothing noticed: the damage surfaces far
    /// away, as a sparse analysis that never revisits a consumer and concludes
    /// a range narrower than the truth.
    ///
    /// [`SsaEditor`]: crate::ir::function::SsaEditor
    fn check_use_lists(&mut self) {
        // Bucket the occurrences by variable with a counting sort, which keeps
        // the whole comparison to a handful of allocations however many
        // variables the function has. An id past the dense part is a row that
        // sits in the wrong slot; the buckets keep it out of the sort's size.
        let mut buckets = VarBuckets::build(self.vars.dense_len(), (0, 0, false), |emit| {
            for record in &self.use_records {
                emit(record.var, (record.block, record.index, record.is_phi));
            }
        });

        let mut recorded_sites: Vec<(usize, usize, bool)> = Vec::new();
        for var in self.ssa.variables() {
            recorded_sites.clear();
            recorded_sites.extend(
                var.uses()
                    .iter()
                    .map(|site| (site.block(), site.index(), site.is_phi_operand())),
            );
            let counted = buckets.get_mut(var.id());
            if recorded_sites.is_empty() && counted.is_empty() {
                continue;
            }
            recorded_sites.sort_unstable();
            counted.sort_unstable();
            if recorded_sites.as_slice() != &*counted {
                self.found.push(
                    Check::UseLists,
                    VerifierError::StaleUseList {
                        var: var.id(),
                        recorded: recorded_sites.len(),
                        actual: counted.len(),
                    },
                );
            }
        }
    }

    /// Checks dominance: every use of a variable must be dominated by its definition.
    ///
    /// For phi operands, the use is considered to be at the end of the predecessor
    /// block (not at the phi's block), following standard SSA semantics.
    ///
    /// Only reachable blocks are checked, and reachability is taken from the
    /// exception-aware dominator tree so that handler and filter code - which is
    /// unreachable from the entry block alone - is verified rather than silently
    /// skipped. Block indices read straight from the IR are queried with the
    /// non-asserting `contains_checked`, since malformed IR may record a def or
    /// predecessor block past the block count and the verifier must report such
    /// IR rather than panic on it.
    fn check_dominance(&mut self) {
        let Some(cfg) = self.cfg.take() else {
            return;
        };
        // Exception-aware, so handler and filter code is checked at all: rooted
        // at the entry block alone, every such block is unreachable and no
        // dominance violation inside one can be seen.
        let dominance = EhDominance::of(&cfg);
        let block_count = self.block_count.max(1);
        let tree = dominance.tree();
        let mut reachable = BitSet::new(block_count);
        for block_idx in 0..block_count {
            // Reachability the tree already knows, over the exception-aware
            // view -- so a handler is checked rather than skipped.
            if tree.is_reachable(NodeId::new(block_idx)) {
                reachable.insert_checked(block_idx);
            }
        }

        // Check instruction uses
        let mut use_buf: Vec<SsaVarId> = Vec::new();
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            if !reachable.contains(block_idx) {
                continue;
            }

            for instr in block.instructions() {
                // Reuse one scratch buffer instead of `SsaOp::uses()`, which
                // allocates a fresh `Vec` per instruction - this runs over every
                // instruction of every function that gets verified.
                use_buf.clear();
                instr.op().for_each_use(|var| use_buf.push(var));
                for &used_var in &use_buf {
                    if let Some(def_block) =
                        self.vars.get(used_var).and_then(VarFacts::definition_block)
                    {
                        // `def_block` is a variable's recorded `DefSite::block`,
                        // which malformed IR can put past the block count.
                        // `contains` asserts; an out-of-range block is by
                        // definition unreachable, so `false` is also the correct
                        // answer. The verifier is the net for malformed IR and
                        // must not abort on a shape it exists to report.
                        if !reachable.contains_checked(def_block) {
                            continue;
                        }
                        // Definition must dominate use block
                        if def_block < block_count
                            && block_idx < block_count
                            && !dominance.definition_is_well_formed(def_block, block_idx)
                        {
                            self.found.push(
                                Check::Dominance,
                                VerifierError::DominanceViolation {
                                    var: used_var,
                                    def_block,
                                    use_block: block_idx,
                                },
                            );
                        }
                    }
                }
            }

            // Check phi operand uses: the use is at the end of the predecessor
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    let used_var = operand.value();
                    let pred_block = operand.predecessor();
                    if let Some(def_block) =
                        self.vars.get(used_var).and_then(VarFacts::definition_block)
                    {
                        // Both indices come straight out of the IR - `pred_block`
                        // is a raw phi-operand predecessor, which
                        // `check_phi_operands` explicitly tolerates naming a
                        // non-existent block.
                        if !reachable.contains_checked(def_block)
                            || !reachable.contains_checked(pred_block)
                        {
                            continue;
                        }
                        // Definition must dominate the predecessor block
                        if def_block < block_count
                            && pred_block < block_count
                            && !dominance.definition_is_well_formed(def_block, pred_block)
                        {
                            self.found.push(
                                Check::Dominance,
                                VerifierError::DominanceViolation {
                                    var: used_var,
                                    def_block,
                                    use_block: pred_block,
                                },
                            );
                        }
                    }
                }
            }
        }
        self.cfg = Some(cfg);
    }
}

/// Which checks a scan runs.
#[derive(Debug, Clone, Copy)]
enum Depth {
    /// Only the index-bounds check.
    IndexBounds,
    /// Every check up to and including a [`VerifyLevel`].
    Level(VerifyLevel),
}

/// SSA verifier that validates invariants at configurable depth.
///
/// The verifier checks SSA form correctness at three levels (Quick, Standard, Full)
/// and reports all violations as `VerifierError` values. It is the safety net for
/// all SSA transformations, catching invariant violations early.
pub struct SsaVerifier<'a, T: Target> {
    /// Reference to the SSA function being verified.
    ssa: &'a SsaFunction<T>,
}

impl<'a, T: Target> SsaVerifier<'a, T> {
    /// Creates a new verifier for the given SSA function.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>) -> Self {
        Self { ssa }
    }

    /// Reports every index in the function that names something it does not
    /// hold, and nothing else.
    ///
    /// The same scan [`verify`](Self::verify) runs first at every level, offered
    /// on its own for the one caller that needs it separately: a host that has
    /// just built a function by hand, or received one it does not trust, and
    /// wants the cheap structural gate before handing it to analyses whose
    /// block- and variable-sized sets are indexed by what the IR says.
    ///
    /// # Returns
    ///
    /// Every out-of-range index found, in discovery order; empty when every
    /// index the function names exists.
    #[must_use]
    pub fn index_bounds_only(self) -> Vec<VerifierError> {
        Scan::new(self.ssa, Depth::IndexBounds).run()
    }

    /// Runs verification at the specified level and returns all errors found.
    ///
    /// The three levels are cumulative and gated by cost. The Quick checks
    /// always run because they are single linear scans and catch the corruption
    /// every later check then reasons from: table density, single definition,
    /// block structure, placeholders/self-references, and the exception table.
    /// Standard builds the CFG once and adds the phi-operand, def-use,
    /// registration, use-list and per-op shape/effect checks. Full additionally
    /// computes dominance over the exception-aware CFG and dominator tree, so
    /// handler and filter code is reachable and checked at all rather than
    /// dismissed as unreachable from the entry block alone.
    ///
    /// All of them share one walk over the blocks: an operation's operands are
    /// read out once and handed to every check that wants them, and a check that
    /// needs a fact about the whole function concludes when the walk ends. The
    /// errors come back as if each check had run alone, check by check, each in
    /// the order it found them.
    ///
    /// Consumes the verifier so the accumulated error list can be handed back by
    /// value.
    ///
    /// # Arguments
    ///
    /// * `level` - How deep to verify; each level first runs every lower
    ///   level's checks.
    ///
    /// # Returns
    ///
    /// Every invariant violation found, check by check and in discovery order
    /// within each; empty when the SSA is well-formed at `level`.
    pub fn verify(self, level: VerifyLevel) -> Vec<VerifierError> {
        Scan::new(self.ssa, Depth::Level(level)).run()
    }
}

/// Returns the greatest common divisor, with one for two zero operands.
fn greatest_common_divisor(mut left: u128, mut right: u128) -> u128 {
    while let Some(remainder) = left.checked_rem(right) {
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            float::FloatRounding,
            instruction::SsaInstruction,
            ops::{
                CarrylessPart, CmpKind, Computation, ControlEffect, FloatFunctionKind, Kernel,
                KindedVecData, LaneControl, MemoryAccessSemantics, MemoryEffectLocation, PacKind,
                RuntimeValueKind, SsaEffectKind, SsaEffects, SsaOp, TrapClass, VectorBinaryKind,
                VectorCastKind, VectorCompareKind, VectorCryptoKind, VectorElement,
                VectorElementKind, VectorFaultMode, VectorFixedKind, VectorInactivePolicy,
                VectorMaskBinaryKind, VectorPredicateLayout, VectorReduceKind, VectorRounding,
                VectorStatus, VectorTernaryKind, VectorUnaryKind,
            },
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        target::VectorTailPolicy,
        testing::{MockTarget, MockType},
    };

    /// **An effect summary that contradicts itself is reported.** No typed
    /// operation derives one, which is the invariant the check guards: an
    /// atomic effect without atomic semantics or an ordering, a fence without
    /// fence semantics, and memory-free effects that reference memory are
    /// each an invalid native operation, and a consistent summary is none.
    #[test]
    fn an_effect_summary_that_contradicts_itself_is_reported() {
        let ssa = SsaFunction::<MockTarget>::new(0, 0);
        let report = |effects: SsaEffects| {
            let mut found = Findings::new();
            OperationScan {
                ssa: &ssa,
                found: &mut found,
            }
            .check_effect_summary(0, 0, effects);
            found.into_errors()
        };
        let invalid = |effects| {
            report(effects)
                .iter()
                .any(|error| matches!(error, VerifierError::InvalidNativeOperation { .. }))
        };
        assert!(invalid(SsaEffects {
            kind: SsaEffectKind::Atomic,
            may_throw: false,
            memory: MemoryEffectLocation::Unknown,
            memory_semantics: MemoryAccessSemantics::Normal,
            volatile: false,
            ordering: None,
            trap: TrapClass::None,
            control: ControlEffect::None,
        }));
        assert!(invalid(SsaEffects {
            memory_semantics: MemoryAccessSemantics::Normal,
            ..SsaEffects::new(SsaEffectKind::Fence, false)
        }));
        assert!(invalid(
            SsaEffects::pure().with_memory(MemoryEffectLocation::Unknown)
        ));
        assert!(report(SsaEffects::pure()).is_empty());
        assert!(report(SsaEffects::new(SsaEffectKind::Write, false)).is_empty());
    }

    /// A variable nothing defines and nothing reads is cosmetic; one read with
    /// no definition is not.
    #[test]
    fn only_an_unread_orphan_is_cosmetic() {
        let var = SsaVarId::from_index(3);
        assert!(VerifierError::OrphanVariable { var }.is_cosmetic());
        assert!(
            !VerifierError::UndefinedUse {
                block: 0,
                instr_idx: 0,
                var,
            }
            .is_cosmetic()
        );
    }

    /// The value a handler is entered with has a definition like any other, and
    /// the verifier demands one.
    ///
    /// The older model left it undefined and exempted the shape by position,
    /// which every edit to the entry block could take away. `RuntimeValue` says
    /// what the position only implied, so the exemption is gone: the same
    /// function without it reads a value nothing produces, and that is now
    /// reported.
    #[test]
    fn a_runtime_supplied_value_needs_its_definition() {
        /// A catch handler reading a value it is entered with, defined by a
        /// `RuntimeValue` when `defined` and left undefined otherwise.
        fn handler_reading_an_entry_value(defined: bool) -> SsaFunction<MockTarget> {
            let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
            let object = ssa.create_variable(
                VariableOrigin::Local(0),
                1,
                if defined {
                    DefSite::instruction(2, 0)
                } else {
                    DefSite::phi(2)
                },
                MockType::I32,
            );

            let mut entry = SsaBlock::new(0);
            entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
            ssa.add_block(entry);

            let mut guard = SsaBlock::new(1);
            guard.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(guard);

            let mut handler = SsaBlock::new(2);
            if defined {
                handler.add_instruction(SsaInstruction::synthetic(SsaOp::RuntimeValue {
                    dest: object,
                    kind: RuntimeValueKind::ExceptionObject,
                }));
            }
            handler.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(object))));
            ssa.add_block(handler);

            let mut join = SsaBlock::new(3);
            join.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(join);

            ssa.set_exception_handlers(vec![SsaExceptionHandler {
                flags: 0,
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
            ssa
        }

        let defined = handler_reading_an_entry_value(true);
        for level in [VerifyLevel::Standard, VerifyLevel::Full] {
            let errors = SsaVerifier::new(&defined).verify(level);
            assert!(
                !errors.iter().any(|error| matches!(
                    error,
                    VerifierError::UndefinedUse { .. }
                        | VerifierError::UnregisteredVariable { .. }
                        | VerifierError::MisplacedRuntimeValue { .. }
                )),
                "a defined entry value verifies clean at {level:?}: {errors:?}"
            );
        }

        let undefined = handler_reading_an_entry_value(false);
        let errors = SsaVerifier::new(&undefined).verify(VerifyLevel::Standard);
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, VerifierError::UndefinedUse { .. })),
            "without the definition the handler reads a value nothing produces: {errors:?}"
        );
    }

    /// A runtime-supplied definition is justified by where it stands, so the
    /// verifier checks exactly that, and says which way it is wrong.
    #[test]
    fn a_runtime_supplied_value_is_pinned_to_where_the_runtime_supplies_it() {
        /// The exception object defined in `block`, after `preceding` other
        /// instructions, with the handler clause given `flags`.
        fn object_defined_at(
            block: usize,
            preceding: usize,
            flags: u32,
        ) -> SsaFunction<MockTarget> {
            let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
            let object = ssa.create_variable(
                VariableOrigin::Local(0),
                1,
                DefSite::instruction(block, preceding),
                MockType::I32,
            );

            for idx in 0..3 {
                let mut current = SsaBlock::new(idx);
                if idx == block {
                    for _ in 0..preceding {
                        current.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
                    }
                    current.add_instruction(SsaInstruction::synthetic(SsaOp::RuntimeValue {
                        dest: object,
                        kind: RuntimeValueKind::ExceptionObject,
                    }));
                }
                current.add_instruction(SsaInstruction::synthetic(match idx {
                    0 => SsaOp::Jump { target: 1 },
                    1 => SsaOp::Jump { target: 2 },
                    _ => SsaOp::returns(None),
                }));
                ssa.add_block(current);
            }

            ssa.set_exception_handlers(vec![SsaExceptionHandler {
                flags,
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
            ssa
        }

        /// The placement complaints one function draws.
        fn placements(ssa: &SsaFunction<MockTarget>) -> Vec<RuntimeValuePlacement> {
            SsaVerifier::new(ssa)
                .verify(VerifyLevel::Quick)
                .into_iter()
                .filter_map(|error| match error {
                    VerifierError::MisplacedRuntimeValue { reason, .. } => Some(reason),
                    _ => None,
                })
                .collect()
        }

        // Leading in the catch handler's entry: the one legal position.
        assert_eq!(placements(&object_defined_at(2, 0, 0)), Vec::new());

        // The same block, one instruction later: something already ran.
        assert_eq!(
            placements(&object_defined_at(2, 1, 0)),
            vec![RuntimeValuePlacement::NotLeading]
        );

        // An ordinary block, which no dispatch reaches.
        assert_eq!(
            placements(&object_defined_at(1, 0, 0)),
            vec![RuntimeValuePlacement::NotARuntimeEntry]
        );

        // A finally handler is entered for an exit, holding no object. `0x2` is
        // the finally flag.
        assert_eq!(
            placements(&object_defined_at(2, 0, 0x2)),
            vec![RuntimeValuePlacement::NotARuntimeEntry],
            "a finally is entered with no exception object"
        );
    }

    /// Region extents compare rational scalable widths and fixed prefixes
    /// without substituting any minimum runtime length.
    #[test]
    fn region_extents_preserve_runtime_scale_and_fixed_terms() {
        assert_eq!(RegionExtent::new(0, 256, 2), RegionExtent::new(0, 128, 1));
        assert_eq!(
            RegionExtent::new(16, 128, 1)
                .unwrap()
                .plus(RegionExtent::new(8, 32, 2).unwrap()),
            RegionExtent::new(24, 144, 1)
        );
        assert_ne!(RegionExtent::new(128, 0, 1), RegionExtent::new(0, 128, 1));
        assert_eq!(RegionExtent::new(0, 1, 0), None);
        assert_eq!(
            RegionExtent::new(u128::MAX, 0, 1)
                .unwrap()
                .plus(RegionExtent::new(1, 0, 1).unwrap()),
            None
        );
        assert_eq!(
            RegionExtent::of::<MockTarget>(&MockType::NxV4I32),
            RegionExtent::new(0, 128, 1)
        );
        assert_eq!(
            RegionExtent::of::<MockTarget>(&MockType::NxMask4),
            RegionExtent::new(0, 4, 1)
        );
        assert_eq!(
            RegionExtent::of::<MockTarget>(&MockType::Mask4),
            RegionExtent::new(4, 0, 1)
        );
        assert_eq!(RegionExtent::of::<MockTarget>(&MockType::Unknown), None);
    }

    /// The three operations keep fixed, scalable and predicate storage widths
    /// distinct, while accepting an unknown input width conservatively.
    #[test]
    fn region_operations_verify_runtime_widths_and_offsets() {
        let nx = MockType::NxV4I32;
        let wide = MockType::NxV8I32;
        let fixed = MockType::V4I32;
        for (kind, base, value, offset, target, valid) in [
            (0, nx, nx, MockType::I64, nx, true),
            (0, fixed, nx, MockType::I64, nx, true),
            (1, nx, nx, MockType::I64, nx, true),
            (1, nx, nx, MockType::I64, wide, false),
            (1, fixed, nx, MockType::I64, fixed, true),
            (2, nx, nx, MockType::I64, wide, true),
            (2, nx, nx, MockType::I64, nx, false),
            (2, fixed, nx, MockType::I64, wide, false),
            (
                2,
                MockType::V16I4,
                MockType::V16I4,
                MockType::I64,
                fixed,
                true,
            ),
            (
                2,
                MockType::Mask4,
                MockType::Mask4,
                MockType::I64,
                MockType::I8,
                true,
            ),
            (2, MockType::Unknown, nx, MockType::I64, wide, true),
            (
                1,
                MockType::NxMask4,
                nx,
                MockType::I64,
                MockType::NxMask4,
                true,
            ),
            (0, nx, nx, MockType::F64, nx, false),
            (1, nx, nx, nx, nx, false),
            (0, nx, nx, MockType::Unknown, nx, true),
        ] {
            let errors = region_fixture(kind, [base, offset, value], target);
            if valid {
                assert!(
                    errors.is_empty(),
                    "kind={kind}, target={target:?}: {errors:?}"
                );
            } else {
                assert!(
                    errors
                        .iter()
                        .any(|error| matches!(error, VerifierError::InvalidVectorOperation { .. })),
                    "kind={kind}, target={target:?}: {errors:?}"
                );
            }
        }
    }

    /// Builds one region operation over three entry values and verifies its
    /// complete SSA graph, not only its shape metadata.
    fn region_fixture(
        kind: u8,
        inputs: [MockType; 3],
        target_type: MockType,
    ) -> Vec<VerifierError> {
        let mut ssa = SsaFunction::<MockTarget>::new(4, 0);
        let inputs = inputs
            .into_iter()
            .chain([MockType::I64])
            .enumerate()
            .map(|(index, ty)| {
                ssa.create_variable(
                    VariableOrigin::Argument(u16::try_from(index).unwrap()),
                    0,
                    DefSite::entry(),
                    ty,
                )
            })
            .collect::<Vec<_>>();
        let dest = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 0),
            target_type,
        );
        let op = match kind {
            0 => SsaOp::VectorSlice {
                dest,
                vector: inputs[0],
                offset: inputs[1],
                length: inputs[3],
                target_type,
            },
            1 => SsaOp::VectorInsertSlice {
                dest,
                vector: inputs[0],
                offset: inputs[1],
                value: inputs[2],
                target_type,
            },
            2 => SsaOp::VectorConcat {
                dest,
                left: inputs[0],
                right: inputs[2],
                target_type,
            },
            _ => unreachable!(),
        };
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(op));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest))));
        ssa.add_block(block);
        ssa.refresh_def_sites();
        ssa.recompute_uses();
        SsaVerifier::new(&ssa).verify(VerifyLevel::Full)
    }

    /// A scalar operation computes at one width: operands of two widths are
    /// rejected, a shift's amount is any width, and an operand whose width is
    /// not fixed is not compared.
    #[test]
    fn a_scalar_operation_computes_at_one_width() {
        /// Verifies `dest = left xor right`, or `dest = left << right` when
        /// `shift`, under the types given.
        fn widths(shift: bool, [left, right, dest]: [MockType; 3]) -> Vec<VerifierError> {
            let mut ssa = SsaFunction::<MockTarget>::new(2, 0);
            let [left, right] = [(0, left), (1, right)].map(|(index, ty)| {
                ssa.create_variable(VariableOrigin::Argument(index), 0, DefSite::entry(), ty)
            });
            let dest =
                ssa.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(0, 0), dest);
            let op = if shift {
                SsaOp::Shl {
                    dest,
                    value: left,
                    amount: right,
                    flags: None,
                }
            } else {
                SsaOp::Xor {
                    dest,
                    left,
                    right,
                    flags: None,
                }
            };
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(op));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest))));
            ssa.add_block(block);
            ssa.refresh_def_sites();
            ssa.recompute_uses();
            SsaVerifier::new(&ssa).verify(VerifyLevel::Full)
        }
        let (byte, word, long, unsigned, unknown) = (
            MockType::I8,
            MockType::I32,
            MockType::I64,
            MockType::U32,
            MockType::Unknown,
        );
        for (shift, types, valid) in [
            (false, [word, unsigned, word], true),
            (false, [byte, word, word], false),
            (false, [word, word, long], false),
            (false, [word, unknown, word], true),
            (true, [word, byte, word], true),
            (true, [word, word, byte], false),
        ] {
            let mismatched = widths(shift, types)
                .iter()
                .any(|error| matches!(error, VerifierError::InvalidOperandWidths { .. }));
            assert_eq!(mismatched, !valid, "shift={shift}, types={types:?}");
        }
    }

    /// **Every comparison reads two values of one width, and a float
    /// function its floats.** A boolean comparison, a compare-and-branch and
    /// a float compare's flags of an 80-bit and a 64-bit float are each
    /// rejected, and so is `fscale` of them; each of two 80-bit floats is
    /// accepted, the boolean `fscale` result here being no float the rule
    /// holds.
    #[test]
    fn every_comparison_reads_two_values_of_one_width() {
        fn rejected(form: usize, right: MockType) -> bool {
            let mut ssa = SsaFunction::<MockTarget>::new(2, 0);
            let left = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::F80,
            );
            let right =
                ssa.create_variable(VariableOrigin::Argument(1), 0, DefSite::entry(), right);
            let dest = ssa.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, 0),
                MockType::Bool,
            );
            let mut block = SsaBlock::new(0);
            let terminator = match form {
                0 => {
                    block.add_instruction(SsaInstruction::synthetic(SsaOp::Cgt {
                        dest,
                        left,
                        right,
                        unsigned: false,
                        signaling: false,
                    }));
                    SsaOp::returns(Some(dest))
                }
                3 => {
                    block.add_instruction(SsaInstruction::synthetic(SsaOp::FloatFunction(
                        Box::new(KindedVecData {
                            kind: FloatFunctionKind::Scale,
                            outputs: vec![dest],
                            inputs: vec![left, right],
                        }),
                    )));
                    SsaOp::returns(None)
                }
                1 => SsaOp::BranchCmp {
                    left,
                    right,
                    cmp: CmpKind::Gt,
                    unsigned: false,
                    true_target: 1,
                    false_target: 1,
                },
                _ => {
                    block.add_instruction(SsaInstruction::synthetic(SsaOp::FloatCompareFlags {
                        flags: dest,
                        left,
                        right,
                        signaling: false,
                    }));
                    SsaOp::returns(Some(dest))
                }
            };
            block.add_instruction(SsaInstruction::synthetic(terminator));
            ssa.add_block(block);
            let mut exit = SsaBlock::new(1);
            exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(exit);
            ssa.refresh_def_sites();
            ssa.recompute_uses();
            SsaVerifier::new(&ssa)
                .verify(VerifyLevel::Full)
                .iter()
                .any(|error| matches!(error, VerifierError::InvalidOperandWidths { .. }))
        }
        for form in 0..3 {
            assert!(
                rejected(form, MockType::F64),
                "form {form} of 80 and 64 bits"
            );
        }
        for form in 0..4 {
            assert!(!rejected(form, MockType::F80), "form {form} of one width");
        }
    }

    /// **A rounding is stated only where it decides the result**: on the
    /// float lanes of an addition, a square root, a fused multiply-add or a
    /// reduction's sum, and on no integer lanes and no minimum.
    #[test]
    fn a_rounding_is_stated_only_on_an_operation_that_rounds() {
        fn rejected(op: SsaOp<MockTarget>) -> bool {
            let mut ssa = SsaFunction::<MockTarget>::new(3, 0);
            for index in 0..3 {
                ssa.create_variable(
                    VariableOrigin::Argument(index),
                    0,
                    DefSite::entry(),
                    MockType::F64,
                );
            }
            // The fourth variable, the one every operation here defines.
            let dest = ssa.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, 0),
                MockType::F64,
            );
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(op));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest))));
            ssa.add_block(block);
            ssa.refresh_def_sites();
            ssa.recompute_uses();
            SsaVerifier::new(&ssa)
                .verify(VerifyLevel::Full)
                .iter()
                .any(|error| matches!(error, VerifierError::InvalidVectorOperation { .. }))
        }
        let var = SsaVarId::from_index;
        let up = |element: VectorElement| element.rounded(FloatRounding::Up);
        let binary = |kind, element| SsaOp::VectorBinary {
            dest: var(3),
            left: var(0),
            right: var(1),
            kind,
            element,
            environment: None,
        };
        let float = VectorElement {
            scalar: true,
            ..VectorElement::float(64)
        };
        assert!(!rejected(binary(VectorBinaryKind::Add, up(float))));
        assert!(!rejected(binary(VectorBinaryKind::Min, float)));
        assert!(rejected(binary(VectorBinaryKind::Min, up(float))));
        assert!(rejected(binary(
            VectorBinaryKind::Add,
            up(VectorElement::integer(64, None))
        )));
        assert!(!rejected(SsaOp::VectorUnary {
            dest: var(3),
            value: var(0),
            kind: VectorUnaryKind::Sqrt,
            element: up(float),
            environment: None,
        }));
        assert!(rejected(SsaOp::VectorUnary {
            dest: var(3),
            value: var(0),
            kind: VectorUnaryKind::Neg,
            element: up(float),
            environment: None,
        }));
        assert!(!rejected(SsaOp::VectorTernary {
            dest: var(3),
            first: var(0),
            second: var(1),
            third: var(2),
            kind: VectorTernaryKind::Fma,
            element: up(float),
            environment: None,
        }));
        assert!(!rejected(SsaOp::VectorReduce {
            dest: var(3),
            value: var(0),
            kind: VectorReduceKind::Add,
            element: up(float),
            initial: None,
            ordered: true,
            groups: None,
            environment: None,
        }));
    }

    #[test]
    fn computed_mask_tail_limit_requires_integer_storage() {
        for limit_type in [
            MockType::I64,
            MockType::F64,
            MockType::Ref,
            MockType::NxMask4,
        ] {
            let kind = Computation {
                canonical_nan: false,
                kernel: Kernel::Compare {
                    relation: VectorCompareKind::Eq,
                    signaling: false,
                },
                element: VectorElement::integer(32, None),
                predicate: VectorPredicateLayout::Packed,
                rounding: crate::ir::ops::VectorRounding::None,
                status: VectorStatus::None,
                lanes: Some(LaneControl {
                    mask_tail: true,
                    tail: VectorTailPolicy::Agnostic,
                    inactive: None,
                    no_passthrough: false,
                }),
            };
            let mut function = SsaFunction::<MockTarget>::new(6, 0);
            let inputs = [
                MockType::I64,
                MockType::I64,
                limit_type,
                MockType::NxMask4,
                MockType::NxV4I32,
                MockType::NxV4I32,
            ]
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                function.create_variable(
                    VariableOrigin::Argument(index as u16),
                    0,
                    DefSite::entry(),
                    ty,
                )
            })
            .collect();
            let output = function.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, 0),
                MockType::NxMask4,
            );
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::compute(
                kind,
                vec![output],
                inputs,
            )));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(output))));
            function.add_block(block);
            function.refresh_def_sites();
            function.recompute_uses();
            assert_eq!(
                SsaVerifier::new(&function)
                    .verify(VerifyLevel::Full)
                    .is_empty(),
                limit_type == MockType::I64,
                "{limit_type:?}"
            );
        }
    }

    #[test]
    fn controlled_execution_verifier_checks_header_arity_and_passthrough_layouts() {
        let kind = Computation {
            canonical_nan: false,
            kernel: Kernel::Binary(VectorBinaryKind::Add),
            element: VectorElement::integer(32, None),
            predicate: VectorPredicateLayout::Packed,
            rounding: VectorRounding::None,
            status: VectorStatus::None,
            lanes: Some(LaneControl {
                mask_tail: false,
                tail: VectorTailPolicy::Undisturbed,
                inactive: Some(VectorInactivePolicy::Undisturbed),
                no_passthrough: false,
            }),
        };
        for (source_types, target, remove_input, valid) in [
            (
                [
                    MockType::I64,
                    MockType::I64,
                    MockType::NxMask4,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                ],
                MockType::NxV4I32,
                false,
                true,
            ),
            (
                [
                    MockType::I64,
                    MockType::I64,
                    MockType::Mask4,
                    MockType::V4I32,
                    MockType::V4I32,
                    MockType::V4I32,
                ],
                MockType::V4I32,
                false,
                true,
            ),
            (
                [
                    MockType::F64,
                    MockType::I64,
                    MockType::NxMask4,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                ],
                MockType::NxV4I32,
                false,
                false,
            ),
            (
                [
                    MockType::I64,
                    MockType::F64,
                    MockType::NxMask4,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                ],
                MockType::NxV4I32,
                false,
                false,
            ),
            (
                [
                    MockType::I64,
                    MockType::I64,
                    MockType::F64,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                ],
                MockType::NxV4I32,
                false,
                false,
            ),
            (
                [
                    MockType::I64,
                    MockType::I64,
                    MockType::NxMask4,
                    MockType::V4I32,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                ],
                MockType::NxV4I32,
                false,
                false,
            ),
            (
                [
                    MockType::I64,
                    MockType::I64,
                    MockType::NxMask4,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                    MockType::NxV4I32,
                ],
                MockType::NxV4I32,
                true,
                false,
            ),
        ] {
            let mut function = SsaFunction::<MockTarget>::new(6, 0);
            let mut inputs: Vec<_> = source_types
                .into_iter()
                .enumerate()
                .map(|(index, ty)| {
                    function.create_variable(
                        VariableOrigin::Argument(index as u16),
                        0,
                        DefSite::entry(),
                        ty,
                    )
                })
                .collect();
            if remove_input {
                inputs.pop();
            }
            let output = function.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, 0),
                target,
            );
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::compute(
                kind,
                vec![output],
                inputs,
            )));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(output))));
            function.add_block(block);
            function.refresh_def_sites();
            function.recompute_uses();
            let errors = SsaVerifier::new(&function).verify(VerifyLevel::Full);
            assert_eq!(errors.is_empty(), valid, "{source_types:?}: {errors:?}");
            if !valid {
                assert!(
                    errors
                        .iter()
                        .any(|error| matches!(error, VerifierError::InvalidVectorOperation { .. }))
                );
            }
        }
    }

    #[test]
    fn controlled_integer_arithmetic_rejects_managed_references_as_data() {
        let kind = Computation {
            canonical_nan: false,
            kernel: Kernel::Binary(VectorBinaryKind::Add),
            element: VectorElement::integer(32, None),
            predicate: VectorPredicateLayout::Packed,
            rounding: crate::ir::ops::VectorRounding::None,
            status: VectorStatus::None,
            lanes: Some(LaneControl {
                mask_tail: false,
                tail: VectorTailPolicy::Undisturbed,
                inactive: None,
                no_passthrough: false,
            }),
        };
        let mut function = SsaFunction::<MockTarget>::new(5, 0);
        let inputs = [
            MockType::I64,
            MockType::I64,
            MockType::NxV4I32,
            MockType::NxV4I32,
            MockType::Ref,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            function.create_variable(
                VariableOrigin::Argument(index as u16),
                0,
                DefSite::entry(),
                ty,
            )
        })
        .collect();
        let output = function.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 0),
            MockType::NxV4I32,
        );
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::compute(
            kind,
            vec![output],
            inputs,
        )));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(output))));
        function.add_block(block);
        function.refresh_def_sites();
        function.recompute_uses();
        let errors = SsaVerifier::new(&function).verify(VerifyLevel::Full);
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, VerifierError::InvalidVectorOperation { .. })),
            "managed references have no integer lane representation: {errors:?}"
        );
    }

    /// Verifies one complete controlled interface, including every result slot.
    fn controlled_storage_fixture(
        kind: Computation,
        input_types: &[MockType],
        output_types: &[MockType],
    ) -> Vec<VerifierError> {
        let mut function = SsaFunction::<MockTarget>::new(input_types.len(), 0);
        let inputs = input_types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                function.create_variable(
                    VariableOrigin::Argument(u16::try_from(index).unwrap()),
                    0,
                    DefSite::entry(),
                    *ty,
                )
            })
            .collect();
        let outputs: Vec<_> = output_types
            .iter()
            .map(|ty| {
                function.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(0, 0), *ty)
            })
            .collect();
        let result = outputs.first().copied();
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::compute(
            kind, outputs, inputs,
        )));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(result)));
        function.add_block(block);
        function.refresh_def_sites();
        function.recompute_uses();
        SsaVerifier::new(&function).verify(VerifyLevel::Full)
    }

    /// A whole-value kernel's interface is exact: a pointer authentication
    /// without its key, a random draw without the flags reporting its
    /// success, or a modular conversion reading the flags as well as its
    /// double states some other computation.
    #[test]
    fn whole_kernel_interfaces_are_exact() {
        let whole = |kernel: Kernel| {
            Computation::whole(
                kernel,
                VectorElement::integer(kernel.fixed_integer_bits().unwrap_or(64), None),
                VectorRounding::None,
            )
        };
        let word = MockType::I64;
        for (kernel, inputs, outputs) in [
            (Kernel::PointerAuth(PacKind::Sign), 4, 1),
            (Kernel::PointerAuth(PacKind::AuthenticateDiversified), 5, 1),
            (
                Kernel::Random {
                    from_entropy: false,
                },
                0,
                2,
            ),
            (Kernel::FloatToIntModular, 1, 1),
        ] {
            let exact = controlled_storage_fixture(
                whole(kernel),
                &vec![word; inputs],
                &vec![word; outputs],
            );
            assert!(exact.is_empty(), "{kernel:?}: {exact:?}");
            for (inputs, outputs) in [(inputs + 1, outputs), (inputs, outputs + 1)]
                .into_iter()
                .chain(inputs.checked_sub(1).map(|fewer| (fewer, outputs)))
                .chain(outputs.checked_sub(1).map(|fewer| (inputs, fewer)))
            {
                let errors = controlled_storage_fixture(
                    whole(kernel),
                    &vec![word; inputs],
                    &vec![word; outputs],
                );
                assert!(
                    errors.iter().any(|error| matches!(
                        error,
                        VerifierError::InvalidVectorOperation { reason, .. }
                            if reason == "computation interface differs from its kernel"
                    )),
                    "{kernel:?} {inputs}/{outputs}: {errors:?}"
                );
            }
        }
    }

    #[test]
    fn every_controlled_kernel_requires_bit_storage_in_each_data_slot() {
        let vector = MockType::NxV4I32;
        let mut cases = vec![
            (Kernel::Move, vec![MockType::I32], vec![vector]),
            (
                Kernel::Unary(VectorUnaryKind::Popcount),
                vec![vector],
                vec![vector],
            ),
            (
                Kernel::Binary(VectorBinaryKind::Add),
                vec![vector, MockType::I32],
                vec![vector],
            ),
            (
                Kernel::IntegerDivide {
                    remainder: false,
                    unsigned: true,
                },
                vec![vector, MockType::I32],
                vec![vector],
            ),
            (
                Kernel::ModuloShift {
                    kind: VectorBinaryKind::Shl,
                },
                vec![vector, MockType::I64],
                vec![vector],
            ),
            (
                Kernel::MultiplyHigh {
                    signed_left: true,
                    signed_right: false,
                },
                vec![vector, MockType::I32],
                vec![vector],
            ),
            (
                Kernel::Compare {
                    relation: crate::ir::ops::VectorCompareKind::Eq,
                    signaling: false,
                },
                vec![vector, MockType::I32],
                vec![MockType::NxMask4],
            ),
            (
                Kernel::Cast {
                    from: VectorCastKind::Signed,
                    to: VectorCastKind::Float,
                },
                vec![vector],
                vec![MockType::V2F64],
            ),
            (
                Kernel::Extend { signed: true },
                vec![MockType::I16],
                vec![vector],
            ),
            (
                Kernel::WidenBinary {
                    kind: VectorBinaryKind::Add,
                    wide_left: false,
                    signed_left: false,
                    signed_right: false,
                },
                vec![MockType::I16, MockType::I16],
                vec![vector],
            ),
            (
                Kernel::FusedMultiplyAdd {
                    negate_product: false,
                    negate_addend: false,
                },
                vec![MockType::V2F64; 3],
                vec![MockType::V2F64],
            ),
            (
                Kernel::IntegerMultiplyAdd {
                    subtract_product: false,
                },
                vec![vector, MockType::I32, vector],
                vec![vector],
            ),
            (
                Kernel::Reduce {
                    kind: VectorReduceKind::Add,
                    ordered: false,
                },
                vec![vector, MockType::I32],
                vec![vector],
            ),
            (Kernel::Gather, vec![vector, MockType::I64], vec![vector]),
            (
                Kernel::Slide {
                    up: true,
                    scalar_fill: false,
                },
                vec![vector, MockType::I64],
                vec![vector],
            ),
            (
                Kernel::Slide {
                    up: false,
                    scalar_fill: true,
                },
                vec![vector, MockType::I32],
                vec![vector],
            ),
            (
                Kernel::Fixed {
                    kind: VectorFixedKind::AverageAdd,
                },
                vec![vector, vector],
                vec![vector],
            ),
            (
                Kernel::Fixed {
                    kind: VectorFixedKind::AverageSub,
                },
                vec![vector, vector],
                vec![vector],
            ),
            (
                Kernel::Fixed {
                    kind: VectorFixedKind::FractionalMultiply,
                },
                vec![vector, vector],
                vec![vector, MockType::I1],
            ),
            (
                Kernel::Fixed {
                    kind: VectorFixedKind::Shift,
                },
                vec![vector, vector],
                vec![vector],
            ),
            (
                Kernel::Fixed {
                    kind: VectorFixedKind::Clip,
                },
                vec![vector, vector],
                vec![vector, MockType::I1],
            ),
            (
                Kernel::WidenMultiplyAdd {
                    signed_left: true,
                    signed_right: false,
                },
                vec![vector, vector, vector],
                vec![vector],
            ),
            (
                Kernel::WidenMultiplyAdd {
                    signed_left: false,
                    signed_right: true,
                },
                vec![vector, vector, vector],
                vec![vector],
            ),
            (Kernel::Index, vec![], vec![vector]),
            (Kernel::InsertScalar, vec![MockType::I32], vec![vector]),
            (
                Kernel::NarrowShift { unsigned: true },
                vec![MockType::I64, MockType::I32],
                vec![vector],
            ),
        ];
        for load in [false, true] {
            for fields in [1, 2] {
                let mut inputs = vec![MockType::Ptr, MockType::I64];
                let outputs = if load {
                    vec![vector; usize::from(fields)]
                } else {
                    inputs.extend(vec![vector; usize::from(fields)]);
                    vec![]
                };
                cases.push((
                    Kernel::Memory {
                        load,
                        access_bits: 8,
                        sign_extend: false,
                        byte_stream: false,
                        coalescible: false,
                        indexed: false,
                        ordered: false,
                        fields,
                        fault: VectorFaultMode::Normal,
                    },
                    inputs,
                    outputs,
                ));
            }
        }
        for fault in [VectorFaultMode::FirstFault, VectorFaultMode::FaultOnlyFirst] {
            cases.push((
                Kernel::Memory {
                    load: true,
                    access_bits: 8,
                    sign_extend: false,
                    byte_stream: false,
                    coalescible: false,
                    indexed: false,
                    ordered: false,
                    fields: 1,
                    fault,
                },
                vec![MockType::Ptr, MockType::I64],
                vec![
                    vector,
                    if fault == VectorFaultMode::FirstFault {
                        MockType::NxMask4
                    } else {
                        MockType::I64
                    },
                ],
            ));
        }
        cases.extend([
            (
                Kernel::Binary(VectorBinaryKind::SatAdd),
                vec![vector, MockType::I32],
                vec![vector, MockType::I1],
            ),
            (Kernel::Move, vec![MockType::BF16], vec![MockType::BF16]),
            (Kernel::Move, vec![MockType::F64], vec![MockType::F64]),
            (Kernel::Move, vec![MockType::Ptr], vec![MockType::Ptr]),
            (
                Kernel::Cast {
                    from: VectorCastKind::BFloat16,
                    to: VectorCastKind::Float,
                },
                vec![MockType::BF16],
                vec![MockType::F64],
            ),
            (
                Kernel::FusedMultiplyAdd {
                    negate_product: true,
                    negate_addend: false,
                },
                vec![MockType::V2F64; 3],
                vec![MockType::V2F64, MockType::I8],
            ),
        ]);
        for relation in [
            VectorMaskBinaryKind::And,
            VectorMaskBinaryKind::Or,
            VectorMaskBinaryKind::Xor,
            VectorMaskBinaryKind::AndNot,
            VectorMaskBinaryKind::OrNot,
            VectorMaskBinaryKind::Nand,
            VectorMaskBinaryKind::Nor,
            VectorMaskBinaryKind::Xnor,
        ] {
            cases.push((
                Kernel::MaskBinary(relation),
                vec![MockType::NxMask4, MockType::NxMask4],
                vec![MockType::NxMask4],
            ));
        }
        for subtract in [false, true] {
            for mask_result in [false, true] {
                cases.push((
                    Kernel::IntegerCarry {
                        subtract,
                        mask_result,
                    },
                    vec![vector, vector, MockType::NxMask4],
                    vec![if mask_result {
                        MockType::NxMask4
                    } else {
                        vector
                    }],
                ));
            }
        }
        cases.push((
            Kernel::Select,
            vec![vector, vector, MockType::NxMask4],
            vec![vector],
        ));
        for prefix in [
            crate::ir::ops::VectorMaskPrefix::Before,
            crate::ir::ops::VectorMaskPrefix::Including,
            crate::ir::ops::VectorMaskPrefix::Only,
        ] {
            cases.push((
                Kernel::MaskPrefix(prefix),
                vec![MockType::NxMask4],
                vec![MockType::NxMask4],
            ));
        }
        cases.push((Kernel::MaskIota, vec![MockType::NxMask4], vec![vector]));
        for first in [false, true] {
            cases.push((
                Kernel::MaskSummary { first },
                vec![MockType::NxMask4],
                vec![MockType::I64],
            ));
        }
        cases.push((
            Kernel::ReduceWide {
                kind: VectorReduceKind::Add,
                ordered: false,
            },
            vec![MockType::I16, vector],
            vec![vector],
        ));
        cases.push((
            Kernel::Compress,
            vec![vector, MockType::NxMask4],
            vec![vector],
        ));
        for count in [vector, MockType::I64] {
            cases.push((
                Kernel::WidenBinary {
                    kind: VectorBinaryKind::Shl,
                    wide_left: false,
                    signed_left: false,
                    signed_right: false,
                },
                vec![vector, count],
                vec![vector],
            ));
        }
        for (negate, xor) in [(false, false), (true, false), (false, true)] {
            cases.push((
                Kernel::SignInject { negate, xor },
                vec![MockType::V2F64, MockType::F64],
                vec![MockType::V2F64],
            ));
        }
        cases.push((Kernel::FloatClassify, vec![MockType::V2F64], vec![vector]));
        for unsigned in [false, true] {
            for nan_maximum in [false, true] {
                cases.push((
                    Kernel::FloatToInteger {
                        unsigned,
                        nan_maximum,
                    },
                    vec![MockType::V2F64],
                    vec![vector, MockType::I8],
                ));
            }
        }
        for square_root in [false, true] {
            cases.push((
                Kernel::Estimate7 { square_root },
                vec![MockType::V2F64],
                vec![MockType::V2F64, MockType::I8],
            ));
        }
        for part in [CarrylessPart::Low, CarrylessPart::High] {
            cases.push((
                Kernel::CarrylessMultiply { part },
                vec![MockType::V4I32, MockType::I64],
                vec![MockType::V4I32],
            ));
        }
        for primitive in [
            VectorCryptoKind::AesEncrypt,
            VectorCryptoKind::AesEncryptLast,
            VectorCryptoKind::AesDecrypt,
            VectorCryptoKind::AesDecryptKeyMix,
            VectorCryptoKind::AesDecryptLast,
            VectorCryptoKind::Sm4Rounds,
            VectorCryptoKind::AesExpand128,
            VectorCryptoKind::AesExpand256,
            VectorCryptoKind::GhashMultiply,
            VectorCryptoKind::GhashMultiplyAdd,
            VectorCryptoKind::Sha2Schedule4,
            VectorCryptoKind::Sha2Compress2Low,
            VectorCryptoKind::Sha2Compress2High,
            VectorCryptoKind::Sm3Expand8,
            VectorCryptoKind::Sm3Compress2,
            VectorCryptoKind::Sm4Expand4,
        ] {
            let mut data = vec![MockType::V4I32; primitive.group_input_count().unwrap()];
            if matches!(
                primitive,
                VectorCryptoKind::AesExpand128
                    | VectorCryptoKind::AesExpand256
                    | VectorCryptoKind::Sm3Compress2
                    | VectorCryptoKind::Sm4Expand4
            ) {
                *data.last_mut().unwrap() = MockType::I32;
            }
            cases.push((Kernel::Crypto(primitive), data, vec![MockType::V4I32]));
        }
        assert_eq!(cases.len(), 89);
        let mut slots = 0;
        for (kernel, data, output_types) in cases {
            let mut kind = Computation {
                canonical_nan: false,
                kernel,
                element: VectorElement::integer(32, None),
                predicate: VectorPredicateLayout::Packed,
                rounding: crate::ir::ops::VectorRounding::None,
                status: VectorStatus::None,
                lanes: Some(LaneControl {
                    mask_tail: false,
                    tail: VectorTailPolicy::Undisturbed,
                    inactive: None,
                    no_passthrough: false,
                }),
            };
            match kernel {
                Kernel::Crypto(primitive) => {
                    kind.element = VectorElement::integer(
                        if matches!(
                            primitive,
                            VectorCryptoKind::Sm3Expand8 | VectorCryptoKind::Sm3Compress2
                        ) {
                            256
                        } else {
                            128
                        },
                        None,
                    );
                }
                Kernel::SignInject { .. } => {
                    kind.element = VectorElement::float(64);
                }
                Kernel::FloatToInteger { unsigned, .. } => {
                    kind.element.signed = Some(!unsigned);
                    kind.rounding = VectorRounding::FloatingEnvironment;
                    kind.status = VectorStatus::FloatingExceptions;
                }
                Kernel::Estimate7 { square_root } => {
                    kind.element = VectorElement::float(64);
                    kind.status = VectorStatus::FloatingExceptions;
                    kind.rounding = if square_root {
                        VectorRounding::None
                    } else {
                        VectorRounding::FloatingEnvironment
                    };
                }
                Kernel::ReduceWide { .. } => kind.element.signed = Some(false),
                Kernel::MaskBinary(_) | Kernel::MaskPrefix(_) | Kernel::MaskSummary { .. } => {
                    kind.element.bits = 1
                }
                Kernel::IntegerDivide { unsigned, .. } | Kernel::NarrowShift { unsigned } => {
                    kind.element.signed = Some(!unsigned)
                }
                Kernel::MultiplyHigh { signed_left, .. }
                | Kernel::WidenBinary { signed_left, .. }
                | Kernel::WidenMultiplyAdd { signed_left, .. } => {
                    kind.element.signed = Some(signed_left)
                }
                Kernel::Cast { .. } | Kernel::FusedMultiplyAdd { .. } => {
                    kind.element.kind = VectorElementKind::Float;
                    kind.element.bits = 64;
                }
                Kernel::Fixed { kind: fixed } => {
                    kind.element.signed = Some(true);
                    kind.rounding = VectorRounding::FixedEnvironment;
                    kind.status = if fixed.saturates() {
                        VectorStatus::Saturation
                    } else {
                        VectorStatus::None
                    };
                }
                Kernel::Binary(VectorBinaryKind::SatAdd) => {
                    kind.element.signed = Some(false);
                    kind.status = VectorStatus::Saturation;
                }
                Kernel::Move if output_types.first() == Some(&MockType::BF16) => {
                    kind.element.kind = VectorElementKind::BFloat16;
                    kind.element.bits = 16;
                }
                Kernel::Move if output_types.first() == Some(&MockType::F64) => {
                    kind.element.kind = VectorElementKind::Float;
                    kind.element.bits = 64;
                }
                Kernel::Move if output_types.first() == Some(&MockType::Ptr) => {
                    kind.element.kind = VectorElementKind::Unknown;
                    kind.element.bits = 64;
                }
                _ => {}
            }
            if matches!(
                kernel,
                Kernel::FusedMultiplyAdd {
                    negate_product: true,
                    ..
                }
            ) {
                kind.status = VectorStatus::FloatingExceptions;
            }
            assert!(kind.is_valid(), "{kernel:?}");
            let mut inputs = vec![MockType::I64, MockType::I64];
            if matches!(
                kind.rounding,
                VectorRounding::FloatingEnvironment | VectorRounding::FixedEnvironment
            ) {
                inputs.push(MockType::I32);
            }
            inputs.extend(output_types.iter().take(kind.data_outputs()).copied());
            inputs.extend(data);
            assert_eq!(inputs.len(), kind.input_count());
            assert_eq!(output_types.len(), kind.output_count());
            let errors = controlled_storage_fixture(kind, &inputs, &output_types);
            assert!(errors.is_empty(), "{kernel:?}: {errors:?}");
            for index in kind.control_input_count()..inputs.len() {
                let mut bad = inputs.clone();
                bad[index] = MockType::Ref;
                let errors = controlled_storage_fixture(kind, &bad, &output_types);
                assert!(
                    errors
                        .iter()
                        .any(|error| error.to_string().contains("bit representation")),
                    "{kernel:?}, input={index}: {errors:?}"
                );
                bad[index] = MockType::Unknown;
                let errors = controlled_storage_fixture(kind, &bad, &output_types);
                // An unknown passthrough still needs its result to agree.
                let mut outputs = output_types.clone();
                if index < kind.control_input_count() + kind.data_outputs() {
                    outputs[index - kind.control_input_count()] = MockType::Unknown;
                }
                let errors = if outputs != output_types {
                    controlled_storage_fixture(kind, &bad, &outputs)
                } else {
                    errors
                };
                assert!(
                    errors.is_empty(),
                    "unknown {kernel:?}, input={index}: {errors:?}"
                );
                slots += 1;
            }
            for index in 0..output_types.len() {
                let mut bad = output_types.clone();
                bad[index] = MockType::Ref;
                let mut old = inputs.clone();
                if index < kind.data_outputs() {
                    old[kind.control_input_count() + index] = MockType::Ref;
                }
                let errors = controlled_storage_fixture(kind, &old, &bad);
                assert!(
                    errors
                        .iter()
                        .any(|error| error.to_string().contains("bit representation")),
                    "{kernel:?}, output={index}: {errors:?}"
                );
                bad[index] = MockType::Unknown;
                if index < kind.data_outputs() {
                    old[kind.control_input_count() + index] = MockType::Unknown;
                }
                let errors = controlled_storage_fixture(kind, &old, &bad);
                assert!(
                    errors.is_empty(),
                    "unknown {kernel:?}, output={index}: {errors:?}"
                );
                slots += 1;
            }
        }
        // Mask relations add four slots; each scan adds one source, old value and output.
        assert_eq!(
            slots,
            153 + 8 * 4 + 4 * 5 + 5 + 6 * 3 + 4 * 4 + 39 + 8 + 10 * 4 + 6 * 5
        );
    }
}
