//! Reading a flag through the operations that compose a flags value.
//!
//! A flags value is not always one operation's: an instruction that sets some
//! condition flags and leaves the rest keeps the others from the instruction
//! before it ([`SsaOp::MergeFlags`]), and a flag no operation describes is set
//! from a boolean ([`SsaOp::InsertFlag`]). A [`SsaOp::ReadFlags`] of such a
//! value reads whichever input the bits it selects came from, and [`route`]
//! says which: the flags value to read instead, or the boolean itself.
//!
//! A branch on a condition no single flag bit carries is a tree of flag reads
//! a host's lift expands it into; [`fold_flag_conditions`] turns it back into
//! the comparison the program made, from the relations the host recorded, and
//! [`fold_result_flag_reads`] reads the zero and sign flags of an operation
//! that sets them from its result as that result's comparison with zero, and
//! the carry of an addition or a subtraction as the unsigned relation it
//! records.
//!
//! # Examples
//!
//! ```text
//! // v2 = mergeflags v0, v1, ZF        zero from v1, everything else from v0
//! // v3 = readflags v2, CF             -> readflags v0, CF
//! // v4 = readflags v2, ZF             -> readflags v1, ZF
//! // v6 = insertflag v2, v5, CF        carry is the boolean v5
//! // v7 = readflags v6, CF             -> v5
//! ```

use crate::{
    Result,
    ir::{
        CmpKind, ConstValue, PhiNode, PhiOperand, SsaBlock, SsaEditOptions, SsaFunction,
        SsaInstruction, SsaRollbackPolicy,
        function::SsaEditor,
        lanes::VectorConst,
        ops::{FlagCondition, FlagsMask, SsaOp},
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::{Target, VectorShape},
};

/// How many composing operations a read is followed through.
///
/// Each step leaves one definition behind, and a function's flags values
/// compose at most once per flag-setting instruction, so a real chain is short;
/// the bound only keeps a malformed chain from costing more than it could be
/// worth.
const MOST_COMPOSITIONS_FOLLOWED: usize = 64;

/// What a read of some flags resolves to once the composition is seen through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagSource {
    /// The same bits, read from this flags value.
    Flags(SsaVarId),
    /// The boolean the one bit read was set from.
    Bit(SsaVarId),
    /// The bits read from one of two flags values, which a select chooses by
    /// `condition`: a shift's flags where a count of zero keeps the flags it
    /// inherited, or where a count past the width leaves another shift's.
    Chosen {
        /// The select's condition.
        condition: SsaVarId,
        /// The flags it chooses where the condition holds.
        true_flags: SsaVarId,
        /// The flags it chooses otherwise.
        false_flags: SsaVarId,
    },
}

/// Resolves a read of `mask` from `flags` through the operations that composed
/// it.
///
/// A read whose bits all come from one input follows that input; one that
/// spans the bits of two inputs stops where it is, since no single source
/// holds them all.
///
/// # Arguments
///
/// * `ssa` - The function holding the definitions.
/// * `flags` - The flags value read.
/// * `mask` - The bits read.
///
/// # Returns
///
/// The source the read resolves to, or `None` when it already reads its source
/// (nothing composed `flags`, or the read spans two inputs at the first step).
#[must_use]
pub fn route<T: Target>(
    ssa: &SsaFunction<T>,
    flags: SsaVarId,
    mask: FlagsMask,
) -> Option<FlagSource> {
    if mask.is_empty() {
        return None;
    }
    let mut current = flags;
    for _ in 0..MOST_COMPOSITIONS_FOLLOWED {
        current = match ssa.get_definition(current) {
            Some(SsaOp::MergeFlags {
                base,
                update,
                mask: merged,
                ..
            }) => {
                if merged.contains(mask) {
                    *update
                } else if !merged.intersects(mask) {
                    *base
                } else {
                    break;
                }
            }
            Some(SsaOp::InsertFlag {
                base,
                bit,
                mask: inserted,
                ..
            }) => {
                if *inserted == mask {
                    return Some(FlagSource::Bit(*bit));
                } else if !inserted.intersects(mask) {
                    *base
                } else {
                    break;
                }
            }
            Some(SsaOp::Copy { src, .. }) => *src,
            Some(SsaOp::Select {
                condition,
                true_val,
                false_val,
                ..
            }) => {
                return Some(FlagSource::Chosen {
                    condition: *condition,
                    true_flags: *true_val,
                    false_flags: *false_val,
                });
            }
            _ => break,
        };
    }
    (current != flags).then_some(FlagSource::Flags(current))
}

/// How deep a join of flags values is followed before giving up.
///
/// A flags phi whose operand is another flags phi is a nest of loop headers;
/// two levels covers the shapes a compiler emits and keeps a malformed cycle
/// from running away.
const MAX_JOIN_DEPTH: usize = 2;

/// The relation a flag-setting instruction states about its operands, as the
/// host recorded it.
///
/// Recorded because the IR cannot say it: a comparison lowers to a `Sub` with
/// a flags output, and so does a real subtraction. Only the lift knows which
/// one is a comparison, and folding a condition against an arithmetic `Sub`
/// would state a relation the program never made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompareRelation {
    /// Left-hand operand of the comparison.
    pub left: SsaVarId,
    /// Right-hand operand of the comparison.
    pub right: SsaVarId,
    /// Whether the comparison happened only under a guard (AArch64 `ccmp`).
    ///
    /// A guarded relation holds on one path and not the other, and
    /// `SsaOp::BranchCmp` has nowhere to put the guard, so a fold declines it
    /// rather than assert the unguarded relation.
    pub conditional: bool,
    /// Whether the comparison compares floats (`fcmp`, `ucomiss`): a
    /// condition reads its flags by its float reading
    /// ([`ExpandedCondition::float`]), a NaN unordered with everything.
    pub float: bool,
}

/// A branch whose ordering condition the lift expanded into flag reads.
///
/// `Above`, `BelowOrEqual`, `Greater`, `GreaterOrEqual`, `Less` and
/// `LessOrEqual` are not single flag bits, so the lift expands them into a
/// `ReadFlags`/`Ceq`/`And` tree. Across a block boundary the comparison that
/// produced those flags is out of reach until phis resolve the block's flag
/// read, so the lift records the intent and [`fold_flag_conditions`] completes
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandedCondition {
    /// Block whose terminator branches on the expanded condition.
    pub block_idx: usize,
    /// Comparison the condition encodes after an integer comparison.
    pub cmp: CmpKind,
    /// Whether the comparison reads its operands as unsigned.
    pub unsigned: bool,
    /// The comparison the condition encodes after a float compare, and
    /// whether it also holds unordered, or `None` for a condition that states
    /// no float relation (an unordered test). The host's flag layout decides
    /// it: AArch64's `b.gt` after `fcmp` is an ordered greater-than, x86's
    /// `ja` after `ucomiss` too.
    pub float: Option<(CmpKind, bool)>,
}

/// One side of the comparison a fold needs, as it must be materialised.
enum Side {
    /// Every edge brings the same value, so the value itself is the operand.
    Same(SsaVarId),
    /// The edges disagree, so the operand is a phi over them.
    Join(Vec<(usize, SsaVarId)>),
}

/// One planned terminator rewrite.
struct Fold {
    /// Block whose terminator is replaced.
    block_idx: usize,
    /// The comparison's left operand, and how to materialise it.
    left: Side,
    /// The comparison's right operand, and how to materialise it.
    right: Side,
    /// The comparison the expanded tree stated.
    cmp: CmpKind,
    /// Whether that comparison reads its operands as unsigned.
    unsigned: bool,
}

/// Rewrites each expanded condition's branch into the comparison the program
/// made.
///
/// For each recorded branch the flags value its block reads is looked up in
/// `relation`, and the two-way branch becomes a `BranchCmp` of the relation's
/// operands. The expanded tree is pure and becomes unreachable; dead-code
/// elimination removes it.
///
/// # Soundness
///
/// - **Only recorded comparisons.** A flags value folds only when `relation`
///   answers for it, never because its defining op happens to be a `Sub`.
/// - **Only one flags value.** A block reading two different flags values is
///   ambiguous about which produced the condition, and is skipped.
/// - **Never a guarded relation.** A conditional compare holds on one path
///   only, and `SsaOp::BranchCmp` has nowhere to put the guard.
/// - **The reading of the compare's kind.** A float compare's flags read as
///   the float relation the condition states after one
///   ([`ExpandedCondition::float`]), whose `unsigned` is the unordered case;
///   a condition with no float reading, or flags joined from a float and an
///   integer compare, is skipped.
///
/// # A flags value that arrives through a join
///
/// The block that tests the flags is often a loop header, the flags reaching
/// it through a phi: one comparison on the way in, another on the back edge.
/// A join of flags values is a join of relations. When every operand of the
/// phi is a flags value with an unguarded relation, the condition is the
/// comparison of two new phis, one over the left operands and one over the
/// right, placed in the same block and reading the same edges. Either side
/// collapses to the value itself when every edge brings the same one. One
/// unrecorded or guarded operand declines the join whole.
///
/// # Arguments
///
/// * `ir` - The function, rewritten in place.
/// * `conditions` - The branches the lift expanded.
/// * `relation` - The relation a flags value states, where the lift recorded
///   one.
///
/// # Returns
///
/// `true` when at least one terminator was rewritten. All or nothing: a fold
/// that fails partway restores the function as it was.
pub fn fold_flag_conditions<T: Target>(
    ir: &mut SsaFunction<T>,
    conditions: &[ExpandedCondition],
    relation: impl Fn(SsaVarId) -> Option<CompareRelation>,
) -> bool {
    let mut plan = Vec::new();
    for expanded in conditions {
        let Some(block) = ir.block(expanded.block_idx) else {
            continue;
        };
        if !matches!(block.control_terminator(), Some(SsaOp::Branch { .. })) {
            continue;
        }
        let Some(flags) = sole_flags_read(block) else {
            continue;
        };
        let Some(relations) = relations_of(ir, &relation, flags, 0) else {
            continue;
        };
        // The condition reads a float compare's flags as the float relation
        // its layout states, and an integer one's as the integer relation;
        // flags joined from both kinds state neither.
        let float = relations.iter().all(|(_, relation)| relation.float);
        let reading = if float {
            expanded.float
        } else if relations.iter().all(|(_, relation)| !relation.float) {
            Some((expanded.cmp, expanded.unsigned))
        } else {
            None
        };
        let Some((cmp, unsigned)) = reading else {
            continue;
        };
        let Some((left, right)) = sides_of(&relations) else {
            continue;
        };
        plan.push(Fold {
            block_idx: expanded.block_idx,
            left,
            right,
            cmp,
            unsigned,
        });
    }
    if plan.is_empty() {
        return false;
    }
    let mut folded = false;
    let options = SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure);
    let report = ir.edit(options, |editor| {
        for fold in plan {
            let Some((true_target, false_target)) = editor
                .function()
                .block(fold.block_idx)
                .and_then(|block| block.control_terminator())
                .and_then(SsaOp::two_way_targets)
            else {
                continue;
            };
            let left = materialise(editor, fold.block_idx, fold.left)?;
            let right = materialise(editor, fold.block_idx, fold.right)?;
            editor.replace_terminator_op(
                fold.block_idx,
                SsaOp::BranchCmp {
                    left,
                    right,
                    cmp: fold.cmp,
                    unsigned: fold.unsigned,
                    true_target,
                    false_target,
                },
            )?;
            folded = true;
        }
        Ok(())
    });
    report.is_ok() && folded
}

/// Rewrites every read of a flag an operation sets from its operands and
/// result into the expression it states.
///
/// ZF is `result == 0` and SF is `result < 0` for every operation that states
/// them from its result, and saying so is what makes the read legible and
/// gives the value analyses a relation to use. A read that reaches the
/// operation through the compositions of the flags value ([`route`]) reads
/// flags the operation stated. Division and remainder are left out; where
/// they state flags at all, the machine leaves ZF and SF undefined. Only an
/// integer result of 8, 16, 32 or 64 bits is compared.
///
/// CF of `sum = left + right` is `sum < left`, unsigned: the addition wrapped
/// exactly when the sum came out below an operand. That holds on every
/// architecture, and it is how a 32-bit target's 64-bit addition reads - `lo =
/// a_lo + b_lo; hi = a_hi + b_hi + (lo < a_lo)` - rather than through a flags
/// value no source names. A subtraction's carry is the borrow, `left < right`,
/// or its complement, `left >= right`, as the conventions say the architecture
/// sets it, and a negation's is that of `0 - x`: `x != 0` borrowing, `x == 0`
/// not; it stays a read where the host does not say.
///
/// OF is the signed overflow, the same on every architecture: an addition
/// overflows where both operands share a sign the result does not,
/// `((left ^ result) & (right ^ result)) < 0`, a subtraction where the
/// operands differ in sign and the result's differs from the minuend's,
/// `((left ^ right) & (left ^ result)) < 0`, and a negation of the least
/// value, `x == MIN`. Where the conventions say a logical operation clears the
/// carry and the overflow (x86), a read of either of an `and`, `or` or `xor`
/// is false; ARM's logical operations take their carry from the shifter and
/// keep their overflow, which stay reads. Where they say the parity and
/// adjust flags are x86's, PF is the even parity of the result's low byte,
/// for every operation that states it from its result but a multiplication,
/// which leaves it undefined, and AF is the carry out of bit 3, bit 4 of
/// `left ^ right ^ result`, of an addition, a subtraction or a negation.
///
/// A multiplication's and a shift's carry and overflow are no expression of
/// their result and operands alone, and stay reads.
///
/// # Arguments
///
/// * `ir` - The function whose flag reads are rewritten.
/// * `conventions` - What the flags mean where architectures disagree.
///
/// # Returns
///
/// `true` when at least one read was rewritten.
pub fn fold_result_flag_reads<T: Target>(
    ir: &mut SsaFunction<T>,
    conventions: FlagConventions,
) -> bool {
    // A read of flags a select chooses is the select of two reads, each a read
    // to fold, so the rewrite runs again while it rewrites anything.
    let mut folded = false;
    for _ in 0..MOST_COMPOSITIONS_FOLLOWED {
        if !fold_result_flag_reads_once(ir, conventions) {
            break;
        }
        folded = true;
    }
    folded
}

/// One round of [`fold_result_flag_reads`]: rewrites the reads there are now.
fn fold_result_flag_reads_once<T: Target>(
    ir: &mut SsaFunction<T>,
    conventions: FlagConventions,
) -> bool {
    let mut plan: Vec<(usize, usize, SsaVarId, ResultFlag)> = Vec::new();
    for (block_idx, block) in ir.iter_blocks() {
        for (instr_idx, instruction) in block.instructions().iter().enumerate() {
            let SsaOp::ReadFlags { dest, flags, mask } = instruction.op() else {
                continue;
            };
            if let Some(read) = ResultFlag::read(ir, *flags, *mask, conventions) {
                plan.push((block_idx, instr_idx, *dest, read));
            }
        }
    }
    if plan.is_empty() {
        return false;
    }
    // Last first within a block, so an insertion leaves the planned indices
    // before it where they were.
    plan.sort_by(|a, b| (a.0, b.1).cmp(&(b.0, a.1)));
    let mut folded = false;
    let options = SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure);
    let report = ir.edit(options, |editor| {
        for (block_idx, instr_idx, dest, read) in plan {
            let mut lowering = FlagLowering {
                editor: &mut *editor,
                block_idx,
                instr_idx,
                inserted: 0,
            };
            if lowering.lower(dest, read)? {
                folded = true;
            }
        }
        Ok(())
    });
    report.is_ok() && folded
}

/// Rewrites every branch on one flag an operation sets from its operands and
/// result into a branch on that flag's value, stated as
/// [`fold_result_flag_reads`] states a read of it: x86 `test eax, eax; jne`
/// branches on `eax & eax == 0`, its targets exchanged, and the `and` whose
/// flags it read is left to dead-code elimination where nothing else reads it.
///
/// A branch on flags a select chooses between two operations' (ARM's
/// conditional compares) branches on the select of the two flags' values,
/// and one on a flag set from a boolean ([`SsaOp::InsertFlag`], ARM's `sbcs`
/// carry and overflow, which its borrow-in decides) branches on the boolean.
///
/// The flag's value is computed before the branch, which then branches on it,
/// its targets exchanged for a condition that tests the flag clear. Only where
/// the value can be stated: a branch on a flag of a value that is no integer
/// of 8, 16, 32 or 64 bits, or that no result states (a multiplication's
/// carry), stays a branch on the flags.
///
/// # Arguments
///
/// * `ir` - The function whose flag branches are rewritten.
/// * `conventions` - What the flags mean where architectures disagree.
///
/// # Returns
///
/// `true` when at least one branch was rewritten.
pub fn fold_result_flag_branches<T: Target>(
    ir: &mut SsaFunction<T>,
    conventions: FlagConventions,
) -> bool {
    let mut plan: Vec<(usize, SsaVarId, FlagsMask, ResultFlag, bool, usize, usize)> = Vec::new();
    // Branches on a flag set from a boolean: the block, the boolean, and the
    // targets where it holds and where it does not.
    let mut on_bits: Vec<(usize, SsaVarId, usize, usize)> = Vec::new();
    for (block_idx, block) in ir.iter_blocks() {
        let Some(SsaOp::BranchFlags {
            flags,
            condition,
            true_target,
            false_target,
        }) = block.control_terminator()
        else {
            continue;
        };
        let (mask, when_set) = match condition {
            FlagCondition::Carry => (FlagsMask::CARRY, true),
            FlagCondition::NotCarry => (FlagsMask::CARRY, false),
            FlagCondition::Zero => (FlagsMask::ZERO, true),
            FlagCondition::NotZero => (FlagsMask::ZERO, false),
            FlagCondition::Overflow => (FlagsMask::OVERFLOW, true),
            FlagCondition::NotOverflow => (FlagsMask::OVERFLOW, false),
            FlagCondition::Negative => (FlagsMask::SIGN, true),
            FlagCondition::Positive => (FlagsMask::SIGN, false),
            FlagCondition::ParityEven => (FlagsMask::PARITY, true),
            FlagCondition::ParityOdd => (FlagsMask::PARITY, false),
        };
        // A flag set from a boolean is the boolean.
        if let Some(FlagSource::Bit(bit)) = route(ir, *flags, mask) {
            let (holds, fails) = if when_set {
                (*true_target, *false_target)
            } else {
                (*false_target, *true_target)
            };
            on_bits.push((block_idx, bit, holds, fails));
            continue;
        }
        // Flags a select chooses (ARM's `cmp r0, #61; cmpne r0, #0; bne`
        // keeps the first compare's flags where `r0` is 61): the branch is on
        // the select of the two flags' values, each a read the read fold
        // states in turn.
        let read = ResultFlag::read(ir, *flags, mask, conventions);
        if let Some(read) = read {
            plan.push((
                block_idx,
                *flags,
                mask,
                read,
                when_set,
                *true_target,
                *false_target,
            ));
        }
    }
    if plan.is_empty() && on_bits.is_empty() {
        return false;
    }
    // One session for the whole plan: a session's repair compacts the
    // variables, and the plan names them by the ids they had when it was made.
    let mut folded = false;
    let options = SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure);
    let report = ir.edit(options, |editor| {
        for (block_idx, bit, holds, fails) in on_bits {
            editor.replace_terminator_op(
                block_idx,
                SsaOp::Branch {
                    condition: bit,
                    true_target: holds,
                    false_target: fails,
                },
            )?;
            folded = true;
        }
        for (block_idx, flags, mask, read, when_set, true_target, false_target) in plan {
            let Some(at) = editor
                .function()
                .block(block_idx)
                .map(|block| block.instructions().len().saturating_sub(1))
            else {
                continue;
            };
            let bit = editor.create_fresh_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(block_idx, at),
                T::comparison_result_type().unwrap_or_else(T::unknown_type),
            );
            // The read the value is stated in place of, which the lowering
            // replaces with its last step.
            editor.insert_instruction(
                block_idx,
                at,
                SsaInstruction::synthetic(SsaOp::ReadFlags {
                    dest: bit,
                    flags,
                    mask,
                }),
            )?;
            let mut lowering = FlagLowering {
                editor: &mut *editor,
                block_idx,
                instr_idx: at,
                inserted: 0,
            };
            if !lowering.lower(bit, read)? {
                editor.nop_instruction(block_idx, at)?;
                continue;
            }
            let (true_target, false_target) = if when_set {
                (true_target, false_target)
            } else {
                (false_target, true_target)
            };
            editor.replace_terminator_op(
                block_idx,
                SsaOp::Branch {
                    condition: bit,
                    true_target,
                    false_target,
                },
            )?;
            folded = true;
        }
        Ok(())
    });
    report.is_ok() && folded
}

/// What an architecture's flags mean where architectures disagree, which
/// [`fold_result_flag_reads`] reads them by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagConventions {
    /// What a subtraction's carry says.
    pub subtraction: SubtractionCarry,
    /// Whether a logical operation clears the carry and the overflow (x86's
    /// `and`, `or`, `xor`, `test`); ARM's take their carry from the shifter
    /// and keep their overflow.
    pub logic_clears_carry_and_overflow: bool,
    /// Whether the parity and adjust flags are x86's: PF the even parity of
    /// the result's low byte, AF the carry out of bit 3.
    pub parity_and_adjust: bool,
}

impl FlagConventions {
    /// x86's: the carry of a subtraction is its borrow, a logical operation
    /// clears the carry and the overflow, and PF and AF are stated.
    pub const X86: Self = Self {
        subtraction: SubtractionCarry::Borrow,
        logic_clears_carry_and_overflow: true,
        parity_and_adjust: true,
    };

    /// ARM's and AArch64's: the carry of a subtraction is set without a
    /// borrow, and a logical operation's carry and overflow are not its own.
    pub const ARM: Self = Self {
        subtraction: SubtractionCarry::NoBorrow,
        logic_clears_carry_and_overflow: false,
        parity_and_adjust: false,
    };

    /// No convention: only the flags every architecture agrees on fold.
    pub const UNSTATED: Self = Self {
        subtraction: SubtractionCarry::Unstated,
        logic_clears_carry_and_overflow: false,
        parity_and_adjust: false,
    };
}

/// Emits the expression a flag read is rewritten into, before the read, and
/// replaces the read with its last step.
struct FlagLowering<'e, 'f, T: Target> {
    /// The editor of the function.
    editor: &'e mut SsaEditor<'f, T>,
    /// The block of the read.
    block_idx: usize,
    /// The read's index before anything is inserted.
    instr_idx: usize,
    /// How many instructions have been inserted before the read.
    inserted: usize,
}

impl<T: Target> FlagLowering<'_, '_, T> {
    /// The type of `var`.
    fn type_of(&self, var: SsaVarId) -> Option<T::Type> {
        self.editor
            .function()
            .variable(var)
            .map(|variable| variable.var_type().clone())
    }

    /// A fresh variable of `ty`, defined where the read is.
    fn fresh(&mut self, ty: T::Type) -> SsaVarId {
        let site =
            DefSite::instruction(self.block_idx, self.instr_idx.saturating_add(self.inserted));
        self.editor
            .create_fresh_variable(VariableOrigin::Phi, 0, site, ty)
    }

    /// Inserts `op` before the read.
    fn emit(&mut self, op: SsaOp<T>) -> Result<()> {
        self.editor.insert_instruction(
            self.block_idx,
            self.instr_idx.saturating_add(self.inserted),
            SsaInstruction::synthetic(op),
        )?;
        self.inserted = self.inserted.saturating_add(1);
        Ok(())
    }

    /// A fresh `ty` defined as `value`.
    fn constant(&mut self, ty: T::Type, value: ConstValue<T>) -> Result<SsaVarId> {
        let dest = self.fresh(ty);
        self.emit(SsaOp::Const { dest, value })?;
        Ok(dest)
    }

    /// A fresh `ty` defined by `op` of it.
    fn step(&mut self, ty: &T::Type, op: impl FnOnce(SsaVarId) -> SsaOp<T>) -> Result<SsaVarId> {
        let dest = self.fresh(ty.clone());
        self.emit(op(dest))?;
        Ok(dest)
    }

    /// Replaces the read with `op`.
    fn finish(&mut self, op: SsaOp<T>) -> Result<bool> {
        self.editor.replace_instruction_op(
            self.block_idx,
            self.instr_idx.saturating_add(self.inserted),
            op,
        )?;
        Ok(true)
    }

    /// The vector type of `var` and its shape, where it is a vector.
    fn vector(&self, var: SsaVarId) -> Option<(T::Type, VectorShape)> {
        let ty = self.type_of(var)?;
        let shape = T::vector_shape(&ty)?;
        Some((ty, shape))
    }

    /// The integer type of `var` and its width, where it is an integer of 8,
    /// 16, 32 or 64 bits.
    fn integer(&self, var: SsaVarId) -> Option<(T::Type, u32)> {
        let ty = self.type_of(var).filter(T::is_integer)?;
        let bits = T::bit_width(&ty).filter(|bits| matches!(bits, 8 | 16 | 32 | 64))?;
        Some((ty, bits))
    }

    /// Rewrites the read defining `dest` into `read`'s expression; `false`
    /// where its values are not integers it can state it of.
    fn lower(&mut self, dest: SsaVarId, read: ResultFlag) -> Result<bool> {
        let Some(flag_type) = self.type_of(dest) else {
            return Ok(false);
        };
        match read {
            ResultFlag::Known(value) => self.finish(SsaOp::Const {
                dest,
                value: ConstValue::from_bool(value),
            }),
            ResultFlag::Chosen {
                condition,
                true_flags,
                false_flags,
                mask,
            } => {
                let read = |flags: SsaVarId| {
                    move |bit| SsaOp::ReadFlags {
                        dest: bit,
                        flags,
                        mask,
                    }
                };
                let true_val = self.step(&flag_type, read(true_flags))?;
                let false_val = self.step(&flag_type, read(false_flags))?;
                self.finish(SsaOp::Select {
                    dest,
                    condition,
                    true_val,
                    false_val,
                })
            }
            // A vector is zero where every bit is clear (x86 `ptest`); it has
            // no sign.
            ResultFlag::Zero(result) if self.vector(result).is_some() => {
                let Some((ty, shape)) = self.vector(result) else {
                    return Ok(false);
                };
                let Some(zero) = VectorConst::zeroed(shape.lane_bits, shape.lane_count) else {
                    return Ok(false);
                };
                let zero = self.constant(ty, ConstValue::Vector(zero))?;
                self.finish(SsaOp::Ceq {
                    dest,
                    left: result,
                    right: zero,
                    signaling: false,
                })
            }
            ResultFlag::Zero(result) | ResultFlag::Sign(result) => {
                let Some((ty, bits)) = self.integer(result) else {
                    return Ok(false);
                };
                let zero = self.constant(ty, integer_const(bits, 0))?;
                self.finish(if matches!(read, ResultFlag::Sign(_)) {
                    SsaOp::Clt {
                        dest,
                        left: result,
                        right: zero,
                        unsigned: false,
                        signaling: false,
                    }
                } else {
                    SsaOp::Ceq {
                        dest,
                        left: result,
                        right: zero,
                        signaling: false,
                    }
                })
            }
            ResultFlag::Carry { sum, left } => {
                if self.integer(sum).is_none() {
                    return Ok(false);
                }
                self.finish(SsaOp::Clt {
                    dest,
                    left: sum,
                    right: left,
                    unsigned: true,
                    signaling: false,
                })
            }
            ResultFlag::Borrow {
                left,
                right,
                set_when_borrowing,
            } => {
                if self.integer(left).is_none() {
                    return Ok(false);
                }
                if set_when_borrowing {
                    return self.finish(SsaOp::Clt {
                        dest,
                        left,
                        right,
                        unsigned: true,
                        signaling: false,
                    });
                }
                // `left >= right`: the borrow's comparison, negated.
                let below = self.step(&flag_type, |below| SsaOp::Clt {
                    dest: below,
                    left,
                    right,
                    unsigned: true,
                    signaling: false,
                })?;
                self.finish(SsaOp::BoolNot { dest, value: below })
            }
            ResultFlag::NegationBorrow {
                operand,
                set_when_borrowing,
            } => {
                let Some((ty, bits)) = self.integer(operand) else {
                    return Ok(false);
                };
                let zero = self.constant(ty, integer_const(bits, 0))?;
                self.finish(if set_when_borrowing {
                    SsaOp::Cgt {
                        dest,
                        left: operand,
                        right: zero,
                        unsigned: true,
                        signaling: false,
                    }
                } else {
                    SsaOp::Ceq {
                        dest,
                        left: operand,
                        right: zero,
                        signaling: false,
                    }
                })
            }
            ResultFlag::Overflow {
                left,
                right,
                result,
                subtraction,
            } => {
                let Some((ty, bits)) = self.integer(result) else {
                    return Ok(false);
                };
                let xor = |this: &mut Self, a, b| {
                    this.step(&ty, |dest| SsaOp::Xor {
                        dest,
                        left: a,
                        right: b,
                        flags: None,
                    })
                };
                let (first, second) = if subtraction {
                    (xor(self, left, right)?, xor(self, left, result)?)
                } else {
                    (xor(self, left, result)?, xor(self, right, result)?)
                };
                let both = self.step(&ty, |dest| SsaOp::And {
                    dest,
                    left: first,
                    right: second,
                    flags: None,
                })?;
                let zero = self.constant(ty.clone(), integer_const(bits, 0))?;
                self.finish(SsaOp::Clt {
                    dest,
                    left: both,
                    right: zero,
                    unsigned: false,
                    signaling: false,
                })
            }
            ResultFlag::NegationOverflow(operand) => {
                let Some((ty, bits)) = self.integer(operand) else {
                    return Ok(false);
                };
                let least = i64::MIN >> 64u32.saturating_sub(bits);
                let least = self.constant(ty, integer_const(bits, least))?;
                self.finish(SsaOp::Ceq {
                    dest,
                    left: operand,
                    right: least,
                    signaling: false,
                })
            }
            ResultFlag::Parity(result) => {
                let Some((ty, bits)) = self.integer(result) else {
                    return Ok(false);
                };
                // Fold the low byte's bits into bit 0 with three shifted
                // exclusive-ors: even parity leaves it clear.
                let low = self.constant(ty.clone(), integer_const(bits, 0xff))?;
                let mut folded = self.step(&ty, |dest| SsaOp::And {
                    dest,
                    left: result,
                    right: low,
                    flags: None,
                })?;
                for by in [4, 2, 1] {
                    let amount = self.constant(ty.clone(), integer_const(bits, by))?;
                    let shifted = self.step(&ty, |dest| SsaOp::Shr {
                        dest,
                        value: folded,
                        amount,
                        unsigned: true,
                        flags: None,
                    })?;
                    folded = self.step(&ty, |dest| SsaOp::Xor {
                        dest,
                        left: folded,
                        right: shifted,
                        flags: None,
                    })?;
                }
                let one = self.constant(ty.clone(), integer_const(bits, 1))?;
                let odd = self.step(&ty, |dest| SsaOp::And {
                    dest,
                    left: folded,
                    right: one,
                    flags: None,
                })?;
                let zero = self.constant(ty, integer_const(bits, 0))?;
                self.finish(SsaOp::Ceq {
                    dest,
                    left: odd,
                    right: zero,
                    signaling: false,
                })
            }
            ResultFlag::Adjust {
                left,
                right,
                result,
            } => {
                let Some((ty, bits)) = self.integer(result) else {
                    return Ok(false);
                };
                let mut mixed = self.step(&ty, |dest| SsaOp::Xor {
                    dest,
                    left,
                    right: result,
                    flags: None,
                })?;
                if let Some(right) = right {
                    mixed = self.step(&ty, |dest| SsaOp::Xor {
                        dest,
                        left: mixed,
                        right,
                        flags: None,
                    })?;
                }
                let four = self.constant(ty.clone(), integer_const(bits, 4))?;
                let shifted = self.step(&ty, |dest| SsaOp::Shr {
                    dest,
                    value: mixed,
                    amount: four,
                    unsigned: true,
                    flags: None,
                })?;
                let one = self.constant(ty.clone(), integer_const(bits, 1))?;
                let bit = self.step(&ty, |dest| SsaOp::And {
                    dest,
                    left: shifted,
                    right: one,
                    flags: None,
                })?;
                let zero = self.constant(ty, integer_const(bits, 0))?;
                self.finish(SsaOp::Cgt {
                    dest,
                    left: bit,
                    right: zero,
                    unsigned: true,
                    signaling: false,
                })
            }
        }
    }
}

/// The integer constant `value` of `bits` bits, its low bits kept.
fn integer_const<T: Target>(bits: u32, value: i64) -> ConstValue<T> {
    match bits {
        8 => ConstValue::I8(value as i8),
        16 => ConstValue::I16(value as i16),
        32 => ConstValue::I32(value as i32),
        _ => ConstValue::I64(value),
    }
}

/// What a subtraction's carry flag says, which architectures disagree on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtractionCarry {
    /// Set when the subtraction borrows, `left < right` unsigned: x86.
    Borrow,
    /// Set when it does not, `left >= right` unsigned: ARM and AArch64.
    NoBorrow,
    /// Not stated: a read of it stays a read.
    Unstated,
}

/// Which flag of an operation's result a read is rewritten from.
#[derive(Debug, Clone, Copy)]
enum ResultFlag {
    /// ZF, of the result the operation stated it from.
    Zero(SsaVarId),
    /// SF, likewise.
    Sign(SsaVarId),
    /// CF of an addition: its sum, and the operand an overflow leaves it below.
    Carry {
        /// The addition's result.
        sum: SsaVarId,
        /// Its left operand.
        left: SsaVarId,
    },
    /// CF of a subtraction: its operands, and which way round the flag is.
    Borrow {
        /// The minuend.
        left: SsaVarId,
        /// The subtrahend.
        right: SsaVarId,
        /// Whether the flag is set when the subtraction borrows.
        set_when_borrowing: bool,
    },
    /// CF of a negation, that of `0 - operand`.
    NegationBorrow {
        /// The negated value.
        operand: SsaVarId,
        /// Whether the flag is set when the subtraction borrows.
        set_when_borrowing: bool,
    },
    /// OF of an addition or a subtraction: the signed overflow.
    Overflow {
        /// The left operand.
        left: SsaVarId,
        /// The right operand.
        right: SsaVarId,
        /// The result.
        result: SsaVarId,
        /// Whether the operation subtracts.
        subtraction: bool,
    },
    /// OF of a negation: the operand is the least value.
    NegationOverflow(SsaVarId),
    /// PF: the even parity of the result's low byte.
    Parity(SsaVarId),
    /// AF: bit 4 of `left ^ right ^ result`, the carry out of bit 3, `right`
    /// absent for a negation, whose left operand is zero.
    Adjust {
        /// The left operand, or the negated value.
        left: SsaVarId,
        /// The right operand.
        right: Option<SsaVarId>,
        /// The result.
        result: SsaVarId,
    },
    /// A flag the operation sets to a value whatever its operands.
    Known(bool),
    /// `mask` read from whichever of two flags values a select chooses: the
    /// select of the two reads, each of which folds on its own.
    Chosen {
        /// The select's condition.
        condition: SsaVarId,
        /// The flags it chooses where the condition holds.
        true_flags: SsaVarId,
        /// The flags it chooses otherwise.
        false_flags: SsaVarId,
        /// The bits read.
        mask: FlagsMask,
    },
}

impl ResultFlag {
    /// The read of `mask` from `flags` as the expression it folds to, seen
    /// through the operations that composed the flags ([`route`]): the
    /// expression of the operation that defines them, or the select of the
    /// reads of two flags values a select chooses, which fold in turn, where
    /// an expression states the read of either. `None` where none does: a
    /// multiplication's carry, chosen or not, or a choice between two.
    fn read<T: Target>(
        ir: &SsaFunction<T>,
        flags: SsaVarId,
        mask: FlagsMask,
        conventions: FlagConventions,
    ) -> Option<Self> {
        let states = |flags: SsaVarId, depth: usize| -> bool {
            let mut pending = vec![(flags, depth)];
            while let Some((flags, depth)) = pending.pop() {
                match route(ir, flags, mask) {
                    Some(FlagSource::Bit(_)) => {}
                    Some(FlagSource::Chosen {
                        true_flags,
                        false_flags,
                        ..
                    }) if depth < MOST_COMPOSITIONS_FOLLOWED => {
                        pending.push((true_flags, depth.saturating_add(1)));
                        pending.push((false_flags, depth.saturating_add(1)));
                    }
                    Some(FlagSource::Chosen { .. }) => return false,
                    Some(FlagSource::Flags(source)) => {
                        let stated = ir
                            .get_definition(source)
                            .and_then(|op| Self::of(op, source, mask, conventions));
                        if stated.is_none() {
                            return false;
                        }
                    }
                    None => {
                        let stated = ir
                            .get_definition(flags)
                            .and_then(|op| Self::of(op, flags, mask, conventions));
                        if stated.is_none() {
                            return false;
                        }
                    }
                }
            }
            true
        };
        match route(ir, flags, mask) {
            Some(FlagSource::Bit(_)) => None,
            Some(FlagSource::Chosen {
                condition,
                true_flags,
                false_flags,
            }) => (states(true_flags, 1) || states(false_flags, 1)).then_some(Self::Chosen {
                condition,
                true_flags,
                false_flags,
                mask,
            }),
            Some(FlagSource::Flags(source)) => ir
                .get_definition(source)
                .and_then(|op| Self::of(op, source, mask, conventions)),
            None => ir
                .get_definition(flags)
                .and_then(|op| Self::of(op, flags, mask, conventions)),
        }
    }

    /// The flag `mask` of the flags value `flags` that `op` defines, as the
    /// expression of its operands and result it states, where it states one.
    fn of<T: Target>(
        op: &SsaOp<T>,
        flags: SsaVarId,
        mask: FlagsMask,
        conventions: FlagConventions,
    ) -> Option<Self> {
        let logical = matches!(
            op,
            SsaOp::And { flags: Some(stated), .. }
                | SsaOp::Or { flags: Some(stated), .. }
                | SsaOp::Xor { flags: Some(stated), .. }
                if *stated == flags
        );
        let subtraction = conventions.subtraction;
        if mask == FlagsMask::ZERO {
            return result_stating_flags(op, flags).map(Self::Zero);
        }
        if mask == FlagsMask::SIGN {
            return result_stating_flags(op, flags).map(Self::Sign);
        }
        if mask == FlagsMask::PARITY {
            return result_stating_flags(op, flags)
                .filter(|_| conventions.parity_and_adjust && !matches!(op, SsaOp::Mul { .. }))
                .map(Self::Parity);
        }
        if (mask == FlagsMask::CARRY || mask == FlagsMask::OVERFLOW) && logical {
            return conventions
                .logic_clears_carry_and_overflow
                .then_some(Self::Known(false));
        }
        match op {
            SsaOp::Add {
                dest,
                left,
                right,
                flags: Some(stated),
            } if *stated == flags => match mask {
                FlagsMask::CARRY => Some(Self::Carry {
                    sum: *dest,
                    left: *left,
                }),
                FlagsMask::OVERFLOW => Some(Self::Overflow {
                    left: *left,
                    right: *right,
                    result: *dest,
                    subtraction: false,
                }),
                FlagsMask::ADJUST if conventions.parity_and_adjust => Some(Self::Adjust {
                    left: *left,
                    right: Some(*right),
                    result: *dest,
                }),
                _ => None,
            },
            SsaOp::Sub {
                dest,
                left,
                right,
                flags: Some(stated),
            } if *stated == flags => match mask {
                FlagsMask::CARRY if subtraction != SubtractionCarry::Unstated => {
                    Some(Self::Borrow {
                        left: *left,
                        right: *right,
                        set_when_borrowing: subtraction == SubtractionCarry::Borrow,
                    })
                }
                FlagsMask::OVERFLOW => Some(Self::Overflow {
                    left: *left,
                    right: *right,
                    result: *dest,
                    subtraction: true,
                }),
                FlagsMask::ADJUST if conventions.parity_and_adjust => Some(Self::Adjust {
                    left: *left,
                    right: Some(*right),
                    result: *dest,
                }),
                _ => None,
            },
            SsaOp::Neg {
                dest,
                operand,
                flags: Some(stated),
            } if *stated == flags => match mask {
                FlagsMask::CARRY if subtraction != SubtractionCarry::Unstated => {
                    Some(Self::NegationBorrow {
                        operand: *operand,
                        set_when_borrowing: subtraction == SubtractionCarry::Borrow,
                    })
                }
                FlagsMask::OVERFLOW => Some(Self::NegationOverflow(*operand)),
                FlagsMask::ADJUST if conventions.parity_and_adjust => Some(Self::Adjust {
                    left: *operand,
                    right: None,
                    result: *dest,
                }),
                _ => None,
            },
            _ => None,
        }
    }
}

/// The result an operation states its zero and sign flags from, when
/// `flags` is its flags output and it states them that way.
fn result_stating_flags<T: Target>(op: &SsaOp<T>, flags: SsaVarId) -> Option<SsaVarId> {
    let (dest, stated) = match op {
        SsaOp::Add { dest, flags, .. }
        | SsaOp::Sub { dest, flags, .. }
        | SsaOp::Mul { dest, flags, .. }
        | SsaOp::And { dest, flags, .. }
        | SsaOp::Or { dest, flags, .. }
        | SsaOp::Xor { dest, flags, .. }
        | SsaOp::Shl { dest, flags, .. }
        | SsaOp::Shr { dest, flags, .. }
        | SsaOp::Neg { dest, flags, .. }
        | SsaOp::Not { dest, flags, .. } => (*dest, *flags),
        _ => return None,
    };
    (stated == Some(flags)).then_some(dest)
}

/// The relation each edge into `flags` states, one entry per edge.
///
/// A flags value with a recorded relation answers directly, with no edge to
/// name. A flags value that is a phi answers for each of its operands,
/// resolved the same way, within [`MAX_JOIN_DEPTH`]. `None` where any operand
/// states nothing, or states a guarded relation.
fn relations_of<T: Target>(
    ir: &SsaFunction<T>,
    relation: &impl Fn(SsaVarId) -> Option<CompareRelation>,
    flags: SsaVarId,
    depth: usize,
) -> Option<Vec<(Option<usize>, CompareRelation)>> {
    if let Some(recorded) = relation(flags) {
        return (!recorded.conditional).then(|| vec![(None, recorded)]);
    }
    if depth >= MAX_JOIN_DEPTH {
        return None;
    }
    let phi = ir
        .iter_blocks()
        .flat_map(|(_, block)| block.phi_nodes())
        .find(|phi| phi.result() == flags)?;

    let mut resolved = Vec::new();
    for operand in phi.operands() {
        let nested = relations_of(ir, relation, operand.value(), depth.saturating_add(1))?;
        // A nested join contributes its own edges only where this one has none
        // to give it: an operand that is itself a phi is resolved in *its*
        // block, and the value it yields arrives here on this edge.
        let [(_, nested)] = nested.as_slice() else {
            return None;
        };
        resolved.push((Some(operand.predecessor()), *nested));
    }
    (!resolved.is_empty()).then_some(resolved)
}

/// How each side of the comparison is materialised, given the per-edge
/// relations.
fn sides_of(relations: &[(Option<usize>, CompareRelation)]) -> Option<(Side, Side)> {
    if let [(None, relation)] = relations {
        return Some((Side::Same(relation.left), Side::Same(relation.right)));
    }
    let mut lefts = Vec::new();
    let mut rights = Vec::new();
    for (predecessor, relation) in relations {
        let predecessor = (*predecessor)?;
        lefts.push((predecessor, relation.left));
        rights.push((predecessor, relation.right));
    }
    Some((side(lefts), side(rights)))
}

/// One side, collapsed to the value itself when every edge brings it.
fn side(edges: Vec<(usize, SsaVarId)>) -> Side {
    let first = edges.first().map(|(_, value)| *value);
    match first {
        Some(value) if edges.iter().all(|(_, other)| *other == value) => Side::Same(value),
        _ => Side::Join(edges),
    }
}

/// The single flags value `block` reads, or `None` when it reads none or more
/// than one.
fn sole_flags_read<T: Target>(block: &SsaBlock<T>) -> Option<SsaVarId> {
    let mut found: Option<SsaVarId> = None;
    for instruction in block.instructions() {
        let SsaOp::ReadFlags { flags, .. } = instruction.op() else {
            continue;
        };
        match found {
            Some(seen) if seen != *flags => return None,
            Some(_) => {}
            None => found = Some(*flags),
        }
    }
    found
}

/// Gives one side of the comparison a value in `block_idx`, adding a phi when
/// the edges disagree.
fn materialise<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    block_idx: usize,
    side: Side,
) -> Result<SsaVarId> {
    let edges = match side {
        Side::Same(value) => return Ok(value),
        Side::Join(edges) => edges,
    };
    // The comparison's own operand type, taken from an edge rather than
    // invented: the phi stands for the same value each edge already carries.
    let var_type = edges
        .first()
        .and_then(|(_, value)| editor.function().variable(*value))
        .map_or_else(T::unknown_type, |variable| variable.var_type().clone());
    let result = editor.create_variable(VariableOrigin::Phi, 0, DefSite::phi(block_idx), var_type);
    let mut phi = PhiNode::new(result, VariableOrigin::Phi);
    for (predecessor, value) in edges {
        phi.add_operand(PhiOperand::new(value, predecessor));
    }
    editor.add_phi(block_idx, phi)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::testing::{MACHINE, MockTarget, MockType};

    /// A function with one block holding `ops`, over `count` variables.
    fn function(count: usize, ops: Vec<SsaOp<MockTarget>>) -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        for _ in 0..count {
            ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(0), MockType::Unknown);
        }
        let mut block = SsaBlock::new(0);
        for op in ops {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();
        ssa
    }

    fn v(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    /// **A read follows the input its bits came from.** Zero was merged from
    /// `v1`, so a zero read reads `v1`; carry was not, so a carry read reads
    /// `v0`; a read of both spans two inputs and stays where it is.
    #[test]
    fn a_read_follows_the_input_its_bits_came_from() {
        let ssa = function(
            3,
            vec![SsaOp::MergeFlags {
                dest: v(2),
                base: v(0),
                update: v(1),
                mask: FlagsMask::ZERO,
            }],
        );
        assert_eq!(
            route(&ssa, v(2), FlagsMask::ZERO),
            Some(FlagSource::Flags(v(1)))
        );
        assert_eq!(
            route(&ssa, v(2), FlagsMask::CARRY),
            Some(FlagSource::Flags(v(0)))
        );
        assert_eq!(
            route(&ssa, v(2), FlagsMask::ZERO.union(FlagsMask::CARRY)),
            None
        );
    }

    /// **An inserted bit is its boolean, and the rest pass through.** Through
    /// two steps: the carry read is the boolean `v3`, and a zero read goes past
    /// the insert and the merge to `v1`.
    #[test]
    fn an_inserted_bit_is_its_boolean() {
        let ssa = function(
            5,
            vec![
                SsaOp::MergeFlags {
                    dest: v(2),
                    base: v(0),
                    update: v(1),
                    mask: FlagsMask::ZERO,
                },
                SsaOp::InsertFlag {
                    dest: v(4),
                    base: v(2),
                    bit: v(3),
                    mask: FlagsMask::CARRY,
                },
            ],
        );
        assert_eq!(
            route(&ssa, v(4), FlagsMask::CARRY),
            Some(FlagSource::Bit(v(3)))
        );
        assert_eq!(
            route(&ssa, v(4), FlagsMask::ZERO),
            Some(FlagSource::Flags(v(1)))
        );
    }

    /// The control: flags no composition defined are already their source.
    #[test]
    fn uncomposed_flags_resolve_to_nothing() {
        let ssa = function(2, vec![]);
        assert_eq!(route(&ssa, v(1), FlagsMask::ZERO), None);
    }

    /// The constant the read of `mask` folds to, after `fold_result_flag_reads`
    /// with `conventions`, of the flags `operation` defines from the 32-bit
    /// constants `left` and `right`, each instruction evaluated by
    /// [`Machine::evaluate`](crate::analysis::Machine::evaluate), as a boolean; `None`
    /// where the read stays a read.
    fn folded_flag(
        operation: &str,
        (left, right): (i32, i32),
        mask: FlagsMask,
        conventions: FlagConventions,
    ) -> Option<bool> {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let x = defined_at(&mut ir, 0, 0);
        let y = defined_at(&mut ir, 0, 1);
        let result = defined_at(&mut ir, 0, 2);
        let flags = defined_at(&mut ir, 0, 2);
        let read = defined_at(&mut ir, 0, 3);
        let op = match operation {
            "add" => SsaOp::Add {
                dest: result,
                left: x,
                right: y,
                flags: Some(flags),
            },
            "sub" => SsaOp::Sub {
                dest: result,
                left: x,
                right: y,
                flags: Some(flags),
            },
            "and" => SsaOp::And {
                dest: result,
                left: x,
                right: y,
                flags: Some(flags),
            },
            "xor" => SsaOp::Xor {
                dest: result,
                left: x,
                right: y,
                flags: Some(flags),
            },
            "mul" => SsaOp::Mul {
                dest: result,
                left: x,
                right: y,
                flags: Some(flags),
            },
            _ => SsaOp::Neg {
                dest: result,
                operand: x,
                flags: Some(flags),
            },
        };
        let mut ops = vec![
            SsaOp::Const {
                dest: x,
                value: ConstValue::I32(left),
            },
            SsaOp::Const {
                dest: y,
                value: ConstValue::I32(right),
            },
            op,
        ];
        ops.push(SsaOp::ReadFlags {
            dest: read,
            flags,
            mask,
        });
        ops.push(SsaOp::returns(Some(read)));
        add_block(&mut ir, 0, ops);
        ir.recompute_uses();
        fold_result_flag_reads(&mut ir, conventions);
        let mut known: HashMap<SsaVarId, ConstValue<MockTarget>> = HashMap::new();
        for instruction in ir.block(0)?.instructions() {
            let op = instruction.op();
            if let (Some(dest), Some(value)) = (
                op.dest(),
                MACHINE.evaluate(op, None, |var| known.get(&var).cloned()),
            ) {
                known.insert(dest, value);
            }
        }
        // The edit renumbers what it leaves unused, so the read is what the
        // block returns.
        let returned =
            ir.block(0)?
                .instructions()
                .iter()
                .find_map(|instruction| match instruction.op() {
                    SsaOp::Return { value, .. } => *value,
                    _ => None,
                })?;
        // A comparison folds to a boolean, or to an integer of 0 or 1.
        match known.get(&returned)? {
            ConstValue::True => Some(true),
            ConstValue::False => Some(false),
            ConstValue::I32(value) => Some(*value != 0),
            _ => None,
        }
    }

    /// **Every arithmetic flag of a constant operation folds to its value.**
    /// The x86 conventions: the zero, sign, carry, overflow, parity and adjust
    /// flags of additions, subtractions and a negation, the carry and
    /// overflow a logical operation clears, against the values Intel's manual
    /// states for each pair: `0x7fffffff + 1` overflows into the sign with
    /// even parity and a carry out of bit 3, `0xffffffff + 1` carries out and
    /// is zero, `3 - 5` borrows, `0x80000000 - 1` overflows, `neg 0x80000000`
    /// overflows and borrows, `0xf0 & 0x0f` is zero.
    #[test]
    fn every_arithmetic_flag_of_a_constant_operation_folds() {
        let x86 = FlagConventions::X86;
        let t = Some(true);
        let f = Some(false);
        for (operation, operands, mask, expected) in [
            ("add", (0x7fff_ffff, 1), FlagsMask::CARRY, &f),
            ("add", (0x7fff_ffff, 1), FlagsMask::OVERFLOW, &t),
            ("add", (0x7fff_ffff, 1), FlagsMask::SIGN, &t),
            ("add", (0x7fff_ffff, 1), FlagsMask::ZERO, &f),
            ("add", (0x7fff_ffff, 1), FlagsMask::PARITY, &t),
            ("add", (0x7fff_ffff, 1), FlagsMask::ADJUST, &t),
            ("add", (-1, 1), FlagsMask::CARRY, &t),
            ("add", (-1, 1), FlagsMask::OVERFLOW, &f),
            ("add", (-1, 1), FlagsMask::ZERO, &t),
            ("add", (i32::MIN, i32::MIN), FlagsMask::CARRY, &t),
            ("add", (i32::MIN, i32::MIN), FlagsMask::OVERFLOW, &t),
            ("add", (i32::MIN, i32::MIN), FlagsMask::ADJUST, &f),
            ("add", (5, 3), FlagsMask::PARITY, &f),
            ("add", (5, 3), FlagsMask::ADJUST, &f),
            ("add", (0xf, 1), FlagsMask::ADJUST, &t),
            ("add", (0xf, 1), FlagsMask::PARITY, &f),
            ("sub", (3, 5), FlagsMask::CARRY, &t),
            ("sub", (3, 5), FlagsMask::OVERFLOW, &f),
            ("sub", (3, 5), FlagsMask::SIGN, &t),
            ("sub", (3, 5), FlagsMask::PARITY, &f),
            ("sub", (3, 5), FlagsMask::ADJUST, &t),
            ("sub", (i32::MIN, 1), FlagsMask::OVERFLOW, &t),
            ("sub", (i32::MIN, 1), FlagsMask::CARRY, &f),
            ("sub", (5, 5), FlagsMask::ZERO, &t),
            ("sub", (5, 5), FlagsMask::PARITY, &t),
            ("neg", (0, 0), FlagsMask::CARRY, &f),
            ("neg", (0, 0), FlagsMask::ZERO, &t),
            ("neg", (5, 0), FlagsMask::CARRY, &t),
            ("neg", (5, 0), FlagsMask::OVERFLOW, &f),
            ("neg", (i32::MIN, 0), FlagsMask::OVERFLOW, &t),
            ("neg", (i32::MIN, 0), FlagsMask::CARRY, &t),
            ("and", (0xf0, 0x0f), FlagsMask::ZERO, &t),
            ("and", (0xf0, 0x0f), FlagsMask::CARRY, &f),
            ("and", (0xf0, 0x0f), FlagsMask::OVERFLOW, &f),
            ("and", (0xf0, 0x0f), FlagsMask::PARITY, &t),
            ("xor", (0x7f, 0), FlagsMask::PARITY, &f),
            ("xor", (-1, 0), FlagsMask::SIGN, &t),
        ] {
            assert_eq!(
                &folded_flag(operation, operands, mask, x86),
                expected,
                "{operation} {operands:?} {mask:?}"
            );
        }
    }

    /// **A flag an architecture defines its own way folds only under its
    /// conventions.** ARM's subtraction carries where it does not borrow (`3 -
    /// 5` clears it, `5 - 3` sets it), and its logical operations keep the
    /// shifter's carry and their overflow, so those reads stay; no
    /// architecture's convention states the parity of ARM's flags, nor any
    /// flag of a multiplication but its zero and sign, which stay reads too.
    #[test]
    fn an_architecture_specific_flag_folds_only_under_its_conventions() {
        let arm = FlagConventions::ARM;
        let x86 = FlagConventions::X86;
        assert_eq!(
            folded_flag("sub", (3, 5), FlagsMask::CARRY, arm),
            Some(false)
        );
        assert_eq!(
            folded_flag("sub", (5, 3), FlagsMask::CARRY, arm),
            Some(true)
        );
        assert_eq!(
            folded_flag("neg", (5, 0), FlagsMask::CARRY, arm),
            Some(false)
        );
        assert_eq!(folded_flag("and", (1, 1), FlagsMask::CARRY, arm), None);
        assert_eq!(folded_flag("and", (1, 1), FlagsMask::OVERFLOW, arm), None);
        assert_eq!(folded_flag("add", (1, 1), FlagsMask::PARITY, arm), None);
        assert_eq!(
            folded_flag("add", (1, 1), FlagsMask::OVERFLOW, arm),
            Some(false)
        );
        assert_eq!(
            folded_flag("sub", (3, 5), FlagsMask::CARRY, FlagConventions::UNSTATED),
            None
        );
        assert_eq!(folded_flag("mul", (2, 3), FlagsMask::CARRY, x86), None);
        assert_eq!(folded_flag("mul", (2, 3), FlagsMask::OVERFLOW, x86), None);
        assert_eq!(folded_flag("mul", (2, 3), FlagsMask::PARITY, x86), None);
        assert_eq!(folded_flag("mul", (0, 3), FlagsMask::ZERO, x86), Some(true));
    }

    /// A value the function was entered with, defined by no instruction.
    fn live_in(ir: &mut SsaFunction<MockTarget>) -> SsaVarId {
        ir.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I32,
        )
    }

    /// A value defined by instruction `index` of block `block`.
    fn defined_at(ir: &mut SsaFunction<MockTarget>, block: usize, index: usize) -> SsaVarId {
        ir.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(block, index),
            MockType::I32,
        )
    }

    /// Adds `ops` as block `index`.
    fn add_block(ir: &mut SsaFunction<MockTarget>, index: usize, ops: Vec<SsaOp<MockTarget>>) {
        let mut block = SsaBlock::new(index);
        for op in ops {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        ir.add_block(block);
    }

    /// The `Greater` expansion the lift emits when no single flag bit carries
    /// the condition, over `flags` and `second` (the flags its second read
    /// reads), branching to blocks 1 and 2.
    fn expansion(
        ir: &mut SsaFunction<MockTarget>,
        block: usize,
        start: usize,
        flags: SsaVarId,
        second: SsaVarId,
    ) -> Vec<SsaOp<MockTarget>> {
        let sign = defined_at(ir, block, start);
        let overflow = defined_at(ir, block, start + 1);
        let condition = defined_at(ir, block, start + 2);
        vec![
            SsaOp::ReadFlags {
                dest: sign,
                flags,
                mask: FlagsMask::SIGN,
            },
            SsaOp::ReadFlags {
                dest: overflow,
                flags: second,
                mask: FlagsMask::OVERFLOW,
            },
            SsaOp::Ceq {
                dest: condition,
                left: sign,
                right: overflow,
                signaling: false,
            },
            SsaOp::Branch {
                condition,
                true_target: 1,
                false_target: 2,
            },
        ]
    }

    /// `cmp a, b` followed by the expansion of `a > b`, branching on it.
    ///
    /// `split_flags` gives the second read a *different* flags value, the
    /// ambiguous shape the fold must decline.
    fn expanded_branch(
        split_flags: bool,
    ) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId, SsaVarId) {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let left = live_in(&mut ir);
        let right = live_in(&mut ir);
        let other_flags = live_in(&mut ir);
        // `Sub` defines both its result and its flags output.
        let scratch = defined_at(&mut ir, 0, 0);
        let flags = defined_at(&mut ir, 0, 0);
        let second = if split_flags { other_flags } else { flags };
        let mut entry = vec![SsaOp::Sub {
            dest: scratch,
            left,
            right,
            flags: Some(flags),
        }];
        entry.extend(expansion(&mut ir, 0, 1, flags, second));
        add_block(&mut ir, 0, entry);
        for index in 1..3 {
            add_block(&mut ir, index, vec![SsaOp::returns(None)]);
        }
        (ir, left, right, flags)
    }

    /// The branch of block 0, expanded from `a > b`.
    fn greater_at_entry() -> [ExpandedCondition; 1] {
        [ExpandedCondition {
            block_idx: 0,
            cmp: CmpKind::Gt,
            unsigned: false,
            float: Some((CmpKind::Gt, false)),
        }]
    }

    /// The relation recorded for `flags`, and none for anything else.
    fn recorded(
        flags: SsaVarId,
        left: SsaVarId,
        right: SsaVarId,
        conditional: bool,
    ) -> impl Fn(SsaVarId) -> Option<CompareRelation> {
        move |asked| {
            (asked == flags).then_some(CompareRelation {
                left,
                right,
                conditional,
                float: false,
            })
        }
    }

    /// The condition the lift could not fold across a block boundary becomes
    /// the comparison the program makes.
    #[test]
    fn a_recorded_comparison_folds_into_a_branch_cmp() {
        let (mut ir, left, right, flags) = expanded_branch(false);
        assert!(fold_flag_conditions(
            &mut ir,
            &greater_at_entry(),
            recorded(flags, left, right, false)
        ));
        let terminator = ir.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(
                terminator,
                Some(SsaOp::BranchCmp {
                    left: l,
                    right: r,
                    cmp: CmpKind::Gt,
                    unsigned: false,
                    true_target: 1,
                    false_target: 2,
                }) if *l == left && *r == right
            ),
            "expected a comparison branch, got {terminator:?}"
        );
    }

    /// A float compare's flags fold by the condition's float reading: the
    /// condition that is a signed greater-than after an integer compare is an
    /// unordered `nle` after this float one, which the branch states, and a
    /// condition with no float reading leaves the flag reads as they are.
    #[test]
    fn a_float_compare_folds_by_its_float_reading() {
        let float = |flags, left, right| {
            move |asked| {
                (asked == flags).then_some(CompareRelation {
                    left,
                    right,
                    conditional: false,
                    float: true,
                })
            }
        };
        let (mut ir, left, right, flags) = expanded_branch(false);
        let unordered = [ExpandedCondition {
            block_idx: 0,
            cmp: CmpKind::Gt,
            unsigned: false,
            float: Some((CmpKind::Lt, true)),
        }];
        assert!(fold_flag_conditions(
            &mut ir,
            &unordered,
            float(flags, left, right)
        ));
        let terminator = ir.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(
                terminator,
                Some(SsaOp::BranchCmp {
                    cmp: CmpKind::Lt,
                    unsigned: true,
                    ..
                })
            ),
            "expected the float relation, got {terminator:?}"
        );
        let (mut ir, left, right, flags) = expanded_branch(false);
        let no_reading = [ExpandedCondition {
            float: None,
            ..greater_at_entry()[0]
        }];
        assert!(!fold_flag_conditions(
            &mut ir,
            &no_reading,
            float(flags, left, right)
        ));
    }

    /// A conditional compare established its relation on one path only, so
    /// the fold declines rather than state it unconditionally.
    #[test]
    fn a_guarded_relation_is_never_folded() {
        let (mut ir, left, right, flags) = expanded_branch(false);
        assert!(!fold_flag_conditions(
            &mut ir,
            &greater_at_entry(),
            recorded(flags, left, right, true)
        ));
        assert!(matches!(
            ir.block(0).and_then(SsaBlock::control_terminator),
            Some(SsaOp::Branch { .. })
        ));
    }

    /// Two flags values in one block leave it unsaid which produced the
    /// condition.
    #[test]
    fn two_flags_values_in_a_block_are_ambiguous() {
        let (mut ir, left, right, flags) = expanded_branch(true);
        assert!(!fold_flag_conditions(
            &mut ir,
            &greater_at_entry(),
            recorded(flags, left, right, false)
        ));
        assert!(matches!(
            ir.block(0).and_then(SsaBlock::control_terminator),
            Some(SsaOp::Branch { .. })
        ));
    }

    /// Most flag definitions are arithmetic and state no relation, so an
    /// unrecorded flags value folds to nothing.
    #[test]
    fn an_unrecorded_flags_value_does_not_fold() {
        let (mut ir, _, _, _) = expanded_branch(false);
        assert!(!fold_flag_conditions(&mut ir, &greater_at_entry(), |_| {
            None
        }));
        assert!(matches!(
            ir.block(0).and_then(SsaBlock::control_terminator),
            Some(SsaOp::Branch { .. })
        ));
    }

    /// Flags reaching a loop header through a phi fold into a comparison of
    /// phis over the edges: the left operands differ and join, the right one
    /// is the same on both edges and is used as it is.
    ///
    /// `B0: cmp a, k; jump B3. B4: cmp b, k; jump B3. B3: f = phi(B0: f0,
    /// B4: f1); a > k?` with blocks 1 and 2 the branch's targets.
    #[test]
    fn flags_joined_at_a_phi_fold_into_a_comparison_of_phis() {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let a = live_in(&mut ir);
        let b = live_in(&mut ir);
        let k = live_in(&mut ir);
        let scratch0 = defined_at(&mut ir, 0, 0);
        let f0 = defined_at(&mut ir, 0, 0);
        let scratch1 = defined_at(&mut ir, 4, 0);
        let f1 = defined_at(&mut ir, 4, 0);
        let f = ir.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
        add_block(
            &mut ir,
            0,
            vec![
                SsaOp::Sub {
                    dest: scratch0,
                    left: a,
                    right: k,
                    flags: Some(f0),
                },
                SsaOp::Jump { target: 3 },
            ],
        );
        add_block(&mut ir, 1, vec![SsaOp::returns(None)]);
        add_block(&mut ir, 2, vec![SsaOp::Jump { target: 4 }]);
        let header = expansion(&mut ir, 3, 0, f, f);
        add_block(&mut ir, 3, header);
        add_block(
            &mut ir,
            4,
            vec![
                SsaOp::Sub {
                    dest: scratch1,
                    left: b,
                    right: k,
                    flags: Some(f1),
                },
                SsaOp::Jump { target: 3 },
            ],
        );
        let mut phi = PhiNode::new(f, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(f0, 0));
        phi.add_operand(PhiOperand::new(f1, 4));
        ir.block_mut(3).unwrap().add_phi(phi);
        ir.recompute_uses();

        let relation = move |flags: SsaVarId| {
            let left = if flags == f0 {
                a
            } else if flags == f1 {
                b
            } else {
                return None;
            };
            Some(CompareRelation {
                left,
                right: k,
                conditional: false,
                float: false,
            })
        };
        let conditions = [ExpandedCondition {
            block_idx: 3,
            cmp: CmpKind::Gt,
            unsigned: false,
            float: None,
        }];
        assert!(fold_flag_conditions(&mut ir, &conditions, relation));
        let header = ir.block(3).unwrap();
        let Some(SsaOp::BranchCmp { left, right, .. }) = header.control_terminator() else {
            panic!("expected a comparison branch in the header");
        };
        assert_eq!(*right, k, "the right side is the same on every edge");
        let joined = header
            .phi_nodes()
            .iter()
            .find(|phi| phi.result() == *left)
            .expect("the left side is a phi in the header");
        let mut operands: Vec<(usize, SsaVarId)> = joined
            .operands()
            .iter()
            .map(|operand| (operand.predecessor(), operand.value()))
            .collect();
        operands.sort_unstable();
        assert_eq!(operands, [(0, a), (4, b)]);
    }

    /// **The carry of an addition is its unsigned overflow, and of a
    /// subtraction the borrow or its complement.** `lo = x + y` carries
    /// exactly when `lo < x`, unsigned, so the carry read a 32-bit target's
    /// 64-bit addition makes - `hi = a + b + carry` - is that comparison. A
    /// subtraction's carry is `x < y` where it is set on a borrow (x86) and
    /// `x >= y` where it is set without one (ARM). The control states no
    /// convention, and the subtraction's carry stays a read.
    #[test]
    fn the_carry_of_an_addition_is_its_unsigned_overflow() {
        let folded = |subtraction: SubtractionCarry| {
            let mut ir = SsaFunction::<MockTarget>::new(0, 0);
            let x = live_in(&mut ir);
            let y = live_in(&mut ir);
            let sum = defined_at(&mut ir, 0, 0);
            let flags = defined_at(&mut ir, 0, 0);
            let carry = defined_at(&mut ir, 0, 1);
            let difference = defined_at(&mut ir, 0, 2);
            let borrow_flags = defined_at(&mut ir, 0, 2);
            let borrow = defined_at(&mut ir, 0, 3);
            add_block(
                &mut ir,
                0,
                vec![
                    SsaOp::Add {
                        dest: sum,
                        left: x,
                        right: y,
                        flags: Some(flags),
                    },
                    SsaOp::ReadFlags {
                        dest: carry,
                        flags,
                        mask: FlagsMask::CARRY,
                    },
                    SsaOp::Sub {
                        dest: difference,
                        left: x,
                        right: y,
                        flags: Some(borrow_flags),
                    },
                    SsaOp::ReadFlags {
                        dest: borrow,
                        flags: borrow_flags,
                        mask: FlagsMask::CARRY,
                    },
                    SsaOp::returns(Some(carry)),
                ],
            );
            ir.recompute_uses();
            assert!(fold_result_flag_reads(
                &mut ir,
                FlagConventions {
                    subtraction,
                    ..FlagConventions::UNSTATED
                }
            ));
            let ops: Vec<SsaOp<MockTarget>> = ir
                .block(0)
                .unwrap()
                .instructions()
                .iter()
                .map(|instruction| instruction.op().clone())
                .collect();
            (ops, x, y, sum, carry, borrow)
        };

        let (ops, x, y, sum, carry, borrow) = folded(SubtractionCarry::Unstated);
        assert!(ops.iter().any(|op| matches!(
            op,
            SsaOp::Clt { dest, left, right, unsigned: true, .. }
                if *dest == carry && *left == sum && *right == x
        )));
        assert!(
            ops.iter()
                .any(|op| matches!(op, SsaOp::ReadFlags { dest, .. } if *dest == borrow)),
            "the control: a subtraction's carry stays a read"
        );

        let (ops, _, _, _, _, _) = folded(SubtractionCarry::Borrow);
        assert!(ops.iter().any(|op| matches!(
            op,
            SsaOp::Clt { dest, left, right, unsigned: true, .. }
                if *dest == borrow && *left == x && *right == y
        )));

        // A carry set where nothing is borrowed is `x >= y`: the borrow's
        // comparison, negated.
        let (ops, _, _, _, _, _) = folded(SubtractionCarry::NoBorrow);
        let below = ops.iter().find_map(|op| match op {
            SsaOp::Clt {
                dest,
                left,
                right,
                unsigned: true,
                ..
            } if *left == x && *right == y => Some(*dest),
            _ => None,
        });
        assert!(
            ops.iter().any(|op| matches!(
                op,
                SsaOp::BoolNot { dest, value } if *dest == borrow && Some(*value) == below
            )),
            "{ops:?}"
        );
        assert!(!ops.iter().any(|op| matches!(op, SsaOp::ReadFlags { .. })));
    }

    /// A read of the zero flag of an operation that sets it from its result
    /// is that result's comparison with zero, and a read of the sign flag is
    /// its comparison below zero.
    #[test]
    fn a_result_flag_read_is_a_comparison_with_zero() {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let x = live_in(&mut ir);
        let y = live_in(&mut ir);
        let sum = defined_at(&mut ir, 0, 0);
        let flags = defined_at(&mut ir, 0, 0);
        let zero = defined_at(&mut ir, 0, 1);
        let negative = defined_at(&mut ir, 0, 2);
        add_block(
            &mut ir,
            0,
            vec![
                SsaOp::Add {
                    dest: sum,
                    left: x,
                    right: y,
                    flags: Some(flags),
                },
                SsaOp::ReadFlags {
                    dest: zero,
                    flags,
                    mask: FlagsMask::ZERO,
                },
                SsaOp::ReadFlags {
                    dest: negative,
                    flags,
                    mask: FlagsMask::SIGN,
                },
                SsaOp::returns(Some(zero)),
            ],
        );
        ir.recompute_uses();

        assert!(fold_result_flag_reads(&mut ir, FlagConventions::UNSTATED));
        let ops: Vec<&SsaOp<MockTarget>> = ir
            .block(0)
            .unwrap()
            .instructions()
            .iter()
            .map(SsaInstruction::op)
            .collect();
        assert!(ops.iter().any(|op| matches!(
            op,
            SsaOp::Ceq { dest, left, .. } if *dest == zero && *left == sum
        )));
        assert!(ops.iter().any(|op| matches!(
            op,
            SsaOp::Clt { dest, left, unsigned: false, .. } if *dest == negative && *left == sum
        )));
        assert!(
            !ops.iter().any(|op| matches!(op, SsaOp::ReadFlags { .. })),
            "{ops:?}"
        );
    }

    /// **A read of flags a select chooses is the select of the reads.** A
    /// shift by a register keeps the flags it inherited where the count is
    /// zero and states its own otherwise: the zero flag of `c ? add : sub` is
    /// `c ? sum == 0 : difference == 0`, each read folded as it would be on
    /// its own, and no flags read is left.
    #[test]
    fn a_flag_read_of_a_choice_is_a_choice_of_reads() {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let (x, y, condition) = (live_in(&mut ir), live_in(&mut ir), live_in(&mut ir));
        let sum = defined_at(&mut ir, 0, 0);
        let added = defined_at(&mut ir, 0, 0);
        let difference = defined_at(&mut ir, 0, 1);
        let subtracted = defined_at(&mut ir, 0, 1);
        let chosen = defined_at(&mut ir, 0, 2);
        let zero = defined_at(&mut ir, 0, 3);
        add_block(
            &mut ir,
            0,
            vec![
                SsaOp::Add {
                    dest: sum,
                    left: x,
                    right: y,
                    flags: Some(added),
                },
                SsaOp::Sub {
                    dest: difference,
                    left: x,
                    right: y,
                    flags: Some(subtracted),
                },
                SsaOp::Select {
                    dest: chosen,
                    condition,
                    true_val: added,
                    false_val: subtracted,
                },
                SsaOp::ReadFlags {
                    dest: zero,
                    flags: chosen,
                    mask: FlagsMask::ZERO,
                },
                SsaOp::returns(Some(zero)),
            ],
        );
        ir.recompute_uses();

        assert!(fold_result_flag_reads(&mut ir, FlagConventions::UNSTATED));
        let ops: Vec<&SsaOp<MockTarget>> = ir
            .block(0)
            .unwrap()
            .instructions()
            .iter()
            .map(SsaInstruction::op)
            .collect();
        let compares = |var: SsaVarId, result: SsaVarId| {
            ops.iter().any(
                |op| matches!(op, SsaOp::Ceq { dest, left, .. } if *dest == var && *left == result),
            )
        };
        assert!(
            ops.iter().any(|op| matches!(
                op,
                SsaOp::Select { dest, condition: c, true_val, false_val }
                    if *dest == zero && *c == condition
                        && compares(*true_val, sum) && compares(*false_val, difference)
            )),
            "{ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(op, SsaOp::ReadFlags { .. })),
            "{ops:?}"
        );
    }

    /// **A branch on flags a select chooses branches on the select of their
    /// values.** ARM's `cmp r0, #61; cmpne r0, #0; bne` keeps the first
    /// compare's flags where `r0` is 61 and takes the second's otherwise: the
    /// branch is on the select of `r0 - 61 == 0` and `r0 - 0 == 0`, its
    /// targets exchanged, and no flags are read or branched on once the reads
    /// fold. The control: a branch on flags a select chooses between a
    /// multiplication's carries, which no result states, stays.
    #[test]
    fn a_branch_on_chosen_flags_branches_on_the_chosen_values() {
        let chosen = |multiply: bool| {
            let mut ir = SsaFunction::<MockTarget>::new(0, 0);
            let x = live_in(&mut ir);
            let condition = live_in(&mut ir);
            let constant = defined_at(&mut ir, 0, 0);
            let first = defined_at(&mut ir, 0, 1);
            let first_flags = defined_at(&mut ir, 0, 1);
            let second = defined_at(&mut ir, 0, 2);
            let second_flags = defined_at(&mut ir, 0, 2);
            let flags = defined_at(&mut ir, 0, 3);
            let computed = |dest, flags| {
                if multiply {
                    SsaOp::Mul {
                        dest,
                        left: x,
                        right: constant,
                        flags: Some(flags),
                    }
                } else {
                    SsaOp::Sub {
                        dest,
                        left: x,
                        right: constant,
                        flags: Some(flags),
                    }
                }
            };
            add_block(
                &mut ir,
                0,
                vec![
                    SsaOp::Const {
                        dest: constant,
                        value: ConstValue::I32(61),
                    },
                    computed(first, first_flags),
                    computed(second, second_flags),
                    SsaOp::Select {
                        dest: flags,
                        condition,
                        true_val: second_flags,
                        false_val: first_flags,
                    },
                    SsaOp::BranchFlags {
                        flags,
                        condition: if multiply {
                            FlagCondition::Carry
                        } else {
                            FlagCondition::NotZero
                        },
                        true_target: 1,
                        false_target: 2,
                    },
                ],
            );
            add_block(&mut ir, 1, vec![SsaOp::returns(None)]);
            add_block(&mut ir, 2, vec![SsaOp::returns(None)]);
            ir.recompute_uses();
            ir
        };

        let mut ir = chosen(false);
        assert!(fold_result_flag_branches(&mut ir, FlagConventions::ARM));
        assert!(fold_result_flag_reads(&mut ir, FlagConventions::ARM));
        let ops: Vec<&SsaOp<MockTarget>> = ir
            .block(0)
            .into_iter()
            .flat_map(SsaBlock::instructions)
            .map(SsaInstruction::op)
            .collect();
        let selected =
            ops.iter().find_map(|op| {
                match op {
            SsaOp::Select { dest, true_val, false_val, .. }
                if ops.iter().any(|defining| {
                    matches!(defining, SsaOp::Ceq { dest: zero, .. } if zero == true_val)
                }) && ops.iter().any(|defining| {
                    matches!(defining, SsaOp::Ceq { dest: zero, .. } if zero == false_val)
                }) =>
            {
                Some(*dest)
            }
            _ => None,
        }
            });
        assert!(
            ops.iter().any(|op| matches!(
                op,
                SsaOp::Branch { condition, true_target: 2, false_target: 1 } if Some(*condition) == selected
            )),
            "{ops:?}"
        );
        assert!(
            !ops.iter()
                .any(|op| matches!(op, SsaOp::ReadFlags { .. } | SsaOp::BranchFlags { .. })),
            "{ops:?}"
        );

        let mut ir = chosen(true);
        assert!(!fold_result_flag_branches(&mut ir, FlagConventions::ARM));
    }

    /// **A branch on a flag set from a boolean branches on the boolean.**
    /// ARM's `sbcs` sets its carry from its borrow-in, and the lift inserts
    /// the carry it computes ([`SsaOp::InsertFlag`]): `bcs` is a branch on that
    /// boolean, and `bcc` on it with its targets exchanged.
    #[test]
    fn a_branch_on_an_inserted_flag_branches_on_its_boolean() {
        for (condition, holds, fails) in [
            (FlagCondition::Carry, 1, 2),
            (FlagCondition::NotCarry, 2, 1),
        ] {
            let mut ir = SsaFunction::<MockTarget>::new(0, 0);
            let x = live_in(&mut ir);
            let y = live_in(&mut ir);
            let carry = live_in(&mut ir);
            let difference = defined_at(&mut ir, 0, 0);
            let flags = defined_at(&mut ir, 0, 0);
            let inserted = defined_at(&mut ir, 0, 1);
            add_block(
                &mut ir,
                0,
                vec![
                    SsaOp::Sub {
                        dest: difference,
                        left: x,
                        right: y,
                        flags: Some(flags),
                    },
                    SsaOp::InsertFlag {
                        dest: inserted,
                        base: flags,
                        bit: carry,
                        mask: FlagsMask::CARRY,
                    },
                    SsaOp::BranchFlags {
                        flags: inserted,
                        condition,
                        true_target: 1,
                        false_target: 2,
                    },
                ],
            );
            add_block(&mut ir, 1, vec![SsaOp::returns(None)]);
            add_block(&mut ir, 2, vec![SsaOp::returns(None)]);
            ir.recompute_uses();

            assert!(fold_result_flag_branches(&mut ir, FlagConventions::ARM));
            assert!(
                matches!(
                    ir.block(0).and_then(SsaBlock::control_terminator),
                    Some(SsaOp::Branch { condition: bit, true_target, false_target })
                        if *bit == carry && *true_target == holds && *false_target == fails
                ),
                "{condition:?}: {:?}",
                ir.block(0).and_then(SsaBlock::control_terminator)
            );
        }
    }

    /// **A branch on a flag an operation states from its result branches on
    /// the result.** x86 `test eax, eax; jne` is a branch on the zero flag of
    /// `and eax, eax`, clear: it branches on `eax & eax == 0` with its targets
    /// exchanged, and nothing reads the `and`'s flags any more. A branch on the
    /// carry of a multiplication, which no result states, stays, and so does
    /// one on the sign of an untyped difference (x86 `cmp dword [ebp+8], 0;
    /// jns`), which no comparison with zero can state.
    #[test]
    fn a_branch_on_a_result_flag_branches_on_the_result() {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let x = live_in(&mut ir);
        let masked = defined_at(&mut ir, 0, 0);
        let flags = defined_at(&mut ir, 0, 0);
        add_block(
            &mut ir,
            0,
            vec![
                SsaOp::And {
                    dest: masked,
                    left: x,
                    right: x,
                    flags: Some(flags),
                },
                SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::NotZero,
                    true_target: 1,
                    false_target: 2,
                },
            ],
        );
        add_block(&mut ir, 1, vec![SsaOp::returns(None)]);
        add_block(&mut ir, 2, vec![SsaOp::returns(None)]);
        ir.recompute_uses();

        assert!(fold_result_flag_branches(&mut ir, FlagConventions::X86));
        let ops: Vec<&SsaOp<MockTarget>> = ir
            .block(0)
            .unwrap()
            .instructions()
            .iter()
            .map(SsaInstruction::op)
            .collect();
        let zero = ops.iter().find_map(|op| match op {
            SsaOp::Ceq { dest, left, .. } if *left == masked => Some(*dest),
            _ => None,
        });
        assert!(
            ops.iter().any(|op| matches!(
                op,
                SsaOp::Branch { condition, true_target: 2, false_target: 1 } if Some(*condition) == zero
            )),
            "{ops:?}"
        );
        assert!(
            !ops.iter()
                .any(|op| matches!(op, SsaOp::ReadFlags { .. } | SsaOp::BranchFlags { .. })),
            "{ops:?}"
        );

        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let x = live_in(&mut ir);
        let product = defined_at(&mut ir, 0, 0);
        let flags = defined_at(&mut ir, 0, 0);
        add_block(
            &mut ir,
            0,
            vec![
                SsaOp::Mul {
                    dest: product,
                    left: x,
                    right: x,
                    flags: Some(flags),
                },
                SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::Carry,
                    true_target: 1,
                    false_target: 2,
                },
            ],
        );
        add_block(&mut ir, 1, vec![SsaOp::returns(None)]);
        add_block(&mut ir, 2, vec![SsaOp::returns(None)]);
        ir.recompute_uses();
        assert!(!fold_result_flag_branches(&mut ir, FlagConventions::X86));

        // Nor does a branch on the sign of a result of no integer width,
        // which no comparison with zero can state: it is left as it was.
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let x = live_in(&mut ir);
        let zero = defined_at(&mut ir, 0, 0);
        let difference = ir.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 1),
            MockType::Unknown,
        );
        let flags = defined_at(&mut ir, 0, 1);
        add_block(
            &mut ir,
            0,
            vec![
                SsaOp::Const {
                    dest: zero,
                    value: ConstValue::I32(0),
                },
                SsaOp::Sub {
                    dest: difference,
                    left: x,
                    right: zero,
                    flags: Some(flags),
                },
                SsaOp::BranchFlags {
                    flags,
                    condition: FlagCondition::Positive,
                    true_target: 1,
                    false_target: 2,
                },
            ],
        );
        add_block(&mut ir, 1, vec![SsaOp::returns(None)]);
        add_block(&mut ir, 2, vec![SsaOp::returns(None)]);
        ir.recompute_uses();
        assert!(!fold_result_flag_branches(&mut ir, FlagConventions::X86));
        assert!(
            matches!(
                ir.block(0).and_then(|block| block.control_terminator()),
                Some(SsaOp::BranchFlags { .. })
            ),
            "{ir:?}"
        );
    }

    /// **Every branch a fold plans is stated over the values its plan names.**
    /// Two blocks branch on the zero flag of an `and`, and the block both
    /// reach opens with a phi nothing reads, which an edit session's repair
    /// removes, renumbering every value after it. Folded one session per branch,
    /// the second branch was stated over the ids the first session's repair
    /// had moved: gcc 8's `-O3` `x86_64` `grep` branched on the overflow of a
    /// value no block defined.
    #[test]
    fn every_planned_branch_fold_names_the_values_it_planned() {
        let mut ir = SsaFunction::<MockTarget>::new(0, 0);
        let unread = ir.create_variable(VariableOrigin::Phi, 0, DefSite::phi(2), MockType::I32);
        let x = live_in(&mut ir);
        let first = defined_at(&mut ir, 0, 0);
        let first_flags = defined_at(&mut ir, 0, 0);
        let second = defined_at(&mut ir, 1, 0);
        let second_flags = defined_at(&mut ir, 1, 0);
        add_block(
            &mut ir,
            0,
            vec![
                SsaOp::And {
                    dest: first,
                    left: x,
                    right: x,
                    flags: Some(first_flags),
                },
                SsaOp::BranchFlags {
                    flags: first_flags,
                    condition: FlagCondition::NotZero,
                    true_target: 1,
                    false_target: 2,
                },
            ],
        );
        add_block(
            &mut ir,
            1,
            vec![
                SsaOp::And {
                    dest: second,
                    left: first,
                    right: x,
                    flags: Some(second_flags),
                },
                SsaOp::BranchFlags {
                    flags: second_flags,
                    condition: FlagCondition::Zero,
                    true_target: 2,
                    false_target: 3,
                },
            ],
        );
        add_block(&mut ir, 2, vec![SsaOp::returns(None)]);
        add_block(&mut ir, 3, vec![SsaOp::returns(None)]);
        let mut phi = PhiNode::new(unread, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(first, 0));
        phi.add_operand(PhiOperand::new(second, 1));
        ir.block_mut(2).expect("the join").add_phi(phi);
        ir.recompute_uses();

        assert!(fold_result_flag_branches(&mut ir, FlagConventions::X86));
        assert_eq!(ir.validate(), Ok(()));
        for block_idx in [0, 1] {
            let block = ir.block(block_idx).expect("the block");
            let anded =
                block
                    .instructions()
                    .iter()
                    .find_map(|instruction| match instruction.op() {
                        SsaOp::And { dest, .. } => Some(*dest),
                        _ => None,
                    });
            let zero = block
                .instructions()
                .iter()
                .find_map(|instruction| match instruction.op() {
                    SsaOp::Ceq { dest, left, .. } if Some(*left) == anded => Some(*dest),
                    _ => None,
                });
            assert!(
                matches!(
                    block.control_terminator(),
                    Some(SsaOp::Branch { condition, .. }) if Some(*condition) == zero
                ),
                "block {block_idx}: {:?}",
                block.instructions()
            );
        }
    }
}
