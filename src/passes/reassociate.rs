//! Reassociation pass - reorders nested associative operations to enable
//! constant folding.
//!
//! # Algorithm
//!
//! Detects and rewrites nested binary operations where both the inner
//! and outer operations have a constant operand:
//!
//! - `(x + c1) + c2` → `x + (c1 + c2)` (and Sub/Mul/And/Or/Xor variants)
//! - `(x << c1) << c2` → `x << (c1 + c2)` (and the Shr variant)
//! - `(x * c1) << c2` → `x * (c1 << c2)` and `(x << c1) * c2` →
//!   `x * (c2 << c1)`: a product and a left shift both scale by a constant,
//!   and two scalings are one, so `(i * 3) << 3` is `i * 24`, the stride a
//!   reader of an array of 24-byte records looks for
//!
//! After rewriting, the two constant operands are combined into a single
//! constant value that a later constant-propagation pass can fold.
//!
//! # Details
//!
//! 1. Find all constant variables via [`SsaFunction::find_constants`].
//! 2. For each instruction, check if its defining op matches the outer
//!    pattern: one operand is a constant and the other is defined by the
//!    same op kind with a constant on the corresponding side.
//! 3. If the inner result has only one use, the rewrite is valid: the
//!    inner constant is replaced with the combined value, the inner op
//!    is rewritten as `x ∘ c1`, and the outer op is rewritten as a `Copy`
//!    of the inner result. A shift under a product folds into the
//!    product's constant instead, the outer one, since a shift's count is
//!    no factor: the outer op is rewritten as `x * c2`, and the shift, read
//!    by nothing, is left to dead-code elimination. A constant on the left of
//!    a commutative outer op is read as if on its right.
//! 4. The `combine` step uses `ptr_size` so wraparound on `NativeInt`
//!    constants matches the host's pointer width.
//!
//! # Supported Operations
//!
//! | Outer kind | Inner kind | Combine op | Example |
//! |-----------|-----------|-----------|---------|
//! | Add | Add | Add | `(x + 3) + 7 → x + 10` |
//! | Sub | Sub | Add | `(x - 3) - 2 → x - 5` |
//! | Mul | Mul | Mul | `(x * 3) * 5 → x * 15` |
//! | And | And | And | `(x & 0xF0) & 0x0F → x & 0` |
//! | Or | Or | Or | `(x \| 0xF0) \| 0x0F → x \| 0xFF` |
//! | Xor | Xor | Xor | `(x ^ 0xF0) ^ 0x0F → x ^ 0xFF` |
//! | Shl | Shl | Add | `(x << 2) << 3 → x << 5` |
//! | Shr | Shr | Add | `(x >> 2) >> 3 → x >> 5` |
//! | Shl | Mul | Shl | `(x * 3) << 3 → x * 24` |
//! | Mul | Shl | Shl | `(x << 1) * 3 → x * 6` |

use std::collections::BTreeMap;

use rustc_hash::FxHashSet;

use crate::{
    Result,
    analysis::DefUseIndex,
    events::EventKind,
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// A detected reassociation opportunity: `(base_var op const1) op const2`.
#[derive(Debug)]
struct ReassociationCandidate<T: Target> {
    /// Block containing the outer instruction.
    block_idx: usize,
    /// Index of the outer instruction within its block.
    instr_idx: usize,
    /// Destination variable of the outer operation's result.
    dest: SsaVarId,
    /// The non-constant base variable in the inner operation.
    base_var: SsaVarId,
    /// Variable holding the first constant `c1`.
    const1_var: SsaVarId,
    /// Variable holding the second constant `c2`.
    const2_var: SsaVarId,
    /// Value of the first constant.
    const1_value: ConstValue<T>,
    /// Value of the second constant.
    const2_value: ConstValue<T>,
    /// Block containing the inner instruction.
    inner_block: usize,
    /// Index of the inner instruction within its block.
    inner_instr: usize,
    /// Destination variable of the inner operation's result.
    inner_dest: SsaVarId,
    /// The operation kind the chain becomes: both ops' kind for a
    /// [`Chain::Same`], `Mul` for a product and a shift.
    op_kind: OpKind,
    /// How the two constants fold, and into which of them.
    chain: Chain,
}

/// The shape of a chain of two constant operations, which says how its
/// constants fold and which one holds the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chain {
    /// `(x op c1) op c2`, one kind twice: `c1` becomes the two folded by
    /// [`OpKind::combine`].
    Same,
    /// `(x * c1) << c2`, a product shifted: `c1` becomes `c1 << c2`.
    ProductShifted,
    /// `(x << c1) * c2`, a shift multiplied: `c2` becomes `c2 << c1`, the
    /// product's own constant, as a shift's count is no factor.
    ShiftMultiplied,
}

impl Chain {
    /// Folds a chain's constants, inner `c1` and outer `c2`, into the one
    /// constant the rewritten chain reads, evaluated at `ptr_size`.
    ///
    /// `None` where the fold is undefined: mismatched kinds, or a shift by a
    /// count outside the shifted constant's width, which
    /// [`ConstValue::shl`] declines.
    fn fold<T: Target>(
        self,
        kind: OpKind,
        c1: &ConstValue<T>,
        c2: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        match self {
            Chain::Same => kind.combine(c1, c2, ptr_size),
            Chain::ProductShifted => c1.shl(c2, ptr_size),
            Chain::ShiftMultiplied => c2.shl(c1, ptr_size),
        }
    }

    /// The `(x a c1) b c2 → x k (c1 f c2)` description of the rewrite, for
    /// the [`EventKind::ConstantFolded`] event.
    fn describe(self, kind: OpKind) -> String {
        let (inner, outer, fold) = match self {
            Chain::Same => (kind.name(), kind.name(), kind.combine_name()),
            Chain::ProductShifted => ("mul", "shl", "shl"),
            Chain::ShiftMultiplied => ("shl", "mul", "shl"),
        };
        format!(
            "reassociate: (x {inner} c1) {outer} c2 → x {} (c1 {fold} c2)",
            kind.name()
        )
    }
}

/// The kind of binary operation involved in a reassociation pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpKind {
    /// Integer addition (`Add`).
    Add,
    /// Integer subtraction (`Sub`).
    Sub,
    /// Integer multiplication (`Mul`).
    Mul,
    /// Bitwise AND (`And`).
    And,
    /// Bitwise OR (`Or`).
    Or,
    /// Bitwise XOR (`Xor`).
    Xor,
    /// Left shift (`Shl`).
    Shl,
    /// Right shift (`Shr`), with an `unsigned` flag.
    Shr { unsigned: bool },
}

impl OpKind {
    /// Folds the two constant operands of a nested chain into one.
    ///
    /// Dispatches to the [`ConstValue`] arithmetic that reproduces the chain's
    /// effect: additive kinds (`Add`/`Sub`) and both shifts add their
    /// operands - `(x - c1) - c2` subtracts `c1 + c2`, and `(x << c1) << c2` shifts by
    /// `c1 + c2` - while `Mul`/`And`/`Or`/`Xor` fold with their own operator.
    /// Every operation is evaluated at `ptr_size` so a `NativeInt` fold wraps
    /// at the host pointer width, matching how the rewritten instruction will
    /// evaluate at runtime.
    ///
    /// # Arguments
    ///
    /// * `c1` - The inner constant.
    /// * `c2` - The outer constant.
    /// * `ptr_size` - Host pointer width governing `NativeInt` wraparound.
    ///
    /// # Returns
    ///
    /// The combined constant, or `None` if the fold is undefined for these
    /// values (e.g. mismatched constant kinds).
    fn combine<T: Target>(
        self,
        c1: &ConstValue<T>,
        c2: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        match self {
            OpKind::Add | OpKind::Sub | OpKind::Shl | OpKind::Shr { .. } => c1.add(c2, ptr_size),
            OpKind::Mul => c1.mul(c2, ptr_size),
            OpKind::And => c1.bitwise_and(c2, ptr_size),
            OpKind::Or => c1.bitwise_or(c2, ptr_size),
            OpKind::Xor => c1.bitwise_xor(c2, ptr_size),
        }
    }

    /// Returns the mnemonic for this operation.
    ///
    /// Used to render the `(x op c1) op c2 → x op (c1 ...)` description in the
    /// [`EventKind::ConstantFolded`] event; the `Shr` unsigned variant is
    /// distinguished as `shr.un`.
    ///
    /// # Returns
    ///
    /// A static mnemonic string for the kind.
    fn name(self) -> &'static str {
        match self {
            OpKind::Add => "add",
            OpKind::Sub => "sub",
            OpKind::Mul => "mul",
            OpKind::And => "and",
            OpKind::Or => "or",
            OpKind::Xor => "xor",
            OpKind::Shl => "shl",
            OpKind::Shr { unsigned: false } => "shr",
            OpKind::Shr { unsigned: true } => "shr.un",
        }
    }

    /// Returns the mnemonic of the operator that [`combine`](Self::combine)
    /// uses to fold the two constants.
    ///
    /// This differs from [`name`](Self::name) for the non-associative kinds:
    /// `Sub` and both shifts fold their operands with addition, so they report
    /// `"add"` here even though the chain itself subtracts or shifts. Used only
    /// to render the event description accurately.
    ///
    /// # Returns
    ///
    /// The static mnemonic of the folding operator.
    fn combine_name(self) -> &'static str {
        match self {
            OpKind::Add | OpKind::Mul | OpKind::And | OpKind::Or | OpKind::Xor => self.name(),
            OpKind::Sub | OpKind::Shl | OpKind::Shr { .. } => "add",
        }
    }

    /// Reports whether a constant may appear on either side of this operation.
    ///
    /// Gates the "constant on the left" branch of [`check_reassociation`]:
    /// `(c1 + x) + c2` can reassociate because addition commutes, but
    /// `(c1 - x) - c2` cannot, since moving the base out from under the
    /// subtraction changes its sign. Shifts are non-commutative for the same
    /// reason - the base is always the shifted value, never the count.
    ///
    /// # Returns
    ///
    /// `true` for `Add`/`Mul`/`And`/`Or`/`Xor`; `false` for `Sub` and the
    /// shifts.
    const fn is_commutative(self) -> bool {
        match self {
            OpKind::Add | OpKind::Mul | OpKind::And | OpKind::Or | OpKind::Xor => true,
            OpKind::Sub | OpKind::Shl | OpKind::Shr { .. } => false,
        }
    }
}

/// Decodes a reassociable binary op, declining any whose `flags` definition is
/// still read.
///
/// `flags` is a real secondary definition (`SsaOp::defs()` yields it,
/// `SsaOp::ReadFlags` and `SsaOp::BranchFlags` consume it) and `make_op` below
/// reproduces none of them - it builds every replacement with `flags: None`, and
/// additionally rewrites the outer op into a `Copy`, dropping a second one.
/// Forwarding the flags through would be wrong rather than merely incomplete:
/// the carry of a combined `add x, (c1 + c2)` is not the carry of the two
/// chained adds it replaces.
///
/// The condition is *liveness*, not presence. A lifter emits flags for nearly
/// every arithmetic instruction while only a few feed a conditional branch, so
/// rejecting every flag-producing op would disable this pass on essentially all
/// native code. A definition nothing reads can be dropped with the op that made
/// it, which is what dead-code elimination would do regardless.
///
/// # Arguments
///
/// * `op` - The instruction operation to classify.
/// * `uses` - Per-variable use counts, consulted to decide whether the op's
///   `flags` definition is dead.
///
/// # Returns
///
/// `Some((kind, dest, left, right))` for a reassociable op with dead flags,
/// where `left`/`right` are its two value operands; `None` otherwise.
fn get_op_kind<T: Target>(
    op: &SsaOp<T>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<(OpKind, SsaVarId, SsaVarId, SsaVarId)> {
    let flags_are_dead =
        |flags: &Option<SsaVarId>| flags.is_none_or(|f| uses.get(&f).copied().unwrap_or(0) == 0);
    match op {
        SsaOp::Add {
            dest,
            left,
            right,
            flags,
        } if flags_are_dead(flags) => Some((OpKind::Add, *dest, *left, *right)),
        SsaOp::Sub {
            dest,
            left,
            right,
            flags,
        } if flags_are_dead(flags) => Some((OpKind::Sub, *dest, *left, *right)),
        SsaOp::Mul {
            dest,
            left,
            right,
            flags,
        } if flags_are_dead(flags) => Some((OpKind::Mul, *dest, *left, *right)),
        SsaOp::And {
            dest,
            left,
            right,
            flags,
        } if flags_are_dead(flags) => Some((OpKind::And, *dest, *left, *right)),
        SsaOp::Or {
            dest,
            left,
            right,
            flags,
        } if flags_are_dead(flags) => Some((OpKind::Or, *dest, *left, *right)),
        SsaOp::Xor {
            dest,
            left,
            right,
            flags,
        } if flags_are_dead(flags) => Some((OpKind::Xor, *dest, *left, *right)),
        SsaOp::Shl {
            dest,
            value,
            amount,
            flags,
        } if flags_are_dead(flags) => Some((OpKind::Shl, *dest, *value, *amount)),
        SsaOp::Shr {
            dest,
            value,
            amount,
            unsigned,
            flags,
        } if flags_are_dead(flags) => Some((
            OpKind::Shr {
                unsigned: *unsigned,
            },
            *dest,
            *value,
            *amount,
        )),
        _ => None,
    }
}

/// Builds a replacement operation of `kind` from the given operands.
///
/// The inverse of [`get_op_kind`], used to reconstruct the inner op after its
/// constant is rewritten. Every replacement is built with `flags: None`, which
/// is why [`get_op_kind`] only admits ops whose flags are dead - a live flags
/// reader would be left dangling. For the shifts, the positional `left`/`right`
/// pair maps onto the named `value`/`amount` fields.
///
/// # Arguments
///
/// * `kind` - The operation kind to construct.
/// * `dest` - Destination variable of the result.
/// * `left` - First operand (the shifted `value` for shifts).
/// * `right` - Second operand (the shift `amount` for shifts).
///
/// # Returns
///
/// The constructed [`SsaOp`] with no flags output.
fn make_op<T: Target>(kind: OpKind, dest: SsaVarId, left: SsaVarId, right: SsaVarId) -> SsaOp<T> {
    match kind {
        OpKind::Add => SsaOp::Add {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Sub => SsaOp::Sub {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Mul => SsaOp::Mul {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::And => SsaOp::And {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Or => SsaOp::Or {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Xor => SsaOp::Xor {
            dest,
            left,
            right,
            flags: None,
        },
        OpKind::Shl => SsaOp::Shl {
            dest,
            value: left,
            amount: right,
            flags: None,
        },
        OpKind::Shr { unsigned } => SsaOp::Shr {
            dest,
            value: left,
            amount: right,
            unsigned,
            flags: None,
        },
    }
}

/// Scans every instruction and collects the reassociation opportunities.
///
/// A thin driver over [`check_reassociation`]: it visits each instruction in
/// program order and keeps the ones that match the nested-constant pattern.
/// Collecting into a `Vec` up front separates detection from mutation, so
/// [`apply_reassociations`] can edit the function without invalidating the
/// def-use index the scan relied on.
///
/// # Arguments
///
/// * `ssa` - The SSA function to scan.
/// * `constants` - Map of variables proven constant, from [`SsaFunction::find_constants`].
/// * `index` - Def-use index used to locate each operand's defining op.
/// * `uses` - Per-variable use counts gating single-use and dead-flags checks.
///
/// # Returns
///
/// Every detected [`ReassociationCandidate`], in program order.
fn find_candidates<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    index: &DefUseIndex<T>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Vec<ReassociationCandidate<T>> {
    let mut candidates = Vec::new();
    for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
        if let Some(candidate) = check_reassociation(
            ssa,
            instr.op(),
            block_idx,
            instr_idx,
            constants,
            index,
            uses,
        ) {
            candidates.push(candidate);
        }
    }
    candidates
}

/// Tests one instruction for the nested-constant pattern and builds a candidate.
///
/// The op must itself be a reassociable kind with a constant operand (`...
/// op c2`, either side of a commutative op), and its non-constant operand must
/// be defined by the *same* kind with its own constant (`(x op c1) op c2`), or
/// be a product and a left shift, one under the other ([`Chain`]). Three
/// conditions then guard
/// safety: the inner result must have a single use (or the shared intermediate
/// would be corrupted), the inner constant must be private to this chain
/// (`apply_reassociations` rewrites its defining `Const` in place), and for the
/// shifts each count and their sum must lie in `0..width` - a count outside
/// that range has no convention-independent meaning, and a chain whose sum
/// reaches the width saturates where a single shift would not. The
/// commutative kinds are retried with the constant on the inner left.
///
/// # Arguments
///
/// * `ssa` - The SSA function, used to read the base operand's type width.
/// * `op` - The candidate outer operation.
/// * `block_idx` - Block of the outer instruction.
/// * `instr_idx` - Index of the outer instruction within its block.
/// * `constants` - Map of variables proven constant.
/// * `index` - Def-use index used to locate the inner defining op.
/// * `uses` - Per-variable use counts for the single-use, private-constant, and
///   dead-flags checks.
///
/// # Returns
///
/// A [`ReassociationCandidate`] if the pattern and all safety conditions hold,
/// otherwise `None`.
fn check_reassociation<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    block_idx: usize,
    instr_idx: usize,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    index: &DefUseIndex<T>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<ReassociationCandidate<T>> {
    let (outer_kind, dest, outer_left, outer_right) = get_op_kind(op, uses)?;
    // A constant on the left of a commutative op is the same chain.
    let (outer_left, outer_right) = if constants.contains_key(&outer_right) {
        (outer_left, outer_right)
    } else if outer_kind.is_commutative() && constants.contains_key(&outer_left) {
        (outer_right, outer_left)
    } else {
        return None;
    };
    let c2_value = constants.get(&outer_right)?;

    let (inner_block, inner_instr, inner_op) = index.full_definition(outer_left)?;
    let (inner_kind, inner_dest, inner_left, inner_right) = get_op_kind(inner_op, uses)?;

    let chain = match (inner_kind, outer_kind) {
        (inner, outer) if inner == outer => Chain::Same,
        (OpKind::Mul, OpKind::Shl) => Chain::ProductShifted,
        (OpKind::Shl, OpKind::Mul) => Chain::ShiftMultiplied,
        _ => return None,
    };
    let op_kind = match chain {
        Chain::Same => outer_kind,
        Chain::ProductShifted | Chain::ShiftMultiplied => OpKind::Mul,
    };
    let inner_uses = uses.get(&inner_dest).copied().unwrap_or(0);
    if inner_uses > 1 {
        return None;
    }

    // `apply_reassociations` rewrites the instruction defining the inner
    // constant *in place*, so that constant must be private to this chain. The
    // `inner_uses` check above guards the intermediate result, which is a
    // different variable - a constant materialized once and read by several
    // instructions would be silently rewritten under all of them.
    let const_is_private = |const_var: SsaVarId| uses.get(&const_var).copied().unwrap_or(0) <= 1;

    // `(x << c1) << c2 == x << (c1 + c2)` only while every count is in
    // range. A shift by a count outside `0..width` has no meaning this crate
    // commits to - ISAs disagree (x86 masks by 31 or 63, AArch64 by the
    // register width), so `ConstValue::shl`/`shr` decline to fold one - and
    // summing it with another count manufactures a valid-looking one:
    // `(x << -1) << 2` is not `x << 1`. And once `c1 + c2` reaches the width
    // the chained shifts saturate - to zero, or to -1 for an arithmetic right
    // shift of a negative value - while a single shift by the sum is out of
    // range. The other six kinds are width-independent, so this only gates
    // the shifts.
    //
    // A product and a shift are one scaling, `(x * c1) << c2 == x * (c1 <<
    // c2)` modulo the width, while the shift's count is in range for the
    // value it shifts: the product's result, or `x`.
    let shift_stays_in_width = |base: SsaVarId, c1: &ConstValue<T>, c2: &ConstValue<T>| {
        let (shifted, counts) = match chain {
            Chain::Same if matches!(outer_kind, OpKind::Shl | OpKind::Shr { .. }) => (
                base,
                vec![
                    c1.as_i64(),
                    c2.as_i64(),
                    c1.as_i64()
                        .zip(c2.as_i64())
                        .and_then(|(a, b)| a.checked_add(b)),
                ],
            ),
            Chain::Same => return true,
            Chain::ProductShifted => (inner_dest, vec![c2.as_i64()]),
            Chain::ShiftMultiplied => (base, vec![c1.as_i64()]),
        };
        let Some(width) = ssa
            .variable(shifted)
            .map(|var| var.var_type().clone())
            .and_then(|ty| T::bit_width(&ty))
        else {
            // No width information means no proof; refuse rather than guess.
            return false;
        };
        counts
            .into_iter()
            .all(|count| count.is_some_and(|count| (0..i64::from(width)).contains(&count)))
    };
    // The constant the rewrite folds into is rewritten in place, so it too
    // must be private: the inner one, or the product's own for a shift
    // multiplied.
    let folded_is_private = |c1_var: SsaVarId| match chain {
        Chain::ShiftMultiplied => const_is_private(outer_right),
        Chain::Same | Chain::ProductShifted => const_is_private(c1_var),
    };

    if let Some(c1_value) = constants.get(&inner_right) {
        if !folded_is_private(inner_right) || !shift_stays_in_width(inner_left, c1_value, c2_value)
        {
            return None;
        }
        return Some(ReassociationCandidate {
            block_idx,
            instr_idx,
            dest,
            base_var: inner_left,
            const1_var: inner_right,
            const2_var: outer_right,
            const1_value: c1_value.clone(),
            const2_value: c2_value.clone(),
            inner_block,
            inner_instr,
            inner_dest,
            op_kind,
            chain,
        });
    }

    if inner_kind.is_commutative()
        && let Some(c1_value) = constants.get(&inner_left)
    {
        if !folded_is_private(inner_left) || !shift_stays_in_width(inner_right, c1_value, c2_value)
        {
            return None;
        }
        return Some(ReassociationCandidate {
            block_idx,
            instr_idx,
            dest,
            base_var: inner_right,
            const1_var: inner_left,
            const2_var: outer_right,
            const1_value: c1_value.clone(),
            const2_value: c2_value.clone(),
            inner_block,
            inner_instr,
            inner_dest,
            op_kind,
            chain,
        });
    }

    None
}

/// Applies the detected candidates in one editor session.
///
/// Each candidate is realized as three in-place op replacements: the inner
/// `Const` becomes the combined constant, the inner op is rebuilt against that
/// constant, and the outer op collapses to a `Copy` of the inner result (which
/// a later copy-propagation pass removes). A shift multiplied is two: the
/// product's `Const` becomes the combined factor and the product reads `x`
/// directly, leaving the shift unread. A folded constant is found where it is
/// defined, in whatever block. Positions already touched by an
/// overlapping candidate are skipped so chained rewrites never rewrite the same
/// instruction twice, and each candidate re-checks that both instructions still
/// exist before editing, since a prior rewrite may have changed the block.
///
/// # Arguments
///
/// * `ssa` - The SSA function to transform in place.
/// * `candidates` - The opportunities from [`find_candidates`].
/// * `env` - The pass environment: where events are recorded, naming the method.
/// * `ptr_size` - Host pointer width governing `NativeInt` constant combination.
///
/// # Returns
///
/// `true` if any candidate was rewritten (or, on a failed edit session, to
/// force the pass-group transaction to verify and roll back the mutated
/// function).
fn apply_reassociations<T>(
    ssa: &mut SsaFunction<T>,
    candidates: Vec<ReassociationCandidate<T>>,
    env: &PassEnv<'_, T>,
    ptr_size: PointerSize,
) -> bool
where
    T: Target,
{
    let mut modified: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut changed = false;

    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for candidate in candidates {
            // Skip if either position has already been rewritten by a prior
            // overlapping candidate. This avoids ambiguous chain rewrites.
            if modified.contains(&(candidate.inner_block, candidate.inner_instr))
                || modified.contains(&(candidate.block_idx, candidate.instr_idx))
            {
                continue;
            }

            let Some(combined) = candidate.chain.fold(
                candidate.op_kind,
                &candidate.const1_value,
                &candidate.const2_value,
                ptr_size,
            ) else {
                continue;
            };

            // The constant the chain folds into, wherever it is defined: a
            // constant hoisted out of a loop lies in another block than the
            // ops reading it.
            let folded = match candidate.chain {
                Chain::Same | Chain::ProductShifted => candidate.const1_var,
                Chain::ShiftMultiplied => candidate.const2_var,
            };
            // The folded value is that variable's, of its type: a shift's two
            // counts may be of two widths, `int32` 44 and `int64` 7, and their
            // sum is still a count of the first. A value the type does not hold
            // is no rewrite.
            let Some(combined) = editor
                .function()
                .variable(folded)
                .map(|variable| variable.var_type().clone())
                .and_then(|ty| {
                    let (bits, _) = combined.integer_pattern(ptr_size)?;
                    ConstValue::from_bits_of_type(bits, &ty)
                })
                .filter(|retyped| {
                    retyped.as_i64().is_some() && retyped.as_i64() == combined.as_i64()
                        || retyped.integer_pattern(ptr_size) == combined.integer_pattern(ptr_size)
                })
            else {
                continue;
            };
            let Some((const_block, const_instr_idx)) = editor
                .function()
                .blocks()
                .iter()
                .enumerate()
                .find_map(|(position, block)| {
                    block
                        .instructions()
                        .iter()
                        .position(|instr| {
                            matches!(
                                instr.op(),
                                SsaOp::Const { dest, .. } if *dest == folded
                            )
                        })
                        .map(|instr| (position, instr))
                })
            else {
                continue;
            };

            let inner_exists = editor
                .function()
                .block(candidate.inner_block)
                .and_then(|block| block.instruction(candidate.inner_instr))
                .is_some();
            let outer_exists = editor
                .function()
                .block(candidate.block_idx)
                .and_then(|block| block.instruction(candidate.instr_idx))
                .is_some();
            if !inner_exists || !outer_exists {
                continue;
            }

            editor.replace_instruction_op(
                const_block,
                const_instr_idx,
                SsaOp::Const {
                    dest: folded,
                    value: combined.clone(),
                },
            )?;
            if candidate.chain == Chain::ShiftMultiplied {
                // `x * c2` in the product's place; the shift it read is read
                // by nothing.
                editor.replace_instruction_op(
                    candidate.block_idx,
                    candidate.instr_idx,
                    make_op(
                        candidate.op_kind,
                        candidate.dest,
                        candidate.base_var,
                        folded,
                    ),
                )?;
            } else {
                editor.replace_instruction_op(
                    candidate.inner_block,
                    candidate.inner_instr,
                    make_op(
                        candidate.op_kind,
                        candidate.inner_dest,
                        candidate.base_var,
                        folded,
                    ),
                )?;
                editor.replace_instruction_op(
                    candidate.block_idx,
                    candidate.instr_idx,
                    SsaOp::Copy {
                        dest: candidate.dest,
                        src: candidate.inner_dest,
                    },
                )?;
            }

            modified.insert((candidate.inner_block, candidate.inner_instr));
            modified.insert((candidate.block_idx, candidate.instr_idx));

            env.record(EventKind::ConstantFolded)
                .location(candidate.instr_idx)
                .message(candidate.chain.describe(candidate.op_kind));
            changed = true;
        }
        Ok(())
    });

    if result.is_err() {
        // The session runs under `SsaRollbackPolicy::Never`, so a failed edit or
        // boundary repair leaves the edits applied - the function is mutated and
        // possibly mid-repair. Reporting "unchanged" would make the pass-group
        // transaction skip **both** verification and rollback, keeping damaged
        // IR and keeping it unchecked. Report the change so the transaction
        // verifies this function and rolls it back.
        return true;
    }

    changed
}

/// Pass that reorders nested associative operations to enable constant
/// folding.
///
/// Rewrites
/// `(x op c1) op c2` into `x op (c1 combined c2)` so the constants
/// collapse. Wraps at the pointer width of the context's machine
/// ([`AnalysisContext::machine`](crate::analysis::AnalysisContext::machine))
/// on native-size integers.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReassociationPass;

impl ReassociationPass {
    /// Creates a new reassociation pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for ReassociationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "reassociation"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Reorder associative operations to expose simplification opportunities"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run the reassociation pass on `ssa`.
    ///
    /// Rewrites nested associative operations `(x op c1) op c2` into
    /// `x op (c1 combined c2)` so the constants can be folded.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any operation was rewritten.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let ptr_size = env.context().machine().ptr_size;
        let constants = ssa.find_constants();
        let index = DefUseIndex::<T>::build_with_ops(ssa);
        let uses = ssa.count_uses();
        let candidates = find_candidates(ssa, &constants, &index, &uses);
        Ok(apply_reassociations(ssa, candidates, env, ptr_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            ops::FlagsMask,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType, mock_op_at, run_mock_pass_boundary},
    };

    fn add(
        c1: ConstValue<MockTarget>,
        c2: ConstValue<MockTarget>,
    ) -> Option<ConstValue<MockTarget>> {
        OpKind::Add.combine(&c1, &c2, PointerSize::Bit64)
    }

    /// `combine` on `Add` sums the two constants.
    #[test]
    fn op_kind_combine_add() {
        assert_eq!(
            add(ConstValue::I32(5), ConstValue::I32(3)),
            Some(ConstValue::I32(8))
        );
    }

    /// `combine` on `Xor` bitwise-XORs the two constants.
    #[test]
    fn op_kind_combine_xor() {
        let r = OpKind::Xor.combine(
            &ConstValue::<MockTarget>::I32(0xF0),
            &ConstValue::<MockTarget>::I32(0x0F),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(0xFF)));
    }

    /// `combine` on `Mul` multiplies the two constants.
    #[test]
    fn op_kind_combine_mul() {
        let r = OpKind::Mul.combine(
            &ConstValue::<MockTarget>::I32(7),
            &ConstValue::<MockTarget>::I32(11),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(77)));
    }

    /// `combine` on `And` bitwise-ANDs the two constants.
    #[test]
    fn op_kind_combine_and() {
        let r = OpKind::And.combine(
            &ConstValue::<MockTarget>::I32(0xF0),
            &ConstValue::<MockTarget>::I32(0x33),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(0x30)));
    }

    /// `combine` on `Or` bitwise-ORs the two constants.
    #[test]
    fn op_kind_combine_or() {
        let r = OpKind::Or.combine(
            &ConstValue::<MockTarget>::I32(0xF0),
            &ConstValue::<MockTarget>::I32(0x0F),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(0xFF)));
    }

    /// Sub uses Add internally for combining (`(x - c1) - c2` → `x - (c1 + c2)`).
    #[test]
    fn op_kind_combine_sub() {
        let r = OpKind::Sub.combine(
            &ConstValue::<MockTarget>::I32(5),
            &ConstValue::<MockTarget>::I32(3),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(8)));
    }

    /// Shl uses Add for shift-amount combining (`(x << c1) << c2` → `x << (c1 + c2)`).
    #[test]
    fn op_kind_combine_shl() {
        let r = OpKind::Shl.combine(
            &ConstValue::<MockTarget>::I32(2),
            &ConstValue::<MockTarget>::I32(3),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(5)));
    }

    /// `combine` on a signed `Shr` adds the two shift counts.
    #[test]
    fn op_kind_combine_shr() {
        let r = OpKind::Shr { unsigned: false }.combine(
            &ConstValue::<MockTarget>::I32(2),
            &ConstValue::<MockTarget>::I32(3),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(5)));
    }

    /// `combine` on an unsigned `Shr` adds the two shift counts.
    #[test]
    fn op_kind_combine_shr_unsigned() {
        let r = OpKind::Shr { unsigned: true }.combine(
            &ConstValue::<MockTarget>::I32(1),
            &ConstValue::<MockTarget>::I32(2),
            PointerSize::Bit64,
        );
        assert_eq!(r, Some(ConstValue::I32(3)));
    }

    /// `is_commutative` is true for the additive/bitwise kinds and false for
    /// `Sub` and the shifts.
    #[test]
    fn op_kind_is_commutative() {
        assert!(OpKind::Add.is_commutative());
        assert!(OpKind::Mul.is_commutative());
        assert!(OpKind::And.is_commutative());
        assert!(OpKind::Or.is_commutative());
        assert!(OpKind::Xor.is_commutative());
        assert!(!OpKind::Sub.is_commutative());
        assert!(!OpKind::Shl.is_commutative());
        assert!(!OpKind::Shr { unsigned: false }.is_commutative());
        assert!(!OpKind::Shr { unsigned: true }.is_commutative());
    }

    /// Associative kinds report their own mnemonic as `combine_name`.
    #[test]
    fn op_kind_combine_name_associative() {
        // Associative ops report their own name as combine_name.
        assert_eq!(OpKind::Add.combine_name(), "add");
        assert_eq!(OpKind::Mul.combine_name(), "mul");
        assert_eq!(OpKind::And.combine_name(), "and");
        assert_eq!(OpKind::Or.combine_name(), "or");
        assert_eq!(OpKind::Xor.combine_name(), "xor");
    }

    /// Non-associative kinds report `add` as `combine_name`.
    #[test]
    fn op_kind_combine_name_non_associative() {
        // Non-associative ops fall back to add for the constant fold.
        assert_eq!(OpKind::Sub.combine_name(), "add");
        assert_eq!(OpKind::Shl.combine_name(), "add");
        assert_eq!(OpKind::Shr { unsigned: false }.combine_name(), "add");
        assert_eq!(OpKind::Shr { unsigned: true }.combine_name(), "add");
    }

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    fn local_at(
        ssa: &mut SsaFunction<MockTarget>,
        idx: u16,
        block: usize,
        instr: usize,
    ) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    fn run_reassociate(
        ssa: &mut SsaFunction<MockTarget>,
        label: &str,
        log: &EventLog<MockTarget>,
    ) -> bool {
        run_mock_pass_boundary(ssa, label, |ssa| {
            ReassociationPass.run(
                ssa,
                &PassEnv::new(&0u32, log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        })
    }

    /// `(x + 3) + 7` reassociates and folds to `x + 10`.
    #[test]
    fn reassociate_add_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(7),
        }));
        // (x + 3) + 7 → x + (3 + 7) = x + 10
        block.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "add reassociation", &log);
        assert!(changed, "constant add reassociation should fire");
        assert!(log.has(EventKind::ConstantFolded));
    }

    /// The chain `(x * c1) <outer> c2` over a parameter-like `x`, its
    /// constants hoisted into an entry block apart from the ops, as a loop
    /// over an array reads them: `outer` builds the outer op from the inner
    /// result and `c2`. Returns the function after the pass, with the inner
    /// op at block 1 index 0 and the outer at index 1.
    fn hoisted_chain(
        inner: impl FnOnce(SsaVarId, SsaVarId, SsaVarId) -> SsaOp<MockTarget>,
        outer: impl FnOnce(SsaVarId, SsaVarId, SsaVarId) -> SsaOp<MockTarget>,
        c1: ConstValue<MockTarget>,
        c2: ConstValue<MockTarget>,
        inner_reads: usize,
    ) -> (SsaFunction<MockTarget>, EventLog<MockTarget>) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let first = local_at(&mut ssa, 1, 0, 1);
        let second = local_at(&mut ssa, 2, 0, 2);
        let product = local_at(&mut ssa, 3, 1, 0);
        let scaled = local_at(&mut ssa, 4, 1, 1);
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(5),
        }));
        entry.add_instruction(instr(SsaOp::Const {
            dest: first,
            value: c1,
        }));
        entry.add_instruction(instr(SsaOp::Const {
            dest: second,
            value: c2,
        }));
        entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
        let mut body = SsaBlock::new(1);
        body.add_instruction(instr(inner(product, x, first)));
        body.add_instruction(instr(outer(scaled, product, second)));
        let returned = if inner_reads > 1 {
            let both = local_at(&mut ssa, 5, 1, 2);
            body.add_instruction(instr(SsaOp::Add {
                dest: both,
                left: scaled,
                right: product,
                flags: None,
            }));
            both
        } else {
            scaled
        };
        body.add_instruction(instr(SsaOp::returns(Some(returned))));
        ssa.add_block(entry);
        ssa.add_block(body);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        run_reassociate(&mut ssa, "scaling chain", &log);
        (ssa, log)
    }

    /// The constant a block's op `index` reads at `operand` (0 the left or
    /// shifted, 1 the right or count), where an entry-block `Const` defines it.
    fn read_constant(
        ssa: &SsaFunction<MockTarget>,
        index: usize,
        operand: usize,
    ) -> Option<ConstValue<MockTarget>> {
        let read = match mock_op_at(ssa, 1, index) {
            SsaOp::Mul { left, right, .. } => [*left, *right],
            SsaOp::Shl { value, amount, .. } => [*value, *amount],
            _ => return None,
        }[operand];
        ssa.find_constants().get(&read).cloned()
    }

    /// **A product shifted is one product.** `(i * 3) << 3`, as a RISC-V
    /// search over 24-byte records computes its index, is `i * 24`, its
    /// constants hoisted out of the loop into the entry block; the shift
    /// becomes a copy. A shift by a count no narrower than the product's
    /// width, or a product something else reads, stays as it is.
    #[test]
    fn a_product_shifted_is_one_product() {
        let product = |dest, left, right| SsaOp::Mul {
            dest,
            left,
            right,
            flags: None,
        };
        let shift = |dest, value, amount| SsaOp::Shl {
            dest,
            value,
            amount,
            flags: None,
        };
        let (ssa, log) = hoisted_chain(product, shift, ConstValue::I32(3), ConstValue::I32(3), 1);
        assert!(log.has(EventKind::ConstantFolded));
        assert!(matches!(mock_op_at(&ssa, 1, 0), SsaOp::Mul { .. }));
        assert_eq!(read_constant(&ssa, 0, 1), Some(ConstValue::I32(24)));
        assert!(
            matches!(mock_op_at(&ssa, 1, 1), SsaOp::Copy { .. }),
            "the shift is the product: {:?}",
            mock_op_at(&ssa, 1, 1)
        );

        let (ssa, _) = hoisted_chain(product, shift, ConstValue::I32(3), ConstValue::I32(32), 1);
        assert!(
            matches!(mock_op_at(&ssa, 1, 1), SsaOp::Shl { .. }),
            "a count of the width shifts out every bit"
        );
        let (ssa, _) = hoisted_chain(product, shift, ConstValue::I32(3), ConstValue::I32(3), 2);
        assert!(
            matches!(mock_op_at(&ssa, 1, 1), SsaOp::Shl { .. }),
            "a product read elsewhere is not rewritten under its reader"
        );
        assert_eq!(read_constant(&ssa, 0, 1), Some(ConstValue::I32(3)));
    }

    /// **A shift multiplied is one product.** `(i << 1) * 3` is `i * 6`, and
    /// so is `3 * (i << 1)`: the product's own constant takes the factor, as
    /// the shift's count is no factor, and the product reads `i`. A shift by
    /// a count outside the width stays.
    #[test]
    fn a_shift_multiplied_is_one_product() {
        let shift = |dest, value, amount| SsaOp::Shl {
            dest,
            value,
            amount,
            flags: None,
        };
        for constant_left in [false, true] {
            let product = move |dest, left, right| SsaOp::Mul {
                dest,
                left: if constant_left { right } else { left },
                right: if constant_left { left } else { right },
                flags: None,
            };
            let (ssa, log) =
                hoisted_chain(shift, product, ConstValue::I32(1), ConstValue::I32(3), 1);
            assert!(log.has(EventKind::ConstantFolded));
            let SsaOp::Mul { left, right, .. } = mock_op_at(&ssa, 1, 1) else {
                panic!("the product stays one: {:?}", mock_op_at(&ssa, 1, 1));
            };
            let constants = ssa.find_constants();
            let mut read: Vec<_> = [left, right]
                .into_iter()
                .map(|operand| constants.get(operand).cloned())
                .collect();
            read.sort_by_key(|constant| constant.as_ref().and_then(ConstValue::as_i64));
            assert_eq!(
                read,
                [Some(ConstValue::I32(5)), Some(ConstValue::I32(6))],
                "the product reads i, not the shift, by 6: constant left {constant_left}"
            );
        }
        let product = |dest, left, right| SsaOp::Mul {
            dest,
            left,
            right,
            flags: None,
        };
        let (ssa, _) = hoisted_chain(shift, product, ConstValue::I32(32), ConstValue::I32(3), 1);
        assert_eq!(
            read_constant(&ssa, 1, 1),
            Some(ConstValue::I32(3)),
            "a count of the width has no one meaning"
        );
    }

    /// `(x << c1) << c2` is `x << (c1 + c2)` only while the combined count stays
    /// below the operand's width. `(x << 20) << 20` on a 32-bit value
    /// saturates to zero, while `x << 40` is out of range: x86 masks it to
    /// `x << 8`, a completely different value.
    #[test]
    fn chained_shifts_past_the_operand_width_are_not_combined() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(20),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(20),
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: inner,
            value: x,
            amount: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: outer,
            value: inner,
            amount: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_reassociate(&mut ssa, "overwide shift chain", &log);

        assert!(
            matches!(mock_op_at(&ssa, 0, 4), SsaOp::Shl { .. }),
            "40 >= 32, so the shifts must not be combined; got {:?}",
            mock_op_at(&ssa, 0, 4)
        );
    }

    /// A chain that stays inside the width must still combine.
    #[test]
    fn chained_shifts_within_the_operand_width_are_combined() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: inner,
            value: x,
            amount: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: outer,
            value: inner,
            amount: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "in-width shift chain", &log);
        assert!(changed, "2 + 3 = 5 < 32, so the shifts should combine");
    }

    /// **A folded constant keeps its variable's type.** A shift's two counts
    /// may be of two widths: `(x << 44) << 7L`, as strength reduction states
    /// a product by 128 over an `int64`, sums the counts into the `int32` that
    /// held 44, and that variable holds an `int32` 51, not an `int64` one a
    /// host would load as the wider value.
    #[test]
    fn a_folded_constant_keeps_its_variables_type() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let typed = |ssa: &mut SsaFunction<MockTarget>, at: usize, ty: MockType| {
            ssa.create_variable(
                VariableOrigin::Local(u16::try_from(at).expect("few")),
                0,
                DefSite::instruction(0, at),
                ty,
            )
        };
        let x = typed(&mut ssa, 0, MockType::I64);
        let c1 = typed(&mut ssa, 1, MockType::I32);
        let c2 = typed(&mut ssa, 2, MockType::I64);
        let inner = typed(&mut ssa, 3, MockType::I64);
        let outer = typed(&mut ssa, 4, MockType::I64);
        let mut block = SsaBlock::new(0);
        for (dest, value) in [
            (x, ConstValue::I64(1)),
            (c1, ConstValue::I32(44)),
            (c2, ConstValue::I64(7)),
        ] {
            block.add_instruction(instr(SsaOp::Const { dest, value }));
        }
        block.add_instruction(instr(SsaOp::Shl {
            dest: inner,
            value: x,
            amount: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: outer,
            value: inner,
            amount: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_reassociate(&mut ssa, "mixed count widths", &log));
        assert_eq!(
            ssa.find_constants().get(&c1).cloned(),
            Some(ConstValue::I32(51)),
            "{ssa}"
        );
    }

    /// A negative count is out of range, so `(x << -1) << 2` is not `x << 1`
    /// even though the counts sum to one.
    #[test]
    fn a_chain_with_a_negative_shift_count_is_not_combined() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(-1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: inner,
            value: x,
            amount: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: outer,
            value: inner,
            amount: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "negative shift count", &log);

        assert!(!changed, "-1 is not a shift count, so nothing combines");
        assert!(
            matches!(mock_op_at(&ssa, 0, 4), SsaOp::Shl { amount, .. } if *amount == c2),
            "the outer shift must be left as it is; got {:?}",
            mock_op_at(&ssa, 0, 4)
        );
    }

    /// A flags definition that something reads blocks the rewrite: `make_op`
    /// builds every replacement with `flags: None` and turns the outer op into a
    /// `Copy`, so both flags definitions would be dropped while a reader remains.
    #[test]
    fn a_chain_with_live_flags_is_not_reassociated() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let inner_flags = local_at(&mut ssa, 4, 0, 3);
        let outer = local_at(&mut ssa, 5, 0, 4);
        let read = local_at(&mut ssa, 6, 0, 5);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: x,
            right: c1,
            flags: Some(inner_flags),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        // The inner op's carry is read, so the chain may not be rewritten.
        block.add_instruction(instr(SsaOp::ReadFlags {
            dest: read,
            flags: inner_flags,
            mask: FlagsMask::CARRY,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(read))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_reassociate(&mut ssa, "live flags reassociation", &log);

        assert!(
            matches!(mock_op_at(&ssa, 0, 3), SsaOp::Add { flags: Some(_), .. }),
            "the flag-producing inner add must survive; got {:?}",
            mock_op_at(&ssa, 0, 3)
        );
        assert_eq!(ssa.validate(), Ok(()));
    }

    /// ...but a flags definition nothing reads must not block it, or the pass
    /// would never fire on native code, where nearly every arithmetic
    /// instruction defines flags.
    #[test]
    fn a_chain_with_dead_flags_is_still_reassociated() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let inner_flags = local_at(&mut ssa, 4, 0, 3);
        let outer = local_at(&mut ssa, 5, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: x,
            right: c1,
            // Defined but never read.
            flags: Some(inner_flags),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "dead flags reassociation", &log);

        assert!(
            changed,
            "a dead flags definition must not block reassociation"
        );
    }

    /// The rewrite overwrites the instruction defining the inner constant *in
    /// place*, so that constant must belong to this chain alone. The
    /// `inner_uses > 1` guard checks the intermediate *result*, not the
    /// constant, so a constant shared with an unrelated instruction is silently
    /// rewritten under it.
    ///
    /// `c1 = Const 3; t = x + c1; r = t + c2; s = x + c1` - combining into
    /// `c1 = Const 10` also changes `s` from `x + 3` to `x + 10`.
    #[test]
    fn a_constant_shared_with_another_instruction_is_not_overwritten() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);
        let other = local_at(&mut ssa, 5, 0, 5);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(7),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        // A second, independent user of the same constant.
        block.add_instruction(instr(SsaOp::Add {
            dest: other,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(other))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_reassociate(&mut ssa, "shared constant reassociation", &log);

        let c1_value = ssa.blocks().iter().find_map(|block| {
            block
                .instructions()
                .iter()
                .find_map(|instr| match instr.op() {
                    SsaOp::Const { dest, value } if *dest == c1 => Some(value.clone()),
                    _ => None,
                })
        });
        assert_eq!(
            c1_value,
            Some(ConstValue::I32(3)),
            "the shared constant must keep its value; the other `x + c1` still reads it"
        );
    }

    /// `(x * 3) * 5` reassociates and folds to `x * 15`.
    #[test]
    fn reassociate_mul_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(5),
        }));
        // (x * 3) * 5 → x * (3 * 5) = x * 15
        block.add_instruction(instr(SsaOp::Mul {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "mul reassociation", &log);
        assert!(changed, "constant mul reassociation should fire");
    }

    /// `(x & 0xF0) & 0x0F` reassociates and folds to `x & 0`.
    #[test]
    fn reassociate_and_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(0xFF),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(0xF0),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(0x0F),
        }));
        // (x & 0xF0) & 0x0F → x & (0xF0 & 0x0F) = x & 0
        block.add_instruction(instr(SsaOp::And {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::And {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "and reassociation", &log);
        assert!(changed, "constant and reassociation should fire");
    }

    /// `(x | 0xF0) | 0x0F` reassociates and folds to `x | 0xFF`.
    #[test]
    fn reassociate_or_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(0x00),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(0xF0),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(0x0F),
        }));
        // (x | 0xF0) | 0x0F → x | (0xF0 | 0x0F) = x | 0xFF
        block.add_instruction(instr(SsaOp::Or {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Or {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "or reassociation", &log);
        assert!(changed, "constant or reassociation should fire");
    }

    /// A non-nested expression yields no candidates and no changes.
    #[test]
    fn no_candidates_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let y = local_at(&mut ssa, 1, 0, 1);
        let z = local_at(&mut ssa, 2, 0, 2);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(2),
        }));
        // No nested expression with constants
        block.add_instruction(instr(SsaOp::Add {
            dest: z,
            left: x,
            right: y,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(z))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "no-candidate reassociation", &log);
        assert!(
            !changed,
            "no nested constant pattern should not trigger reassociation"
        );
    }

    /// A commutative chain with the inner constant on the left still fires.
    #[test]
    fn reassociate_commutative_constant_on_left() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(7),
        }));
        // (3 + x) + 7 = x + (3 + 7) = x + 10
        block.add_instruction(instr(SsaOp::Add {
            dest: inner,
            left: c1,
            right: x,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "commutative-left reassociation", &log);
        assert!(
            changed,
            "commutative reassociation with constant on left should fire"
        );
    }

    /// `(x << 2) << 3` reassociates and folds to `x << 5`.
    #[test]
    fn reassociate_shift_combines_amounts() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(3),
        }));
        // (x << 2) << 3 → x << (2 + 3) = x << 5
        block.add_instruction(instr(SsaOp::Shl {
            dest: inner,
            value: x,
            amount: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Shl {
            dest: outer,
            value: inner,
            amount: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "shift reassociation", &log);
        assert!(changed, "shift reassociation should fire");
    }

    /// `(x - 3) - 2` reassociates and folds to `x - 5`.
    #[test]
    fn reassociate_sub_combines_constants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let x = local_at(&mut ssa, 0, 0, 0);
        let c1 = local_at(&mut ssa, 1, 0, 1);
        let c2 = local_at(&mut ssa, 2, 0, 2);
        let inner = local_at(&mut ssa, 3, 0, 3);
        let outer = local_at(&mut ssa, 4, 0, 4);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: x,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: c2,
            value: ConstValue::I32(2),
        }));
        // (x - 3) - 2 → x - (3 + 2) = x - 5
        block.add_instruction(instr(SsaOp::Sub {
            dest: inner,
            left: x,
            right: c1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Sub {
            dest: outer,
            left: inner,
            right: c2,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(outer))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "sub reassociation", &log);
        assert!(changed, "sub reassociation should fire");
    }

    /// An empty function reassociates nothing and reports no change.
    #[test]
    fn reassociate_empty_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_reassociate(&mut ssa, "empty reassociation", &log);
        assert!(!changed);
    }
}
