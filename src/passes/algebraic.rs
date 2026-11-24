//! Algebraic simplification pass - transforms redundant operations using
//! the identities catalogued in [`crate::analysis::algebraic`], the
//! conversion-chain rule in [`crate::analysis::convert`], and the idiom folds
//! in [`crate::analysis::address`], [`crate::analysis::byteswap`],
//! [`crate::analysis::rotate`] and [`crate::analysis::mba`].
//!
//! Pure-SSA and target-agnostic. Hosts run [`AlgebraicSimplificationPass`]
//! directly or register it with the scheduler.
//!
//! # Algorithm
//!
//! 1. Find all constant-valued variables via [`SsaFunction::find_constants`],
//!    and count every variable's uses once via [`SsaFunction::count_uses`].
//!    Both are taken before any edit, so they are exact for the whole scan.
//! 2. Offer each instruction to the rules below in order; the first match
//!    wins. An op with a second definition is considered only when that
//!    definition is a flags output nothing reads, since every replacement
//!    either has no flags output or sets its flags differently.
//!    1. **Constant folding** ([`Machine::evaluate`]): every value operand is
//!       a known constant, so the op is a literal. A computation over lanes
//!       whose restart is at or beyond its length executes no element, so
//!       each of its results is what it leaves ([`executing_nothing`](crate::analysis::algebraic::executing_nothing)).
//!    2. **Flag routing** ([`flags::route`](crate::analysis::flags::route)): a `ReadFlags` of a flags value
//!       `MergeFlags` or `InsertFlag` composed reads the input its bits came
//!       from.
//!    3. **Conversion-chain collapse** ([`collapse_conversion_chain`](crate::analysis::convert::collapse_conversion_chain)): an
//!       integer conversion reading another may read straight past it -
//!       `(uint16_t)(uint64_t)(uint32_t)x` is one conversion, not three. A
//!       narrowing conversion of an operation on values extended from the
//!       narrow width is the operation at that width
//!       ([`narrow_truncated_operation`](crate::analysis::convert::narrow_truncated_operation)).
//!       **Reshape-chain collapse** ([`reshape::collapse_reshape`](crate::analysis::reshape::collapse_reshape)) does the
//!       same for the reinterprets, shuffles and extracts that move a value's
//!       bits between vector shapes.
//!    4. **Constant-index address folding**
//!       ([`address::fold_constant_index`](crate::analysis::address::fold_constant_index)): a `PtrAdd` with a constant index
//!       is a base plus a displacement, and one adding nothing to a base of
//!       its own type is that base.
//!    5. **Known bits** ([`Reader::known_constant`](crate::analysis::bits::Reader::known_constant), [`Reader::redundant_mask`](crate::analysis::bits::Reader::redundant_mask),
//!       [`Reader::demanded_field`](crate::analysis::bits::Reader::demanded_field), [`Reader::copied_value`](crate::analysis::bits::Reader::copied_value)): a bit operation every result
//!       bit of which is known is a constant - a field read back after a write
//!       of it into an unknown container - an `and` that keeps every bit its
//!       operand can have is its operand, a field read back is the value
//!       written into it, whether or not that value is known, and a value's
//!       bits moved back where they were are the value.
//!    6. **Byte-swap, rotate and MBA idioms** ([`byteswap::fold`](crate::analysis::byteswap::fold),
//!       [`rotate::fold`](crate::analysis::rotate::fold), [`mba::fold`](crate::analysis::mba::fold)): a whole subtree matched through the
//!       operands' defining instructions - a masked shift chain that reverses
//!       the bytes, a shift pair that rotates, a mixed boolean-arithmetic
//!       tangle that spells a single operation. Beside them, **decided
//!       selects** ([`select::fold`](crate::analysis::select::fold)): an arm that is a select on the same
//!       condition or its negation is the arm that condition picks.
//!    7. **Algebraic identities** ([`simplify_op`](crate::analysis::algebraic::simplify_op)):
//!       - `x ^ x` → 0, `x - x` → 0, `x & x` → `x`, `x | x` → `x`
//!       - `x + 0` → `x`, `x * 1` → `x`, `x * 0` → 0
//!       - `x << 0` → `x`, `x >> 0` → `x`
//!       - `ceq(x, x)` → 1, `clt(x, x)` → 0, `cgt(x, x)` → 0
//!       - `div(x, 1)` → `x`, `rem(x, 1)` → 0
//! 3. Replace each match with a `Const`, a `Copy`, or a rewritten operation
//!    (a collapsed conversion, an address without its index, or the operation
//!    an idiom spells), all in one edit scope.
//! 4. Report one [`EventKind::ConstantFolded`] event per replacement.
//!
//! # Complexity
//!
//! O(n log n + C·d + I·k) - the constant scan and the use count are each one
//! pass over the `n` instructions into an ordered map; the rule scan offers
//! each instruction to every rule with map lookups; each of the `C` integer
//! conversions walks the `d` links it collapses through; and each idiom
//! matcher walks a bounded subtree of `k` defining instructions (at most eight
//! byte-swap terms, a fixed depth for the others) for the `I` operations of
//! the kinds it tops. No rule rescans the function.

mod addresses;
mod apply;
mod constants;
mod conversions;
mod identities;
mod idioms;
mod operations;
mod site;

use std::collections::{BTreeMap, BTreeSet};

use self::{apply::apply_simplifications, site::Site};
use crate::{
    Error, Result,
    analysis::{
        address::AddressWalk,
        cfg::SsaCfg,
        consts::Machine,
        convert::{NarrowPhiOperand, NarrowValue, Readers, narrowable_phis},
        distribute::Factoring,
        doubleword::{DoublewordFold, Regrouping, StorePair, WideLoad},
        reshape::{ReshapedPhis, SourceOperand},
    },
    events::EventKind,
    facts::Substitution,
    graph::{NodeId, algorithms::compute_dominators},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    pointer::PointerSize,
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Rewrites every phi joining one place as a phi of that place on every edge,
/// for the phi passes to collapse: each operand is one value, or moves back
/// to it through the address model's moves alone
/// ([`AddressWalk::same_place_ancestors`](crate::analysis::address::AddressWalk::same_place_ancestors)), that value of the phi's kind,
/// pointer or integer, and width, and defined above the phi's block, where
/// it dominates it.
///
/// x86-64 stages a call's stack arguments on one arm, `sub rsp, 8; push rax;
/// call; pop rcx; pop rsi`: the stack pointer leaving that arm is the one
/// that entered it, which the other arm leaves untouched, so the join is that
/// value. Left a phi of two values of one place, every address taken past it
/// was the stack pointer's, and the frame's variables were read through it
/// (shadow's `login` passed `&tty` as `reg_1[4]`). Only the phi is rewritten:
/// a value moved back stays the address it is elsewhere, which names a place
/// of the frame where the value it moved from is the stack pointer.
///
/// # Arguments
///
/// * `ssa` - The function, edited in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
/// * `ptr_size` - The target pointer width, at which displacements wrap.
///
/// # Returns
///
/// `true` if any phi was rewritten; else `false`.
fn join_one_place<T>(ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>, ptr_size: PointerSize) -> bool
where
    T: Target,
{
    let cfg = SsaCfg::from_ssa(ssa);
    let dominators = compute_dominators(&cfg, NodeId::new(0));
    // Each phi joining one place, the place, and the predecessors whose
    // operands name it another way.
    let mut joined: Vec<(usize, usize, SsaVarId, Vec<usize>)> = Vec::new();
    for (block_idx, block) in ssa.blocks().iter().enumerate() {
        for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
            let result = phi.result();
            let mut operands: Vec<SsaVarId> = phi
                .operands()
                .iter()
                .map(PhiOperand::value)
                .filter(|value| *value != result)
                .collect();
            operands.sort_unstable();
            operands.dedup();
            // One distinct operand is the trivial phi dead-code removal
            // answers for.
            if operands.len() < 2 {
                continue;
            }
            let ancestors: Vec<Vec<SsaVarId>> = operands
                .iter()
                .map(|operand| AddressWalk::new(ssa, ptr_size).same_place_ancestors(*operand))
                .collect();
            // The kind and width of the phi's storage, which are what its
            // replacement must hold: what a type lane states of a pointer's
            // pointee changes as types are recovered, and the join decided on
            // it would be decided again once they are.
            let shape = |value: SsaVarId| {
                ssa.variable(value).map(|variable| {
                    let ty = variable.var_type();
                    (T::is_pointer(ty), T::bit_width(ty))
                })
            };
            let result_shape = shape(result);
            // Defined in a block strictly above the phi's, the place is there
            // at the end of every predecessor.
            let place = operands.iter().copied().find(|candidate| {
                ssa.variable(*candidate).is_some_and(|variable| {
                    let defined = variable.def_site().block();
                    shape(*candidate) == result_shape
                        && defined != block_idx
                        && dominators.dominates(NodeId::new(defined), NodeId::new(block_idx))
                }) && operands
                    .iter()
                    .zip(&ancestors)
                    .all(|(operand, from)| operand == candidate || from.contains(candidate))
            });
            if let Some(place) = place {
                let other: Vec<usize> = phi
                    .operands()
                    .iter()
                    .filter(|operand| operand.value() != place && operand.value() != result)
                    .map(PhiOperand::predecessor)
                    .collect();
                joined.push((block_idx, phi_idx, place, other));
            }
        }
    }
    if joined.is_empty() {
        return false;
    }
    // Every edge carries the place itself: the phi is then the trivial one
    // the phi passes collapse, by their own checks of what may read it.
    let mut rewritten = false;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, phi_idx, place, other) in joined {
            for predecessor in other {
                if editor.set_phi_operand(block_idx, phi_idx, predecessor, place)? {
                    rewritten = true;
                }
            }
            env.record(EventKind::ConstantFolded)
                .location(block_idx)
                .message("a phi of one place is that place".to_string());
        }
        Ok(())
    });
    rewritten && result.is_ok()
}

/// Rewrites every web of phis joining one reshape of values as that reshape
/// of the phis joining the values ([`ReshapedPhis`]).
///
/// Each phi of a web is replaced by a phi of the source type, joining the
/// values its own joined values are reshapes of, the source phis of the phis
/// it joined, and the source constants defined at the end of their
/// predecessors; its result is redefined at the top of its block as the
/// reshape of that phi, so every reader keeps the value it read and the
/// collapse of the reshapes reading it reaches the source phi.
///
/// # Arguments
///
/// * `ssa` - The function, edited in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// `true` if any web was rewritten, or the edit failed, leaving the function
/// to be cleaned; else `false`.
fn unreshape_phis<T>(ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> bool
where
    T: Target,
{
    let readers = Readers::of(ssa);
    let webs = ReshapedPhis::of(ssa, &readers);
    if webs.is_empty() {
        return false;
    }
    let sources = ReshapedPhis::released(&webs, &readers);
    // Each phi's result and the source phi standing for it.
    let mut unreshaped: Vec<(SsaVarId, SsaVarId)> = Vec::new();
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for web in webs {
            let mut source_phis = BTreeMap::new();
            for phi in &web.phis {
                // The source phi is the variable the reshaped one was.
                let origin = editor
                    .function()
                    .variable(phi.result)
                    .map_or(VariableOrigin::Phi, |variable| variable.origin());
                source_phis.insert(
                    phi.result,
                    editor.create_fresh_variable(
                        origin,
                        0,
                        DefSite::phi(phi.block),
                        web.source_type.clone(),
                    ),
                );
            }
            for phi in web.phis {
                let joined = *source_phis
                    .get(&phi.result)
                    .ok_or_else(|| Error::new("a web phi has no source phi"))?;
                let mut node = PhiNode::new(joined, phi.origin);
                for (predecessor, _, source) in phi.operands {
                    let value = match source {
                        SourceOperand::Value(value) => value,
                        SourceOperand::Phi(result) => *source_phis
                            .get(&result)
                            .ok_or_else(|| Error::new("a web phi joins a phi outside its web"))?,
                        SourceOperand::Constant(value) => {
                            editor.constant_at_exit(predecessor, web.source_type.clone(), value)?
                        }
                    };
                    node.add_operand(PhiOperand::new(value, predecessor));
                }
                // A removed phi moves the later ones of its block, so each is
                // found by its result.
                let index = editor
                    .function()
                    .block(phi.block)
                    .and_then(|block| {
                        block
                            .phi_nodes()
                            .iter()
                            .position(|node| node.result() == phi.result)
                    })
                    .ok_or_else(|| Error::new("a web phi is missing from its block"))?;
                editor.remove_phi(phi.block, index)?;
                editor.add_phi(phi.block, node)?;
                let (mut operand, mut operand_type) = (joined, web.source_type.clone());
                let last = web.steps.len().saturating_sub(1);
                for (at, step) in web.steps.iter().enumerate() {
                    let dest = if at == last {
                        phi.result
                    } else {
                        editor.create_fresh_variable(
                            VariableOrigin::Phi,
                            0,
                            DefSite::instruction(phi.block, at),
                            step.result_type().clone(),
                        )
                    };
                    editor.insert_instruction(
                        phi.block,
                        at,
                        SsaInstruction::synthetic(step.op(dest, operand, &operand_type)),
                    )?;
                    operand = dest;
                    operand_type = step.result_type().clone();
                }
                unreshaped.push((phi.result, joined));
                env.record(EventKind::ConstantFolded)
                    .location(0)
                    .message("a phi of reshaped values is the reshape of their phi".to_string());
            }
        }
        Ok(())
    });
    // A failed edit leaves the function mutated; see `apply_simplifications`.
    let _ = result;
    // The source phi holds what the phi held, in the register the joined
    // values are in, and a reshape it no longer joins holds its source.
    for (reshaped, source) in unreshaped.into_iter().chain(sources) {
        ssa.note_substitution(reshaped, source, Substitution::Algebraic);
    }
    true
}

/// Narrows every web of phis of values extended from one narrower width
/// ([`narrowable_phis`]).
///
/// Each narrow phi joins the narrow values, the narrow phis of the phis
/// narrowed with it, a constant defined at the end of its predecessor at the
/// narrow width, and a wide value no reader sees past the narrow bits
/// truncated there; each wide phi's result is redefined as its narrow phi's
/// extension at the top of its block, so every reader keeps the value it read
/// and the rules that narrow an operation can reach past it.
///
/// # Arguments
///
/// * `ssa` - The function, edited in place.
/// * `env` - The pass environment: where events are recorded, naming the
///   method, and the machine whose pointer size is the width of the native
///   integers.
/// * `reads_narrow` - Operations the caller knows read only the narrow bits
///   of an extended value ([`narrowable_phis`]).
///
/// # Returns
///
/// `true` if any phi was narrowed, or the edit failed, leaving the function
/// to be cleaned; else `false`.
pub fn narrow_phis<T>(
    ssa: &mut SsaFunction<T>,
    env: &PassEnv<'_, T>,
    reads_narrow: impl Fn(&SsaOp<T>) -> bool,
) -> bool
where
    T: Target,
{
    let ptr_size = env.context().machine().ptr_size;
    let readers = Readers::of(ssa);
    let plans = narrowable_phis(ssa, &readers, ptr_size, reads_narrow);
    if plans.is_empty() {
        return false;
    }
    let narrowing: BTreeSet<SsaVarId> = plans.iter().map(|plan| plan.result).collect();
    // Each extension the narrow phis stop reading, nothing else reading it,
    // and the narrow value standing for it from then on.
    let mut sources: Vec<(SsaVarId, SsaVarId)> = Vec::new();
    let mut planned = Vec::with_capacity(plans.len());
    for plan in plans {
        let Some(phi) = ssa.block(plan.block).and_then(|block| {
            block
                .phi_nodes()
                .iter()
                .find(|phi| phi.result() == plan.result)
        }) else {
            continue;
        };
        for (operand, (_, narrow)) in phi.operands().iter().zip(&plan.operands) {
            if let NarrowPhiOperand::Value(source) = narrow
                && operand.value() != *source
                && readers.read_only_by(operand.value(), &narrowing)
            {
                sources.push((operand.value(), *source));
            }
        }
        let Some(wide_type) = ssa
            .variable(plan.result)
            .map(|variable| variable.var_type().clone())
        else {
            continue;
        };
        planned.push((phi.origin(), wide_type, plan));
    }
    // Each wide phi's result and the narrow phi standing for it.
    let mut narrowed: Vec<(SsaVarId, SsaVarId)> = Vec::new();
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        // Every narrow phi first, as the phis of a web join each other's.
        let mut narrow_of: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
        for (_, _, plan) in &planned {
            // The narrow phi is the variable the wide one was, its extension
            // what the wide one's readers read.
            let origin = editor
                .function()
                .variable(plan.result)
                .map_or(VariableOrigin::Phi, |variable| variable.origin());
            narrow_of.insert(
                plan.result,
                editor.create_fresh_variable(
                    origin,
                    0,
                    DefSite::phi(plan.block),
                    plan.narrow_type.clone(),
                ),
            );
        }
        for (origin, wide_type, plan) in planned {
            let wide = plan.result;
            let block_idx = plan.block;
            let narrow = *narrow_of
                .get(&wide)
                .ok_or_else(|| Error::new("a narrowed phi has no narrow phi"))?;
            let mut node = PhiNode::new(narrow, origin);
            for (predecessor, operand) in plan.operands {
                let value = match operand {
                    NarrowPhiOperand::Value(value) => value,
                    NarrowPhiOperand::Joined(joined) => *narrow_of
                        .get(&joined)
                        .ok_or_else(|| Error::new("a narrow phi joins a phi left wide"))?,
                    NarrowPhiOperand::Constant(value) => {
                        editor.constant_at_exit(predecessor, plan.narrow_type.clone(), value)?
                    }
                    NarrowPhiOperand::Truncated(value) => {
                        let at = editor
                            .function()
                            .block(predecessor)
                            .map_or(0, |block| block.instructions().len());
                        let low = editor.create_fresh_variable(
                            VariableOrigin::Phi,
                            0,
                            DefSite::instruction(predecessor, at),
                            plan.narrow_type.clone(),
                        );
                        editor.insert_before_terminator(
                            predecessor,
                            SsaInstruction::synthetic(SsaOp::IntConv {
                                dest: low,
                                operand: value,
                                target: plan.narrow_type.clone(),
                                overflow_check: false,
                                unsigned: true,
                            }),
                        )?;
                        low
                    }
                };
                node.add_operand(PhiOperand::new(value, predecessor));
            }
            // A removed phi moves the later ones of its block, so each is
            // found by its result.
            let index = editor
                .function()
                .block(block_idx)
                .and_then(|block| {
                    block
                        .phi_nodes()
                        .iter()
                        .position(|phi| phi.result() == wide)
                })
                .ok_or_else(|| Error::new("a narrowed phi is missing from its block"))?;
            editor.remove_phi(block_idx, index)?;
            editor.add_phi(block_idx, node)?;
            narrowed.push((wide, narrow));
            // Reinterpreted back first, so the extension reads its source as
            // the joined values were read.
            let (source, at) = match plan.reinterpreted {
                Some(ty) => {
                    let viewed = editor.create_fresh_variable(
                        VariableOrigin::Phi,
                        0,
                        DefSite::instruction(block_idx, 0),
                        ty.clone(),
                    );
                    editor.insert_instruction(
                        block_idx,
                        0,
                        SsaInstruction::synthetic(SsaOp::IntConv {
                            dest: viewed,
                            operand: narrow,
                            target: ty,
                            overflow_check: false,
                            unsigned: true,
                        }),
                    )?;
                    (viewed, 1)
                }
                None => (narrow, 0),
            };
            editor.insert_instruction(
                block_idx,
                at,
                SsaInstruction::synthetic(SsaOp::IntConv {
                    dest: wide,
                    operand: source,
                    target: wide_type,
                    overflow_check: false,
                    unsigned: plan.unsigned,
                }),
            )?;
            env.record(EventKind::ConstantFolded)
                .location(0)
                .message("a phi of extended values is the extension of their phi".to_string());
        }
        Ok(())
    });
    // A failed edit leaves the function mutated; see `apply_simplifications`.
    let _ = result;
    // The narrow phi holds what the wide one held, at its width: the
    // registers the joined values are in, which the wide phi's extension no
    // longer is once its readers read the narrow phi.
    for (wide, narrow) in narrowed {
        ssa.note_substitution(wide, narrow, Substitution::Algebraic);
    }
    // And a joined extension dies with the wide phi, its narrow value read in
    // its place: the register the extension was written to holds that value.
    for (extension, source) in sources {
        ssa.note_substitution(extension, source, Substitution::Algebraic);
    }
    true
}

/// The type of simplification applied to a redundant operation.
#[derive(Debug, Clone)]
enum Simplification<T: Target> {
    /// Replace the operation's result with a known constant value.
    Constant(ConstValue<T>),
    /// Replace the operation's result with a copy of another variable.
    Copy(SsaVarId),
    /// Replace the operation with a different one computing the same value.
    ///
    /// Produced by the conversion-chain collapse (an `IntConv` reading further
    /// up its own chain), the constant-index address fold (a `PtrAdd` without
    /// its index), and the mixed boolean-arithmetic fold (the operation the
    /// obfuscated subtree computes).
    Rewrite(Box<SsaOp<T>>),
    /// Replace the operation with one computing the same bits as a different
    /// kind of value, retyping the destination to match.
    ///
    /// Produced by the lea-lowered mixed boolean-arithmetic fold: the lift
    /// typed the address it computed as a pointer, and the identity proves the
    /// sum is integer addition. See [`mba::MbaFold::dest_type`](crate::analysis::mba::MbaFold::dest_type).
    Retyped(Box<SsaOp<T>>, T::Type),
    /// Replace an operation of several results with one operation defining
    /// each: the first replaces the operation, the rest are inserted after it
    /// in order, and an operation of no results is replaced by nothing.
    /// Produced by the fold of results computed together
    /// ([`SsaOp::folds_results`]: the double-width arithmetic, the exact
    /// float functions of several results), one constant each, and by a
    /// computation over lanes that executes no element
    /// ([`executing_nothing`](crate::analysis::algebraic::executing_nothing)).
    Results(Vec<SsaOp<T>>),
    /// Replace a narrowing conversion with the operation it truncates, at the
    /// narrow width ([`narrow_truncated_operation`](crate::analysis::convert::narrow_truncated_operation)), a comparison of
    /// extended values with the comparison of the values extended
    /// ([`narrow_extended_comparison`](crate::analysis::convert::narrow_extended_comparison)), or a comparison in halves with the
    /// comparison of the whole values ([`doubleword::comparison`](crate::analysis::doubleword::comparison)): each
    /// listed operand the operation reads is replaced by a value of the given
    /// type, defined just before it (a whole comparison's constant side at
    /// the double width).
    Narrowed(Box<SsaOp<T>>, Vec<(SsaVarId, NarrowValue<T>)>, T::Type),
    /// Replace a composition of two halves with the double-width sum they
    /// compute ([`Doubleword::fold`](crate::analysis::doubleword::Doubleword::fold)): each operand is composed from its
    /// halves just before it. Where a half is read apart as well, every reader
    /// the sum dominates reads that half of the sum instead, so the halves'
    /// own arithmetic is left to no reader.
    Doubleword {
        /// The fold.
        fold: DoublewordFold,
        /// Whether either half has a reader besides the composition.
        read_apart: bool,
    },
    /// Replace a double-width sum adding values' halves apart with the sum of
    /// the values ([`Doubleword::regroup`](crate::analysis::doubleword::Doubleword::regroup)): the compositions of what
    /// is left of each high half are built from their halves and the terms
    /// summed just before it.
    Regrouped(Box<Regrouping>),
    /// Replace two stores of a double-width value's halves with one store of
    /// the value ([`Doubleword::store_pair`](crate::analysis::doubleword::Doubleword::store_pair)): the halves are composed just
    /// before the later store, which stores the composition, and the earlier
    /// store is removed.
    StorePair(StorePair),
    /// Replace a composition of two word loads with the load of both
    /// ([`Doubleword::wide_load`](crate::analysis::doubleword::Doubleword::wide_load)), and remove the word loads.
    WideLoad(Box<WideLoad<T>>),
    /// Replace a join of two operations that share an operand with the
    /// shared operand taken out ([`distribute::factored`](crate::analysis::distribute::factored)): the join of the
    /// other operands is defined just before it.
    Factored(Factoring),
    /// Replace a right shift of a left shift with the part of the value the
    /// two select, extended and moved where they leave it
    /// ([`idioms::shifted_part`]): the steps are defined just before it.
    ShiftedPart(Box<ShiftedPart<T>>),
}

/// The part of a value a left shift and a right shift select
/// ([`idioms::shifted_part`]): the bits `low..low + bits` of `source`, extended back
/// to the shifts' width by sign or zero, and moved up by `up` bits.
#[derive(Debug, Clone)]
struct ShiftedPart<T: Target> {
    /// The value shifted left.
    source: SsaVarId,
    /// The lowest bit of `source` the part keeps.
    low: u32,
    /// How many bits the part keeps.
    bits: u32,
    /// Whether the right shift is arithmetic, which extends the part by its
    /// sign rather than by zero.
    signed: bool,
    /// How far up the part ends: the left shift's amount less the right's.
    up: u32,
    /// The integer type the part's width names, of the part's sign, where the
    /// width is an integer type's (a power of two of at least a byte) and the
    /// target has one: the part is then its truncation, extended. `None` for
    /// a part of another width, which a mask keeps.
    narrow: Option<T::Type>,
    /// The type the shifts work at.
    wide: T::Type,
    /// Its width.
    width: u32,
    /// The width of the right shift's amount, which each new amount takes.
    amount_bits: u32,
    /// The type of the right shift's amount.
    amount: T::Type,
}

/// A single instruction identified as a candidate for algebraic simplification.
#[derive(Debug)]
struct SimplificationCandidate<T: Target> {
    /// Index of the block containing the instruction.
    block_idx: usize,
    /// Index of the instruction within the block.
    instr_idx: usize,
    /// Destination variable of the instruction's result; for a store pair,
    /// the value the later store writes; for a block operation, the
    /// direction flag it stops reading.
    dest: SsaVarId,
    /// The simplification to apply: a constant, a copy, or a rewritten operation.
    simplification: Simplification<T>,
    /// Definitions the simplification reads, rewritten in place before it,
    /// keyed by the variable each defines.
    ///
    /// Empty for every rule whose replacement reads only values the matched
    /// subtree already holds, which is all of them but the widened
    /// mixed-boolean-arithmetic identity. See [`mba::MbaFold::also`](crate::analysis::mba::MbaFold::also) for why
    /// that one cannot be a single operation, and why the rewrite it asks for
    /// creates nothing.
    also: Vec<(SsaVarId, SsaOp<T>)>,
    /// Human-readable description of the applied identity.
    description: &'static str,
}

/// Collect every instruction that can be algebraically simplified.
///
/// One linear scan over all instructions, offering each to
/// [`Site::simplification`] and keeping the ones that match. Collecting into a
/// `Vec` here - rather than rewriting during the walk - lets the scan borrow
/// the function immutably; the mutation happens later in
/// [`apply_simplifications`], where no rewrite can invalidate this iteration.
///
/// A value an earlier candidate folds to a constant is that constant for the
/// operations after it, so a chain of constant operations folds in one scan
/// rather than one link per run: the lowering of a single instruction can be
/// a dozen dependent operations (a bit interleave, a grouped reversal), and
/// folded a link at a time it outlasts the normalizer's iteration cap. The
/// constant is the operation's value whether or not its own rewrite is
/// applied, so reading it is sound either way.
///
/// # Arguments
///
/// * `ssa` - The SSA function to scan.
/// * `constants` - Constant-valued variables from [`SsaFunction::find_constants`],
///   consulted by the identity and constant-fold checks, and extended with
///   each constant a candidate folds to.
/// * `uses` - Use counts from [`SsaFunction::count_uses`], for the dead-flags
///   checks.
/// * `ptr_size` - Target pointer width, used when folding pointer-typed operands.
///
/// # Returns
///
/// One [`SimplificationCandidate`] per simplifiable instruction, in scan order.
fn find_candidates<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &mut BTreeMap<SsaVarId, ConstValue<T>>,
    uses: &BTreeMap<SsaVarId, usize>,
    machine: Machine,
) -> Vec<SimplificationCandidate<T>> {
    let mut candidates = Vec::new();
    // The stores a store pair takes, by block and position: a store is one
    // half of at most one pair, or two pairs would each store the word it
    // writes and remove the store the other put in its place.
    let mut paired: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
        let op = instr.op();
        if let Some(candidate) =
            Site::new(ssa, op, block_idx, instr_idx, constants, uses, machine).simplification()
        {
            if let Simplification::StorePair(pair) = &candidate.simplification {
                if paired.contains(&(block_idx, pair.first)) {
                    continue;
                }
                paired.extend([(block_idx, pair.first), (block_idx, instr_idx)]);
            }
            match &candidate.simplification {
                Simplification::Constant(value) => {
                    constants.insert(candidate.dest, value.clone());
                }
                Simplification::Results(results) => {
                    for result in results {
                        if let SsaOp::Const { dest, value } = result {
                            constants.insert(*dest, value.clone());
                        }
                    }
                }
                _ => {}
            }
            candidates.push(candidate);
        }
    }
    candidates
}

/// Pass that simplifies algebraic identities.
///
/// Rewrites operations like `x xor x → 0`, `x + 0 → x`, `x * 1 → x`,
/// `ceq(x, x) → 1`, etc.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - only replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlgebraicSimplificationPass;

impl AlgebraicSimplificationPass {
    /// Creates a new algebraic simplification pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for AlgebraicSimplificationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "algebraic-simplification"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Simplify algebraic identities (x xor x = 0, x + 0 = x, etc.)"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run algebraic simplification on `ssa`.
    ///
    /// Scans all instructions, applies the algebraic identities and the
    /// conversion-chain collapse, and replaces each redundant computation with a
    /// constant, a copy, or a conversion reading further up its own chain.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any operation was rewritten, `false` otherwise.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let machine = env.context().machine();
        let mut constants = ssa.find_constants();
        // Counted once for the whole scan, and exact for all of it: candidates are
        // collected before any edit. Counting per operation would make the scan
        // quadratic in the function's size.
        let uses = ssa.count_uses();
        let candidates = find_candidates(ssa, &mut constants, &uses, machine);
        let simplified = apply_simplifications(ssa, candidates, env);
        let narrowed = narrow_phis(ssa, env, |_| false);
        let joined = join_one_place(ssa, env, machine.ptr_size);
        Ok(unreshape_phis(ssa, env) || joined || narrowed || simplified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Error, NullListener, Result,
        analysis::{
            consts::{ConstEvaluator, Destination},
            context::AnalysisContext,
            doubleword,
            execution::StaticExecution,
        },
        events::EventLog,
        ir::{
            FloatBits, FloatExceptionLayout, FloatExceptions, FloatFormat, FloatRules, NanEncoding,
            OutOfRangeConversion, SsaBlockBuilder, SsaDefSpec, SsaFunctionBuilder,
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            float::FloatRounding,
            instruction::SsaInstruction,
            lanes::VectorConst,
            ops::{
                AccessQualifiers, BinaryOpKind, CmpKind, CompareBlkData, FlagsMask,
                FloatFunctionKind, KindedVecData, VectorBinaryKind, VectorElement,
                VectorShuffleSelector,
            },
            variable::{DefSite, VariableOrigin, VariableRole},
        },
        passes::DeadCodeEliminationPass,
        pointer::PointerSize,
        target::{Endianness, VectorShuffleLane, VectorShuffleMask},
        testing::{
            MACHINE, MockConvLink, MockConversionChain, MockTarget, MockType, mock_i32, mock_i64,
            mock_op_at, mock_ptr, run_mock_pass_boundary,
        },
    };

    /// Build an empty mock SSA function with no blocks.
    fn build_function() -> SsaFunction<MockTarget> {
        SsaFunction::<MockTarget>::new(0, 0)
    }

    /// Build a function with a single block that performs `dest = left ^ right`.
    fn build_xor(
        left_var: SsaVarId,
        right_var: SsaVarId,
        dest_var: SsaVarId,
    ) -> SsaFunction<MockTarget> {
        let mut f = build_function();
        f.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        f.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        f.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        // Const 5 -> left
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: left_var,
            value: ConstValue::I32(5),
        }));
        // Const 5 -> right
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: right_var,
            value: ConstValue::I32(5),
        }));
        // dest = left ^ right (which equals left ^ left's value -> 0 if left == right)
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Xor {
            dest: dest_var,
            left: left_var,
            right: left_var,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest_var))));
        f.add_block(block);
        f.recompute_uses();
        f
    }

    /// **A double-width product of constants is two constants, and a fold below
    /// it in its block still lands on its own instruction.** `u64::MAX * 3`
    /// unsigned is `3 * 2^64 - 3`: low `u64::MAX - 2`, high `2`. The fold
    /// inserts the second constant, so an `add` of constants after it in the
    /// block, folded in the same run, must be found where it moved to rather
    /// than where it was.
    #[test]
    fn a_wide_product_of_constants_folds_to_both_halves() {
        let mut f = build_function();
        for index in 0..7u16 {
            f.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, index as usize),
                if index < 4 {
                    MockType::U64
                } else {
                    MockType::I32
                },
            );
        }
        let var = SsaVarId::from_index;
        let (a, b, low, high, c, d, sum) = (var(0), var(1), var(2), var(3), var(4), var(5), var(6));
        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        for op in [
            SsaOp::Const {
                dest: a,
                value: ConstValue::U64(u64::MAX),
            },
            SsaOp::Const {
                dest: b,
                value: ConstValue::U64(3),
            },
            SsaOp::WideMul {
                low,
                high,
                left: a,
                right: b,
                unsigned: true,
            },
            SsaOp::Const {
                dest: c,
                value: ConstValue::I32(2),
            },
            SsaOp::Const {
                dest: d,
                value: ConstValue::I32(3),
            },
            SsaOp::Add {
                dest: sum,
                left: c,
                right: d,
                flags: None,
            },
            SsaOp::returns(Some(sum)),
        ] {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        f.add_block(block);
        f.recompute_uses();

        let log = EventLog::<MockTarget>::new();
        assert!(
            AlgebraicSimplificationPass.run(
                &mut f,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ) == Ok(true)
        );
        let ops: Vec<&SsaOp<MockTarget>> = f.blocks()[0]
            .instructions()
            .iter()
            .map(SsaInstruction::op)
            .collect();
        assert_eq!(
            ops[2],
            &SsaOp::Const {
                dest: low,
                value: ConstValue::U64(u64::MAX - 2)
            }
        );
        assert_eq!(
            ops[3],
            &SsaOp::Const {
                dest: high,
                value: ConstValue::U64(2)
            }
        );
        assert_eq!(
            ops[6],
            &SsaOp::Const {
                dest: sum,
                value: ConstValue::I32(5)
            }
        );
        assert!(!ops.iter().any(|op| matches!(op, SsaOp::WideMul { .. })));
        assert_defs_at_their_sites(&f);
    }

    /// **A partial remainder of constants is three constants: the remainder,
    /// the condition codes and the quotient's low bit.** `15 fprem 2`
    /// truncates the quotient to 7, so the remainder is 1, `C0` (the carry
    /// flag) and `C3` (the zero flag) hold its bits 2 and 1, and `C1` its bit
    /// 0. A reduction the instruction leaves incomplete (`2^70 fprem 3`) is no
    /// constant and stays.
    #[test]
    fn a_partial_remainder_of_constants_folds_to_every_result() {
        let x87 = |value: f64| {
            FloatBits::new(FloatFormat::Double, u128::from(value.to_bits()))
                .and_then(|double| double.exactly(FloatFormat::X87Extended))
                .expect("an f64 is an x87 extended float exactly")
        };
        let remainder_of = |value: f64, divisor: f64| {
            let mut f = build_function();
            for (index, ty) in [
                MockType::F80,
                MockType::F80,
                MockType::F80,
                MockType::U32,
                MockType::U8,
            ]
            .into_iter()
            .enumerate()
            {
                f.create_variable(
                    VariableOrigin::Local(u16::try_from(index).expect("five locals")),
                    0,
                    DefSite::instruction(0, index.min(2)),
                    ty,
                );
            }
            let var = SsaVarId::from_index;
            let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
            for op in [
                SsaOp::Const {
                    dest: var(0),
                    value: ConstValue::Float(x87(value)),
                },
                SsaOp::Const {
                    dest: var(1),
                    value: ConstValue::Float(x87(divisor)),
                },
                SsaOp::FloatFunction(Box::new(KindedVecData {
                    outputs: vec![var(2), var(3), var(4)],
                    inputs: vec![var(0), var(1)],
                    kind: FloatFunctionKind::Rem,
                })),
                SsaOp::returns(Some(var(2))),
            ] {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            f.add_block(block);
            f.recompute_uses();
            let log = EventLog::<MockTarget>::new();
            assert!(
                AlgebraicSimplificationPass
                    .run(
                        &mut f,
                        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            f
        };

        let f = remainder_of(15.0, 2.0);
        let ops: Vec<&SsaOp<MockTarget>> = f.blocks()[0]
            .instructions()
            .iter()
            .map(SsaInstruction::op)
            .collect();
        let var = SsaVarId::from_index;
        assert_eq!(
            ops[2..5],
            [
                &SsaOp::Const {
                    dest: var(2),
                    value: ConstValue::Float(x87(1.0)),
                },
                &SsaOp::Const {
                    dest: var(3),
                    value: ConstValue::U32(u32::from(
                        FlagsMask::CARRY.bits() | FlagsMask::ZERO.bits()
                    )),
                },
                &SsaOp::Const {
                    dest: var(4),
                    value: ConstValue::U8(1),
                },
            ]
        );
        assert_defs_at_their_sites(&f);

        let f = remainder_of(2.0_f64.powi(70), 3.0);
        assert!(
            f.blocks()[0]
                .instructions()
                .iter()
                .any(|instr| matches!(instr.op(), SsaOp::FloatFunction(_))),
            "an incomplete reduction stays"
        );
    }

    /// Asserts every definition in the first block sits where its def site
    /// says.
    fn assert_defs_at_their_sites(f: &SsaFunction<MockTarget>) {
        let ops: Vec<&SsaOp<MockTarget>> = f.blocks()[0]
            .instructions()
            .iter()
            .map(SsaInstruction::op)
            .collect();
        for (index, op) in ops.iter().enumerate() {
            for def in op.defs() {
                let site = f.variable(def).map(|variable| variable.def_site());
                assert_eq!(site.and_then(|site| site.index()), Some(index), "{op:?}");
            }
        }
    }

    /// **A conversion of a float its integer does not hold folds by the
    /// target's rule.** A NaN, an infinity and a value past the width convert
    /// to x86's integer indefinite, to Arm's saturation with a NaN zero, and
    /// to RISC-V's with a NaN the maximum, signed and unsigned; a value in
    /// range converts exactly whatever the rule, and with no rule stated an
    /// out-of-range one stays.
    #[test]
    fn an_out_of_range_conversion_folds_by_the_targets_rule() {
        let ssa = typed_fn(&[MockType::F64, MockType::I32, MockType::U32]);
        let (x, signed, unsigned) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let convert = |dest, target, is_unsigned| SsaOp::<MockTarget>::FloatToInt {
            dest,
            operand: x,
            target,
            overflow_check: false,
            unsigned: is_unsigned,
            source: None,
        };
        let fold =
            |op: &SsaOp<MockTarget>, value: f64, rule: Option<crate::ir::OutOfRangeConversion>| {
                let constants = BTreeMap::from([(x, ConstValue::from(value))]);
                let machine = Machine {
                    float: FloatRules {
                        out_of_range: rule,
                        default_nan: None,
                        nan_conversion: None,
                        nan_operands: None,
                        extended_nan_operands: None,

                        program_flushes_subnormals: false,

                        nan_encoding: NanEncoding::Ieee,
                    },
                    ..MACHINE
                };
                match Site::new(&ssa, op, 0, 0, &constants, &BTreeMap::new(), machine)
                    .simplification()?
                    .simplification
                {
                    Simplification::Constant(value) => value.integer_pattern(PointerSize::Bit64),
                    _ => None,
                }
                .map(|(bits, _)| bits)
            };
        let x86 = Some(OutOfRangeConversion::Indefinite);
        let arm = Some(OutOfRangeConversion::Saturate { nan_maximum: false });
        let riscv = Some(OutOfRangeConversion::Saturate { nan_maximum: true });
        let mips = Some(OutOfRangeConversion::Maximum);
        let to_i32 = convert(signed, MockType::I32, false);
        let to_u32 = convert(unsigned, MockType::U32, true);
        for (value, rule, i32_bits, u32_bits) in [
            (f64::NAN, x86, 0x8000_0000u64, 0xffff_ffffu64),
            (3e9, x86, 0x8000_0000, 3_000_000_000),
            (-1.0, x86, 0xffff_ffff, 0xffff_ffff),
            (f64::NEG_INFINITY, x86, 0x8000_0000, 0xffff_ffff),
            (f64::NAN, arm, 0, 0),
            (3e9, arm, 0x7fff_ffff, 3_000_000_000),
            (-1.0, arm, 0xffff_ffff, 0),
            (f64::INFINITY, arm, 0x7fff_ffff, 0xffff_ffff),
            (f64::NAN, riscv, 0x7fff_ffff, 0xffff_ffff),
            (-3e9, riscv, 0x8000_0000, 0),
            (-1.5, riscv, 0xffff_ffff, 0),
            (5e9, riscv, 0x7fff_ffff, 0xffff_ffff),
            (f64::NAN, mips, 0x7fff_ffff, 0x7fff_ffff),
            (-3e9, mips, 0x7fff_ffff, 0x7fff_ffff),
            (5e9, mips, 0x7fff_ffff, 0x7fff_ffff),
            (f64::NEG_INFINITY, mips, 0x7fff_ffff, 0x7fff_ffff),
            (-1.5, mips, 0xffff_ffff, 0x7fff_ffff),
        ] {
            assert_eq!(
                fold(&to_i32, value, rule),
                Some(i32_bits),
                "{value} to i32 by {rule:?}"
            );
            assert_eq!(
                fold(&to_u32, value, rule),
                Some(u32_bits),
                "{value} to u32 by {rule:?}"
            );
        }
        assert_eq!(fold(&to_i32, 3e9, None), None, "no rule, no fold");
        assert_eq!(fold(&to_i32, f64::NAN, None), None);
    }

    /// `x - x` is NaN when `x` is NaN, and `x == x` is *false* for NaN. `Sub`,
    /// `Ceq`, `Clt` and `Cgt` carry no float discriminator - `ConstValue` folds
    /// `F32`/`F64` pairs for all of them - so the operand's declared type is the
    /// only thing that can distinguish the sound case from the unsound one.
    #[test]
    fn self_cancelling_identities_are_refused_on_float_operands() {
        let ssa = typed_fn(&[MockType::F64, MockType::F64, MockType::F64]);
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();

        for op in [
            SsaOp::<MockTarget>::Sub {
                dest,
                left: x,
                right: x,
                flags: None,
            },
            SsaOp::Ceq {
                dest,
                left: x,
                right: x,
                signaling: false,
            },
            SsaOp::Clt {
                dest,
                left: x,
                right: x,
                unsigned: false,
                signaling: false,
            },
        ] {
            assert!(
                Site::new(&ssa, &op, 0, 0, &constants, &BTreeMap::new(), MACHINE)
                    .simplification()
                    .is_none(),
                "{op:?} must not fold on a float operand"
            );
        }
    }

    /// The same identities on integers must still fold - the guard must not cost
    /// the optimization it protects.
    #[test]
    fn self_cancelling_identities_still_fold_on_integer_operands() {
        let ssa = int_fn();
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();

        let sub: SsaOp<MockTarget> = SsaOp::Sub {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let candidate = Site::new(&ssa, &sub, 0, 0, &constants, &BTreeMap::new(), MACHINE)
            .simplification()
            .expect("x - x folds for integers");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    /// `xor rax, rax` is the ubiquitous zeroing idiom. Folding it to an `I32`
    /// zero redefines a 64-bit destination with a 32-bit constant.
    #[test]
    fn self_cancellation_materialises_a_constant_of_the_operand_width() {
        let ssa = typed_fn(&[MockType::I64, MockType::I64, MockType::I64]);
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();

        let xor: SsaOp<MockTarget> = SsaOp::Xor {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let candidate = Site::new(&ssa, &xor, 0, 0, &constants, &BTreeMap::new(), MACHINE)
            .simplification()
            .expect("x ^ x folds for integers");
        assert!(
            matches!(
                candidate.simplification,
                Simplification::Constant(ConstValue::I64(0))
            ),
            "a 64-bit operand must yield a 64-bit zero, got {:?}",
            candidate.simplification
        );
    }

    /// Three `I32` variables (v0..v2), enough for any binary op's operands and
    /// destination. `Site::simplification` reads the *operand* type to decide
    /// whether a self-cancelling identity is sound and at what width to
    /// materialise its constant.
    fn int_fn() -> SsaFunction<MockTarget> {
        typed_fn(&[MockType::I32, MockType::I32, MockType::I32])
    }

    /// Same, with caller-chosen types.
    fn typed_fn(types: &[MockType]) -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        for (idx, ty) in types.iter().enumerate() {
            ssa.create_variable(
                VariableOrigin::Local(u16::try_from(idx).unwrap_or(0)),
                0,
                DefSite::instruction(0, idx),
                *ty,
            );
        }
        ssa
    }

    /// An MBA identity whose flags output something reads is left alone.
    ///
    /// The operation an identity folds to sets flags differently from the
    /// tangle it replaces: `a + ~b + 1` sets the carry at `a = b = 0` where
    /// `a - b` clears it, and `(a ^ b) + ((a & b) << 1)` over 8-bit
    /// `a = b = 0x80` leaves carry and overflow clear where `a + b` sets both.
    /// Carrying the flags output across would hand its reader those different
    /// flags, so the fold is refused, as rotate and byte-swap refuse theirs.
    #[test]
    fn an_mba_identity_is_refused_while_its_flags_output_is_read() {
        let shapes: [(&str, Body); 2] = [
            (
                "(a + ~b) + 1",
                Box::new(|block, args| {
                    let not = block.not(mock_i32(), args[1])?;
                    let inner = block.add(mock_i32(), args[0], not)?;
                    let one = block.const_i32(mock_i32(), 1)?;
                    let (_, flags) = block.binary_op_with_flags(
                        mock_i32(),
                        mock_i32(),
                        BinaryOpKind::Add,
                        inner,
                        one,
                        false,
                    )?;
                    block.read_flags(mock_i32(), flags, FlagsMask::CARRY)
                }),
            ),
            (
                "(a ^ b) + ((a & b) << 1)",
                Box::new(|block, args| {
                    let xor = block.xor(mock_i32(), args[0], args[1])?;
                    let and = block.and(mock_i32(), args[0], args[1])?;
                    let one = block.const_i32(mock_i32(), 1)?;
                    let twice = block.shl(mock_i32(), and, one)?;
                    let (_, flags) = block.binary_op_with_flags(
                        mock_i32(),
                        mock_i32(),
                        BinaryOpKind::Add,
                        xor,
                        twice,
                        false,
                    )?;
                    block.read_flags(mock_i32(), flags, FlagsMask::CARRY)
                }),
            ),
        ];
        for (name, body) in shapes {
            let case = Case {
                name: name.to_string(),
                args: vec![MockType::I32, MockType::I32],
                body,
                folds_to: None,
            };
            let mut ssa = build_case(&case);
            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            run_mock_pass_boundary(&mut ssa, name, |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let folded: Vec<_> = log
                .iter()
                .filter(|event| event.message.starts_with("mba"))
                .map(|event| event.message.clone())
                .collect();
            assert!(
                folded.is_empty(),
                "{name}: the flags are read, so no MBA fold may apply; got {folded:?}"
            );
        }
    }

    /// A lea-lowered MBA sum is folded to an `Add` whose destination is the
    /// terms' integer, on the variable and on the instruction alike.
    ///
    /// The lift typed the address as a pointer. Keeping that type on an `Add`
    /// would have integer arithmetic define a pointer, and a type inference
    /// reading the destination would carry the pointer into whatever the sum
    /// flows to: a returned `int` read back as an address.
    #[test]
    fn a_folded_lea_sum_is_retyped_as_the_integer_it_adds() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut sum = None;
        builder
            .in_block(0, |block| {
                let a = block.load_arg(mock_i32(), 0)?;
                let b = block.load_arg(mock_i32(), 1)?;
                let xor = block.xor(mock_i32(), a, b)?;
                let and = block.and(mock_i32(), a, b)?;
                let dest =
                    block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: xor,
                        index: Some(and),
                        stride: 2,
                        offset: 0,
                        result_type: MockType::Ptr,
                    })?;
                sum = Some(dest);
                block.ret(Some(dest))?;
                Ok(())
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let sum = sum.expect("sum");
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "lea mba", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(
                    &method,
                    &log,
                    AnalysisContext::new(Machine {
                        ptr_size: PointerSize::Bit32,
                        ..MACHINE
                    }),
                ),
            ) == Ok(true)
        });
        let site = ssa.variable(sum).expect("sum survives").def_site();
        let instr = &ssa.block(site.block()).expect("block").instructions()
            [site.index().expect("an instruction defines it")];
        assert!(
            matches!(instr.op(), SsaOp::Add { .. }),
            "the tangle folds to an Add: {:?}",
            instr.op()
        );
        assert_eq!(
            ssa.variable(sum).map(|var| var.var_type()),
            Some(&MockType::I32)
        );
        assert_eq!(instr.result_type(), Some(&MockType::I32));
    }

    /// An index the code scales before it adds it is the address's scaled
    /// index: `ptradd p, (i << 2), 1` is `ptradd p, i, 4`, as AArch64 `-O0`
    /// computes `a[i]` with `lsl x0, x0, #2; str w1, [x1, x0]`. A 32-bit
    /// shift under a 64-bit address wraps where the address does not, and
    /// stays.
    #[test]
    fn an_index_scaled_before_it_is_added_is_the_stride() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
        let mut addresses = Vec::new();
        let mut wide_index = None;
        builder
            .in_block(0, |block| {
                let pointer = block.load_arg(mock_ptr(), 0)?;
                let wide = block.load_arg(mock_i64(), 1)?;
                let narrow = block.load_arg(mock_i32(), 2)?;
                wide_index = Some(wide);
                let two = block.const_i32(mock_i32(), 2)?;
                for (index, ty) in [(wide, mock_i64()), (narrow, mock_i32())] {
                    let scaled = block.shl(ty, index, two)?;
                    let dest =
                        block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: pointer,
                            index: Some(scaled),
                            stride: 1,
                            offset: 8,
                            result_type: MockType::Ptr,
                        })?;
                    addresses.push(dest);
                }
                block.ret(None)?;
                Ok(())
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "scaled index", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let op = |address: SsaVarId| {
            let site = ssa.variable(address).expect("address survives").def_site();
            ssa.block(site.block()).expect("block").instructions()
                [site.index().expect("an instruction defines it")]
            .op()
            .clone()
        };
        assert!(
            matches!(
                op(addresses[0]),
                SsaOp::PtrAdd { index: Some(index), stride: 4, offset: 8, .. }
                    if Some(index) == wide_index
            ),
            "{:?}",
            op(addresses[0])
        );
        assert!(
            matches!(op(addresses[1]), SsaOp::PtrAdd { stride: 1, .. }),
            "the control: {:?}",
            op(addresses[1])
        );
    }

    /// An address adding nothing to a pointer is the pointer, so an access
    /// through it is an access through the pointer itself. One over an
    /// integer is the conversion to a pointer, and one adding an offset is a
    /// different address: both stay.
    #[test]
    fn an_address_adding_nothing_to_a_pointer_is_the_pointer() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut addresses = Vec::new();
        builder
            .in_block(0, |block| {
                let pointer = block.load_arg(mock_ptr(), 0)?;
                let integer = block.load_arg(mock_i64(), 1)?;
                for (base, offset) in [(pointer, 0), (integer, 0), (pointer, 4)] {
                    let dest =
                        block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base,
                            index: None,
                            stride: 0,
                            offset,
                            result_type: MockType::Ptr,
                        })?;
                    addresses.push(dest);
                }
                block.ret(None)?;
                Ok(())
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "identity address", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let ops: Vec<_> = addresses
            .iter()
            .map(|address| {
                let site = ssa.variable(*address).expect("address survives").def_site();
                ssa.block(site.block()).expect("block").instructions()
                    [site.index().expect("an instruction defines it")]
                .op()
                .clone()
            })
            .collect();
        assert!(matches!(ops[0], SsaOp::Copy { .. }), "{:?}", ops[0]);
        assert!(matches!(ops[1], SsaOp::PtrAdd { .. }), "{:?}", ops[1]);
        assert!(matches!(ops[2], SsaOp::PtrAdd { .. }), "{:?}", ops[2]);
    }

    /// Two addresses from one base differ by their displacements, symbols
    /// included - which constant evaluation cannot say, because a symbol
    /// folds to no number. A difference with an unknown operand stays.
    #[test]
    fn an_address_difference_folds_to_its_displacement() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let mut results = Vec::new();
        builder
            .in_block(0, |block| {
                let unknown = block.load_arg(mock_i64(), 0)?;
                let start = block.const_value(mock_ptr(), ConstValue::Symbol(0x1000))?;
                let end = block.const_value(mock_ptr(), ConstValue::Symbol(0x1000))?;
                let past = block.const_value(mock_ptr(), ConstValue::Symbol(0x1010))?;
                results.push(block.sub(mock_i64(), end, start)?);
                results.push(block.sub(mock_i64(), past, start)?);
                results.push(block.sub(mock_i64(), past, unknown)?);
                let first = block.add(mock_i64(), results[0], results[1])?;
                let sum = block.add(mock_i64(), first, results[2])?;
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "address difference", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let constants = ssa.find_constants();
        assert_eq!(constants.get(&results[0]), Some(&ConstValue::I64(0)));
        assert_eq!(constants.get(&results[1]), Some(&ConstValue::I64(16)));
        assert_eq!(constants.get(&results[2]), None, "an unknown operand stays");
    }

    /// Two addresses from one base compare as their displacements: two
    /// materialisations of one symbol are equal and order unsigned by
    /// displacement, a value and itself plus 8 differ, and nothing orders them
    /// signed, nor a value against itself plus 8. A float's comparison with
    /// itself stays: it is false on NaN.
    #[test]
    fn an_address_comparison_folds_by_its_displacements() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut results = Vec::new();
        builder
            .in_block(0, |block| {
                let pointer = block.load_arg(mock_ptr(), 0)?;
                let float = block.load_arg(SsaDefSpec::tmp(MockType::F64), 1)?;
                let start = block.const_value(mock_ptr(), ConstValue::Symbol(0x1000))?;
                let end = block.const_value(mock_ptr(), ConstValue::Symbol(0x1000))?;
                let past = block.const_value(mock_ptr(), ConstValue::Symbol(0x1010))?;
                let eight = block.const_i64(mock_i64(), 8)?;
                let moved = block.add(mock_ptr(), pointer, eight)?;
                results.push(block.ceq(mock_i32(), end, start)?);
                results.push(block.ceq(mock_i32(), past, start)?);
                results.push(block.clt(mock_i32(), start, past, true)?);
                results.push(block.cgt(mock_i32(), start, past, true)?);
                results.push(block.clt(mock_i32(), start, end, false)?);
                results.push(block.clt(mock_i32(), start, past, false)?);
                results.push(block.ceq(mock_i32(), moved, pointer)?);
                results.push(block.clt(mock_i32(), pointer, moved, true)?);
                results.push(block.ceq(mock_i32(), float, float)?);
                let mut sum = results[0];
                for result in &results[1..] {
                    sum = block.add(mock_i32(), sum, *result)?;
                }
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "address comparison", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let constants = ssa.find_constants();
        let decided: Vec<Option<bool>> = results
            .iter()
            .map(|result| {
                constants
                    .get(result)
                    .and_then(ConstValue::as_i64)
                    .map(|value| value != 0)
            })
            .collect();
        assert_eq!(
            decided,
            [
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                None,
                Some(false),
                None,
                None
            ]
        );
    }

    /// The pass decides whether a flags output is read from the use counts it
    /// takes once, not from a fresh count per operation.
    ///
    /// Counting per operation is quadratic in the function's size (5.5 s for
    /// 8000 flag-setting adds in a release build). Here the counts handed to
    /// `Site::simplification` say the flags are unread while the function
    /// reads them: a check that recounted would see the reader and refuse, so
    /// a fold proves the counts passed in are the ones consulted.
    #[test]
    fn the_dead_flags_check_reads_the_counts_it_is_given() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let mut and_op = None;
        builder
            .in_block(0, |block| {
                let x = block.load_arg(mock_i32(), 0)?;
                let (dest, flags) = block.binary_op_with_flags(
                    mock_i32(),
                    mock_i32(),
                    BinaryOpKind::And,
                    x,
                    x,
                    false,
                )?;
                and_op = Some(SsaOp::And {
                    dest,
                    left: x,
                    right: x,
                    flags: Some(flags),
                });
                let observed = block.read_flags(mock_i32(), flags, FlagsMask::ZERO)?;
                block.ret(Some(observed))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let op = and_op.expect("and op");
        let constants = ssa.find_constants();
        let check = |uses: &BTreeMap<SsaVarId, usize>| {
            Site::new(&ssa, &op, 0, 0, &constants, uses, MACHINE).simplification()
        };
        assert!(
            check(&ssa.count_uses()).is_none(),
            "the real counts see the reader"
        );
        assert!(
            check(&BTreeMap::new()).is_some(),
            "counts saying the flags are unread must be the ones obeyed"
        );
    }

    /// The pass stays near-linear on a long run of flag-setting arithmetic,
    /// the shape where a per-operation use count is quadratic.
    #[test]
    fn a_long_run_of_flag_setting_adds_is_scanned_once() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        builder
            .in_block(0, |block| {
                let x = block.load_arg(mock_i32(), 0)?;
                let mut acc = x;
                for _ in 0..4000 {
                    let (value, _) = block.binary_op_with_flags(
                        mock_i32(),
                        mock_i32(),
                        BinaryOpKind::Add,
                        acc,
                        x,
                        false,
                    )?;
                    acc = value;
                }
                block.ret(Some(acc))
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let method = 0u32;
        assert!(
            !(AlgebraicSimplificationPass.run(
                &mut ssa,
                &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true))
        );
    }

    /// An identity whose flags output nothing reads still folds.
    ///
    /// `x & x` is `x` for every bit pattern, but on x86 the instruction that
    /// spells it also sets flags, giving the op two definitions. When nothing
    /// reads those flags the second definition is dead weight, and refusing the
    /// fold on account of it would leave the catalogue unreachable on x86
    /// while AArch64 and MIPS, which set no flags there, fold the same
    /// identity.
    #[test]
    fn an_identity_folds_when_its_flags_output_is_dead() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let (mut and_op, mut value) = (None, None);
        builder
            .in_block(0, |block| {
                let x = block.load_arg(mock_i32(), 0)?;
                let dest = block.and(mock_i32(), x, x)?;
                let flags = block.const_i32(mock_i32(), 0)?;
                value = Some(x);
                and_op = Some(SsaOp::And {
                    dest,
                    left: x,
                    right: x,
                    flags: Some(flags),
                });
                block.ret(Some(dest))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let constants = ssa.find_constants();
        let candidate = Site::new(
            &ssa,
            &and_op.expect("and op"),
            0,
            0,
            &constants,
            &ssa.count_uses(),
            MACHINE,
        )
        .simplification()
        .expect("an identity whose flags nothing reads must fold");
        let x = value.expect("operand");
        assert!(
            matches!(candidate.simplification, Simplification::Copy(v) if v == x),
            "x & x is x, so the fold is a copy of the operand"
        );
    }

    /// Control: the same identity is refused while something reads the flags.
    ///
    /// `Copy` replaces the instruction with one that has no flags output, so
    /// folding here would leave the reader naming a variable with no
    /// definition.
    #[test]
    fn an_identity_is_refused_while_its_flags_output_is_read() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let mut and_op = None;
        builder
            .in_block(0, |block| {
                let x = block.load_arg(mock_i32(), 0)?;
                let dest = block.and(mock_i32(), x, x)?;
                let flags = block.const_i32(mock_i32(), 0)?;
                let observed = block.read_flags(mock_i32(), flags, FlagsMask::ZERO)?;
                and_op = Some(SsaOp::And {
                    dest,
                    left: x,
                    right: x,
                    flags: Some(flags),
                });
                block.ret(Some(observed))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let constants = ssa.find_constants();
        assert!(
            Site::new(
                &ssa,
                &and_op.expect("and op"),
                0,
                0,
                &constants,
                &ssa.count_uses(),
                MACHINE,
            )
            .simplification()
            .is_none(),
            "a live flags reader must keep its definition"
        );
    }

    /// Drives a chain collapse through the whole pass, not just through
    /// `Site::simplification`.
    ///
    /// A test that inspects the candidate `Site::simplification` returns covers
    /// the *rule* and nothing downstream of it. This one covers the write path:
    /// `apply_simplifications`' `Simplification::Rewrite` arm, the
    /// `replace_instruction_op` after it, and the edit scope's repair, which
    /// renumbers every `SsaVarId`.
    ///
    /// That renumbering is why the structure is asserted by *scanning* the
    /// block rather than by saved `SsaVarId`s: a stale id would make the
    /// assertion fail for the wrong reason, or pass for one.
    #[test]
    fn run_collapses_a_conversion_chain_and_reports_it() {
        // I32 → I8 → I64 → I16. Only the outer conversion collapses, and it
        // stops on the I8 link, so exactly one rewrite is expected.
        let mut ssa = MockConversionChain::build(
            MockType::I32,
            &[
                MockConvLink::new(MockType::I8, true),
                MockConvLink::new(MockType::I64, true),
                MockConvLink::new(MockType::I16, true),
            ],
        )
        .into_function();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0x5Au32;

        let changed = run_mock_pass_boundary(&mut ssa, "conversion chain collapse", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });

        assert!(changed, "the chain collapses");
        assert_eq!(log.count_kind(EventKind::ConstantFolded), 1);

        let narrowing = ssa
            .all_instructions()
            .find_map(|instr| match instr.op() {
                SsaOp::IntConv {
                    dest,
                    target: MockType::I8,
                    ..
                } => Some(*dest),
                _ => None,
            })
            .expect("the narrowing link survives");
        let collapsed = ssa
            .all_instructions()
            .find_map(|instr| match instr.op() {
                SsaOp::IntConv {
                    operand,
                    target: MockType::I16,
                    ..
                } => Some(*operand),
                _ => None,
            })
            .expect("the outer conversion survives");

        assert_eq!(
            collapsed, narrowing,
            "the surviving conversion must read the narrowing link's result directly"
        );
    }

    /// The mirror of the write path: a pass run over a chain the rule refuses
    /// must leave the function alone and report nothing, so a `changed` of
    /// `true` cannot come from having merely *looked* at a conversion.
    #[test]
    fn run_leaves_a_checked_conversion_alone() {
        let mut ssa = MockConversionChain::build(
            MockType::I32,
            &[
                MockConvLink::new(MockType::I8, false),
                MockConvLink::checked(MockType::I16, false),
            ],
        )
        .into_function();
        let before = ssa.clone();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0x5Bu32;

        let changed = run_mock_pass_boundary(&mut ssa, "checked conversion", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });

        assert!(!changed, "a checked conversion is not simplifiable");
        assert!(log.is_empty());
        assert_eq!(
            ssa.instruction_count(),
            before.instruction_count(),
            "nothing was rewritten"
        );
        assert!(
            ssa.all_instructions().any(|instr| matches!(
                instr.op(),
                SsaOp::IntConv {
                    overflow_check: true,
                    ..
                }
            )),
            "the overflow check is still there"
        );
    }

    /// `x ^ x` is rewritten to a constant zero end-to-end through [`AlgebraicSimplificationPass`].
    /// A copy and a fill whose direction flag is the constant clear walk
    /// upward and read no flag; one whose flag is set keeps reading it.
    #[test]
    fn a_block_with_a_clear_direction_flag_walks_upward() {
        for (flag, upward) in [(ConstValue::False, true), (ConstValue::True, false)] {
            let mut ssa = typed_fn(&[
                MockType::U64,
                MockType::U64,
                MockType::U8,
                MockType::Bool,
                MockType::U64,
                MockType::U64,
                MockType::U32,
                MockType::U32,
                MockType::U8,
                MockType::U64,
                MockType::U64,
                MockType::U64,
                MockType::U64,
                MockType::U32,
            ]);
            let v = SsaVarId::from_index;
            let mut block = SsaBlock::new(0);
            for op in [
                SsaOp::Const {
                    dest: v(3),
                    value: flag.clone(),
                },
                SsaOp::CopyBlk {
                    dest_addr: v(0),
                    src_addr: v(1),
                    size: v(1),
                    element_bytes: 1,
                    backward: Some(v(3)),
                    src_access: AccessQualifiers::FLAT,
                    dest_access: AccessQualifiers::FLAT,
                },
                SsaOp::InitBlk {
                    dest_addr: v(0),
                    value: v(2),
                    size: v(1),
                    backward: Some(v(3)),
                    access: AccessQualifiers::FLAT,
                },
                SsaOp::ScanBlk {
                    remaining: v(4),
                    next: v(5),
                    flags: v(6),
                    address: v(0),
                    value: v(2),
                    count: v(1),
                    flags_in: Some(v(7)),
                    backward: Some(v(3)),
                    element_bytes: 1,
                    while_equal: false,
                    access: AccessQualifiers::FLAT,
                },
                SsaOp::LoadBlk {
                    value: v(8),
                    next: v(9),
                    address: v(0),
                    count: v(1),
                    value_in: Some(v(2)),
                    backward: Some(v(3)),
                    element_bytes: 1,
                    access: AccessQualifiers::FLAT,
                },
                SsaOp::CompareBlk(Box::new(CompareBlkData {
                    remaining: v(10),
                    left_next: v(11),
                    right_next: v(12),
                    flags: v(13),
                    left: v(0),
                    right: v(1),
                    count: v(1),
                    flags_in: Some(v(7)),
                    backward: Some(v(3)),
                    element_bytes: 1,
                    while_equal: true,
                    left_access: AccessQualifiers::FLAT,
                    right_access: AccessQualifiers::FLAT,
                })),
            ] {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            assert!(
                AlgebraicSimplificationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            let directions: Vec<Option<SsaVarId>> = ssa
                .iter_instructions()
                .filter_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::CopyBlk { backward, .. }
                    | SsaOp::InitBlk { backward, .. }
                    | SsaOp::ScanBlk { backward, .. }
                    | SsaOp::LoadBlk { backward, .. } => Some(*backward),
                    SsaOp::CompareBlk(data) => Some(data.backward),
                    _ => None,
                })
                .collect();
            assert_eq!(directions.len(), 5, "{ssa}");
            assert!(
                directions
                    .iter()
                    .all(|backward| backward.is_none() == upward),
                "{flag:?}: {ssa}"
            );
        }
    }

    /// **A repeated block operation whose count is never zero is never
    /// empty.** A scan, a load and a compare of a count of `-1` (x86's
    /// `strlen` idiom, `or ecx, -1; repne scasb`), or of a count with a bit
    /// known set, read nothing of what an empty run would leave: the flags or
    /// the value they took in for it go. A count that may be zero keeps them.
    #[test]
    fn a_repeated_block_operation_whose_count_is_never_zero_takes_no_empty_input() {
        let v = SsaVarId::from_index;
        for (name, never_zero) in [("-1", true), ("count | 1", true), ("an argument", false)] {
            let mut ssa = typed_fn(&[
                MockType::U64,
                MockType::U64,
                MockType::U8,
                MockType::U64,
                MockType::U64,
                MockType::U64,
                MockType::U32,
                MockType::U32,
                MockType::U8,
                MockType::U64,
                MockType::U64,
                MockType::U64,
                MockType::U64,
                MockType::U32,
                MockType::U64,
            ]);
            let mut block = SsaBlock::new(0);
            let count = match name {
                "-1" => vec![SsaOp::Const {
                    dest: v(3),
                    value: ConstValue::U64(u64::MAX),
                }],
                "count | 1" => vec![
                    SsaOp::Const {
                        dest: v(14),
                        value: ConstValue::U64(1),
                    },
                    SsaOp::Or {
                        dest: v(3),
                        left: v(1),
                        right: v(14),
                        flags: None,
                    },
                ],
                _ => vec![SsaOp::Copy {
                    dest: v(3),
                    src: v(1),
                }],
            };
            for op in count.into_iter().chain([
                SsaOp::ScanBlk {
                    remaining: v(4),
                    next: v(5),
                    flags: v(6),
                    address: v(0),
                    value: v(2),
                    count: v(3),
                    flags_in: Some(v(7)),
                    backward: None,
                    element_bytes: 1,
                    while_equal: false,
                    access: AccessQualifiers::FLAT,
                },
                SsaOp::LoadBlk {
                    value: v(8),
                    next: v(9),
                    address: v(0),
                    count: v(3),
                    value_in: Some(v(2)),
                    backward: None,
                    element_bytes: 1,
                    access: AccessQualifiers::FLAT,
                },
                SsaOp::CompareBlk(Box::new(CompareBlkData {
                    remaining: v(10),
                    left_next: v(11),
                    right_next: v(12),
                    flags: v(13),
                    left: v(0),
                    right: v(1),
                    count: v(3),
                    flags_in: Some(v(7)),
                    backward: None,
                    element_bytes: 1,
                    while_equal: true,
                    left_access: AccessQualifiers::FLAT,
                    right_access: AccessQualifiers::FLAT,
                })),
            ]) {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            assert!(
                AlgebraicSimplificationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            let inputs: Vec<bool> = ssa
                .iter_instructions()
                .filter_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::ScanBlk { flags_in, .. } => Some(flags_in.is_some()),
                    SsaOp::LoadBlk { value_in, .. } => Some(value_in.is_some()),
                    SsaOp::CompareBlk(data) => Some(data.flags_in.is_some()),
                    _ => None,
                })
                .collect();
            assert_eq!(inputs, [!never_zero; 3], "{name}: {ssa}");
        }
    }

    /// **A shuffle by a known index selects by mask.** Indices 3, 0, 5 and
    /// 1 of two four-lane tables are the left's lanes 3 and 0, the right's
    /// lane 1 and the left's lane 1, and the index goes; an index past both
    /// tables selects no lane, and the shuffle is left reading its index.
    #[test]
    fn a_shuffle_by_a_known_index_selects_by_mask() {
        for (indices, selected) in [
            (
                [3, 0, 5, 1],
                Some(vec![
                    VectorShuffleLane::Left(3),
                    VectorShuffleLane::Left(0),
                    VectorShuffleLane::Right(1),
                    VectorShuffleLane::Left(1),
                ]),
            ),
            ([3, 0, 9, 1], None),
        ] {
            let mut ssa = typed_fn(&[MockType::V4I32; 4]);
            let v = SsaVarId::from_index;
            let mut block = SsaBlock::new(0);
            for op in [
                SsaOp::Const {
                    dest: v(2),
                    value: ConstValue::Vector(VectorConst::from_lanes(32, indices).unwrap()),
                },
                SsaOp::VectorShuffle {
                    dest: v(3),
                    left: v(0),
                    right: Some(v(1)),
                    selector: VectorShuffleSelector::Index(v(2)),
                },
                SsaOp::returns(Some(v(3))),
            ] {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            ssa.add_block(block);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            assert!(
                AlgebraicSimplificationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            let selector =
                ssa.iter_instructions()
                    .find_map(|(_, _, instruction)| match instruction.op() {
                        SsaOp::VectorShuffle { selector, .. } => Some(selector.clone()),
                        _ => None,
                    });
            let expected = match selected {
                Some(lanes) => VectorShuffleSelector::Mask(VectorShuffleMask::new(lanes)),
                None => VectorShuffleSelector::Index(v(2)),
            };
            assert_eq!(selector, Some(expected), "{ssa}");
        }
    }

    /// **An operation rounding by a known mode states the mode.** A lane
    /// division of doubles reading the constant mode 1 rounds toward zero, as
    /// one stating that rounding does, and reads no mode, and so does the
    /// division a raise is of; one reading the mode 5, which no rule names, is
    /// left reading it.
    #[test]
    fn an_operation_rounding_by_a_known_mode_states_it() {
        for (mode, stated) in [(1, Some(FloatRounding::TowardZero)), (5, None)] {
            let mut ssa = typed_fn(&[
                MockType::F64,
                MockType::F64,
                MockType::F64,
                MockType::I32,
                MockType::U8,
            ]);
            let v = SsaVarId::from_index;
            let mut block = SsaBlock::new(0);
            for op in [
                SsaOp::Const {
                    dest: v(3),
                    value: ConstValue::I32(mode),
                },
                SsaOp::VectorBinary {
                    dest: v(2),
                    left: v(0),
                    right: v(1),
                    kind: VectorBinaryKind::Div,
                    element: VectorElement::float(64).as_scalar(),
                    environment: Some(v(3)),
                },
                SsaOp::FloatRaise {
                    dest: v(4),
                    op: Box::new(SsaOp::VectorBinary {
                        dest: v(2),
                        left: v(0),
                        right: v(1),
                        kind: VectorBinaryKind::Div,
                        element: VectorElement::float(64).as_scalar(),
                        environment: Some(v(3)),
                    }),
                    lanes: None,
                    raises: FloatExceptions::IEEE,
                    layout: FloatExceptionLayout::IEEE,
                },
                SsaOp::returns(Some(v(2))),
            ] {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            ssa.add_block(block);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            assert!(
                AlgebraicSimplificationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            let raised = ssa
                .iter_instructions()
                .find_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::FloatRaise { op, .. } => match op.as_ref() {
                        SsaOp::VectorBinary {
                            element,
                            environment,
                            ..
                        } => Some((element.rounding, *environment)),
                        _ => None,
                    },
                    _ => None,
                });
            let divided = ssa
                .iter_instructions()
                .find_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::VectorBinary {
                        element,
                        environment,
                        ..
                    } => Some((element.rounding, *environment)),
                    _ => None,
                });
            let expected = match stated {
                Some(rounding) => (Some(rounding), None),
                None => (None, Some(v(3))),
            };
            assert_eq!(divided, Some(expected), "mode {mode}: {ssa}");
            assert_eq!(raised, Some(expected), "mode {mode}: {ssa}");
        }
    }

    #[test]
    fn rewrites_xor_self_to_const_zero() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let mut ssa = build_xor(left, right, dest);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0xABu32;

        let changed = run_mock_pass_boundary(&mut ssa, "xor-self algebraic rewrite", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed);
        assert_eq!(log.count_kind(EventKind::ConstantFolded), 1);

        // The third instruction should now be a Const, not Xor.
        match mock_op_at(&ssa, 0, 2) {
            SsaOp::Const {
                value: ConstValue::I32(0),
                dest: d,
            } => assert_eq!(*d, dest),
            other => panic!("expected Const I32(0), got {other:?}"),
        }
    }

    /// **A flag read of a composed flags value reads its source.** The carry of
    /// an `InsertFlag` is its boolean, so the read becomes a copy of it; the
    /// zero flag passes through to the merged update. The control reads the
    /// uncomposed flags `v0`, and stays.
    #[test]
    fn a_flag_read_is_routed_through_its_composition() {
        let mut ssa = typed_fn(&[MockType::Unknown; 9]);
        let v = SsaVarId::from_index;
        let mut block = SsaBlock::new(0);
        for op in [
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
            SsaOp::ReadFlags {
                dest: v(5),
                flags: v(4),
                mask: FlagsMask::CARRY,
            },
            SsaOp::ReadFlags {
                dest: v(6),
                flags: v(4),
                mask: FlagsMask::ZERO,
            },
            SsaOp::ReadFlags {
                dest: v(7),
                flags: v(0),
                mask: FlagsMask::ZERO,
            },
        ] {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            AlgebraicSimplificationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );

        assert!(matches!(mock_op_at(&ssa, 0, 2), SsaOp::Copy { src, .. } if *src == v(3)));
        assert!(matches!(
            mock_op_at(&ssa, 0, 3),
            SsaOp::ReadFlags { flags, mask, .. } if *flags == v(1) && *mask == FlagsMask::ZERO
        ));
        assert!(matches!(
            mock_op_at(&ssa, 0, 4),
            SsaOp::ReadFlags { flags, .. } if *flags == v(0)
        ));
    }

    /// The pass runs and rewrites correctly when driven with [`NullListener`].
    #[test]
    fn null_listener_runs_silently() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let mut ssa = build_xor(left, right, dest);
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "null-listener algebraic rewrite", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed);
    }

    /// A function with nothing to simplify reports no change and logs nothing.
    #[test]
    fn no_op_returns_false_and_records_nothing() {
        // Build an empty function with no instructions to simplify.
        let mut ssa = build_function();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let changed = run_mock_pass_boundary(&mut ssa, "empty algebraic pass", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
        assert!(log.is_empty());
    }

    /// `x / 1` is recognised as a copy of the dividend.
    #[test]
    fn check_simplification_div_by_one() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(right, ConstValue::I32(1))].into();
        let op: SsaOp<MockTarget> = SsaOp::Div {
            dest,
            left,
            right,
            unsigned: false,
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("div-by-1 should simplify");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == left));
    }

    /// `x % 1` is recognised as constant zero.
    #[test]
    fn check_simplification_rem_by_one() {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(right, ConstValue::I32(1))].into();
        let op: SsaOp<MockTarget> = SsaOp::Rem {
            dest,
            left,
            right,
            unsigned: false,
            flags: None,
        };
        let candidate = Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE)
            .simplification()
            .expect("rem-by-1 simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    /// `ceq(x, x)` is recognised as constant one.
    #[test]
    fn check_simplification_ceq_same_var() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: x,
            right: x,
            signaling: false,
        };
        let candidate = Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE)
            .simplification()
            .expect("ceq(x,x) simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(1))
        ));
    }

    /// `clt(x, x)` is recognised as constant zero.
    #[test]
    fn check_simplification_clt_same_var() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: x,
            right: x,
            unsigned: false,
            signaling: false,
        };
        let candidate = Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE)
            .simplification()
            .expect("clt(x,x) simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    /// `cgt(x, x)` is recognised as constant zero.
    #[test]
    fn check_simplification_cgt_same_var() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: x,
            right: x,
            unsigned: false,
            signaling: false,
        };
        let candidate = Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE)
            .simplification()
            .expect("cgt(x,x) simplifies");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    /// `x + 0` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_add_identity() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x + 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// `x * 1` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_mul_identity() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(1))].into();
        let op: SsaOp<MockTarget> = SsaOp::Mul {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x * 1 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// `x * 0` is recognised as constant zero.
    #[test]
    fn check_simplification_mul_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Mul {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x * 0 should simplify to 0");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    /// `x - x` is recognised as constant zero.
    #[test]
    fn check_simplification_sub_self() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Sub {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x - x should simplify to 0");
        assert!(matches!(
            candidate.simplification,
            Simplification::Constant(ConstValue::I32(0))
        ));
    }

    /// The operations [`absorbs_a_join_it_is_part_of`] builds over the
    /// arguments `x` and `y`, the last of them the candidate.
    #[derive(Debug, Clone, Copy)]
    enum Absorbing {
        /// `x & (x | y)`.
        AndOfOr,
        /// `x | (x & y)`.
        OrOfAnd,
        /// `x & ~(x | y)`.
        AndOfNotOr,
        /// `x | ~(x & y)`.
        OrOfNotAnd,
        /// `x & (x ^ -1 | y)`: a complement spelt as an `xor` with every bit
        /// set, under `|`.
        AndOfComplementOr,
        /// `x | (~x & y)`.
        OrOfComplementAnd,
        /// `x & ~(y | y)`: no part of the join is `x`.
        Control,
    }

    /// **A value absorbs a join it is part of.** Over the arguments `x` and
    /// `y`: `x & (x | y)` and `x | (x & y)` are `x`; `x & ~(x | y)` is none
    /// and `x | ~(x & y)` every bit set; `x & (~x | y)` is `x & y` and `x |
    /// (~x & y)` is `x | y`, the complement spelt as `not` or as an `xor` with
    /// every bit set. A join with no part `x`, `x & ~(y | y)`, absorbs
    /// nothing.
    #[test]
    fn absorbs_a_join_it_is_part_of() {
        let candidate = |shape: Absorbing| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
            let mut operands = None;
            builder
                .in_block(0, |block| {
                    let x = block.load_arg(mock_i32(), 0)?;
                    let y = block.load_arg(mock_i32(), 1)?;
                    let not =
                        |block: &mut SsaBlockBuilder<'_, MockTarget>, v| block.not(mock_i32(), v);
                    let result = match shape {
                        Absorbing::AndOfOr => {
                            let join = block.or(mock_i32(), x, y)?;
                            block.and(mock_i32(), x, join)?
                        }
                        Absorbing::OrOfAnd => {
                            let join = block.and(mock_i32(), y, x)?;
                            block.or(mock_i32(), join, x)?
                        }
                        Absorbing::AndOfNotOr => {
                            let join = block.or(mock_i32(), y, x)?;
                            let complement = not(block, join)?;
                            block.and(mock_i32(), x, complement)?
                        }
                        Absorbing::OrOfNotAnd => {
                            let join = block.and(mock_i32(), x, y)?;
                            let complement = not(block, join)?;
                            block.or(mock_i32(), complement, x)?
                        }
                        Absorbing::AndOfComplementOr => {
                            let ones = block.const_i32(mock_i32(), -1)?;
                            let complement = block.xor(mock_i32(), x, ones)?;
                            let join = block.or(mock_i32(), complement, y)?;
                            block.and(mock_i32(), x, join)?
                        }
                        Absorbing::OrOfComplementAnd => {
                            let complement = not(block, x)?;
                            let join = block.and(mock_i32(), y, complement)?;
                            block.or(mock_i32(), x, join)?
                        }
                        Absorbing::Control => {
                            let join = block.or(mock_i32(), y, y)?;
                            let complement = not(block, join)?;
                            block.and(mock_i32(), x, complement)?
                        }
                    };
                    operands = Some((x, y, result));
                    block.ret(Some(result))?;
                    Ok(())
                })
                .expect("build block");
            let ssa = builder.finish().expect("finish");
            let (x, y, result) = operands.expect("built");
            let op = ssa.get_definition(result).expect("the candidate").clone();
            let mut constants = BTreeMap::new();
            for block in ssa.blocks() {
                for instruction in block.instructions() {
                    if let SsaOp::Const { dest, value } = instruction.op() {
                        constants.insert(*dest, value.clone());
                    }
                }
            }
            let found = Site::new(&ssa, &op, 0, 0, &constants, &ssa.count_uses(), MACHINE)
                .simplification()
                .map(|candidate| candidate.simplification);
            (x, y, found)
        };
        let (x, _, found) = candidate(Absorbing::AndOfOr);
        assert!(
            matches!(found, Some(Simplification::Copy(v)) if v == x),
            "{found:?}"
        );
        let (x, _, found) = candidate(Absorbing::OrOfAnd);
        assert!(
            matches!(found, Some(Simplification::Copy(v)) if v == x),
            "{found:?}"
        );
        let (_, _, found) = candidate(Absorbing::AndOfNotOr);
        assert!(
            matches!(&found, Some(Simplification::Constant(value)) if value.is_zero()),
            "{found:?}"
        );
        let (_, _, found) = candidate(Absorbing::OrOfNotAnd);
        assert!(
            matches!(&found, Some(Simplification::Constant(value)) if value.is_all_ones()),
            "{found:?}"
        );
        let (x, y, found) = candidate(Absorbing::AndOfComplementOr);
        assert!(
            matches!(&found, Some(Simplification::Rewrite(op))
                if matches!(**op, SsaOp::And { left, right, .. } if left == x && right == y)),
            "{found:?}"
        );
        let (x, y, found) = candidate(Absorbing::OrOfComplementAnd);
        assert!(
            matches!(&found, Some(Simplification::Rewrite(op))
                if matches!(**op, SsaOp::Or { left, right, .. } if left == x && right == y)),
            "{found:?}"
        );
        let (_, _, found) = candidate(Absorbing::Control);
        assert!(found.is_none(), "{found:?}");
    }

    /// `x & x` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_and_self() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::And {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x & x should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// `x | x` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_or_self() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Or {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x | x should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// `x & -1` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_and_with_all_ones() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(-1))].into();
        let op: SsaOp<MockTarget> = SsaOp::And {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x & -1 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// `x | 0` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_or_with_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Or {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x | 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// `x ^ 0` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_xor_with_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Xor {
            dest,
            left: x,
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x ^ 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// `x << 0` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_shl_by_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Shl {
            dest,
            value: x,
            amount: SsaVarId::from_index(1),
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x << 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// **A leading-zero count shifted by its width's bits is the zero test.**
    /// ARM's `clz r0, r1; lsr r0, r0, #5` lifts as `(v == 0 ? 32 : 31 -
    /// bsr(v)) >> 5`, which is `zext(v == 0)`; shifted by any other amount it
    /// is a number of leading zeros, and stays.
    #[test]
    fn a_leading_zero_count_shifted_by_its_width_is_a_zero_test() {
        let folded = |shift: i32| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            let mut ids = Vec::new();
            builder
                .in_block(0, |block| {
                    let value = block.load_arg(mock_i32(), 0)?;
                    let zero = block.const_value(mock_i32(), ConstValue::I32(0))?;
                    let test = block.ceq(mock_i32(), value, zero)?;
                    let width = block.const_value(mock_i32(), ConstValue::I32(32))?;
                    let top = block.const_value(mock_i32(), ConstValue::I32(31))?;
                    let index = block.bit_scan_reverse(mock_i32(), value)?;
                    let below = block.sub(mock_i32(), top, index)?;
                    let count = block.select(mock_i32(), test, width, below)?;
                    let amount = block.const_value(mock_i32(), ConstValue::I32(shift))?;
                    let shifted = block.shr(mock_i32(), count, amount, true)?;
                    ids.extend([test, shifted]);
                    block.ret(Some(shifted))?;
                    Ok(())
                })
                .expect("build block");
            let mut ssa = builder.finish().expect("finish");
            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "zero test", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let (test, shifted) = (ids[0], ids[1]);
            ssa.recorded_definition(shifted)
                .map(|found| found.instruction().op().clone())
                .is_some_and(|op| matches!(op, SsaOp::IntConv { operand, unsigned: true, .. } if operand == test))
        };
        assert!(
            folded(5),
            "shifted by five, the count of 32 bits is the zero test"
        );
        assert!(
            !folded(4),
            "control: shifted by four, it is a count and stays"
        );
    }

    /// **A value shifted up and back down is the part it keeps.** Each shift
    /// pair at 64 bits computes, for every input, what it did before the
    /// pass, evaluated: RV64's zero-extension of a word (`<< 32 >> 32`), the
    /// word zero-extended and scaled by eight (`<< 32 >> 29`), a word and a
    /// byte sign-extended, the bits a longer right shift extracts
    /// (`<< 32 >> 35`), a 15-bit part and a bit kept by a mask, and the
    /// signed 15-bit part and bit, which no type or mask states and which
    /// stay shift pairs.
    #[test]
    fn a_value_shifted_up_and_back_down_is_the_part_it_keeps() {
        let inputs: [i64; 8] = [
            0,
            1,
            -1,
            0x1234_5678_9abc_def0,
            0x8000_0000,
            0x7fff_ffff,
            i64::MIN,
            0x0000_0001_8000_00ff,
        ];
        let cases: [(i64, i64, bool, bool); 10] = [
            (32, 32, true, true),
            (32, 29, true, true),
            (32, 32, false, true),
            (56, 56, false, true),
            (32, 35, true, true),
            (48, 56, false, true),
            (49, 49, true, true),
            (49, 49, false, false),
            (63, 63, true, true),
            (63, 63, false, false),
        ];
        for (left, right, unsigned, folds) in cases {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            let mut ids = Vec::new();
            builder
                .in_block(0, |block| {
                    let value = block.load_arg(mock_i64(), 0)?;
                    let up = block.const_value(mock_i64(), ConstValue::I64(left))?;
                    let shifted = block.shl(mock_i64(), value, up)?;
                    let down = block.const_value(mock_i64(), ConstValue::I64(right))?;
                    let part = block.shr(mock_i64(), shifted, down, unsigned)?;
                    ids.extend([value, part]);
                    block.ret(Some(part))?;
                    Ok(())
                })
                .expect("build block");
            let mut ssa = builder.finish().expect("finish");
            let (value, part) = (ids[0], ids[1]);
            let evaluate = |ssa: &SsaFunction<MockTarget>| {
                inputs
                    .iter()
                    .map(|input| {
                        let mut evaluator = ConstEvaluator::new(ssa, MACHINE);
                        evaluator.set_known(value, ConstValue::I64(*input));
                        evaluator
                            .evaluate_var(part)
                            .and_then(|got| got.integer_pattern(PointerSize::Bit64))
                    })
                    .collect::<Vec<_>>()
            };
            let before = evaluate(&ssa);
            assert!(before.iter().all(Option::is_some), "the oracle evaluates");
            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "shifted part", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let pair = ssa.recorded_definition(part).is_some_and(|found| {
                matches!(found.instruction().op(), SsaOp::Shr { value, .. }
                if ssa.recorded_definition(*value).is_some_and(|inner| {
                    matches!(inner.instruction().op(), SsaOp::Shl { .. })
                }))
            });
            assert_eq!(
                !pair, folds,
                "<< {left} >> {right} (unsigned {unsigned}): folded {}",
                !pair
            );
            assert_eq!(
                evaluate(&ssa),
                before,
                "<< {left} >> {right} (unsigned {unsigned}) computes what it did"
            );
        }
    }

    /// **Shifted parts fold beside the other rules of their block.** Two
    /// shift pairs and an addition of zero between them fold in one run, and
    /// the block computes what it did for every input: the pairs' steps are
    /// inserted, and a rule naming its instruction by position after one of
    /// them must still find its own.
    #[test]
    fn shifted_parts_fold_beside_the_other_rules_of_their_block() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let mut ids = Vec::new();
        builder
            .in_block(0, |block| {
                let value = block.load_arg(mock_i64(), 0)?;
                let up = block.const_value(mock_i64(), ConstValue::I64(32))?;
                let shifted = block.shl(mock_i64(), value, up)?;
                let down = block.const_value(mock_i64(), ConstValue::I64(29))?;
                let scaled = block.shr(mock_i64(), shifted, down, true)?;
                let zero = block.const_value(mock_i64(), ConstValue::I64(0))?;
                let same = block.add(mock_i64(), scaled, zero)?;
                let byte_up = block.const_value(mock_i64(), ConstValue::I64(56))?;
                let high = block.shl(mock_i64(), value, byte_up)?;
                let byte = block.shr(mock_i64(), high, byte_up, false)?;
                let joined = block.xor(mock_i64(), same, byte)?;
                ids.extend([value, joined]);
                block.ret(Some(joined))?;
                Ok(())
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let (value, joined) = (ids[0], ids[1]);
        let inputs: [i64; 5] = [0, -1, 0x1234_5678_9abc_def0, 0x8000_0080, i64::MIN];
        let evaluate = |ssa: &SsaFunction<MockTarget>| {
            inputs
                .iter()
                .map(|input| {
                    let mut evaluator = ConstEvaluator::new(ssa, MACHINE);
                    evaluator.set_known(value, ConstValue::I64(*input));
                    evaluator
                        .evaluate_var(joined)
                        .and_then(|got| got.integer_pattern(PointerSize::Bit64))
                })
                .collect::<Vec<_>>()
        };
        let before = evaluate(&ssa);
        assert!(before.iter().all(Option::is_some), "the oracle evaluates");
        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "shifted parts", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert_eq!(evaluate(&ssa), before, "the block computes what it did");
    }

    /// `x >> 0` is recognised as a copy of `x`.
    #[test]
    fn check_simplification_shr_by_zero() {
        let x = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> =
            [(SsaVarId::from_index(1), ConstValue::I32(0))].into();
        let op: SsaOp<MockTarget> = SsaOp::Shr {
            dest,
            value: x,
            amount: SsaVarId::from_index(1),
            unsigned: false,
            flags: None,
        };
        let result =
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE).simplification();
        let candidate = result.expect("x >> 0 should simplify to x");
        assert!(matches!(candidate.simplification, Simplification::Copy(v) if v == x));
    }

    /// A plain `x + y` with no constant operand matches no identity.
    #[test]
    fn check_simplification_no_match_for_unknown_op() {
        let x = SsaVarId::from_index(0);
        let y = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let constants: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: x,
            right: y,
            flags: None,
        };
        assert!(
            Site::new(&int_fn(), &op, 0, 0, &constants, &BTreeMap::new(), MACHINE)
                .simplification()
                .is_none()
        );
    }

    /// Several independent identities in one function are all simplified in a single run.
    #[test]
    fn run_simplifies_multiple_ops_in_single_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);

        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );
        ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(0, 3),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(0),
        }));
        // x + 0 -> x
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        // x - x -> 0
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Sub {
            dest: v3,
            left: v0,
            right: v0,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v3))));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        ssa.recompute_uses();
        let changed =
            run_mock_pass_boundary(&mut ssa, "multiple algebraic simplifications", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed);
        // Should have simplified both ops
        assert!(log.count_kind(EventKind::ConstantFolded) >= 2);
    }

    /// A chain of dependent constant operations folds in one run, every link
    /// reading the constant the one before it folds to: twenty shifts of `1`
    /// by one bit each are `1 << 20` after a single run. Folded a link per
    /// run, as when each operation saw only the constants the function held
    /// before the scan, the chain outlasts the normalizer's iteration cap.
    #[test]
    fn a_chain_of_constant_operations_folds_in_one_run() {
        const LINKS: usize = 20;
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut block = SsaBlock::new(0);
        let define = |ssa: &mut SsaFunction<MockTarget>, at: usize, ty: MockType| {
            ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::instruction(0, at), ty)
        };
        let one = define(&mut ssa, 0, MockType::I64);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: one,
            value: ConstValue::I64(1),
        }));
        let amount = define(&mut ssa, 1, MockType::I32);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: amount,
            value: ConstValue::I32(1),
        }));
        let mut current = one;
        for link in 0..LINKS {
            let dest = define(&mut ssa, link + 2, MockType::I64);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Shl {
                dest,
                value: current,
                amount,
                flags: None,
            }));
            current = dest;
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(current))));
        ssa.add_block(block);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = run_mock_pass_boundary(&mut ssa, "a constant chain", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed);
        let folded = ssa.find_constants();
        assert_eq!(folded.get(&current), Some(&ConstValue::I64(1 << LINKS)));
    }

    /// A block of only a `Const` and a `Return` is left untouched.
    #[test]
    fn run_with_no_simplifications_on_constant_only_block() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v0))));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        ssa.recompute_uses();
        let changed = run_mock_pass_boundary(&mut ssa, "constant-only algebraic no-op", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
        assert!(log.is_empty());
    }

    /// A function body built over its loaded arguments, returning the value the
    /// function returns.
    type Body = Box<dyn Fn(&mut SsaBlockBuilder<'_, MockTarget>, &[SsaVarId]) -> Result<SsaVarId>>;

    /// Whether the value a folded function returns is defined by the op the
    /// fold is meant to produce.
    type Folded = fn(&SsaOp<MockTarget>) -> bool;

    /// One shape the property test drives through the whole pass.
    struct Case {
        /// Names the shape in a failure message.
        name: String,
        /// The argument types, in order.
        args: Vec<MockType>,
        /// Builds the shape over the loaded arguments.
        body: Body,
        /// The op the returned value must be defined by after the pass, or
        /// `None` when the shape must merely keep its value.
        folds_to: Option<Folded>,
    }

    /// A constant of `ty` holding `value`, wrapped to the type's width.
    fn konst(ty: MockType, value: i64) -> ConstValue<MockTarget> {
        match ty {
            MockType::I8 => ConstValue::I8(value as i8),
            MockType::I16 => ConstValue::I16(value as i16),
            MockType::I32 => ConstValue::I32(value as i32),
            MockType::U8 => ConstValue::U8(value as u8),
            MockType::U16 => ConstValue::U16(value as u16),
            MockType::U32 => ConstValue::U32(value as u32),
            MockType::U64 => ConstValue::U64(value as u64),
            _ => ConstValue::I64(value),
        }
    }

    /// An integer conversion of a constant: extend from the source's width by
    /// `unsigned`, then wrap to `target`. `MockTarget` supplies no constant
    /// conversion of its own.
    fn convert(
        value: &ConstValue<MockTarget>,
        target: MockType,
        unsigned: bool,
    ) -> Option<ConstValue<MockTarget>> {
        // A comparison's boolean is the one-bit zero or one a carry extends.
        let (width, raw) = match value {
            ConstValue::True => (1, 1),
            ConstValue::False => (1, 0),
            _ => (
                value.bit_width(PointerSize::Bit64)?,
                match value.as_i64() {
                    Some(raw) => raw as u64,
                    None => value.as_u64()?,
                },
            ),
        };
        let shift = 64 - width;
        let extended = if unsigned {
            (raw << shift) >> shift
        } else {
            (((raw << shift) as i64) >> shift) as u64
        };
        Some(konst(target, extended as i64))
    }

    /// The bit width of an integer test type.
    fn bits(ty: MockType) -> u32 {
        MockTarget::bit_width(&ty).unwrap_or(0)
    }

    /// Every value of an 8-bit type; boundary values plus a deterministic
    /// pseudo-random spread for the wider ones.
    fn samples(ty: MockType) -> Vec<i64> {
        if ty == MockType::I8 {
            return (-128..128).collect();
        }
        let mut out = vec![
            0,
            1,
            -1,
            2,
            3,
            0x7f,
            0x80,
            0xff,
            0x100,
            0x7fff,
            0x8000,
            0xffff,
            0x1234_5678,
            0x8765_4321,
            i64::from(i32::MIN),
            i64::from(i32::MAX),
            0x0102_0304_0506_0708,
            i64::MIN,
            i64::MAX,
        ];
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..24 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            out.push(state as i64);
        }
        out
    }

    /// Every combination of samples for `args`.
    fn inputs(args: &[MockType]) -> Vec<Vec<ConstValue<MockTarget>>> {
        let mut out: Vec<Vec<ConstValue<MockTarget>>> = vec![Vec::new()];
        for ty in args {
            let values = samples(*ty);
            out = out
                .into_iter()
                .flat_map(|prefix| {
                    values.iter().map(move |value| {
                        let mut next = prefix.clone();
                        next.push(konst(*ty, *value));
                        next
                    })
                })
                .collect();
        }
        out
    }

    /// Builds a one-block function loading `args` and returning `body`'s value.
    fn build_case(case: &Case) -> SsaFunction<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(case.args.len(), 0);
        builder
            .in_block(0, |block| {
                let mut loaded = Vec::new();
                for (index, ty) in case.args.iter().enumerate() {
                    let index = u16::try_from(index).expect("argument index");
                    loaded.push(block.load_arg(SsaDefSpec::tmp(*ty), index)?);
                }
                let result = (case.body)(block, &loaded)?;
                block.ret(Some(result))
            })
            .expect("build block");
        builder.finish().expect("finish")
    }

    /// Runs a one-block function on concrete arguments, op by op, through the
    /// crate's own width-aware constant evaluator.
    fn interpret(
        ssa: &SsaFunction<MockTarget>,
        args: &[ConstValue<MockTarget>],
    ) -> Option<ConstValue<MockTarget>> {
        interpret_in(ssa, args, Endianness::Little)
    }

    /// Interprets `ssa` on `args` as [`interpret`] does, with a memory of
    /// `byte_order` that its indirect stores write and its indirect loads
    /// read, addresses being the integers their bases hold and a byte nothing
    /// wrote reading as zero.
    fn interpret_in(
        ssa: &SsaFunction<MockTarget>,
        args: &[ConstValue<MockTarget>],
        byte_order: Endianness,
    ) -> Option<ConstValue<MockTarget>> {
        let mut values: BTreeMap<SsaVarId, ConstValue<MockTarget>> = BTreeMap::new();
        let mut memory: BTreeMap<u64, u8> = BTreeMap::new();
        let address = |values: &BTreeMap<SsaVarId, ConstValue<MockTarget>>, var: SsaVarId| {
            values
                .get(&var)?
                .integer_pattern(PointerSize::Bit64)
                .map(|(bits, _)| bits)
        };
        for (_, _, instr) in ssa.iter_instructions() {
            match instr.op() {
                SsaOp::Nop => {}
                SsaOp::PtrAdd {
                    dest,
                    base,
                    index,
                    stride,
                    offset,
                    ..
                } => {
                    let scaled = match index {
                        Some(index) => address(&values, *index)?.wrapping_mul(*stride),
                        None => 0,
                    };
                    let at = address(&values, *base)?
                        .wrapping_add(scaled)
                        .wrapping_add_signed(*offset);
                    values.insert(*dest, ConstValue::U64(at));
                }
                SsaOp::PtrToInt {
                    dest,
                    operand,
                    target,
                } => {
                    let converted = convert(values.get(operand)?, *target, true)?;
                    values.insert(*dest, converted);
                }
                SsaOp::StoreIndirect {
                    addr,
                    value,
                    value_type,
                    ..
                } => {
                    let at = address(&values, *addr)?;
                    let bytes = values
                        .get(value)?
                        .to_bytes(byte_order, PointerSize::Bit64)?;
                    let width = usize::try_from(MockTarget::bit_width(value_type)? / 8).ok()?;
                    for (step, byte) in (0u64..).zip(bytes.iter().take(width)) {
                        memory.insert(at.wrapping_add(step), *byte);
                    }
                }
                SsaOp::LoadIndirect {
                    dest,
                    addr,
                    value_type,
                    ..
                } => {
                    let at = address(&values, *addr)?;
                    let width = u64::from(MockTarget::bit_width(value_type)? / 8);
                    let bytes: Vec<u8> = (0..width)
                        .map(|step| memory.get(&at.wrapping_add(step)).copied().unwrap_or(0))
                        .collect();
                    let value = ConstValue::from_typed_bytes(&bytes, value_type, byte_order)?;
                    values.insert(*dest, value);
                }
                SsaOp::LoadArg { dest, arg_index } => {
                    values.insert(*dest, args.get(usize::from(*arg_index))?.clone());
                }
                SsaOp::Return {
                    value: Some(value), ..
                } => return values.get(value).cloned(),
                SsaOp::Copy { dest, src } => {
                    let value = values.get(src)?.clone();
                    values.insert(*dest, value);
                }
                SsaOp::IntConv {
                    dest,
                    operand,
                    target,
                    unsigned,
                    ..
                } => {
                    let converted = convert(values.get(operand)?, *target, *unsigned)?;
                    values.insert(*dest, converted);
                }
                op => {
                    let folded = MACHINE.evaluate(op, Destination::of_operation(ssa, op), |var| {
                        values.get(&var).cloned()
                    });
                    if let (Some(dest), Some(value)) = (op.dest(), folded) {
                        values.insert(dest, value);
                    }
                }
            }
        }
        None
    }

    /// The op defining the value `ssa` returns.
    fn returned_op(ssa: &SsaFunction<MockTarget>) -> Option<&SsaOp<MockTarget>> {
        let SsaOp::Return {
            value: Some(value), ..
        } = ssa.block(0)?.control_terminator()?
        else {
            return None;
        };
        ssa.get_definition(*value)
    }

    /// Runs the pass over `case` and checks the function computes the same
    /// value before and after for every sampled input: the function after.
    fn assert_preserves(case: &Case) -> SsaFunction<MockTarget> {
        let before = build_case(case);
        let mut after = before.clone();
        let method = 0u32;
        run_mock_pass_boundary(&mut after, &case.name, |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        if let Some(folded) = case.folds_to {
            let op = returned_op(&after);
            assert!(
                op.is_some_and(folded),
                "{}: expected the fold, the result is defined by {op:?}\n{after}",
                case.name
            );
        }
        for input in inputs(&case.args) {
            let expected = interpret(&before, &input);
            assert!(
                expected.is_some(),
                "{}: the unfolded shape does not evaluate on {input:?}\n{before}",
                case.name
            );
            let actual = interpret(&after, &input);
            assert_eq!(
                expected, actual,
                "{}: input {input:?} changes value\nbefore:\n{before}\nafter:\n{after}",
                case.name
            );
        }
        after
    }

    /// `left op right` at `ty`, with a flags output nothing reads when `flags`.
    fn bin(
        block: &mut SsaBlockBuilder<'_, MockTarget>,
        ty: MockType,
        kind: BinaryOpKind,
        left: SsaVarId,
        right: SsaVarId,
        flags: bool,
    ) -> Result<SsaVarId> {
        if flags {
            let (value, _) = block.binary_op_with_flags(
                SsaDefSpec::tmp(ty),
                SsaDefSpec::tmp(ty),
                kind,
                left,
                right,
                false,
            )?;
            Ok(value)
        } else {
            block.binary_op(SsaDefSpec::tmp(ty), kind, left, right, false)
        }
    }

    /// Every MBA identity the fold knows, at `ty`, with and without a dead
    /// flags output on the top operation.
    fn mba_cases(ty: MockType) -> Vec<Case> {
        let mut cases = Vec::new();
        for flags in [false, true] {
            let d = move || SsaDefSpec::<MockTarget>::tmp(ty);
            let shapes: Vec<(&str, Body, Folded)> = vec![
                (
                    "(a ^ b) + ((a & b) << 1)",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let xor = block.xor(d(), a, b)?;
                        let and = block.and(d(), a, b)?;
                        let one = block.const_value(d(), konst(ty, 1))?;
                        let twice = block.shl(d(), and, one)?;
                        bin(block, ty, BinaryOpKind::Add, xor, twice, flags)
                    }),
                    |op| matches!(op, SsaOp::Add { .. }),
                ),
                (
                    "(a ^ b) + (a & b) * 2",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let xor = block.xor(d(), a, b)?;
                        let and = block.and(d(), a, b)?;
                        let two = block.const_value(d(), konst(ty, 2))?;
                        let twice = block.mul(d(), and, two)?;
                        bin(block, ty, BinaryOpKind::Add, twice, xor, flags)
                    }),
                    |op| matches!(op, SsaOp::Add { .. }),
                ),
                (
                    "(a ^ b) + ((a & b) + (a & b))",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let xor = block.xor(d(), a, b)?;
                        let and = block.and(d(), a, b)?;
                        let twice = block.add(d(), and, and)?;
                        bin(block, ty, BinaryOpKind::Add, xor, twice, flags)
                    }),
                    |op| matches!(op, SsaOp::Add { .. }),
                ),
                (
                    "(a | b) + (a & b)",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let or = block.or(d(), a, b)?;
                        let and = block.and(d(), b, a)?;
                        bin(block, ty, BinaryOpKind::Add, or, and, flags)
                    }),
                    |op| matches!(op, SsaOp::Add { .. }),
                ),
                (
                    "(a + ~b) + 1",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let not = block.not(d(), b)?;
                        let inner = block.add(d(), a, not)?;
                        let one = block.const_value(d(), konst(ty, 1))?;
                        bin(block, ty, BinaryOpKind::Add, inner, one, flags)
                    }),
                    |op| matches!(op, SsaOp::Sub { .. }),
                ),
                (
                    "(a | b) - (a & b)",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let or = block.or(d(), a, b)?;
                        let and = block.and(d(), a, b)?;
                        bin(block, ty, BinaryOpKind::Sub, or, and, flags)
                    }),
                    |op| matches!(op, SsaOp::Xor { .. }),
                ),
                (
                    "(a + b) - (a | b)",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let add = block.add(d(), a, b)?;
                        let or = block.or(d(), a, b)?;
                        bin(block, ty, BinaryOpKind::Sub, add, or, flags)
                    }),
                    |op| matches!(op, SsaOp::And { .. }),
                ),
                (
                    "(a & b) | (a ^ b)",
                    Box::new(move |block, args| {
                        let (a, b) = (args[0], args[1]);
                        let and = block.and(d(), a, b)?;
                        let xor = block.xor(d(), a, b)?;
                        bin(block, ty, BinaryOpKind::Or, and, xor, flags)
                    }),
                    |op| matches!(op, SsaOp::Or { left, right, .. } if left != right),
                ),
            ];
            for (name, body, folded) in shapes {
                cases.push(Case {
                    name: format!("{ty:?} flags={flags}: {name}"),
                    args: vec![ty, ty],
                    body,
                    folds_to: Some(folded),
                });
            }
        }
        cases
    }

    /// Every rotate spelling the fold knows, at `ty`.
    fn rotate_cases(ty: MockType) -> Vec<Case> {
        let width = bits(ty);
        let d = move || SsaDefSpec::<MockTarget>::tmp(ty);
        let amount = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let mut cases = Vec::new();
        for flags in [false, true] {
            for left_first in [true, false] {
                cases.push(Case {
                    name: format!(
                        "{ty:?} flags={flags}: masked-negation {}",
                        if left_first { "rol" } else { "ror" }
                    ),
                    args: vec![ty, MockType::I32],
                    body: Box::new(move |block, args| {
                        let (x, r) = (args[0], args[1]);
                        let mask = block.const_i32(amount(), (width - 1) as i32)?;
                        let masked = block.and(amount(), r, mask)?;
                        let negated = block.neg(amount(), r)?;
                        let negated = block.and(amount(), negated, mask)?;
                        let (by_left, by_right) = if left_first {
                            (masked, negated)
                        } else {
                            (negated, masked)
                        };
                        let shl = block.shl(d(), x, by_left)?;
                        let shr = block.shr(d(), x, by_right, true)?;
                        bin(block, ty, BinaryOpKind::Or, shl, shr, flags)
                    }),
                    folds_to: Some(|op| matches!(op, SsaOp::Rol { .. } | SsaOp::Ror { .. })),
                });
            }
            for k in 1..width {
                cases.push(Case {
                    name: format!("{ty:?} flags={flags}: (x << {k}) | (x >> {})", width - k),
                    args: vec![ty],
                    body: Box::new(move |block, args| {
                        let x = args[0];
                        let by_left = block.const_i32(amount(), k as i32)?;
                        let by_right = block.const_i32(amount(), (width - k) as i32)?;
                        let shl = block.shl(d(), x, by_left)?;
                        let shr = block.shr(d(), x, by_right, true)?;
                        bin(block, ty, BinaryOpKind::Or, shr, shl, flags)
                    }),
                    folds_to: Some(|op| matches!(op, SsaOp::Rol { .. } | SsaOp::BSwap { .. })),
                });
            }
        }
        cases
    }

    /// How one byte-reversal term is spelled.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum MaskAt {
        /// `(x shift n) & mask`, the mask naming the destination byte, omitted
        /// where the shift alone leaves one byte standing.
        AfterShift,
        /// `(x shift n) & mask` on every term, the top lane's mask included:
        /// that mask has the sign bit set, so it reads as a negative constant.
        AfterShiftEveryLane,
        /// `(x & mask) shift n`, the mask naming the source byte.
        BeforeShift,
    }

    /// The byte reversal of a `ty` value, one term per byte, or-ed together.
    fn bswap_body(ty: MockType, mask_at: MaskAt, flags: bool) -> Body {
        let d = move || SsaDefSpec::<MockTarget>::tmp(ty);
        let lanes = i64::from(bits(ty) / 8);
        Box::new(move |block, args| {
            let x = args[0];
            let mut acc: Option<SsaVarId> = None;
            for from in 0..lanes {
                let to = lanes - 1 - from;
                let distance = 8 * (to - from).abs();
                let by = block.const_i32(SsaDefSpec::tmp(MockType::I32), distance as i32)?;
                let edge = from == 0 || from == lanes - 1;
                let term = match mask_at {
                    // The compiler omits the mask where the shift alone leaves
                    // one byte standing.
                    MaskAt::AfterShift | MaskAt::AfterShiftEveryLane => {
                        let shifted = if to > from {
                            block.shl(d(), x, by)?
                        } else {
                            block.shr(d(), x, by, true)?
                        };
                        if edge && mask_at == MaskAt::AfterShift {
                            shifted
                        } else {
                            let mask = block.const_value(d(), konst(ty, 0xff << (8 * to)))?;
                            block.and(d(), shifted, mask)?
                        }
                    }
                    MaskAt::BeforeShift => {
                        let mask = block.const_value(d(), konst(ty, 0xff << (8 * from)))?;
                        let masked = block.and(d(), x, mask)?;
                        if to > from {
                            block.shl(d(), masked, by)?
                        } else {
                            block.shr(d(), masked, by, true)?
                        }
                    }
                };
                acc = Some(match acc {
                    None => term,
                    Some(prev) if from == lanes - 1 => {
                        bin(block, ty, BinaryOpKind::Or, prev, term, flags)?
                    }
                    Some(prev) => block.or(d(), prev, term)?,
                });
            }
            acc.ok_or_else(|| Error::new("no lanes"))
        })
    }

    /// Every byte-reversal spelling the fold knows.
    fn bswap_cases() -> Vec<Case> {
        let mut cases = Vec::new();
        for ty in [MockType::I16, MockType::I32, MockType::I64] {
            for mask_at in [
                MaskAt::AfterShift,
                MaskAt::AfterShiftEveryLane,
                MaskAt::BeforeShift,
            ] {
                for flags in [false, true] {
                    cases.push(Case {
                        name: format!("{ty:?} flags={flags}: bswap, mask {mask_at:?}"),
                        args: vec![ty],
                        body: bswap_body(ty, mask_at, flags),
                        folds_to: Some(|op| matches!(op, SsaOp::BSwap { .. })),
                    });
                }
            }
        }
        cases
    }

    /// Shapes with a width conversion between the idiom's parts. Each looks
    /// like an identity to a matcher that peels every conversion, and none is
    /// one: the conversion changes which bits the outer operation sees.
    fn conversion_cases() -> Vec<Case> {
        let i8d = || SsaDefSpec::<MockTarget>::tmp(MockType::I8);
        let i16d = || SsaDefSpec::<MockTarget>::tmp(MockType::I16);
        let i32d = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        let mut cases = vec![
            Case {
                name: "(u64)(a ^ b) + ((u64)(a & b) << 1) over I32".to_string(),
                args: vec![MockType::I32, MockType::I32],
                body: Box::new(move |block, args| {
                    let (a, b) = (args[0], args[1]);
                    let xor = block.xor(i32d(), a, b)?;
                    let and = block.and(i32d(), a, b)?;
                    let wide_xor = block.conv(i64d(), xor, MockType::I64, false, true)?;
                    let wide_and = block.conv(i64d(), and, MockType::I64, false, true)?;
                    let one = block.const_i64(i64d(), 1)?;
                    let twice = block.shl(i64d(), wide_and, one)?;
                    block.add(i64d(), wide_xor, twice)
                }),
                folds_to: None,
            },
            Case {
                name: "a32 + zext32(~b8) + 1".to_string(),
                args: vec![MockType::I32, MockType::I8],
                body: Box::new(move |block, args| {
                    let (a, b) = (args[0], args[1]);
                    let not = block.not(i8d(), b)?;
                    let wide = block.conv(i32d(), not, MockType::I32, false, true)?;
                    let inner = block.add(i32d(), a, wide)?;
                    let one = block.const_i32(i32d(), 1)?;
                    block.add(i32d(), inner, one)
                }),
                folds_to: None,
            },
            Case {
                name: "(x << 8) | (zext32(trunc8(x)) >> 24)".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let x = args[0];
                    let narrow = block.conv(i8d(), x, MockType::I8, false, false)?;
                    let wide = block.conv(i32d(), narrow, MockType::I32, false, true)?;
                    let c8 = block.const_i32(i32d(), 8)?;
                    let c24 = block.const_i32(i32d(), 24)?;
                    let shl = block.shl(i32d(), x, c8)?;
                    let shr = block.shr(i32d(), wide, c24, true)?;
                    block.or(i32d(), shl, shr)
                }),
                folds_to: None,
            },
            Case {
                name: "bswap chain with a term reading zext32(trunc16(x))".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let x = args[0];
                    let narrow = block.conv(i16d(), x, MockType::I16, false, false)?;
                    let wide = block.conv(i32d(), narrow, MockType::I32, false, true)?;
                    let c8 = block.const_i32(i32d(), 8)?;
                    let c24 = block.const_i32(i32d(), 24)?;
                    let top = block.shl(i32d(), x, c24)?;
                    let second = block.shl(i32d(), x, c8)?;
                    let m2 = block.const_i32(i32d(), 0x00ff_0000)?;
                    let second = block.and(i32d(), second, m2)?;
                    let third = block.shr(i32d(), wide, c8, true)?;
                    let m1 = block.const_i32(i32d(), 0x0000_ff00)?;
                    let third = block.and(i32d(), third, m1)?;
                    let low = block.shr(i32d(), x, c24, true)?;
                    let acc = block.or(i32d(), top, second)?;
                    let acc = block.or(i32d(), acc, third)?;
                    block.or(i32d(), acc, low)
                }),
                folds_to: None,
            },
            Case {
                name: "bswap chain through a truncated 64-bit or".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let x = args[0];
                    let c8 = block.const_i32(i32d(), 8)?;
                    let c24 = block.const_i32(i32d(), 24)?;
                    let top = block.shl(i32d(), x, c24)?;
                    let second = block.shl(i32d(), x, c8)?;
                    let m2 = block.const_i32(i32d(), 0x00ff_0000)?;
                    let second = block.and(i32d(), second, m2)?;
                    let third = block.shr(i32d(), x, c8, true)?;
                    let m1 = block.const_i32(i32d(), 0x0000_ff00)?;
                    let third = block.and(i32d(), third, m1)?;
                    let wide_third = block.conv(i64d(), third, MockType::I64, false, false)?;
                    let wide_x = block.conv(i64d(), x, MockType::I64, false, false)?;
                    let low = block.shr(i64d(), wide_x, c24, true)?;
                    let wide_or = block.or(i64d(), wide_third, low)?;
                    let narrow_or = block.conv(i32d(), wide_or, MockType::I32, false, false)?;
                    let acc = block.or(i32d(), top, second)?;
                    block.or(i32d(), acc, narrow_or)
                }),
                folds_to: None,
            },
        ];
        // A same-width conversion changes no bit, so the idiom still folds
        // through one.
        cases.push(Case {
            name: "(u32)(a ^ b) + (u32)((a & b) << 1) over I32".to_string(),
            args: vec![MockType::I32, MockType::I32],
            body: Box::new(move |block, args| {
                let (a, b) = (args[0], args[1]);
                let u32d = SsaDefSpec::<MockTarget>::tmp(MockType::U32);
                let xor = block.xor(i32d(), a, b)?;
                let and = block.and(i32d(), a, b)?;
                let one = block.const_i32(i32d(), 1)?;
                let twice = block.shl(i32d(), and, one)?;
                let xor = block.conv(u32d.clone(), xor, MockType::U32, false, false)?;
                let twice = block.conv(u32d, twice, MockType::U32, false, false)?;
                let xor = block.conv(i32d(), xor, MockType::I32, false, true)?;
                let twice = block.conv(i32d(), twice, MockType::I32, false, true)?;
                block.add(i32d(), xor, twice)
            }),
            folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
        });
        cases
    }

    /// Every MBA identity computes the same value after the pass as before,
    /// exhaustively over 8-bit operands and on a sample at 16, 32 and 64 bits.
    #[test]
    fn every_mba_identity_keeps_its_value() {
        for ty in [MockType::I8, MockType::I16, MockType::I32, MockType::I64] {
            for case in mba_cases(ty) {
                assert_preserves(&case);
            }
        }
    }

    /// Every rotate spelling computes the same value after the pass as before.
    #[test]
    fn every_rotate_spelling_keeps_its_value() {
        for ty in [MockType::I8, MockType::I16, MockType::I32, MockType::I64] {
            for case in rotate_cases(ty) {
                assert_preserves(&case);
            }
        }
    }

    /// How the high half of a double-width sum or difference adds its carry.
    #[derive(Clone, Copy)]
    enum Carried {
        /// `(ah + bh) + c` (x86 `adc`).
        After,
        /// `c + (ah + bh)` (MIPS `addu` after `sltu`).
        Before,
        /// `ah + (bh + c)`.
        Inside,
        /// `ah + c`: the second operand is a constant with no high half.
        Alone,
    }

    /// A function of two `u64` arguments returning their sum (or difference
    /// when `subtract`) computed a 32-bit half at a time and composed back:
    /// `carry` builds the carry or borrow from the low result and the operands'
    /// low halves, `carried` places it, and `compose` builds the composition
    /// from the low and the shifted high half.
    fn halves_case(
        name: &str,
        subtract: bool,
        carried: Carried,
        carry: fn(
            &mut SsaBlockBuilder<'_, MockTarget>,
            SsaVarId,
            SsaVarId,
            SsaVarId,
        ) -> Result<SsaVarId>,
        folds: bool,
    ) -> Case {
        let args = match carried {
            Carried::Alone => vec![MockType::U64],
            _ => vec![MockType::U64, MockType::U64],
        };
        Case {
            name: name.to_string(),
            args,
            body: Box::new(move |block, args| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let split = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                    let low = block.conv(u32d(), value, MockType::U32, false, true)?;
                    let shifted = block.shr(u64d(), value, shift, true)?;
                    let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                    Ok::<_, Error>((low, high))
                };
                let (al, ah) = split(block, args[0])?;
                let (bl, bh) = match args.get(1) {
                    Some(b) => {
                        let (bl, bh) = split(block, *b)?;
                        (bl, Some(bh))
                    }
                    None => (
                        block.const_value(u32d(), ConstValue::U32(0x8000_0005))?,
                        None,
                    ),
                };
                let low = if subtract {
                    block.sub(u32d(), al, bl)?
                } else {
                    block.add(u32d(), al, bl)?
                };
                let flag = carry(block, low, al, bl)?;
                let flag = block.conv(u32d(), flag, MockType::U32, false, true)?;
                let combine = |block: &mut SsaBlockBuilder<'_, MockTarget>, left, right| {
                    if subtract {
                        block.sub(u32d(), left, right)
                    } else {
                        block.add(u32d(), left, right)
                    }
                };
                let high = match (carried, bh) {
                    (Carried::After, Some(bh)) => {
                        let partial = combine(block, ah, bh)?;
                        combine(block, partial, flag)?
                    }
                    (Carried::Before, Some(bh)) if !subtract => {
                        let partial = block.add(u32d(), ah, bh)?;
                        block.add(u32d(), flag, partial)?
                    }
                    // `ah - b - bh`, the borrow subtracted first.
                    (Carried::Before, Some(bh)) => {
                        let partial = block.sub(u32d(), ah, flag)?;
                        block.sub(u32d(), partial, bh)?
                    }
                    (Carried::Inside, Some(bh)) => {
                        let partial = block.add(u32d(), bh, flag)?;
                        combine(block, ah, partial)?
                    }
                    _ => combine(block, ah, flag)?,
                };
                let low = block.conv(u64d(), low, MockType::U64, false, true)?;
                let high = block.conv(u64d(), high, MockType::U64, false, true)?;
                let high = block.shl(u64d(), high, shift)?;
                block.or(u64d(), high, low)
            }),
            folds_to: if folds {
                Some(if subtract {
                    |op| matches!(op, SsaOp::Sub { .. })
                } else {
                    |op| matches!(op, SsaOp::Add { .. })
                })
            } else {
                Some(|op| matches!(op, SsaOp::Or { .. }))
            },
        }
    }

    /// A double-width sum or difference computed in halves and composed back
    /// is the sum or difference, in every spelling of the carry and every
    /// association of the high half, and keeps its value on every sample. The
    /// controls: a signed carry, a carry against neither addend and a borrow
    /// the wrong way round are no carry, and the composition stays.
    #[test]
    fn a_double_width_sum_computed_in_halves_is_the_sum() {
        let lt = |block: &mut SsaBlockBuilder<'_, MockTarget>, low, al, _bl| {
            block.clt(SsaDefSpec::tmp(MockType::I1), low, al, true)
        };
        let lt_second = |block: &mut SsaBlockBuilder<'_, MockTarget>, low, _al, bl| {
            block.clt(SsaDefSpec::tmp(MockType::I1), low, bl, true)
        };
        let gt = |block: &mut SsaBlockBuilder<'_, MockTarget>, low, al, _bl| {
            block.cgt(SsaDefSpec::tmp(MockType::I1), al, low, true)
        };
        let signed = |block: &mut SsaBlockBuilder<'_, MockTarget>, low, al, _bl| {
            block.clt(SsaDefSpec::tmp(MockType::I1), low, al, false)
        };
        let borrow = |block: &mut SsaBlockBuilder<'_, MockTarget>, _low, al, bl| {
            block.clt(SsaDefSpec::tmp(MockType::I1), al, bl, true)
        };
        let borrow_gt = |block: &mut SsaBlockBuilder<'_, MockTarget>, _low, al, bl| {
            block.cgt(SsaDefSpec::tmp(MockType::I1), bl, al, true)
        };
        let backwards = |block: &mut SsaBlockBuilder<'_, MockTarget>, _low, al, bl| {
            block.clt(SsaDefSpec::tmp(MockType::I1), bl, al, true)
        };
        let unrelated = |block: &mut SsaBlockBuilder<'_, MockTarget>, low, _al, _bl| {
            let one = block.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(1))?;
            block.clt(SsaDefSpec::tmp(MockType::I1), low, one, true)
        };
        for case in [
            halves_case("adc", false, Carried::After, lt, true),
            halves_case("sltu then addu", false, Carried::Before, lt, true),
            halves_case("carry inside", false, Carried::Inside, lt, true),
            halves_case(
                "carry against the second addend",
                false,
                Carried::After,
                lt_second,
                true,
            ),
            halves_case("carry spelled cgt", false, Carried::After, gt, true),
            halves_case("constant addend", false, Carried::Alone, lt, true),
            halves_case("sbb", true, Carried::After, borrow, true),
            halves_case("borrow first", true, Carried::Before, borrow, true),
            halves_case("borrow inside", true, Carried::Inside, borrow, true),
            halves_case("borrow spelled cgt", true, Carried::After, borrow_gt, true),
            halves_case("constant subtrahend", true, Carried::Alone, borrow, true),
            halves_case("signed carry", false, Carried::After, signed, false),
            halves_case("unrelated carry", false, Carried::After, unrelated, false),
            halves_case("backwards borrow", true, Carried::After, backwards, false),
        ] {
            assert_preserves(&case);
        }
    }

    /// How a double-width sum adds halves apart ([`regroup_case`]).
    #[derive(Clone, Copy, Debug)]
    enum Apart {
        /// `((vh + wh) << 32 | xl) + vl + wl`: two values' halves added.
        Added,
        /// `((xh - vh) << 32 | xl) - vl`: a value's halves subtracted.
        Subtracted,
        /// `(vh << 32) + vl + x`: a high half shifted alone.
        Alone,
        /// `((xh - vh) << 32 | xl) + vl`: the high half subtracted, the low
        /// one added, which are no value.
        OppositeSigns,
        /// `((xh + wh) << 32 | xl) + vl`: halves of two values.
        TwoValues,
        /// `(sext16(vh) << 32 | xl) + vl`: a high half narrower than a half,
        /// its extension placed above it.
        Narrow,
    }

    /// A function of three `u64` arguments `x`, `v` and `w` returning a sum
    /// that adds halves apart as `shape` spells it.
    fn regroup_case(shape: Apart) -> Case {
        Case {
            name: format!("{shape:?}"),
            args: vec![MockType::U64, MockType::U64, MockType::U64],
            body: Box::new(move |block, args| {
                let u16d = || SsaDefSpec::tmp(MockType::U16);
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let split = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                    let low = block.conv(u32d(), value, MockType::U32, false, true)?;
                    let shifted = block.shr(u64d(), value, shift, true)?;
                    let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                    Ok::<_, Error>((low, high))
                };
                let widen = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                    block.conv(u64d(), value, MockType::U64, false, true)
                };
                let (xl, xh) = split(block, args[0])?;
                let (vl, vh) = split(block, args[1])?;
                let (_, wh) = split(block, args[2])?;
                let wl = block.conv(u32d(), args[2], MockType::U32, false, true)?;
                let compose = |block: &mut SsaBlockBuilder<'_, MockTarget>, high| {
                    let high = widen(block, high)?;
                    let high = block.shl(u64d(), high, shift)?;
                    let low = widen(block, xl)?;
                    block.or(u64d(), high, low)
                };
                let low = widen(block, vl)?;
                match shape {
                    Apart::Added => {
                        let high = block.add(u32d(), vh, wh)?;
                        let composed = compose(block, high)?;
                        let sum = block.add(u64d(), composed, low)?;
                        let other = widen(block, wl)?;
                        block.add(u64d(), sum, other)
                    }
                    Apart::Subtracted => {
                        let high = block.sub(u32d(), xh, vh)?;
                        let composed = compose(block, high)?;
                        block.sub(u64d(), composed, low)
                    }
                    Apart::Alone => {
                        let high = widen(block, vh)?;
                        let high = block.shl(u64d(), high, shift)?;
                        let sum = block.add(u64d(), high, low)?;
                        block.add(u64d(), sum, args[0])
                    }
                    Apart::OppositeSigns => {
                        let high = block.sub(u32d(), xh, vh)?;
                        let composed = compose(block, high)?;
                        block.add(u64d(), composed, low)
                    }
                    Apart::TwoValues => {
                        let high = block.add(u32d(), xh, wh)?;
                        let composed = compose(block, high)?;
                        block.add(u64d(), composed, low)
                    }
                    Apart::Narrow => {
                        let high = block.conv(u16d(), vh, MockType::U16, false, true)?;
                        let high = block.conv(u64d(), high, MockType::U64, false, false)?;
                        let high = block.shl(u64d(), high, shift)?;
                        let low_x = widen(block, xl)?;
                        let composed = block.or(u64d(), high, low_x)?;
                        block.add(u64d(), composed, low)
                    }
                }
            }),
            folds_to: Some(|op| matches!(op, SsaOp::Add { .. } | SsaOp::Sub { .. })),
        }
    }

    /// The arguments a function's returned sum adds or subtracts whole: the
    /// indices of the arguments among the terms its additions, subtractions
    /// and copies at the result's width reach.
    fn summed_arguments(ssa: &SsaFunction<MockTarget>) -> Vec<u16> {
        let Some(SsaOp::Return {
            value: Some(value), ..
        }) = ssa.block(0).and_then(|block| block.control_terminator())
        else {
            return Vec::new();
        };
        let mut pending = vec![*value];
        let mut seen = BTreeSet::new();
        let mut arguments = Vec::new();
        while let Some(value) = pending.pop() {
            if !seen.insert(value) {
                continue;
            }
            match ssa.get_definition(value) {
                Some(SsaOp::Add { left, right, .. } | SsaOp::Sub { left, right, .. }) => {
                    pending.extend([*left, *right]);
                }
                Some(SsaOp::Copy { src, .. }) => pending.push(*src),
                Some(SsaOp::LoadArg { arg_index, .. }) => arguments.push(*arg_index),
                _ => {}
            }
        }
        arguments.sort_unstable();
        arguments
    }

    /// **A double-width sum adding a value's halves apart adds the value.** A
    /// sum in halves folded before its operands' halves were known to be one
    /// value's (a callee's two results later found one integer) adds each
    /// low half zero-extended and collects each high half in the high half it
    /// shifts into place; the two of one value, added or subtracted alike, are
    /// the value. Every shape keeps its value on every sample. The controls
    /// pair nothing: a high half of the other sign, halves of two values, and
    /// a high half narrower than a half, whose extension lands inside the sum.
    #[test]
    fn a_double_width_sum_adding_a_values_halves_apart_adds_the_value() {
        for (shape, whole) in [
            (Apart::Added, vec![1, 2]),
            (Apart::Subtracted, vec![1]),
            (Apart::Alone, vec![0, 1]),
            (Apart::OppositeSigns, vec![]),
            (Apart::TwoValues, vec![]),
            (Apart::Narrow, vec![]),
        ] {
            let after = assert_preserves(&regroup_case(shape));
            assert_eq!(
                summed_arguments(&after),
                whole,
                "{shape:?}: the arguments the sum reads whole\n{after}"
            );
        }
    }

    /// **A composition built from an untyped low half states its width.** A
    /// sum in halves whose first operand's low half has no type yet (an
    /// argument slot before typing) is composed with that half read at the
    /// half width, its high half summed as a half-wide integer, so the
    /// composition reads as two halves to every later fold
    /// ([`doubleword::halves`]), as one regrouping the sum once its operands'
    /// halves are known to be one value's.
    #[test]
    fn a_composition_of_an_untyped_low_half_reads_as_halves() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
        builder
            .in_block(0, |block| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let al = block.load_arg(SsaDefSpec::tmp(MockType::Unknown), 0)?;
                let ah = block.load_arg(u32d(), 1)?;
                let b = block.load_arg(u64d(), 2)?;
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let bl = block.conv(u32d(), b, MockType::U32, false, true)?;
                let shifted = block.shr(u64d(), b, shift, true)?;
                let bh = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                let low = block.add(u32d(), al, bl)?;
                let carry = block.clt(SsaDefSpec::tmp(MockType::I1), low, bl, true)?;
                let carry = block.conv(u32d(), carry, MockType::U32, false, true)?;
                let high = block.add(u32d(), ah, bh)?;
                let high = block.add(u32d(), high, carry)?;
                let low = block.conv(u64d(), low, MockType::U64, false, true)?;
                let high = block.conv(u64d(), high, MockType::U64, false, true)?;
                let high = block.shl(u64d(), high, shift)?;
                let composed = block.or(u64d(), high, low)?;
                block.ret(Some(composed))
            })
            .expect("build block");
        let mut ssa = builder.finish().expect("finish");
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "untyped low half", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let constants = ssa.find_constants();
        let compositions: Vec<_> = ssa
            .iter_instructions()
            .filter(|(_, _, instruction)| matches!(instruction.op(), SsaOp::Or { .. }))
            .map(|(_, _, instruction)| {
                doubleword::halves(&ssa, instruction.op(), &constants).is_some()
            })
            .collect();
        assert!(
            !compositions.is_empty() && compositions.iter().all(|reads| *reads),
            "every composition the fold built reads as two halves\n{ssa}"
        );
    }

    /// Builds the low halves' borrow of a comparison in halves from the first
    /// value's low half and the constant's ([`comparison_case`]).
    type BorrowOf =
        fn(&mut SsaBlockBuilder<'_, MockTarget>, SsaVarId, SsaVarId) -> Result<SsaVarId>;

    /// How the high halves of a comparison in halves take the low halves'
    /// borrow ([`comparison_case`]).
    #[derive(Clone, Copy)]
    enum Borrowed {
        /// `ah <u b`, the second's high half zero.
        Alone,
        /// `ah <u b + bh`, a constant high half.
        Added(u32),
    }

    /// A function of one `u64` argument comparing it, a 32-bit half at a
    /// time, with `constant`: the low halves' borrow (`borrow`), then the high
    /// halves with it (`borrowed`), unsigned or signed (`unsigned`). `apart`
    /// takes the high half from a second argument, so the halves are two
    /// values'.
    fn comparison_case(
        name: &str,
        constant: u64,
        borrowed: Borrowed,
        unsigned: bool,
        borrow: BorrowOf,
        apart: bool,
    ) -> Case {
        Case {
            name: name.to_string(),
            args: vec![MockType::U64, MockType::U64],
            body: Box::new(move |block, args| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let low = block.conv(u32d(), args[0], MockType::U32, false, true)?;
                let shifted =
                    block.shr(u64d(), if apart { args[1] } else { args[0] }, shift, true)?;
                let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                let constant_low = block.const_value(u32d(), ConstValue::U32(constant as u32))?;
                let flag = borrow(block, low, constant_low)?;
                let flag = block.conv(u32d(), flag, MockType::U32, false, true)?;
                let against = match borrowed {
                    Borrowed::Alone => flag,
                    Borrowed::Added(constant_high) => {
                        let constant_high =
                            block.const_value(u32d(), ConstValue::U32(constant_high))?;
                        block.add(u32d(), flag, constant_high)?
                    }
                };
                block.clt(SsaDefSpec::tmp(MockType::I1), high, against, unsigned)
            }),
            folds_to: Some(|op| matches!(op, SsaOp::Clt { .. })),
        }
    }

    /// **A comparison of a double-width value computed in halves is the
    /// comparison of the value.** The low halves' borrow, then the high halves
    /// against it and the second's high half, is `a < b` over the whole width
    /// (`dash`'s `number`, `n > INT_MAX` on ARM as `(uint32_t)(n >> 32) >=
    /// (uint32_t)n <u 0x80000000`): unsigned and signed, the borrow spelled
    /// `cgt` or negated twice, the constant's high half zero or added. Every
    /// shape keeps its value on every sample; the ones that fold read the
    /// argument whole. The controls fold nothing: a high half whose addition
    /// of the borrow may wrap, halves of two values, a borrow compared signed.
    #[test]
    fn a_comparison_in_halves_is_the_comparison_of_the_value() {
        let lt: BorrowOf =
            |block, low, constant| block.clt(SsaDefSpec::tmp(MockType::I1), low, constant, true);
        let gt: BorrowOf =
            |block, low, constant| block.cgt(SsaDefSpec::tmp(MockType::I1), constant, low, true);
        let negated_twice: BorrowOf = |block, low, constant| {
            let flag = block.clt(SsaDefSpec::tmp(MockType::I1), low, constant, true)?;
            let flag = block.bool_not(SsaDefSpec::tmp(MockType::I1), flag)?;
            block.bool_not(SsaDefSpec::tmp(MockType::I1), flag)
        };
        let signed: BorrowOf =
            |block, low, constant| block.clt(SsaDefSpec::tmp(MockType::I1), low, constant, false);
        let cases = [
            (
                "n <u 0x80000000",
                0x8000_0000,
                Borrowed::Alone,
                true,
                lt,
                false,
                true,
            ),
            (
                "borrow spelled cgt",
                0x8000_0000,
                Borrowed::Alone,
                true,
                gt,
                false,
                true,
            ),
            (
                "borrow negated twice",
                0x8000_0000,
                Borrowed::Alone,
                true,
                negated_twice,
                false,
                true,
            ),
            (
                "a constant high half",
                0x0000_0005_0000_0010,
                Borrowed::Added(5),
                true,
                lt,
                false,
                true,
            ),
            (
                "signed",
                0x0000_0002_8000_0000,
                Borrowed::Added(2),
                false,
                lt,
                false,
                true,
            ),
            (
                "a high half that may wrap",
                0xffff_ffff_0000_0010,
                Borrowed::Added(u32::MAX),
                true,
                lt,
                false,
                false,
            ),
            (
                "halves of two values",
                0x8000_0000,
                Borrowed::Alone,
                true,
                lt,
                true,
                false,
            ),
            (
                "a borrow compared signed",
                0x8000_0000,
                Borrowed::Alone,
                true,
                signed,
                false,
                false,
            ),
        ];
        for (name, constant, borrowed, unsigned, borrow, apart, folds) in cases {
            let case = comparison_case(name, constant, borrowed, unsigned, borrow, apart);
            assert_preserves(&case);
            let mut after = build_case(&case);
            run_mock_pass_boundary(&mut after, name, |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let whole = match returned_op(&after) {
                Some(SsaOp::Clt { left, right, .. }) => {
                    after.bit_width(*left) == Some(64) && after.bit_width(*right) == Some(64)
                }
                _ => false,
            };
            assert_eq!(whole, folds, "{name}\n{after}");
        }
    }

    /// A sum carried onto a double-width value already summed in halves,
    /// `v = a + b; v += c` with `c` a word, is the sum of the three: every
    /// composition folds away, none left composing a partial sum with a stray
    /// carry, and the value is kept on every sample.
    #[test]
    fn a_chain_of_double_width_sums_is_one_sum() {
        let case = Case {
            name: "a + b + c".to_string(),
            args: vec![MockType::U64, MockType::U64, MockType::U32],
            body: Box::new(|block, args| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let flag = || SsaDefSpec::tmp(MockType::I1);
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let split = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                    let low = block.conv(u32d(), value, MockType::U32, false, true)?;
                    let shifted = block.shr(u64d(), value, shift, true)?;
                    let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                    Ok::<_, Error>((low, high))
                };
                let (al, ah) = split(block, args[0])?;
                let (bl, bh) = split(block, args[1])?;
                let low = block.add(u32d(), al, bl)?;
                let carry = block.clt(flag(), low, al, true)?;
                let carry = block.conv(u32d(), carry, MockType::U32, false, true)?;
                let high = block.add(u32d(), ah, bh)?;
                let high = block.add(u32d(), high, carry)?;
                let next = block.add(u32d(), low, args[2])?;
                let carry = block.clt(flag(), next, low, true)?;
                let carry = block.conv(u32d(), carry, MockType::U32, false, true)?;
                let high = block.add(u32d(), high, carry)?;
                let next = block.conv(u64d(), next, MockType::U64, false, true)?;
                let high = block.conv(u64d(), high, MockType::U64, false, true)?;
                let high = block.shl(u64d(), high, shift)?;
                block.or(u64d(), high, next)
            }),
            folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
        };
        let mut folded = build_case(&case);
        let method = 0u32;
        // To its fixpoint, as the pipeline runs it: each round folds the
        // compositions the round before built.
        for _ in 0..8 {
            if !run_mock_pass_boundary(&mut folded, &case.name, |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            }) {
                break;
            }
        }
        let compositions = folded
            .iter_instructions()
            .filter(|(_, _, instruction)| matches!(instruction.op(), SsaOp::Or { .. }))
            .count();
        assert_eq!(compositions, 0, "{folded}");
        assert_preserves(&case);
    }

    /// A chain of double-width sums and differences whose high half
    /// reassociation flattened into one sum, `a + b - c + d` with `b` a
    /// sign-extended word and `c` a zero-extended one, is the chain: the high
    /// half's terms, in no association the steps had, are each step's high
    /// half and each step's carry or borrow, and the fold reads them as such.
    /// Every composition folds away and the value is kept on every sample;
    /// the control, a carry compared against no addend of its step, folds
    /// nothing and keeps the value.
    #[test]
    fn a_flattened_chain_of_double_width_sums_is_the_chain() {
        let chain = |name: &str, related: bool| Case {
            name: name.to_string(),
            args: vec![MockType::U64, MockType::U32, MockType::U64],
            body: Box::new(move |block, args| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let flag = || SsaDefSpec::tmp(MockType::I1);
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let sign = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(31))?;
                let split = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                    let low = block.conv(u32d(), value, MockType::U32, false, true)?;
                    let shifted = block.shr(u64d(), value, shift, true)?;
                    let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                    Ok::<_, Error>((low, high))
                };
                let carry = |block: &mut SsaBlockBuilder<'_, MockTarget>, below, than| {
                    let flag = block.clt(flag(), below, than, true)?;
                    block.conv(u32d(), flag, MockType::U32, false, true)
                };
                let (al, ah) = split(block, args[0])?;
                let bl = args[1];
                let bh = block.shr(u32d(), bl, sign, false)?;
                let (dl, dh) = split(block, args[2])?;
                let cl = block.xor(u32d(), al, dl)?;
                let first = block.add(u32d(), al, bl)?;
                let c1 = carry(block, first, if related { al } else { cl })?;
                let second = block.sub(u32d(), first, cl)?;
                let b2 = carry(block, first, cl)?;
                let third = block.add(u32d(), second, dl)?;
                let c3 = carry(block, third, dl)?;
                // ((c3 + bh) - b2) + ((dh + c1) + ah), no step's own shape.
                let x = block.add(u32d(), c3, bh)?;
                let x = block.sub(u32d(), x, b2)?;
                let y = block.add(u32d(), dh, c1)?;
                let y = block.add(u32d(), y, ah)?;
                let high = block.add(u32d(), x, y)?;
                let low = block.conv(u64d(), third, MockType::U64, false, true)?;
                let high = block.conv(u64d(), high, MockType::U64, false, true)?;
                let high = block.shl(u64d(), high, shift)?;
                block.or(u64d(), high, low)
            }),
            folds_to: related.then_some(|op: &SsaOp<MockTarget>| matches!(op, SsaOp::Add { .. })),
        };
        let compositions = |case: &Case| {
            let mut folded = build_case(case);
            let method = 0u32;
            for _ in 0..8 {
                if !run_mock_pass_boundary(&mut folded, &case.name, |ssa| {
                    AlgebraicSimplificationPass.run(
                        ssa,
                        &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
                    ) == Ok(true)
                }) {
                    break;
                }
            }
            folded
                .iter_instructions()
                .filter(|(_, _, instruction)| matches!(instruction.op(), SsaOp::Or { .. }))
                .count()
        };
        let folded = chain("a + b - c + d", true);
        assert_preserves(&folded);
        assert_eq!(compositions(&folded), 0, "every composition folds");
        let control = chain("a carry of no addend", false);
        assert_preserves(&control);
        assert_eq!(compositions(&control), 1, "the composition stays");
    }

    /// **A half of a join is read back whatever the other half is.** `(u64)
    /// (lo | (u128) b << 64)` is `lo` and `(u64) ((lo | (u128) b << 64) >>
    /// 64)` is `b`, where `lo` is an argument zero-extended and where it is a
    /// 128-bit constant, whose zero high half merges with `b`'s field as any
    /// zero does: x86's `bndmov` stores a bound pair so, and loads it back.
    #[test]
    fn a_half_of_a_wide_join_is_read_back_over_a_constant_half() {
        for constant in [false, true] {
            let case = Case {
                name: "(u64) (lo | (u128) b << 64), (u64) ((...) >> 64)".to_string(),
                args: vec![MockType::U64, MockType::U64],
                body: Box::new(move |block, args| {
                    let wide = || SsaDefSpec::tmp(MockType::U128);
                    let u64d = || SsaDefSpec::tmp(MockType::U64);
                    let shift =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(64))?;
                    let low = if constant {
                        block.const_value(wide(), ConstValue::of_lane(128, 4096).unwrap())?
                    } else {
                        block.conv(wide(), args[0], MockType::U128, false, true)?
                    };
                    let high = block.conv(wide(), args[1], MockType::U128, false, true)?;
                    let high = block.shl(wide(), high, shift)?;
                    let whole = block.or(wide(), low, high)?;
                    let read_low = block.conv(u64d(), whole, MockType::U64, false, true)?;
                    let down = block.shr(wide(), whole, shift, true)?;
                    let read_high = block.conv(u64d(), down, MockType::U64, false, true)?;
                    block.xor(u64d(), read_low, read_high)
                }),
                folds_to: None,
            };
            let mut after = build_case(&case);
            run_mock_pass_boundary(&mut after, &case.name, |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let SsaOp::Xor { left, right, .. } = after
                .iter_instructions()
                .find_map(|(_, _, instruction)| {
                    matches!(instruction.op(), SsaOp::Xor { .. }).then(|| instruction.op().clone())
                })
                .expect("the xor")
            else {
                unreachable!()
            };
            let resolved = |value: SsaVarId| match after.get_definition(value) {
                Some(SsaOp::Copy { src, .. }) => Some(Err(*src)),
                Some(SsaOp::Const { value, .. }) => Some(Ok(value.clone())),
                _ => None,
            };
            let argument = |index: u16| {
                after
                    .iter_instructions()
                    .find_map(|(_, _, instruction)| match instruction.op() {
                        SsaOp::LoadArg { dest, arg_index } if *arg_index == index => Some(*dest),
                        _ => None,
                    })
                    .expect("the argument")
            };
            let low = if constant {
                Ok(ConstValue::U64(4096))
            } else {
                Err(argument(0))
            };
            assert_eq!(resolved(left), Some(low), "{after}");
            assert_eq!(resolved(right), Some(Err(argument(1))), "{after}");
        }
    }

    /// A half of a double-width sum read apart as well, `(int32_t)a` beside
    /// `a = b + c` on a 32-bit target, is read as that half of the sum once the
    /// composition is the sum: a reader the sum dominates reads its truncation,
    /// so the low half's own arithmetic is left to readers before it, and the
    /// value is kept on every sample.
    #[test]
    fn a_half_read_apart_is_read_from_the_sum() {
        let case = Case {
            name: "(a + b) ^ (uint32_t)(a + b)".to_string(),
            args: vec![MockType::U64, MockType::U64],
            body: Box::new(|block, args| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let split = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                    let low = block.conv(u32d(), value, MockType::U32, false, true)?;
                    let shifted = block.shr(u64d(), value, shift, true)?;
                    let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                    Ok::<_, Error>((low, high))
                };
                let (al, ah) = split(block, args[0])?;
                let (bl, bh) = split(block, args[1])?;
                let low = block.add(u32d(), al, bl)?;
                let carry = block.clt(SsaDefSpec::tmp(MockType::I1), low, al, true)?;
                let carry = block.conv(u32d(), carry, MockType::U32, false, true)?;
                let high = block.add(u32d(), ah, bh)?;
                let high = block.add(u32d(), high, carry)?;
                let wide_low = block.conv(u64d(), low, MockType::U64, false, true)?;
                let wide_high = block.conv(u64d(), high, MockType::U64, false, true)?;
                let wide_high = block.shl(u64d(), wide_high, shift)?;
                let whole = block.or(u64d(), wide_high, wide_low)?;
                let apart = block.conv(u64d(), low, MockType::U64, false, true)?;
                block.xor(u64d(), whole, apart)
            }),
            folds_to: None,
        };
        let readers_of_the_low_sum = |ssa: &SsaFunction<MockTarget>| {
            let low = ssa
                .iter_instructions()
                .find_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::Add { dest, .. }
                        if ssa
                            .variable(*dest)
                            .is_some_and(|variable| *variable.var_type() == MockType::U32) =>
                    {
                        Some(*dest)
                    }
                    _ => None,
                })
                .expect("the low sum");
            ssa.iter_instructions()
                .filter(|(_, _, instruction)| {
                    matches!(instruction.op(), SsaOp::IntConv { operand, .. } if *operand == low)
                })
                .count()
        };
        let before = build_case(&case);
        let mut after = before.clone();
        run_mock_pass_boundary(&mut after, &case.name, |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert_eq!(readers_of_the_low_sum(&before), 2, "{before}");
        assert_eq!(
            readers_of_the_low_sum(&after),
            1,
            "only the composition's own widening, dead now, reads it:\n{after}"
        );
        assert_preserves(&case);
    }

    /// A store is a half of one pair at most. Three adjacent words `x`, its
    /// sign and its sign's sign each pair with the next as a value with its
    /// sign above it, and the middle store with either neighbour; merged with
    /// both, the second merge would remove the store the first put in the
    /// middle one's place and lose the first word. One pair is merged, and
    /// what the three words read back is the same on every sample. Two
    /// constants stored to adjacent words, an `int` array's initialisation,
    /// are two values and stay two stores; read back whole, as a `long long`
    /// zeroed a word at a time is, they are one, and one store.
    #[test]
    fn a_store_is_a_half_of_one_pair() {
        let build = |constant: bool, whole: bool| -> SsaFunction<MockTarget> {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            builder
                .in_block(0, |block| {
                    let u32d = || SsaDefSpec::tmp(MockType::U32);
                    let u64d = || SsaDefSpec::tmp(MockType::U64);
                    let x = block.load_arg(SsaDefSpec::tmp(MockType::I32), 0)?;
                    let fill =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(31))?;
                    let words = if constant {
                        vec![
                            block.const_value(u32d(), ConstValue::U32(1))?,
                            block.const_value(u32d(), ConstValue::U32(2))?,
                        ]
                    } else {
                        let sign = block.shr(SsaDefSpec::tmp(MockType::I32), x, fill, false)?;
                        let again = block.shr(SsaDefSpec::tmp(MockType::I32), sign, fill, false)?;
                        vec![x, sign, again]
                    };
                    let base = block.const_value(u64d(), ConstValue::U64(0x1000))?;
                    let mut last = base;
                    for (offset, word) in (0i64..).step_by(4).zip(&words) {
                        let at = block.emit_def(u64d(), |dest| SsaOp::PtrAdd {
                            dest,
                            base,
                            index: None,
                            stride: 1,
                            offset,
                            result_type: MockType::U64,
                        })?;
                        let ty = if constant {
                            MockType::U32
                        } else {
                            MockType::I32
                        };
                        block.store_indirect(at, *word, ty)?;
                        last = at;
                    }
                    let first = if whole {
                        block.load_indirect(u64d(), base, MockType::U64)?
                    } else {
                        let word = block.load_indirect(u32d(), base, MockType::U32)?;
                        block.conv(u64d(), word, MockType::U64, false, true)?
                    };
                    let tail = block.load_indirect(u32d(), last, MockType::U32)?;
                    let tail = block.conv(u64d(), tail, MockType::U64, false, true)?;
                    let read = block.xor(u64d(), first, tail)?;
                    block.ret(Some(read))
                })
                .expect("the case builds");
            builder.finish().expect("the case is valid")
        };
        let widths = |ssa: &SsaFunction<MockTarget>| -> Vec<u32> {
            ssa.iter_instructions()
                .filter_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::StoreIndirect { value_type, .. } => MockTarget::bit_width(value_type),
                    _ => None,
                })
                .collect()
        };
        for (constant, whole) in [(false, true), (true, false), (true, true)] {
            let before = build(constant, whole);
            let mut after = before.clone();
            run_mock_pass_boundary(&mut after, "a run of words", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(
                        &0u32,
                        &NullListener,
                        AnalysisContext::new(Machine {
                            byte_order: Some(Endianness::Little),
                            ..MACHINE
                        }),
                    ),
                ) == Ok(true)
            });
            let expected = match (constant, whole) {
                (true, false) => vec![32, 32],
                (true, true) => vec![64],
                (false, _) => vec![64, 32],
            };
            assert_eq!(widths(&after), expected, "{after}");
            for input in inputs(&[MockType::I32]) {
                assert_eq!(
                    interpret_in(&before, &input, Endianness::Little),
                    interpret_in(&after, &input, Endianness::Little),
                    "{input:?}:\nbefore:\n{before}\nafter:\n{after}"
                );
            }
        }
    }

    /// Two stores of a double-width sum's halves to the two words of one cell,
    /// `*p = a + b` on a 32-bit target, are one store of the sum: the low half
    /// in the word the byte order puts it in, whichever is stored first. What
    /// a load of the cell reads back is the same on every sample, in either
    /// byte order, and an access between the two that misses the earlier
    /// store's word changes nothing. The controls keep both stores: a load of
    /// the earlier store's word between them, the access elsewhere in a block a
    /// handler protects (a fault there reaches a handler that reads memory),
    /// words not adjacent, halves no composition reduces, and the halves in
    /// the words the other byte order puts them in.
    #[test]
    fn a_double_width_value_stored_as_its_halves_is_one_store() {
        #[derive(Clone, Copy, PartialEq)]
        enum Shape {
            Paired,
            Elsewhere,
            Protected,
            LoadBetween,
            Apart,
            Unrelated,
            HighFirst,
        }
        let build = |shape: Shape, order: Endianness| -> SsaFunction<MockTarget> {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
            builder
                .in_block(0, |block| {
                    let u32d = || SsaDefSpec::tmp(MockType::U32);
                    let u64d = || SsaDefSpec::tmp(MockType::U64);
                    let a = block.load_arg(u64d(), 0)?;
                    let b = block.load_arg(u64d(), 1)?;
                    let shift =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                    let split = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                        let low = block.conv(u32d(), value, MockType::U32, false, true)?;
                        let shifted = block.shr(u64d(), value, shift, true)?;
                        let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                        Ok::<_, Error>((low, high))
                    };
                    let (al, ah) = split(block, a)?;
                    let (bl, bh) = split(block, b)?;
                    let low = block.add(u32d(), al, bl)?;
                    let carry = block.clt(SsaDefSpec::tmp(MockType::I1), low, al, true)?;
                    let carry = block.conv(u32d(), carry, MockType::U32, false, true)?;
                    let high = block.add(u32d(), ah, bh)?;
                    let high = block.add(u32d(), high, carry)?;
                    let high = if shape == Shape::Unrelated {
                        block.add(u32d(), ah, bh)?
                    } else {
                        high
                    };
                    let base = block.const_value(u64d(), ConstValue::U64(0x1000))?;
                    let at = |block: &mut SsaBlockBuilder<'_, MockTarget>, offset| {
                        block.emit_def(u64d(), |dest| SsaOp::PtrAdd {
                            dest,
                            base,
                            index: None,
                            stride: 1,
                            offset,
                            result_type: MockType::U64,
                        })
                    };
                    let step = if shape == Shape::Apart { 8 } else { 4 };
                    let (lower, upper) = match order {
                        Endianness::Little => (low, high),
                        Endianness::Big => (high, low),
                    };
                    let (lower, upper) = if shape == Shape::HighFirst {
                        (upper, lower)
                    } else {
                        (lower, upper)
                    };
                    let up = at(block, step)?;
                    block.store_indirect(up, upper, MockType::U32)?;
                    if matches!(shape, Shape::Elsewhere | Shape::Protected) {
                        let elsewhere = at(block, 0x100)?;
                        let read = block.load_indirect(u32d(), elsewhere, MockType::U32)?;
                        block.store_indirect(elsewhere, read, MockType::U32)?;
                    }
                    if shape == Shape::LoadBetween {
                        let read = block.load_indirect(u32d(), up, MockType::U32)?;
                        let elsewhere = at(block, 0x100)?;
                        block.store_indirect(elsewhere, read, MockType::U32)?;
                    }
                    let down = at(block, 0)?;
                    block.store_indirect(down, lower, MockType::U32)?;
                    let whole = if shape == Shape::Apart {
                        let first = block.load_indirect(u32d(), down, MockType::U32)?;
                        let second = block.load_indirect(u32d(), up, MockType::U32)?;
                        let first = block.conv(u64d(), first, MockType::U64, false, true)?;
                        let second = block.conv(u64d(), second, MockType::U64, false, true)?;
                        let second = block.shl(u64d(), second, shift)?;
                        block.or(u64d(), second, first)?
                    } else {
                        block.load_indirect(u64d(), down, MockType::U64)?
                    };
                    block.ret(Some(whole))
                })
                .expect("the case builds");
            if shape == Shape::Protected {
                builder
                    .in_block(1, |block| block.ret(None))
                    .expect("the handler builds");
            }
            let mut ssa = builder.finish().expect("the case is valid");
            if shape == Shape::Protected {
                ssa.set_exception_handlers(vec![SsaExceptionHandler {
                    flags: 0,
                    try_offset: 0,
                    try_length: 1,
                    handler_offset: 1,
                    handler_length: 1,
                    class_token_or_filter: 0,
                    protected_range: BlockRange::new(0, 1),
                    handler_range: BlockRange::new(1, 2),
                    filter_range: None,
                    selector: None,
                }]);
            }
            ssa
        };
        let stores = |ssa: &SsaFunction<MockTarget>| -> Vec<u32> {
            ssa.iter_instructions()
                .filter_map(|(_, _, instruction)| match instruction.op() {
                    SsaOp::StoreIndirect { value_type, .. } => MockTarget::bit_width(value_type),
                    _ => None,
                })
                .collect()
        };
        for order in [Endianness::Little, Endianness::Big] {
            for shape in [
                Shape::Paired,
                Shape::Elsewhere,
                Shape::Protected,
                Shape::LoadBetween,
                Shape::Apart,
                Shape::Unrelated,
                Shape::HighFirst,
            ] {
                let before = build(shape, order);
                let mut after = before.clone();
                let method = 0u32;
                run_mock_pass_boundary(&mut after, "store pair", |ssa| {
                    AlgebraicSimplificationPass.run(
                        ssa,
                        &PassEnv::new(
                            &method,
                            &NullListener,
                            AnalysisContext::new(Machine {
                                byte_order: Some(order),
                                ..MACHINE
                            }),
                        ),
                    ) == Ok(true)
                });
                let words = |ssa: &SsaFunction<MockTarget>, bits: u32| {
                    stores(ssa).iter().filter(|width| **width == bits).count()
                };
                let merged = words(&after, 64) == 1
                    && words(&after, 32).checked_add(2) == Some(words(&before, 32));
                assert_eq!(
                    merged,
                    matches!(shape, Shape::Paired | Shape::Elsewhere),
                    "{order:?}:\n{after}"
                );
                for input in inputs(&[MockType::U64, MockType::U64]) {
                    let expected = interpret_in(&before, &input, order);
                    assert!(expected.is_some(), "the case evaluates:\n{before}");
                    assert_eq!(
                        expected,
                        interpret_in(&after, &input, order),
                        "{order:?} {input:?}:\nbefore:\n{before}\nafter:\n{after}"
                    );
                }
            }
        }
    }

    /// A difference taken from a double-width difference already computed in
    /// halves, `v = a - b; v -= c`, is the difference of the three, every
    /// composition folded away and the value kept on every sample.
    #[test]
    fn a_chain_of_double_width_differences_is_one_difference() {
        let case = Case {
            name: "a - b - c".to_string(),
            args: vec![MockType::U64, MockType::U64, MockType::U32],
            body: Box::new(|block, args| {
                let u32d = || SsaDefSpec::tmp(MockType::U32);
                let u64d = || SsaDefSpec::tmp(MockType::U64);
                let flag = || SsaDefSpec::tmp(MockType::I1);
                let shift = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                let split = |block: &mut SsaBlockBuilder<'_, MockTarget>, value| {
                    let low = block.conv(u32d(), value, MockType::U32, false, true)?;
                    let shifted = block.shr(u64d(), value, shift, true)?;
                    let high = block.conv(u32d(), shifted, MockType::U32, false, true)?;
                    Ok::<_, Error>((low, high))
                };
                let (al, ah) = split(block, args[0])?;
                let (bl, bh) = split(block, args[1])?;
                let low = block.sub(u32d(), al, bl)?;
                let borrow = block.clt(flag(), al, bl, true)?;
                let borrow = block.conv(u32d(), borrow, MockType::U32, false, true)?;
                let high = block.sub(u32d(), ah, bh)?;
                let high = block.sub(u32d(), high, borrow)?;
                let next = block.sub(u32d(), low, args[2])?;
                let borrow = block.clt(flag(), low, args[2], true)?;
                let borrow = block.conv(u32d(), borrow, MockType::U32, false, true)?;
                let high = block.sub(u32d(), high, borrow)?;
                let next = block.conv(u64d(), next, MockType::U64, false, true)?;
                let high = block.conv(u64d(), high, MockType::U64, false, true)?;
                let high = block.shl(u64d(), high, shift)?;
                block.or(u64d(), high, next)
            }),
            folds_to: Some(|op| matches!(op, SsaOp::Sub { .. })),
        };
        let mut folded = build_case(&case);
        let method = 0u32;
        for _ in 0..8 {
            if !run_mock_pass_boundary(&mut folded, &case.name, |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            }) {
                break;
            }
        }
        let compositions = folded
            .iter_instructions()
            .filter(|(_, _, instruction)| matches!(instruction.op(), SsaOp::Or { .. }))
            .count();
        assert_eq!(compositions, 0, "{folded}");
        assert_preserves(&case);
    }

    /// **Two loads of adjacent words composed are the load of both.** `[p + 8]`
    /// and `[p + 12]` composed low then high are the 8-byte load at `p + 8` on
    /// a little-endian target; on a big-endian one the same composition is
    /// not, and the composition of `[p + 12]` low and `[p + 8]` high is. The
    /// word loads go with the composition: left in place, as a load that may
    /// fault is by dead code elimination, the memory pass would compose the
    /// wide load back out of them. The controls: a store between the loads, a
    /// word read apart as well, and an unknown byte order leave the
    /// composition.
    #[test]
    fn two_loads_of_adjacent_words_composed_are_one_load() {
        /// The offset from `p` of the 8-byte load the returned value is, the
        /// low word loaded at `p + low` and the high at `p + high`, with a
        /// store between them when `stored` and the low word read apart as
        /// well when `shared`; `None` when it is no such load, or a word load
        /// is left beside it.
        fn folded(
            low: i64,
            high: i64,
            (stored, shared): (bool, bool),
            order: Option<Endianness>,
        ) -> Option<i64> {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            builder
                .in_block(0, |block| {
                    let pointer = block.load_arg(SsaDefSpec::tmp(MockType::Ptr), 0)?;
                    let word = |block: &mut SsaBlockBuilder<'_, MockTarget>, offset| {
                        let addr = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| {
                            SsaOp::PtrAdd {
                                dest,
                                base: pointer,
                                index: None,
                                stride: 0,
                                offset,
                                result_type: MockType::Ptr,
                            }
                        })?;
                        block.emit_def(SsaDefSpec::tmp(MockType::U32), |dest| SsaOp::LoadIndirect {
                            dest,
                            addr,
                            value_type: MockType::U32,
                            access: AccessQualifiers::FLAT,
                        })
                    };
                    let lo = word(block, low)?;
                    if stored {
                        block.emit_no_defs(SsaOp::StoreIndirect {
                            addr: pointer,
                            value: lo,
                            value_type: MockType::U32,
                            access: AccessQualifiers::FLAT,
                        })?;
                    }
                    let hi = word(block, high)?;
                    if shared {
                        block.add(SsaDefSpec::tmp(MockType::U32), lo, lo)?;
                    }
                    let u64d = || SsaDefSpec::tmp(MockType::U64);
                    let shift =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                    let lo = block.conv(u64d(), lo, MockType::U64, false, true)?;
                    let hi = block.conv(u64d(), hi, MockType::U64, false, true)?;
                    let hi = block.shl(u64d(), hi, shift)?;
                    let whole = block.or(u64d(), hi, lo)?;
                    block.ret(Some(whole))
                })
                .expect("build block");
            let mut ssa = builder.finish().expect("finish");
            assert!(
                AlgebraicSimplificationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(
                            &0u32,
                            &NullListener,
                            AnalysisContext::new(Machine {
                                byte_order: order,
                                ..MACHINE
                            })
                        )
                    )
                    .is_ok()
            );
            let words = ssa
                .iter_instructions()
                .filter(|(_, _, instruction)| {
                    matches!(
                        instruction.op(),
                        SsaOp::LoadIndirect {
                            value_type: MockType::U32,
                            ..
                        }
                    )
                })
                .count();
            if words > 0 {
                return None;
            }
            let SsaOp::LoadIndirect {
                addr,
                value_type: MockType::U64,
                ..
            } = returned_op(&ssa)?
            else {
                return None;
            };
            match ssa.get_definition(*addr)? {
                SsaOp::PtrAdd { offset, .. } => Some(*offset),
                _ => None,
            }
        }
        let (little, big) = (Some(Endianness::Little), Some(Endianness::Big));
        let plain = (false, false);
        assert_eq!(folded(8, 12, plain, little), Some(8));
        assert_eq!(folded(12, 8, plain, big), Some(8));
        assert_eq!(folded(8, 12, plain, big), None);
        assert_eq!(folded(8, 12, (true, false), little), None);
        assert_eq!(folded(8, 12, (false, true), little), None);
        assert_eq!(folded(8, 12, plain, None), None);
    }

    /// **A composition names the halves it places side by side.** `(hi << 32)
    /// | zext(lo)` over two 32-bit values is the pair `lo`, `hi`, either
    /// operand order of the `or`, and both are whole halves; over two 16-bit
    /// values extended straight to 64 bits it is the same pair, of halves
    /// narrower than a half. A shift by another amount, and a low half
    /// sign-extended rather than zero-extended, compose no halves.
    #[test]
    fn a_composition_names_its_halves() {
        let composed = |amount: u8, zero_low: bool, swapped: bool, part: MockType| {
            let case = Case {
                name: String::new(),
                args: vec![part, part],
                body: Box::new(move |block, args| {
                    let u64d = || SsaDefSpec::tmp(MockType::U64);
                    let shift =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(amount))?;
                    let low = block.conv(u64d(), args[0], MockType::U64, false, zero_low)?;
                    let high = block.conv(u64d(), args[1], MockType::U64, false, true)?;
                    let high = block.shl(u64d(), high, shift)?;
                    if swapped {
                        block.or(u64d(), low, high)
                    } else {
                        block.or(u64d(), high, low)
                    }
                }),
                folds_to: None,
            };
            let ssa = build_case(&case);
            let op = returned_op(&ssa).expect("a composition").clone();
            let halves = doubleword::halves(&ssa, &op, &ssa.find_constants())?;
            let argument = |value: SsaVarId| {
                ssa.get_definition(value).and_then(|op| match op {
                    SsaOp::LoadArg { arg_index, .. } => Some(*arg_index),
                    _ => None,
                })
            };
            Some((
                argument(halves.low)?,
                argument(halves.high)?,
                halves.whole(&ssa),
            ))
        };
        let word = MockType::I32;
        assert_eq!(composed(32, true, false, word), Some((0, 1, true)));
        assert_eq!(composed(32, true, true, word), Some((0, 1, true)));
        assert_eq!(
            composed(32, true, false, MockType::I16),
            Some((0, 1, false)),
            "halves narrower than a half"
        );
        assert_eq!(composed(16, true, false, word), None, "another amount");
        assert_eq!(
            composed(32, false, false, word),
            None,
            "a sign-extended low half"
        );
    }

    /// An `int` composed with its sign above it, as a 32-bit target passes one
    /// promoted to `long long`, is its sign extension. The control: a logical
    /// shift fills zeros, no sign, and the composition stays.
    #[test]
    fn a_value_composed_with_its_sign_is_its_extension() {
        for (name, signed) in [("sign fill", true), ("zero fill", false)] {
            assert_preserves(&Case {
                name: name.to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let u64d = || SsaDefSpec::tmp(MockType::U64);
                    let fill =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(31))?;
                    let shift =
                        block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(32))?;
                    let high = block.shr(SsaDefSpec::tmp(MockType::I32), args[0], fill, !signed)?;
                    let low = block.conv(u64d(), args[0], MockType::U64, false, true)?;
                    let high = block.conv(u64d(), high, MockType::U64, false, true)?;
                    let high = block.shl(u64d(), high, shift)?;
                    block.or(u64d(), high, low)
                }),
                folds_to: Some(if signed {
                    |op| {
                        matches!(
                            op,
                            SsaOp::IntConv {
                                unsigned: false,
                                ..
                            }
                        )
                    }
                } else {
                    |op| matches!(op, SsaOp::Or { .. })
                }),
            });
        }
    }

    /// **A value split by masks and joined again is the value, or the value
    /// under the masks' union.** `(x & m) | (x & ~m)` is `x` whatever `m` is,
    /// and so are `(x & m) ^ (x & ~m)` and its dual `(x | m) & (x | ~m)`; over
    /// constant masks `(x & a) | (x & b)` is `x & (a | b)`, `(x & a) ^ (x &
    /// b)` is `x & (a ^ b)` and `(x | a) & (x | b)` is `x | (a & b)`, and a
    /// mask two values share comes out, `(x & a) | (y & a)` being `(x | y) &
    /// a`. At two widths, each kept on every sample and folded to fewer bit
    /// operations. The controls keep their value and fold nothing: two parts
    /// that share no operand, and two values under two variable masks.
    #[test]
    fn a_value_split_by_masks_and_joined_is_the_value() {
        for ty in [MockType::U8, MockType::U32] {
            let d = move || SsaDefSpec::<MockTarget>::tmp(ty);
            let shapes: Vec<(&str, usize, bool, Body)> = vec![
                (
                    "(x & m) | (x & ~m)",
                    0,
                    true,
                    Box::new(move |block, args| {
                        let m = block.not(d(), args[1])?;
                        let a = block.and(d(), args[0], args[1])?;
                        let b = block.and(d(), args[0], m)?;
                        block.or(d(), a, b)
                    }),
                ),
                (
                    "(x & m) ^ (x & ~m)",
                    0,
                    true,
                    Box::new(move |block, args| {
                        let m = block.not(d(), args[1])?;
                        let a = block.and(d(), args[0], args[1])?;
                        let b = block.and(d(), m, args[0])?;
                        block.xor(d(), a, b)
                    }),
                ),
                (
                    "(x | m) & (x | ~m)",
                    0,
                    true,
                    Box::new(move |block, args| {
                        let m = block.not(d(), args[1])?;
                        let a = block.or(d(), args[0], args[1])?;
                        let b = block.or(d(), args[0], m)?;
                        block.and(d(), a, b)
                    }),
                ),
                (
                    "(x & a) | (x & b)",
                    1,
                    true,
                    Box::new(move |block, args| {
                        let a = block.const_value(d(), konst(ty, 0x50))?;
                        let b = block.const_value(d(), konst(ty, 0x0c))?;
                        let a = block.and(d(), args[0], a)?;
                        let b = block.and(d(), b, args[0])?;
                        block.or(d(), a, b)
                    }),
                ),
                (
                    "(x & a) ^ (x & b)",
                    1,
                    true,
                    Box::new(move |block, args| {
                        let a = block.const_value(d(), konst(ty, 0xf0))?;
                        let b = block.const_value(d(), konst(ty, 0x3c))?;
                        let a = block.and(d(), args[0], a)?;
                        let b = block.and(d(), args[0], b)?;
                        block.xor(d(), a, b)
                    }),
                ),
                (
                    "(x | a) & (x | b)",
                    1,
                    true,
                    Box::new(move |block, args| {
                        let a = block.const_value(d(), konst(ty, 0xf0))?;
                        let b = block.const_value(d(), konst(ty, 0x3c))?;
                        let a = block.or(d(), args[0], a)?;
                        let b = block.or(d(), args[0], b)?;
                        block.and(d(), a, b)
                    }),
                ),
                (
                    "(x & a) | (y & a)",
                    2,
                    true,
                    Box::new(move |block, args| {
                        let a = block.const_value(d(), konst(ty, 0x5a))?;
                        let x = block.and(d(), args[0], a)?;
                        let y = block.and(d(), a, args[1])?;
                        block.or(d(), x, y)
                    }),
                ),
                (
                    "control: (x & m) | (~x & ~m)",
                    0,
                    false,
                    Box::new(move |block, args| {
                        let x = block.not(d(), args[0])?;
                        let m = block.not(d(), args[1])?;
                        let a = block.and(d(), args[0], args[1])?;
                        let b = block.and(d(), x, m)?;
                        block.or(d(), a, b)
                    }),
                ),
                (
                    "control: (x & m) | (y & ~m)",
                    0,
                    false,
                    Box::new(move |block, args| {
                        let one = block.const_value(d(), konst(ty, 1))?;
                        let y = block.add(d(), args[0], one)?;
                        let m = block.not(d(), args[1])?;
                        let a = block.and(d(), args[0], args[1])?;
                        let b = block.and(d(), y, m)?;
                        block.or(d(), a, b)
                    }),
                ),
            ];
            for (name, kept, folds, body) in shapes {
                let case = Case {
                    name: format!("{name} at {ty:?}"),
                    args: vec![ty, ty],
                    body,
                    folds_to: None,
                };
                let before = build_case(&case);
                let mut after = before.clone();
                for _ in 0..4 {
                    if !run_mock_pass_boundary(&mut after, &case.name, |ssa| {
                        AlgebraicSimplificationPass.run(
                            ssa,
                            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                        ) == Ok(true)
                    }) {
                        break;
                    }
                }
                // Counted live: the operations a fold leaves unread go with
                // dead code.
                let bitwise = |ssa: &SsaFunction<MockTarget>| {
                    let mut live = ssa.clone();
                    assert!(
                        DeadCodeEliminationPass
                            .run(
                                &mut live,
                                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                            )
                            .is_ok()
                    );
                    live.iter_instructions()
                        .filter(|(_, _, instruction)| {
                            matches!(
                                instruction.op(),
                                SsaOp::And { .. } | SsaOp::Or { .. } | SsaOp::Xor { .. }
                            )
                        })
                        .count()
                };
                if folds {
                    assert_eq!(bitwise(&after), kept, "{}:\n{after}", case.name);
                } else {
                    assert_eq!(bitwise(&after), bitwise(&before), "{}:\n{after}", case.name);
                }
                assert_preserves(&case);
            }
        }
    }

    /// Every byte-reversal spelling computes the same value after the pass.
    #[test]
    fn every_byte_reversal_spelling_keeps_its_value() {
        for case in bswap_cases() {
            assert_preserves(&case);
        }
    }

    /// The shapes a 64-bit machine computing 32-bit values leaves - an
    /// operation at the register's width truncated to the value's, a
    /// complement spelled as `xor` with all ones, a negation spelled as a
    /// subtraction from zero, a carry sum over wide terms - each with the
    /// operation the pass must leave defining the result.
    fn narrowing_cases() -> Vec<Case> {
        let i32d = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        vec![
            Case {
                name: "trunc32(sext64(a) ^ sext64(b))".to_string(),
                args: vec![MockType::I32, MockType::I32],
                body: Box::new(move |block, args| {
                    let a = block.conv(i64d(), args[0], MockType::I64, false, false)?;
                    let b = block.conv(i64d(), args[1], MockType::I64, false, false)?;
                    let wide = block.xor(i64d(), a, b)?;
                    block.conv(i32d(), wide, MockType::I32, false, false)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Xor { .. })),
            },
            Case {
                name: "trunc32(zext64(a) * sext64(b))".to_string(),
                args: vec![MockType::I32, MockType::I32],
                body: Box::new(move |block, args| {
                    let a = block.conv(i64d(), args[0], MockType::I64, false, true)?;
                    let b = block.conv(i64d(), args[1], MockType::I64, false, false)?;
                    let wide = block.mul(i64d(), a, b)?;
                    block.conv(i32d(), wide, MockType::I32, false, true)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Mul { .. })),
            },
            Case {
                name: "trunc32(sext64(a) + 0x1_0000_0005)".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let a = block.conv(i64d(), args[0], MockType::I64, false, false)?;
                    let k = block.const_i64(i64d(), 0x1_0000_0005)?;
                    let wide = block.add(i64d(), a, k)?;
                    block.conv(i32d(), wide, MockType::I32, false, false)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
            },
            Case {
                name: "trunc32(sext64(a) ^ zext64(b8))".to_string(),
                args: vec![MockType::I32, MockType::I8],
                body: Box::new(move |block, args| {
                    let a = block.conv(i64d(), args[0], MockType::I64, false, false)?;
                    let b = block.conv(i64d(), args[1], MockType::I64, false, true)?;
                    let wide = block.xor(i64d(), a, b)?;
                    block.conv(i32d(), wide, MockType::I32, false, false)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Xor { .. })),
            },
            Case {
                name: "trunc32(a + b) of 64-bit arguments".to_string(),
                args: vec![MockType::I64, MockType::I64],
                body: Box::new(move |block, args| {
                    let wide = block.add(i64d(), args[0], args[1])?;
                    block.conv(i32d(), wide, MockType::I32, false, false)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
            },
            Case {
                name: "trunc32(a >> 1) of a 64-bit argument is not narrow".to_string(),
                args: vec![MockType::I64],
                body: Box::new(move |block, args| {
                    let k = block.const_i32(i32d(), 1)?;
                    let wide = block.shr(i64d(), args[0], k, true)?;
                    block.conv(i32d(), wide, MockType::I32, false, false)
                }),
                folds_to: None,
            },
            Case {
                name: "ptrtoint32(a + b) of 64-bit arguments".to_string(),
                args: vec![MockType::I64, MockType::I64],
                body: Box::new(move |block, args| {
                    let address =
                        block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: args[0],
                            index: Some(args[1]),
                            stride: 1,
                            offset: 0,
                            result_type: MockType::Ptr,
                        })?;
                    block.emit_def(i32d(), |dest| SsaOp::PtrToInt {
                        dest,
                        operand: address,
                        target: MockType::I32,
                    })
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
            },
            Case {
                name: "ptrtoint32(a + 5) of a 64-bit argument".to_string(),
                args: vec![MockType::I64],
                body: Box::new(move |block, args| {
                    let address =
                        block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: args[0],
                            index: None,
                            stride: 1,
                            offset: 5,
                            result_type: MockType::Ptr,
                        })?;
                    block.emit_def(i32d(), |dest| SsaOp::PtrToInt {
                        dest,
                        operand: address,
                        target: MockType::I32,
                    })
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
            },
            Case {
                name: "ptrtoint32(a + b * 4) of 64-bit arguments".to_string(),
                args: vec![MockType::I64, MockType::I64],
                body: Box::new(move |block, args| {
                    let address =
                        block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: args[0],
                            index: Some(args[1]),
                            stride: 4,
                            offset: 0,
                            result_type: MockType::Ptr,
                        })?;
                    block.emit_def(i32d(), |dest| SsaOp::PtrToInt {
                        dest,
                        operand: address,
                        target: MockType::I32,
                    })
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
            },
            Case {
                name: "trunc32(sext64(a) << 31)".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let a = block.conv(i64d(), args[0], MockType::I64, false, false)?;
                    let k = block.const_i32(i32d(), 31)?;
                    let wide = block.shl(i64d(), a, k)?;
                    block.conv(i32d(), wide, MockType::I32, false, false)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Shl { .. })),
            },
            Case {
                name: "trunc32(sext64(a) << 32) is not a narrow shift".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let a = block.conv(i64d(), args[0], MockType::I64, false, false)?;
                    let k = block.const_i32(i32d(), 32)?;
                    let wide = block.shl(i64d(), a, k)?;
                    block.conv(i32d(), wide, MockType::I32, false, false)
                }),
                folds_to: None,
            },
            Case {
                name: "a ^ -1".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let ones = block.const_i32(i32d(), -1)?;
                    block.xor(i32d(), args[0], ones)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Not { .. })),
            },
            Case {
                name: "0 - a".to_string(),
                args: vec![MockType::I32],
                body: Box::new(move |block, args| {
                    let zero = block.const_i32(i32d(), 0)?;
                    block.sub(i32d(), zero, args[0])
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Neg { .. })),
            },
            Case {
                name: "a + -b".to_string(),
                args: vec![MockType::I32, MockType::I32],
                body: Box::new(move |block, args| {
                    let negated = block.neg(i32d(), args[1])?;
                    block.add(i32d(), args[0], negated)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Sub { .. })),
            },
            Case {
                name: "-a + b".to_string(),
                args: vec![MockType::I32, MockType::I32],
                body: Box::new(move |block, args| {
                    let negated = block.neg(i32d(), args[0])?;
                    block.add(i32d(), negated, args[1])
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Sub { .. })),
            },
            Case {
                name: "a - -b".to_string(),
                args: vec![MockType::I32, MockType::I32],
                body: Box::new(move |block, args| {
                    let negated = block.neg(i32d(), args[1])?;
                    block.sub(i32d(), args[0], negated)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Add { .. })),
            },
            Case {
                name: "-a - b keeps its value".to_string(),
                args: vec![MockType::I32, MockType::I32],
                body: Box::new(move |block, args| {
                    let negated = block.neg(i32d(), args[0])?;
                    block.sub(i32d(), negated, args[1])
                }),
                folds_to: None,
            },
            Case {
                name: "--a".to_string(),
                args: vec![MockType::I64],
                body: Box::new(move |block, args| {
                    let once = block.neg(i64d(), args[0])?;
                    block.neg(i64d(), once)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Copy { .. })),
            },
            Case {
                name: "~~a".to_string(),
                args: vec![MockType::I64],
                body: Box::new(move |block, args| {
                    let once = block.not(i64d(), args[0])?;
                    block.not(i64d(), once)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Copy { .. })),
            },
            Case {
                name: "trunc32(a ^ b) + (trunc32(a & b) << 1) over I64 terms".to_string(),
                args: vec![MockType::I64, MockType::I64],
                body: Box::new(move |block, args| {
                    let (a, b) = (args[0], args[1]);
                    let xor = block.xor(i64d(), a, b)?;
                    let and = block.and(i64d(), a, b)?;
                    let narrow_xor = block.conv(i32d(), xor, MockType::I32, false, true)?;
                    let narrow_and = block.conv(i32d(), and, MockType::I32, false, true)?;
                    let one = block.const_i32(i32d(), 1)?;
                    let twice = block.shl(i32d(), narrow_and, one)?;
                    block.add(i32d(), narrow_xor, twice)
                }),
                folds_to: Some(|op| matches!(op, SsaOp::Copy { .. })),
            },
        ]
    }

    /// A phi of a sign-extended 32-bit value and a constant a sign extension
    /// gives is the sign extension of the 32-bit phi: the narrow phi joins the
    /// value and the narrowed constant, and the wide result is redefined as its
    /// extension, so the return still reads the same 64-bit value.
    ///
    /// The controls: a phi whose constant no 32-bit value extends to
    /// (`0x1_0000_0000`) is left alone, and so is one whose result is read
    /// whole - returned as the 64-bit value - where the narrow phi would only
    /// add the extension back.
    #[test]
    fn a_phi_of_extended_values_is_the_extension_of_their_phi() {
        for (constant, read_narrow, narrows) in [
            (-1_i64, true, true),
            (0x1_0000_0000, true, false),
            (-1, false, false),
        ] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
            builder
                .in_block(0, |block| {
                    let flag = block.load_arg(mock_i32(), 1)?;
                    block.branch(flag, 1, 2)
                })
                .expect("entry");
            let mut wide = None;
            builder
                .in_block(1, |block| {
                    let x = block.load_arg(mock_i32(), 0)?;
                    wide = Some(block.conv(mock_i64(), x, MockType::I64, false, false)?);
                    block.jump(3)
                })
                .expect("extending arm");
            let mut konst = None;
            builder
                .in_block(2, |block| {
                    konst = Some(block.const_i64(mock_i64(), constant)?);
                    block.jump(3)
                })
                .expect("constant arm");
            builder
                .in_block(3, |block| {
                    let joined = block.phi(
                        mock_i64(),
                        [(1, wide.expect("wide")), (2, konst.expect("constant"))],
                    )?;
                    let read = if read_narrow {
                        block.conv(mock_i32(), joined, MockType::I32, false, false)?
                    } else {
                        joined
                    };
                    block.ret(Some(read))
                })
                .expect("join");
            let mut ssa = builder.finish().expect("finish");
            let method = 0u32;
            let changed = run_mock_pass_boundary(&mut ssa, "phi narrowing", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let join = ssa.block(3).expect("join block");
            let phi = join.phi_nodes().first().expect("the join keeps a phi");
            let phi_type = ssa.variable(phi.result()).map(|v| *v.var_type());
            if narrows {
                assert!(changed);
                assert_eq!(phi_type, Some(MockType::I32), "{ssa}");
                assert!(
                    join.instructions().first().is_some_and(|instr| matches!(
                        instr.op(),
                        SsaOp::IntConv { operand, target: MockType::I64, unsigned: false, .. }
                            if *operand == phi.result()
                    )),
                    "the wide value is the narrow phi sign-extended: {ssa}"
                );
                assert!(
                    ssa.block(2)
                        .is_some_and(|block| block.instructions().iter().any(|instr| {
                            matches!(
                                instr.op(),
                                SsaOp::Const {
                                    value: ConstValue::I32(-1),
                                    ..
                                }
                            )
                        })),
                    "the constant joins at the narrow width: {ssa}"
                );
            } else {
                assert_eq!(phi_type, Some(MockType::I64), "{ssa}");
            }
        }
    }

    /// A phi of sign extensions of a signed reading of unsigned values joins
    /// the unsigned values, and is read signed again before the extension: the
    /// RV64 `addw` result is a `uint32_t` sum the lift sign-extends through an
    /// `int32_t` view, and `(int64_t)(uint32_t)x` would be a zero extension.
    #[test]
    fn a_phi_of_reinterpreted_extensions_joins_the_unsigned_values() {
        let u32d = || SsaDefSpec::<MockTarget>::tmp(MockType::U32);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        builder
            .in_block(0, |block| {
                let flag = block.load_arg(mock_i32(), 1)?;
                block.branch(flag, 1, 2)
            })
            .expect("entry");
        let mut wide = None;
        builder
            .in_block(1, |block| {
                let x = block.load_arg(u32d(), 0)?;
                let viewed = block.conv(mock_i32(), x, MockType::I32, false, true)?;
                wide = Some(block.conv(mock_i64(), viewed, MockType::I64, false, false)?);
                block.jump(3)
            })
            .expect("extending arm");
        let mut konst = None;
        builder
            .in_block(2, |block| {
                konst = Some(block.const_i64(mock_i64(), -2_128_831_035)?);
                block.jump(3)
            })
            .expect("constant arm");
        builder
            .in_block(3, |block| {
                let joined = block.phi(
                    mock_i64(),
                    [(1, wide.expect("wide")), (2, konst.expect("constant"))],
                )?;
                let read = block.conv(u32d(), joined, MockType::U32, false, true)?;
                block.ret(Some(read))
            })
            .expect("join");
        let mut ssa = builder.finish().expect("finish");
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "reinterpreted phi narrowing", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let join = ssa.block(3).expect("join block");
        let phi = join.phi_nodes().first().expect("the join keeps a phi");
        assert_eq!(
            ssa.variable(phi.result()).map(|v| *v.var_type()),
            Some(MockType::U32),
            "{ssa}"
        );
        assert!(
            ssa.block(2)
                .is_some_and(|block| block.instructions().iter().any(|instr| {
                    matches!(
                        instr.op(),
                        SsaOp::Const {
                            value: ConstValue::U32(0x811c_9dc5),
                            ..
                        }
                    )
                })),
            "the constant joins as the unsigned value it holds: {ssa}"
        );
        let (viewed, extended) = (
            join.instructions().first().map(|instr| instr.op()),
            join.instructions().get(1).map(|instr| instr.op()),
        );
        let Some(SsaOp::IntConv {
            dest: view,
            operand,
            target: MockType::I32,
            ..
        }) = viewed
        else {
            panic!("the narrow phi is viewed signed first: {ssa}");
        };
        assert_eq!(*operand, phi.result());
        assert!(
            matches!(
                extended,
                Some(SsaOp::IntConv { operand, target: MockType::I64, unsigned: false, .. })
                    if operand == view
            ),
            "then sign-extended: {ssa}"
        );
    }

    /// Each narrowing shape keeps its value and leaves the narrow operation.
    #[test]
    fn a_narrowed_operation_keeps_its_value() {
        for case in narrowing_cases() {
            assert_preserves(&case);
        }
    }

    /// Comparisons of 32-bit values extended to 64 bits, against each other
    /// or a constant: each kind of extension under each reading of the
    /// order, and the shapes the rule leaves wide - a constant no narrow value
    /// extends to, extensions of two kinds.
    fn comparison_cases() -> Vec<Case> {
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        // `a` and `b` extended by `zero_a` and `zero_b`, compared by `compare`.
        type Compare =
            fn(&mut SsaBlockBuilder<'_, MockTarget>, SsaVarId, SsaVarId) -> Result<SsaVarId>;
        let less = |unsigned: bool| -> Compare {
            if unsigned {
                |block, a, b| block.clt(mock_i32(), a, b, true)
            } else {
                |block, a, b| block.clt(mock_i32(), a, b, false)
            }
        };
        let mut cases = Vec::new();
        for (zero_a, zero_b) in [(false, false), (true, true), (false, true)] {
            for unsigned in [false, true] {
                let compare = less(unsigned);
                cases.push(Case {
                    name: format!(
                        "{}ext64(a) <{} {}ext64(b)",
                        if zero_a { "z" } else { "s" },
                        if unsigned { "u" } else { "s" },
                        if zero_b { "z" } else { "s" }
                    ),
                    args: vec![MockType::I32, MockType::I32],
                    body: Box::new(move |block, args| {
                        let a = block.conv(i64d(), args[0], MockType::I64, false, zero_a)?;
                        let b = block.conv(i64d(), args[1], MockType::I64, false, zero_b)?;
                        compare(block, a, b)
                    }),
                    folds_to: None,
                });
            }
        }
        for (zero, constant) in [
            (false, -1_i64),
            (false, 0x7fff_ffff),
            (false, 0x1_0000_0000),
            (true, 0xffff_ffff),
            (true, -1),
        ] {
            for unsigned in [false, true] {
                let compare = less(unsigned);
                cases.push(Case {
                    name: format!(
                        "{}ext64(a) <{} {constant:#x}",
                        if zero { "z" } else { "s" },
                        if unsigned { "u" } else { "s" }
                    ),
                    args: vec![MockType::I32],
                    body: Box::new(move |block, args| {
                        let a = block.conv(i64d(), args[0], MockType::I64, false, zero)?;
                        let k = block.const_i64(i64d(), constant)?;
                        compare(block, a, k)
                    }),
                    folds_to: None,
                });
            }
        }
        cases.push(Case {
            name: "sext64(a) == sext64(b)".to_string(),
            args: vec![MockType::I32, MockType::I32],
            body: Box::new(move |block, args| {
                let a = block.conv(i64d(), args[0], MockType::I64, false, false)?;
                let b = block.conv(i64d(), args[1], MockType::I64, false, false)?;
                block.ceq(mock_i32(), a, b)
            }),
            folds_to: None,
        });
        cases
    }

    /// Each comparison of extended values keeps its value.
    #[test]
    fn a_narrowed_comparison_keeps_its_value() {
        for case in comparison_cases() {
            assert_preserves(&case);
        }
    }

    /// **A comparison of values extended from one narrower width compares
    /// those values.** `sext64(a) < 5` is `a < 5` at 32 bits, signed, and
    /// `zext64(a) < zext64(b)` is `a < b` unsigned, whatever the wide
    /// comparison read; a branch on `sext64(a) >= 0` branches on `a >= 0`.
    /// The controls: a constant no 32-bit value sign-extends to
    /// (`0x1_0000_0000`) and a sign extension against a zero extension are
    /// left wide.
    #[test]
    fn a_comparison_of_extended_values_compares_them_at_their_width() {
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        let compared = |zero: (bool, bool), constant: Option<i64>, unsigned: bool| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
            builder
                .in_block(0, |block| {
                    let a = block.load_arg(mock_i32(), 0)?;
                    let b = block.load_arg(mock_i32(), 1)?;
                    let a = block.conv(i64d(), a, MockType::I64, false, zero.0)?;
                    let b = match constant {
                        Some(constant) => block.const_i64(i64d(), constant)?,
                        None => block.conv(i64d(), b, MockType::I64, false, zero.1)?,
                    };
                    let less = block.clt(mock_i32(), a, b, unsigned)?;
                    block.ret(Some(less))
                })
                .expect("compare");
            let mut ssa = builder.finish().expect("finish");
            run_mock_pass_boundary(&mut ssa, "comparison narrowing", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let Some(SsaOp::Clt {
                left,
                right,
                unsigned,
                ..
            }) = returned_op(&ssa)
            else {
                panic!("the comparison stays: {ssa}");
            };
            (
                ssa.bit_width(*left),
                ssa.bit_width(*right),
                *unsigned,
                ssa.to_string(),
            )
        };
        let (left, right, unsigned, ssa) = compared((false, false), Some(5), false);
        assert_eq!(
            (left, right, unsigned),
            (Some(32), Some(32), false),
            "{ssa}"
        );
        let (left, right, unsigned, ssa) = compared((true, true), None, false);
        assert_eq!((left, right, unsigned), (Some(32), Some(32), true), "{ssa}");
        let (left, _, _, ssa) = compared((false, false), Some(0x1_0000_0000), false);
        assert_eq!(
            left,
            Some(64),
            "no narrow value extends to the constant: {ssa}"
        );
        let (left, _, _, ssa) = compared((false, true), None, false);
        assert_eq!(left, Some(64), "two kinds of extension: {ssa}");

        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        builder
            .in_block(0, |block| {
                let a = block.load_arg(mock_i32(), 0)?;
                let a = block.conv(i64d(), a, MockType::I64, false, false)?;
                let zero = block.const_i64(i64d(), 0)?;
                block.branch_cmp(a, zero, CmpKind::Ge, false, 1, 2)
            })
            .expect("branch");
        for target in [1, 2] {
            builder
                .in_block(target, |block| {
                    let result =
                        block.const_i32(mock_i32(), i32::try_from(target).expect("small"))?;
                    block.ret(Some(result))
                })
                .expect("arm");
        }
        let mut ssa = builder.finish().expect("finish");
        run_mock_pass_boundary(&mut ssa, "branch narrowing", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let Some(SsaOp::BranchCmp { left, right, .. }) =
            ssa.block(0).and_then(|block| block.control_terminator())
        else {
            panic!("the branch stays: {ssa}");
        };
        assert_eq!(
            (ssa.bit_width(*left), ssa.bit_width(*right)),
            (Some(32), Some(32)),
            "{ssa}"
        );
    }

    /// **A phi a comparison reads against a constant its extension gives back
    /// is narrowed, and so is the comparison.** RV64's `int` loop counter is
    /// a phi of sign-extended `addw` results, which its 64-bit `blt` compares
    /// with the bound: the comparison reads the counter at 32 bits. The
    /// control: compared with a constant no 32-bit value sign-extends to, the
    /// phi is read whole and stays.
    #[test]
    fn a_phi_a_comparison_reads_at_its_width_is_narrowed() {
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        for (bound, narrows) in [(10_i64, true), (0x1_0000_0000, false)] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
            builder
                .in_block(0, |block| {
                    let flag = block.load_arg(mock_i32(), 1)?;
                    block.branch(flag, 1, 2)
                })
                .expect("entry");
            let mut wide = None;
            builder
                .in_block(1, |block| {
                    let x = block.load_arg(mock_i32(), 0)?;
                    wide = Some(block.conv(i64d(), x, MockType::I64, false, false)?);
                    block.jump(3)
                })
                .expect("extending arm");
            let mut konst = None;
            builder
                .in_block(2, |block| {
                    konst = Some(block.const_i64(i64d(), -1)?);
                    block.jump(3)
                })
                .expect("constant arm");
            builder
                .in_block(3, |block| {
                    let joined = block.phi(
                        mock_i64(),
                        [(1, wide.expect("wide")), (2, konst.expect("constant"))],
                    )?;
                    let limit = block.const_i64(i64d(), bound)?;
                    let less = block.clt(mock_i32(), joined, limit, false)?;
                    block.ret(Some(less))
                })
                .expect("join");
            let mut ssa = builder.finish().expect("finish");
            // The phi narrows first, and the comparison of its extension the
            // run after, as the normalizer's fixpoint runs it.
            for _ in 0..4 {
                if !run_mock_pass_boundary(&mut ssa, "compared phi narrowing", |ssa| {
                    AlgebraicSimplificationPass.run(
                        ssa,
                        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                    ) == Ok(true)
                }) {
                    break;
                }
            }
            let join = ssa.block(3).expect("join block");
            let phi = join.phi_nodes().first().expect("the join keeps a phi");
            let phi_width = ssa.bit_width(phi.result());
            if narrows {
                assert_eq!(phi_width, Some(32), "{ssa}");
                let Some(SsaOp::Clt { left, .. }) = join
                    .instructions()
                    .iter()
                    .map(|instr| instr.op())
                    .find(|op| matches!(op, SsaOp::Clt { .. }))
                else {
                    panic!("the comparison stays: {ssa}");
                };
                assert_eq!(*left, phi.result(), "it compares the narrow phi: {ssa}");
            } else {
                assert_eq!(phi_width, Some(64), "{ssa}");
            }
        }
    }

    /// **A wide value no reader sees past the narrow bits joins truncated.**
    /// A loop counter enters as a whole 64-bit register and comes back as its
    /// `addw`'s sign extension; read only at its low 32 bits, its phi joins the
    /// register truncated and the 32-bit sum, so one width is live across the
    /// back edge. Here a diamond of the two: the wide argument is truncated at
    /// the end of its arm. The controls: read whole, or compared with a
    /// constant (whose comparison reads the high bits the truncation drops), the
    /// phi stays wide, and so does one whose extension another phi reads too
    /// (the truncation would be a row more, no extension dying for it) and one
    /// joining no extension at all.
    #[test]
    fn a_wide_value_read_at_the_narrow_bits_joins_truncated() {
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        #[derive(Clone, Copy, PartialEq)]
        enum Read {
            Low,
            Whole,
            Compared,
            /// Low, and the extension read by another phi, which keeps it.
            LowExtensionKept,
        }
        for (read, extending, narrows) in [
            (Read::Low, true, true),
            (Read::Whole, true, false),
            (Read::Compared, true, false),
            (Read::LowExtensionKept, true, false),
            (Read::Low, false, false),
        ] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
            builder
                .in_block(0, |block| {
                    let flag = block.load_arg(mock_i32(), 1)?;
                    block.branch(flag, 1, 2)
                })
                .expect("entry");
            let mut narrowed_arm = None;
            builder
                .in_block(1, |block| {
                    narrowed_arm = Some(if extending {
                        let x = block.load_arg(mock_i32(), 0)?;
                        block.conv(i64d(), x, MockType::I64, false, false)?
                    } else {
                        block.load_arg(mock_i64(), 0)?
                    });
                    block.jump(3)
                })
                .expect("extending arm");
            let mut whole = None;
            builder
                .in_block(2, |block| {
                    whole = Some(block.load_arg(mock_i64(), 2)?);
                    block.jump(3)
                })
                .expect("wide arm");
            builder
                .in_block(3, |block| {
                    let joined = block.phi(
                        mock_i64(),
                        [
                            (1, narrowed_arm.expect("extension")),
                            (2, whole.expect("wide")),
                        ],
                    )?;
                    let result = match read {
                        Read::Low => block.conv(mock_i32(), joined, MockType::I32, false, true)?,
                        Read::Whole => joined,
                        Read::Compared => {
                            let limit = block.const_i64(i64d(), 10)?;
                            block.clt(mock_i32(), joined, limit, false)?
                        }
                        Read::LowExtensionKept => {
                            let other = block.phi(
                                mock_i64(),
                                [
                                    (1, narrowed_arm.expect("extension")),
                                    (2, whole.expect("wide")),
                                ],
                            )?;
                            let low = block.conv(mock_i32(), joined, MockType::I32, false, true)?;
                            let widened = block.conv(i64d(), low, MockType::I64, false, true)?;
                            block.add(i64d(), widened, other)?
                        }
                    };
                    block.ret(Some(result))
                })
                .expect("join");
            let mut ssa = builder.finish().expect("finish");
            let changed = run_mock_pass_boundary(&mut ssa, "truncated phi narrowing", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let join = ssa.block(3).expect("join block");
            // A narrowed phi is added again, after the others of its block.
            let narrowed = join
                .phi_nodes()
                .iter()
                .any(|phi| ssa.bit_width(phi.result()) == Some(32));
            if narrows {
                assert!(changed);
                assert!(narrowed, "{ssa}");
                assert!(
                    ssa.block(2)
                        .is_some_and(|block| block.instructions().iter().any(|instr| matches!(
                            instr.op(),
                            SsaOp::IntConv { operand, target: MockType::I32, .. }
                                if Some(*operand) == whole
                        ))),
                    "the wide argument joins truncated in its arm: {ssa}"
                );
            } else {
                assert!(!narrowed, "{ssa}");
            }
        }
    }

    /// **A wide value never joins truncated into a truth value.** A join of a
    /// widened `bool` and a wide argument, read only at the `bool`'s width,
    /// would narrow to the `bool`'s type and truncate the argument into it,
    /// but a truth value holds 0 or 1 however many bits store it, which a
    /// truncated integer need not be. The phi stays wide. The control is
    /// `a_wide_value_read_at_the_narrow_bits_joins_truncated`'s narrowing
    /// case, the same join of an integer's extension.
    #[test]
    fn a_wide_value_never_joins_truncated_into_a_truth_value() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
        builder
            .in_block(0, |block| {
                let flag = block.load_arg(mock_i32(), 1)?;
                block.branch(flag, 1, 2)
            })
            .expect("entry");
        let mut widened = None;
        builder
            .in_block(1, |block| {
                let truth = block.load_arg(SsaDefSpec::tmp(MockType::Bool), 0)?;
                widened = Some(block.conv(mock_i32(), truth, MockType::I32, false, true)?);
                block.jump(3)
            })
            .expect("truth arm");
        let mut whole = None;
        builder
            .in_block(2, |block| {
                whole = Some(block.load_arg(mock_i32(), 2)?);
                block.jump(3)
            })
            .expect("wide arm");
        builder
            .in_block(3, |block| {
                let joined = block.phi(
                    mock_i32(),
                    [(1, widened.expect("widened")), (2, whole.expect("wide"))],
                )?;
                let low = block.conv(
                    SsaDefSpec::tmp(MockType::I1),
                    joined,
                    MockType::I1,
                    false,
                    true,
                )?;
                block.ret(Some(low))
            })
            .expect("join");
        let mut ssa = builder.finish().expect("finish");
        run_mock_pass_boundary(&mut ssa, "truth value truncation", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let truth_typed = ssa.blocks().iter().any(|block| {
            block.instructions().iter().any(|instr| {
                matches!(
                    instr.op(),
                    SsaOp::IntConv {
                        target: MockType::Bool,
                        ..
                    }
                )
            }) || block.phi_nodes().iter().any(|phi| {
                ssa.variable(phi.result())
                    .is_some_and(|v| *v.var_type() == MockType::Bool)
            })
        });
        assert!(
            !truth_typed,
            "nothing is truncated into a truth value: {ssa}"
        );
    }

    /// **A phi's value never joins truncated.** The phis a value flows
    /// through are one web, which the lowering gives one variable; truncating
    /// the inner phi's value where it enters the outer one would split the web,
    /// and move the copy the narrowing saves to the edge it enters by. The
    /// control is `a_wide_value_read_at_the_narrow_bits_joins_truncated`'s
    /// narrowing case, the same join of an extension and a wide value that is
    /// no phi's.
    #[test]
    fn a_phi_value_joined_with_an_extension_stays_wide() {
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
        builder
            .in_block(0, |block| {
                let flag = block.load_arg(mock_i32(), 1)?;
                block.branch(flag, 1, 2)
            })
            .expect("entry");
        let mut extension = None;
        builder
            .in_block(1, |block| {
                let x = block.load_arg(mock_i32(), 0)?;
                extension = Some(block.conv(i64d(), x, MockType::I64, false, false)?);
                block.jump(5)
            })
            .expect("extending arm");
        let mut first = None;
        builder
            .in_block(2, |block| {
                first = Some(block.load_arg(mock_i64(), 2)?);
                let flag = block.load_arg(mock_i32(), 1)?;
                block.branch(flag, 3, 4)
            })
            .expect("inner fork");
        let mut second = None;
        builder
            .in_block(3, |block| {
                second = Some(block.load_arg(mock_i64(), 3)?);
                block.jump(4)
            })
            .expect("inner arm");
        let mut inner = None;
        builder
            .in_block(4, |block| {
                inner = Some(block.phi(
                    mock_i64(),
                    [(2, first.expect("first")), (3, second.expect("second"))],
                )?);
                block.jump(5)
            })
            .expect("inner join");
        builder
            .in_block(5, |block| {
                let joined = block.phi(
                    mock_i64(),
                    [
                        (1, extension.expect("extension")),
                        (4, inner.expect("inner")),
                    ],
                )?;
                let low = block.conv(mock_i32(), joined, MockType::I32, false, true)?;
                block.ret(Some(low))
            })
            .expect("outer join");
        let mut ssa = builder.finish().expect("finish");
        run_mock_pass_boundary(&mut ssa, "phi web narrowing", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let join = ssa.block(5).expect("outer join block");
        let phi = join.phi_nodes().first().expect("the join keeps a phi");
        assert_eq!(ssa.bit_width(phi.result()), Some(64), "{ssa}");
    }

    /// **A phi of extensions a whole-register use reads narrows for the phi it
    /// enters.** A word loop counter enters from a phi of sign-extended words
    /// that a 64-bit computation also reads; the phi it enters wants only the
    /// word. The first narrows, its readers keeping its extension, and the
    /// second joins that extension and narrows too. The controls: the entered
    /// phi read whole, or joining a value that is no extension, leaves both
    /// wide, the first having no phi to open.
    #[test]
    fn a_phi_of_extensions_narrows_for_the_word_phi_it_enters() {
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        #[derive(Clone, Copy, PartialEq)]
        enum Entered {
            Low,
            Whole,
            JoinsWhole,
        }
        for entered in [Entered::Low, Entered::Whole, Entered::JoinsWhole] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
            builder
                .in_block(0, |block| {
                    let flag = block.load_arg(mock_i32(), 3)?;
                    block.branch(flag, 1, 2)
                })
                .expect("entry");
            let (mut first, mut second, mut third) = (None, None, None);
            builder
                .in_block(1, |block| {
                    let x = block.load_arg(mock_i32(), 0)?;
                    first = Some(block.conv(i64d(), x, MockType::I64, false, false)?);
                    block.jump(3)
                })
                .expect("first arm");
            builder
                .in_block(2, |block| {
                    let x = block.load_arg(mock_i32(), 1)?;
                    second = Some(block.conv(i64d(), x, MockType::I64, false, false)?);
                    block.jump(3)
                })
                .expect("second arm");
            let (mut outer, mut wide) = (None, None);
            builder
                .in_block(3, |block| {
                    let joined = block.phi(
                        mock_i64(),
                        [(1, first.expect("first")), (2, second.expect("second"))],
                    )?;
                    outer = Some(joined);
                    wide = Some(block.add(i64d(), joined, joined)?);
                    let flag = block.load_arg(mock_i32(), 3)?;
                    block.branch(flag, 4, 5)
                })
                .expect("outer join");
            builder
                .in_block(4, |block| {
                    third = Some(if entered == Entered::JoinsWhole {
                        block.load_arg(mock_i64(), 2)?
                    } else {
                        let x = block.load_arg(mock_i32(), 2)?;
                        block.conv(i64d(), x, MockType::I64, false, false)?
                    });
                    block.jump(5)
                })
                .expect("third arm");
            builder
                .in_block(5, |block| {
                    let joined = block.phi(
                        mock_i64(),
                        [(3, outer.expect("outer")), (4, third.expect("third"))],
                    )?;
                    let read = if entered == Entered::Whole {
                        joined
                    } else {
                        let low = block.conv(mock_i32(), joined, MockType::I32, false, true)?;
                        block.conv(i64d(), low, MockType::I64, false, true)?
                    };
                    let sum = block.add(i64d(), read, wide.expect("wide"))?;
                    block.ret(Some(sum))
                })
                .expect("inner join");
            let mut ssa = builder.finish().expect("finish");
            // One round narrows the phi whose readers open, the next the one
            // it entered.
            for _ in 0..3 {
                run_mock_pass_boundary(&mut ssa, "chained phi narrowing", |ssa| {
                    AlgebraicSimplificationPass.run(
                        ssa,
                        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                    ) == Ok(true)
                });
            }
            let width_at = |at: usize| {
                ssa.block(at)
                    .and_then(|block| block.phi_nodes().first())
                    .and_then(|phi| ssa.bit_width(phi.result()))
            };
            let expected = if entered == Entered::Low { 32 } else { 64 };
            assert_eq!(width_at(3), Some(expected), "{ssa}");
            assert_eq!(width_at(5), Some(expected), "{ssa}");
        }
    }

    /// **A web of phis of extensions narrows as one.** A loop a compiler
    /// copied for each way a test could go carries its word counter
    /// zero-extended through every join: entered from a phi of a phi, and
    /// read only by the loop's truncation, no phi of the web is the one a
    /// single phi's narrowing reads narrow. The web narrows whole in one run,
    /// each phi joining the narrow phis of the others. The controls: the
    /// loop's phi read whole, where no operation reads a phi at its low bits,
    /// and a sign extension joined beside the zero extensions, which leaves the
    /// phis joining it wide and the phi they read with no narrow reader.
    #[test]
    fn a_web_of_phis_of_extensions_narrows_as_one() {
        let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
        #[derive(Clone, Copy, PartialEq)]
        enum Web {
            Low,
            Whole,
            Mixed,
        }
        for web in [Web::Low, Web::Whole, Web::Mixed] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
            builder
                .in_block(0, |block| {
                    let flag = block.load_arg(mock_i32(), 3)?;
                    block.branch(flag, 1, 2)
                })
                .expect("entry");
            let (mut entered, mut zero, mut beside) = (None, None, None);
            builder
                .in_block(1, |block| {
                    let x = block.load_arg(mock_i32(), 0)?;
                    entered = Some(block.conv(i64d(), x, MockType::I64, false, true)?);
                    block.jump(3)
                })
                .expect("extending arm");
            builder
                .in_block(2, |block| {
                    zero = Some(block.const_i64(mock_i64(), 0)?);
                    block.jump(3)
                })
                .expect("constant arm");
            let mut first = None;
            builder
                .in_block(3, |block| {
                    first = Some(block.phi(
                        mock_i64(),
                        [(1, entered.expect("entered")), (2, zero.expect("zero"))],
                    )?);
                    let flag = block.load_arg(mock_i32(), 3)?;
                    block.branch(flag, 4, 5)
                })
                .expect("first join");
            builder
                .in_block(4, |block| {
                    let y = block.load_arg(mock_i32(), 1)?;
                    beside =
                        Some(block.conv(i64d(), y, MockType::I64, false, web != Web::Mixed)?);
                    block.jump(5)
                })
                .expect("second arm");
            let mut second = None;
            builder
                .in_block(5, |block| {
                    second = Some(block.phi(
                        mock_i64(),
                        [(3, first.expect("first")), (4, beside.expect("beside"))],
                    )?);
                    block.jump(6)
                })
                .expect("second join");
            let counter = builder
                .in_block(6, |block| block.empty_phi(mock_i64()))
                .expect("loop phi");
            let stepped = builder
                .in_block(6, |block| {
                    let word = block.conv(mock_i32(), counter, MockType::I32, false, true)?;
                    let one = block.const_i32(mock_i32(), 1)?;
                    let next = block.add(mock_i32(), word, one)?;
                    let stepped = block.conv(i64d(), next, MockType::I64, false, true)?;
                    let flag = block.load_arg(mock_i32(), 2)?;
                    block.branch(flag, 6, 7)?;
                    Ok(stepped)
                })
                .expect("loop");
            builder
                .add_phi_operand(6, counter, 5, second.expect("second"))
                .expect("entering");
            builder
                .add_phi_operand(6, counter, 6, stepped)
                .expect("round");
            builder
                .in_block(7, |block| {
                    if web == Web::Whole {
                        block.ret(Some(counter))
                    } else {
                        let word = block.conv(mock_i32(), counter, MockType::I32, false, true)?;
                        block.ret(Some(word))
                    }
                })
                .expect("exit");
            let mut ssa = builder.finish().expect("finish");
            run_mock_pass_boundary(&mut ssa, "phi web narrowing", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let width_at = |at: usize| {
                ssa.block(at)
                    .and_then(|block| block.phi_nodes().first())
                    .and_then(|phi| ssa.bit_width(phi.result()))
            };
            let expected = if web == Web::Low { 32 } else { 64 };
            for join in [3, 5, 6] {
                assert_eq!(width_at(join), Some(expected), "B{join}: {ssa}");
            }
        }
    }

    /// **A value with its low bit cleared, plus its half, is three halves**,
    /// and a value other than the halved one, or a mask other than `~1`, is
    /// not.
    #[test]
    fn a_cleared_value_and_its_half_are_three_halves() {
        for (mask, other, folds) in [(-2i64, false, true), (-4, false, false), (-2, true, false)] {
            let i64d = || SsaDefSpec::<MockTarget>::tmp(MockType::I64);
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            builder
                .in_block(0, |block| {
                    let x = block.load_arg(mock_i64(), 0)?;
                    let y = block.load_arg(mock_i64(), 1)?;
                    let bits = block.const_i64(i64d(), mask)?;
                    let cleared = block.and(i64d(), x, bits)?;
                    let one = block.const_i64(i64d(), 1)?;
                    let half = block.shr(i64d(), if other { y } else { x }, one, true)?;
                    let sum = block.add(i64d(), cleared, half)?;
                    block.ret(Some(sum))
                })
                .expect("the block");
            let mut ssa = builder.finish().expect("finish");
            run_mock_pass_boundary(&mut ssa, "tripled half", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let multiplied = ssa
                .block(0)
                .expect("the block")
                .instructions()
                .iter()
                .any(|row| matches!(row.op(), SsaOp::Mul { .. }));
            assert_eq!(multiplied, folds, "{ssa}");
        }
    }

    /// A width conversion inside an idiom is kept unless it changes no bit.
    #[test]
    fn a_width_conversion_inside_an_idiom_keeps_its_value() {
        for case in conversion_cases() {
            assert_preserves(&case);
        }
    }

    /// **A copy of a constant into another sign is that constant.** `v0 =
    /// int32_t 1; v1 = uint32_t copy v0; ret v1`: copy propagation refuses a
    /// copy whose readers would read another type, so the fold restates the
    /// constant in the destination's type, `uint32_t 1`. The control: a copy
    /// within one type is copy propagation's, and stays a copy here.
    #[test]
    fn a_copy_of_a_constant_into_another_sign_is_that_constant() {
        let fold = |copied: MockType| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
            let one = ssa.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            );
            let copy = ssa.create_variable(
                VariableOrigin::Local(1),
                0,
                DefSite::instruction(0, 1),
                copied,
            );
            let mut block = SsaBlock::new(0);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: one,
                value: ConstValue::I32(1),
            }));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
                dest: copy,
                src: one,
            }));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(copy))));
            ssa.add_block(block);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "a copied constant", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            mock_op_at(&ssa, 0, 1).clone()
        };
        assert!(
            matches!(
                fold(MockType::U32),
                SsaOp::Const {
                    value: ConstValue::U32(1),
                    ..
                }
            ),
            "restated unsigned"
        );
        assert!(
            matches!(fold(MockType::I32), SsaOp::Copy { .. }),
            "left to copy propagation"
        );
    }

    /// **An address moved back to its base is its base.** A stack pointer
    /// moved down for a buffer and up again, `inttoptr(ptrtoint(p) - 32) +
    /// 32`, is `p`, as is an integer moved down and up, `(x - 32) + 32`. The
    /// control: moved up by another amount, it is another address.
    #[test]
    fn an_address_moved_back_to_its_base_is_its_base() {
        let fold = |ty: MockType, back: i64| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
            let at = |ssa: &mut SsaFunction<MockTarget>, index: usize, ty: MockType| {
                ssa.create_variable(
                    VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
                    0,
                    DefSite::instruction(0, index),
                    ty,
                )
            };
            let pointer = ty == MockType::Ptr;
            let base = at(&mut ssa, 0, ty);
            let integer = at(&mut ssa, 1, MockType::I64);
            let down = at(&mut ssa, 2, MockType::I64);
            let lowered = at(&mut ssa, 3, MockType::I64);
            let moved = at(&mut ssa, 4, ty);
            let up = at(&mut ssa, 5, MockType::I64);
            let back_at = at(&mut ssa, 6, ty);
            let mut block = SsaBlock::new(0);
            let mut push =
                |op: SsaOp<MockTarget>| block.add_instruction(SsaInstruction::synthetic(op));
            push(SsaOp::LoadArg {
                dest: base,
                arg_index: 0,
            });
            if pointer {
                push(SsaOp::PtrToInt {
                    dest: integer,
                    operand: base,
                    target: MockType::I64,
                });
            } else {
                push(SsaOp::Copy {
                    dest: integer,
                    src: base,
                });
            }
            push(SsaOp::Const {
                dest: down,
                value: ConstValue::I64(-32),
            });
            push(SsaOp::Add {
                dest: lowered,
                left: integer,
                right: down,
                flags: None,
            });
            if pointer {
                push(SsaOp::IntToPtr {
                    dest: moved,
                    operand: lowered,
                    target: MockType::Ptr,
                });
            } else {
                push(SsaOp::Copy {
                    dest: moved,
                    src: lowered,
                });
            }
            push(SsaOp::Const {
                dest: up,
                value: ConstValue::I64(back),
            });
            push(SsaOp::Add {
                dest: back_at,
                left: moved,
                right: up,
                flags: None,
            });
            push(SsaOp::returns(Some(back_at)));
            ssa.add_block(block);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "an address moved back", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            matches!(mock_op_at(&ssa, 0, 6), SsaOp::Copy { src, .. } if *src == base)
        };
        assert!(fold(MockType::Ptr, 32), "the pointer");
        assert!(fold(MockType::I64, 32), "the integer");
        assert!(!fold(MockType::Ptr, 16), "the control: another address");
    }

    /// **A join of the stack pointer an arm moved for a call's arguments and
    /// back is the stack pointer.** x86-64 stages a call's stack arguments
    /// on one arm with `sub rsp, 8; push rax` and releases them with `pop
    /// rcx; pop rsi`: from `sp`, already `0x48` into the frame,
    /// `inttoptr(ptrtoint(sp) - 8) - 8`, then `+ 16`. The other arm leaves
    /// `sp` as it is, and the phi joining the two is `sp`. The control:
    /// moved back by another amount, the arm leaves another place, and the
    /// phi stays.
    #[test]
    fn a_join_of_the_stack_pointer_moved_and_back_is_the_stack_pointer() {
        let joined = |back: i64| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 0);
            let at = |ssa: &mut SsaFunction<MockTarget>, block: usize, index: usize, ty| {
                ssa.create_variable(
                    VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
                    0,
                    DefSite::instruction(block, index),
                    ty,
                )
            };
            let entry = at(&mut ssa, 0, 0, MockType::Ptr);
            let test = at(&mut ssa, 0, 1, MockType::I32);
            let sp = at(&mut ssa, 0, 2, MockType::Ptr);
            let integer = at(&mut ssa, 1, 0, MockType::I64);
            let eight = at(&mut ssa, 1, 1, MockType::I64);
            let staged = at(&mut ssa, 1, 2, MockType::I64);
            let moved = at(&mut ssa, 1, 3, MockType::Ptr);
            let pushed = at(&mut ssa, 1, 4, MockType::Ptr);
            let up = at(&mut ssa, 1, 5, MockType::I64);
            let released = at(&mut ssa, 1, 6, MockType::Ptr);
            let join = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::Ptr);
            let push = |block: &mut SsaBlock<MockTarget>, op: SsaOp<MockTarget>| {
                block.add_instruction(SsaInstruction::synthetic(op));
            };
            let mut b0 = SsaBlock::new(0);
            push(
                &mut b0,
                SsaOp::LoadArg {
                    dest: entry,
                    arg_index: 0,
                },
            );
            push(
                &mut b0,
                SsaOp::LoadArg {
                    dest: test,
                    arg_index: 1,
                },
            );
            push(
                &mut b0,
                SsaOp::PtrAdd {
                    dest: sp,
                    base: entry,
                    index: None,
                    stride: 0,
                    offset: -0x48,
                    result_type: MockType::Ptr,
                },
            );
            push(
                &mut b0,
                SsaOp::Branch {
                    condition: test,
                    true_target: 1,
                    false_target: 2,
                },
            );
            let mut b1 = SsaBlock::new(1);
            push(
                &mut b1,
                SsaOp::PtrToInt {
                    dest: integer,
                    operand: sp,
                    target: MockType::I64,
                },
            );
            push(
                &mut b1,
                SsaOp::Const {
                    dest: eight,
                    value: ConstValue::I64(8),
                },
            );
            push(
                &mut b1,
                SsaOp::Sub {
                    dest: staged,
                    left: integer,
                    right: eight,
                    flags: None,
                },
            );
            push(
                &mut b1,
                SsaOp::IntToPtr {
                    dest: moved,
                    operand: staged,
                    target: MockType::Ptr,
                },
            );
            push(
                &mut b1,
                SsaOp::Sub {
                    dest: pushed,
                    left: moved,
                    right: eight,
                    flags: None,
                },
            );
            push(
                &mut b1,
                SsaOp::Const {
                    dest: up,
                    value: ConstValue::I64(back),
                },
            );
            push(
                &mut b1,
                SsaOp::Add {
                    dest: released,
                    left: pushed,
                    right: up,
                    flags: None,
                },
            );
            push(&mut b1, SsaOp::Jump { target: 3 });
            let mut b2 = SsaBlock::new(2);
            push(&mut b2, SsaOp::Jump { target: 3 });
            let mut b3 = SsaBlock::new(3);
            let mut phi = PhiNode::new(join, VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(released, 1));
            phi.add_operand(PhiOperand::new(sp, 2));
            b3.add_phi(phi);
            push(&mut b3, SsaOp::returns(Some(join)));
            for block in [b0, b1, b2, b3] {
                ssa.add_block(block);
            }
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "a join of one place", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            // Every edge carries `sp`, or the phi is gone.
            ssa.block(3).is_some_and(|block| {
                block
                    .phi_nodes()
                    .iter()
                    .all(|phi| phi.operands().iter().all(|operand| operand.value() == sp))
            })
        };
        assert!(joined(16), "the stack pointer before the arguments");
        assert!(!joined(8), "the control: another place");
    }

    /// **A join of one place is decided by its storage, not by what types
    /// say of it.** An integer `0x48` past its root, moved down 8 on one arm
    /// and back up, joins the value the other arm leaves: typed `u64` where the value is `i64`, as
    /// a type lane may state one sign of a register and another of the
    /// join, the phi is that value still, so the decision does not wait on
    /// types being recovered and is not made again once they are. The
    /// control: joined as a pointer, the phi holds another kind, and stays.
    #[test]
    fn a_join_of_one_place_is_decided_by_its_storage() {
        let joined = |join_type: MockType| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 0);
            let at = |ssa: &mut SsaFunction<MockTarget>, block: usize, index: usize, ty| {
                ssa.create_variable(
                    VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
                    0,
                    DefSite::instruction(block, index),
                    ty,
                )
            };
            let root = at(&mut ssa, 0, 0, MockType::I64);
            let test = at(&mut ssa, 0, 1, MockType::I32);
            let into = at(&mut ssa, 0, 2, MockType::I64);
            let value = at(&mut ssa, 0, 3, MockType::I64);
            let eight = at(&mut ssa, 1, 0, MockType::I64);
            let down = at(&mut ssa, 1, 1, MockType::I64);
            let back = at(&mut ssa, 1, 2, MockType::I64);
            let join = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), join_type);
            let push = |block: &mut SsaBlock<MockTarget>, op: SsaOp<MockTarget>| {
                block.add_instruction(SsaInstruction::synthetic(op));
            };
            let mut b0 = SsaBlock::new(0);
            push(
                &mut b0,
                SsaOp::LoadArg {
                    dest: root,
                    arg_index: 0,
                },
            );
            push(
                &mut b0,
                SsaOp::LoadArg {
                    dest: test,
                    arg_index: 1,
                },
            );
            push(
                &mut b0,
                SsaOp::Const {
                    dest: into,
                    value: ConstValue::I64(0x48),
                },
            );
            push(
                &mut b0,
                SsaOp::Add {
                    dest: value,
                    left: root,
                    right: into,
                    flags: None,
                },
            );
            push(
                &mut b0,
                SsaOp::Branch {
                    condition: test,
                    true_target: 1,
                    false_target: 2,
                },
            );
            let mut b1 = SsaBlock::new(1);
            push(
                &mut b1,
                SsaOp::Const {
                    dest: eight,
                    value: ConstValue::I64(8),
                },
            );
            push(
                &mut b1,
                SsaOp::Sub {
                    dest: down,
                    left: value,
                    right: eight,
                    flags: None,
                },
            );
            push(
                &mut b1,
                SsaOp::Add {
                    dest: back,
                    left: down,
                    right: eight,
                    flags: None,
                },
            );
            push(&mut b1, SsaOp::Jump { target: 3 });
            let mut b2 = SsaBlock::new(2);
            push(&mut b2, SsaOp::Jump { target: 3 });
            let mut b3 = SsaBlock::new(3);
            let mut phi = PhiNode::new(join, VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(back, 1));
            phi.add_operand(PhiOperand::new(value, 2));
            b3.add_phi(phi);
            push(&mut b3, SsaOp::returns(Some(join)));
            for block in [b0, b1, b2, b3] {
                ssa.add_block(block);
            }
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "a join decided by its storage", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            ssa.block(3).is_some_and(|block| {
                block.phi_nodes().iter().all(|phi| {
                    phi.operands()
                        .iter()
                        .all(|operand| operand.value() == value)
                })
            })
        };
        assert!(joined(MockType::U64), "another sign of one storage");
        assert!(!joined(MockType::Ptr), "the control: another kind");
    }

    /// **An address moved back to its pointer base as an integer is that
    /// pointer converted.** RISC-V moves a list on with a pointer add and back
    /// with an integer one, `(ap + 56) + -56`: no copy, the types differ, but
    /// `ptrtoint ap`. The controls: moved back by another amount it is another
    /// address, and measured from the entry stack pointer it is a place in
    /// the frame, which keeps its form.
    #[test]
    fn an_address_moved_back_as_an_integer_is_its_base_converted() {
        let fold = |back: i64, stack: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
            let at = |ssa: &mut SsaFunction<MockTarget>, index: usize, ty: MockType| {
                ssa.create_variable(
                    VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
                    0,
                    DefSite::instruction(0, index),
                    ty,
                )
            };
            let base = at(&mut ssa, 0, MockType::Ptr);
            if stack && let Some(variable) = ssa.variable_mut(base) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            let moved = at(&mut ssa, 1, MockType::Ptr);
            let amount = at(&mut ssa, 2, MockType::I64);
            let back_at = at(&mut ssa, 3, MockType::I64);
            let mut block = SsaBlock::new(0);
            let mut push =
                |op: SsaOp<MockTarget>| block.add_instruction(SsaInstruction::synthetic(op));
            push(SsaOp::LoadArg {
                dest: base,
                arg_index: 0,
            });
            push(SsaOp::PtrAdd {
                dest: moved,
                base,
                index: None,
                stride: 0,
                offset: 56,
                result_type: MockType::Ptr,
            });
            push(SsaOp::Const {
                dest: amount,
                value: ConstValue::I64(back),
            });
            push(SsaOp::Add {
                dest: back_at,
                left: moved,
                right: amount,
                flags: None,
            });
            push(SsaOp::returns(Some(back_at)));
            ssa.add_block(block);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "an address moved back as an integer", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            matches!(
                mock_op_at(&ssa, 0, 3),
                SsaOp::PtrToInt { operand, .. } if *operand == base
            )
        };
        assert!(fold(-56, false), "the pointer converted");
        assert!(!fold(-48, false), "the control: another address");
        assert!(!fold(-56, true), "the control: a place in the frame");
    }

    /// **An address a loop carries round is no address of itself.** `p =
    /// phi(arg, q); q = inttoptr(ptrtoint(p) - 32) + 32` moves the pointer
    /// down and back each time round: `q` is `p`, and folds to it, but the
    /// walk stops at the phi, so nothing folds to itself, and the phi is left
    /// joining the argument and itself, for the phi passes to collapse.
    #[test]
    fn an_address_a_loop_carries_round_is_no_address_of_itself() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let var = |ssa: &mut SsaFunction<MockTarget>, block: usize, index: usize, ty: MockType| {
            ssa.create_variable(
                VariableOrigin::Local(u16::try_from(index).unwrap_or(0)),
                0,
                DefSite::instruction(block, index),
                ty,
            )
        };
        let arg = var(&mut ssa, 0, 0, MockType::Ptr);
        let carried = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::Ptr);
        let integer = var(&mut ssa, 1, 0, MockType::I64);
        let down = var(&mut ssa, 1, 1, MockType::I64);
        let lowered = var(&mut ssa, 1, 2, MockType::I64);
        let moved = var(&mut ssa, 1, 3, MockType::Ptr);
        let up = var(&mut ssa, 1, 4, MockType::I64);
        let back = var(&mut ssa, 1, 5, MockType::Ptr);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: arg,
            arg_index: 0,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(carried, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(arg, 0));
        phi.add_operand(PhiOperand::new(back, 1));
        b1.add_phi(phi);
        for op in [
            SsaOp::PtrToInt {
                dest: integer,
                operand: carried,
                target: MockType::I64,
            },
            SsaOp::Const {
                dest: down,
                value: ConstValue::I64(-32),
            },
            SsaOp::Add {
                dest: lowered,
                left: integer,
                right: down,
                flags: None,
            },
            SsaOp::IntToPtr {
                dest: moved,
                operand: lowered,
                target: MockType::Ptr,
            },
            SsaOp::Const {
                dest: up,
                value: ConstValue::I64(32),
            },
            SsaOp::Add {
                dest: back,
                left: moved,
                right: up,
                flags: None,
            },
        ] {
            b1.add_instruction(SsaInstruction::synthetic(op));
        }
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b1);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "an address a loop carries", |ssa| {
            AlgebraicSimplificationPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(
            matches!(mock_op_at(&ssa, 1, 5), SsaOp::Copy { src, .. } if *src == carried),
            "the round trip is the carried pointer: {ssa}"
        );
    }

    /// A `double` an AArch64 loop carries in a `q` register: the register's
    /// storage is joined at the loop head and at the exit, each joined value
    /// the `double` written into the low lane (reinterpreted, shuffled beside
    /// a zero lane, reinterpreted as the storage), each phi read back out by
    /// the reverse. Three times round, the body adds 1.5 and leaves early on
    /// the second pass, so the exit joins the head's register and the body's.
    ///
    /// `entry` is the register's value entering the loop; `whole`, whether the
    /// exit returns the register rather than the `double` read out of it.
    fn lane_loop(entry: ConstValue<MockTarget>, whole: bool) -> SsaFunction<MockTarget> {
        let tmp = SsaDefSpec::<MockTarget>::tmp;
        let unpack = |block: &mut SsaBlockBuilder<'_, MockTarget>, register: SsaVarId| {
            let lanes = block.emit_def(tmp(MockType::V2I64), |dest| SsaOp::VectorReinterpret {
                dest,
                value: register,
                target_type: MockType::V2I64,
            })?;
            let low = block.emit_def(tmp(MockType::V1I64), |dest| SsaOp::VectorShuffle {
                dest,
                left: lanes,
                right: None,
                selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                    VectorShuffleLane::Left(0),
                ])),
            })?;
            block.emit_def(tmp(MockType::F64), |dest| SsaOp::Bitcast {
                dest,
                operand: low,
                target: MockType::F64,
            })
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.ensure_block(3);
        let (start, counter_start) = builder
            .in_block(0, |block| {
                let start = block.const_value(tmp(MockType::V1I128), entry)?;
                let counter = block.const_i32(tmp(MockType::I32), 0)?;
                block.jump(1)?;
                Ok((start, counter))
            })
            .expect("entry");
        let (register, counter) = builder
            .in_block(1, |block| {
                let register =
                    block.empty_phi(SsaDefSpec::new(VariableOrigin::Local(0), MockType::V1I128))?;
                let counter =
                    block.empty_phi(SsaDefSpec::new(VariableOrigin::Local(1), MockType::I32))?;
                let limit = block.const_i32(tmp(MockType::I32), 3)?;
                let more = block.clt(tmp(MockType::I32), counter, limit, false)?;
                block.branch(more, 2, 3)?;
                Ok((register, counter))
            })
            .expect("head");
        let (packed, next) = builder
            .in_block(2, |block| {
                let value = unpack(block, register)?;
                let step = block.const_value(tmp(MockType::F64), ConstValue::from(1.5_f64))?;
                let sum = block.add(tmp(MockType::F64), value, step)?;
                let scalar =
                    block.emit_def(tmp(MockType::V1I64), |dest| SsaOp::VectorReinterpret {
                        dest,
                        value: sum,
                        target_type: MockType::V1I64,
                    })?;
                let widened =
                    block.emit_def(tmp(MockType::V2I64), |dest| SsaOp::VectorShuffle {
                        dest,
                        left: scalar,
                        right: None,
                        selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                            VectorShuffleLane::Left(0),
                            VectorShuffleLane::Zero,
                        ])),
                    })?;
                let packed =
                    block.emit_def(tmp(MockType::V1I128), |dest| SsaOp::VectorReinterpret {
                        dest,
                        value: widened,
                        target_type: MockType::V1I128,
                    })?;
                let one = block.const_i32(tmp(MockType::I32), 1)?;
                let next = block.add(tmp(MockType::I32), counter, one)?;
                let two = block.const_i32(tmp(MockType::I32), 2)?;
                let early = block.ceq(tmp(MockType::I32), next, two)?;
                block.branch(early, 3, 1)?;
                Ok((packed, next))
            })
            .expect("body");
        for (result, from_entry, from_body) in
            [(register, start, packed), (counter, counter_start, next)]
        {
            builder
                .add_phi_operand(1, result, 0, from_entry)
                .expect("entry edge");
            builder
                .add_phi_operand(1, result, 2, from_body)
                .expect("back edge");
        }
        builder
            .in_block(3, |block| {
                let out = block.phi(
                    SsaDefSpec::new(VariableOrigin::Local(0), MockType::V1I128),
                    [(1, register), (2, packed)],
                )?;
                let read = if whole { out } else { unpack(block, out)? };
                block.ret(Some(read))
            })
            .expect("exit");
        builder.finish().expect("finish")
    }

    /// The constant `ssa` returns, executed.
    fn returned(ssa: &SsaFunction<MockTarget>) -> Option<ConstValue<MockTarget>> {
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(ssa, &context).run();
        facts.returned().and_then(|value| value.constant().cloned())
    }

    /// A 128-bit register constant of two 64-bit lanes.
    fn register(low: u64, high: u64) -> ConstValue<MockTarget> {
        ConstValue::Vector(
            VectorConst::from_lanes(128, [u128::from(low) | (u128::from(high) << 64)])
                .expect("one lane"),
        )
    }

    /// **A phi of a value's reshapes is the reshape of the value's phi.** The
    /// loop's register phis, head and exit, become `double` phis - the entry's
    /// zero register joining as `0.0`, a register holding `2.0` as `2.0` - and
    /// each register is the `double` written back into it at the top of its
    /// block, so the function returns what it returned (executed, against the
    /// sums worked by hand: 3.0 from zero, 5.0 from 2.0). Once the algebraic
    /// rules have run to a fixed point, the `double` the exit returns is the
    /// exit's `double` phi itself.
    ///
    /// The controls: a register entering with its upper lane set is no
    /// reshape of any `double`, and a register the exit returns whole is read
    /// past the `double`; both webs are left as they are.
    #[test]
    fn a_phi_of_reshaped_values_is_the_reshape_of_their_phi() {
        let two = 2.0_f64.to_bits();
        for (entry, whole, expected) in [
            (register(0, 0), false, Some(3.0_f64)),
            (register(two, 0), false, Some(5.0)),
            (register(two, 1), false, None),
            (register(0, 0), true, None),
        ] {
            let mut ssa = lane_loop(entry, whole);
            let before = returned(&ssa);
            let changed = run_mock_pass_boundary(&mut ssa, "reshaped phis", |ssa| {
                unreshape_phis(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                )
            });
            let phi_types: Vec<_> = [1, 3]
                .into_iter()
                .filter_map(|block| ssa.block(block))
                .flat_map(|block| block.phi_nodes())
                .filter_map(|phi| ssa.variable(phi.result()).map(|v| *v.var_type()))
                .filter(|ty| *ty != MockType::I32)
                .collect();
            let Some(sum) = expected else {
                assert!(!changed, "{ssa}");
                assert_eq!(phi_types, [MockType::V1I128, MockType::V1I128], "{ssa}");
                continue;
            };
            assert!(changed, "{ssa}");
            assert_eq!(phi_types, [MockType::F64, MockType::F64], "{ssa}");
            assert_eq!(before, Some(ConstValue::from(sum)), "{ssa}");
            assert_eq!(returned(&ssa), before, "the same value returns: {ssa}");
            while run_mock_pass_boundary(&mut ssa, "algebraic", |ssa| {
                AlgebraicSimplificationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            }) {}
            let exit = ssa.block(3).expect("exit");
            let joined = exit.phi_nodes().first().expect("the exit's phi").result();
            let Some(SsaOp::Return {
                value: Some(value), ..
            }) = exit.instructions().last().map(|instr| instr.op())
            else {
                panic!("the exit returns: {ssa}");
            };
            assert!(
                matches!(
                    ssa.recorded_definition(*value).map(|definition| definition.op()),
                    Some(SsaOp::Copy { src, .. }) if *src == joined
                ),
                "the double returned is the exit's double phi: {ssa}"
            );
            assert_eq!(returned(&ssa), before, "the same value returns: {ssa}");
        }
    }
}
