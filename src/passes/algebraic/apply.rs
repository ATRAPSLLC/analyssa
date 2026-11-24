//! Applying the simplifications a round found: each candidate rewrites its
//! instruction, those that insert instructions from the bottom of each block
//! up.

use std::collections::BTreeSet;

use crate::{
    Error, Result,
    analysis::{
        cfg::SsaCfg,
        convert::NarrowValue,
        distribute::Factoring,
        doubleword,
        doubleword::{DoublewordFold, Regrouping, StorePair, Term, WideLoad},
    },
    events::EventKind,
    facts::Substitution,
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        function::{SsaEditOptions, SsaEditor, SsaFunction},
        instruction::SsaInstruction,
        integer::BitWidth,
        ops::SsaOp,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::algebraic::{ShiftedPart, Simplification, SimplificationCandidate},
    scheduling::PassEnv,
    target::Target,
};

/// Rewrite every candidate in a single edit scope and report each change.
///
/// Opens one [`SsaEditOptions`] edit and, for each candidate, materialises its
/// [`Simplification`] into a concrete op - `Const`, `Copy`, or the pre-built
/// rewritten op - and replaces the instruction in place. One
/// [`EventKind::ConstantFolded`] event carrying the candidate's identity
/// description is pushed per rewrite.
///
/// On edit failure the function returns `true` even though `changed` may be
/// unset: the session runs under `SsaRollbackPolicy::Never`, so a failed edit
/// or boundary repair leaves the IR mutated and possibly mid-repair. Reporting
/// "unchanged" would make the pass-group transaction skip both verification and
/// rollback, keeping damaged IR unchecked; reporting the change forces the
/// transaction to verify and roll back.
///
/// # Arguments
///
/// * `ssa` - The SSA function to rewrite in place.
/// * `candidates` - Simplifications gathered by [`find_candidates`](crate::passes::algebraic::find_candidates).
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// `true` if any operation was rewritten (or the edit failed, per above), else `false`.
pub fn apply_simplifications<T>(
    ssa: &mut SsaFunction<T>,
    candidates: Vec<SimplificationCandidate<T>>,
    env: &PassEnv<'_, T>,
) -> bool
where
    T: Target,
{
    let mut changed = false;
    // Every other rewrite keeps the value its destination holds, so rewrites
    // of one round compose in any order. A fold that redefines what it reads
    // ([`SimplificationCandidate::also`]) does not: it gives those values new
    // ones, and its own result reads them through the readers it matched -
    // `trunc32(a ^ b) + (trunc32(a & b) << 1)` makes `a ^ b` the sum and the
    // result its truncation. A rewrite of one of those readers in the same
    // round - the truncation narrowed to `trunc32(a) ^ trunc32(b)` - would
    // leave the result reading the old value. So the values it redefines and
    // their readers are its for the round, and a rewrite of any of them waits
    // for the next.
    let redefined: BTreeSet<SsaVarId> = candidates
        .iter()
        .flat_map(|candidate| candidate.also.iter().map(|(var, _)| *var))
        .collect();
    let candidates: Vec<SimplificationCandidate<T>> = if redefined.is_empty() {
        candidates
    } else {
        let mut claimed = redefined.clone();
        for block in ssa.blocks() {
            for instruction in block.instructions() {
                let op = instruction.op();
                if op.uses().iter().any(|used| redefined.contains(used)) {
                    claimed.extend(op.dest());
                }
            }
        }
        candidates
            .into_iter()
            .filter(|candidate| !candidate.also.is_empty() || !claimed.contains(&candidate.dest))
            .collect()
    };
    // A fold of several results, a narrowing that defines narrow constants
    // and a shifted part's steps insert instructions, which moves every later
    // one in their block; the other candidates name their instruction by
    // position, so these go last, from the bottom of each block up.
    let (mut inserting, candidates): (Vec<_>, Vec<_>) =
        candidates.into_iter().partition(|candidate| {
            matches!(
                candidate.simplification,
                Simplification::Results(_)
                    | Simplification::Narrowed(..)
                    | Simplification::Doubleword { .. }
                    | Simplification::Regrouped(_)
                    | Simplification::StorePair(_)
                    | Simplification::Factored(_)
                    | Simplification::ShiftedPart(_)
            )
        });
    inserting
        .sort_by_key(|candidate| std::cmp::Reverse((candidate.block_idx, candidate.instr_idx)));
    // Which readers of a half a sum built in its place reaches. The edits
    // below insert and rewrite values but no terminator, so the tree stays
    // the function's.
    let dominators = inserting
        .iter()
        .any(|candidate| {
            matches!(
                candidate.simplification,
                Simplification::Doubleword {
                    read_apart: true,
                    ..
                }
            )
        })
        .then(|| compute_dominators(&SsaCfg::from_ssa(ssa), NodeId::new(0)));
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for candidate in candidates.into_iter().chain(inserting) {
            let description = candidate.description;
            if let Some(location) = candidate.apply(editor, dominators.as_ref())? {
                changed = true;
                env.record(EventKind::ConstantFolded)
                    .location(location)
                    .message(description.to_string());
            }
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

/// Has every reader of `fold`'s halves that the sum `wide`, at `at`, dominates
/// read that half of the sum instead: the low half its truncation, the high
/// half its truncation shifted down by the half width, defined just after it.
///
/// The halves are one double-width value's, so either is that value's part
/// wherever the value is known; read apart, their own arithmetic, the carries
/// included, would stay for a reader the sum already answers.
fn reread_halves<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    (block, at): (usize, usize),
    wide: SsaVarId,
    fold: &DoublewordFold,
    dominators: Option<&DominatorTree>,
) -> Result<()> {
    let [low, high] = fold.halves;
    let type_of = |editor: &SsaEditor<'_, T>, var: SsaVarId| {
        editor
            .function()
            .variable(var)
            .map(|variable| variable.var_type().clone())
    };
    let (Some(low_type), Some(high_type), Some(wide_type)) = (
        type_of(editor, low),
        type_of(editor, high),
        type_of(editor, wide),
    ) else {
        return Ok(());
    };
    let mut position = at.saturating_add(1);
    let mut define =
        |editor: &mut SsaEditor<'_, T>, ty: &T::Type, op: &dyn Fn(SsaVarId) -> SsaOp<T>| {
            let dest = editor.create_fresh_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(block, position),
                ty.clone(),
            );
            editor.insert_instruction(block, position, SsaInstruction::synthetic(op(dest)))?;
            position = position.saturating_add(1);
            Ok::<_, Error>(dest)
        };
    let narrow = |operand: SsaVarId, target: T::Type| {
        move |dest: SsaVarId| SsaOp::IntConv {
            dest,
            operand,
            target: target.clone(),
            overflow_check: false,
            unsigned: true,
        }
    };
    let lower = define(editor, &low_type, &narrow(wide, low_type.clone()))?;
    let amount = fold.amount;
    let shifted = define(editor, &wide_type, &|dest| SsaOp::Shr {
        dest,
        value: wide,
        amount,
        unsigned: true,
        flags: None,
    })?;
    let upper = define(editor, &high_type, &narrow(shifted, high_type.clone()))?;
    editor.replace_uses_checked_with(low, lower, dominators, Substitution::Algebraic);
    editor.replace_uses_checked_with(high, upper, dominators, Substitution::Algebraic);
    Ok(())
}

/// Composes a value of type `wide` from a `low` and a `high` half just before
/// position `at` of `block`, advancing `at` past what it defines: the low half
/// zero-extended, or-ed with the high half's terms summed at the half width,
/// zero-extended and shifted up by `amount`; the low half alone where the high
/// one is zero, the high half alone where there is no low one, and `None`
/// where neither is.
fn compose_halves<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    block: usize,
    at: &mut usize,
    wide: &T::Type,
    low: Option<SsaVarId>,
    high: &[Term],
    amount: SsaVarId,
) -> Result<Option<SsaVarId>> {
    let define = |editor: &mut SsaEditor<'_, T>,
                  at: &mut usize,
                  ty: &T::Type,
                  op: &dyn Fn(SsaVarId) -> SsaOp<T>|
     -> Result<SsaVarId> {
        let dest = editor.create_fresh_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(block, *at),
            ty.clone(),
        );
        editor.insert_instruction(block, *at, SsaInstruction::synthetic(op(dest)))?;
        *at = at.saturating_add(1);
        Ok(dest)
    };
    let extend = |operand: SsaVarId| {
        move |dest: SsaVarId| SsaOp::IntConv {
            dest,
            operand,
            target: wide.clone(),
            overflow_check: false,
            unsigned: true,
        }
    };
    // A half is an integer half the composition's width, whatever types the
    // values it is computed from were given.
    let half = T::bit_width(wide)
        .and_then(|bits| bits.checked_div(2))
        .and_then(|bits| T::integer_type(bits, false));
    let lower = match low {
        None => None,
        Some(low) => {
            // A low half of no stated width is read at the half width, as the
            // half-wide arithmetic that made it a half reads it: its zero
            // extension then states which bits of the composition it holds.
            let low = match &half {
                Some(half) if editor.function().bit_width(low).is_none() => {
                    define(editor, at, half, &|dest| SsaOp::IntConv {
                        dest,
                        operand: low,
                        target: half.clone(),
                        overflow_check: false,
                        unsigned: true,
                    })?
                }
                _ => low,
            };
            Some(define(editor, at, wide, &extend(low))?)
        }
    };
    let Some(half) = half else {
        return Ok(lower);
    };
    // The high half's terms, summed at the half width: a high half that is a
    // single added term is that term.
    let mut summed: Option<SsaVarId> = None;
    for term in high {
        summed = Some(match summed {
            None if !term.negative => term.value,
            None => define(editor, at, &half, &|dest| SsaOp::Neg {
                dest,
                operand: term.value,
                flags: None,
            })?,
            Some(sum) if term.negative => define(editor, at, &half, &|dest| SsaOp::Sub {
                dest,
                left: sum,
                right: term.value,
                flags: None,
            })?,
            Some(sum) => define(editor, at, &half, &|dest| SsaOp::Add {
                dest,
                left: sum,
                right: term.value,
                flags: None,
            })?,
        });
    }
    let Some(summed) = summed else {
        return Ok(lower);
    };
    let upper = define(editor, at, wide, &extend(summed))?;
    let shifted = define(editor, at, wide, &|dest| SsaOp::Shl {
        dest,
        value: upper,
        amount,
        flags: None,
    })?;
    let Some(lower) = lower else {
        return Ok(Some(shifted));
    };
    define(editor, at, wide, &|dest| SsaOp::Or {
        dest,
        left: shifted,
        right: lower,
        flags: None,
    })
    .map(Some)
}

impl<T: Target> SimplificationCandidate<T> {
    /// Applies the candidate through `editor`: the position of the
    /// instruction it leaves rewritten, which its event names, or `None`
    /// where the function no longer has what the candidate found.
    /// `dominators` is the function's dominance, where a double-width sum
    /// read apart needs it.
    pub fn apply(
        self,
        editor: &mut SsaEditor<'_, T>,
        dominators: Option<&DominatorTree>,
    ) -> Result<Option<usize>> {
        let Self {
            block_idx,
            instr_idx,
            dest,
            simplification,
            also,
            ..
        } = self;
        let mut rewrite = Rewrite {
            editor,
            block: block_idx,
            instr: instr_idx,
        };
        match simplification {
            Simplification::Results(results) => rewrite.results(results),
            Simplification::Factored(factoring) => rewrite.factored(dest, factoring),
            Simplification::WideLoad(wide) => rewrite.wide_load(dest, *wide),
            Simplification::Doubleword { fold, read_apart } => {
                rewrite.doubleword(dest, dominators, fold, read_apart)
            }
            Simplification::Regrouped(regrouping) => rewrite.regrouped(dest, *regrouping),
            Simplification::StorePair(pair) => rewrite.store_pair(pair),
            Simplification::ShiftedPart(part) => rewrite.shifted_part(dest, part),
            Simplification::Narrowed(op, inserted, narrow_type) => {
                rewrite.narrowed(*op, inserted, narrow_type)
            }
            other => rewrite.replaced(dest, other, also),
        }
    }
}

/// One candidate's rewrite: the editor it goes through and the instruction it
/// rewrites, the position every applier starts from.
struct Rewrite<'e, 's, T: Target> {
    /// The checked editor the rewrite goes through.
    editor: &'e mut SsaEditor<'s, T>,
    /// The block of the instruction rewritten.
    block: usize,
    /// The instruction's position in its block.
    instr: usize,
}

impl<T: Target> Rewrite<'_, '_, T> {
    /// Places the constants of every result an operation defines: the first in its place, the rest after it.
    /// ([`Simplification::Results`])
    fn results(&mut self, results: Vec<SsaOp<T>>) -> Result<Option<usize>> {
        let mut results = results.into_iter();
        self.editor.replace_instruction_op(
            self.block,
            self.instr,
            results.next().unwrap_or(SsaOp::Nop),
        )?;
        for (offset, result) in (1..).zip(results) {
            self.editor.insert_instruction(
                self.block,
                self.instr.saturating_add(offset),
                SsaInstruction::synthetic(result),
            )?;
        }
        Ok(Some(self.instr))
    }

    /// Takes a shared operand out of a join: the masks joined first, then the operand under them.
    /// ([`Simplification::Factored`])
    fn factored(&mut self, dest: SsaVarId, factoring: Factoring) -> Result<Option<usize>> {
        let Some(ty) = self
            .editor
            .function()
            .variable(dest)
            .map(|variable| variable.var_type().clone())
        else {
            return Ok(None);
        };
        let joined = self.editor.create_fresh_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(self.block, self.instr),
            ty,
        );
        self.editor.insert_instruction(
            self.block,
            self.instr,
            SsaInstruction::synthetic(factoring.outer.op(joined, factoring.left, factoring.right)),
        )?;
        let at = self.instr.saturating_add(1);
        self.editor.replace_instruction_op(
            self.block,
            at,
            factoring.inner.op(dest, factoring.shared, joined),
        )?;
        Ok(Some(at))
    }

    /// Loads two adjacent words as one, where no word load also writes a slot the function reads as storage.
    /// ([`Simplification::WideLoad`])
    fn wide_load(&mut self, dest: SsaVarId, wide: WideLoad<T>) -> Result<Option<usize>> {
        let WideLoad {
            load,
            loads,
            words: [start, rest],
        } = wide;
        // A word load into a version of a slot the function reads as
        // storage also writes that slot; the fold would drop the write.
        let slots = self.editor.function().storage_slots();
        let writes_slot = loads.iter().any(|&position| {
            self.editor
                .function()
                .block(self.block)
                .and_then(|block| block.instruction(position))
                .is_some_and(|instr| slots.written_by(self.editor.function(), instr.op()))
        });
        if writes_slot {
            return Ok(None);
        }
        self.editor
            .replace_instruction_op(self.block, self.instr, load)?;
        // The one load reads the value where its first word load
        // read, which is where the value is kept.
        self.editor.note_whole(dest, start, Some(rest));
        for position in loads {
            self.editor.nop_instruction(self.block, position)?;
        }
        Ok(Some(self.instr))
    }

    /// Computes a double-width sum built in halves at its width, and has the readers of its halves read the sum where it reaches them.
    /// ([`Simplification::Doubleword`])
    fn doubleword(
        &mut self,
        dest: SsaVarId,
        dominators: Option<&DominatorTree>,
        fold: DoublewordFold,
        read_apart: bool,
    ) -> Result<Option<usize>> {
        let Some(wide) = self
            .editor
            .function()
            .variable(dest)
            .map(|variable| variable.var_type().clone())
        else {
            return Ok(None);
        };
        let mut at = self.instr;
        let mut sum: Option<SsaVarId> = None;
        let last = fold.operands.len().saturating_sub(1);
        for (position, operand) in fold.operands.iter().enumerate() {
            let Some(composed) = compose_halves(
                self.editor,
                self.block,
                &mut at,
                &wide,
                Some(operand.low),
                &operand.high,
                fold.amount,
            )?
            else {
                return Ok(None);
            };
            // An operand read from where it was kept - a `long long`
            // reloaded from its two words - is kept there whole.
            if let [
                doubleword::Term {
                    value: high,
                    negative: false,
                },
            ] = operand.high.as_slice()
                && let Some([start, rest]) = fold.in_memory_order(operand.low, *high)
            {
                self.editor.note_whole(composed, start, Some(rest));
            }
            let Some(left) = sum else {
                sum = Some(composed);
                continue;
            };
            let op = |dest: SsaVarId| {
                if operand.negative {
                    SsaOp::Sub {
                        dest,
                        left,
                        right: composed,
                        flags: None,
                    }
                } else {
                    SsaOp::Add {
                        dest,
                        left,
                        right: composed,
                        flags: None,
                    }
                }
            };
            let summed = if position == last {
                self.editor
                    .replace_instruction_op(self.block, at, op(dest))?;
                dest
            } else {
                let dest = self.editor.create_fresh_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(self.block, at),
                    wide.clone(),
                );
                self.editor.insert_instruction(
                    self.block,
                    at,
                    SsaInstruction::synthetic(op(dest)),
                )?;
                at = at.saturating_add(1);
                sum = Some(dest);
                dest
            };
            // Each sum along the way is kept where its halves were: a
            // `long long` accumulated in registers and spilled to its
            // slot across a call.
            if let Some((low, Some(high))) =
                fold.partial_halves.get(position.saturating_sub(1)).copied()
                && let Some([start, rest]) = fold.in_memory_order(low, high)
            {
                self.editor.note_whole(summed, start, Some(rest));
            }
        }
        if read_apart {
            reread_halves(self.editor, (self.block, at), dest, &fold, dominators)?;
        }
        Ok(Some(at))
    }

    /// Sums the values whose halves a double-width sum added apart, with what is left of its high halves and its other terms.
    /// ([`Simplification::Regrouped`])
    fn regrouped(&mut self, dest: SsaVarId, regrouping: Regrouping) -> Result<Option<usize>> {
        let Some(wide) = self
            .editor
            .function()
            .variable(dest)
            .map(|variable| variable.var_type().clone())
        else {
            return Ok(None);
        };
        let Regrouping {
            whole,
            composed,
            kept,
            amount,
        } = regrouping;
        // The sum starts from a term it adds; one of only subtracted terms is
        // left as it is.
        if composed.iter().all(|composition| composition.negative)
            && whole.iter().chain(&kept).all(|term| term.negative)
        {
            return Ok(None);
        }
        let mut at = self.instr;
        let mut terms: Vec<Term> = Vec::with_capacity(
            composed
                .len()
                .saturating_add(whole.len())
                .saturating_add(kept.len()),
        );
        for composition in &composed {
            if let Some(value) = compose_halves(
                self.editor,
                self.block,
                &mut at,
                &wide,
                composition.low,
                &composition.high,
                amount,
            )? {
                terms.push(Term {
                    value,
                    negative: composition.negative,
                });
            }
        }
        terms.extend(whole);
        terms.extend(kept);
        let Some(start) = terms.iter().position(|term| !term.negative) else {
            return Ok(None);
        };
        let first = terms.remove(start);
        if terms.is_empty() {
            self.editor.replace_instruction_op(
                self.block,
                at,
                SsaOp::Copy {
                    dest,
                    src: first.value,
                },
            )?;
            return Ok(Some(at));
        }
        let last = terms.len().saturating_sub(1);
        let mut sum = first.value;
        for (position, term) in terms.into_iter().enumerate() {
            let op = |dest: SsaVarId| {
                if term.negative {
                    SsaOp::Sub {
                        dest,
                        left: sum,
                        right: term.value,
                        flags: None,
                    }
                } else {
                    SsaOp::Add {
                        dest,
                        left: sum,
                        right: term.value,
                        flags: None,
                    }
                }
            };
            if position == last {
                self.editor
                    .replace_instruction_op(self.block, at, op(dest))?;
            } else {
                let partial = self.editor.create_fresh_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(self.block, at),
                    wide.clone(),
                );
                self.editor.insert_instruction(
                    self.block,
                    at,
                    SsaInstruction::synthetic(op(partial)),
                )?;
                at = at.saturating_add(1);
                sum = partial;
            }
        }
        Ok(Some(at))
    }

    /// Stores a double-width value whose halves were stored to one cell's two words as one store.
    /// ([`Simplification::StorePair`])
    fn store_pair(&mut self, pair: StorePair) -> Result<Option<usize>> {
        let Some(wide) = pair
            .half
            .checked_mul(2)
            .and_then(|bits| T::integer_type(bits, false))
        else {
            return Ok(None);
        };
        let Some(amount_type) = T::integer_type(8, false) else {
            return Ok(None);
        };
        let Some(amount_value) = u8::try_from(pair.half)
            .ok()
            .and_then(|half| ConstValue::integer(8, false, i64::from(half)))
        else {
            return Ok(None);
        };
        let mut at = self.instr;
        let amount = self.editor.create_fresh_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(self.block, at),
            amount_type,
        );
        self.editor.insert_instruction(
            self.block,
            at,
            SsaInstruction::synthetic(SsaOp::Const {
                dest: amount,
                value: amount_value,
            }),
        )?;
        at = at.saturating_add(1);
        let Some(composed) = compose_halves(
            self.editor,
            self.block,
            &mut at,
            &wide,
            Some(pair.low),
            &[Term {
                value: pair.high,
                negative: false,
            }],
            amount,
        )?
        else {
            return Ok(None);
        };
        self.editor.replace_instruction_op(
            self.block,
            at,
            SsaOp::StoreIndirect {
                addr: pair.address,
                value: composed,
                value_type: wide,
                access: pair.space,
            },
        )?;
        self.editor.nop_instruction(self.block, pair.first)?;
        Ok(Some(at))
    }

    /// Writes the part of a value a pair of shifts keeps as its extension and move.
    /// ([`Simplification::ShiftedPart`])
    fn shifted_part(&mut self, dest: SsaVarId, part: Box<ShiftedPart<T>>) -> Result<Option<usize>> {
        let part = part.as_ref();
        let block = self.block;
        let mut at = self.instr;
        let mut define =
            |editor: &mut SsaEditor<'_, T>, ty: &T::Type, op: &dyn Fn(SsaVarId) -> SsaOp<T>| {
                let dest = editor.create_fresh_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(block, at),
                    ty.clone(),
                );
                editor.insert_instruction(block, at, SsaInstruction::synthetic(op(dest)))?;
                at = at.saturating_add(1);
                Ok::<_, Error>(dest)
            };
        let amount = |shift: u32| ConstValue::integer(part.amount_bits, false, i64::from(shift));
        let (Some(down), Some(up)) = (amount(part.low), amount(part.up)) else {
            return Ok(None);
        };
        // The part's bits brought down to the bottom.
        let mut kept = part.source;
        if part.low > 0 {
            let shift = define(self.editor, &part.amount, &|dest| SsaOp::Const {
                dest,
                value: down.clone(),
            })?;
            kept = define(self.editor, &part.wide, &|dest| SsaOp::Shr {
                dest,
                value: kept,
                amount: shift,
                unsigned: true,
                flags: None,
            })?;
        }
        // Extended to the shifts' width: the truncation to the
        // part's type, extended, or a mask that keeps it.
        let wide = part.wide.clone();
        let signed = part.signed;
        let extend: Box<dyn Fn(SsaVarId) -> SsaOp<T>> = match &part.narrow {
            Some(narrow) => {
                let narrowed = define(self.editor, narrow, &|dest| SsaOp::IntConv {
                    dest,
                    operand: kept,
                    target: narrow.clone(),
                    overflow_check: false,
                    unsigned: !signed,
                })?;
                Box::new(move |dest| SsaOp::IntConv {
                    dest,
                    operand: narrowed,
                    target: wide.clone(),
                    overflow_check: false,
                    unsigned: !signed,
                })
            }
            None => {
                let Some(mask) = (part.bits != 0)
                    .then(|| BitWidth::new(part.bits).mask())
                    .and_then(|mask| i64::try_from(mask).ok())
                    .and_then(|mask| ConstValue::integer(part.width, false, mask))
                else {
                    return Ok(None);
                };
                let mask = define(self.editor, &part.wide, &|dest| SsaOp::Const {
                    dest,
                    value: mask.clone(),
                })?;
                Box::new(move |dest| SsaOp::And {
                    dest,
                    left: kept,
                    right: mask,
                    flags: None,
                })
            }
        };
        let op = if part.up > 0 {
            let whole = define(self.editor, &part.wide, &*extend)?;
            let shift = define(self.editor, &part.amount, &|dest| SsaOp::Const {
                dest,
                value: up.clone(),
            })?;
            SsaOp::Shl {
                dest,
                value: whole,
                amount: shift,
                flags: None,
            }
        } else {
            extend(dest)
        };
        self.editor.replace_instruction_op(self.block, at, op)?;
        Ok(Some(at))
    }

    /// Computes an operation at the narrow width, defining first the narrow values it reads.
    /// ([`Simplification::Narrowed`])
    fn narrowed(
        &mut self,
        op: SsaOp<T>,
        inserted: Vec<(SsaVarId, NarrowValue<T>)>,
        narrow_type: T::Type,
    ) -> Result<Option<usize>> {
        let mut op = op;
        let mut at = self.instr;
        // Each wide value an entry defined at the narrow width, for a
        // later entry that reads it (a scaled index).
        let mut defined: Vec<(SsaVarId, SsaVarId)> = Vec::new();
        for (wide, value) in inserted {
            let narrow = self.editor.create_fresh_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(self.block, at),
                narrow_type.clone(),
            );
            let definition = match value {
                NarrowValue::Scaled { index, stride } => {
                    let index = defined
                        .iter()
                        .find(|(wide, _)| *wide == index)
                        .map_or(index, |(_, narrow)| *narrow);
                    let factor = self.editor.create_fresh_variable(
                        VariableOrigin::Phi,
                        0,
                        DefSite::instruction(self.block, at),
                        narrow_type.clone(),
                    );
                    self.editor.insert_instruction(
                        self.block,
                        at,
                        SsaInstruction::synthetic(SsaOp::Const {
                            dest: factor,
                            value: stride,
                        }),
                    )?;
                    at = at.saturating_add(1);
                    SsaOp::Mul {
                        dest: narrow,
                        left: index,
                        right: factor,
                        flags: None,
                    }
                }
                NarrowValue::Constant(value) => SsaOp::Const {
                    dest: narrow,
                    value,
                },
                NarrowValue::Extension { source, unsigned } => SsaOp::IntConv {
                    dest: narrow,
                    operand: source,
                    target: narrow_type.clone(),
                    overflow_check: false,
                    unsigned,
                },
                NarrowValue::Truncation { source } => SsaOp::IntConv {
                    dest: narrow,
                    operand: source,
                    target: narrow_type.clone(),
                    overflow_check: false,
                    unsigned: false,
                },
            };
            self.editor.insert_instruction(
                self.block,
                at,
                SsaInstruction::synthetic(definition),
            )?;
            at = at.saturating_add(1);
            op.replace_uses(wide, narrow);
            defined.push((wide, narrow));
        }
        self.editor.replace_instruction_op(self.block, at, op)?;
        Ok(Some(at))
    }

    /// Replaces the instruction by the one operation `simplification` states
    /// (a constant, a copy, a rewrite, a retyped rewrite), first restating the
    /// values the candidate also redefines (`also`).
    fn replaced(
        &mut self,
        dest: SsaVarId,
        simplification: Simplification<T>,
        also: Vec<(SsaVarId, SsaOp<T>)>,
    ) -> Result<Option<usize>> {
        let (new_op, dest_type) = match simplification {
            Simplification::Constant(value) => (SsaOp::Const { dest, value }, None),
            Simplification::Copy(src) => (SsaOp::Copy { dest, src }, None),
            Simplification::Rewrite(op) => (*op, None),
            Simplification::Retyped(op, dest_type) => (*op, Some(dest_type)),
            Simplification::Results(_)
            | Simplification::Narrowed(..)
            | Simplification::Doubleword { .. }
            | Simplification::Regrouped(_)
            | Simplification::StorePair(_)
            | Simplification::WideLoad(_)
            | Simplification::Factored(_)
            | Simplification::ShiftedPart(_) => return Ok(None),
        };
        // Before the replacement, because it reads what these produce.
        // Each names a variable rather than a position: a recogniser works
        // over values, and the definition's coordinates are the function's
        // to know. A variable with no instruction definition is a phi,
        // which no recogniser asks to rewrite.
        for (var, op) in also {
            let Some((block, instruction)) = self
                .editor
                .function()
                .recorded_definition(var)
                .map(|definition| (definition.block_index(), definition.instruction_index()))
            else {
                continue;
            };
            self.editor.replace_instruction_op(block, instruction, op)?;
        }
        match dest_type {
            Some(dest_type) => self
                .editor
                .replace_instruction_op_as(self.block, self.instr, new_op, dest_type)?,
            None => self
                .editor
                .replace_instruction_op(self.block, self.instr, new_op)?,
        }
        Ok(Some(self.instr))
    }
}
