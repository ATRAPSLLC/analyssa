//! Merging blocks: duplicates, repeated tests, forwarders, split loops and
//! unconditional successors.

use std::collections::BTreeMap;

use crate::{
    error::{Error, Result},
    facts::{Derivation, InstructionFacts, Join},
    ir::{
        block::SsaBlock,
        function::{
            SsaFunction,
            editor::{SsaEditScope, SsaEditor},
        },
        instruction::SsaInstruction,
        ops::{OperandRole, SsaOp},
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, SsaVariable},
    },
    target::Target,
};

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Replaces the copy of `original` that `copy` ends in, from instruction
    /// `at`, by a jump to `original`, whose phis take `entering` along the new
    /// edge, in their order.
    ///
    /// The caller has established that the copy computes what `original`
    /// computes under the values `entering` names, and that the blocks both
    /// branch to take the same values from each ([`crate::passes::unthread`]):
    /// every value keeps its meaning, and what the copy defined is read
    /// nowhere. The original's rows, row for row, do the copy's work on its
    /// path now, so each takes in the facts of the row it stands for
    /// ([`InstructionFacts::absorb_twin`]): the copy's call is a call site of
    /// the original's. What the copy's ways out ran after it runs on no way
    /// now ([`BlockProvenance::forget_exits`]), and a copy that was the whole
    /// block leaves it standing for nothing
    /// ([`BlockProvenance::clear`]): an edge sent past it runs past nothing.
    /// The graph changed, so the caller declares the session repaired
    /// ([`Self::declare_repaired`]), and finishing it retires the copy's
    /// values.
    ///
    /// [`BlockProvenance::forget_exits`]: crate::ir::block::BlockProvenance::forget_exits
    /// [`BlockProvenance::clear`]: crate::ir::block::BlockProvenance::clear
    ///
    /// # Arguments
    ///
    /// * `copy` - The block ending in the copy.
    /// * `original` - The block it copies.
    /// * `at` - Where the copy begins in `copy`.
    /// * `entering` - The value each of `original`'s phis takes from `copy`.
    ///
    /// # Errors
    ///
    /// Returns an error if either block is missing, or `entering` does not
    /// name one value per phi of `original`.
    pub fn merge_duplicate(
        &mut self,
        copy: usize,
        original: usize,
        at: usize,
        entering: &[SsaVarId],
    ) -> Result<()> {
        let successors = self
            .ssa
            .block(copy)
            .map(SsaBlock::successors)
            .ok_or_else(|| Error::new(format!("missing block B{copy}")))?;
        let phis = self
            .ssa
            .block(original)
            .map(|block| block.phi_nodes().len())
            .ok_or_else(|| Error::new(format!("missing block B{original}")))?;
        if phis != entering.len() {
            return Err(Error::new(format!(
                "B{original} has {phis} phis and {} values enter them from B{copy}",
                entering.len()
            )));
        }
        let twins: Vec<T::InstructionFacts> = self
            .ssa
            .block(copy)
            .and_then(|block| block.instructions().get(at..))
            .unwrap_or_default()
            .iter()
            .map(|row| row.facts().clone())
            .collect();
        if let Some(block) = self.ssa.block_mut(original) {
            for (row, twin) in block.instructions_mut().iter_mut().zip(&twins) {
                row.facts_mut().absorb_twin(twin);
            }
        }
        if let Some(block) = self.ssa.block_mut(copy) {
            block.instructions_mut().truncate(at);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: original }));
            // The copy's ways out are the original's now, which run their
            // own code: what the copy's ran after it runs nowhere. A copy that
            // was the whole block leaves it standing for nothing at all: the
            // original's rows hold its code, as their twins.
            if at == 0 {
                block.provenance_mut().clear();
            } else {
                block.provenance_mut().forget_exits();
            }
        }
        for target in successors {
            if let Some(block) = self.ssa.block_mut(target) {
                for phi in block.phi_nodes_mut() {
                    phi.operands_mut().retain(|operand| {
                        operand.is_exceptional() || operand.predecessor() != copy
                    });
                }
            }
        }
        if let Some(block) = self.ssa.block_mut(original) {
            for (phi, value) in block.phi_nodes_mut().iter_mut().zip(entering) {
                phi.add_operand(PhiOperand::new(*value, copy));
            }
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(())
    }

    /// Merges the test a compiler copied ahead of a loop it rotated into the
    /// loop's own test, which then runs first: `guard`, outside the loop,
    /// branches to `forwarder`, which only jumps on to `header`, and past
    /// the loop; `latch` tests at the loop's foot, going on to `header` or
    /// out. `guard` jumps to `forwarder`, which jumps to `latch`, whose phis
    /// take `entering` from it, in their order; `header` takes nothing from
    /// `forwarder` any more, and the block `guard` branched to past the loop
    /// nothing from `guard`. The rows of `guard` that computed only what its
    /// branch read go with the branch.
    ///
    /// The caller has established that `latch`, run with `entering` in its
    /// phis' places, asks what `guard` asked, goes on to `header` where
    /// `guard` went into the loop, and leaves it for a way that reaches,
    /// computing only values, the block `guard` branched to past it, holding
    /// what `guard` gave it there ([`crate::passes::unthread`]); and that
    /// each phi of `header` takes from `latch` a phi of `latch` whose
    /// `entering` value is what it took from `forwarder`. Every value keeps
    /// its meaning, and the loop is entered at its test alone. The latch's
    /// branch does the guard's work on its path now, so it takes in the
    /// facts of the guard's branch ([`InstructionFacts::absorb_twin`]), and
    /// what the guard's way out ran after it runs on no way now
    /// ([`BlockProvenance::forget_exits`]). The graph changed, so the caller
    /// declares the session repaired ([`Self::declare_repaired`]).
    ///
    /// [`BlockProvenance::forget_exits`]: crate::ir::block::BlockProvenance::forget_exits
    ///
    /// # Arguments
    ///
    /// * `guard` - The block ending in the copied test.
    /// * `forwarder` - The block `guard` enters the loop through.
    /// * `latch` - The loop's test.
    /// * `header` - The block `latch` goes on to and `forwarder` enters.
    /// * `entering` - The value each of `latch`'s phis takes from
    ///   `forwarder`.
    ///
    /// # Errors
    ///
    /// Returns an error if a block is missing, `guard` does not branch two
    /// ways, one of them to `forwarder`, `forwarder` does not jump to
    /// `header`, or `entering` does not name one value per phi of `latch`.
    pub fn unrotate(
        &mut self,
        guard: usize,
        forwarder: usize,
        latch: usize,
        header: usize,
        entering: &[SsaVarId],
    ) -> Result<()> {
        let missing = |block: usize| Error::new(format!("missing block B{block}"));
        let guarding = self.ssa.block(guard).ok_or_else(|| missing(guard))?;
        let Some((true_target, false_target)) = guarding
            .control_terminator()
            .and_then(SsaOp::two_way_targets)
        else {
            return Err(Error::new(format!("B{guard} does not branch two ways")));
        };
        let skipped = match (true_target == forwarder, false_target == forwarder) {
            (true, false) => false_target,
            (false, true) => true_target,
            _ => {
                return Err(Error::new(format!(
                    "B{guard} does not branch to B{forwarder} and elsewhere"
                )));
            }
        };
        let terminator = guarding
            .instructions()
            .last()
            .ok_or_else(|| Error::new(format!("B{guard} has no terminator")))?;
        let read = terminator.op().uses();
        let twin = terminator.facts().clone();
        // The guard's rows that computed only what its branch read.
        let asked: Vec<(usize, usize)> = guarding
            .instructions()
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                row.op().is_pure() && row.op().defs().any(|value| read.contains(&value))
            })
            .map(|(index, _)| (guard, index))
            .collect();
        let phis = self
            .ssa
            .block(latch)
            .map(|block| block.phi_nodes().len())
            .ok_or_else(|| missing(latch))?;
        if phis != entering.len() {
            return Err(Error::new(format!(
                "B{latch} has {phis} phis and {} values enter them from B{forwarder}",
                entering.len()
            )));
        }
        if !self.redirect_terminator_target(forwarder, header, latch)? {
            return Err(Error::new(format!(
                "B{forwarder} does not jump to B{header}"
            )));
        }
        if let Some(row) = self
            .ssa
            .block_mut(latch)
            .and_then(|block| block.instructions_mut().last_mut())
        {
            row.facts_mut().absorb_twin(&twin);
        }
        self.replace_terminator_op(guard, SsaOp::Jump { target: forwarder })?;
        if let Some(block) = self.ssa.block_mut(guard) {
            block.provenance_mut().forget_exits();
        }
        if let Some(block) = self.ssa.block_mut(skipped) {
            for phi in block.phi_nodes_mut() {
                phi.operands_mut()
                    .retain(|operand| operand.is_exceptional() || operand.predecessor() != guard);
            }
        }
        if let Some(block) = self.ssa.block_mut(header) {
            for phi in block.phi_nodes_mut() {
                phi.operands_mut().retain(|operand| {
                    operand.is_exceptional() || operand.predecessor() != forwarder
                });
            }
        }
        if let Some(block) = self.ssa.block_mut(latch) {
            for (phi, value) in block.phi_nodes_mut().iter_mut().zip(entering) {
                phi.add_operand(PhiOperand::new(*value, forwarder));
            }
        }
        self.remove_unread_instructions(asked);
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(())
    }

    /// Rejoins the loop at `second` into the loop at `header` it was split
    /// from: each of `second`'s latches goes back to `header` instead, whose
    /// phis take from it what `second`'s phis took, and a trip takes either
    /// loop's body by `header`'s test.
    ///
    /// The caller has established that `header`'s test leaves its loop for
    /// `second` once and for good, along a path computing only values, and
    /// that nothing `second`'s loop changes is read by that test or path
    /// ([`crate::passes::rejoin`]): re-tested after each trip of the second
    /// body, the test takes it again. Each of `second`'s phis continues one of
    /// `header`'s (`continues` names it) or takes a value from before the loop,
    /// which becomes a phi of `header`'s that the first body leaves as it is.
    /// The graph changed, so the caller declares the session repaired
    /// ([`Self::declare_repaired`]).
    ///
    /// # Arguments
    ///
    /// * `header` - The first loop's header.
    /// * `second` - The second loop's header.
    /// * `entering` - `header`'s predecessors outside its loop.
    /// * `first_latches` - The first loop's latches.
    /// * `second_latches` - The second loop's latches.
    /// * `continues` - For each of `second`'s phis, in order, the phi of
    ///   `header`'s it continues, or `None` for one taking a value from before
    ///   the loop.
    ///
    /// # Errors
    ///
    /// Returns an error if a block is missing, `continues` does not name one
    /// entry per phi of `second`, or a latch does not branch to `second`.
    pub fn rejoin_split_loop(
        &mut self,
        header: usize,
        second: usize,
        entering: &[usize],
        first_latches: &[usize],
        second_latches: &[usize],
        continues: &[Option<SsaVarId>],
    ) -> Result<()> {
        let second_phis: Vec<PhiNode> = self
            .ssa
            .block(second)
            .map(|block| block.phi_nodes().to_vec())
            .ok_or_else(|| Error::new(format!("missing block B{second}")))?;
        if second_phis.len() != continues.len() {
            return Err(Error::new(format!(
                "B{second} has {} phis and {} continuations",
                second_phis.len(),
                continues.len()
            )));
        }
        for latch in second_latches {
            if !self.redirect_terminator_target(*latch, second, header)? {
                return Err(Error::new(format!("B{latch} does not branch to B{second}")));
            }
        }
        let continued: BTreeMap<SsaVarId, &PhiNode> = continues
            .iter()
            .zip(&second_phis)
            .filter_map(|(first, phi)| Some(((*first)?, phi)))
            .collect();
        // What the second body hands each latch's edge: the value its phi
        // continuing a header phi takes from the latch, and a header phi it
        // continues none of unchanged.
        let from_latch =
            |phi: &PhiNode, latch: usize| phi.operand_from(latch).map(PhiOperand::value);
        if let Some(block) = self.ssa.block_mut(header) {
            for phi in block.phi_nodes_mut() {
                let result = phi.result();
                for latch in second_latches {
                    let value = continued
                        .get(&result)
                        .and_then(|second| from_latch(second, *latch))
                        .unwrap_or(result);
                    phi.add_operand(PhiOperand::new(value, *latch));
                }
            }
        }
        let mut replacements: Vec<(SsaVarId, SsaVarId)> = Vec::new();
        for (phi, first) in second_phis.iter().zip(continues) {
            if let Some(first) = first {
                replacements.push((phi.result(), *first));
                continue;
            }
            // A value from before the loop the first body leaves as it is.
            let Some(before) = phi
                .normal_operands()
                .find(|operand| !second_latches.contains(&operand.predecessor()))
                .map(PhiOperand::value)
            else {
                return Err(Error::new(format!(
                    "B{second}'s phi {} takes nothing from before its loop",
                    phi.result()
                )));
            };
            let (version, ty) = self
                .ssa
                .variable(phi.result())
                .map(|variable| (variable.version(), variable.var_type().clone()))
                .ok_or_else(|| Error::new(format!("missing variable {}", phi.result())))?;
            let result =
                self.create_fresh_variable(phi.origin(), version, DefSite::phi(header), ty);
            let mut joined = PhiNode::new(result, phi.origin());
            for pred in entering {
                joined.add_operand(PhiOperand::new(before, *pred));
            }
            for latch in first_latches {
                joined.add_operand(PhiOperand::new(result, *latch));
            }
            for latch in second_latches {
                joined.add_operand(PhiOperand::new(
                    from_latch(phi, *latch).unwrap_or(result),
                    *latch,
                ));
            }
            if let Some(block) = self.ssa.block_mut(header) {
                block.add_phi(joined);
            }
            replacements.push((phi.result(), result));
        }
        if let Some(block) = self.ssa.block_mut(second) {
            block.phi_nodes_mut().clear();
        }
        for (old, new) in replacements {
            let _ = self.ssa.replace_uses_including_phis(old, new);
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(())
    }

    /// Computes in `test` what its two arms `first` and `second` compute
    /// alike, and sends each of its edges to where that arm went on.
    ///
    /// Each arm is entered from `test` alone, holds no phi, computes values
    /// only and jumps on, and `renamed` pairs each value `second` defines
    /// with the one `first` defines in its place ([`crate::passes::unthread`]).
    /// `first`'s rows move to the end of `test`, ahead of its branch; what
    /// read a value of `second`'s reads `first`'s; `test` branches to where
    /// each arm jumped, and the blocks there take from `test` what they took
    /// from the arm; both arms, entered from nowhere, go. Every value keeps
    /// its meaning; the graph changed, so the caller declares the session
    /// repaired ([`Self::declare_repaired`]).
    ///
    /// # Arguments
    ///
    /// * `test` - The block whose branch enters the arms.
    /// * `first` - The arm whose rows move into `test`.
    /// * `second` - The arm whose values become `first`'s.
    /// * `renamed` - Each value `second` defines, with `first`'s in its place.
    ///
    /// # Errors
    ///
    /// Returns an error if an arm does not exist or does not end in a jump,
    /// both arms jump to one block, or `test`'s terminator does not branch to
    /// both.
    pub fn merge_forwarders(
        &mut self,
        test: usize,
        first: usize,
        second: usize,
        renamed: &[(SsaVarId, SsaVarId)],
    ) -> Result<()> {
        let jumps_to = |arm: usize| match self.ssa.block(arm).and_then(SsaBlock::control_terminator)
        {
            Some(SsaOp::Jump { target }) => Ok(*target),
            _ => Err(Error::new(format!("B{arm} does not end in a jump"))),
        };
        let (first_to, second_to) = (jumps_to(first)?, jumps_to(second)?);
        if first_to == second_to {
            return Err(Error::new(format!(
                "B{first} and B{second} both jump to B{first_to}"
            )));
        }
        let rows = self
            .ssa
            .block(first)
            .map(|block| block.instructions().len().saturating_sub(1))
            .ok_or_else(|| Error::new(format!("missing block B{first}")))?;
        for row in 0..rows {
            let moved = self.detach_instruction(first, row)?;
            self.insert_before_terminator(test, moved)?;
        }
        for (theirs, mine) in renamed {
            let _ = self.ssa.replace_uses_including_phis(*theirs, *mine);
        }
        if !self.redirect_terminator_target(test, first, first_to)?
            || !self.redirect_terminator_target(test, second, second_to)?
        {
            return Err(Error::new(format!(
                "B{test}'s terminator does not branch to B{first} and B{second}"
            )));
        }
        for (arm, to) in [(first, first_to), (second, second_to)] {
            self.replace_phi_predecessor(to, arm, test)?;
            self.ssa.block_mut(arm).map(SsaBlock::clear);
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(())
    }

    /// Gives the terminators of `tests` one block of their own, which the
    /// tests jump to and which branches as each of them did.
    ///
    /// The tests end in one branch to the same blocks, alike but for the
    /// values they read ([`crate::passes::unthread`]): the new block's
    /// terminator is the first test's, the decoded branch it stands for with
    /// it, and reads, where every test reads one value, that value, and
    /// elsewhere a phi of what each reads. It holds the host block the first
    /// test's branch lies in, the last that test carries
    /// ([`BlockProvenance::carry`](crate::ir::block::BlockProvenance::carry)), and each edge out of it runs through
    /// what that branch's edge to the same block did. Each test ends in a
    /// jump to it that stands for no decoded code; every other test's edges
    /// ran through what they did on its way to the one test, so its jump runs
    /// through that ([`BlockProvenance::retarget`](crate::ir::block::BlockProvenance::retarget)). Each block the tests
    /// branch to takes one value from the new block in place of the tests'
    /// own: theirs where they agree, a phi of them where they do not. Every
    /// value keeps its meaning; the graph changed, so the caller declares the
    /// session repaired ([`Self::declare_repaired`]).
    ///
    /// # Arguments
    ///
    /// * `tests` - The blocks whose terminators become one; at least two.
    ///
    /// # Returns
    ///
    /// The new block's index.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than two tests are named, a test has no
    /// terminator, or the tests' terminators branch to different blocks, or a
    /// block they branch to lacks a phi operand from one of them.
    pub fn merge_tests(&mut self, tests: &[usize]) -> Result<usize> {
        let terminators: Vec<SsaOp<T>> = tests
            .iter()
            .map(|test| {
                self.ssa
                    .block(*test)
                    .and_then(SsaBlock::control_terminator)
                    .cloned()
                    .ok_or_else(|| Error::new(format!("block B{test} has no terminator")))
            })
            .collect::<Result<_>>()?;
        let [first, rest @ ..] = terminators.as_slice() else {
            return Err(Error::new("no tests to merge".to_string()));
        };
        if rest.is_empty() {
            return Err(Error::new("one test merges with nothing".to_string()));
        }
        // The tests ask one question, some negated with their targets
        // exchanged: they branch to the same blocks in either order.
        let successors = first.successors();
        let sorted = |op: &SsaOp<T>| {
            let mut targets = op.successors();
            targets.sort_unstable();
            targets
        };
        if rest.iter().any(|op| sorted(op) != sorted(first)) {
            return Err(Error::new(format!(
                "the tests {tests:?} branch to different blocks"
            )));
        }
        let merged = self.ssa.block_count();
        let mut block = SsaBlock::new(merged);
        // A value for each set of values the tests supply, one per test: the
        // value itself where they agree, else a phi of the new block - one
        // phi for one set, however many readers want it. A phi a successor's
        // phi takes stands for that phi's variable, `stands_for`: the new block
        // is a latch the tests' ways round share, and the value is that
        // variable's split across it ([`Derivation::LoopSplit`]), held where
        // it is held. Any other joins the tests' values.
        let mut minted: BTreeMap<Vec<SsaVarId>, SsaVarId> = BTreeMap::new();
        let mut join = |ssa: &mut SsaFunction<T>,
                        block: &mut SsaBlock<T>,
                        values: &[SsaVarId],
                        stands_for: Option<SsaVarId>| {
            let &value = values.first()?;
            if values.iter().all(|other| *other == value) {
                return Some(value);
            }
            if let Some(result) = minted.get(values) {
                return Some(*result);
            }
            let result = match stands_for.and_then(|variable| ssa.variable(variable)) {
                Some(variable) => {
                    let (parent, origin, var_type) = (
                        variable.id(),
                        variable.origin(),
                        variable.var_type().clone(),
                    );
                    ssa.create_variable_derived(
                        parent,
                        origin,
                        0,
                        DefSite::phi(merged),
                        var_type,
                        Derivation::LoopSplit,
                    )
                }
                None => {
                    let (origin, var_type) = ssa
                        .variable(value)
                        .map(|var| (var.origin(), var.var_type().clone()))?;
                    ssa.create_variable_joined(
                        values,
                        origin,
                        0,
                        DefSite::phi(merged),
                        var_type,
                        Join::PlacedPhi,
                    )
                }
            };
            let origin = ssa.variable(result).map(SsaVariable::origin)?;
            let mut phi = PhiNode::new(result, origin);
            for (test, value) in tests.iter().zip(values) {
                phi.add_operand(PhiOperand::new(*value, *test));
            }
            block.add_phi(phi);
            minted.insert(values.to_vec(), result);
            Some(result)
        };
        // What each block the tests branch to takes from the new block: a
        // value of the variable that block's phi stands for, which the
        // terminator reads in turn where it reads those values.
        let mut entering: Vec<(usize, usize, SsaVarId)> = Vec::new();
        for target in &successors {
            let phis = self
                .ssa
                .block(*target)
                .map(|block| block.phi_nodes().to_vec())
                .ok_or_else(|| Error::new(format!("missing block B{target}")))?;
            for (index, phi) in phis.iter().enumerate() {
                let values: Vec<SsaVarId> = tests
                    .iter()
                    .map(|test| {
                        phi.operand_from(*test)
                            .map(PhiOperand::value)
                            .ok_or_else(|| {
                                Error::new(format!("B{target}'s phi takes nothing from B{test}"))
                            })
                    })
                    .collect::<Result<_>>()?;
                let value = join(&mut self.ssa, &mut block, &values, Some(phi.result()))
                    .ok_or_else(|| Error::new(format!("B{target}'s phi reads no variable")))?;
                entering.push((*target, index, value));
            }
        }
        // The terminator, reading at each operand what every test reads there.
        let read: Vec<Vec<SsaVarId>> = terminators
            .iter()
            .map(|op| {
                let mut uses = Vec::new();
                op.visit_operands(|role, var| {
                    if role == OperandRole::Use {
                        uses.push(var);
                    }
                });
                uses
            })
            .collect();
        let count = read.first().map_or(0, Vec::len);
        if read.iter().any(|uses| uses.len() != count) {
            return Err(Error::new(format!(
                "the tests {tests:?} read different numbers of values"
            )));
        }
        let mut chosen = Vec::with_capacity(count);
        for position in 0..count {
            let values: Vec<SsaVarId> = read
                .iter()
                .filter_map(|uses| uses.get(position).copied())
                .collect();
            let value = join(&mut self.ssa, &mut block, &values, None)
                .ok_or_else(|| Error::new(format!("an operand of {tests:?} is no variable")))?;
            chosen.push(value);
        }
        let mut terminator = first.clone();
        let mut operands = chosen.into_iter();
        terminator.visit_operands_mut(|role, var| {
            if role == OperandRole::Use
                && let Some(value) = operands.next()
            {
                *var = value;
            }
        });
        // The branch moves to the new block: the first test's instruction, the
        // decoded branch it stands for with it, reading what the new block
        // reads. Each test ends in a jump there, standing for no decoded code,
        // as a block a pass made ends.
        let mut moved = tests
            .first()
            .and_then(|test| self.ssa.block(*test))
            .and_then(|block| block.instructions().last())
            .cloned()
            .ok_or_else(|| Error::new(format!("the tests {tests:?} have no terminator")))?;
        moved.set_op(terminator);
        block.add_instruction(moved);
        // The branch is the first test's: the new block holds the code of the
        // host block it lies in, the last the first test carries, and the
        // edges out of it, which leave the new block now, run through what
        // they ran through ([`BlockProvenance::hand_passed_to`]). Every other
        // test keeps the host block its own branch lay in, and its jump runs
        // through what its own way out ran through: that code ran on its way
        // to the one test.
        if let Some(first) = tests.first().and_then(|test| self.ssa.block_mut(*test)) {
            if let Some(&host) = first.provenance().carried().last() {
                block.provenance_mut().carry(host);
            }
            first
                .provenance_mut()
                .hand_passed_to(block.provenance_mut());
        }
        self.ssa.add_block(block);
        for test in tests {
            let block = self
                .ssa
                .block_mut(*test)
                .ok_or_else(|| Error::new(format!("missing block B{test}")))?;
            block.instructions_mut().pop();
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: merged }));
            block.provenance_mut().retarget(|_| Some(merged));
        }
        for (target, index, value) in entering {
            if let Some(phi) = self
                .ssa
                .block_mut(target)
                .and_then(|block| block.phi_nodes_mut().get_mut(index))
            {
                phi.operands_mut().retain(|operand| {
                    operand.is_exceptional() || !tests.contains(&operand.predecessor())
                });
                phi.add_operand(PhiOperand::new(value, merged));
            }
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(merged)
    }

    /// Coalesces a block into its unconditional predecessor.
    ///
    /// `predecessor_idx` must end with `Jump { target: successor_idx }`.
    /// The predecessor's jump is removed, successor phis are converted into
    /// `Copy` instructions, successor instructions are appended, self-targets
    /// inside the moved instructions are redirected to the predecessor, and
    /// the successor block is cleared. Phi operands in other blocks that
    /// referenced the successor are rewritten to the predecessor.
    ///
    /// # Arguments
    ///
    /// * `predecessor_idx` - Block that absorbs the successor; must end in
    ///   `Jump { target: successor_idx }`.
    /// * `successor_idx` - Block merged into the predecessor and then cleared.
    ///
    /// # Returns
    ///
    /// `true` if the merge happened, `false` if the successor had no
    /// instructions to move.
    ///
    /// # Errors
    ///
    /// Returns an error if either block is missing, or if the predecessor does
    /// not end in an unconditional jump to the successor.
    pub fn coalesce_unconditional_successor(
        &mut self,
        predecessor_idx: usize,
        successor_idx: usize,
    ) -> Result<bool> {
        let Some(predecessor) = self.ssa.block(predecessor_idx) else {
            return Err(Error::new(format!("missing block B{predecessor_idx}")));
        };
        if !matches!(
            predecessor.control_terminator(),
            Some(SsaOp::Jump { target }) if *target == successor_idx
        ) {
            return Err(Error::new(format!(
                "block B{predecessor_idx} does not jump to B{successor_idx}"
            )));
        }

        let Some(successor) = self.ssa.block(successor_idx) else {
            return Err(Error::new(format!("missing block B{successor_idx}")));
        };
        if successor.instructions().is_empty() {
            return Ok(false);
        }

        let phi_copies: Vec<SsaInstruction<T>> = successor
            .phi_nodes()
            .iter()
            .filter_map(|phi| {
                let operand = phi.operands().first()?;
                let dest = phi.result();
                let src = operand.value();
                if dest == src {
                    return None;
                }
                Some(SsaInstruction::synthetic(SsaOp::Copy { dest, src }))
            })
            .collect();
        let successor_instrs = successor.instructions().to_vec();
        let successor_provenance = successor.provenance().clone();

        let Some(predecessor) = self.ssa.block_mut(predecessor_idx) else {
            return Err(Error::new(format!("missing block B{predecessor_idx}")));
        };
        // The predecessor holds the successor's code from here on, so it stands
        // for the host blocks the successor did.
        predecessor
            .provenance_mut()
            .absorb(successor_idx, &successor_provenance);
        let instrs = predecessor.instructions_mut();
        instrs.pop();
        instrs.extend(phi_copies);
        instrs.extend(successor_instrs);
        for instr in instrs {
            instr
                .op_mut()
                .redirect_target(successor_idx, predecessor_idx);
        }

        if let Some(successor) = self.ssa.block_mut(successor_idx) {
            successor.clear();
        }

        let block_count = self.ssa.block_count();
        for block_idx in 0..block_count {
            if block_idx == predecessor_idx || block_idx == successor_idx {
                continue;
            }
            self.retarget_phi_operands(block_idx, successor_idx, predecessor_idx)?;
        }

        // **No reconstruction.** The successor had the predecessor as its only
        // way in, so the predecessor dominated everything the successor did;
        // moving the successor's code to the end of the predecessor leaves
        // every definition dominating every use it did before, and the phis
        // that named the successor now name the block that holds its code.
        // Nothing a rebuild would compute is missing - and a rebuild recomputes
        // every use from its rename group, which on optimised SSA, where a
        // group can hold unrelated values, reads some of them as undefined.
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(true)
    }
}
