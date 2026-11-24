//! Turning a front end's output into SSA, and saying what happened.
//!
//! A lifter emits blocks, instructions and an exception table. What it does not
//! emit is the part every front end otherwise writes for itself: where the phis
//! go, which definition each read names, and the order the finishing steps have
//! to run in. That order is not obvious and getting it wrong is quiet - prune
//! before placing phis and a handler loses its operands, refresh def sites
//! before renaming and they describe the wrong values, compact variables before
//! verifying and the report names ids that no longer exist.
//!
//! This module is that sequence, stated once. A host calls
//! [`SsaFunction::construct_ssa`] when it wants the phis placed and the reads
//! resolved, and [`SsaFunction::finish_construction`] when it placed its own
//! and wants only the finishing and the verdict.
//!
//! # Why one engine
//!
//! Construction and reconstruction are the same algorithm over different
//! starting points: Cytron placement over the iterated dominance frontier, then
//! a dominator-tree rename. `construct_ssa` therefore runs one reconstruction
//! engine for both rather than a second copy of it. A second copy is how the crate's consumers ended up with three of them,
//! each with its own rounding of the same corners - one that rooted its
//! dominator walk at the entry block alone, so a block only a handler reached
//! was never renamed at all.
//!
//! # The contract
//!
//! In this order, and the order is the contract:
//!
//! 1. the entry rule, so nothing downstream sees an entry with predecessors;
//! 2. phi placement and renaming, for [`SsaFunction::construct_ssa`];
//! 3. instruction identities, so every instruction has one before a host keys
//!    anything by it;
//! 4. address-taken slots, before anything treats a slot's load as a value;
//! 5. the unreachable prune, rooted at the entry *and* every handler and filter
//!    entry, so a handler is not mistaken for dead code;
//! 6. definition sites and use lists, rebuilt from the finished blocks;
//! 7. the optional tidy-up: trivial and dead phis, then variable compaction;
//! 8. the optional settling of conversions ([`SsaFunction::settle_conversions`]),
//!    once every read is answered and every phi typed, so the verdict judges
//!    the conversions a front end settles on rather than the ones it guessed;
//! 9. verification, at the requested level, into the report;
//! 10. [`SsaFunction::reset_variable_remap`] last, so the remap a host reads
//!     afterwards starts from the finished function. Where the call moved the
//!     ids held when it began is reported in [`ConstructionReport::variables`].

use crate::{
    Error, Result,
    analysis::{SsaCfg, SsaVerifier, VerifierError, VerifyLevel, exceptions::FunctionRoots},
    ir::{
        function::{
            RebuildReport, SsaFunction, TrivialPhiOptions, VariableRemap, rebuild::SsaRebuilder,
            transforms::Cleanup,
        },
        ops::convert::ConversionIntent,
    },
    target::Target,
};

/// What to do about an entry block a terminator names.
///
/// Every analysis here reads the entry block as the one place control starts,
/// and a predecessor makes that false: the entry's phis would need an operand
/// for an edge from outside the function, and its dominance answers change. A
/// lift whose first instruction heads a loop produces exactly this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EntryRule {
    /// Split the entry behind a fresh, empty block when it has predecessors.
    ///
    /// The default, and what a lifter wants: every block moves up by one and
    /// the new block 0 jumps to the old entry. A host that keys anything by
    /// block index has to renumber, which is why this is stated rather than
    /// implied.
    #[default]
    SplitWhenNeeded,
    /// Leave the numbering alone.
    ///
    /// For a host whose block indices are already published and cannot move.
    /// The verifier then reports the entry's predecessors
    /// ([`VerifierError::EntryHasPredecessors`]) and the report carries it.
    Keep,
}

/// What construction should do beyond placing and renaming.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaConstructOptions {
    /// What to do about an entry block with predecessors.
    pub entry_rule: EntryRule,
    /// Remove blocks control cannot reach, from the entry or from a handler.
    pub prune_unreachable: bool,
    /// Fold phis with one distinct operand, and drop phis nothing reads.
    pub simplify_phis: bool,
    /// Renumber variables to close the gaps construction leaves.
    ///
    /// A host that holds [`SsaVarId`](crate::ir::SsaVarId)s across the call
    /// reads [`ConstructionReport::variables`] afterwards to find where they
    /// went.
    pub compact_variables: bool,
    /// The level to verify at, or `None` to skip verification.
    ///
    /// The result goes in the report rather than the return value: a host that
    /// wants construction to fail on a finding decides that itself, and one
    /// that wants to log and continue can.
    pub verify: Option<VerifyLevel>,
    /// Restate every conversion for the type its operand settles on, as the
    /// intent says the front end meant it ([`SsaFunction::settle_conversions`]),
    /// or `None` to leave each as the front end chose it.
    ///
    /// A front end that chose a conversion before construction answered its
    /// operand - a register read whose definition another block supplies -
    /// states its intent here, so the conversion is settled once every phi
    /// has its type and before the verdict judges it.
    pub settle_conversions: Option<ConversionIntent>,
}

impl Default for SsaConstructOptions {
    /// Everything on, verifying at [`VerifyLevel::Standard`].
    ///
    /// The settings a lift wants: a front end's output is exactly the input
    /// whose defects should be found here rather than four passes later.
    fn default() -> Self {
        Self {
            entry_rule: EntryRule::default(),
            prune_unreachable: true,
            simplify_phis: true,
            compact_variables: true,
            verify: Some(VerifyLevel::Standard),
            settle_conversions: None,
        }
    }
}

impl SsaConstructOptions {
    /// The settings with no verification, for a host that verifies itself.
    #[must_use]
    pub fn unverified() -> Self {
        Self {
            verify: None,
            ..Self::default()
        }
    }

    /// The settings that place phis and rename and do nothing else: the
    /// numbering of blocks and variables stays as it was, nothing is pruned,
    /// simplified or verified, and an entry block with predecessors is
    /// refused rather than split.
    ///
    /// For a host, or a test, that holds block indices and variable ids across
    /// the call.
    #[must_use]
    pub const fn renaming_only() -> Self {
        Self {
            entry_rule: EntryRule::Keep,
            prune_unreachable: false,
            simplify_phis: false,
            compact_variables: false,
            verify: None,
            settle_conversions: None,
        }
    }

    /// Sets the level to verify at.
    ///
    /// # Arguments
    ///
    /// * `level` - The depth to verify at.
    #[must_use]
    pub const fn verified_at(mut self, level: VerifyLevel) -> Self {
        self.verify = Some(level);
        self
    }

    /// Sets what to do about an entry block with predecessors.
    ///
    /// # Arguments
    ///
    /// * `rule` - The rule to apply.
    #[must_use]
    pub const fn with_entry_rule(mut self, rule: EntryRule) -> Self {
        self.entry_rule = rule;
        self
    }
}

/// What construction did, and what it found.
///
/// Returned rather than logged: a host decides for itself whether a finding is
/// fatal, and a lift that reports "4.4% of methods did not verify" needs the
/// findings, not a warning per method.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstructionReport {
    /// Whether the entry block was split off behind a fresh one.
    ///
    /// Every block index moved up by one when it was.
    pub entry_split: bool,
    /// Blocks removed as unreachable.
    pub blocks_pruned: usize,
    /// Phis the function holds now.
    pub phis: usize,
    /// Instructions that were given an identity here.
    pub instructions_identified: usize,
    /// Conversions restated for their operands' settled types
    /// ([`SsaConstructOptions::settle_conversions`]).
    pub conversions_settled: usize,
    /// What the verifier found, empty when it was not run.
    pub findings: Vec<VerifierError>,
    /// What placing the phis and renaming did, for
    /// [`SsaFunction::construct_ssa`]; the default for
    /// [`SsaFunction::finish_construction`], which places none.
    pub reconstruction: RebuildReport,
    /// Where each variable the function held when the call began went.
    ///
    /// The function's own record ([`SsaFunction::variable_remap`]) restarts
    /// from the finished function, so this is the only place a host that holds
    /// ids across the call - a table keyed by the values its lift created -
    /// learns what compaction did to them.
    pub variables: VariableRemap,
}

impl ConstructionReport {
    /// Whether verification ran and found nothing.
    ///
    /// `false` for a report whose options skipped verification: nothing was
    /// checked, which is not the same as nothing being wrong.
    #[must_use]
    pub fn verified_clean(&self, options: &SsaConstructOptions) -> bool {
        options.verify.is_some() && self.findings.is_empty()
    }
}

impl<T: Target> SsaFunction<T> {
    /// Places the phis, resolves every read, and finishes the function.
    ///
    /// The entry point for a front end that emits definitions and reads without
    /// saying which definition each read names. Every value that is a version of
    /// one storage location must share a rename group
    /// ([`SsaFunction::set_rename_group`], or
    /// [`VariableOrigin::canonical_rename_group`]), because that grouping is
    /// what says two definitions are the same variable; ungrouped values are
    /// left as the distinct temporaries they are.
    ///
    /// [`VariableOrigin::canonical_rename_group`]: crate::ir::VariableOrigin::canonical_rename_group
    ///
    /// # Arguments
    ///
    /// * `options` - What to do beyond placing and renaming.
    ///
    /// # Returns
    ///
    /// What construction did and what verification found.
    ///
    /// # Errors
    ///
    /// Returns an error when reconstruction cannot resolve a read - a value
    /// used before any definition of its group reaches it - or when the entry
    /// rule's split fails.
    pub fn construct_ssa(&mut self, options: &SsaConstructOptions) -> Result<ConstructionReport> {
        // The phases address blocks by position and by the id each block
        // records; a table where the two differ has no meaning to construct
        // from, and is refused before anything is touched.
        if let Some((slot, block)) = self
            .blocks
            .iter()
            .enumerate()
            .find(|(slot, block)| block.id() != *slot)
        {
            return Err(Error::new(format!(
                "SSA construction needs a dense block table, but slot {slot} holds a block \
                 recording id {}",
                block.id()
            )));
        }
        let mut report = ConstructionReport::default();
        self.reset_variable_remap();
        self.apply_entry_rule(options, &mut report)?;
        // Placement and renaming are rooted at the entry, so an entry a
        // terminator names would get wrong dominance and a phi with no operand
        // for the edge from outside the function. Refused by name rather than
        // attempted: `EntryRule::Keep` is a host promising the invariant holds,
        // and when it does not the repair is the other rule.
        if matches!(options.entry_rule, EntryRule::Keep) && self.entry_has_predecessors() {
            return Err(Error::new(
                "cannot construct SSA: a terminator names the entry block, and EntryRule::Keep \
                 forbids splitting it; use EntryRule::SplitWhenNeeded, or finish_construction to \
                 have it reported instead",
            ));
        }
        report.reconstruction = SsaRebuilder::new(self).rebuild()?;
        self.finish_construction_inner(options, &mut report);
        Ok(report)
    }

    /// Finishes a function whose phis the host placed itself.
    ///
    /// Everything [`Self::construct_ssa`] does except the placement and the
    /// rename: the entry rule, identities, address-taken slots, the prune, the
    /// def-site and use-list rebuild, the optional tidy-up, and the verdict. A
    /// host that has already resolved its own reads calls this so the finishing
    /// order is the crate's rather than its own.
    ///
    /// # Arguments
    ///
    /// * `options` - What to do while finishing.
    ///
    /// # Returns
    ///
    /// What finishing did and what verification found.
    ///
    /// # Errors
    ///
    /// Returns an error when the entry rule's split fails.
    pub fn finish_construction(
        &mut self,
        options: &SsaConstructOptions,
    ) -> Result<ConstructionReport> {
        let mut report = ConstructionReport::default();
        self.reset_variable_remap();
        self.apply_entry_rule(options, &mut report)?;
        self.finish_construction_inner(options, &mut report);
        Ok(report)
    }

    /// Whether any terminator names the entry block.
    fn entry_has_predecessors(&self) -> bool {
        let mut found = false;
        for block in self.blocks() {
            block.for_each_successor(|succ| found |= succ == 0);
        }
        found
    }

    /// Applies the entry rule, recording whether it moved anything.
    fn apply_entry_rule(
        &mut self,
        options: &SsaConstructOptions,
        report: &mut ConstructionReport,
    ) -> Result<()> {
        if matches!(options.entry_rule, EntryRule::SplitWhenNeeded) {
            report.entry_split = self.ensure_entry_has_no_predecessors();
        }
        Ok(())
    }

    /// The finishing sequence, shared by both entry points.
    ///
    /// The order is the contract; see the module documentation for why each
    /// step has to come where it does.
    fn finish_construction_inner(
        &mut self,
        options: &SsaConstructOptions,
        report: &mut ConstructionReport,
    ) {
        report.instructions_identified = self.assign_instr_ids();
        self.refresh_address_taken();

        if options.prune_unreachable {
            report.blocks_pruned = self.prune_unreachable_blocks();
        }

        self.refresh_def_sites();
        self.recompute_uses();
        self.settle_phi_types();

        if options.simplify_phis {
            self.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });
            self.eliminate_dead_phis();
            self.refresh_def_sites();
            self.recompute_uses();
        }
        if options.compact_variables {
            self.compact_variables();
            self.refresh_def_sites();
            self.recompute_uses();
        }
        if let Some(intent) = options.settle_conversions {
            report.conversions_settled = self.settle_conversions(intent);
        }

        report.phis = self
            .blocks()
            .iter()
            .map(|block| block.phi_nodes().len())
            .sum();

        if let Some(level) = options.verify {
            let mut findings = SsaVerifier::new(self).verify(level);
            // A variable nothing defines and nothing reads is an id compaction
            // would reclaim, and a host that turned compaction off asked for
            // those ids to stay. The prune above makes them itself - emptying an
            // unreachable block unregisters nothing - so reporting them would
            // fail every function with dead code on a request the options made.
            // A read with no definition is a different finding and stays.
            if !options.compact_variables {
                findings.retain(|finding| !finding.is_cosmetic());
            }
            report.findings = findings;
        }

        // Where the call moved the values the host held when it began, which
        // the reset below would otherwise discard.
        report.variables = self.variable_remap().clone();
        // Last: what the function records from here is measured from the
        // finished function, whose ids are the ones a host keys new tables by.
        self.reset_variable_remap();
    }

    /// Empties every block control cannot reach, and returns how many.
    ///
    /// Rooted at the entry *and* at every handler and filter entry, because
    /// those are entered by the runtime rather than by a terminator: rooting at
    /// the entry alone calls every handler dead and deletes the code that runs
    /// when something goes wrong.
    ///
    /// The blocks stay in place, emptied, rather than being removed. Block
    /// indices are how the exception table names its ranges and how a host
    /// names its own records, so closing the gaps is a renumbering a caller has
    /// to ask for.
    ///
    /// # Returns
    ///
    /// The number of blocks emptied.
    pub fn prune_unreachable_blocks(&mut self) -> usize {
        let block_count = self.block_count();
        if block_count == 0 {
            return 0;
        }

        let reachable = FunctionRoots::of(self).live_blocks(&SsaCfg::from_ssa(self));

        let mut pruned = 0usize;
        for block_idx in 0..block_count {
            if reachable.contains_checked(block_idx) {
                continue;
            }
            if let Some(block) = self.block_mut(block_idx) {
                let was_empty = block.instructions().is_empty() && block.phi_nodes().is_empty();
                block.clear();
                if !was_empty {
                    pruned = pruned.saturating_add(1);
                }
            }
        }
        if pruned > 0 {
            self.prune_phi_operands(&reachable);
            self.recompute_uses();
        }
        pruned
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            instruction::SsaInstruction,
            ops::SsaOp,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// A block nothing reaches is dead code: clearing it forgets the host
    /// block it stood for, and the live one keeps its own.
    #[test]
    fn pruning_forgets_what_an_unreached_block_stood_for() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        for (id, host) in [(0, 0x10), (1, 0x20)] {
            let mut block = SsaBlock::new(id);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            block.provenance_mut().carry(host);
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        assert_eq!(ssa.prune_unreachable_blocks(), 1);
        assert_eq!(
            ssa.block(0).map(|block| block.provenance().carried()),
            Some(&[0x10][..])
        );
        assert!(
            ssa.block(1)
                .is_some_and(|block| block.provenance().is_empty())
        );
    }

    /// `B0: local0 = 1; jump B1. B1: local0 = 2; jump B1` - an entry a
    /// terminator names, which the entry rule has to deal with.
    fn entry_with_a_predecessor() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
        let first = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: first,
            value: ConstValue::I32(1),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(entry);

        let mut loop_block = SsaBlock::new(1);
        loop_block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 0 }));
        ssa.add_block(loop_block);
        ssa.recompute_uses();
        ssa
    }

    /// **Compaction says where the ids a host held went.** A variable nothing
    /// defines is dropped and the one after it moves down; the report maps the
    /// ids the function held when the call began to where they are now, and the
    /// function's own record restarts from the finished numbering.
    #[test]
    fn compaction_reports_where_held_ids_went() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let dropped = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let kept = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: kept,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(kept))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let report = ssa
            .finish_construction(&SsaConstructOptions::default())
            .unwrap_or_else(|_| process::abort());
        assert_eq!(
            report.variables.get(dropped),
            None,
            "the undefined variable is gone"
        );
        assert_eq!(
            report.variables.get(kept),
            Some(dropped),
            "the kept variable moved down into the freed id",
        );
        assert!(
            ssa.variable_remap().is_identity(),
            "the function's record starts from the finished numbering",
        );
    }

    #[test]
    fn the_entry_rule_splits_an_entry_a_terminator_names() {
        let mut ssa = entry_with_a_predecessor();
        let report = ssa
            .construct_ssa(&SsaConstructOptions::default())
            .unwrap_or_else(|_| process::abort());
        assert!(report.entry_split);
        assert!(
            !report
                .findings
                .iter()
                .any(|finding| matches!(finding, VerifierError::EntryHasPredecessors { .. })),
            "{:?}",
            report.findings
        );

        // `Keep` is a promise the invariant holds. Where it does not,
        // construction refuses by name rather than reconstructing against
        // dominance it knows is wrong.
        let mut kept = entry_with_a_predecessor();
        let options = SsaConstructOptions::default().with_entry_rule(EntryRule::Keep);
        assert!(kept.construct_ssa(&options).is_err());

        // Finishing such a function is allowed, and reports it.
        let report = kept
            .finish_construction(&options)
            .unwrap_or_else(|_| process::abort());
        assert!(!report.entry_split);
        assert!(
            report
                .findings
                .iter()
                .any(|finding| matches!(finding, VerifierError::EntryHasPredecessors { .. })),
            "{:?}",
            report.findings
        );
    }

    #[test]
    fn a_handler_is_not_pruned_as_unreachable() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
        let value = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );

        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(entry);

        // Only the runtime enters B1.
        let mut handler = SsaBlock::new(1);
        handler.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(7),
        }));
        handler.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(value))));
        ssa.add_block(handler);

        let mut tail = SsaBlock::new(2);
        tail.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(tail);

        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(2, 3),
            handler_range: BlockRange::new(1, 2),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let report = ssa
            .construct_ssa(&SsaConstructOptions::default())
            .unwrap_or_else(|_| process::abort());
        assert_eq!(report.blocks_pruned, 0, "the handler runs: {ssa}");
        assert!(
            ssa.block(1)
                .is_some_and(|block| !block.instructions().is_empty()),
            "{ssa}"
        );
    }

    #[test]
    fn an_unreachable_block_is_emptied_and_counted() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(entry);
        let mut orphan = SsaBlock::new(1);
        orphan.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(orphan);
        ssa.recompute_uses();

        let report = ssa
            .finish_construction(&SsaConstructOptions::default())
            .unwrap_or_else(|_| process::abort());
        assert_eq!(report.blocks_pruned, 1);
        assert!(
            ssa.block(1)
                .is_some_and(|block| block.instructions().is_empty())
        );
    }

    /// `B0: return`, and an unreachable `B1: v = 1; return v`.
    fn dead_block_defining_a_value() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        // Version 1: a version 0 with no definition is an entry value, which
        // neither the verifier nor compaction treats as dead.
        let value = ssa.create_variable(
            VariableOrigin::Local(0),
            1,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(entry);
        let mut dead = SsaBlock::new(1);
        dead.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(1),
        }));
        dead.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(value))));
        ssa.add_block(dead);
        ssa.recompute_uses();
        ssa
    }

    /// **Keeping ids is not a defect the verdict reports.** Pruning the dead
    /// block leaves its value registered with no definition and no read. A
    /// host that turns compaction off - to keep the ids it holds - was told
    /// its function did not verify, and a lift refused a well-formed function
    /// for having dead code.
    #[test]
    fn a_pruned_value_is_not_a_finding_when_ids_are_kept() {
        let mut ssa = dead_block_defining_a_value();
        let options = SsaConstructOptions {
            compact_variables: false,
            simplify_phis: false,
            ..SsaConstructOptions::default()
        };
        let report = ssa
            .finish_construction(&options)
            .unwrap_or_else(|_| process::abort());
        assert_eq!(report.blocks_pruned, 1);
        assert!(report.findings.is_empty(), "{:?}", report.findings);
        assert!(
            !SsaVerifier::new(&ssa)
                .verify(VerifyLevel::Standard)
                .is_empty(),
            "the value is still registered; only the verdict leaves it out"
        );
    }

    /// Control for [`a_pruned_value_is_not_a_finding_when_ids_are_kept`]: with
    /// compaction on, the id is reclaimed, so there is nothing to leave out.
    #[test]
    fn a_pruned_value_is_reclaimed_when_ids_are_compacted() {
        let mut ssa = dead_block_defining_a_value();
        let report = ssa
            .finish_construction(&SsaConstructOptions::default())
            .unwrap_or_else(|_| process::abort());
        assert!(report.findings.is_empty(), "{:?}", report.findings);
        assert_eq!(ssa.variables().len(), 0);
    }

    /// A diamond where both arms assign the same local and the join reads it -
    /// the shape every front end emits, with no phi anywhere in it.
    ///
    /// This is the whole point of the entry point: the caller says which values
    /// are versions of one storage, and construction works out that the join
    /// needs a merge and which definition the read names.
    #[test]
    fn construction_places_the_phi_a_front_end_did_not() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
        let condition = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let declare = |ssa: &mut SsaFunction<MockTarget>, version, block| {
            let id = ssa.create_variable(
                VariableOrigin::Local(0),
                version,
                DefSite::instruction(block, 0),
                MockType::I32,
            );
            ssa.set_rename_group(id, 0);
            id
        };
        let left = declare(&mut ssa, 0, 1);
        let right = declare(&mut ssa, 1, 2);
        let read = declare(&mut ssa, 2, 3);

        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: condition,
            value: ConstValue::I32(1),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(entry);

        for (block_id, dest, value) in [(1usize, left, 10i32), (2, right, 20)] {
            let mut arm = SsaBlock::new(block_id);
            arm.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            }));
            arm.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(arm);
        }

        // The join reads the local, spelled as a front end spells it: a fresh
        // load of the slot, with no statement of which arm's value it names.
        let mut join = SsaBlock::new(3);
        join.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocal {
            dest: read,
            local_index: 0,
        }));
        join.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(read))));
        ssa.add_block(join);
        ssa.recompute_uses();

        let report = ssa
            .construct_ssa(&SsaConstructOptions::default())
            .unwrap_or_else(|_| process::abort());
        assert!(report.findings.is_empty(), "{:?}\n{ssa}", report.findings);
        assert_eq!(report.phis, 1, "the join merges the two arms: {ssa}");

        let merged = ssa
            .block(3)
            .and_then(|block| block.phi_nodes().first())
            .map(PhiNode::result);
        let returned = ssa.block(3).and_then(|block| {
            block
                .instructions()
                .iter()
                .find_map(|instr| match instr.op() {
                    SsaOp::Return { value, .. } => *value,
                    _ => None,
                })
        });
        assert_eq!(returned, merged, "and the read names that merge: {ssa}");
    }

    /// A dispatcher `B2` is entered from `B0`, from its case `B5` that sets
    /// local 0, and back from case `B3` through the forwarding block `B1`.
    /// Neither `B3` nor `B1` touches local 0, so the edge from `B1` carries the
    /// merge itself - what the local holds after `B5` ran - not the value the
    /// function was entered with.
    #[test]
    fn a_back_edge_that_leaves_a_variable_alone_carries_its_merge() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 2);
        let var = |ssa: &mut SsaFunction<MockTarget>, local: u16, version, site, group| {
            let id =
                ssa.create_variable(VariableOrigin::Local(local), version, site, MockType::I32);
            ssa.set_rename_group(id, group);
            id
        };
        let entry = var(&mut ssa, 0, 0, DefSite::entry(), 0);
        let first = var(&mut ssa, 1, 0, DefSite::instruction(0, 0), 1);
        let back = var(&mut ssa, 1, 1, DefSite::instruction(3, 0), 1);
        let after_set = var(&mut ssa, 1, 2, DefSite::instruction(5, 1), 1);
        let set = var(&mut ssa, 0, 1, DefSite::instruction(5, 0), 0);
        let state = var(&mut ssa, 1, 3, DefSite::instruction(2, 0), 1);
        let read = var(&mut ssa, 0, 2, DefSite::instruction(4, 0), 0);
        let blocks: [Vec<SsaOp<MockTarget>>; 6] = [
            vec![
                SsaOp::Const {
                    dest: first,
                    value: ConstValue::I32(2),
                },
                SsaOp::Jump { target: 2 },
            ],
            vec![SsaOp::Jump { target: 2 }],
            vec![
                SsaOp::LoadLocal {
                    dest: state,
                    local_index: 1,
                },
                SsaOp::Switch {
                    value: state,
                    targets: vec![3, 4, 5],
                    default: 3,
                },
            ],
            vec![
                SsaOp::Const {
                    dest: back,
                    value: ConstValue::I32(1),
                },
                SsaOp::Jump { target: 1 },
            ],
            vec![
                SsaOp::LoadLocal {
                    dest: read,
                    local_index: 0,
                },
                SsaOp::returns(Some(read)),
            ],
            vec![
                SsaOp::Const {
                    dest: set,
                    value: ConstValue::I32(7),
                },
                SsaOp::Const {
                    dest: after_set,
                    value: ConstValue::I32(0),
                },
                SsaOp::Jump { target: 2 },
            ],
        ];
        for (index, ops) in blocks.into_iter().enumerate() {
            let mut block = SsaBlock::new(index);
            for op in ops {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        ssa.construct_ssa(&SsaConstructOptions::unverified())
            .unwrap_or_else(|_| process::abort());

        let merge = ssa
            .block(2)
            .and_then(|block| {
                block.phi_nodes().iter().find(|phi| {
                    phi.operands().iter().any(|operand| {
                        ssa.variable(operand.value())
                            .is_some_and(|v| v.origin() == VariableOrigin::Local(0))
                    })
                })
            })
            .cloned()
            .expect("the dispatcher merges local 0");
        assert_eq!(merge.operand_from(0).map(PhiOperand::value), Some(entry));
        assert_eq!(
            merge.operand_from(1).map(PhiOperand::value),
            Some(merge.result()),
            "the local goes round B3 and B1 unchanged: {ssa}"
        );
        let returned = ssa
            .block(4)
            .and_then(|block| match block.control_terminator() {
                Some(SsaOp::Return { value, .. }) => *value,
                _ => None,
            });
        assert_eq!(
            returned,
            Some(merge.result()),
            "and the case reads the merge"
        );
    }

    #[test]
    fn a_report_without_verification_is_not_a_clean_one() {
        let mut ssa = entry_with_a_predecessor();
        let options = SsaConstructOptions::unverified();
        let report = ssa
            .construct_ssa(&options)
            .unwrap_or_else(|_| process::abort());
        assert!(report.findings.is_empty());
        assert!(
            !report.verified_clean(&options),
            "nothing was checked, which is not the same as nothing being wrong"
        );
    }
}
