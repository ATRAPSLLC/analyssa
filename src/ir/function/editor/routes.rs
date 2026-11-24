//! Routing edges: splitting them, routing them through or past blocks, and
//! the values that pass through.

use crate::{
    analysis::cfg::SsaCfg,
    error::{Error, Result},
    facts::Derivation,
    graph::{NodeId, algorithms::compute_dominators},
    ir::{
        block::{BlockProvenance, SsaBlock},
        function::editor::{SsaEditScope, SsaEditor},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId},
    },
    target::Target,
};

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Gives the edges from each of `sources` into `target` one block of their
    /// own.
    ///
    /// The new block jumps to `target`; each source's terminator names it in
    /// `target`'s place, and `target`'s phis name it for the value those
    /// edges carried, which must be one value per phi: the edges carry the
    /// same values, and only then are they one edge on. Nothing else moves:
    /// every value that reached `target` along the edges still does, and the
    /// new block, whose predecessors are the sources, is dominated by
    /// whatever dominates all of them. A phi operand for a throw inside a
    /// source names a point in that source and stays with it. The split
    /// disturbs no value, so it needs no repair declared, and a session of
    /// splits alone ends without any ([`SsaEditScope::EdgesSplit`]): no value
    /// is renumbered, and a caller holding facts about values keeps them.
    ///
    /// # Arguments
    ///
    /// * `sources` - The blocks whose edges into `target` go through the new
    ///   block; at least one.
    /// * `target` - The block the edges enter.
    ///
    /// # Returns
    ///
    /// The new block's index, or `None` when a source's terminator does not
    /// transfer to `target`, when the edges carry different values into a phi
    /// of `target`, or when the split would break the function's exception
    /// structure ([`SsaFunction::retarget_respects_regions`](crate::ir::function::SsaFunction::retarget_respects_regions)): the new block
    /// lies in no region, so an edge that stays inside one cannot pass
    /// through it.
    ///
    /// # Errors
    ///
    /// Returns an error if a source has no terminator or `target` does not
    /// exist.
    pub fn split_edges(&mut self, sources: &[usize], target: usize) -> Result<Option<usize>> {
        let split = self.ssa.block_count();
        let Some(phis) = self.ssa.block(target).map(SsaBlock::phi_nodes) else {
            return Err(Error::new(format!("missing block B{target}")));
        };
        if sources.is_empty() {
            return Ok(None);
        }
        // One value per phi along the edges: the new block carries it.
        let mut values: Vec<Option<SsaVarId>> = Vec::with_capacity(phis.len());
        for phi in phis {
            let mut carried = sources.iter().map(|&source| {
                phi.normal_operands()
                    .find(|operand| operand.predecessor() == source)
                    .map(PhiOperand::value)
            });
            let first = carried.next().flatten();
            if carried.any(|value| value != first) {
                return Ok(None);
            }
            values.push(first);
        }
        let mut terminators = Vec::with_capacity(sources.len());
        for &source in sources {
            let Some(mut op) = self
                .ssa
                .block(source)
                .and_then(SsaBlock::control_terminator)
                .cloned()
            else {
                return Err(Error::new(format!("block B{source} has no terminator")));
            };
            if !op.redirect_target(target, split)
                || !self.ssa.retarget_respects_regions(source, target, split)
            {
                return Ok(None);
            }
            terminators.push((source, op));
        }
        if !self.ssa.retarget_respects_regions(split, target, target) {
            return Ok(None);
        }

        // Through the function's own operations: the editor's would widen the
        // session to a repairing scope, and nothing here needs repair.
        let mut block = SsaBlock::new(split);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target }));
        self.ssa.add_block(block);
        for (source, op) in terminators {
            let Some(last) = self
                .ssa
                .block(source)
                .and_then(|block| block.instructions().len().checked_sub(1))
            else {
                return Err(Error::new(format!("block B{source} has no terminator")));
            };
            self.ssa.replace_instruction_op(source, last, op);
            // What a source's edge passed it still passes, on its way into
            // the new block, ahead of the copies the block makes.
            if let Some(block) = self.ssa.block_mut(source) {
                block
                    .provenance_mut()
                    .retarget(|edge| Some(if edge == target { split } else { edge }));
            }
        }
        let Some(entered) = self.ssa.block_mut(target) else {
            return Err(Error::new(format!("missing block B{target}")));
        };
        for (phi, value) in entered.phi_nodes_mut().iter_mut().zip(values) {
            phi.operands_mut().retain(|operand| {
                operand.is_exceptional() || !sources.contains(&operand.predecessor())
            });
            if let Some(value) = value {
                phi.add_operand(PhiOperand::new(value, split));
            }
        }
        self.mark_changed(SsaEditScope::EdgesSplit);
        Ok(Some(split))
    }

    /// Sends each of `froms`' edges into `to` through `through` instead,
    /// which goes on to `to` from there: each terminator names `through` in
    /// `to`'s place, and `to`'s phis no longer take a value from them.
    ///
    /// The caller has established that `through` computes only values, has
    /// no phis, and decides on each edge's way to go on to `to`
    /// ([`crate::passes::unthread`]). Each phi of `to` takes from `through`
    /// what it took from the edges, but those `joined` names: for each of
    /// them, `through` takes a phi of its own, of what `to` took from it on
    /// each way in it had and of what `to` took from each edge on the new
    /// ones, and `to` takes that from it, the phi's value rebuilt on the
    /// rerouted paths ([`Derivation::Reconstruction`]). What `to` took from
    /// `through` is defined above it, so each way in has it. Where `through`
    /// reaches `to` by `via`, a block it alone enters that only jumps there,
    /// `to` takes these from `via` instead, and a constant `via` defines for
    /// it is defined again at the end of each way into `through`, the one
    /// number wherever it is written.
    ///
    /// Each `(value, versions)` of `rejoined` is a value defined above
    /// `through` that each edge holds as its entry of `versions`, defined
    /// where the edge leaves: the value itself, or another version of the
    /// variable the edge carries. `through` takes a phi of `value` on every
    /// way in it had and the versions on the new ones, rebuilt on the
    /// rerouted paths, and reads it: its instructions, and the phi operands
    /// it passes on but to `to`, whose phis that take `value` from it the
    /// caller names among `joined`, read the phi. A block `through` comes to dominate,
    /// where `value` no longer reaches, reads the phi too: past an edge that
    /// held no `value`, only the ways in that held it go on there. Every
    /// value keeps its meaning. The graph changed, so the caller declares
    /// the session repaired ([`Self::declare_repaired`]).
    ///
    /// # Arguments
    ///
    /// * `froms` - The blocks whose edges move.
    /// * `to` - The block the edges entered.
    /// * `through` - The block the edges enter instead.
    /// * `via` - The block `through` reaches `to` by, where it does not go
    ///   there itself.
    /// * `joined` - The indices of `to`'s phis whose value `through` joins.
    /// * `rejoined` - Each value `through` reads or passes on that it joins,
    ///   with each edge's version of it, in the order of `froms`.
    ///
    /// # Errors
    ///
    /// Returns an error if a block of `froms` has no terminator or one that
    /// does not name `to`, `to` or `through` does not exist, a phi `joined`
    /// names lacks an operand from an edge or from `through`, or a value
    /// `rejoined` names has no variable or not one version per edge.
    pub fn route_through(
        &mut self,
        froms: &[usize],
        to: usize,
        through: usize,
        via: Option<usize>,
        joined: &[usize],
        rejoined: &[(SsaVarId, Vec<SsaVarId>)],
    ) -> Result<()> {
        if self.ssa.block(through).is_none() {
            return Err(Error::new(format!("missing block B{through}")));
        }
        // The block `to` is entered from on `through`'s way.
        let arm = via.unwrap_or(through);
        let before = {
            let cfg = SsaCfg::from_ssa(&self.ssa);
            compute_dominators(&cfg, NodeId::new(0))
        };
        // The ways into `through` before the edges join them.
        let entered: Vec<usize> = self
            .ssa
            .blocks()
            .iter()
            .enumerate()
            .filter(|(_, block)| block.successors().contains(&through))
            .map(|(index, _)| index)
            .collect();
        let mut joins: Vec<(usize, PhiNode)> = Vec::with_capacity(joined.len());
        for &index in joined {
            let phi = self
                .ssa
                .block(to)
                .and_then(|block| block.phi_nodes().get(index))
                .cloned()
                .ok_or_else(|| Error::new(format!("B{to} has no phi {index}")))?;
            let missing = || Error::new(format!("B{to}'s phi {index} lacks an operand it joins"));
            let theirs = phi.operand_from(arm).ok_or_else(missing)?.value();
            let ours = froms
                .iter()
                .map(|from| phi.operand_from(*from).map(PhiOperand::value))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(missing)?;
            let (origin, var_type) = self
                .ssa
                .variable(phi.result())
                .map(|var| (var.origin(), var.var_type().clone()))
                .ok_or_else(|| Error::new(format!("B{to}'s phi {index} has no variable")))?;
            let rebuilt = self.ssa.create_variable_derived(
                phi.result(),
                origin,
                0,
                DefSite::phi(through),
                var_type,
                Derivation::Reconstruction,
            );
            let mut join = PhiNode::new(rebuilt, origin);
            // A constant `to` took from `via` is defined there, past
            // `through`: each way into `through` defines it again.
            let defined = self.ssa.variable(theirs).map(|var| var.def_site().block());
            let constant = match self.ssa.get_definition(theirs) {
                Some(SsaOp::Const { value, .. }) => Some(value.clone()),
                _ => None,
            };
            let theirs_type = self
                .ssa
                .variable(theirs)
                .map(|var| var.var_type().clone())
                .ok_or_else(|| Error::new(format!("{theirs:?} has no variable")))?;
            for &predecessor in &entered {
                let held = defined.is_some_and(|block| {
                    before.dominates(NodeId::new(block), NodeId::new(predecessor))
                });
                let value = match (&constant, held) {
                    (_, true) => theirs,
                    (Some(value), false) => {
                        self.constant_at_exit(predecessor, theirs_type.clone(), value.clone())?
                    }
                    (None, false) => {
                        return Err(Error::new(format!(
                            "{theirs:?}, which B{to} takes on B{through}'s way, does not reach B{predecessor}"
                        )));
                    }
                };
                join.add_operand(PhiOperand::new(value, predecessor));
            }
            for (&from, value) in froms.iter().zip(ours) {
                join.add_operand(PhiOperand::new(value, from));
            }
            joins.push((index, join));
        }
        for &from in froms {
            if !self.redirect_terminator_target(from, to, through)? {
                return Err(Error::new(format!(
                    "B{from}'s terminator does not branch to B{to}"
                )));
            }
        }
        if !rejoined.is_empty() {
            let cfg = SsaCfg::from_ssa(&self.ssa);
            let dominators = compute_dominators(&cfg, NodeId::new(0));
            let beneath: Vec<usize> = (0..self.ssa.block_count())
                .filter(|block| dominators.dominates(NodeId::new(through), NodeId::new(*block)))
                .collect();
            for (value, versions) in rejoined {
                let value = *value;
                if versions.len() != froms.len() {
                    return Err(Error::new(format!(
                        "{value:?} has {} versions for {} edges",
                        versions.len(),
                        froms.len()
                    )));
                }
                let (origin, var_type, defined) = self
                    .ssa
                    .variable(value)
                    .map(|var| (var.origin(), var.var_type().clone(), var.def_site().block()))
                    .ok_or_else(|| Error::new(format!("{value:?} has no variable")))?;
                let holds =
                    |block: usize| dominators.dominates(NodeId::new(defined), NodeId::new(block));
                let rebuilt = self.ssa.create_variable_derived(
                    value,
                    origin,
                    0,
                    DefSite::phi(through),
                    var_type,
                    Derivation::Reconstruction,
                );
                let mut join = PhiNode::new(rebuilt, origin);
                for &predecessor in &entered {
                    let arriving = if holds(predecessor) { value } else { rebuilt };
                    join.add_operand(PhiOperand::new(arriving, predecessor));
                }
                for (&from, &version) in froms.iter().zip(versions) {
                    join.add_operand(PhiOperand::new(version, from));
                }
                // What reads the phi: `through` itself, and each block it
                // dominates that `value` no longer reaches.
                for &block in &beneath {
                    if (block == through || !holds(block))
                        && let Some(reader) = self.ssa.block_mut(block)
                    {
                        reader.replace_uses(value, rebuilt, &[]);
                    }
                }
                // `to`'s phis take what each edge carried ([`joined`]).
                for (index, block) in self.ssa.blocks_mut().iter_mut().enumerate() {
                    for phi in block.phi_nodes_mut() {
                        for operand in phi.operands_mut() {
                            let predecessor = operand.predecessor();
                            if operand.value() == value
                                && ((predecessor == through && index != to)
                                    || (beneath.contains(&predecessor) && !holds(predecessor)))
                            {
                                *operand = PhiOperand::new(rebuilt, predecessor);
                            }
                        }
                    }
                }
                if let Some(block) = self.ssa.block_mut(through) {
                    block.add_phi(join);
                }
            }
        }
        for (index, join) in joins {
            let rebuilt = join.result();
            if let Some(block) = self.ssa.block_mut(through) {
                block.add_phi(join);
            }
            if let Some(phi) = self
                .ssa
                .block_mut(to)
                .and_then(|block| block.phi_nodes_mut().get_mut(index))
                && let Some(operand) = phi
                    .operands_mut()
                    .iter_mut()
                    .find(|operand| operand.predecessor() == arm)
            {
                *operand = PhiOperand::new(rebuilt, arm);
            }
        }
        let block = self
            .ssa
            .block_mut(to)
            .ok_or_else(|| Error::new(format!("missing block B{to}")))?;
        for phi in block.phi_nodes_mut() {
            phi.operands_mut().retain(|operand| {
                operand.is_exceptional() || !froms.contains(&operand.predecessor())
            });
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(())
    }

    /// Sends `from`'s edge into `to` to `onward` instead: `from`'s terminator
    /// names `onward` in `to`'s place, each phi of `onward` takes `values`'
    /// entry in its place along the new edge, and `to`'s phis no longer take
    /// a value from `from`.
    ///
    /// The caller has established that `onward`, entered from `from` with
    /// those values, goes on to `to` through blocks that compute what `from`
    /// carried ([`crate::passes::unthread`]): a compiler that knew the answer
    /// on `from`'s way jumped past it. Each value is defined where `from`
    /// ends. Every value keeps its meaning; the graph changed, so the caller
    /// declares the session repaired ([`Self::declare_repaired`]).
    ///
    /// # Arguments
    ///
    /// * `from` - The block whose edge moves.
    /// * `to` - The block the edge entered.
    /// * `onward` - The block the edge enters instead.
    /// * `values` - What each phi of `onward` takes from `from`, in order.
    ///
    /// # Errors
    ///
    /// Returns an error if `onward` does not exist, `values` does not name a
    /// value for each of its phis, `onward` already takes a value from
    /// `from`, or `from`'s terminator does not branch to `to`.
    pub fn send_onward(
        &mut self,
        from: usize,
        to: usize,
        onward: usize,
        values: &[SsaVarId],
    ) -> Result<()> {
        let phis = self
            .ssa
            .block(onward)
            .map(|block| block.phi_nodes().len())
            .ok_or_else(|| Error::new(format!("missing block B{onward}")))?;
        if phis != values.len() {
            return Err(Error::new(format!(
                "B{onward} has {phis} phis and {} values to take",
                values.len()
            )));
        }
        let entered = self.ssa.block(onward).is_some_and(|block| {
            block
                .phi_nodes()
                .iter()
                .any(|phi| phi.operand_from(from).is_some())
        });
        if entered {
            return Err(Error::new(format!(
                "B{onward} already takes a value from B{from}"
            )));
        }
        if !self.redirect_terminator_target(from, to, onward)? {
            return Err(Error::new(format!(
                "B{from}'s terminator does not branch to B{to}"
            )));
        }
        if let Some(block) = self.ssa.block_mut(onward) {
            for (phi, value) in block.phi_nodes_mut().iter_mut().zip(values) {
                phi.add_operand(PhiOperand::new(*value, from));
            }
        }
        let block = self
            .ssa
            .block_mut(to)
            .ok_or_else(|| Error::new(format!("missing block B{to}")))?;
        for phi in block.phi_nodes_mut() {
            phi.operands_mut()
                .retain(|operand| operand.is_exceptional() || operand.predecessor() != from);
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(())
    }

    /// Moves `block`'s rows from `at` on, its terminator among them, into a
    /// block of their own, which `block` then jumps to, so that another way
    /// in can join the moved rows where they begin.
    ///
    /// Each value of `carried` - defined in `block`, by a phi or a row before
    /// `at`, and read past it - takes a phi in the new block with that one
    /// operand from `block`, the same value under a new name
    /// ([`Derivation::Rename`]), and whatever read it at or beneath the new
    /// block reads the phi: a moved row, a block the new block dominates, or
    /// a phi operand coming from one. A way into the new block added later
    /// gives each phi its own value ([`Self::merge_duplicate`]). The new block
    /// holds the code of the host block `block`'s last rows lie in, the last
    /// it carries, and its edges run through what `block`'s did
    /// ([`BlockProvenance::hand_passed_to`]); `block`'s one way out, its jump,
    /// runs through nothing. The graph changed, so the caller declares the
    /// session repaired ([`Self::declare_repaired`]).
    ///
    /// # Arguments
    ///
    /// * `block` - The block split.
    /// * `at` - The first row moved; at least one row stays and at least the
    ///   terminator moves.
    /// * `carried` - The values that take a phi in the new block.
    ///
    /// # Returns
    ///
    /// The new block's index.
    ///
    /// # Errors
    ///
    /// Returns an error if `block` does not exist, `at` keeps no row or moves
    /// none, the function has exception handlers (whose regions name their
    /// blocks), or a carried value does not exist.
    pub fn split_off(&mut self, block: usize, at: usize, carried: &[SsaVarId]) -> Result<usize> {
        if self.ssa.has_exception_handlers() {
            return Err(Error::new(
                "a block of a function with exception handlers is not split".to_string(),
            ));
        }
        let rows = self
            .ssa
            .block(block)
            .map(|block| block.instructions().len())
            .ok_or_else(|| Error::new(format!("missing block B{block}")))?;
        if at == 0 || at >= rows {
            return Err(Error::new(format!(
                "B{block} split at {at} of {rows} rows keeps or moves nothing"
            )));
        }
        let split = self.ssa.block_count();
        let mut moved = SsaBlock::new(split);
        let successors = {
            let source = self
                .ssa
                .block_mut(block)
                .ok_or_else(|| Error::new(format!("missing block B{block}")))?;
            for row in source.instructions_mut().drain(at..) {
                moved.add_instruction(row);
            }
            source.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: split }));
            if let Some(&host) = source.provenance().carried().last() {
                moved.provenance_mut().carry(host);
            }
            source
                .provenance_mut()
                .hand_passed_to(moved.provenance_mut());
            moved.successors()
        };
        self.ssa.add_block(moved);
        for successor in successors {
            self.retarget_phi_operands(successor, block, split)?;
        }
        if !carried.is_empty() {
            let cfg = SsaCfg::from_ssa(&self.ssa);
            let dominators = compute_dominators(&cfg, NodeId::new(0));
            let beneath: Vec<usize> = (0..self.ssa.block_count())
                .filter(|index| dominators.dominates(NodeId::new(split), NodeId::new(*index)))
                .collect();
            for &value in carried {
                let (origin, var_type) = self
                    .ssa
                    .variable(value)
                    .map(|var| (var.origin(), var.var_type().clone()))
                    .ok_or_else(|| Error::new(format!("{value:?} has no variable")))?;
                let renamed = self.ssa.create_variable_derived(
                    value,
                    origin,
                    0,
                    DefSite::phi(split),
                    var_type,
                    Derivation::Rename,
                );
                for &index in &beneath {
                    if let Some(found) = self.ssa.block_mut(index) {
                        found.replace_uses(value, renamed, &[]);
                    }
                }
                for found in self.ssa.blocks_mut() {
                    for phi in found.phi_nodes_mut() {
                        for operand in phi.operands_mut() {
                            if operand.value() == value && beneath.contains(&operand.predecessor())
                            {
                                *operand = PhiOperand::new(renamed, operand.predecessor());
                            }
                        }
                    }
                }
                let mut phi = PhiNode::new(renamed, origin);
                phi.add_operand(PhiOperand::new(value, block));
                if let Some(found) = self.ssa.block_mut(split) {
                    found.add_phi(phi);
                }
            }
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(split)
    }

    /// Replaces one predecessor in phi operands with one or more predecessors.
    ///
    /// For each phi in `block_idx`, the first operand whose predecessor is
    /// `old_pred` is rewritten to the first value in `new_preds`. Additional
    /// predecessors receive duplicate operands carrying the same value. This
    /// preserves phi value attribution when multiple incoming edges are
    /// redirected through a single removed block.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose phis are rewritten.
    /// * `old_pred` - The incoming predecessor being expanded.
    /// * `new_preds` - Predecessors that replace it; an empty slice is a no-op.
    ///
    /// # Returns
    ///
    /// The number of phi operands rewritten or added.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist, or if a predecessor in
    /// `new_preds` already carries a different value for a phi than
    /// `old_pred` brought.
    pub fn expand_phi_predecessor(
        &mut self,
        block_idx: usize,
        old_pred: usize,
        new_preds: &[usize],
    ) -> Result<usize> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        if new_preds.is_empty() {
            return Ok(0);
        }

        let mut updated = 0usize;
        for phi in block.phi_nodes_mut() {
            // The edge being replaced is a terminator's. An operand for a throw
            // from the same block describes a different arrival, which the
            // redirect does not touch.
            let Some(value) = phi
                .normal_operands()
                .find(|operand| operand.predecessor() == old_pred)
                .map(|operand| operand.value())
            else {
                continue;
            };

            // Drop the edge being replaced, then re-add one operand per new
            // predecessor - but never for a predecessor the phi already names.
            //
            // A predecessor can reach this block *both* directly and through the
            // block being bypassed; redirecting then collapses the two paths onto
            // one edge. Adding an operand regardless leaves two operands for that
            // single edge, which has no defined meaning and which consumers
            // disagree about: `PhiNode::operand_from` returns the first and
            // discards the rest, while SCCP meets them all and yields Bottom.
            // `retarget_phi_predecessor` already guards this case; this is the
            // same guard for the one-to-many form.
            //
            // When an operand for that predecessor already exists it is kept as
            // it stands: it describes the direct edge, which is the edge that
            // survives - provided it carries the same value. One that does not
            // would leave the collapsed edge two values, and keeping either
            // silently drops the other.
            for &pred in new_preds {
                if pred == old_pred {
                    continue;
                }
                if let Some(existing) = phi
                    .normal_operands()
                    .find(|operand| operand.predecessor() == pred)
                    .map(PhiOperand::value)
                    && existing != value
                {
                    return Err(Error::new(format!(
                        "B{block_idx}: B{pred} already carries {existing}, not the {value} \
                         B{old_pred} brought"
                    )));
                }
            }
            phi.operands_mut()
                .retain(|operand| operand.is_exceptional() || operand.predecessor() != old_pred);
            updated = updated.saturating_add(1);

            for &pred in new_preds {
                if phi
                    .normal_operands()
                    .any(|operand| operand.predecessor() == pred)
                {
                    continue;
                }
                phi.add_operand(PhiOperand::new(value, pred));
                updated = updated.saturating_add(1);
            }
        }

        if updated > 0 {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(updated)
    }

    /// Rewrites `block_idx`'s edge into a forwarding chain to go straight to
    /// where the chain ends.
    ///
    /// `entry` must start a chain of blocks that each hold only a jump and no
    /// phis, whose last block `last` jumps to `target`. The destination's phis
    /// gain an operand for `block_idx` carrying what arrived from `last`: the
    /// chain defines nothing, so that is what this path always brought.
    ///
    /// **No reconstruction.** Every path over the new edge is a path the
    /// function already had, through blocks that define nothing, so every
    /// definition still dominates every use it did. The phi operands are the
    /// only thing a changed edge can leave wrong, and they are set here.
    ///
    /// The chain's host blocks still run on the new edge, with nothing of them
    /// left to execute, so `block_idx` records them as passed through
    /// ([`BlockProvenance::pass_through`]).
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose terminator is rewritten.
    /// * `entry` - The first block of the chain, which the terminator names.
    /// * `target` - Where the chain ends.
    /// * `last` - The chain's last block, the one that jumps to `target`.
    ///
    /// # Returns
    ///
    /// `true` if the terminator named `entry` and was rewritten.
    ///
    /// # Errors
    ///
    /// Returns an error if the chain is not a chain of forwarding blocks
    /// ending at `target` through `last`, or if a phi of `target` cannot take
    /// the operand (see [`Self::duplicate_phi_edge`]).
    pub fn bypass_forwarding_chain(
        &mut self,
        block_idx: usize,
        entry: usize,
        target: usize,
        last: usize,
    ) -> Result<bool> {
        let mut current = entry;
        let mut chain = Vec::new();
        for _ in 0..=self.ssa.block_count() {
            let next = self
                .ssa
                .block(current)
                .and_then(SsaBlock::is_trampoline)
                .ok_or_else(|| Error::new(format!("B{current} is not a forwarding block")))?;
            chain.push(current);
            if next == target {
                if current != last {
                    return Err(Error::new(format!(
                        "the chain from B{entry} reaches B{target} through B{current}, not B{last}"
                    )));
                }
                break;
            }
            current = next;
        }
        if self.ssa.block(current).and_then(SsaBlock::is_trampoline) != Some(target) {
            return Err(Error::new(format!(
                "the chain from B{entry} does not end at B{target}"
            )));
        }

        let Some(mut op) = self
            .ssa
            .block(block_idx)
            .and_then(SsaBlock::control_terminator)
            .cloned()
        else {
            return Err(Error::new(format!("block B{block_idx} has no terminator")));
        };
        if !op.redirect_target(entry, target) {
            return Ok(false);
        }
        self.add_edge_like(target, last, block_idx)?;
        let Some(last_idx) = self
            .ssa
            .block(block_idx)
            .and_then(|block| block.instructions().len().checked_sub(1))
        else {
            return Err(Error::new(format!("block B{block_idx} has no terminator")));
        };
        self.replace_instruction_op(block_idx, last_idx, op)?;
        if let Some(block) = self.ssa.block_mut(block_idx) {
            block
                .provenance_mut()
                .retarget(|edge| Some(if edge == entry { target } else { edge }));
        }
        self.record_passed_through(block_idx, target, &chain);
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(true)
    }

    /// Records that an edge out of `block_idx` now runs past `bypassed`: the
    /// host blocks they stand for still run on that edge, with nothing of them
    /// left to execute ([`BlockProvenance::pass_through`]).
    ///
    /// Called by every rewrite that sends an edge around blocks instead of
    /// through them. The bypassed blocks keep their own record for whatever
    /// still reaches them; one nothing reaches any more is cleared as dead,
    /// and its host blocks stay accounted for here.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block whose edge was rewritten.
    /// * `successor` - The block the rewritten edge now leads to.
    /// * `bypassed` - The blocks the edge no longer passes through.
    pub fn record_passed_through(
        &mut self,
        block_idx: usize,
        successor: usize,
        bypassed: &[usize],
    ) {
        let mut passed = BlockProvenance::default();
        for block in bypassed.iter().filter_map(|&index| self.ssa.block(index)) {
            passed.pass_through(block.provenance());
        }
        if let Some(block) = self.ssa.block_mut(block_idx) {
            block.provenance_mut().pass_through_on(successor, &passed);
        }
    }

    /// Restates the edges `block_idx`'s record passes host blocks on, as a
    /// rewrite that retargets them leaves them ([`BlockProvenance::retarget`]):
    /// `target` says where the edge to each successor leads now, or `None`
    /// for nowhere it knows, which passes its hosts on every edge out.
    pub fn retarget_passed(&mut self, block_idx: usize, target: impl Fn(usize) -> Option<usize>) {
        if let Some(block) = self.ssa.block_mut(block_idx) {
            block.provenance_mut().retarget(target);
        }
    }
}
