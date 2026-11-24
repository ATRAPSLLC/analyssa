//! Scoped SSA reconstruction for the values one edit disturbs.
//!
//! A transform that adds a control-flow edge can make a value's definition stop
//! reaching some of its reads: control now arrives at a block without having
//! passed the definition. Re-placing the phis that answer for it is standard
//! SSA construction, but it does **not** need to be done for the whole
//! function. A renaming [`SsaFunction::construct_ssa`] is the whole-function
//! form, and it reconstructs by rename group - the set of SSA values that are versions of
//! one storage location - which only IR whose frontend states that grouping,
//! and keeps it complete through every pass, can use.
//!
//! This module is the other form: **name the values the edit disturbed and the
//! value each new edge carries, and reconstruct exactly those.** Nothing else
//! is touched, no grouping is consulted, and the answer is decided from the
//! graph as it stands after the edit.
//!
//! # What it does
//!
//! For one [`ValueUpdate`], the definitions of "the variable" are the block
//! defining `value` plus the tail of each new edge. Phis go at the iterated
//! dominance frontier of those blocks, pruned to where the variable is live, and
//! every read of `value` is rewritten to the definition that now reaches it. A
//! read the edit did not disturb resolves to `value` itself and is left alone.
//!
//! # What it refuses
//!
//! A phi needs a value on every incoming edge. When a predecessor has none -
//! the variable is not defined on any path reaching it - the reconstruction is
//! abandoned rather than invented, because the alternative is a read answered
//! by a value that never reached it. The caller's edit session then rolls back.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    BitSet, Error, Result,
    analysis::cfg::SsaCfg,
    facts::Derivation,
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominance_frontiers, compute_dominators},
    },
    ir::{
        function::SsaFunction,
        phi::{PhiEdge, PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId},
    },
    target::Target,
};

/// One value to reconstruct, and where the variable it names is defined now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueUpdate {
    /// The value whose reads the edit may have cut off from their definition.
    ///
    /// Its own definition is one of the variable's definitions too, when the
    /// edit left it reachable. When the edit deleted it - a dispatcher's phi
    /// that unflattening removes - [`Self::definitions`] is the whole story.
    pub value: SsaVarId,
    /// `(block, value)`: at the end of `block`, the variable this reconstructs
    /// holds `value`.
    ///
    /// Per block rather than per edge, because that is what a definition is: a
    /// block that assigns the variable assigns it for every successor. An edit
    /// that reroutes a branch states the definitions its rerouting exposed -
    /// for a dispatcher, the values its cases write back.
    pub definitions: Vec<(usize, SsaVarId)>,
}

/// What a reconstruction changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reconstruction {
    /// Phi nodes placed.
    pub phis: usize,
    /// Reads, instruction operands and phi operands, rewritten to the
    /// definition that now reaches them.
    pub rewritten: usize,
}

impl Reconstruction {
    /// Whether the reconstruction changed the function.
    #[must_use]
    pub const fn changed(&self) -> bool {
        self.phis > 0 || self.rewritten > 0
    }
}

/// The definition reaching a point: a value that already exists, or one of the
/// phis this reconstruction is about to place.
///
/// A phi cannot be named by id while the plan is being made - its result is
/// minted only once every placement is known, and a phi's own operands can name
/// another phi - so it is named by the block it goes in and resolved at the
/// end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Def {
    /// A value the function already has.
    Value(SsaVarId),
    /// The phi this reconstruction places in the given block.
    NewPhi(usize),
}

/// Where the variable stands at the end of each block.
struct Reaching {
    /// Definition reaching the end of each block, once the walk has visited it.
    at_end: BTreeMap<usize, Def>,
    /// Blocks the dominator-tree walk entered, which is exactly the blocks the
    /// program can reach. A predecessor outside this set is an edge from code
    /// that no longer runs: the edit's boundary drops it, and asking it for a
    /// value would refuse a reconstruction over a question nothing asks.
    reachable: BTreeSet<usize>,
}

impl Reaching {
    /// The definition a successor's phi should read for the edge from `block`.
    fn on(&self, block: usize) -> Option<Def> {
        self.at_end.get(&block).copied()
    }
}

impl<T: Target> SsaFunction<T> {
    /// Reconstructs the values an edge-adding edit disturbed.
    ///
    /// Each update names one value and where the variable it stands for is
    /// defined now; the reconstruction places the phis those definitions call
    /// for and rewrites every read to the one that reaches it. A read the edit
    /// did not disturb still resolves to `value` and is left alone. Each update
    /// is reconstructed on its own, against the graph as it stands.
    ///
    /// # Arguments
    ///
    /// * `updates` - The values to reconstruct and the values the new edges
    ///   carry for them.
    ///
    /// # Returns
    ///
    /// The phis placed and the reads rewritten.
    ///
    /// # Errors
    ///
    /// Returns an error when a phi the reconstruction needs has a predecessor
    /// the variable does not reach, which is the case it refuses to invent a
    /// value for.
    pub fn reconstruct_values(&mut self, updates: &[ValueUpdate]) -> Result<Reconstruction> {
        if updates.is_empty() || self.blocks.is_empty() {
            return Ok(Reconstruction::default());
        }
        let placed = {
            let cfg = SsaCfg::from_ssa(self);
            let dominators = compute_dominators(&cfg, NodeId::new(0));
            let frontiers = compute_dominance_frontiers(&cfg, &dominators);
            let mut plans = Vec::new();
            for update in updates {
                plans.push(plan_update(self, &cfg, &dominators, &frontiers, update)?);
            }
            plans
        };
        let mut total = Reconstruction::default();
        for plan in placed {
            let applied = apply_plan(self, plan)?;
            total.phis = total.phis.saturating_add(applied.phis);
            total.rewritten = total.rewritten.saturating_add(applied.rewritten);
        }
        Ok(total)
    }
}

/// One reconstruction, resolved against the pre-edit graph and ready to apply.
struct Plan {
    /// The value being reconstructed.
    value: SsaVarId,
    /// Blocks that receive a new phi, in block order.
    phi_blocks: Vec<usize>,
    /// Where the variable stands at the end of every block.
    reaching: Reaching,
    /// Instruction reads to rewrite: `(block, instruction, replacement)`.
    reads: Vec<(usize, usize, Def)>,
    /// Phi operands to rewrite: `(block, phi index, operand index, replacement)`.
    operands: Vec<(usize, usize, usize, Def)>,
}

/// Decides where `update`'s phis go and what every read of it resolves to.
fn plan_update<T: Target>(
    ssa: &SsaFunction<T>,
    cfg: &SsaCfg<'_, T>,
    dominators: &DominatorTree,
    frontiers: &[BitSet],
    update: &ValueUpdate,
) -> Result<Plan> {
    let value = update.value;
    let block_count = ssa.blocks.len();
    let own_def = definition_block(ssa, value);

    let stated: BTreeMap<usize, SsaVarId> = update
        .definitions
        .iter()
        .filter(|(block, _)| *block < block_count)
        .map(|(block, value)| (*block, *value))
        .collect();
    let mut def_blocks: BTreeSet<usize> = stated.keys().copied().collect();
    if let Some(block) = own_def {
        def_blocks.insert(block);
    }

    // A block that states a definition kills the variable for everything above
    // it, exactly as the value's own definition does - the definition holds
    // from the end of the block, so nothing before it needs the variable to
    // arrive. Without this a stated definition inside a loop header asks for a
    // phi *at* that header, whose operand on the entry edge is a value the
    // variable does not have there.
    //
    // A block that also *reads* the value is not a kill: the read happens
    // before the definition it states.
    let mut kills: BTreeSet<usize> = own_def.into_iter().collect();
    kills.extend(
        stated
            .keys()
            .copied()
            .filter(|block| !reads_value(ssa, *block, value)),
    );
    let live = live_blocks(ssa, cfg, value, &kills, block_count);
    let phi_blocks = phi_placement(&def_blocks, frontiers, &live, block_count);

    let placed: BTreeSet<usize> = phi_blocks.iter().copied().collect();

    let mut reaching = Reaching {
        at_end: BTreeMap::new(),
        reachable: BTreeSet::new(),
    };
    let mut reads = Vec::new();
    let mut operands = Vec::new();
    walk(
        ssa,
        dominators,
        value,
        own_def,
        &stated,
        &placed,
        &mut reaching,
        &mut reads,
        &mut operands,
        cfg,
    );

    // Every new phi must have a value on every edge into its block. One that
    // does not is a path the variable never reached, and inventing an operand
    // there answers a read with a value that never arrived.
    for block in &phi_blocks {
        for predecessor in cfg.block_predecessors(*block) {
            if !reaching.reachable.contains(predecessor) {
                continue;
            }
            if reaching.on(*predecessor).is_none() {
                return Err(Error::new(format!(
                    "cannot reconstruct {value:?}: block B{block} is entered from \
                     B{predecessor}, which the value does not reach"
                )));
            }
            // A throw leaves its block from wherever it happened, so what
            // reaches the handler is every value the block held, not the one it
            // ends with. This walk knows only the latter. When the block
            // defines the variable the two differ and the operand set cannot be
            // stated here, so the reconstruction is refused rather than
            // understated - the same rule as a value that does not reach at
            // all. When it does not define it, the block held one value
            // throughout and the end value is exactly that one.
            if cfg.is_exception_edge(*predecessor, *block) && def_blocks.contains(predecessor) {
                return Err(Error::new(format!(
                    "cannot reconstruct {value:?}: a throw in B{predecessor} reaches B{block} \
                     from part-way through a block that redefines the variable"
                )));
            }
        }
    }

    Ok(Plan {
        value,
        phi_blocks,
        reaching,
        reads,
        operands,
    })
}

/// The block that defines `value`, whether by phi or by instruction.
fn definition_block<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> Option<usize> {
    for (block_idx, block) in ssa.blocks.iter().enumerate() {
        if block.phi_nodes().iter().any(|phi| phi.result() == value) {
            return Some(block_idx);
        }
        for instr in block.instructions() {
            if instr.op().defs().any(|dest| dest == value) {
                return Some(block_idx);
            }
        }
    }
    None
}

/// The blocks the variable is live *into*, so a phi is placed only where one is
/// read.
///
/// Backward from every block that reads `value` - a phi operand counts at the
/// predecessor whose edge supplies it - and stopping at a block that defines
/// the variable, because nothing above such a block needs it to arrive.
fn live_blocks<T: Target>(
    ssa: &SsaFunction<T>,
    cfg: &SsaCfg<'_, T>,
    value: SsaVarId,
    kills: &BTreeSet<usize>,
    block_count: usize,
) -> BitSet {
    let mut live = BitSet::new(block_count);
    let mut pending: Vec<usize> = Vec::new();
    for (block_idx, block) in ssa.blocks.iter().enumerate() {
        if block
            .instructions()
            .iter()
            .any(|instr| instr.op().uses_var(value))
        {
            pending.push(block_idx);
        }
        for phi in block.phi_nodes() {
            for operand in phi.operands() {
                if operand.value() == value {
                    pending.push(operand.predecessor());
                }
            }
        }
    }
    while let Some(block_idx) = pending.pop() {
        if block_idx >= block_count || kills.contains(&block_idx) || !live.insert(block_idx) {
            continue;
        }
        pending.extend(cfg.block_predecessors(block_idx).iter().copied());
    }
    live
}

/// Whether `block` reads `value`, which is what stops a definition it states
/// from killing the variable above it.
///
/// # Arguments
///
/// * `ssa` - The SSA function being reconstructed over.
/// * `block` - The block to inspect.
/// * `value` - The value whose reads are looked for.
fn reads_value<T: Target>(ssa: &SsaFunction<T>, block: usize, value: SsaVarId) -> bool {
    ssa.block(block).is_some_and(|this| {
        this.instructions()
            .iter()
            .any(|instr| instr.op().uses_var(value))
    }) || throws_value(ssa, block, value)
}

/// Whether a handler's phi names `value` for a throw in `block`.
///
/// A throw reads the variable where it happens, part-way through the block, so
/// it is a read the block makes before any definition it states for its end.
///
/// # Arguments
///
/// * `ssa` - The SSA function being reconstructed over.
/// * `block` - The block the throw is in.
/// * `value` - The value the handler's operand names.
fn throws_value<T: Target>(ssa: &SsaFunction<T>, block: usize, value: SsaVarId) -> bool {
    ssa.blocks.iter().any(|handler| {
        handler.phi_nodes().iter().any(|phi| {
            phi.operands().iter().any(|operand| {
                operand.is_exceptional()
                    && operand.predecessor() == block
                    && operand.value() == value
            })
        })
    })
}

/// The blocks that receive a phi: the iterated dominance frontier of the
/// definitions, kept where the variable is live.
fn phi_placement(
    def_blocks: &BTreeSet<usize>,
    frontiers: &[BitSet],
    live: &BitSet,
    block_count: usize,
) -> Vec<usize> {
    let mut placed = BitSet::new(block_count);
    let mut pending: Vec<usize> = def_blocks.iter().copied().collect();
    let mut seen: BTreeSet<usize> = def_blocks.clone();
    while let Some(block_idx) = pending.pop() {
        let Some(frontier) = frontiers.get(block_idx) else {
            continue;
        };
        for target in frontier.iter() {
            if !live.contains(target) || !placed.insert(target) {
                continue;
            }
            if seen.insert(target) {
                pending.push(target);
            }
        }
    }
    placed.iter().collect()
}

/// Walks the dominator tree recording what reaches each point, and collects the
/// reads and phi operands the new definitions answer.
#[expect(clippy::too_many_arguments, reason = "one walk, all of its outputs")]
fn walk<T: Target>(
    ssa: &SsaFunction<T>,
    dominators: &DominatorTree,
    value: SsaVarId,
    own_def: Option<usize>,
    stated: &BTreeMap<usize, SsaVarId>,
    placed: &BTreeSet<usize>,
    reaching: &mut Reaching,
    reads: &mut Vec<(usize, usize, Def)>,
    operands: &mut Vec<(usize, usize, usize, Def)>,
    cfg: &SsaCfg<'_, T>,
) {
    let mut stack = vec![(0usize, None::<Def>)];
    while let Some((block_idx, incoming)) = stack.pop() {
        let Some(block) = ssa.blocks.get(block_idx) else {
            continue;
        };
        reaching.reachable.insert(block_idx);
        // A phi placed here answers for the variable from the top of the block;
        // otherwise what reaches here is what reached the end of the immediate
        // dominator.
        let mut current = if placed.contains(&block_idx) {
            Some(Def::NewPhi(block_idx))
        } else {
            incoming
        };
        // The value's own phi defines it at the top of its block, ahead of
        // every instruction there: a read in the block reads the phi, and so
        // does a throw out of it.
        if Some(block_idx) == own_def && block.phi_nodes().iter().any(|phi| phi.result() == value) {
            current = Some(Def::Value(value));
        }
        let at_entry = current;

        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            if instr.op().uses_var(value)
                && let Some(reaching_def) = current
                && reaching_def != Def::Value(value)
            {
                reads.push((block_idx, instr_idx, reaching_def));
            }
            if instr.op().defs().any(|dest| dest == value) {
                current = Some(Def::Value(value));
            }
        }
        // A definition the caller stated holds from the end of its block, so it
        // supersedes whatever the block's own instructions left standing.
        if let Some(stated_value) = stated.get(&block_idx) {
            current = Some(Def::Value(*stated_value));
        }
        if let Some(reaching_def) = current {
            reaching.at_end.insert(block_idx, reaching_def);
        }

        // An existing phi in a successor that reads the value on this edge now
        // reads whatever reaches the end of this block instead.
        for successor in cfg.block_successors(block_idx) {
            let Some(target) = ssa.blocks.get(*successor) else {
                continue;
            };
            for (phi_idx, phi) in target.phi_nodes().iter().enumerate() {
                // By operand index, not by predecessor: one predecessor can
                // supply a normal operand and several exceptional ones, and
                // each states a different thing about the same edge. Rewriting
                // by predecessor would collapse them into one normal operand,
                // which claims the block ran to completion.
                for (operand_idx, operand) in phi.operands().iter().enumerate() {
                    if operand.predecessor() != block_idx || operand.value() != value {
                        continue;
                    }
                    // A throw is seen where it happened. One naming `value` in
                    // a block that does not define it came while the block
                    // still held what it was entered with - before any
                    // definition stated for the block's end, which is what
                    // a transfer out of it carries instead.
                    let edge = if operand.is_exceptional() {
                        if own_def == Some(block_idx) {
                            continue;
                        }
                        at_entry
                    } else {
                        reaching.on(block_idx)
                    };
                    let Some(edge) = edge else {
                        continue;
                    };
                    if edge != Def::Value(value) {
                        operands.push((*successor, phi_idx, operand_idx, edge));
                    }
                }
            }
        }

        for child in dominators.children(NodeId::new(block_idx)) {
            stack.push((child.index(), current));
        }
    }
}

/// Creates the planned phis, fills every operand, and rewrites the reads.
fn apply_plan<T: Target>(ssa: &mut SsaFunction<T>, plan: Plan) -> Result<Reconstruction> {
    let Plan {
        value,
        phi_blocks,
        reaching,
        reads,
        operands,
    } = plan;
    if phi_blocks.is_empty() && reads.is_empty() && operands.is_empty() {
        return Ok(Reconstruction::default());
    }

    let (origin, var_type) = ssa
        .variable(value)
        .map(|var| (var.origin(), var.var_type().clone()))
        .ok_or_else(|| Error::new(format!("cannot reconstruct missing value {value:?}")))?;

    // Mint the results first, then resolve the placeholders the walk recorded:
    // a phi's own operands can name another phi, so no order of creation makes
    // every id known in advance.
    let mut results: BTreeMap<usize, SsaVarId> = BTreeMap::new();
    for block_idx in &phi_blocks {
        // Each result stands for `value` on a path that no longer reaches its
        // definition, so it starts from what the host knew about `value`; the
        // join over the operands runs once they are filled in, at the repair
        // boundary this edit ends at.
        let result = ssa.create_variable_derived(
            value,
            origin,
            0,
            DefSite::phi(*block_idx),
            var_type.clone(),
            Derivation::Reconstruction,
        );
        results.insert(*block_idx, result);
    }
    let resolve = |def: Def| -> Option<SsaVarId> {
        match def {
            Def::Value(var) => Some(var),
            Def::NewPhi(block) => results.get(&block).copied(),
        }
    };

    // One entry per operand the phi needs, not per predecessor: an operand for
    // a throw states what a handler may see, which is a different claim from an
    // operand for a transfer a terminator names, and one predecessor can owe
    // both. A protected block that leaves to its own `finally` is the case -
    // it gets there by running to the leave, and by throwing before it.
    let predecessors: BTreeMap<usize, Vec<(usize, PhiEdge)>> = {
        let cfg = SsaCfg::from_ssa(ssa);
        phi_blocks
            .iter()
            .map(|block| {
                let mut edges = Vec::new();
                for pred in cfg.block_predecessors(*block) {
                    if cfg.terminator_successors(*pred).contains(block) {
                        edges.push((*pred, PhiEdge::Normal));
                    }
                    if cfg.is_exception_edge(*pred, *block) {
                        edges.push((*pred, PhiEdge::Exceptional));
                    }
                }
                edges.dedup();
                (*block, edges)
            })
            .collect()
    };

    for block_idx in &phi_blocks {
        let Some(result) = results.get(block_idx).copied() else {
            continue;
        };
        let mut phi = PhiNode::new(result, origin);
        for (predecessor, edge) in predecessors.get(block_idx).into_iter().flatten() {
            // An edge from code the program can no longer enter asks nothing:
            // the edit's boundary drops the block and the operand with it.
            if !reaching.reachable.contains(predecessor) {
                continue;
            }
            let carried = reaching.on(*predecessor).and_then(resolve).ok_or_else(|| {
                Error::new(format!(
                    "cannot reconstruct {value:?}: B{block_idx} is entered from \
                         B{predecessor}, which the value does not reach"
                ))
            })?;
            // Planning refused every throwing predecessor that redefines the
            // variable, so the one value reaching the end of this one is the
            // one it held throughout, and a single operand states it exactly.
            match edge {
                PhiEdge::Exceptional => {
                    phi.add_exceptional_operand(*predecessor, carried);
                }
                PhiEdge::Normal => phi.add_operand(PhiOperand::new(carried, *predecessor)),
            }
        }
        if let Some(block) = ssa.block_mut(*block_idx) {
            block.add_phi(phi);
        }
    }

    let mut rewritten = 0usize;
    for (block_idx, instr_idx, replacement) in reads {
        let Some(replacement) = resolve(replacement) else {
            continue;
        };
        if let Some(block) = ssa.block_mut(block_idx)
            && let Some(instr) = block.instructions_mut().get_mut(instr_idx)
        {
            instr.op_mut().replace_uses(value, replacement);
            rewritten = rewritten.saturating_add(1);
        }
    }
    for (block_idx, phi_idx, operand_idx, replacement) in operands {
        let Some(replacement) = resolve(replacement) else {
            continue;
        };
        if let Some(block) = ssa.block_mut(block_idx)
            && let Some(phi) = block.phi_nodes_mut().get_mut(phi_idx)
            && let Some(operand) = phi.operands_mut().get_mut(operand_idx)
        {
            // In place, so the operand stays the kind of edge it was.
            operand.set_value(replacement);
            rewritten = rewritten.saturating_add(1);
        }
    }

    ssa.refresh_def_sites();
    ssa.recompute_uses();
    Ok(Reconstruction {
        phis: phi_blocks.len(),
        rewritten,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            function::SsaEditOptions,
            instruction::SsaInstruction,
            ops::SsaOp,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::VariableOrigin,
        },
        testing::{MockTarget, MockType},
    };

    /// An instruction with no recorded source position.
    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    /// `b0 -> b1 -> b2`, with `b1` defining the value `b2` returns.
    ///
    /// Adding an edge `b0 -> b2` is then exactly the shape an edge-adding edit
    /// leaves behind: `b2`'s read is no longer reached by `b1`'s definition on
    /// every path.
    fn straight_line() -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let early = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let late = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: early,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: late,
            value: ConstValue::I32(2),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(Some(late))));
        ssa.add_block(b2);
        ssa.recompute_uses();
        (ssa, early, late)
    }

    /// A read the new edge cut off is answered by a phi that merges what each
    /// path brings, and the read is rewritten to it.
    /// **A reconstruction that only rewrites reads is a change.** `b1` now
    /// states `early` for the variable, so `b2`'s read is answered by it with
    /// no join to place; the session that asked must still see the function
    /// change and repair it at its boundary.
    #[test]
    fn a_reconstruction_rewriting_only_reads_changes_the_function() {
        let (mut ssa, early, late) = straight_line();
        let mut reconstructed = Reconstruction::default();
        let report = ssa
            .edit(SsaEditOptions::new(), |editor| {
                reconstructed = editor.reconstruct_values(&[ValueUpdate {
                    value: late,
                    definitions: vec![(1, early)],
                }])?;
                Ok(())
            })
            .expect("the definition reaches the read");
        assert_eq!(
            reconstructed,
            Reconstruction {
                phis: 0,
                rewritten: 1
            }
        );
        assert!(report.changed, "the read was rewritten");
        assert!(matches!(
            ssa.block(2).and_then(|block| block.control_terminator()),
            Some(SsaOp::Return { value: Some(value), .. }) if *value == early
        ));
    }

    #[test]
    fn a_read_the_new_edge_reaches_gets_a_phi_for_both_paths() {
        let (mut ssa, early, late) = straight_line();
        // The edit: `b0` now branches to `b2` as well, carrying `early`.
        if let Some(block) = ssa.block_mut(0)
            && let Some(terminator) = block.instructions_mut().last_mut()
        {
            terminator.set_op(SsaOp::Branch {
                condition: early,
                true_target: 1,
                false_target: 2,
            });
        }

        let placed = ssa
            .reconstruct_values(&[ValueUpdate {
                value: late,
                definitions: vec![(0, early), (1, late)],
            }])
            .expect("the value is defined on both paths");
        assert_eq!(placed.phis, 1, "one join, one phi");

        let phis = ssa.block(2).map(|block| block.phi_nodes().to_vec());
        let phi = phis
            .as_ref()
            .and_then(|phis| phis.first())
            .expect("a phi at the join");
        let mut operands: Vec<(usize, SsaVarId)> = phi
            .operands()
            .iter()
            .map(|operand| (operand.predecessor(), operand.value()))
            .collect();
        operands.sort_unstable();
        assert_eq!(
            operands,
            vec![(0, early), (1, late)],
            "each path brings its own definition"
        );
        assert_eq!(
            ssa.block(2)
                .and_then(|block| block.instructions().first())
                .map(|instr| instr.op().clone()),
            Some(SsaOp::returns(Some(phi.result()))),
            "the read is answered by the phi, not by the definition it lost"
        );
    }

    /// **A read in a phi's own block reads the phi.** The dispatcher `b1`
    /// joins `v = phi(c from b0, w from b2)` and branches on `v`; `b2` loops
    /// back and `b3` returns `v`. With `c` stated for `v` at the end of `b0`,
    /// the branch in `b1` and the return in `b3` still read the phi, which
    /// defines `v` at `b1`'s top, ahead of every instruction there: answered
    /// by what reaches from `b0`, the dispatcher would always take `c`'s case.
    #[test]
    fn a_read_in_a_phis_own_block_reads_the_phi() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let variable = |ssa: &mut SsaFunction<MockTarget>, local, site| {
            ssa.create_variable(VariableOrigin::Local(local), 0, site, MockType::I32)
        };
        let c = variable(&mut ssa, 0, DefSite::instruction(0, 0));
        let v = variable(&mut ssa, 1, DefSite::phi(1));
        let w = variable(&mut ssa, 2, DefSite::instruction(2, 0));
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: c,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(v, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(c, 0));
        phi.add_operand(PhiOperand::new(w, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: v,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: w,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(Some(v))));
        ssa.add_block(b3);
        ssa.recompute_uses();

        ssa.reconstruct_values(&[ValueUpdate {
            value: v,
            definitions: vec![(0, c)],
        }])
        .expect("the update applies");
        assert!(
            matches!(
                ssa.block(1).and_then(|block| block.control_terminator()),
                Some(SsaOp::Branch { condition, .. }) if *condition == v
            ),
            "the dispatcher branches on the phi"
        );
        assert!(
            matches!(
                ssa.block(3).and_then(|block| block.control_terminator()),
                Some(SsaOp::Return { value: Some(value), .. }) if *value == v
            ),
            "the exit returns the phi"
        );
    }

    /// A read the new edge does not reach is left exactly as it was.
    #[test]
    fn a_read_the_edit_did_not_disturb_is_untouched() {
        let (mut ssa, early, late) = straight_line();
        let before = ssa
            .block(2)
            .map(|block| format!("{:?}", block.instructions()));

        // No edge added, so nothing is cut off: the definition still reaches.
        let placed = ssa
            .reconstruct_values(&[ValueUpdate {
                value: late,
                definitions: vec![(1, late)],
            }])
            .expect("nothing to do");

        assert_eq!(placed, Reconstruction::default(), "no join, no rewrite");
        assert_eq!(
            ssa.block(2)
                .map(|block| format!("{:?}", block.instructions())),
            before,
            "the read still names the definition that reaches it"
        );
        assert_eq!(
            ssa.block(0)
                .and_then(|block| block.instructions().first())
                .map(|instr| instr.op().dest()),
            Some(Some(early)),
            "and nothing else moved"
        );
    }

    /// A join the value does not reach on every edge is refused, not invented.
    #[test]
    fn a_join_the_value_does_not_reach_is_refused() {
        let (mut ssa, early, late) = straight_line();
        if let Some(block) = ssa.block_mut(0)
            && let Some(terminator) = block.instructions_mut().last_mut()
        {
            terminator.set_op(SsaOp::Branch {
                condition: early,
                true_target: 1,
                false_target: 2,
            });
        }

        // `b0` is named as carrying nothing, so the join has an edge with no
        // value: the reconstruction must say so rather than pick one.
        let refused = ssa.reconstruct_values(&[ValueUpdate {
            value: late,
            definitions: vec![(1, late)],
        }]);
        assert!(
            refused.is_err(),
            "an edge with no value is a refusal, not a guess"
        );
    }

    /// B0: `outer = 7`; B1 (try): branch; B2 (try): `inner = 9`, then back to
    /// B2 along an edge an edit just added; B3: return; B4 handler.
    ///
    /// The handler sees `outer` for a throw in B2 before `inner` is assigned.
    /// Along the new edge that throw comes on a later round, when the local
    /// already holds `inner`: what it sees is what B2 is entered with, a phi
    /// at B2 - not what B2 ends with, which the throw came before.
    #[test]
    fn a_throw_part_way_through_a_block_reads_what_the_block_was_entered_with() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 1);
        let condition = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let outer = ssa.create_variable(
            VariableOrigin::Local(0),
            1,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let inner = ssa.create_variable(
            VariableOrigin::Local(0),
            2,
            DefSite::instruction(2, 0),
            MockType::I32,
        );
        let seen = ssa.create_variable(VariableOrigin::Local(0), 3, DefSite::phi(4), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: outer,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: inner,
            value: ConstValue::I32(9),
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);
        let mut b4 = SsaBlock::new(4);
        let mut phi = PhiNode::new(seen, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::exceptional(outer, 1));
        phi.add_operand(PhiOperand::exceptional(outer, 2));
        phi.add_operand(PhiOperand::exceptional(inner, 2));
        b4.add_phi(phi);
        b4.add_instruction(instr(SsaOp::returns(Some(seen))));
        ssa.add_block(b4);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, 3),
            handler_range: BlockRange::new(4, 5),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let placed = ssa.reconstruct_values(&[ValueUpdate {
            value: outer,
            definitions: vec![(2, inner)],
        }]);
        assert_eq!(
            placed.ok().map(|placed| placed.phis),
            Some(1),
            "one phi, at B2's entry:\n{ssa:?}"
        );

        let entered = ssa
            .block(2)
            .and_then(|block| block.phi_nodes().first())
            .expect("B2 merges what it is entered with");
        assert_eq!(entered.operand_from(1).map(PhiOperand::value), Some(outer));
        assert_eq!(entered.operand_from(2).map(PhiOperand::value), Some(inner));

        let handler = ssa
            .block(4)
            .and_then(|block| block.phi_nodes().first())
            .expect("the handler keeps its phi");
        let from_b2: Vec<SsaVarId> = handler
            .exceptional_operands()
            .filter(|operand| operand.predecessor() == 2)
            .map(PhiOperand::value)
            .collect();
        assert_eq!(
            from_b2,
            vec![entered.result(), inner],
            "a throw before the assignment sees the entry merge, one after it the assignment"
        );
    }
}
