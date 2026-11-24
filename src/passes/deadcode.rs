//! Dead code elimination passes - both per-method and interprocedural.
//!
//! # Per-method DCE ([`DeadCodeEliminationPass`])
//!
//! Performs the following steps each iteration
//! ([`DeadCodeEliminationPass::run_iteration`]) until convergence:
//!
//! 1. **Reachable blocks** - BFS from the entry block (plus exception
//!    handler roots) to determine which blocks are live.
//! 2. **Clear unreachable blocks** - instructions and phis in blocks not
//!    reachable from entry or any handler are removed.
//! 3. **Remove Nops** - `Nop` stack-simulation artefacts are stripped, which
//!    simplifies the CFG for subsequent block merging.
//! 4. **Prune phi operands** - remove phi operands from unreachable
//!    predecessors.
//! 5. **Simplify trivial phis** - phis where all operands are the same
//!    variable (or self-referential) are replaced with a `Copy` or removed.
//! 6. **Recompute reachability** after phi changes.
//! 7. **Liveness via reverse dataflow** - compute which variables are
//!    live at each point using an RPO traversal and a worklist.
//! 8. **Remove dead phis** - phis whose results are not live.
//! 9. **Remove dead pure definitions** - instructions with no side effects
//!    whose results are not live.
//! 10. **Clean up Nops** created by dead-def removal.
//!
//! Each mutating step runs through the checked SSA editor and performs
//! boundary repair before the next analysis step.
//!
//! # Global DCE ([`DeadMethodEliminationPass`])
//!
//! Interprocedural dead-method detection. Traverses the call graph from
//! [`World::entry_points`] and marks every method that is not transitively
//! reachable as dead. The host's [`World::callees`] decides how call edges
//! are resolved.
//!
//! # Shared utilities
//!
//! - [`find_dead_tails`] - identifies unreachable code after a terminator
//!   (shared with the control flow pass).

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, VecDeque},
};

use rustc_hash::FxHashSet;

use crate::{
    Result,
    analysis::{budget::Budget, cfg::SsaCfg, exceptions::FunctionRoots, phis::PhiAnalyzer},
    bitset::BitSet,
    events::{EventBuilder, EventKind, EventListener},
    graph::{NodeId, algorithms},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::SsaOp,
        phi::PhiNode,
        variable::{SsaVarId, VariableOrigin},
        varstore::{VarMap, VarSet},
    },
    scheduling::{FunctionPass, ModificationScope, ModulePass, PassEnv},
    target::Target,
    world::World,
};

/// Identify blocks with unreachable code after a terminator.
///
/// Scans all blocks for instructions that follow a terminator (return,
/// throw, etc.) within the same block. These are unreachable and should
/// be removed.
///
/// # Arguments
///
/// * `ssa` - The SSA function to scan.
///
/// # Returns
///
/// A vector of `(block_idx, first_dead_instr_idx)` pairs describing the
/// unreachable instruction spans.
#[must_use]
pub fn find_dead_tails<T: Target>(ssa: &SsaFunction<T>) -> Vec<(usize, usize)> {
    ssa.iter_blocks()
        .filter_map(|(block_idx, block)| {
            let last_idx = block.instruction_count().checked_sub(1)?;
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if instr.op().is_terminator() && instr_idx < last_idx {
                    return Some((block_idx, instr_idx.saturating_add(1)));
                }
            }
            None
        })
        .collect()
}

/// The values something the function does reads, as dead code elimination
/// decides them: an operand of an operation with an effect, a value returned
/// or thrown, and every value a live one is computed from.
///
/// A load is an effect of its own (it may fault), so its address is live, but
/// its result is live only where a live value reads it: a load whose result
/// is not live is a read the function makes and discards.
///
/// # Arguments
///
/// * `ssa` - The SSA function to scan.
///
/// # Returns
///
/// A [`VarSet`] of every live variable.
#[must_use]
pub fn live_values<T: Target>(ssa: &SsaFunction<T>) -> VarSet {
    let cfg = SsaCfg::from_ssa(ssa);
    let reachable = FunctionRoots::of(ssa).live_blocks(&cfg);
    let rpo = compute_reverse_postorder(ssa, &reachable, &cfg);
    compute_live_variables(ssa, &reachable, &rpo)
}

/// Order the reachable blocks in reverse postorder for the liveness sweep.
///
/// Liveness is a backward dataflow, so it converges fastest when blocks are
/// visited in reverse postorder. RPO from the entry block covers the blocks
/// reachable through normal control flow; exception roots (handlers reached
/// only by the runtime) are not on any edge from entry, so each uncovered
/// reachable block seeds its own RPO traversal, appended in block-index order
/// for determinism. Membership is tracked in an incremental [`BitSet`] rather
/// than a linear scan of the growing list, keeping the handler sweep linear
/// instead of quadratic in the block count.
///
/// # Arguments
///
/// * `ssa` - The SSA function being analyzed.
/// * `reachable` - The set of reachable blocks to include.
/// * `cfg` - A CFG already built from `ssa`.
///
/// # Returns
///
/// The reachable block indices in reverse postorder.
fn compute_reverse_postorder<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    cfg: &SsaCfg<T>,
) -> Vec<usize> {
    if ssa.block_count() == 0 || reachable.is_empty() {
        return Vec::new();
    }

    let mut rpo: Vec<usize> = algorithms::reverse_postorder(cfg, NodeId::new(0))
        .into_iter()
        .map(|n: NodeId| n.index())
        .filter(|idx| reachable.contains(*idx))
        .collect();

    let mut in_rpo = BitSet::new(ssa.block_count());
    for &idx in &rpo {
        in_rpo.insert(idx);
    }
    let mut additional: Vec<usize> = reachable
        .iter()
        .filter(|idx| !in_rpo.contains(*idx))
        .collect();
    additional.sort_unstable();

    for &root in &additional {
        // Maintain `in_rpo` incrementally so membership is O(1); a linear
        // `rpo.contains` here would make the handler sweep O(blocks^2).
        for node in algorithms::reverse_postorder(cfg, NodeId::new(root)) {
            let idx = node.index();
            if reachable.contains(idx) && !in_rpo.contains(idx) {
                in_rpo.insert(idx);
                rpo.push(idx);
            }
        }
    }

    rpo
}

/// Compute the set of live variables by reverse dataflow.
///
/// Runs in three phases. Phase 1 seeds a worklist with the roots of liveness:
/// operands of side-effectful (non-pure) ops, and the values carried by
/// `Return` and `Throw` - anything observable outside the function. Phase 2
/// builds a def→uses map (from both phi operands and instruction uses) and
/// propagates liveness backward until the worklist drains, so a value is live
/// whenever a live value uses it. Phase 3 bridges local/argument slots: a live
/// `LoadLocal`/`LoadLocalAddr`/`LoadArg`/`LoadArgAddr` keeps alive every
/// definition sharing
/// its [`VariableOrigin`], and because that can make further loads live it
/// loops to a fixpoint. The def→uses map is a dense [`VarMap`] rather than a
/// tree map because variable ids are dense and it is probed once per worklist
/// pop; instruction operands are gathered once per instruction, not once per
/// definition, so a multi-output op does not re-walk them.
///
/// # Arguments
///
/// * `ssa` - The SSA function being analyzed.
/// * `reachable` - The set of reachable blocks; unreachable blocks contribute
///   no liveness.
/// * `rpo` - Reachable blocks in reverse postorder.
///
/// # Returns
///
/// A [`VarSet`] of every live variable.
fn compute_live_variables<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    rpo: &[usize],
) -> VarSet {
    let mut live = VarSet::new(ssa.var_id_bound());
    let mut worklist = VecDeque::new();

    // Phase 1: roots - operands of side-effectful ops, return values, throws.
    for &block_idx in rpo {
        if !reachable.contains(block_idx) {
            continue;
        }
        if let Some(block) = ssa.block(block_idx) {
            for instr in block.instructions() {
                let op = instr.op();
                if !op.effects().is_pure() {
                    op.for_each_use(|var| {
                        if live.insert(var) {
                            worklist.push_back(var);
                        }
                    });
                }
                if let SsaOp::Return { value: Some(v), .. } = op
                    && live.insert(*v)
                {
                    worklist.push_back(*v);
                }
                if let SsaOp::Throw { exception } = op
                    && live.insert(*exception)
                {
                    worklist.push_back(*exception);
                }
            }
        }
    }

    // Phase 2: backward propagation.
    // Dense, indexed by variable id: this is rebuilt on every DCE iteration and
    // probed once per worklist pop, so a `BTreeMap` paid an O(log n) comparison
    // chain on every access for no benefit - variable ids are dense by
    // construction.
    let mut def_uses: VarMap<Vec<SsaVarId>> = VarMap::new(ssa.var_id_bound());

    let mut origin_defs: BTreeMap<VariableOrigin, Vec<SsaVarId>> = BTreeMap::new();
    let mut load_local_info: Vec<(SsaVarId, VariableOrigin)> = Vec::new();
    let mut use_buf: Vec<SsaVarId> = Vec::new();

    for &block_idx in rpo {
        if !reachable.contains(block_idx) {
            continue;
        }
        if let Some(block) = ssa.block(block_idx) {
            for phi in block.phi_nodes() {
                let def = phi.result();
                for operand in phi.operands() {
                    def_uses.or_insert(def, Vec::new());
                    if let Some(slot) = def_uses.get_mut(def) {
                        slot.push(operand.value());
                    }
                }
                let origin = phi.origin();
                if matches!(
                    origin,
                    VariableOrigin::Local(_) | VariableOrigin::Argument(_)
                ) {
                    origin_defs.entry(origin).or_default().push(def);
                }
            }

            for instr in block.instructions() {
                let op = instr.op();
                let defs: Vec<SsaVarId> = op.defs().collect();
                // Walked once per instruction, not once per definition: an op
                // with a `flags` output has two defs, and re-walking its operands
                // for each of them is pure duplication.
                use_buf.clear();
                op.for_each_use(|use_var| use_buf.push(use_var));
                for &def in &defs {
                    def_uses.or_insert(def, Vec::new());
                    if let Some(slot) = def_uses.get_mut(def) {
                        slot.extend_from_slice(&use_buf);
                    }
                    if let Some(var) = ssa.variable(def) {
                        let origin = var.origin();
                        if matches!(
                            origin,
                            VariableOrigin::Local(_) | VariableOrigin::Argument(_)
                        ) {
                            origin_defs.entry(origin).or_default().push(def);
                        }
                    }
                }

                match op {
                    SsaOp::LoadLocal { dest, local_index } => {
                        load_local_info.push((*dest, VariableOrigin::Local(*local_index)));
                    }
                    SsaOp::LoadLocalAddr { dest, local_index } => {
                        load_local_info.push((*dest, VariableOrigin::Local(*local_index)));
                    }
                    SsaOp::LoadArg { dest, arg_index } | SsaOp::LoadArgAddr { dest, arg_index } => {
                        load_local_info.push((*dest, VariableOrigin::Argument(*arg_index)));
                    }
                    _ => {}
                }
            }
        }
    }

    while let Some(var) = worklist.pop_front() {
        if let Some(uses) = def_uses.get(var) {
            for &use_var in uses {
                if live.insert(use_var) {
                    worklist.push_back(use_var);
                }
            }
        }
    }

    // Phase 3: bridge LoadLocal/LoadArg → defs of corresponding origin. Loop
    // because re-propagation can keep newly-live LoadLocals alive (e.g. a
    // Copy of a LoadLocal's dest going live transitively).
    loop {
        let mut newly_live = false;
        for (dest, origin) in &load_local_info {
            if !live.contains(*dest) {
                continue;
            }
            if let Some(defs) = origin_defs.get(origin) {
                for &def_var in defs {
                    if live.insert(def_var) {
                        worklist.push_back(def_var);
                        newly_live = true;
                    }
                }
            }
        }
        while let Some(var) = worklist.pop_front() {
            if let Some(uses) = def_uses.get(var) {
                for &use_var in uses {
                    if live.insert(use_var) {
                        worklist.push_back(use_var);
                    }
                }
            }
        }
        if !newly_live {
            break;
        }
    }

    live
}

/// Locate pure instructions whose results are all dead.
///
/// Scans reachable blocks for instructions whose effects report
/// `removable_when_unused`, i.e. may be deleted when their results are dead. An
/// instruction with no defs is dead
/// outright; one with defs is dead only if *every* def is absent from `live`.
/// `Pop` is handled separately: a `Pop` is dead not when its own result is
/// unused but when the value it pops is being removed this iteration - either
/// an instruction def collected into `dead_vars` during the first scan, or a
/// dead phi result passed in via `dead_phi_results` - so a second scan catches
/// those. Keeping a stack-balancing `Pop` whose source vanished would leave the
/// operand stack unbalanced.
///
/// # Arguments
///
/// * `ssa` - The SSA function being analyzed.
/// * `reachable` - The set of reachable blocks to scan.
/// * `live` - The live-variable set from [`compute_live_variables`].
/// * `dead_phi_results` - Results of phis being removed this iteration, so a
///   `Pop` of one is recognised as dead.
///
/// # Returns
///
/// `(block_idx, instr_idx)` pairs identifying the dead instructions.
fn find_dead_definitions<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    live: &VarSet,
    dead_phi_results: &VarSet,
) -> Vec<(usize, usize)> {
    let mut dead_vars = VarSet::new(ssa.var_id_bound());
    let mut dead = Vec::new();

    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let op = instr.op();
                if !op.effects().removable_when_unused() {
                    continue;
                }
                if matches!(op, SsaOp::Pop { .. }) {
                    continue;
                }
                let defs: Vec<SsaVarId> = op.defs().collect();
                if defs.is_empty() {
                    dead.push((block_idx, instr_idx));
                } else if defs.iter().all(|def| !live.contains(*def)) {
                    dead.push((block_idx, instr_idx));
                    for def in defs {
                        dead_vars.insert(def);
                    }
                }
            }
        }
    }

    // Dead Pop: operand's definer is being removed in this iteration.
    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if let SsaOp::Pop { value } = instr.op() {
                    let instr_definer_dead = dead_vars.contains(*value);
                    let phi_definer_dead = dead_phi_results.contains(*value);
                    if instr_definer_dead || phi_definer_dead {
                        dead.push((block_idx, instr_idx));
                    }
                }
            }
        }
    }

    dead
}

/// Locate phi nodes whose results are dead.
///
/// A phi is dead when its result variable is not in `live`. Only reachable
/// blocks are scanned, since a phi in an unreachable block is removed by
/// [`clear_unreachable_blocks`] instead. The results feed both
/// [`remove_dead_phis_and_definitions`] and the `dead_phi_results` set that
/// [`find_dead_definitions`] uses to spot now-dead `Pop`s.
///
/// # Arguments
///
/// * `ssa` - The SSA function being analyzed.
/// * `reachable` - The set of reachable blocks to scan.
/// * `live` - The live-variable set from [`compute_live_variables`].
///
/// # Returns
///
/// `(block_idx, phi_idx)` pairs identifying the dead phi nodes.
fn find_dead_phis<T: Target>(
    ssa: &SsaFunction<T>,
    reachable: &BitSet,
    live: &VarSet,
) -> Vec<(usize, usize)> {
    let mut dead = Vec::new();
    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                if !live.contains(phi.result()) {
                    dead.push((block_idx, phi_idx));
                }
            }
        }
    }
    dead
}

/// Removes dead phi nodes and dead pure definitions in a **single** edit session.
///
/// Both sets are computed from one liveness snapshot before any mutation. They
/// must also be *applied* together: an edit session's boundary repair renumbers
/// every [`SsaVarId`] (`repair_ssa` -> `compact_variables`), so splitting the two
/// removals across two sessions would leave the caller's var-id-keyed liveness
/// bitset describing the pre-renumber function while the ids it is queried with
/// come from the post-renumber one - silently deleting live definitions.
///
/// Phi indices and instruction indices live in separate per-block vectors, so the
/// two removals cannot disturb each other's indices; each is applied
/// highest-index-first so earlier removals do not shift later ones.
///
/// # Arguments
///
/// * `ssa` - The SSA function to mutate in place.
/// * `dead_phis` - `(block_idx, phi_idx)` pairs from [`find_dead_phis`].
/// * `dead_defs` - `(block_idx, instr_idx)` pairs from [`find_dead_definitions`].
/// * `env` - The pass environment: where events are recorded, naming the method.
fn remove_dead_phis_and_definitions<T>(
    ssa: &mut SsaFunction<T>,
    dead_phis: &[(usize, usize)],
    dead_defs: &[(usize, usize)],
    env: &PassEnv<'_, T>,
) where
    T: Target,
{
    let mut instructions_by_block: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(block_idx, instr_idx) in dead_defs {
        instructions_by_block
            .entry(block_idx)
            .or_default()
            .push(instr_idx);
    }
    let mut phis_by_block: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(block_idx, phi_idx) in dead_phis {
        phis_by_block.entry(block_idx).or_default().push(phi_idx);
    }

    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, mut indices) in instructions_by_block {
            indices.sort_by(|a, b| b.cmp(a));
            for instr_idx in indices {
                let Some(instr) = editor
                    .function()
                    .block(block_idx)
                    .and_then(|block| block.instructions().get(instr_idx))
                else {
                    continue;
                };

                let defs: Vec<SsaVarId> = instr.op().defs().collect();
                let message = if defs.is_empty() {
                    format!("dead {}", instr.mnemonic())
                } else {
                    format!("dead definition(s) {defs:?}")
                };

                editor.nop_instruction(block_idx, instr_idx)?;
                let location = block_idx.saturating_mul(1000).saturating_add(instr_idx);
                env.record(EventKind::InstructionRemoved)
                    .location(location)
                    .message(message);
            }
        }
        for (block_idx, mut indices) in phis_by_block {
            indices.sort_by(|a, b| b.cmp(a));
            for phi_idx in indices {
                if editor.remove_phi(block_idx, phi_idx).is_ok() {
                    env.record(EventKind::PhiSimplified)
                        .location(block_idx)
                        .message("removed dead phi node".to_string());
                }
            }
        }
        Ok(())
    });
}

/// Gives each phi operand whose value is undefined ([`SsaOp::defines_undefined`])
/// the value of a sibling operand available at the end of its edge.
///
/// The path that brings an undefined value into a phi relies on nothing it
/// holds, so the operand may be any value the edge can carry: one whose
/// definition dominates the end of the edge's predecessor, or that the
/// function is entered with. Of several such siblings the lowest value id
/// is taken, whatever order the phi lists them in. A phi whose defined
/// operands then agree is trivial, and the copies its edges would carry
/// agree too. Only the phi operand changes: every other read of the value
/// keeps it.
///
/// # Arguments
///
/// * `ssa` - The SSA function to mutate in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// The number of operands given a sibling's value.
fn fill_undefined_phi_operands<T>(ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> usize
where
    T: Target,
{
    let undefined: BTreeSet<SsaVarId> = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter(|instruction| instruction.op().defines_undefined())
        .flat_map(|instruction| instruction.op().defs())
        .collect();
    if undefined.is_empty() || ssa.block_count() == 0 {
        return 0;
    }
    let dominators = algorithms::compute_dominators(&SsaCfg::from_ssa(ssa), NodeId::new(0));
    let available = |value: SsaVarId, at: usize| {
        ssa.variable(value).is_some_and(|variable| {
            dominators.dominates(NodeId::new(variable.def_site().block()), NodeId::new(at))
        })
    };
    // (block, phi, edge, value) for every operand to fill.
    let mut fills: Vec<(usize, usize, usize, SsaVarId)> = Vec::new();
    for block in ssa.blocks() {
        for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
            let defined: Vec<SsaVarId> = phi
                .normal_operands()
                .map(|operand| operand.value())
                .filter(|value| !undefined.contains(value) && *value != phi.result())
                .collect();
            for operand in phi.normal_operands() {
                if !undefined.contains(&operand.value()) {
                    continue;
                }
                let edge = operand.predecessor();
                if let Some(value) = defined
                    .iter()
                    .copied()
                    .filter(|value| available(*value, edge))
                    .min()
                {
                    fills.push((block.id(), phi_idx, edge, value));
                }
            }
        }
    }
    if fills.is_empty() {
        return 0;
    }
    let mut filled = 0usize;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        for &(block_idx, phi_idx, edge, value) in &fills {
            editor.set_phi_operand(block_idx, phi_idx, edge, value)?;
            filled = filled.saturating_add(1);
        }
        Ok(())
    });
    if result.is_err() {
        return 0;
    }
    for &(block_idx, _, edge, value) in &fills {
        env.record(EventKind::PhiSimplified)
            .location(block_idx)
            .message(format!("undefined operand from B{edge} given {value}"));
    }
    filled
}

/// Replace or remove trivial phi nodes.
///
/// A phi identified as trivial by the [`PhiAnalyzer`]
/// carries an optional replacement variable: `Some(v)` means every distinct
/// operand resolves to `v`, so the phi becomes a `Copy` from `v`; `None` means
/// the phi is purely self-referential and is deleted outright. Removals within
/// a block are applied highest-index-first so earlier edits do not shift the
/// indices of later ones, and the whole batch runs in one checked edit session.
///
/// # Arguments
///
/// * `ssa` - The SSA function to mutate in place.
/// * `trivial_phis` - `(block_idx, phi_idx, replacement)` triples, where a
///   `None` replacement marks a self-referential phi to delete.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// The number of phis simplified or removed.
fn simplify_trivial_phis<T>(
    ssa: &mut SsaFunction<T>,
    trivial_phis: &[(usize, usize, Option<SsaVarId>)],
    env: &PassEnv<'_, T>,
) -> usize
where
    T: Target,
{
    let mut simplified: usize = 0;

    // A replacement can be the result of another phi this batch removes: a
    // phi trivial to a phi that is trivial in turn. Each is followed through
    // the batch to the value the chain ends at, so no use is rewritten to a
    // result the batch then removes. A chain that ends at a phi that names
    // only itself ends at no value, and its phi waits for the next round.
    let removed: BTreeMap<SsaVarId, Option<SsaVarId>> = trivial_phis
        .iter()
        .filter_map(|&(block_idx, phi_idx, replacement)| {
            let result = ssa.block(block_idx)?.phi_nodes().get(phi_idx)?.result();
            Some((result, replacement))
        })
        .collect();
    let resolve = |start: SsaVarId| -> Option<SsaVarId> {
        let mut value = start;
        let mut seen = BTreeSet::new();
        while let Some(&next) = removed.get(&value) {
            if !seen.insert(value) {
                return None;
            }
            value = next?;
        }
        Some(value)
    };

    let mut by_block: BTreeMap<usize, Vec<(usize, Option<SsaVarId>)>> = BTreeMap::new();
    for &(block_idx, phi_idx, replacement) in trivial_phis {
        let replacement = match replacement {
            Some(value) => match resolve(value) {
                Some(resolved) => Some(resolved),
                None => continue,
            },
            None => None,
        };
        by_block
            .entry(block_idx)
            .or_default()
            .push((phi_idx, replacement));
    }

    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, mut phis) in by_block {
            phis.sort_by_key(|p| Reverse(p.0));
            for (phi_idx, replacement) in phis {
                if let Some(replacement_var) = replacement {
                    if editor
                        .simplify_phi_to_copy(block_idx, phi_idx, replacement_var)
                        .is_ok()
                    {
                        env.record(EventKind::PhiSimplified)
                            .location(block_idx)
                            .message(format!("replaced with {replacement_var}"));
                        simplified = simplified.saturating_add(1);
                    }
                } else if editor.remove_phi(block_idx, phi_idx).is_ok() {
                    env.record(EventKind::PhiSimplified)
                        .location(block_idx)
                        .message("removed self-referential phi".to_string());
                    simplified = simplified.saturating_add(1);
                }
            }
        }
        Ok(())
    });

    simplified
}

/// Clear the contents of every block not in the reachable set.
///
/// Walks all block indices and, for each one absent from `reachable`, empties
/// its instructions and phis via `clear_block`. The block slot itself is kept
/// (block indices remain stable for the rest of the iteration); only its
/// contents go, which severs its outgoing edges and lets subsequent
/// reachability sweeps see the reduced CFG.
///
/// # Arguments
///
/// * `ssa` - The SSA function to mutate in place.
/// * `reachable` - The set of reachable blocks; every other block is cleared.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// The number of blocks cleared.
fn clear_unreachable_blocks<T>(
    ssa: &mut SsaFunction<T>,
    reachable: &BitSet,
    env: &PassEnv<'_, T>,
) -> usize
where
    T: Target,
{
    let mut cleared: Vec<usize> = Vec::new();
    // A failure is recorded on the function for the pass's runner (see
    // `SsaFunction::take_edit_failure`).
    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        cleared = editor.clear_unreached_blocks(reachable);
        Ok(())
    });
    for &block_idx in &cleared {
        env.record(EventKind::BlockRemoved)
            .location(block_idx)
            .message(format!("removed unreachable block {block_idx}"));
    }
    cleared.len()
}

/// Remove every `Nop` in the reachable blocks.
///
/// Front ends emit `Nop`s as stack-simulation artefacts, and removing them
/// flattens the CFG for the subsequent block-merge pass. Every `Nop` goes,
/// including one that is the sole instruction of its block: the session's
/// boundary repair strips all `Nop`s regardless. All removals share one edit
/// session, and events are coalesced into one per block reporting the count,
/// rather than one per `Nop`.
///
/// # Arguments
///
/// * `ssa` - The SSA function to mutate in place.
/// * `reachable` - The set of reachable blocks to scan.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// The number of `Nop` instructions removed.
fn remove_nop_instructions<T>(
    ssa: &mut SsaFunction<T>,
    reachable: &BitSet,
    env: &PassEnv<'_, T>,
) -> usize
where
    T: Target,
{
    let mut nops = Vec::new();
    for block_idx in reachable.iter() {
        if let Some(block) = ssa.block(block_idx) {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if matches!(instr.op(), SsaOp::Nop) {
                    nops.push((block_idx, instr_idx));
                }
            }
        }
    }

    if nops.is_empty() {
        return 0;
    }

    let mut removed = 0usize;
    let _ = ssa.edit(SsaEditOptions::new(), |editor| {
        for (block_idx, instr_idx) in &nops {
            if editor.nop_instruction(*block_idx, *instr_idx).is_ok() {
                removed = removed.saturating_add(1);
            }
        }
        Ok(())
    });

    if removed > 0 {
        let mut per_block: BTreeMap<usize, usize> = BTreeMap::new();
        for (block_idx, _) in nops {
            let entry = per_block.entry(block_idx).or_insert(0);
            *entry = entry.saturating_add(1);
        }
        for (block_idx, count) in per_block {
            env.record(EventKind::InstructionRemoved)
                .location(block_idx)
                .message(format!("removed {count} Nop instructions"));
        }
    }

    removed
}

/// Pass that removes unreachable blocks, unused definitions, and dead phi
/// nodes within a single method.
///
/// Complements [`DeadMethodEliminationPass`], which handles interprocedural
/// DCE.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - removes instructions and phis, and
/// clears unreachable blocks, which removes their outgoing edges. Every edit
/// runs in a checked session that repairs the function at its boundary, so
/// the pass [repairs its own SSA](FunctionPass::repairs_ssa).
#[derive(Debug, Clone, Copy, Default)]
pub struct DeadCodeEliminationPass;

impl<T> FunctionPass<T> for DeadCodeEliminationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "dead-code-elimination"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Remove unreachable blocks, unused definitions, and op-less instructions"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Runs intraprocedural dead-code elimination on one method: removes
    /// unreachable blocks, unused definitions, and dead phi nodes, one
    /// [`Self::run_iteration`] at a time until a round changes nothing or the
    /// context's [`Budget::DeadCodeRounds`] runs out.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to clean up in place.
    /// * `env` - The pass environment: where events are recorded, naming the method.
    ///
    /// # Returns
    ///
    /// `true` if anything was removed.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let mut changed = false;
        let mut rounds = env.context().budget().rounds(Budget::DeadCodeRounds);
        while rounds.next_round() {
            if self.run_iteration(ssa, env) == 0 {
                rounds.settle();
            } else {
                changed = true;
            }
        }
        if let Some(exhaustion) = rounds.exhausted() {
            env.exhausted(exhaustion);
        }
        Ok(changed)
    }
}

impl DeadCodeEliminationPass {
    /// Single iteration of per-method DCE.
    ///
    /// Performs reachability analysis, removes unreachable blocks, strips
    /// nops, prunes phi operands, simplifies trivial
    /// phis, computes liveness, and removes dead phis and definitions.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to clean up in place.
    /// * `env` - The pass environment: where events are recorded, naming the method.
    ///
    /// # Returns
    ///
    /// The total number of changes made this iteration. Zero means the
    /// algorithm has converged.
    pub fn run_iteration<T: Target>(
        &self,
        ssa: &mut SsaFunction<T>,
        env: &PassEnv<'_, T>,
    ) -> usize {
        let mut total_changes: usize = 0;

        // Step 1: reachable blocks - the entry's reach, then every handler, filter
        // and preservation root's.
        //
        // A stale index is *tolerated here and reported elsewhere*, and the two are
        // deliberately different questions. Passes read the exception table through
        // the total accessors (`parts`, `entry_blocks`, `FunctionRoots`), none of
        // which rejects anything, so an out-of-range clause costs a root rather than
        // an abort. Whether the IR is valid is the verifier's question, and
        // `SsaVerifier::check_exception_table` answers it by rejecting that same
        // clause. Pass robustness and IR validity must not be one condition: a pass
        // that refused to run on a table the verifier dislikes could not be run to
        // fix it.
        let reachable = FunctionRoots::of(ssa).live_blocks(&SsaCfg::from_ssa(ssa));

        // Step 2: clear unreachable blocks.
        total_changes =
            total_changes.saturating_add(clear_unreachable_blocks(ssa, &reachable, env));

        // Step 3: remove Nop instructions (stack-simulation artifacts; removing
        // them simplifies the CFG for block merging).
        total_changes = total_changes.saturating_add(remove_nop_instructions(ssa, &reachable, env));

        // Step 4: prune phi operands from unreachable predecessors.
        let mut pruned = 0usize;
        let _ = ssa.edit(SsaEditOptions::new(), |editor| {
            pruned = editor.prune_phi_operands(&reachable);
            Ok(())
        });
        total_changes = total_changes.saturating_add(pruned);
        let reachable_set: BTreeSet<usize> = reachable.iter().collect();

        // Step 4b: an undefined phi operand takes a sibling's value, which can
        // leave the phi trivial for step 5.
        total_changes = total_changes.saturating_add(fill_undefined_phi_operands(ssa, env));

        // Step 5: simplify trivial phis (purely structural).
        let trivial_phis = PhiAnalyzer::new(ssa).find_all_trivial(&reachable_set);
        total_changes =
            total_changes.saturating_add(simplify_trivial_phis(ssa, &trivial_phis, env));

        // Steps 6+7: recompute reachability after phi simplification, then compute
        // liveness. One CFG built here serves both the reachability sweep and the
        // RPO traversal.
        //
        // This CFG must be rebuilt rather than reusing step 1's: step 2 clears
        // unreachable blocks, and `SsaBlock::clear` removes their instructions -
        // terminators included - so their outgoing edges disappear. Forward
        // traversals from entry happen not to observe the difference, since the
        // cleared blocks were already unreachable, but the CFG object itself is
        // genuinely stale and reusing it would rest on that coincidence rather than
        // on an invariant.
        let cfg = SsaCfg::from_ssa(ssa);
        let reachable = FunctionRoots::of(ssa).live_blocks(&cfg);
        let rpo = compute_reverse_postorder(ssa, &reachable, &cfg);
        let live = compute_live_variables(ssa, &reachable, &rpo);

        // Steps 8+9: dead phis and dead pure definitions.
        //
        // Both sets are derived from `live`, which - like `dead_phi_results` - is a
        // `BitSet` indexed by `SsaVarId`. Closing an edit session runs the boundary
        // repair (`repair_ssa` -> `compact_variables`), which **renumbers every
        // `SsaVarId`**, so no var-id-keyed analysis result survives a session
        // boundary. Both removals therefore share ONE session, applied only after
        // every such result has been consumed: removing the phis in their own session
        // and *then* consulting `live` for dead definitions would index the old
        // numbering with new ids and delete definitions that are still live.
        let dead_phis = find_dead_phis(ssa, &reachable, &live);
        let mut dead_phi_results = VarSet::new(ssa.var_id_bound());
        for &(block_idx, phi_idx) in &dead_phis {
            if let Some(result) = ssa
                .block(block_idx)
                .and_then(|b| b.phi_nodes().get(phi_idx))
                .map(PhiNode::result)
            {
                dead_phi_results.insert(result);
            }
        }
        let dead_defs = find_dead_definitions(ssa, &reachable, &live, &dead_phi_results);
        total_changes = total_changes
            .saturating_add(dead_phis.len())
            .saturating_add(dead_defs.len());
        remove_dead_phis_and_definitions(ssa, &dead_phis, &dead_defs, env);

        total_changes
    }
}

/// Global interprocedural dead-method elimination pass.
///
/// Operates over the whole program via [`World<T>`](World), not per-method.
///
/// This pass is declared as `is_global()` and never invoked per-method.
/// It marks all methods not transitively reachable from entry points as
/// dead using [`World::mark_dead`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DeadMethodEliminationPass;

impl<T> ModulePass<T> for DeadMethodEliminationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "dead-method-elimination"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Mark methods that have no live callers and aren't entry points"
    }

    /// Runs interprocedural dead-method elimination over the whole program.
    ///
    /// Walks the [`World`] from its entry points along its call edges and
    /// marks every method not transitively reachable as dead
    /// ([`World::mark_dead`]).
    ///
    /// # Arguments
    ///
    /// * `world` - The program's methods, entry points and call edges.
    /// * `env` - The pass environment: where events are recorded, naming the method.
    ///
    /// # Returns
    ///
    /// `true` if any method was newly marked dead.
    fn run(&self, world: &dyn World<T>, events: &dyn EventListener<T>) -> Result<bool> {
        let entries = world.entry_points();
        let mut live: FxHashSet<T::MethodRef> = entries.iter().cloned().collect();
        let mut worklist: VecDeque<T::MethodRef> = entries.into_iter().collect();

        while let Some(method) = worklist.pop_front() {
            for callee in world.callees(&method) {
                if live.insert(callee.clone()) {
                    worklist.push_back(callee);
                }
            }
        }

        let mut changed = false;
        for method in world.all_methods() {
            if !live.contains(&method) && !world.is_dead(&method) {
                world.mark_dead(&method);
                EventBuilder::new(events, EventKind::MethodMarkedDead)
                    .method(method.clone())
                    .message("method has no live callers");
                changed = true;
            }
        }
        Ok(changed)
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
            exception::{BlockRange, SsaExceptionHandler},
            function::SsaFunction,
            instruction::SsaInstruction,
            ops::SsaOp,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{
            MACHINE, MockTarget, MockType, MockWorld, assert_mock_valid_full,
            run_mock_malformed_cleanup_boundary, run_mock_pass_boundary,
        },
    };

    /// Builds a function with an in-function protected region: block 0 is the
    /// try body, block 1 is where it completes, block 2 is the handler.
    ///
    /// No terminator names block 2 - the runtime enters a handler - which is
    /// exactly why its survival is a question worth asking.
    fn function_with_an_in_function_handler() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        for (index, op) in [
            SsaOp::Jump { target: 1 },
            SsaOp::returns(None),
            SsaOp::returns(None),
        ]
        .into_iter()
        .enumerate()
        {
            let mut block: SsaBlock<MockTarget> = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(op));
            ssa.add_block(block);
        }
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 1,
            handler_offset: 2,
            handler_length: 1,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 1),
            handler_range: BlockRange::new(2, 3),
            filter_range: None,
            selector: None,
        }]);
        ssa
    }

    /// A handler block survives elimination even though nothing branches to it.
    ///
    /// This is the property the exception table buys, and it is bought *here* -
    /// by rooting handler entries from the table - not by the CFG view drawing
    /// a synthetic edge to them. Which matters, because that edge binds for
    /// few clauses in lifted native code: on Windows x64 a handler is a
    /// funclet, a separate function, so a per-function CFG cannot contain the
    /// block at all.
    #[test]
    fn an_in_function_handler_survives_elimination() {
        let mut ssa = function_with_an_in_function_handler();
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            DeadCodeEliminationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );

        assert!(
            ssa.block(2).is_some(),
            "the handler block was eliminated: nothing branches to it, so only \
             the exception table keeps it alive"
        );
        assert_mock_valid_full(&ssa, "handler survival");
    }

    /// **An unread value the implementation chooses is dead.** `v0 =
    /// runtime(unspecified)` that nothing reads is removed.
    #[test]
    fn an_unread_unspecified_value_is_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::RuntimeValue {
            dest: SsaVarId::from_index(0),
            kind: crate::ir::ops::RuntimeValueKind::Unspecified,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            DeadCodeEliminationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        let kept = ssa.blocks().iter().any(|block| {
            block
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.op(), SsaOp::RuntimeValue { .. }))
        });
        assert!(!kept, "the unread value survived");
    }

    /// **The state a return leaves its caller is live.** An accrual of
    /// status flags nothing in the function reads, `v2 = v0 | v1`, stays
    /// while the return names it as state its caller observes; the control,
    /// the same function returning without it, loses the accrual.
    #[test]
    fn the_state_a_return_leaves_its_caller_is_live() {
        let accrued = |state: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
            for index in 0..3 {
                ssa.create_variable(
                    VariableOrigin::Local(index),
                    0,
                    DefSite::instruction(0, usize::from(index)),
                    MockType::I32,
                );
            }
            let var = SsaVarId::from_index;
            let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
            for (dest, value) in [(0, 1), (1, 0x10)] {
                block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                    dest: var(dest),
                    value: ConstValue::I32(value),
                }));
            }
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Or {
                dest: var(2),
                left: var(0),
                right: var(1),
                flags: None,
            }));
            let state = if state { vec![var(2)] } else { Vec::new() };
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns_with(None, state)));
            ssa.add_block(block);
            let log: EventLog<MockTarget> = EventLog::new();
            assert!(
                DeadCodeEliminationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            ssa.blocks().iter().any(|block| {
                block
                    .instructions()
                    .iter()
                    .any(|instruction| matches!(instruction.op(), SsaOp::Or { .. }))
            })
        };
        assert!(accrued(true), "the accrual the return names went");
        assert!(!accrued(false), "an accrual nothing reads stayed");
    }

    /// A `Nop` following a `Return` is reported as a dead tail at index 1.
    #[test]
    fn find_dead_tails_after_terminator() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        ssa.add_block(block);

        let dead = find_dead_tails(&ssa);
        assert_eq!(dead, vec![(0, 1)]);
    }

    /// Global DCE marks a method unreachable from any entry point as dead.
    #[test]
    fn global_dce_marks_unreachable_methods() {
        // Methods 1, 2, 3, 4. Entry: 1. Edges: 1→2, 2→3. Method 4 is
        // unreachable.
        let world = MockWorld::new([1u32, 2, 3, 4], [1u32], [(1u32, 2), (2u32, 3)]);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = DeadMethodEliminationPass.run(&world, &log) == Ok(true);
        assert!(changed);
        assert!(world.is_dead(&4));
        assert!(!world.is_dead(&1));
        assert!(!world.is_dead(&2));
        assert!(!world.is_dead(&3));
        assert_eq!(log.count_kind(EventKind::MethodMarkedDead), 1);
    }

    /// Global DCE reports no change when every method is reachable.
    #[test]
    fn global_dce_no_op_when_all_reachable() {
        let world = MockWorld::new([1u32, 2], [1u32], [(1u32, 2)]);
        let log: EventLog<MockTarget> = EventLog::new();
        let changed = DeadMethodEliminationPass.run(&world, &log) == Ok(true);
        assert!(!changed);
    }

    /// Verifies `find_dead_tails` finds a single dead-tail span when one
    /// `Const` follows a `Return`. This builds the invalid block manually so
    /// the dead-tail scanner can inspect instructions after a terminator.
    #[test]
    fn find_dead_tails_with_dead_code() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let mut block0: SsaBlock<MockTarget> = SsaBlock::new(0);
        block0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        block0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: SsaVarId::from_index(0),
            value: ConstValue::I32(42),
        }));
        ssa.add_block(block0);

        let dead = find_dead_tails(&ssa);
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0], (0, 1));
    }

    /// Multiple `Const` instructions after a `Return` should still report a
    /// single dead-tail span starting at index 1.
    #[test]
    fn find_dead_tails_multiple_dead_instructions() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 0);
        let mut block0: SsaBlock<MockTarget> = SsaBlock::new(0);
        block0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        for (i, v) in [1, 2, 3].iter().enumerate() {
            block0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: SsaVarId::from_index(i),
                value: ConstValue::I32(*v),
            }));
        }
        ssa.add_block(block0);

        let dead = find_dead_tails(&ssa);
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0], (0, 1));
    }

    /// Exercises `SsaOp::successors` for every terminator shape DCE recognizes
    /// during reachability.
    #[test]
    fn successor_extraction() {
        let op: SsaOp<MockTarget> = SsaOp::Jump { target: 5 };
        assert_eq!(op.successors(), vec![5]);

        let cond = SsaVarId::from_index(0);
        let op: SsaOp<MockTarget> = SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        };
        assert_eq!(op.successors(), vec![1, 2]);

        let val = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Switch {
            value: val,
            targets: vec![1, 2, 3],
            default: 4,
        };
        assert_eq!(op.successors(), vec![1, 2, 3, 4]);

        let op: SsaOp<MockTarget> = SsaOp::returns(None);
        assert!(op.successors().is_empty());

        let op: SsaOp<MockTarget> = SsaOp::Leave { target: 3 };
        assert_eq!(op.successors(), vec![3]);
    }

    /// A function with no blocks should be a no-op for DCE.
    #[test]
    fn empty_function() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
    }

    /// A self-referential phi `v0 = phi(v0)` is trivial; DCE should remove it
    /// via `simplify_trivial_phis`.
    #[test]
    fn self_referential_phi_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let phi_var = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(0), MockType::I32);

        let mut block: SsaBlock<MockTarget> = SsaBlock::new(0);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(phi_var, 0)); // self-ref
        block.add_phi(phi);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_malformed_cleanup_boundary(&mut ssa, "self-referential phi DCE", |ssa| {
                DeadCodeEliminationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
        assert!(changed, "self-referential phi should be removed");
        // The trivial phi should have been removed.
        assert!(
            ssa.block(0)
                .expect("entry block should remain")
                .phi_nodes()
                .is_empty()
        );
    }

    /// A block with no predecessors is removed and a `BlockRemoved` event fires.
    #[test]
    fn removes_unreachable_block() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v0))));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(99),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v1))));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "unreachable-block DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "unreachable block should be removed");
        assert!(log.has(EventKind::BlockRemoved));
    }

    /// A pure instruction whose result is never used is removed.
    #[test]
    fn unused_pure_instruction_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        // v1 is computed but never used
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Neg {
            dest: v1,
            operand: v0,
            flags: None,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v0))));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "unused pure instruction DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "unused pure instruction should be removed");
        assert!(log.has(EventKind::InstructionRemoved));
    }

    /// A phi whose result is never used is removed.
    #[test]
    fn dead_phi_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let phi_var = SsaVarId::from_index(2);
        ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v0))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "dead phi DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        // phi_var is never used after the phi, so the phi should be removed
        assert!(changed);
        assert!(log.has(EventKind::PhiSimplified));
    }

    /// `Nop` instructions are stripped, leaving only the terminator.
    #[test]
    fn nop_instructions_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "nop DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "Nop instructions should be removed");
        assert!(log.has(EventKind::InstructionRemoved));
        // Only the Return should remain
        assert_eq!(
            ssa.block(0)
                .expect("entry block should remain")
                .instructions()
                .len(),
            1
        );
    }

    /// A `Nop` that is the only instruction of its block is stripped in the
    /// same iteration as any other `Nop`; nothing is left for a later pass.
    #[test]
    fn a_lone_nop_is_stripped() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            DeadCodeEliminationPass.run_iteration(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ) > 0
        );
        assert!(
            ssa.block(0)
                .is_some_and(|block| block.instructions().is_empty()),
            "the lone Nop is removed"
        );
    }

    /// An unused but side-effecting (overflow-checked) instruction is kept.
    #[test]
    fn side_effect_instruction_not_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let v2 = SsaVarId::from_index(2);
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(20),
        }));
        // Use an overflow-checked arithmetic op (impure - may throw)
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::AddOvf {
            dest: v2,
            left: v0,
            right: v1,
            unsigned: false,
            flags: None,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        // We don't expect specific changes; just verify the function stays valid
        let changed = run_mock_pass_boundary(&mut ssa, "side-effect DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "side-effecting unused instruction must remain");
    }

    /// A CFG whose blocks are all reachable and live is left untouched.
    #[test]
    fn all_blocks_reachable_nothing_removed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(2, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: v0,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v0))));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(2),
        }));
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v1))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "reachable-block DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        // v1 is used, both blocks reachable - no dead code
        assert!(!changed, "reachable live blocks should not be removed");
    }

    /// A single iteration reports zero changes on an already-clean function.
    #[test]
    fn run_iteration_returns_zero_on_no_work() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert_mock_valid_full(&ssa, "DCE iteration no-work before");
        let count = DeadCodeEliminationPass.run_iteration(
            &mut ssa,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
        );
        assert_mock_valid_full(&ssa, "DCE iteration no-work after");
        assert_eq!(count, 0);
    }

    /// Removing a dead definition compacts the variable numbering.
    #[test]
    fn compact_variables_after_dce() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(99),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v0))));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "variable compaction DCE", |ssa| {
            DeadCodeEliminationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed);
        // v1 should be gone after compaction
        assert!(log.has(EventKind::InstructionRemoved));
    }

    /// DCE must never delete a live definition, however the variables happen to
    /// be numbered.
    ///
    /// Liveness is a `BitSet` keyed by `SsaVarId`, and closing an edit session
    /// renumbers every `SsaVarId` (`repair_ssa` -> `compact_variables`). Applying
    /// the dead-phi removal in its own session and *then* consulting that bitset
    /// for dead definitions reads the **old** numbering with **new** ids, deleting
    /// definitions that are still live. Whether a given function trips the
    /// mismatch depends entirely on how its ids permute under compaction, so this
    /// sweeps the id space rather than asserting one hand-built collision.
    ///
    /// The check here is `Full` rather than the `Quick` a release pass group
    /// uses: `Quick` rejects a dangling read, but this sweep is about every
    /// definition DCE could delete, so it checks everything the verifier can.
    #[test]
    fn dce_never_deletes_a_live_definition() {
        // Deterministic xorshift; no `rand` dependency, and a failure is
        // reproducible from the printed seed.
        fn next(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }

        for seed in 1..2_000u64 {
            let mut rng = seed;
            // `slots` constants, all defined in the entry block so each dominates
            // the join, plus the phi result. Ids are permuted across ALL of them -
            // including the phi's - because the corruption needs the dead phi to
            // hold a *low* id: compaction only shifts the ids above the one it
            // drops, so a phi pinned to the highest id renumbers nothing.
            let slots = 4 + (next(&mut rng) % 4) as usize; // 4..=7
            let total = slots + 1;
            let mut perm: Vec<usize> = (0..total).collect();
            for i in (1..total).rev() {
                let j = (next(&mut rng) % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }
            let ids = &perm[..slots]; // const ids, by slot
            let phi_id = perm[slots]; // the dead phi's id

            // Variables must be created in id order, so invert the permutation:
            // `owner[id]` is the slot that owns `id`, or `None` for the phi.
            let mut owner: Vec<Option<usize>> = vec![None; total];
            for (slot, &id) in ids.iter().enumerate() {
                owner[id] = Some(slot);
            }

            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, total);
            for (id, slot) in owner.iter().enumerate() {
                match slot {
                    Some(slot) => ssa.create_variable(
                        VariableOrigin::Local(id as u16),
                        0,
                        DefSite::instruction(0, *slot),
                        MockType::I32,
                    ),
                    None => ssa.create_variable(
                        VariableOrigin::Local(id as u16),
                        3,
                        DefSite::phi(3),
                        MockType::I32,
                    ),
                };
            }
            let var = |slot: usize| SsaVarId::from_index(ids[slot]);

            let mut b0 = SsaBlock::new(0);
            for slot in 0..slots {
                b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                    dest: var(slot),
                    value: ConstValue::I32(slot as i32),
                }));
            }
            b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
                condition: var(0),
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);

            let mut b1 = SsaBlock::new(1);
            b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(b1);

            let mut b2 = SsaBlock::new(2);
            b2.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(b2);

            // A two-operand phi (a one-operand phi is trivial and gets simplified
            // away before the dead-phi step, so the renumbering never happens).
            // Its result is unused -> dead.
            let mut b3 = SsaBlock::new(3);
            let left = (next(&mut rng) % slots as u64) as usize;
            let mut right = (next(&mut rng) % slots as u64) as usize;
            if right == left {
                right = (left + 1) % slots;
            }
            let mut phi = PhiNode::new(
                SsaVarId::from_index(phi_id),
                VariableOrigin::Local(phi_id as u16),
            );
            phi.add_operand(PhiOperand::new(var(left), 1));
            phi.add_operand(PhiOperand::new(var(right), 2));
            b3.add_phi(phi);
            // Return one of the entry constants: that one is unambiguously LIVE,
            // and every other constant is dead.
            let returned_slot = (next(&mut rng) % slots as u64) as usize;
            b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(var(
                returned_slot,
            )))));
            ssa.add_block(b3);
            ssa.recompute_uses();

            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            assert!(
                DeadCodeEliminationPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );

            assert_mock_valid_full(&ssa, &format!("DCE corrupted SSA (seed {seed})"));

            // Validity alone is NOT enough to catch this. When the stale lookup
            // deletes the live definition, `compact_variables` drops the
            // now-undefined variable and `remap_var_ids_in_blocks` silently
            // *aliases* the dangling id onto whichever variable inherits that
            // number (`remap.get(&id).unwrap_or(id)`). The result verifies
            // cleanly and simply returns the WRONG constant. So assert the
            // semantics: the returned value must still be defined by the very
            // constant the function was built to return.
            let returned = ssa
                .blocks()
                .iter()
                .flat_map(SsaBlock::instructions)
                .find_map(|instr| match instr.op() {
                    SsaOp::Return { value, .. } => *value,
                    _ => None,
                })
                .unwrap_or_else(|| panic!("function still returns a value (seed {seed})"));
            let defining_value = ssa
                .blocks()
                .iter()
                .flat_map(SsaBlock::instructions)
                .find_map(|instr| match instr.op() {
                    SsaOp::Const { dest, value } if *dest == returned => Some(value.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    panic!("DCE deleted the live returned definition (seed {seed})")
                });
            assert_eq!(
                defining_value,
                ConstValue::I32(returned_slot as i32),
                "DCE left the function returning the wrong constant (seed {seed})"
            );
        }
    }

    /// A join `B2` of a branch at `B0`: `B1` brings in a register a call left
    /// undefined, and the other edge `kept`, a value defined in `B0` where
    /// `dominating`, in `B3`, which `B0` branches to instead of straight to
    /// `B2`, otherwise. Returns the function and the kept value.
    fn clobbered_join(dominating: bool) -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let kept_block = if dominating { 0 } else { 3 };
        let condition = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let kept = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(kept_block, 0),
            MockType::I32,
        );
        let clobbered = ssa.create_variable(
            VariableOrigin::Local(1),
            1,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let merged =
            ssa.create_variable(VariableOrigin::Local(1), 2, DefSite::phi(2), MockType::I32);
        let constant = |dest, value| {
            SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            })
        };
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(constant(condition, 1));
        if dominating {
            b0.add_instruction(constant(kept, 7));
        }
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: if dominating { 2 } else { 3 },
        }));
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::CallClobber {
            outputs: vec![clobbered],
            machine: None,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(kept, kept_block));
        phi.add_operand(PhiOperand::new(clobbered, 1));
        b2.add_phi(phi);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(merged))));
        ssa.add_block(b0);
        ssa.add_block(b1);
        ssa.add_block(b2);
        if !dominating {
            let mut b3 = SsaBlock::new(3);
            b3.add_instruction(constant(kept, 7));
            b3.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
            ssa.add_block(b3);
        }
        ssa.recompute_uses();
        (ssa, kept)
    }

    /// **An undefined phi operand takes a sibling's value available on its
    /// edge.** The register a call left undefined that `B1` brings into the
    /// join is `B0`'s value, which dominates `B1`, so the join is that value
    /// on both edges. A value defined on another path, in `B3`, is not
    /// available at the end of `B1`, and the operand keeps the clobber.
    #[test]
    fn an_undefined_phi_operand_takes_a_siblings_value_available_on_its_edge() {
        let log: EventLog<MockTarget> = EventLog::new();
        let (mut ssa, kept) = clobbered_join(true);
        assert_eq!(
            fill_undefined_phi_operands(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ),
            1
        );
        let operands: Vec<SsaVarId> = ssa
            .block(2)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| phi.operands().iter().map(PhiOperand::value).collect())
            .unwrap_or_default();
        assert_eq!(operands, vec![kept, kept]);

        let (mut ssa, _) = clobbered_join(false);
        assert_eq!(
            fill_undefined_phi_operands(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ),
            0
        );
    }

    /// **A phi trivial to a phi that is trivial in turn takes the value the
    /// chain ends at.** `B2` merges `v` from both its edges, and `B4` merges
    /// `B2`'s result from both of its: both are trivial in one round, `B4`'s
    /// to `B2`'s result. Rewritten to that result after `B2`'s phi was gone,
    /// the return read a value nothing defines.
    #[test]
    fn a_phi_trivial_to_a_trivial_phi_takes_the_value_the_chain_ends_at() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let condition = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let value = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let first =
            ssa.create_variable(VariableOrigin::Local(1), 1, DefSite::phi(2), MockType::I32);
        let second =
            ssa.create_variable(VariableOrigin::Local(1), 2, DefSite::phi(4), MockType::I32);
        let constant = |dest, value| {
            SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            })
        };
        let branch = |to, past| {
            SsaInstruction::synthetic(SsaOp::Branch {
                condition,
                true_target: to,
                false_target: past,
            })
        };
        let jump = |target| SsaInstruction::synthetic(SsaOp::Jump { target });
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(constant(condition, 1));
        b0.add_instruction(constant(value, 7));
        b0.add_instruction(branch(1, 2));
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(jump(2));
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(first, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(value, 0));
        phi.add_operand(PhiOperand::new(value, 1));
        b2.add_phi(phi);
        b2.add_instruction(branch(3, 4));
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(jump(4));
        let mut b4 = SsaBlock::new(4);
        let mut phi = PhiNode::new(second, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(first, 2));
        phi.add_operand(PhiOperand::new(first, 3));
        b4.add_phi(phi);
        b4.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(second))));
        for block in [b0, b1, b2, b3, b4] {
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let trivial = PhiAnalyzer::new(&ssa).find_all_trivial(&(0..5).collect::<BTreeSet<usize>>());
        assert_eq!(trivial.len(), 2, "{trivial:?}");
        simplify_trivial_phis(
            &mut ssa,
            &trivial,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
        );
        assert_mock_valid_full(&ssa, "chained trivial phis");
    }
}
