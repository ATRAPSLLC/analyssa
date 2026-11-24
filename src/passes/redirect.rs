//! Redirecting control flow past trampolines: where a chain of them ends,
//! whether an edge may go straight there, and the rewrite that sends it.
//!
//! Shared by the control-flow, block-merge, threading, loop and residual
//! passes, which each redirect edges by these rules.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Result,
    analysis::exceptions::FinallyFlow,
    ir::{
        function::{SsaEditor, SsaFunction},
        ops::SsaOp,
    },
    target::Target,
};

/// Where a trampoline chain starting at `start` ends, as the function stands:
/// the last trampoline of the chain and the block it jumps to. `None` when
/// `start` is not one of `trampolines` or the chain loops.
///
/// The chain is followed through the function, not through the map's recorded
/// targets: an edit session that has already rewritten a trampoline's own jump
/// has moved where its chain ends, and a chain end read from before that names
/// a last block the path no longer passes through. The map says only which
/// blocks may be passed through.
pub fn chain_end<T: Target>(
    ssa: &SsaFunction<T>,
    trampolines: &BTreeMap<usize, usize>,
    start: usize,
) -> Option<(usize, usize)> {
    if !trampolines.contains_key(&start) {
        return None;
    }
    let mut last = start;
    let mut visited = BTreeSet::new();
    loop {
        let next = ssa.block(last)?.is_trampoline()?;
        if !trampolines.contains_key(&next) {
            return Some((last, next));
        }
        if !visited.insert(next) {
            return None;
        }
        last = next;
    }
}

/// Whether `pred` can jump straight to `target` in place of going through the
/// chain whose last block is `last`.
///
/// Every phi of `target` must know what arrives from `last` - that is what the
/// new edge brings - and must not already hold something else for `pred`. A
/// predecessor that reaches `target` both directly and through the chain would
/// otherwise have two values on what becomes one edge, and merging them is not
/// semantics-preserving.
fn redirect_is_consistent<T: Target>(
    ssa: &SsaFunction<T>,
    target: usize,
    last: usize,
    pred: usize,
) -> bool {
    let Some(block) = ssa.block(target) else {
        return false;
    };
    block.phi_nodes().iter().all(|phi| {
        let value_from = |from: usize| {
            phi.normal_operands()
                .find(|operand| operand.predecessor() == from)
                .map(|operand| operand.value())
        };
        match (value_from(last), value_from(pred)) {
            (Some(arriving), Some(existing)) => arriving == existing,
            (Some(_), None) => true,
            (None, _) => false,
        }
    })
}

/// A block whose terminator was redirected: the block, then its successors
/// before and after.
type Redirected = (usize, Vec<usize>, Vec<usize>);

/// Rewrites every edge into a trampoline chain to go straight to where the
/// chain ends, keeping the destination's phis consistent.
///
/// Each redirected edge gives the destination's phis an operand for the new
/// predecessor carrying what arrived from the chain's last block - trampolines
/// have no phis, so that is what the path always brought. An edge whose
/// redirect would give one predecessor two values on what becomes one edge is
/// left as it is (see [`redirect_is_consistent`]), and so is an edge into a
/// chain that loops.
///
/// A `leave` that runs a `finally` on its way is not redirected: its target
/// is entered from the `endfinally` that resumes there, not from the `leave`,
/// so the operand a redirect would give the target's phis would name a block
/// that is not their predecessor and leave the real one without one.
///
/// The trampolines themselves are left in place; a caller that clears them
/// clears only the ones nothing jumps to any more.
///
/// # Arguments
///
/// * `editor` - Active edit session over the function.
/// * `trampolines` - Map from each trampoline block to its immediate jump
///   target.
/// * `skip` - Blocks whose own terminators are not rewritten.
///
/// # Returns
///
/// Every block whose terminator changed, with its successors before and after.
///
/// # Errors
///
/// Propagates an editor failure; the consistency check makes none expected.
pub fn redirect_past_trampolines<T: Target>(
    editor: &mut SsaEditor<T>,
    trampolines: &BTreeMap<usize, usize>,
    skip: &dyn Fn(usize) -> bool,
) -> Result<Vec<Redirected>> {
    let mut redirected = Vec::new();
    if trampolines.is_empty() {
        return Ok(redirected);
    }
    let block_count = editor.function().block_count();
    let finally_flow = FinallyFlow::of(editor.function());
    for block_idx in 0..block_count {
        if skip(block_idx) || finally_flow.diverted_target(block_idx).is_some() {
            continue;
        }
        let Some(old_targets) = editor
            .function()
            .block(block_idx)
            .and_then(|block| block.control_terminator())
            .map(SsaOp::successors)
        else {
            continue;
        };
        let mut changed = false;
        let mut seen = BTreeSet::new();
        for &target in &old_targets {
            if !seen.insert(target) {
                continue;
            }
            let Some((last, ultimate)) = chain_end(editor.function(), trampolines, target) else {
                continue;
            };
            if ultimate == target
                || !redirect_is_consistent(editor.function(), ultimate, last, block_idx)
                || !editor
                    .function()
                    .retarget_respects_regions(block_idx, target, ultimate)
            {
                continue;
            }
            if editor.bypass_forwarding_chain(block_idx, target, ultimate, last)? {
                changed = true;
            }
        }
        if changed {
            let new_targets = editor
                .function()
                .block(block_idx)
                .and_then(|block| block.control_terminator())
                .map(SsaOp::successors)
                .unwrap_or_default();
            redirected.push((block_idx, old_targets, new_targets));
        }
    }
    Ok(redirected)
}
