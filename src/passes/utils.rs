//! Shared utility functions used across SSA transformation passes.
//!
//! Provides:
//!
//! - [`is_power_of_two`] - detects whether an integer is a positive power
//!   of two (used by strength reduction).
//! - [`resolve_chain`] - follows a chain of transitive mappings to find
//!   the ultimate target (used by trampoline resolution, copy propagation,
//!   and control flow simplification).
//! - [`resolve_chains`] - resolves a whole list of substitutions that way at
//!   once, so applying them in any order lands every use on its final value.

use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::FxHashSet;

use crate::{analysis::loopanalyzer::SsaLoopAnalysis, ir::function::SsaFunction, target::Target};

/// Test whether an integer is a positive power of two.
///
/// Returns `Some(exponent)` where `exponent` is `log2(value)` if `value`
/// is `2^n` for some `0 ≤ n ≤ 63`. Returns `None` for non-positive or
/// non-power-of-two values.
///
/// Used by strength reduction (`x * 2^n → x << n`, `x / 2^n → x >> n`,
/// `x % 2^n → x & (2^n - 1)`).
///
/// # Arguments
///
/// * `value` - The integer to test. Must be positive and a power of two.
///
/// # Returns
///
/// `Some(exponent)` where `2^exponent == value`, or `None` if `value` is
/// not a positive power of two.
#[must_use]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
pub fn is_power_of_two(value: i64) -> Option<u8> {
    if value <= 0 {
        return None;
    }
    let value = value as u64;
    if value.is_power_of_two() {
        Some(value.trailing_zeros() as u8)
    } else {
        None
    }
}

/// Follow a chain of transitive mappings to find the ultimate target.
///
/// Given a map `{key → value}` where values may themselves be keys,
/// follows the chain until reaching a value that is not a key. Handles
/// cycles by stopping when a previously visited key is encountered.
///
/// Used by:
/// - Trampoline resolution: `block → block → block` becomes `block → ultimate`.
/// - Copy propagation: `var → var → var` becomes `var → ultimate`.
/// - Control flow simplification: resolving branch targets through chains.
///
/// # Arguments
///
/// * `map` - The transitive mapping (e.g., `{1: 2, 2: 3, 3: 4}`).
/// * `start` - The starting key.
///
/// # Returns
///
/// The ultimate target after following the chain. If `start` is not a key
/// in `map`, returns `start` unchanged.
#[must_use]
pub fn resolve_chain<K>(map: &BTreeMap<K, K>, start: K) -> K
where
    K: Copy + Ord,
{
    let mut current = start;
    let mut visited = BTreeSet::new();

    while let Some(&next) = map.get(&current) {
        if !visited.insert(current) {
            break;
        }
        current = next;
    }

    current
}

/// Resolves a list of `(old, new)` substitutions to a fixed point: every key
/// maps straight to the end of its chain.
///
/// Applying substitutions one at a time is the defect this prevents: given
/// `(q -> p)` and `(p -> r)`, rewriting `p`'s uses to `r` and then `q`'s to
/// `p` writes `p` back into use position after its own uses were rewritten -
/// and `p` is typically a definition the caller is about to delete. Resolved,
/// both map to `r`, and no right-hand side is still a key.
///
/// A key given twice resolves through the pair given last. A cycle, which no
/// substitution walking toward earlier definitions produces, stops where it
/// would revisit a key rather than looping.
///
/// # Arguments
///
/// * `pairs` - `(old, new)` pairs, possibly chained.
///
/// # Returns
///
/// Each `old` with the end of its chain.
#[must_use]
pub fn resolve_chains<K>(pairs: &[(K, K)]) -> BTreeMap<K, K>
where
    K: Copy + Ord,
{
    let direct: BTreeMap<K, K> = pairs.iter().copied().collect();
    direct
        .iter()
        .map(|(from, to)| (*from, resolve_chain(&direct, *to)))
        .collect()
}

/// Returns the block indices that hold canonical loop structure - preheaders
/// and latches - which CFG-simplification passes must preserve.
///
/// The loop canonicalizer inserts a single non-loop predecessor (preheader) in
/// front of each loop header and a single unified back-edge block (latch) when a
/// loop has several. Both are forwarding blocks: an empty preheader or unified
/// latch is just a jump, which the control-flow and block-merge passes would
/// otherwise eliminate as a trampoline. Doing so re-exposes the loop's multiple
/// non-loop predecessors / multiple latches, and the canonicalizer re-inserts
/// the block on the next iteration - an infinite oscillation that never reaches
/// a fixpoint. Skipping these blocks keeps the IR in canonical loop-simplify
/// form (the same trade-off LLVM's `simplifycfg` makes under loop-simplify
/// form). Non-trampoline preheaders/latches are unaffected - they are never in
/// the trampoline set to begin with.
///
/// # Arguments
///
/// * `ssa` - The SSA function whose loop forest is analyzed.
///
/// # Returns
///
/// The set of preheader and latch block indices to preserve. Empty if the
/// function contains no loops.
#[must_use]
pub fn loop_canonical_blocks<T: Target>(ssa: &SsaFunction<T>) -> FxHashSet<usize> {
    let mut blocks = FxHashSet::default();
    for loop_info in ssa.analyze_loops().iter() {
        if let Some(preheader) = loop_info.preheader {
            blocks.insert(preheader.index());
        }
        for latch in &loop_info.latches {
            blocks.insert(latch.index());
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight-line chain of mappings resolves to its final target from any
    /// starting point along the chain.
    #[test]
    fn follows_mappings() {
        let mut map = BTreeMap::new();
        map.insert(1, 2);
        map.insert(2, 3);
        map.insert(3, 4);

        assert_eq!(resolve_chain(&map, 1), 4);
        assert_eq!(resolve_chain(&map, 2), 4);
        assert_eq!(resolve_chain(&map, 3), 4);
        assert_eq!(resolve_chain(&map, 4), 4);
    }

    /// A cyclic mapping terminates instead of looping forever, returning one of
    /// the cycle's members.
    #[test]
    fn handles_cycles() {
        let mut map = BTreeMap::new();
        map.insert(1, 2);
        map.insert(2, 1);

        let result = resolve_chain(&map, 1);
        assert!(result == 1 || result == 2);
    }

    /// A single mapping resolves to its direct target.
    #[test]
    fn single_step() {
        let mut map = BTreeMap::new();
        map.insert(5, 10);

        assert_eq!(resolve_chain(&map, 5), 10);
    }

    /// A key absent from an empty map resolves to itself unchanged.
    #[test]
    fn empty_map() {
        let map: BTreeMap<usize, usize> = BTreeMap::new();
        assert_eq!(resolve_chain(&map, 42), 42);
    }

    /// A key mapping to itself terminates rather than spinning on the self-edge.
    #[test]
    fn self_loop() {
        let mut map = BTreeMap::new();
        map.insert(1, 1);

        assert_eq!(resolve_chain(&map, 1), 1);
    }

    #[test]
    fn test_resolve_chains_lands_every_key_on_its_final_value() {
        let resolved = resolve_chains(&[(2, 1), (3, 2), (5, 4)]);
        assert_eq!(resolved.get(&3), Some(&1), "a chain resolves to its end");
        assert_eq!(resolved.get(&2), Some(&1));
        assert_eq!(resolved.get(&5), Some(&4));
        assert!(
            resolved.values().all(|value| !resolved.contains_key(value)),
            "no resolved value is itself still a key: {resolved:?}"
        );
    }

    #[test]
    fn test_resolve_chains_takes_the_last_pair_for_a_repeated_key() {
        let resolved = resolve_chains(&[(1, 2), (1, 3)]);
        assert_eq!(resolved.get(&1), Some(&3));
    }
}
