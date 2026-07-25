//! Fuzzes `SsaFunction::validate` at every level.
//!
//! The verifier is the safety net for malformed IR, so it must *report* bad
//! shapes rather than panic on them. The failure mode this guards against is a
//! raw block or variable index reaching an asserting `BitSet` accessor — for
//! example in `check_dominance` or `place_pruned_phis` — which aborts the
//! process on exactly the input the verifier is being asked to reject.

#![no_main]

use analyssa::analysis::{SsaVerifier, VerifyLevel};
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let Some(ssa) = common::from_bytes(data) else {
        return;
    };
    // Every level must complete. What they report is not asserted — malformed
    // input legitimately produces errors — only that they return.
    for level in [
        VerifyLevel::Quick,
        VerifyLevel::Standard,
        VerifyLevel::Full,
    ] {
        let _ = SsaVerifier::new(&ssa).verify(level);
    }
    let _ = ssa.validate();
});
