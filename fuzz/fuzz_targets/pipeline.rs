//! Fuzzes the normalization passes over malformed IR.
//!
//! Passes run on whatever a frontend produced, and a frontend fed a hostile
//! binary produces IR no compiler would emit. A pass must not panic on it; the
//! checked-edit boundary is allowed to reject the edit, but the process must
//! survive.
//!
//! Each pass runs on its own copy, so one pass's rewrite cannot mask another's
//! crash.

#![no_main]

use analyssa::{PointerSize, events::NullListener, passes};
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let Some(ssa) = common::from_bytes(data) else {
        return;
    };
    let method = 0u32;

    let mut gvn = ssa.clone();
    let _ = passes::gvn::run(&mut gvn, &method, &NullListener);

    let mut dce = ssa.clone();
    let _ = passes::deadcode::run(&mut dce, &method, &NullListener, 8);

    let mut algebraic = ssa.clone();
    let _ = passes::algebraic::run(&mut algebraic, &method, &NullListener);

    let mut copying = ssa.clone();
    let _ = passes::copying::run(&mut copying, &method, &NullListener, 8);

    let mut memory = ssa.clone();
    let _ = passes::memory::run(&mut memory, &method, &NullListener, PointerSize::Bit64);

    let mut blockmerge = ssa.clone();
    let _ = passes::blockmerge::run(&mut blockmerge, &method, &NullListener, 8);

    let mut controlflow = ssa.clone();
    let _ = passes::controlflow::run(&mut controlflow, &method, &NullListener, 8);

    let mut licm = ssa.clone();
    let _ = passes::licm::run(&mut licm, &method, &NullListener);

    let mut reassociate = ssa.clone();
    let _ = passes::reassociate::run(
        &mut reassociate,
        &method,
        &NullListener,
        PointerSize::Bit64,
    );
});
