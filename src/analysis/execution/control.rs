//! Where a body receives the address it returns to, which the executor checks
//! it returns through.

use crate::{
    ir::{CallInputKey, SsaVarId},
    target::TypeLayout,
};

/// Where a call leaves its callee the address the callee returns to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnAddressStorage {
    /// The word at the callee's entry stack pointer, where a call instruction
    /// pushes it (x86): a return reads it from there.
    Stack,
    /// A register, the link register (AArch64's `x30`, Arm's `lr`, RISC-V's
    /// `ra`, MIPS's `$ra`), and each return names the value it returns
    /// through ([`SsaOp::Return`](crate::ir::ops::SsaOp::Return)'s `target`).
    /// A callee's value of the register on entry is what its call left there,
    /// bound from the caller's inputs; `entry` names it so a root, or a
    /// callee whose call states nothing of it, receives the address it
    /// returns to.
    Register {
        /// The body's entry value of the register.
        entry: Option<SsaVarId>,
        /// The key a native call states the register's value under among its
        /// implicit inputs: what the activation's link register holds on
        /// entry, which code it continues in finds there while nothing names
        /// the register.
        key: Option<CallInputKey>,
    },
}

/// How a body returns: the layout of a return address and where the body
/// receives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReturnContract {
    /// Pointer-width layout of a return address.
    pub layout: TypeLayout,
    /// Where the body receives it.
    pub storage: ReturnAddressStorage,
}
