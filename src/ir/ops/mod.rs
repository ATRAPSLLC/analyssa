//! Decomposed SSA operations — the core opcode representation.
//!
//! This module defines [`SsaOp`], the decomposed operation that puts a lifted
//! instruction into clean `result = op(operands)` form, along with the helper
//! types for classifying and extracting operation semantics.
//!
//! # Design Goals
//!
//! - **Explicit definitions**: every value an operation produces is a named SSA
//!   variable, including secondary outputs such as flags bundles and the
//!   high/low halves of wide arithmetic — see [`SsaOp::defs`].
//! - **Explicit operands**: all data dependencies are explicit SSA variables —
//!   no implicit stack.
//! - **Pattern matching**: enum variants enable easy destructuring for analysis
//!   passes.
//! - **Uniform access**: [`SsaOp::dest`], [`SsaOp::uses`],
//!   [`SsaOp::replace_uses`], [`SsaOp::as_binary_op`], and friends work across
//!   every variant.
//!
//! # Operation Categories
//!
//! | Category | Variants |
//! |----------|----------|
//! | Constants | `Const` |
//! | Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Neg`, their `Ovf` variants, and `WideMul` / `WideDiv` |
//! | Bitwise | `And`, `Or`, `Xor`, `Not`, `Shl`, `Shr`, `Rol`, `Ror`, `Rcl`, `Rcr` |
//! | Comparison | `Ceq`, `Clt`, `Cgt`, `BranchCmp` (combined compare-and-branch) |
//! | Conversion | `IntConv`, `IntToPtr`, `PtrToInt`, `IntToFloat`, `FloatToInt`, `FloatConv` |
//! | Control flow | `Jump`, `Branch`, `Switch`, `Return`, `Leave`, `Throw`, `Rethrow` |
//! | Memory | Field load/store, element load/store, indirect load/store |
//! | Atomics | Fences, atomic RMW, compare-exchange, lock-prefixed forms |
//! | Vector | Grouped unary/binary/ternary/compare, load/store, shuffle, reduce, mask ops |
//! | Native | `NativeOpaque`, flag reads/writes, system and compute ops |
//! | Calls | `Call`, `CallVirt`, `CallIndirect` |
//! | Synthetic | `Phi`, `Copy`, `Pop`, `Nop` |
//!
//! # Field Naming Conventions
//!
//! Consistent across all variants:
//! - `dest`: Destination SSA variable for the operation result
//! - `left`, `right`: Binary operands (left / right hand side)
//! - `operand`: Unary operand
//! - `object`: Object instance for field/method operations
//! - `array`, `index`: Array and index for element operations
//! - `addr`: Address for indirect memory operations
//! - `target`, `true_target`, `false_target`: Branch target block indices
//! - `unsigned`: Whether the operation treats values as unsigned
//! - `overflow_check`: Whether the operation checks for overflow
//!
//! # Module Layout
//!
//! [`SsaOp`] has 200 variants, so its surface is split by concern rather than
//! kept in one file. Every submodule is public and every *type* is re-exported
//! here, so the paths callers use for the data model are unchanged; the
//! inherent `impl SsaOp` blocks live in [`classify`], [`control`] and
//! [`display`], which a glob cannot re-export and which callers reach through
//! the methods themselves.
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`def`] | The [`SsaOp`] enum and its variants |
//! | [`kinds`] | Comparison, atomic, and taxonomy vocabulary types |
//! | [`vector`] | Vector / SIMD descriptors and lane semantics |
//! | [`native`] | Machine state accesses and condition-flag semantics |
//! | [`table`] | The [`OpKindTable`] contract shared by every kind enum |
//! | [`effects`] | Effect classification: `effects`, `may_throw`, `is_pure` |
//! | [`visit`] | Operand and definition access and substitution |
//! | [`control`] | Terminator and branch-target queries |
//! | [`classify`] | Taxonomy, similarity tokens, structural views |
//! | [`display`] | Human-readable rendering |
//! | `operands` | The operand policy the boxed-payload variants share |
//!
//! Adding a variant touches the exhaustive matches in [`effects`], [`visit`],
//! [`classify`], and [`display`]; the compile-time exhaustiveness sentinel in
//! the test module names every variant so the build breaks until each is
//! handled deliberately.
//!
//! # The boxed-payload variants
//!
//! Forty-five variants carry a `Box<...Data>` whose whole operand storage is an
//! `outputs` list and an `inputs` list. What those lists *mean* — `outputs` are
//! the operation's definitions, `inputs` are its uses, definitions come first,
//! `outputs[0]` is the primary destination, and the stack effect is
//! `(inputs.len(), outputs.len())` — is written once, in the blanket
//! `KindedOperands` impl in the crate-private `operands` module, and the arms in
//! [`visit`], [`classify`] and [`display`] delegate to it. Restating the policy
//! per variant is what let an arm be present and wrong: exhaustiveness forces an
//! arm to *exist*, and nothing then checks that it assigns the right roles in
//! the right order.
//!
//! A new payload struct joins by naming itself and **all** its fields in the
//! `impl_kinded_payload!` registry at the foot of its file. The field list
//! expands to a destructuring with no `..`, so a payload that later grows a
//! field does not compile until someone decides whether that field is an
//! operand; a payload nobody registers does not compile at the match arm.
//! `SsaOp::defs` needs no arm at all — it is the `Def`- and `FlagsDef`-role
//! operands of [`SsaOp::visit_operands`], so it cannot disagree with the walk
//! that [`SsaOp::dest`], [`SsaOp::flags_dest`] and [`SsaOp::replace_def`] also
//! read.
//!
//! Adding a *kind enum* should also register it with [`table`]: nothing in the
//! type system forces that, so the registry in `table::all_tables` is where a
//! new enum joins the count, spelling and injectivity checks.

pub mod classify;
pub mod control;
pub mod def;
pub mod display;
pub mod effects;
pub mod kinds;
pub mod native;
mod operands;
pub mod table;
pub mod vector;
pub mod visit;

#[cfg(test)]
mod tests;

pub use self::{def::*, effects::*, kinds::*, native::*, table::*, vector::*, visit::*};
