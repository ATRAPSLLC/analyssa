//! One operand policy for the boxed-payload [`SsaOp`](crate::ir::ops::def::SsaOp) variants.
//!
//! Forty-five `SsaOp` variants carry a boxed payload whose whole operand
//! storage is an `outputs` list and an `inputs` list. The policy over those two
//! lists is fixed: `outputs` are the operation's definitions, `inputs` are its
//! uses, definitions are visited first, `outputs[0]` is the primary
//! destination, and the stack effect is `(inputs.len(), outputs.len())`.
//!
//! That policy lives here, in the four bodies of the blanket
//! `impl<P: KindedPayload> KindedOperands for P`, and nowhere else. A blanket
//! impl cannot be specialised, so no payload type can give an input a `Def`
//! role or reverse the ordering, and [`KindedPayload`] is sealed, so no type
//! can join the policy except through [`impl_kinded_payload!`], whose body
//! literally writes `&self.outputs` and `&self.inputs` and therefore cannot
//! transpose them either.
//!
//! # The contract [`KindedPayload`] carries
//!
//! `outputs` and `inputs` are the payload's **complete** operand storage. A
//! payload with a third `SsaVarId` field would silently keep that field out of
//! def-use construction, liveness, renaming and the verifier, so
//! [`impl_kinded_payload!`] takes each payload's full field list and expands to
//! an exhaustive destructuring with no `..`. Adding a field to a registered
//! payload is E0027 at the registration, which forces a decision about whether
//! the new field is an operand.

use crate::ir::{ops::kinds::OperandRole, variable::SsaVarId};

/// Seals [`KindedPayload`] against implementations outside this module's
/// macro.
pub(super) mod sealed {
    /// The seal itself. Only [`impl_kinded_payload!`](super::impl_kinded_payload)
    /// implements it, so a hand-written [`KindedPayload`](super::KindedPayload)
    /// -- which could transpose `outputs` and `inputs`, or answer with a subset
    /// -- does not compile.
    pub trait Sealed {}
}

/// A boxed `SsaOp` payload whose complete operand storage is an `outputs` list
/// and an `inputs` list.
///
/// Implemented only by [`impl_kinded_payload!`]. Implementors get
/// [`KindedOperands`] for free through the blanket impl, which is where the
/// operand policy is written.
pub(super) trait KindedPayload: sealed::Sealed {
    /// Returns the variables this operation defines, in payload order.
    fn outputs(&self) -> &[SsaVarId];

    /// Returns the variables this operation defines, mutably.
    fn outputs_mut(&mut self) -> &mut [SsaVarId];

    /// Returns the variables this operation uses, in payload order.
    fn inputs(&self) -> &[SsaVarId];

    /// Returns the variables this operation uses, mutably.
    fn inputs_mut(&mut self) -> &mut [SsaVarId];
}

/// The operand policy every [`KindedPayload`] obeys.
///
/// Deliberately a blanket impl over `KindedPayload` rather than a per-type one:
/// the role assignment, the definitions-before-uses ordering and the stack
/// effect then exist in exactly three function bodies for the whole crate, and
/// a payload type cannot opt out of any of them.
pub(super) trait KindedOperands: KindedPayload {
    /// Visits every operand with its role: outputs as [`OperandRole::Def`],
    /// then inputs as [`OperandRole::Use`].
    fn visit<F>(&self, f: &mut F)
    where
        F: FnMut(OperandRole, SsaVarId);

    /// Visits every operand mutably, in the order and with the roles
    /// [`Self::visit`] reports.
    fn visit_mut<F>(&mut self, f: &mut F)
    where
        F: FnMut(OperandRole, &mut SsaVarId);

    /// Returns the operation's stack effect as `(pops, pushes)` -- one pop per
    /// input consumed, one push per output defined.
    fn operand_counts(&self) -> (u32, u32);
}

impl<P: KindedPayload> KindedOperands for P {
    fn visit<F>(&self, f: &mut F)
    where
        F: FnMut(OperandRole, SsaVarId),
    {
        for output in self.outputs() {
            f(OperandRole::Def, *output);
        }
        for input in self.inputs() {
            f(OperandRole::Use, *input);
        }
    }

    fn visit_mut<F>(&mut self, f: &mut F)
    where
        F: FnMut(OperandRole, &mut SsaVarId),
    {
        for output in self.outputs_mut() {
            f(OperandRole::Def, output);
        }
        for input in self.inputs_mut() {
            f(OperandRole::Use, input);
        }
    }

    fn operand_counts(&self) -> (u32, u32) {
        // An operand list long enough to truncate would need four billion
        // entries in one instruction.
        #[allow(clippy::cast_possible_truncation)]
        let counts = (self.inputs().len() as u32, self.outputs().len() as u32);
        counts
    }
}

/// Registers payload structs with [`KindedPayload`], and with it the operand
/// policy of [`KindedOperands`].
///
/// Each entry names the struct and **every one of its fields**. The field list
/// expands to a destructuring with no `..`, so a field added to a registered
/// payload fails to compile here (E0027) until someone decides whether it is an
/// operand. Registration is what makes a payload usable in the one-line match
/// arms of `visit_operands`, `visit_operands_mut` and `stack_effect`; a payload
/// nobody registers is E0599 at those arms rather than a silent gap.
///
/// Three entry forms:
///
/// ```text
/// impl_kinded_payload! {
///     VecImm8Data { imm8, outputs, inputs };
///     KindedVecData<K> { kind, outputs, inputs };
///     outputs_only VectorElementCountData { element_bits, multiplier, outputs };
/// }
/// ```
macro_rules! impl_kinded_payload {
    () => {};

    ($ty:ident { $($field:ident),+ $(,)? }; $($rest:tt)*) => {
        impl $crate::ir::ops::operands::sealed::Sealed for $ty {}

        impl $crate::ir::ops::operands::KindedPayload for $ty {
            fn outputs(&self) -> &[$crate::ir::variable::SsaVarId] {
                &self.outputs
            }

            fn outputs_mut(&mut self) -> &mut [$crate::ir::variable::SsaVarId] {
                &mut self.outputs
            }

            fn inputs(&self) -> &[$crate::ir::variable::SsaVarId] {
                &self.inputs
            }

            fn inputs_mut(&mut self) -> &mut [$crate::ir::variable::SsaVarId] {
                &mut self.inputs
            }
        }

        impl_kinded_payload!(@fields $ty ($($field),+));
        impl_kinded_payload!($($rest)*);
    };

    ($ty:ident<$param:ident> { $($field:ident),+ $(,)? }; $($rest:tt)*) => {
        impl<$param> $crate::ir::ops::operands::sealed::Sealed for $ty<$param> {}

        impl<$param> $crate::ir::ops::operands::KindedPayload for $ty<$param> {
            fn outputs(&self) -> &[$crate::ir::variable::SsaVarId] {
                &self.outputs
            }

            fn outputs_mut(&mut self) -> &mut [$crate::ir::variable::SsaVarId] {
                &mut self.outputs
            }

            fn inputs(&self) -> &[$crate::ir::variable::SsaVarId] {
                &self.inputs
            }

            fn inputs_mut(&mut self) -> &mut [$crate::ir::variable::SsaVarId] {
                &mut self.inputs
            }
        }

        impl_kinded_payload!(@generic_fields $ty<$param> ($($field),+));
        impl_kinded_payload!($($rest)*);
    };

    (outputs_only $ty:ident { $($field:ident),+ $(,)? }; $($rest:tt)*) => {
        impl $crate::ir::ops::operands::sealed::Sealed for $ty {}

        impl $crate::ir::ops::operands::KindedPayload for $ty {
            fn outputs(&self) -> &[$crate::ir::variable::SsaVarId] {
                &self.outputs
            }

            fn outputs_mut(&mut self) -> &mut [$crate::ir::variable::SsaVarId] {
                &mut self.outputs
            }

            fn inputs(&self) -> &[$crate::ir::variable::SsaVarId] {
                &[]
            }

            fn inputs_mut(&mut self) -> &mut [$crate::ir::variable::SsaVarId] {
                &mut []
            }
        }

        impl_kinded_payload!(@fields $ty ($($field),+));
        impl_kinded_payload!($($rest)*);
    };

    (@fields $ty:ident ($($field:ident),+)) => {
        const _: () = {
            /// Fails to compile if the payload grows a field the registration
            /// does not name, so a new field cannot join it without someone
            /// deciding whether the field is an operand.
            #[allow(dead_code)]
            fn every_field_is_accounted_for(payload: $ty) {
                let $ty { $($field: _),+ } = payload;
            }
        };
    };

    (@generic_fields $ty:ident<$param:ident> ($($field:ident),+)) => {
        const _: () = {
            /// Fails to compile if the payload grows a field the registration
            /// does not name, so a new field cannot join it without someone
            /// deciding whether the field is an operand.
            #[allow(dead_code)]
            fn every_field_is_accounted_for<$param>(payload: $ty<$param>) {
                let $ty { $($field: _),+ } = payload;
            }
        };
    };
}

pub(super) use impl_kinded_payload;
