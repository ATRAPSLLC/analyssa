//! Operand and definition access for [`SsaOp`] - reading, iterating, and
//! substituting the SSA variables an operation mentions.
//!
//! [`SsaOp::visit_operands`] and [`SsaOp::visit_operands_mut`] are the two
//! exhaustive walks every other accessor here is built on. They must visit the
//! same operands, in the same order, with the same [`OperandRole`]: the shared
//! walk feeds def-use index construction while the mutable one performs
//! renaming, so a divergence corrupts SSA silently rather than failing loudly.
//! The `visit_operands_mut_agrees_with_visit_operands` test pins that
//! invariant across every variant.
//!
//! [`SsaOp::dest`], [`SsaOp::flags_dest`], [`SsaOp::defs`],
//! [`SsaOp::for_each_use`], [`SsaOp::uses`], [`SsaOp::replace_def`] and
//! [`SsaOp::replace_uses`] are all expressed over those two walks, so none of
//! them can disagree with the others about which operands an operation has or
//! what role each one plays. For the forty-five boxed-payload variants the
//! walks themselves delegate to one shared operand policy, so their arms carry
//! no per-variant statement of that policy either.

use std::convert::Infallible;

use crate::{
    ir::{
        float::{FloatExceptionLayout, FloatExceptions},
        ops::{
            calls::{CallData, CallInput, Callee},
            def::SsaOp,
            kinds::{CompareBlkData, OperandRole},
            operands::KindedOperands,
            vector::VectorShuffleSelector,
        },
        variable::SsaVarId,
    },
    target::Target,
};

/// Iterator over the variables an SSA operation defines, in the order
/// [`SsaOp::visit_operands`] reports them.
///
/// Owns its contents. [`SsaOp::defs`] collects the `Def` and `FlagsDef`
/// operands of one walk over the operation, so this iterator cannot disagree
/// with that walk - or with [`SsaOp::dest`], [`SsaOp::flags_dest`] and
/// [`SsaOp::replace_def`], which read the same walk - about what the
/// definitions are, and it does not borrow the operation it came from.
///
/// Two definitions live in fixed slots, so single- and double-result operations
/// allocate nothing. Additional outputs spill to a `Vec`, appended in payload
/// order and reversed once so construction is linear in the number of outputs.
///
/// The shape is chosen for the hot path, because `defs()` runs per instruction
/// in def-use index construction, liveness and the verifier. Two fixed slots
/// the iterator takes from, rather than an inline buffer addressed by a
/// running index, are what keep it there: the slots need no cursor field and
/// their stores fold, and the spill's push is out of line so the allocation it
/// needs is not code the callers that never spill have to step around.
#[derive(Debug, Clone, Default)]
pub struct SsaDefs {
    /// The first definition.
    first: Option<SsaVarId>,
    /// The second definition.
    second: Option<SsaVarId>,
    /// Definitions past the second, reversed after collection so iteration
    /// pops in payload order. Empty for single- and double-result operations.
    rest: Vec<SsaVarId>,
}

impl SsaDefs {
    /// Appends one definition.
    #[inline]
    fn push(&mut self, var: SsaVarId) {
        if self.first.is_none() {
            self.first = Some(var);
        } else if self.second.is_none() {
            self.second = Some(var);
        } else {
            self.push_rest(var);
        }
    }

    /// Appends a definition past the second.
    ///
    /// Kept out of line so spill allocation is absent from the common path
    /// for one or two definitions.
    #[cold]
    #[inline(never)]
    fn push_rest(&mut self, var: SsaVarId) {
        self.rest.push(var);
    }

    /// Returns how many definitions the iterator has yet to yield.
    fn remaining(&self) -> usize {
        usize::from(self.first.is_some())
            .saturating_add(usize::from(self.second.is_some()))
            .saturating_add(self.rest.len())
    }
}

impl Iterator for SsaDefs {
    type Item = SsaVarId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(var) = self.first.take() {
            return Some(var);
        }
        if let Some(var) = self.second.take() {
            return Some(var);
        }
        self.rest.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SsaDefs {}

/// Where an operation's governing mask and passthrough are among its uses
/// ([`SsaOp::uses`]), for one whose mask governs the lanes it writes or
/// accesses ([`SsaOp::governing_uses`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GoverningUses {
    /// The position of the mask.
    pub mask: usize,
    /// The position of the value the lanes the mask leaves inactive keep,
    /// where the operation merges rather than zeroes them.
    pub passthrough: Option<usize>,
}

impl<T: Target> SsaOp<T> {
    /// Returns the destination variable if this operation produces one.
    #[must_use]
    pub fn dest(&self) -> Option<SsaVarId> {
        self.try_visit_operands(|role, var| match role {
            OperandRole::Def => Err(Some(var)),
            OperandRole::Use => Err(None),
            OperandRole::FlagsDef => Ok(()),
        })
        .err()
        .flatten()
    }

    /// The operand whose width this operation computes in, for an operation
    /// whose result has the width of its operands.
    ///
    /// **An operation's width is the operation's, not its destination
    /// variable's recovered type.** `Mul { dest, left, right }` is documented as
    /// `dest = left * right` - one width throughout, `WideMul` being the
    /// separate op for a widening product - so when type recovery leaves `dest`
    /// `Unknown`, the operands still say what width the machine multiplied in.
    ///
    /// Reading the width from `dest` alone made "the destination has no
    /// recovered type" indistinguishable from "this value could be anything":
    /// `passes::ranges` clamps an arithmetic result to its width, and an unknown
    /// width answers `Top`, discarding a bound the operands had proved: `[0,
    /// 32767]` squared is `[0, 1073676289]`, and an untyped destination would
    /// turn it into nothing.
    ///
    /// The operands only settle it when they agree. CIL computes `int32 +
    /// native int` as a native int, so an operand pair of different widths -
    /// or one whose width `width_of` cannot tell - names no single width, and
    /// this answers `None` rather than guess the narrower.
    ///
    /// `None` for anything that is not same-width: loads and stores, calls,
    /// conversions - whose whole purpose is to change width, and which carry
    /// their target type anyway - comparisons, and the terminators.
    ///
    /// The shift and rotate operations answer with the value shifted, never the
    /// shift amount: `dest` and `value` share a width, and `amount` need not.
    /// A rotate is same-width by construction - it is the width that decides
    /// which bits wrap - so `Rol` and `Ror` answer here exactly as `Shl` and
    /// `Shr` do.
    ///
    /// # Arguments
    ///
    /// * `width_of` - The width of an operand's type in bits, when known.
    #[must_use]
    pub fn width_operand(&self, width_of: impl Fn(SsaVarId) -> Option<u32>) -> Option<SsaVarId> {
        let (first, second) = match self {
            Self::Add { left, right, .. }
            | Self::Sub { left, right, .. }
            | Self::Mul { left, right, .. }
            | Self::Div { left, right, .. }
            | Self::Rem { left, right, .. }
            | Self::And { left, right, .. }
            | Self::Or { left, right, .. }
            | Self::Xor { left, right, .. } => (*left, Some(*right)),
            Self::Neg { operand, .. } | Self::Not { operand, .. } => (*operand, None),
            Self::Shl { value, .. }
            | Self::Shr { value, .. }
            | Self::Rol { value, .. }
            | Self::Ror { value, .. } => (*value, None),
            _ => return None,
        };
        let width = width_of(first)?;
        match second {
            Some(other) if width_of(other) != Some(width) => None,
            _ => Some(first),
        }
    }

    /// Returns all variables defined by this operation, in the order
    /// [`Self::visit_operands`] reports them.
    ///
    /// Derived from that walk rather than matched per variant, so it cannot
    /// disagree with [`Self::dest`], [`Self::flags_dest`] or
    /// [`Self::replace_def`] about which operands are definitions: all four
    /// read the same walk. A per-variant match needs an arm for every variant
    /// that defines more than one variable, and the arm a new variant does not
    /// get is a definition silently dropped from the def-use index, liveness,
    /// SCCP, reaching definitions, taint and the verifier's dominance and
    /// uniqueness checks.
    #[must_use]
    #[inline]
    pub fn defs(&self) -> SsaDefs {
        let mut defs = SsaDefs::default();
        match self.try_for_each_def(|var| {
            defs.push(var);
            Ok::<(), Infallible>(())
        }) {
            Ok(()) => (),
            Err(error) => match error {},
        }
        // Store the spill in reverse once, preserving payload order on pop.
        defs.rest.reverse();
        defs
    }

    /// Replaces a definition variable without touching operand uses.
    ///
    /// Rewrites every definition equal to `old_var` - the primary destination,
    /// secondary outputs (high halves, status/fault outputs, native output
    /// lists), and flag outputs - and leaves `Use` operands untouched. This is
    /// used by SSA renaming, where definitions and uses have different scoping
    /// rules and are renamed in separate walks.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The definition variable to rewrite.
    /// * `new_var` - The variable to define in its place.
    ///
    /// # Returns
    ///
    /// `true` if at least one definition was changed.
    pub fn replace_def(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> bool {
        let mut changed = false;
        self.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Def | OperandRole::FlagsDef) && *var == old_var {
                *var = new_var;
                changed = true;
            }
        });
        changed
    }

    /// A return of `value` that leaves its caller no processor state of its
    /// own ([`Self::Return`]).
    #[must_use]
    pub fn returns(value: Option<SsaVarId>) -> Self {
        Self::Return {
            value,
            state: Box::default(),
            results: Box::default(),
            transfer: None,
        }
    }

    /// A return of `value` that leaves its caller the processor state
    /// `state`: the sticky status flags the function accrued, which a caller
    /// reads after the call ([`Self::Return`]).
    #[must_use]
    pub fn returns_with(value: Option<SsaVarId>, state: Vec<SsaVarId>) -> Self {
        Self::Return {
            value,
            state: state.into_boxed_slice(),
            results: Box::default(),
            transfer: None,
        }
    }

    /// A return of `value` that hands its caller `results` beside it, each
    /// under the key a call receives it by ([`CallData::results`]).
    #[must_use]
    pub fn returns_results(value: Option<SsaVarId>, results: Vec<CallInput>) -> Self {
        Self::Return {
            value,
            state: Box::default(),
            results: results.into_boxed_slice(),
            transfer: None,
        }
    }

    /// The raise ([`Self::FloatRaise`]) of the flags of `raises` that
    /// computing `op` sets, at the bits `layout` keeps them at, into `dest`,
    /// over the lanes `lanes` names (every lane for `None`).
    ///
    /// `op`'s definitions become `dest`: the raise reads what `op` reads and
    /// defines nothing but its flags.
    #[must_use]
    pub fn float_raise(
        dest: SsaVarId,
        mut op: Self,
        lanes: Option<SsaVarId>,
        raises: FloatExceptions,
        layout: FloatExceptionLayout,
    ) -> Self {
        op.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Def | OperandRole::FlagsDef) {
                *var = dest;
            }
        });
        Self::FloatRaise {
            dest,
            op: Box::new(op),
            lanes,
            raises,
            layout,
        }
    }

    /// Returns the flags destination if this operation defines flags.
    pub fn flags_dest(&self) -> Option<SsaVarId> {
        self.try_visit_operands(|role, var| match role {
            OperandRole::FlagsDef => Err(Some(var)),
            OperandRole::Use => Err(None),
            OperandRole::Def => Ok(()),
        })
        .err()
        .flatten()
    }

    /// This operation with its two operands exchanged, where exchanging them
    /// computes the same value: an `Add`, `Mul`, `And`, `Or` or `Xor` that
    /// sets no flags. `None` for any other operation.
    ///
    /// A comparison of two computations reads both ways: `v + c` and `c + v`
    /// are one sum.
    #[must_use]
    pub fn commuted(&self) -> Option<Self> {
        Some(match self {
            Self::Add {
                dest,
                left,
                right,
                flags: None,
            } => Self::Add {
                dest: *dest,
                left: *right,
                right: *left,
                flags: None,
            },
            Self::Mul {
                dest,
                left,
                right,
                flags: None,
            } => Self::Mul {
                dest: *dest,
                left: *right,
                right: *left,
                flags: None,
            },
            Self::And {
                dest,
                left,
                right,
                flags: None,
            } => Self::And {
                dest: *dest,
                left: *right,
                right: *left,
                flags: None,
            },
            Self::Or {
                dest,
                left,
                right,
                flags: None,
            } => Self::Or {
                dest: *dest,
                left: *right,
                right: *left,
                flags: None,
            },
            Self::Xor {
                dest,
                left,
                right,
                flags: None,
            } => Self::Xor {
                dest: *dest,
                left: *right,
                right: *left,
                flags: None,
            },
            _ => return None,
        })
    }

    /// Whether `other` is this operation over other operands: every field
    /// alike but the operands, which pair up in order with roles alike, each
    /// pair accepted by `agrees` (a definition of this operation with
    /// `other`'s, a use with its use).
    ///
    /// # Arguments
    ///
    /// * `other` - The operation compared with this one.
    /// * `agrees` - Whether this operation's operand, of the role given,
    ///   stands for `other`'s in its place.
    #[must_use]
    pub fn same_under(
        &self,
        other: &Self,
        mut agrees: impl FnMut(OperandRole, SsaVarId, SsaVarId) -> bool,
    ) -> bool
    where
        T::Type: PartialEq,
    {
        let (mut mine, mut theirs) = (Vec::new(), Vec::new());
        self.visit_operands(|role, var| mine.push((role, var)));
        other.visit_operands(|role, var| theirs.push((role, var)));
        if mine.len() != theirs.len()
            || mine
                .iter()
                .zip(&theirs)
                .any(|((role, own), (other_role, their))| {
                    role != other_role || !agrees(*role, *own, *their)
                })
        {
            return false;
        }
        let mut aligned = self.clone();
        let mut operands = theirs.iter().map(|(_, var)| *var);
        aligned.visit_operands_mut(|_, var| {
            if let Some(their) = operands.next() {
                *var = their;
            }
        });
        aligned == *other
    }

    /// Visits every SSA variable operand together with its [`OperandRole`].
    ///
    /// Operands are visited in payload order: definitions first (the primary
    /// destination, then secondary and flag outputs), then uses in payload
    /// order. This is the single source of truth for operand traversal -
    /// [`Self::dest`], [`Self::flags_dest`], [`Self::for_each_use`],
    /// [`Self::replace_uses`], and [`Self::replace_def`] are expressed over
    /// it.
    pub fn visit_operands<F>(&self, mut f: F)
    where
        F: FnMut(OperandRole, SsaVarId),
    {
        match self.try_visit_operands(|role, var| {
            f(role, var);
            Ok::<(), Infallible>(())
        }) {
            Ok(()) => (),
            Err(error) => match error {},
        }
    }

    /// Visits the uses of a nested raise with constant stack depth and
    /// inner-to-outer mask order.
    ///
    /// The leaf's uses precede every lane mask, and a callback error stops
    /// before the remaining operands. Only the outer definition is visible, and
    /// the caller visits it.
    fn try_visit_raised_uses<E>(
        &self,
        visit: &mut dyn FnMut(OperandRole, SsaVarId) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut first = None;
        let mut masks = Vec::new();
        let mut operation = self;
        while let Self::FloatRaise { op, lanes, .. } = operation {
            if let Some(lanes) = lanes {
                if first.is_none() {
                    first = Some(*lanes);
                } else {
                    masks.push(*lanes);
                }
            }
            operation = op;
        }
        operation.try_for_each_use(|var| visit(OperandRole::Use, var))?;
        for mask in masks.into_iter().rev().chain(first) {
            visit(OperandRole::Use, mask)?;
        }
        Ok(())
    }

    /// Visits operands in payload order, stopping at the first callback error.
    ///
    /// Definitions precede uses, with the same roles, duplicates and order as
    /// [`Self::visit_operands`]. No operand is visited after an error, including
    /// operands in variable-length native payloads. Ordinary operands allocate
    /// nothing; nested raises retain scratch proportional to their lane masks.
    ///
    /// # Errors
    ///
    /// Returns the first error returned by `f`, without inspecting later operands.
    #[inline]
    pub fn try_visit_operands<E, F>(&self, f: F) -> Result<(), E>
    where
        F: FnMut(OperandRole, SsaVarId) -> Result<(), E>,
    {
        self.try_visit_operands_mode::<false, _, _>(f)
    }

    /// Applies the shared operand policy with a compile-time definition-only view.
    ///
    /// The raise arm can stop after its definition without adding a separate
    /// variant check to every ordinary operation's definition query.
    #[allow(clippy::match_same_arms)] // Kept separate for clarity by operation category
    fn try_visit_operands_mode<const DEFINITIONS_ONLY: bool, E, F>(&self, mut f: F) -> Result<(), E>
    where
        F: FnMut(OperandRole, SsaVarId) -> Result<(), E>,
    {
        match self {
            Self::FloatRaise { dest, .. } => {
                f(OperandRole::Def, *dest)?;
                if !DEFINITIONS_ONLY {
                    self.try_visit_raised_uses(&mut f)?;
                }
            }
            Self::VectorSlice {
                dest,
                vector,
                offset,
                length,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *vector)?;
                f(OperandRole::Use, *offset)?;
                f(OperandRole::Use, *length)?;
            }
            Self::VectorInsertSlice {
                dest,
                vector,
                offset,
                value,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *vector)?;
                f(OperandRole::Use, *offset)?;
                f(OperandRole::Use, *value)?;
            }
            Self::VectorConcat {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
            }
            Self::PtrAdd {
                dest, base, index, ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *base)?;
                if let Some(index) = index {
                    f(OperandRole::Use, *index)?;
                }
            }
            Self::Const { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::ArgList { dest }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. }
            | Self::RuntimeValue { dest, .. } => f(OperandRole::Def, *dest)?,

            Self::MergeFlags {
                dest, base, update, ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *base)?;
                f(OperandRole::Use, *update)?;
            }

            Self::InsertFlag {
                dest, base, bit, ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *base)?;
                f(OperandRole::Use, *bit)?;
            }

            Self::CallClobber { outputs, .. } => {
                for output in outputs {
                    f(OperandRole::Def, *output)?;
                }
            }

            Self::Add {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::AddOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Sub {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::SubOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Mul {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::MulOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Div {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Rem {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::And {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Or {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Xor {
                dest,
                left,
                right,
                flags,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v)?;
                }
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
            }

            Self::Neg {
                dest,
                operand,
                flags,
                ..
            }
            | Self::Not {
                dest,
                operand,
                flags,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v)?;
                }
                f(OperandRole::Use, *operand)?;
            }

            Self::Shl {
                dest,
                value,
                amount,
                flags,
                ..
            }
            | Self::Shr {
                dest,
                value,
                amount,
                flags,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v)?;
                }
                f(OperandRole::Use, *value)?;
                f(OperandRole::Use, *amount)?;
            }

            Self::Rol {
                dest,
                value,
                amount,
                ..
            }
            | Self::Ror {
                dest,
                value,
                amount,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *value)?;
                f(OperandRole::Use, *amount)?;
            }

            Self::WideMul {
                low,
                high,
                left,
                right,
                ..
            } => {
                f(OperandRole::Def, *low)?;
                f(OperandRole::Def, *high)?;
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
            }

            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                ..
            } => {
                f(OperandRole::Def, *quotient)?;
                f(OperandRole::Def, *remainder)?;
                f(OperandRole::Use, *high)?;
                f(OperandRole::Use, *low)?;
                f(OperandRole::Use, *divisor)?;
            }

            Self::Ceq {
                dest, left, right, ..
            }
            | Self::Clt {
                dest, left, right, ..
            }
            | Self::Cgt {
                dest, left, right, ..
            }
            | Self::BoolAnd {
                dest, left, right, ..
            }
            | Self::BoolOr {
                dest, left, right, ..
            }
            | Self::BoolXor {
                dest, left, right, ..
            }
            | Self::VectorBinary {
                dest, left, right, ..
            }
            | Self::VectorCompare {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
            }

            Self::FloatCompareFlags {
                flags, left, right, ..
            } => {
                f(OperandRole::FlagsDef, *flags)?;
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
            }

            Self::BranchCmp { left, right, .. } => {
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
            }

            Self::BoolNot { dest, value, .. }
            | Self::Copy {
                dest, src: value, ..
            }
            | Self::IntConv {
                dest,
                operand: value,
                ..
            }
            | Self::IntToPtr {
                dest,
                operand: value,
                ..
            }
            | Self::PtrToInt {
                dest,
                operand: value,
                ..
            }
            | Self::IntToFloat {
                dest,
                operand: value,
                ..
            }
            | Self::FloatToInt {
                dest,
                operand: value,
                ..
            }
            | Self::FloatConv {
                dest,
                operand: value,
                ..
            }
            | Self::Bitcast {
                dest,
                operand: value,
                ..
            }
            | Self::Ckfinite {
                dest,
                operand: value,
                ..
            }
            | Self::FpClassify {
                dest,
                operand: value,
                ..
            }
            | Self::BSwap {
                dest, src: value, ..
            }
            | Self::BRev {
                dest, src: value, ..
            }
            | Self::BitScanForward {
                dest, src: value, ..
            }
            | Self::BitScanReverse {
                dest, src: value, ..
            }
            | Self::Popcount {
                dest, src: value, ..
            }
            | Self::Parity {
                dest, src: value, ..
            }
            | Self::LoadField {
                dest,
                object: value,
                ..
            }
            | Self::LoadFieldAddr {
                dest,
                object: value,
                ..
            }
            | Self::ArrayLength {
                dest, array: value, ..
            }
            | Self::LoadIndirect {
                dest, addr: value, ..
            }
            | Self::ArgNext {
                dest, list: value, ..
            }
            | Self::AtomicLoad {
                dest, addr: value, ..
            }
            | Self::VectorUnary { dest, value, .. }
            | Self::VectorSplat { dest, value, .. }
            | Self::VectorCast { dest, value, .. }
            | Self::VectorReinterpret { dest, value, .. }
            | Self::VectorLoad {
                dest, addr: value, ..
            }
            | Self::VectorBroadcastLoad {
                dest, addr: value, ..
            }
            | Self::VectorExtract {
                dest,
                vector: value,
                ..
            }
            | Self::VectorBitmask { dest, value, .. }
            | Self::NewArr {
                dest,
                length: value,
                ..
            }
            | Self::Box { dest, value, .. }
            | Self::LoadVirtFunctionPtr {
                dest,
                object: value,
                ..
            }
            | Self::LocalAlloc {
                dest, size: value, ..
            }
            | Self::CastClass {
                dest,
                object: value,
                ..
            }
            | Self::IsInst {
                dest,
                object: value,
                ..
            }
            | Self::Unbox {
                dest,
                object: value,
                ..
            }
            | Self::UnboxAny {
                dest,
                object: value,
                ..
            }
            | Self::LoadObj {
                dest,
                src_addr: value,
                ..
            }
            | Self::MakeTypedRef {
                dest,
                address: value,
                ..
            }
            | Self::TypedRefValue {
                dest,
                typed_ref: value,
                ..
            }
            | Self::TypedRefType {
                dest,
                typed_ref: value,
            }
            | Self::ReadFlags {
                dest, flags: value, ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *value)?;
            }

            Self::Branch {
                condition: value, ..
            }
            | Self::Switch { value, .. }
            | Self::StoreStaticField { value, .. }
            | Self::Pop { value }
            | Self::Throw { exception: value }
            | Self::EndFilter { result: value }
            | Self::InitObj {
                dest_addr: value, ..
            }
            | Self::BranchFlags { flags: value, .. } => f(OperandRole::Use, *value)?,

            Self::IndirectBranch { target, state, .. } => {
                f(OperandRole::Use, *target)?;
                for input in state.iter() {
                    f(OperandRole::Use, input.value)?;
                }
            }

            Self::LoadElement {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::LoadElementAddr {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::AtomicExchange {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicLockRmw {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicStoreConditional {
                status: dest,
                addr: a,
                value: b,
                ..
            }
            | Self::VectorInsert {
                dest,
                vector: a,
                value: b,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *a)?;
                f(OperandRole::Use, *b)?;
            }

            Self::StoreField {
                object: a,
                value: b,
                ..
            }
            | Self::StoreIndirect {
                addr: a, value: b, ..
            }
            | Self::AtomicStore {
                addr: a, value: b, ..
            }
            | Self::VectorStore {
                addr: a, value: b, ..
            }
            | Self::CopyObj {
                dest_addr: a,
                src_addr: b,
                ..
            }
            | Self::StoreObj {
                dest_addr: a,
                value: b,
                ..
            } => {
                f(OperandRole::Use, *a)?;
                f(OperandRole::Use, *b)?;
            }

            Self::Select {
                dest,
                condition: a,
                true_val: b,
                false_val: c,
                ..
            }
            | Self::VectorTernary {
                dest,
                first: a,
                second: b,
                third: c,
                ..
            }
            | Self::AtomicPairStoreConditional {
                status: dest,
                addr: a,
                first_value: b,
                second_value: c,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *a)?;
                f(OperandRole::Use, *b)?;
                f(OperandRole::Use, *c)?;
            }

            Self::StoreElement {
                array: a,
                index: b,
                value: c,
                ..
            }
            | Self::VectorMaskedStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::VectorPackStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::MoveBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                ..
            } => {
                f(OperandRole::Use, *a)?;
                f(OperandRole::Use, *b)?;
                f(OperandRole::Use, *c)?;
            }

            // A block operation that walks as a direction flag says reads the
            // flag last.
            Self::InitBlk {
                dest_addr: a,
                value: b,
                size: c,
                backward,
                ..
            }
            | Self::CopyBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                backward,
                ..
            } => {
                f(OperandRole::Use, *a)?;
                f(OperandRole::Use, *b)?;
                f(OperandRole::Use, *c)?;
                if let Some(flag) = backward {
                    f(OperandRole::Use, *flag)?;
                }
            }

            Self::ArgStep {
                dest, next, list, ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Def, *next)?;
                f(OperandRole::Use, *list)?;
            }

            Self::VectorScatter {
                base: a,
                indices: b,
                value: c,
                mask: d,
                ..
            } => {
                f(OperandRole::Use, *a)?;
                f(OperandRole::Use, *b)?;
                f(OperandRole::Use, *c)?;
                f(OperandRole::Use, *d)?;
            }

            Self::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                ..
            } => {
                f(OperandRole::Def, *old)?;
                if let Some(success_v) = success {
                    f(OperandRole::Def, *success_v)?;
                }
                f(OperandRole::Use, *addr)?;
                f(OperandRole::Use, *expected)?;
                f(OperandRole::Use, *desired)?;
            }

            Self::AtomicPairLoad {
                first,
                second,
                addr,
                ..
            } => {
                f(OperandRole::Def, *first)?;
                f(OperandRole::Def, *second)?;
                f(OperandRole::Use, *addr)?;
            }

            Self::AtomicCondRmw {
                old,
                addr,
                compare,
                value,
                ..
            } => {
                f(OperandRole::Def, *old)?;
                f(OperandRole::Use, *addr)?;
                f(OperandRole::Use, *compare)?;
                f(OperandRole::Use, *value)?;
            }

            Self::AtomicPairRmw {
                old_first,
                old_second,
                addr,
                value_first,
                value_second,
                ..
            } => {
                f(OperandRole::Def, *old_first)?;
                f(OperandRole::Def, *old_second)?;
                f(OperandRole::Use, *addr)?;
                f(OperandRole::Use, *value_first)?;
                f(OperandRole::Use, *value_second)?;
            }

            Self::AtomicCheckedRmw {
                old,
                status,
                addr,
                compare,
                value,
                rcw_mask,
                rcws_mask,
                ..
            } => {
                f(OperandRole::Def, *old)?;
                f(OperandRole::Def, *status)?;
                f(OperandRole::Use, *addr)?;
                if let Some(compare) = compare {
                    f(OperandRole::Use, *compare)?;
                }
                f(OperandRole::Use, *value)?;
                f(OperandRole::Use, *rcw_mask)?;
                if let Some(rcws_mask) = rcws_mask {
                    f(OperandRole::Use, *rcws_mask)?;
                }
            }

            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                ..
            } => {
                f(OperandRole::Def, *old_first)?;
                f(OperandRole::Def, *old_second)?;
                f(OperandRole::Use, *addr)?;
                f(OperandRole::Use, *expected_first)?;
                f(OperandRole::Use, *expected_second)?;
                f(OperandRole::Use, *desired_first)?;
                f(OperandRole::Use, *desired_second)?;
            }

            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorMaskedLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPack {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPackLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *value)?;
                f(OperandRole::Use, *mask)?;
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v)?;
                }
            }

            Self::VectorPredicatedBinary {
                dest,
                left,
                right,
                mask,
                passthrough,
                ..
            }
            | Self::VectorGather {
                dest,
                base: left,
                indices: right,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
                f(OperandRole::Use, *mask)?;
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v)?;
                }
            }

            Self::VectorPredicatedTernary {
                dest,
                first,
                second,
                third,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *first)?;
                f(OperandRole::Use, *second)?;
                f(OperandRole::Use, *third)?;
                f(OperandRole::Use, *mask)?;
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v)?;
                }
            }

            Self::VectorFaultingLoad {
                dest,
                fault,
                addr,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                if let Some(fault_v) = fault {
                    f(OperandRole::Def, *fault_v)?;
                }
                f(OperandRole::Use, *addr)?;
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v)?;
                }
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v)?;
                }
            }

            Self::VectorSegmentLoad {
                dests, base, mask, ..
            } => {
                for item in dests {
                    f(OperandRole::Def, *item)?;
                }
                f(OperandRole::Use, *base)?;
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v)?;
                }
            }

            Self::VectorSegmentStore {
                base, values, mask, ..
            } => {
                f(OperandRole::Use, *base)?;
                for item in values {
                    f(OperandRole::Use, *item)?;
                }
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v)?;
                }
            }

            Self::VectorShuffle {
                dest,
                left,
                right,
                selector,
            } => {
                f(OperandRole::Def, *dest)?;
                f(OperandRole::Use, *left)?;
                if let Some(right_v) = right {
                    f(OperandRole::Use, *right_v)?;
                }
                if let VectorShuffleSelector::Index(index) = selector {
                    f(OperandRole::Use, *index)?;
                }
            }

            Self::VectorReduce {
                dest,
                value,
                initial,
                ..
            } => {
                f(OperandRole::Def, *dest)?;
                // The initial value first, as it combines first.
                if let Some(initial) = initial {
                    f(OperandRole::Use, *initial)?;
                }
                f(OperandRole::Use, *value)?;
            }

            Self::SystemOp(data) => data.try_visit(&mut f)?,
            Self::Compute(data) => data.try_visit(&mut f)?,
            Self::Contract(data) => data.try_visit(&mut f)?,
            Self::VectorPredicateWhile(data) => data.try_visit(&mut f)?,
            Self::VectorExtendInLane(data) => data.try_visit(&mut f)?,
            Self::VectorElementCount(data) => data.try_visit(&mut f)?,
            Self::VectorStructLoadReplicate(data) => data.try_visit(&mut f)?,
            Self::VectorSveCompute(data) => data.try_visit(&mut f)?,
            Self::CompareBlk(data) => {
                let CompareBlkData {
                    remaining,
                    left_next,
                    right_next,
                    flags,
                    left,
                    right,
                    count,
                    flags_in,
                    backward,
                    ..
                } = data.as_ref();
                f(OperandRole::Def, *remaining)?;
                f(OperandRole::Def, *left_next)?;
                f(OperandRole::Def, *right_next)?;
                f(OperandRole::Def, *flags)?;
                f(OperandRole::Use, *left)?;
                f(OperandRole::Use, *right)?;
                f(OperandRole::Use, *count)?;
                if let Some(flags_in) = flags_in {
                    f(OperandRole::Use, *flags_in)?;
                }
                if let Some(flag) = backward {
                    f(OperandRole::Use, *flag)?;
                }
            }
            Self::ScanBlk {
                remaining,
                next,
                flags,
                address,
                value,
                count,
                flags_in,
                backward,
                ..
            } => {
                f(OperandRole::Def, *remaining)?;
                f(OperandRole::Def, *next)?;
                f(OperandRole::Def, *flags)?;
                f(OperandRole::Use, *address)?;
                f(OperandRole::Use, *value)?;
                f(OperandRole::Use, *count)?;
                if let Some(flags_in) = flags_in {
                    f(OperandRole::Use, *flags_in)?;
                }
                if let Some(flag) = backward {
                    f(OperandRole::Use, *flag)?;
                }
            }
            Self::LoadBlk {
                value,
                next,
                address,
                count,
                value_in,
                backward,
                ..
            } => {
                f(OperandRole::Def, *value)?;
                f(OperandRole::Def, *next)?;
                f(OperandRole::Use, *address)?;
                f(OperandRole::Use, *count)?;
                if let Some(value_in) = value_in {
                    f(OperandRole::Use, *value_in)?;
                }
                if let Some(flag) = backward {
                    f(OperandRole::Use, *flag)?;
                }
            }
            Self::FloatFunction(data) => data.try_visit(&mut f)?,
            Self::FpuControl(data) => data.try_visit(&mut f)?,

            Self::NewObj { dest, args, .. } => {
                f(OperandRole::Def, *dest)?;
                for item in args {
                    f(OperandRole::Use, *item)?;
                }
            }

            // The result and the values returned beside it, then an indirect
            // call's function pointer, then the ordinary arguments, implicit
            // inputs, then staged stack values.
            Self::Call(call) => {
                let CallData {
                    dest,
                    callee,
                    args,
                    staged,

                    implicit_inputs,
                    results,
                    ..
                } = &**call;
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, *dest_v)?;
                }
                for result in results {
                    f(OperandRole::Def, result.value)?;
                }
                if let Callee::Indirect { fptr, .. } = callee {
                    f(OperandRole::Use, *fptr)?;
                }
                for item in args {
                    f(OperandRole::Use, *item)?;
                }
                for input in implicit_inputs {
                    f(OperandRole::Use, input.value)?;
                }
                for item in staged {
                    f(OperandRole::Use, item.value)?;
                }
            }

            Self::Return {
                value,
                state,
                results,
                transfer,
            } => {
                if let Some(value_v) = value {
                    f(OperandRole::Use, *value_v)?;
                }
                for held in state {
                    f(OperandRole::Use, *held)?;
                }
                for result in results {
                    f(OperandRole::Use, result.value)?;
                }
                if let Some(transfer) = transfer {
                    if let Some(target) = transfer.target {
                        f(OperandRole::Use, target)?;
                    }
                    for input in &transfer.handover {
                        f(OperandRole::Use, input.value)?;
                    }
                }
            }

            Self::Phi { dest, operands } => {
                f(OperandRole::Def, *dest)?;
                for (_, item) in operands {
                    f(OperandRole::Use, *item)?;
                }
            }

            Self::Jump { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::Leave { .. }
            | Self::Nop
            | Self::Break(_)
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::Unreachable => {}
        }
        // The environment's rounding mode an operation rounds by is read
        // after its other operands, a run-time format just before it, so a
        // lowered row can leave the mode out by position.
        if !DEFINITIONS_ONLY && let Some(format) = self.format_source() {
            f(OperandRole::Use, format)?;
        }
        if !DEFINITIONS_ONLY && let Some(environment) = self.environment() {
            f(OperandRole::Use, environment)?;
        }
        Ok(())
    }

    /// Visits every SSA variable operand mutably together with its [`OperandRole`].
    ///
    /// The mutable counterpart of [`Self::visit_operands`], visiting the same
    /// operands in the same order. Substitution passes ([`Self::replace_uses`],
    /// [`Self::replace_def`]) rewrite variables in place through it.
    #[allow(clippy::match_same_arms)] // Kept separate for clarity by operation category
    pub fn visit_operands_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(OperandRole, &mut SsaVarId),
    {
        match self {
            // The operation's definitions follow the raise's, so that they
            // stay its destination however it is renamed.
            Self::FloatRaise {
                dest, op, lanes, ..
            } => {
                f(OperandRole::Def, dest);
                let raised = *dest;
                let mut operation = op.as_mut();
                let mut masks = Vec::new();
                loop {
                    match operation {
                        Self::FloatRaise {
                            dest, op, lanes, ..
                        } => {
                            *dest = raised;
                            if let Some(mask) = lanes.as_mut() {
                                masks.push(mask);
                            }
                            operation = op.as_mut();
                        }
                        leaf => {
                            let uses: &mut dyn FnMut(OperandRole, &mut SsaVarId) =
                                &mut |role, var| {
                                    if role == OperandRole::Use {
                                        f(role, var);
                                    } else {
                                        *var = raised;
                                    }
                                };
                            leaf.visit_operands_mut(uses);
                            break;
                        }
                    }
                }
                for mask in masks.into_iter().rev() {
                    f(OperandRole::Use, mask);
                }
                if let Some(lanes) = lanes {
                    f(OperandRole::Use, lanes);
                }
            }
            Self::VectorSlice {
                dest,
                vector,
                offset,
                length,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, vector);
                f(OperandRole::Use, offset);
                f(OperandRole::Use, length);
            }
            Self::VectorInsertSlice {
                dest,
                vector,
                offset,
                value,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, vector);
                f(OperandRole::Use, offset);
                f(OperandRole::Use, value);
            }
            Self::VectorConcat {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }
            Self::PtrAdd {
                dest, base, index, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, base);
                if let Some(index) = index {
                    f(OperandRole::Use, index);
                }
            }
            Self::Const { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::ArgList { dest }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. }
            | Self::RuntimeValue { dest, .. } => f(OperandRole::Def, dest),

            Self::MergeFlags {
                dest, base, update, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, base);
                f(OperandRole::Use, update);
            }

            Self::InsertFlag {
                dest, base, bit, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, base);
                f(OperandRole::Use, bit);
            }

            Self::CallClobber { outputs, .. } => {
                for output in outputs {
                    f(OperandRole::Def, output);
                }
            }

            Self::Add {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::AddOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Sub {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::SubOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Mul {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::MulOvf {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Div {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Rem {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::And {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Or {
                dest,
                left,
                right,
                flags,
                ..
            }
            | Self::Xor {
                dest,
                left,
                right,
                flags,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, flags_v);
                }
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::Neg {
                dest,
                operand,
                flags,
                ..
            }
            | Self::Not {
                dest,
                operand,
                flags,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, flags_v);
                }
                f(OperandRole::Use, operand);
            }

            Self::Shl {
                dest,
                value,
                amount,
                flags,
                ..
            }
            | Self::Shr {
                dest,
                value,
                amount,
                flags,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, flags_v);
                }
                f(OperandRole::Use, value);
                f(OperandRole::Use, amount);
            }

            Self::Rol {
                dest,
                value,
                amount,
                ..
            }
            | Self::Ror {
                dest,
                value,
                amount,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, value);
                f(OperandRole::Use, amount);
            }

            Self::WideMul {
                low,
                high,
                left,
                right,
                ..
            } => {
                f(OperandRole::Def, low);
                f(OperandRole::Def, high);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                ..
            } => {
                f(OperandRole::Def, quotient);
                f(OperandRole::Def, remainder);
                f(OperandRole::Use, high);
                f(OperandRole::Use, low);
                f(OperandRole::Use, divisor);
            }

            Self::Ceq {
                dest, left, right, ..
            }
            | Self::Clt {
                dest, left, right, ..
            }
            | Self::Cgt {
                dest, left, right, ..
            }
            | Self::BoolAnd {
                dest, left, right, ..
            }
            | Self::BoolOr {
                dest, left, right, ..
            }
            | Self::BoolXor {
                dest, left, right, ..
            }
            | Self::VectorBinary {
                dest, left, right, ..
            }
            | Self::VectorCompare {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::FloatCompareFlags {
                flags, left, right, ..
            } => {
                f(OperandRole::FlagsDef, flags);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::BranchCmp { left, right, .. } => {
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
            }

            Self::BoolNot { dest, value, .. }
            | Self::Copy {
                dest, src: value, ..
            }
            | Self::IntConv {
                dest,
                operand: value,
                ..
            }
            | Self::IntToPtr {
                dest,
                operand: value,
                ..
            }
            | Self::PtrToInt {
                dest,
                operand: value,
                ..
            }
            | Self::IntToFloat {
                dest,
                operand: value,
                ..
            }
            | Self::FloatToInt {
                dest,
                operand: value,
                ..
            }
            | Self::FloatConv {
                dest,
                operand: value,
                ..
            }
            | Self::Bitcast {
                dest,
                operand: value,
                ..
            }
            | Self::Ckfinite {
                dest,
                operand: value,
                ..
            }
            | Self::FpClassify {
                dest,
                operand: value,
                ..
            }
            | Self::BSwap {
                dest, src: value, ..
            }
            | Self::BRev {
                dest, src: value, ..
            }
            | Self::BitScanForward {
                dest, src: value, ..
            }
            | Self::BitScanReverse {
                dest, src: value, ..
            }
            | Self::Popcount {
                dest, src: value, ..
            }
            | Self::Parity {
                dest, src: value, ..
            }
            | Self::LoadField {
                dest,
                object: value,
                ..
            }
            | Self::LoadFieldAddr {
                dest,
                object: value,
                ..
            }
            | Self::ArrayLength {
                dest, array: value, ..
            }
            | Self::LoadIndirect {
                dest, addr: value, ..
            }
            | Self::ArgNext {
                dest, list: value, ..
            }
            | Self::AtomicLoad {
                dest, addr: value, ..
            }
            | Self::VectorUnary { dest, value, .. }
            | Self::VectorSplat { dest, value, .. }
            | Self::VectorCast { dest, value, .. }
            | Self::VectorReinterpret { dest, value, .. }
            | Self::VectorLoad {
                dest, addr: value, ..
            }
            | Self::VectorBroadcastLoad {
                dest, addr: value, ..
            }
            | Self::VectorExtract {
                dest,
                vector: value,
                ..
            }
            | Self::VectorBitmask { dest, value, .. }
            | Self::NewArr {
                dest,
                length: value,
                ..
            }
            | Self::Box { dest, value, .. }
            | Self::LoadVirtFunctionPtr {
                dest,
                object: value,
                ..
            }
            | Self::LocalAlloc {
                dest, size: value, ..
            }
            | Self::CastClass {
                dest,
                object: value,
                ..
            }
            | Self::IsInst {
                dest,
                object: value,
                ..
            }
            | Self::Unbox {
                dest,
                object: value,
                ..
            }
            | Self::UnboxAny {
                dest,
                object: value,
                ..
            }
            | Self::LoadObj {
                dest,
                src_addr: value,
                ..
            }
            | Self::MakeTypedRef {
                dest,
                address: value,
                ..
            }
            | Self::TypedRefValue {
                dest,
                typed_ref: value,
                ..
            }
            | Self::TypedRefType {
                dest,
                typed_ref: value,
            }
            | Self::ReadFlags {
                dest, flags: value, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, value);
            }

            Self::Branch {
                condition: value, ..
            }
            | Self::Switch { value, .. }
            | Self::StoreStaticField { value, .. }
            | Self::Pop { value }
            | Self::Throw { exception: value }
            | Self::EndFilter { result: value }
            | Self::InitObj {
                dest_addr: value, ..
            }
            | Self::BranchFlags { flags: value, .. } => f(OperandRole::Use, value),

            Self::IndirectBranch { target, state, .. } => {
                f(OperandRole::Use, target);
                for input in state.iter_mut() {
                    f(OperandRole::Use, &mut input.value);
                }
            }

            Self::LoadElement {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::LoadElementAddr {
                dest,
                array: a,
                index: b,
                ..
            }
            | Self::AtomicExchange {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicLockRmw {
                dest,
                addr: a,
                value: b,
                ..
            }
            | Self::AtomicStoreConditional {
                status: dest,
                addr: a,
                value: b,
                ..
            }
            | Self::VectorInsert {
                dest,
                vector: a,
                value: b,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
            }

            Self::StoreField {
                object: a,
                value: b,
                ..
            }
            | Self::StoreIndirect {
                addr: a, value: b, ..
            }
            | Self::AtomicStore {
                addr: a, value: b, ..
            }
            | Self::VectorStore {
                addr: a, value: b, ..
            }
            | Self::CopyObj {
                dest_addr: a,
                src_addr: b,
                ..
            }
            | Self::StoreObj {
                dest_addr: a,
                value: b,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
            }

            Self::Select {
                dest,
                condition: a,
                true_val: b,
                false_val: c,
                ..
            }
            | Self::VectorTernary {
                dest,
                first: a,
                second: b,
                third: c,
                ..
            }
            | Self::AtomicPairStoreConditional {
                status: dest,
                addr: a,
                first_value: b,
                second_value: c,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
            }

            Self::StoreElement {
                array: a,
                index: b,
                value: c,
                ..
            }
            | Self::VectorMaskedStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::VectorPackStore {
                addr: a,
                value: b,
                mask: c,
                ..
            }
            | Self::MoveBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
            }

            Self::InitBlk {
                dest_addr: a,
                value: b,
                size: c,
                backward,
                ..
            }
            | Self::CopyBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                backward,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
                if let Some(flag) = backward {
                    f(OperandRole::Use, flag);
                }
            }

            Self::ArgStep {
                dest, next, list, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Def, next);
                f(OperandRole::Use, list);
            }

            Self::VectorScatter {
                base: a,
                indices: b,
                value: c,
                mask: d,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
                f(OperandRole::Use, d);
            }

            Self::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                ..
            } => {
                f(OperandRole::Def, old);
                if let Some(success_v) = success {
                    f(OperandRole::Def, success_v);
                }
                f(OperandRole::Use, addr);
                f(OperandRole::Use, expected);
                f(OperandRole::Use, desired);
            }

            Self::AtomicPairLoad {
                first,
                second,
                addr,
                ..
            } => {
                f(OperandRole::Def, first);
                f(OperandRole::Def, second);
                f(OperandRole::Use, addr);
            }

            Self::AtomicCondRmw {
                old,
                addr,
                compare,
                value,
                ..
            } => {
                f(OperandRole::Def, old);
                f(OperandRole::Use, addr);
                f(OperandRole::Use, compare);
                f(OperandRole::Use, value);
            }

            Self::AtomicPairRmw {
                old_first,
                old_second,
                addr,
                value_first,
                value_second,
                ..
            } => {
                f(OperandRole::Def, old_first);
                f(OperandRole::Def, old_second);
                f(OperandRole::Use, addr);
                f(OperandRole::Use, value_first);
                f(OperandRole::Use, value_second);
            }

            Self::AtomicCheckedRmw {
                old,
                status,
                addr,
                compare,
                value,
                rcw_mask,
                rcws_mask,
                ..
            } => {
                f(OperandRole::Def, old);
                f(OperandRole::Def, status);
                f(OperandRole::Use, addr);
                if let Some(compare) = compare {
                    f(OperandRole::Use, compare);
                }
                f(OperandRole::Use, value);
                f(OperandRole::Use, rcw_mask);
                if let Some(rcws_mask) = rcws_mask {
                    f(OperandRole::Use, rcws_mask);
                }
            }

            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                addr,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                ..
            } => {
                f(OperandRole::Def, old_first);
                f(OperandRole::Def, old_second);
                f(OperandRole::Use, addr);
                f(OperandRole::Use, expected_first);
                f(OperandRole::Use, expected_second);
                f(OperandRole::Use, desired_first);
                f(OperandRole::Use, desired_second);
            }

            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorMaskedLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPack {
                dest,
                value,
                mask,
                passthrough,
                ..
            }
            | Self::VectorPackLoad {
                dest,
                addr: value,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, value);
                f(OperandRole::Use, mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorPredicatedBinary {
                dest,
                left,
                right,
                mask,
                passthrough,
                ..
            }
            | Self::VectorGather {
                dest,
                base: left,
                indices: right,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
                f(OperandRole::Use, mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorPredicatedTernary {
                dest,
                first,
                second,
                third,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, first);
                f(OperandRole::Use, second);
                f(OperandRole::Use, third);
                f(OperandRole::Use, mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorFaultingLoad {
                dest,
                fault,
                addr,
                mask,
                passthrough,
                ..
            } => {
                f(OperandRole::Def, dest);
                if let Some(fault_v) = fault {
                    f(OperandRole::Def, fault_v);
                }
                f(OperandRole::Use, addr);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, mask_v);
                }
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, passthrough_v);
                }
            }

            Self::VectorSegmentLoad {
                dests, base, mask, ..
            } => {
                for item in dests {
                    f(OperandRole::Def, item);
                }
                f(OperandRole::Use, base);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, mask_v);
                }
            }

            Self::VectorSegmentStore {
                base, values, mask, ..
            } => {
                f(OperandRole::Use, base);
                for item in values {
                    f(OperandRole::Use, item);
                }
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, mask_v);
                }
            }

            Self::VectorShuffle {
                dest,
                left,
                right,
                selector,
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                if let Some(right_v) = right {
                    f(OperandRole::Use, right_v);
                }
                if let VectorShuffleSelector::Index(index) = selector {
                    f(OperandRole::Use, index);
                }
            }

            Self::VectorReduce {
                dest,
                value,
                initial,
                ..
            } => {
                f(OperandRole::Def, dest);
                // The initial value first, as it combines first.
                if let Some(initial) = initial {
                    f(OperandRole::Use, initial);
                }
                f(OperandRole::Use, value);
            }

            Self::SystemOp(data) => data.visit_mut(&mut f),
            Self::Compute(data) => data.visit_mut(&mut f),
            Self::Contract(data) => data.visit_mut(&mut f),
            Self::VectorPredicateWhile(data) => data.visit_mut(&mut f),
            Self::VectorExtendInLane(data) => data.visit_mut(&mut f),
            Self::VectorElementCount(data) => data.visit_mut(&mut f),
            Self::VectorStructLoadReplicate(data) => data.visit_mut(&mut f),
            Self::VectorSveCompute(data) => data.visit_mut(&mut f),
            Self::CompareBlk(data) => {
                let CompareBlkData {
                    remaining,
                    left_next,
                    right_next,
                    flags,
                    left,
                    right,
                    count,
                    flags_in,
                    backward,
                    ..
                } = data.as_mut();
                f(OperandRole::Def, remaining);
                f(OperandRole::Def, left_next);
                f(OperandRole::Def, right_next);
                f(OperandRole::Def, flags);
                f(OperandRole::Use, left);
                f(OperandRole::Use, right);
                f(OperandRole::Use, count);
                if let Some(flags_in) = flags_in {
                    f(OperandRole::Use, flags_in);
                }
                if let Some(flag) = backward {
                    f(OperandRole::Use, flag);
                }
            }
            Self::ScanBlk {
                remaining,
                next,
                flags,
                address,
                value,
                count,
                flags_in,
                backward,
                ..
            } => {
                f(OperandRole::Def, remaining);
                f(OperandRole::Def, next);
                f(OperandRole::Def, flags);
                f(OperandRole::Use, address);
                f(OperandRole::Use, value);
                f(OperandRole::Use, count);
                if let Some(flags_in) = flags_in {
                    f(OperandRole::Use, flags_in);
                }
                if let Some(flag) = backward {
                    f(OperandRole::Use, flag);
                }
            }
            Self::LoadBlk {
                value,
                next,
                address,
                count,
                value_in,
                backward,
                ..
            } => {
                f(OperandRole::Def, value);
                f(OperandRole::Def, next);
                f(OperandRole::Use, address);
                f(OperandRole::Use, count);
                if let Some(value_in) = value_in {
                    f(OperandRole::Use, value_in);
                }
                if let Some(flag) = backward {
                    f(OperandRole::Use, flag);
                }
            }
            Self::FloatFunction(data) => data.visit_mut(&mut f),
            Self::FpuControl(data) => data.visit_mut(&mut f),

            Self::NewObj { dest, args, .. } => {
                f(OperandRole::Def, dest);
                for item in args {
                    f(OperandRole::Use, item);
                }
            }

            // The result and the values returned beside it, then an indirect
            // call's function pointer, then the ordinary arguments, implicit
            // inputs, then staged stack values.
            Self::Call(call) => {
                let CallData {
                    dest,
                    callee,
                    args,
                    staged,

                    implicit_inputs,
                    results,
                    ..
                } = &mut **call;
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, dest_v);
                }
                for result in results {
                    f(OperandRole::Def, &mut result.value);
                }
                if let Callee::Indirect { fptr, .. } = callee {
                    f(OperandRole::Use, fptr);
                }
                for item in args {
                    f(OperandRole::Use, item);
                }
                for input in implicit_inputs {
                    f(OperandRole::Use, &mut input.value);
                }
                for item in staged {
                    f(OperandRole::Use, &mut item.value);
                }
            }

            Self::Return {
                value,
                state,
                results,
                transfer,
            } => {
                if let Some(value_v) = value {
                    f(OperandRole::Use, value_v);
                }
                for held in state {
                    f(OperandRole::Use, held);
                }
                for result in results {
                    f(OperandRole::Use, &mut result.value);
                }
                if let Some(transfer) = transfer {
                    if let Some(target) = &mut transfer.target {
                        f(OperandRole::Use, target);
                    }
                    for input in &mut transfer.handover {
                        f(OperandRole::Use, &mut input.value);
                    }
                }
            }

            Self::Phi { dest, operands } => {
                f(OperandRole::Def, dest);
                for (_, item) in operands {
                    f(OperandRole::Use, item);
                }
            }

            Self::Jump { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::Leave { .. }
            | Self::Nop
            | Self::Break(_)
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::Unreachable => {}
        }
        if let Self::VectorCast {
            format: Some(format),
            ..
        } = self
        {
            f(OperandRole::Use, format);
        }
        if let Some(Some(environment)) = self.environment_mut() {
            f(OperandRole::Use, environment);
        }
    }

    /// Calls `f` for every variable used by this operation.
    pub fn for_each_use<F>(&self, mut f: F)
    where
        F: FnMut(SsaVarId),
    {
        match self.try_for_each_use(|var| {
            f(var);
            Ok::<(), Infallible>(())
        }) {
            Ok(()) => (),
            Err(error) => match error {},
        }
    }

    /// Visits uses in payload order, stopping at the first callback error.
    ///
    /// Preserves duplicates and allocates no operand list. Definitions are
    /// skipped; use [`Self::try_visit_operands`] to account for every operand.
    ///
    /// # Errors
    ///
    /// Returns the first error returned by `f`, without visiting later uses.
    pub fn try_for_each_use<E, F>(&self, mut f: F) -> Result<(), E>
    where
        F: FnMut(SsaVarId) -> Result<(), E>,
    {
        self.try_visit_operands(|role, var| {
            if matches!(role, OperandRole::Use) {
                f(var)
            } else {
                Ok(())
            }
        })
    }

    /// Visits definitions in payload order without allocating a definition list.
    ///
    /// Includes primary, secondary and flag definitions. Stops before scanning uses.
    pub fn for_each_def(&self, mut visit: impl FnMut(SsaVarId)) {
        let result = self.try_for_each_def(|var| {
            visit(var);
            Ok::<(), Infallible>(())
        });
        match result {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }

    /// Visits definitions in payload order, stopping before uses or on error.
    ///
    /// Visits primary, secondary and flag definitions without allocating a
    /// definition list. Definitions precede uses in the shared operand policy.
    ///
    /// # Errors
    ///
    /// Returns the first error returned by `f`, without visiting later definitions.
    pub fn try_for_each_def<E, F>(&self, mut f: F) -> Result<(), E>
    where
        F: FnMut(SsaVarId) -> Result<(), E>,
    {
        match self.try_visit_operands_mode::<true, _, _>(|role, var| {
            if matches!(role, OperandRole::Use) {
                Err(None)
            } else {
                f(var).map_err(Some)
            }
        }) {
            Ok(()) | Err(None) => Ok(()),
            Err(Some(error)) => Err(error),
        }
    }

    /// Returns all variables used by this operation.
    #[must_use]
    pub fn uses(&self) -> Vec<SsaVarId> {
        let mut uses = Vec::new();
        self.for_each_use(|var| uses.push(var));
        uses
    }

    /// Where the mask governing the lanes this operation writes or accesses
    /// is among its uses, and the passthrough the lanes it leaves inactive
    /// keep: the predicated lane operations, the masked, faulting, segment
    /// and pack accesses, gathers and scatters. A mask an operation computes
    /// with is data and governs nothing (`Kernel::MaskBinary`), and an operation
    /// over lanes under lane control states its controls in its computation.
    ///
    /// # Returns
    ///
    /// The positions, or `None` for an operation no mask governs.
    #[must_use]
    pub fn governing_uses(&self) -> Option<GoverningUses> {
        // The mask after `before` uses, the passthrough after it.
        let after = |before: usize, passthrough: &Option<SsaVarId>| GoverningUses {
            mask: before,
            passthrough: passthrough.map(|_| before.saturating_add(1)),
        };
        Some(match self {
            Self::VectorPredicatedUnary { passthrough, .. }
            | Self::VectorMaskedLoad { passthrough, .. }
            | Self::VectorPack { passthrough, .. }
            | Self::VectorPackLoad { passthrough, .. } => after(1, passthrough),
            Self::VectorPredicatedBinary { passthrough, .. }
            | Self::VectorGather { passthrough, .. } => after(2, passthrough),
            Self::VectorPredicatedTernary { passthrough, .. } => after(3, passthrough),
            Self::VectorFaultingLoad {
                mask: Some(_),
                passthrough,
                ..
            } => after(1, passthrough),
            Self::VectorSegmentLoad { mask: Some(_), .. } => after(1, &None),
            Self::VectorMaskedStore { .. } | Self::VectorPackStore { .. } => after(2, &None),
            Self::VectorScatter { .. } => after(3, &None),
            Self::VectorSegmentStore {
                values,
                mask: Some(_),
                ..
            } => after(values.len().saturating_add(1), &None),
            _ => return None,
        })
    }

    /// Returns the first variable used by this operation, in operand order.
    ///
    /// Equivalent to `self.uses().first().copied()` without the `Vec`: like
    /// [`use_count`](Self::use_count) and [`uses_var`](Self::uses_var) it walks
    /// [`for_each_use`](Self::for_each_use) directly, so a caller that wants
    /// one operand - the type of the value an identity is being checked
    /// against, say - pays no allocation for it.
    ///
    /// # Returns
    ///
    /// The operation's first use, or `None` when it reads no variables.
    #[must_use]
    pub fn first_use(&self) -> Option<SsaVarId> {
        self.try_for_each_use(Err).err()
    }

    /// Returns the number of variables used by this operation.
    #[must_use]
    pub fn use_count(&self) -> usize {
        let mut count = 0usize;
        self.for_each_use(|_| count = count.saturating_add(1));
        count
    }

    /// Returns `true` if this operation uses (reads) the given variable.
    ///
    /// This is an allocation-free membership test that short-circuits on the
    /// first match, preferable to `self.uses().contains(&var)` in hot paths.
    #[must_use]
    pub fn uses_var(&self, var: SsaVarId) -> bool {
        self.try_for_each_use(|used| if used == var { Err(()) } else { Ok(()) })
            .is_err()
    }

    /// Replaces all uses of `old_var` with `new_var` in this operation.
    ///
    /// This is used for copy propagation and other variable substitution transformations.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable to replace.
    /// * `new_var` - The variable to use instead.
    ///
    /// # Returns
    ///
    /// The number of uses that were replaced.
    pub fn replace_uses(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> usize {
        let mut count: usize = 0;
        self.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Use) && *var == old_var {
                *var = new_var;
                count = count.saturating_add(1);
            }
        });
        count
    }

    /// Rewrites every `Use` operand via `lookup`, in a single pass.
    ///
    /// The batch counterpart to [`Self::replace_uses`]: a substitution with many
    /// `(old → new)` pairs visits each operand **once** and consults `lookup`,
    /// instead of calling `replace_uses` per pair (which re-walks every operand
    /// for every pair - `O(operands × pairs)`). `lookup` returns the replacement
    /// for a used variable, or `None` to leave it unchanged. Returns the number
    /// of operands rewritten.
    pub fn replace_uses_with<F>(&mut self, mut lookup: F) -> usize
    where
        F: FnMut(SsaVarId) -> Option<SsaVarId>,
    {
        let mut count: usize = 0;
        self.visit_operands_mut(|role, var| {
            if matches!(role, OperandRole::Use)
                && let Some(new_var) = lookup(*var)
            {
                *var = new_var;
                count = count.saturating_add(1);
            }
        });
        count
    }

    /// Creates a clone of this operation with all variable IDs remapped.
    ///
    /// This is used for block duplication where all variable references
    /// (both destinations and uses) need to be updated to fresh IDs.
    ///
    /// # Arguments
    ///
    /// * `remap` - A function that maps old variable IDs to new ones.
    ///   If the function returns `None`, the original ID is kept.
    ///
    /// # Returns
    ///
    /// A new `SsaOp` with all variable IDs remapped.
    #[must_use]
    pub fn remap_variables<F>(&self, remap: F) -> Self
    where
        F: Fn(SsaVarId) -> Option<SsaVarId>,
    {
        // Operand remapping is delegated to the single exhaustive operand
        // visitor so this method can never silently drop a variable when a
        // new op variant is added. `visit_operands_mut` walks every
        // Def/FlagsDef/Use; block targets are remapped separately by
        // `remap_branch_targets`, so they are intentionally left untouched.
        let mut out = self.clone();
        out.visit_operands_mut(|_role, var| {
            if let Some(new) = remap(*var) {
                *var = new;
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::ops::{
            AccessQualifiers, VectorBinaryKind, VectorElement, VectorFaultMode, VectorMaskMode,
            VectorPackKind, VectorSegmentLayout, VectorTernaryKind, VectorUnaryKind,
        },
        testing::{MockTarget, MockType},
    };

    /// The operands name the operation's width only when they agree on it.
    ///
    /// `int32 + native int` is a native int in CIL, so a pair of different
    /// widths - or one whose width is unknown - names no single width.
    #[test]
    fn operands_name_a_width_only_when_they_agree() {
        let (left, right) = (SsaVarId::from_index(0), SsaVarId::from_index(1));
        let add: SsaOp<MockTarget> = SsaOp::Add {
            dest: SsaVarId::from_index(2),
            left,
            right,
            flags: None,
        };
        let widths = |narrow: Option<u32>, other: Option<u32>| {
            move |var: SsaVarId| if var == left { narrow } else { other }
        };
        assert_eq!(add.width_operand(widths(Some(32), Some(32))), Some(left));
        assert_eq!(add.width_operand(widths(Some(32), Some(64))), None);
        assert_eq!(add.width_operand(widths(Some(32), None)), None);
        assert_eq!(add.width_operand(widths(None, Some(32))), None);

        // A shift computes in its value's width, whatever the count's is.
        let shift: SsaOp<MockTarget> = SsaOp::Shl {
            dest: SsaVarId::from_index(2),
            value: left,
            amount: right,
            flags: None,
        };
        assert_eq!(shift.width_operand(widths(Some(8), Some(32))), Some(left));
    }

    /// **The mask and the passthrough are found among the uses.** For every
    /// operation a mask governs, the use at the stated mask position is the
    /// operation's mask and the one at the passthrough position its
    /// passthrough, with and without one; an operation computing with a mask,
    /// and a faulting load no mask governs, state none.
    #[test]
    fn the_governing_mask_and_passthrough_are_found_among_the_uses() {
        let v = SsaVarId::from_index;
        let (mask, passthrough) = (v(90), v(91));
        let element = VectorElement::default();
        let merge = VectorMaskMode::Merge;
        let access = AccessQualifiers::default();
        let ty = MockType::V4I32;
        let ops = |passthrough: Option<SsaVarId>| -> Vec<SsaOp<MockTarget>> {
            vec![
                SsaOp::VectorPredicatedUnary {
                    dest: v(0),
                    value: v(1),
                    mask,
                    passthrough,
                    kind: VectorUnaryKind::Neg,
                    element,
                    mode: merge,
                    environment: None,
                },
                SsaOp::VectorPredicatedBinary {
                    dest: v(0),
                    left: v(1),
                    right: v(2),
                    mask,
                    passthrough,
                    kind: VectorBinaryKind::Add,
                    element,
                    mode: merge,
                    environment: None,
                },
                SsaOp::VectorPredicatedTernary {
                    dest: v(0),
                    first: v(1),
                    second: v(2),
                    third: v(3),
                    mask,
                    passthrough,
                    kind: VectorTernaryKind::Fma,
                    element,
                    mode: merge,
                    environment: None,
                },
                SsaOp::VectorMaskedLoad {
                    dest: v(0),
                    addr: v(1),
                    mask,
                    passthrough,
                    vector_type: ty,
                    mode: merge,
                    access,
                },
                SsaOp::VectorGather {
                    dest: v(0),
                    base: v(1),
                    indices: v(2),
                    mask,
                    passthrough,
                    vector_type: ty,
                    mode: merge,
                    access,
                },
                SsaOp::VectorFaultingLoad {
                    dest: v(0),
                    fault: Some(v(5)),
                    addr: v(1),
                    mask: Some(mask),
                    passthrough,
                    vector_type: ty,
                    fault_mode: VectorFaultMode::FirstFault,
                    mask_mode: merge,
                    access,
                },
                SsaOp::VectorPack {
                    dest: v(0),
                    value: v(1),
                    mask,
                    passthrough,
                    vector_type: ty,
                    element_bits: 32,
                    kind: VectorPackKind::Compress,
                    mode: merge,
                },
                SsaOp::VectorPackLoad {
                    dest: v(0),
                    addr: v(1),
                    mask,
                    passthrough,
                    vector_type: ty,
                    element_bits: 32,
                    kind: VectorPackKind::Expand,
                    mode: merge,
                    access,
                },
            ]
        };
        for passthrough in [Some(passthrough), None] {
            for op in ops(passthrough) {
                let uses = op.uses();
                let governing = op.governing_uses().expect("a mask governs it");
                assert_eq!(uses.get(governing.mask), Some(&mask), "{op:?}");
                assert_eq!(
                    governing.passthrough.and_then(|at| uses.get(at)).copied(),
                    passthrough,
                    "{op:?}"
                );
            }
        }
        let stores: Vec<SsaOp<MockTarget>> = vec![
            SsaOp::VectorMaskedStore {
                addr: v(1),
                value: v(2),
                mask,
                vector_type: ty,
                access,
            },
            SsaOp::VectorScatter {
                base: v(1),
                indices: v(2),
                value: v(3),
                mask,
                vector_type: ty,
                access,
            },
            SsaOp::VectorPackStore {
                addr: v(1),
                value: v(2),
                mask,
                vector_type: ty,
                element_bits: 32,
                kind: VectorPackKind::Compress,
                access,
            },
            SsaOp::VectorSegmentStore {
                base: v(1),
                values: vec![v(2), v(3), v(4)],
                mask: Some(mask),
                vector_type: ty,
                segments: 3,
                layout: VectorSegmentLayout::Interleaved,
                access,
            },
            SsaOp::VectorSegmentLoad {
                dests: vec![v(2), v(3)],
                base: v(1),
                mask: Some(mask),
                vector_type: ty,
                segments: 2,
                layout: VectorSegmentLayout::Interleaved,
                access,
            },
        ];
        for op in stores {
            let governing = op.governing_uses().expect("a mask governs it");
            assert_eq!(op.uses().get(governing.mask), Some(&mask), "{op:?}");
            assert_eq!(governing.passthrough, None, "{op:?}");
        }
        let unmasked: SsaOp<MockTarget> = SsaOp::VectorFaultingLoad {
            dest: v(0),
            fault: None,
            addr: v(1),
            mask: None,
            passthrough: None,
            vector_type: ty,
            fault_mode: VectorFaultMode::Normal,
            mask_mode: merge,
            access,
        };
        assert_eq!(unmasked.governing_uses(), None);
        let add: SsaOp<MockTarget> = SsaOp::Add {
            dest: v(0),
            left: v(1),
            right: v(2),
            flags: None,
        };
        assert_eq!(add.governing_uses(), None);
    }
}
