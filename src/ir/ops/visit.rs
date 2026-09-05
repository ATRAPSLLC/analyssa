//! Operand and definition access for [`SsaOp`] — reading, iterating, and
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

use crate::{
    ir::{
        ops::{def::SsaOp, kinds::OperandRole, operands::KindedOperands},
        variable::SsaVarId,
    },
    target::Target,
};

/// Iterator over the variables an SSA operation defines, in the order
/// [`SsaOp::visit_operands`] reports them.
///
/// Owns its contents. [`SsaOp::defs`] collects the `Def` and `FlagsDef`
/// operands of one walk over the operation, so this iterator cannot disagree
/// with that walk — or with [`SsaOp::dest`], [`SsaOp::flags_dest`] and
/// [`SsaOp::replace_def`], which read the same walk — about what the
/// definitions are, and it does not borrow the operation it came from.
///
/// Two definitions live in named slots and any further ones spill to a `Vec`.
/// No operation in the crate defines more than two variables, so the spill
/// stays empty — and therefore unallocated — on every `defs()` call the crate
/// itself makes; a host-built operation with a longer output list allocates
/// once.
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
    /// Definitions past the second, in reverse so that iteration pops. Empty
    /// for every operation the crate defines, and an empty `Vec` does not
    /// allocate.
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
    /// Kept out of line: no operation in the crate reaches it, and an
    /// allocation inlined into the hot path is paid for by every caller that
    /// never spills.
    #[cold]
    #[inline(never)]
    fn push_rest(&mut self, var: SsaVarId) {
        // Reversed on the way in so that iteration is a `pop`, which needs no
        // cursor of its own.
        self.rest.insert(0, var);
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

impl<T: Target> SsaOp<T> {
    /// Returns the destination variable if this operation produces one.
    #[must_use]
    pub fn dest(&self) -> Option<SsaVarId> {
        let mut dest = None;
        self.visit_operands(|role, var| {
            if dest.is_none() && matches!(role, OperandRole::Def) {
                dest = Some(var);
            }
        });
        dest
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
        self.visit_operands(|role, var| {
            if matches!(role, OperandRole::Def | OperandRole::FlagsDef) {
                defs.push(var);
            }
        });
        defs
    }

    /// Replaces a definition variable without touching operand uses.
    ///
    /// Replaces every definition equal to `old_var` — the primary
    /// destination, secondary outputs (high halves, status/fault outputs,
    /// native output lists), and flag outputs. Returns `true` when at least
    /// one definition was changed. This is used by SSA renaming, where
    /// definitions and uses have different scoping rules.
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

    /// Returns the flags destination if this operation defines flags.
    pub fn flags_dest(&self) -> Option<SsaVarId> {
        let mut flags = None;
        self.visit_operands(|role, var| {
            if flags.is_none() && matches!(role, OperandRole::FlagsDef) {
                flags = Some(var);
            }
        });
        flags
    }

    /// Visits every SSA variable operand together with its [`OperandRole`].
    ///
    /// Operands are visited in payload order: definitions first (the primary
    /// destination, then secondary and flag outputs), then uses in payload
    /// order. This is the single source of truth for operand traversal —
    /// [`Self::dest`], [`Self::flags_dest`], [`Self::for_each_use`],
    /// [`Self::replace_uses`], and [`Self::replace_def`] are expressed over
    /// it.
    #[allow(clippy::match_same_arms)] // Kept separate for clarity by operation category
    pub fn visit_operands<F>(&self, mut f: F)
    where
        F: FnMut(OperandRole, SsaVarId),
    {
        match self {
            Self::PtrAdd {
                dest, base, index, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *base);
                if let Some(index) = index {
                    f(OperandRole::Use, *index);
                }
            }
            Self::Const { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. } => f(OperandRole::Def, *dest),

            Self::ComputeFlags { dest, inputs } => {
                f(OperandRole::Def, *dest);
                for input in inputs {
                    f(OperandRole::Use, *input);
                }
            }

            Self::CallClobber { outputs } => {
                for output in outputs {
                    f(OperandRole::Def, *output);
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
                f(OperandRole::Def, *dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v);
                }
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
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
                f(OperandRole::Def, *dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v);
                }
                f(OperandRole::Use, *operand);
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
                f(OperandRole::Def, *dest);
                if let Some(flags_v) = flags {
                    f(OperandRole::FlagsDef, *flags_v);
                }
                f(OperandRole::Use, *value);
                f(OperandRole::Use, *amount);
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
            }
            | Self::Rcl {
                dest,
                value,
                amount,
                ..
            }
            | Self::Rcr {
                dest,
                value,
                amount,
                ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *value);
                f(OperandRole::Use, *amount);
            }

            Self::WideMul {
                low,
                high,
                left,
                right,
                ..
            } => {
                f(OperandRole::Def, *low);
                f(OperandRole::Def, *high);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                ..
            } => {
                f(OperandRole::Def, *quotient);
                f(OperandRole::Def, *remainder);
                f(OperandRole::Use, *high);
                f(OperandRole::Use, *low);
                f(OperandRole::Use, *divisor);
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
            }
            | Self::VectorMaskBinary {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::FloatCompareFlags {
                flags, left, right, ..
            } => {
                f(OperandRole::FlagsDef, *flags);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
            }

            Self::BranchCmp { left, right, .. } => {
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
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
            | Self::VectorMaskUnary {
                dest, mask: value, ..
            }
            | Self::VectorReduce { dest, value, .. }
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
            | Self::ReadFlags {
                dest, flags: value, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *value);
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
            | Self::IndirectBranch { target: value, .. }
            | Self::BranchFlags { flags: value, .. } => f(OperandRole::Use, *value),

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
            | Self::AtomicRmw {
                dest,
                addr: a,
                value: b,
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
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
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
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
            }

            Self::Select {
                dest,
                condition: a,
                true_val: b,
                false_val: c,
                ..
            }
            | Self::CmpXchg {
                dest,
                addr: a,
                expected: b,
                desired: c,
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
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
                f(OperandRole::Use, *c);
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
            | Self::InitBlk {
                dest_addr: a,
                value: b,
                size: c,
                ..
            }
            | Self::CopyBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                ..
            } => {
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
                f(OperandRole::Use, *c);
            }

            Self::VectorScatter {
                base: a,
                indices: b,
                value: c,
                mask: d,
                ..
            } => {
                f(OperandRole::Use, *a);
                f(OperandRole::Use, *b);
                f(OperandRole::Use, *c);
                f(OperandRole::Use, *d);
            }

            Self::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                ..
            } => {
                f(OperandRole::Def, *old);
                if let Some(success_v) = success {
                    f(OperandRole::Def, *success_v);
                }
                f(OperandRole::Use, *addr);
                f(OperandRole::Use, *expected);
                f(OperandRole::Use, *desired);
            }

            Self::AtomicPairLoad {
                first,
                second,
                addr,
                ..
            } => {
                f(OperandRole::Def, *first);
                f(OperandRole::Def, *second);
                f(OperandRole::Use, *addr);
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
                f(OperandRole::Def, *old_first);
                f(OperandRole::Def, *old_second);
                f(OperandRole::Use, *addr);
                f(OperandRole::Use, *expected_first);
                f(OperandRole::Use, *expected_second);
                f(OperandRole::Use, *desired_first);
                f(OperandRole::Use, *desired_second);
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
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *value);
                f(OperandRole::Use, *mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
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
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *left);
                f(OperandRole::Use, *right);
                f(OperandRole::Use, *mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
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
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *first);
                f(OperandRole::Use, *second);
                f(OperandRole::Use, *third);
                f(OperandRole::Use, *mask);
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
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
                f(OperandRole::Def, *dest);
                if let Some(fault_v) = fault {
                    f(OperandRole::Def, *fault_v);
                }
                f(OperandRole::Use, *addr);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v);
                }
                if let Some(passthrough_v) = passthrough {
                    f(OperandRole::Use, *passthrough_v);
                }
            }

            Self::VectorSegmentLoad {
                dests, base, mask, ..
            } => {
                for item in dests {
                    f(OperandRole::Def, *item);
                }
                f(OperandRole::Use, *base);
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v);
                }
            }

            Self::VectorSegmentStore {
                base, values, mask, ..
            } => {
                f(OperandRole::Use, *base);
                for item in values {
                    f(OperandRole::Use, *item);
                }
                if let Some(mask_v) = mask {
                    f(OperandRole::Use, *mask_v);
                }
            }

            Self::VectorShuffle {
                dest, left, right, ..
            } => {
                f(OperandRole::Def, *dest);
                f(OperandRole::Use, *left);
                if let Some(right_v) = right {
                    f(OperandRole::Use, *right_v);
                }
            }

            Self::NativeOpaque(data) => data.visit(&mut f),
            Self::SystemOp(data) => data.visit(&mut f),
            Self::ComputeOp(data) => data.visit(&mut f),
            Self::BcdAdjust(data) => data.visit(&mut f),
            Self::VectorCrypto(data) => data.visit(&mut f),
            Self::TileOp(data) => data.visit(&mut f),
            Self::VectorPermute(data) => data.visit(&mut f),
            Self::VectorMultiplyAdd(data) => data.visit(&mut f),
            Self::VectorPackNarrow(data) => data.visit(&mut f),
            Self::VectorNarrowSaturate(data) => data.visit(&mut f),
            Self::VectorPredicateWhile(data) => data.visit(&mut f),
            Self::VectorPredicateBreak(data) => data.visit(&mut f),
            Self::VectorComplexAdd(data) => data.visit(&mut f),
            Self::VectorCountAdjust(data) => data.visit(&mut f),
            Self::VectorExtendInLane(data) => data.visit(&mut f),
            Self::VectorElementCount(data) => data.visit(&mut f),
            Self::VectorSveAddressGen(data) => data.visit(&mut f),
            Self::FlagAdjust(data) => data.visit(&mut f),
            Self::VectorStructLoadReplicate(data) => data.visit(&mut f),
            Self::VectorSmeMisc(data) => data.visit(&mut f),
            Self::VectorPredicateOp(data) => data.visit(&mut f),
            Self::VectorSveCompute(data) => data.visit(&mut f),
            Self::VectorReverseChunks(data) => data.visit(&mut f),
            Self::VectorMatrixMulAcc(data) => data.visit(&mut f),
            Self::VectorSmeOuterProduct(data) => data.visit(&mut f),
            Self::VectorPredicateGen(data) => data.visit(&mut f),
            Self::VectorFpHelper(data) => data.visit(&mut f),
            Self::VectorSvePermute(data) => data.visit(&mut f),
            Self::VectorTernaryLogic(data)
            | Self::VectorMultiSad(data)
            | Self::VectorClassify(data) => data.visit(&mut f),
            Self::VectorDotProduct(data) => data.visit(&mut f),
            Self::VectorIntDotProduct(data) => data.visit(&mut f),
            Self::VectorStringCompare(data) => data.visit(&mut f),
            Self::VectorBitfield(data) => data.visit(&mut f),
            Self::VectorIntersect(data) => data.visit(&mut f),
            Self::VectorShuffleBits(data) => data.visit(&mut f),
            Self::VectorConditionalMove(data) => data.visit(&mut f),
            Self::VectorHorizontalMinPos(data) => data.visit(&mut f),
            Self::VectorComplexMul(data) => data.visit(&mut f),
            Self::VectorHorizontalReduce(data) => data.visit(&mut f),
            Self::BlockString(data) => data.visit(&mut f),
            Self::WideCompareExchange(data) => data.visit(&mut f),
            Self::FpTranscendental(data) => data.visit(&mut f),
            Self::FpuControl(data) => data.visit(&mut f),

            Self::NewObj { dest, args, .. } => {
                f(OperandRole::Def, *dest);
                for item in args {
                    f(OperandRole::Use, *item);
                }
            }

            Self::Call { dest, args, .. } | Self::CallVirt { dest, args, .. } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, *dest_v);
                }
                for item in args {
                    f(OperandRole::Use, *item);
                }
            }

            Self::CallIndirect {
                dest, fptr, args, ..
            } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, *dest_v);
                }
                f(OperandRole::Use, *fptr);
                for item in args {
                    f(OperandRole::Use, *item);
                }
            }

            Self::Return { value } => {
                if let Some(value_v) = value {
                    f(OperandRole::Use, *value_v);
                }
            }

            Self::Phi { dest, operands } => {
                f(OperandRole::Def, *dest);
                for (_, item) in operands {
                    f(OperandRole::Use, *item);
                }
            }

            Self::Jump { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::Leave { .. }
            | Self::Nop
            | Self::Break(_)
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => {}
        }
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
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. } => f(OperandRole::Def, dest),

            Self::ComputeFlags { dest, inputs } => {
                f(OperandRole::Def, dest);
                for input in inputs {
                    f(OperandRole::Use, input);
                }
            }

            Self::CallClobber { outputs } => {
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
            }
            | Self::Rcl {
                dest,
                value,
                amount,
                ..
            }
            | Self::Rcr {
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
            }
            | Self::VectorMaskBinary {
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
            | Self::VectorMaskUnary {
                dest, mask: value, ..
            }
            | Self::VectorReduce { dest, value, .. }
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
            | Self::IndirectBranch { target: value, .. }
            | Self::BranchFlags { flags: value, .. } => f(OperandRole::Use, value),

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
            | Self::AtomicRmw {
                dest,
                addr: a,
                value: b,
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
            | Self::CmpXchg {
                dest,
                addr: a,
                expected: b,
                desired: c,
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
            | Self::InitBlk {
                dest_addr: a,
                value: b,
                size: c,
                ..
            }
            | Self::CopyBlk {
                dest_addr: a,
                src_addr: b,
                size: c,
                ..
            } => {
                f(OperandRole::Use, a);
                f(OperandRole::Use, b);
                f(OperandRole::Use, c);
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
                dest, left, right, ..
            } => {
                f(OperandRole::Def, dest);
                f(OperandRole::Use, left);
                if let Some(right_v) = right {
                    f(OperandRole::Use, right_v);
                }
            }

            Self::NativeOpaque(data) => data.visit_mut(&mut f),
            Self::SystemOp(data) => data.visit_mut(&mut f),
            Self::ComputeOp(data) => data.visit_mut(&mut f),
            Self::BcdAdjust(data) => data.visit_mut(&mut f),
            Self::VectorCrypto(data) => data.visit_mut(&mut f),
            Self::TileOp(data) => data.visit_mut(&mut f),
            Self::VectorPermute(data) => data.visit_mut(&mut f),
            Self::VectorMultiplyAdd(data) => data.visit_mut(&mut f),
            Self::VectorPackNarrow(data) => data.visit_mut(&mut f),
            Self::VectorNarrowSaturate(data) => data.visit_mut(&mut f),
            Self::VectorPredicateWhile(data) => data.visit_mut(&mut f),
            Self::VectorPredicateBreak(data) => data.visit_mut(&mut f),
            Self::VectorComplexAdd(data) => data.visit_mut(&mut f),
            Self::VectorCountAdjust(data) => data.visit_mut(&mut f),
            Self::VectorExtendInLane(data) => data.visit_mut(&mut f),
            Self::VectorElementCount(data) => data.visit_mut(&mut f),
            Self::VectorSveAddressGen(data) => data.visit_mut(&mut f),
            Self::FlagAdjust(data) => data.visit_mut(&mut f),
            Self::VectorStructLoadReplicate(data) => data.visit_mut(&mut f),
            Self::VectorSmeMisc(data) => data.visit_mut(&mut f),
            Self::VectorPredicateOp(data) => data.visit_mut(&mut f),
            Self::VectorSveCompute(data) => data.visit_mut(&mut f),
            Self::VectorReverseChunks(data) => data.visit_mut(&mut f),
            Self::VectorMatrixMulAcc(data) => data.visit_mut(&mut f),
            Self::VectorSmeOuterProduct(data) => data.visit_mut(&mut f),
            Self::VectorPredicateGen(data) => data.visit_mut(&mut f),
            Self::VectorFpHelper(data) => data.visit_mut(&mut f),
            Self::VectorSvePermute(data) => data.visit_mut(&mut f),
            Self::VectorTernaryLogic(data)
            | Self::VectorMultiSad(data)
            | Self::VectorClassify(data) => data.visit_mut(&mut f),
            Self::VectorDotProduct(data) => data.visit_mut(&mut f),
            Self::VectorIntDotProduct(data) => data.visit_mut(&mut f),
            Self::VectorStringCompare(data) => data.visit_mut(&mut f),
            Self::VectorBitfield(data) => data.visit_mut(&mut f),
            Self::VectorIntersect(data) => data.visit_mut(&mut f),
            Self::VectorShuffleBits(data) => data.visit_mut(&mut f),
            Self::VectorConditionalMove(data) => data.visit_mut(&mut f),
            Self::VectorHorizontalMinPos(data) => data.visit_mut(&mut f),
            Self::VectorComplexMul(data) => data.visit_mut(&mut f),
            Self::VectorHorizontalReduce(data) => data.visit_mut(&mut f),
            Self::BlockString(data) => data.visit_mut(&mut f),
            Self::WideCompareExchange(data) => data.visit_mut(&mut f),
            Self::FpTranscendental(data) => data.visit_mut(&mut f),
            Self::FpuControl(data) => data.visit_mut(&mut f),

            Self::NewObj { dest, args, .. } => {
                f(OperandRole::Def, dest);
                for item in args {
                    f(OperandRole::Use, item);
                }
            }

            Self::Call { dest, args, .. } | Self::CallVirt { dest, args, .. } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, dest_v);
                }
                for item in args {
                    f(OperandRole::Use, item);
                }
            }

            Self::CallIndirect {
                dest, fptr, args, ..
            } => {
                if let Some(dest_v) = dest {
                    f(OperandRole::Def, dest_v);
                }
                f(OperandRole::Use, fptr);
                for item in args {
                    f(OperandRole::Use, item);
                }
            }

            Self::Return { value } => {
                if let Some(value_v) = value {
                    f(OperandRole::Use, value_v);
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
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => {}
        }
    }

    /// Calls `f` for every variable used by this operation.
    pub fn for_each_use<F>(&self, mut f: F)
    where
        F: FnMut(SsaVarId),
    {
        self.visit_operands(|role, var| {
            if matches!(role, OperandRole::Use) {
                f(var);
            }
        });
    }

    /// Returns all variables used by this operation.
    #[must_use]
    pub fn uses(&self) -> Vec<SsaVarId> {
        let mut uses = Vec::new();
        self.for_each_use(|var| uses.push(var));
        uses
    }

    /// Returns the first variable used by this operation, in operand order.
    ///
    /// Equivalent to `self.uses().first().copied()` without the `Vec`: like
    /// [`use_count`](Self::use_count) and [`uses_var`](Self::uses_var) it walks
    /// [`for_each_use`](Self::for_each_use) directly, so a caller that wants
    /// one operand — the type of the value an identity is being checked
    /// against, say — pays no allocation for it.
    ///
    /// # Returns
    ///
    /// The operation's first use, or `None` when it reads no variables.
    #[must_use]
    pub fn first_use(&self) -> Option<SsaVarId> {
        let mut first = None;
        self.for_each_use(|var| {
            if first.is_none() {
                first = Some(var);
            }
        });
        first
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
        let mut found = false;
        self.for_each_use(|used| {
            if used == var {
                found = true;
            }
        });
        found
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
    /// for every pair — `O(operands × pairs)`). `lookup` returns the replacement
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
