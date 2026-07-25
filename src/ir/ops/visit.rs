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

use super::*;
use crate::{ir::variable::SsaVarId, target::Target};

/// Iterator over variables defined by an SSA operation.
pub struct SsaDefs<'a> {
    primary: Option<SsaVarId>,
    secondary: Option<SsaVarId>,
    extra: Option<std::slice::Iter<'a, SsaVarId>>,
}

impl<'a> SsaDefs<'a> {
    /// Creates a definition iterator from optional primary, optional secondary,
    /// and any extra definitions.
    #[must_use]
    pub fn new(
        primary: Option<SsaVarId>,
        secondary: Option<SsaVarId>,
        extra: Option<&'a [SsaVarId]>,
    ) -> Self {
        Self {
            primary,
            secondary,
            extra: extra.map(<[SsaVarId]>::iter),
        }
    }
}

impl Iterator for SsaDefs<'_> {
    type Item = SsaVarId;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(primary) = self.primary.take() {
            return Some(primary);
        }
        if let Some(secondary) = self.secondary.take() {
            return Some(secondary);
        }
        self.extra.as_mut()?.next().copied()
    }
}

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

    /// Returns all variables defined by this operation.
    #[must_use]
    pub fn defs(&self) -> SsaDefs<'_> {
        match self {
            Self::NativeOpaque(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::NativeIntrinsic(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::SystemOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::ComputeOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::BcdAdjust(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorCrypto(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::TileOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPermute(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorMultiplyAdd(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPackNarrow(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorNarrowSaturate(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateWhile(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateBreak(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorComplexAdd(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorCountAdjust(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorExtendInLane(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorElementCount(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSveAddressGen(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::FlagAdjust(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorStructLoadReplicate(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSmeMisc(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateOp(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSveCompute(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorReverseChunks(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorMatrixMulAcc(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSmeOuterProduct(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorPredicateGen(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorFpHelper(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSvePermute(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorTernaryLogic(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorDotProduct(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorMultiSad(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorIntDotProduct(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorStringCompare(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorBitfield(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorIntersect(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorShuffleBits(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorConditionalMove(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorHorizontalMinPos(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorComplexMul(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorClassify(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorHorizontalReduce(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::BlockString(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::WideCompareExchange(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::CallClobber { outputs } => SsaDefs::new(None, None, Some(outputs)),
            Self::ComputeFlags { dest, .. } => SsaDefs::new(Some(*dest), None, None),
            Self::FpTranscendental(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::FpuControl(data) => SsaDefs::new(None, None, Some(&data.outputs)),
            Self::VectorSegmentLoad { dests, .. } => SsaDefs::new(None, None, Some(dests)),
            Self::VectorFaultingLoad { dest, fault, .. } => SsaDefs::new(Some(*dest), *fault, None),
            Self::AtomicCmpXchg { old, success, .. } => SsaDefs::new(Some(*old), *success, None),
            Self::AtomicPairLoad { first, second, .. } => {
                SsaDefs::new(Some(*first), Some(*second), None)
            }
            Self::AtomicPairCmpXchg {
                old_first,
                old_second,
                ..
            } => SsaDefs::new(Some(*old_first), Some(*old_second), None),
            Self::WideMul { low, high, .. } => SsaDefs::new(Some(*low), Some(*high), None),
            Self::WideDiv {
                quotient,
                remainder,
                ..
            } => SsaDefs::new(Some(*quotient), Some(*remainder), None),
            _ => SsaDefs::new(self.dest(), self.flags_dest(), None),
        }
    }

    /// Sets the destination variable for operations that produce a result.
    ///
    /// This is used during SSA renaming to update the dest after assigning
    /// new SSA variable IDs. Returns `true` if the dest was updated.
    ///
    /// # Arguments
    ///
    /// * `new_dest` - The new destination variable ID
    pub fn set_dest(&mut self, new_dest: SsaVarId) -> bool {
        match self {
            Self::Const { dest, .. }
            | Self::Add { dest, .. }
            | Self::AddOvf { dest, .. }
            | Self::Sub { dest, .. }
            | Self::SubOvf { dest, .. }
            | Self::Mul { dest, .. }
            | Self::MulOvf { dest, .. }
            | Self::WideMul { low: dest, .. }
            | Self::Div { dest, .. }
            | Self::Rem { dest, .. }
            | Self::WideDiv { quotient: dest, .. }
            | Self::Neg { dest, .. }
            | Self::And { dest, .. }
            | Self::Or { dest, .. }
            | Self::Xor { dest, .. }
            | Self::Not { dest, .. }
            | Self::Shl { dest, .. }
            | Self::Shr { dest, .. }
            | Self::Ceq { dest, .. }
            | Self::Clt { dest, .. }
            | Self::Cgt { dest, .. }
            | Self::BoolAnd { dest, .. }
            | Self::BoolOr { dest, .. }
            | Self::BoolXor { dest, .. }
            | Self::BoolNot { dest, .. }
            | Self::IntConv { dest, .. }
            | Self::IntToPtr { dest, .. }
            | Self::PtrToInt { dest, .. }
            | Self::IntToFloat { dest, .. }
            | Self::FloatToInt { dest, .. }
            | Self::FloatConv { dest, .. }
            | Self::Bitcast { dest, .. }
            | Self::LoadField { dest, .. }
            | Self::LoadStaticField { dest, .. }
            | Self::LoadFieldAddr { dest, .. }
            | Self::LoadStaticFieldAddr { dest, .. }
            | Self::LoadElement { dest, .. }
            | Self::LoadElementAddr { dest, .. }
            | Self::PtrAdd { dest, .. }
            | Self::ArrayLength { dest, .. }
            | Self::LoadIndirect { dest, .. }
            | Self::NewObj { dest, .. }
            | Self::NewArr { dest, .. }
            | Self::CastClass { dest, .. }
            | Self::IsInst { dest, .. }
            | Self::Box { dest, .. }
            | Self::Unbox { dest, .. }
            | Self::UnboxAny { dest, .. }
            | Self::SizeOf { dest, .. }
            | Self::LoadToken { dest, .. }
            | Self::LoadFunctionPtr { dest, .. }
            | Self::LoadVirtFunctionPtr { dest, .. }
            | Self::LoadArg { dest, .. }
            | Self::LoadLocal { dest, .. }
            | Self::LoadArgAddr { dest, .. }
            | Self::LoadLocalAddr { dest, .. }
            | Self::Copy { dest, .. }
            | Self::Ckfinite { dest, .. }
            | Self::FpClassify { dest, .. }
            | Self::LocalAlloc { dest, .. }
            | Self::LoadObj { dest, .. }
            | Self::Phi { dest, .. }
            | Self::Rol { dest, .. }
            | Self::Ror { dest, .. }
            | Self::Rcl { dest, .. }
            | Self::Rcr { dest, .. }
            | Self::BSwap { dest, .. }
            | Self::BRev { dest, .. }
            | Self::BitScanForward { dest, .. }
            | Self::BitScanReverse { dest, .. }
            | Self::Popcount { dest, .. }
            | Self::Parity { dest, .. }
            | Self::ComputeFlags { dest, .. }
            | Self::Select { dest, .. }
            | Self::CmpXchg { dest, .. }
            | Self::AtomicRmw { dest, .. }
            | Self::AtomicLoad { dest, .. }
            | Self::AtomicExchange { dest, .. }
            | Self::AtomicLockRmw { dest, .. }
            | Self::AtomicStoreConditional { status: dest, .. }
            | Self::AtomicPairLoad { first: dest, .. }
            | Self::AtomicPairStoreConditional { status: dest, .. }
            | Self::AtomicPairCmpXchg {
                old_first: dest, ..
            }
            | Self::ReadFlags { dest, .. }
            | Self::VectorUnary { dest, .. }
            | Self::VectorBinary { dest, .. }
            | Self::VectorTernary { dest, .. }
            | Self::VectorPredicatedUnary { dest, .. }
            | Self::VectorPredicatedBinary { dest, .. }
            | Self::VectorPredicatedTernary { dest, .. }
            | Self::VectorCompare { dest, .. }
            | Self::VectorLoad { dest, .. }
            | Self::VectorMaskedLoad { dest, .. }
            | Self::VectorBroadcastLoad { dest, .. }
            | Self::VectorGather { dest, .. }
            | Self::VectorFaultingLoad { dest, .. }
            | Self::VectorExtract { dest, .. }
            | Self::VectorInsert { dest, .. }
            | Self::VectorSplat { dest, .. }
            | Self::VectorShuffle { dest, .. }
            | Self::VectorCast { dest, .. }
            | Self::VectorReinterpret { dest, .. }
            | Self::VectorPack { dest, .. }
            | Self::VectorPackLoad { dest, .. }
            | Self::VectorMaskUnary { dest, .. }
            | Self::VectorMaskBinary { dest, .. }
            | Self::VectorReduce { dest, .. }
            | Self::VectorBitmask { dest, .. } => {
                *dest = new_dest;
                true
            }

            Self::NativeOpaque(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::CallClobber { outputs } => {
                if let Some(first) = outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::NativeIntrinsic(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::SystemOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::ComputeOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::BcdAdjust(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorCrypto(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::TileOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPermute(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorMultiplyAdd(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPackNarrow(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorNarrowSaturate(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateWhile(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateBreak(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorComplexAdd(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorCountAdjust(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorExtendInLane(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorElementCount(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSveAddressGen(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::FlagAdjust(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorStructLoadReplicate(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSmeMisc(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateOp(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSveCompute(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorReverseChunks(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorMatrixMulAcc(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSmeOuterProduct(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorPredicateGen(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorFpHelper(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSvePermute(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorTernaryLogic(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorDotProduct(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorMultiSad(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorIntDotProduct(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorStringCompare(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorBitfield(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorIntersect(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorShuffleBits(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorConditionalMove(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorHorizontalMinPos(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorComplexMul(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorClassify(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorHorizontalReduce(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::BlockString(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::WideCompareExchange(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::FpTranscendental(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::FpuControl(data) => {
                if let Some(first) = data.outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::VectorSegmentLoad { dests: outputs, .. } => {
                if let Some(first) = outputs.first_mut() {
                    *first = new_dest;
                    true
                } else {
                    false
                }
            }
            Self::AtomicCmpXchg { old, .. } => {
                *old = new_dest;
                true
            }

            Self::Call { dest, .. }
            | Self::CallVirt { dest, .. }
            | Self::CallIndirect { dest, .. } => {
                *dest = Some(new_dest);
                true
            }

            // Operations that don't produce a result - cannot set dest
            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::AtomicStore { .. }
            | Self::FloatCompareFlags { .. }
            | Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::Pop { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::Leave { .. }
            | Self::InitBlk { .. }
            | Self::CopyBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::StoreObj { .. }
            | Self::Nop
            | Self::Break
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::BranchFlags { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackStore { .. }
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => false,
        }
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

            Self::NativeOpaque(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

            Self::NativeIntrinsic(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::SystemOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::ComputeOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::BcdAdjust(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorCrypto(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::TileOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPermute(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorMultiplyAdd(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPackNarrow(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorNarrowSaturate(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateWhile(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateBreak(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorComplexAdd(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorCountAdjust(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorExtendInLane(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorElementCount(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
            }
            Self::VectorSveAddressGen(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::FlagAdjust(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorStructLoadReplicate(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSmeMisc(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateOp(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSveCompute(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorReverseChunks(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorMatrixMulAcc(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSmeOuterProduct(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorPredicateGen(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorFpHelper(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorSvePermute(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorTernaryLogic(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorDotProduct(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorMultiSad(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorIntDotProduct(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorStringCompare(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorBitfield(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorIntersect(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorShuffleBits(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorConditionalMove(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorHorizontalMinPos(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorComplexMul(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorClassify(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::VectorHorizontalReduce(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::BlockString(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }
            Self::WideCompareExchange(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

            Self::FpTranscendental(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

            Self::FpuControl(data) => {
                for item in &data.outputs {
                    f(OperandRole::Def, *item);
                }
                for item in &data.inputs {
                    f(OperandRole::Use, *item);
                }
            }

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
            | Self::Break
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

            Self::NativeOpaque(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

            Self::NativeIntrinsic(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::SystemOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::ComputeOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::BcdAdjust(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorCrypto(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::TileOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPermute(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorMultiplyAdd(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPackNarrow(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorNarrowSaturate(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateWhile(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateBreak(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorComplexAdd(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorCountAdjust(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorExtendInLane(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorElementCount(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
            }
            Self::VectorSveAddressGen(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::FlagAdjust(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorStructLoadReplicate(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSmeMisc(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateOp(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSveCompute(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorReverseChunks(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorMatrixMulAcc(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSmeOuterProduct(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorPredicateGen(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorFpHelper(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorSvePermute(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorTernaryLogic(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorDotProduct(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorMultiSad(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorIntDotProduct(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorStringCompare(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorBitfield(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorIntersect(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorShuffleBits(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorConditionalMove(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorHorizontalMinPos(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorComplexMul(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorClassify(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::VectorHorizontalReduce(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::BlockString(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }
            Self::WideCompareExchange(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

            Self::FpTranscendental(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

            Self::FpuControl(data) => {
                for item in &mut data.outputs {
                    f(OperandRole::Def, item);
                }
                for item in &mut data.inputs {
                    f(OperandRole::Use, item);
                }
            }

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
            | Self::Break
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
