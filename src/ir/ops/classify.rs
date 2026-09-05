//! Operation taxonomy, similarity tokens, and structural views of [`SsaOp`].
//!
//! Two distinct consumers: the optimization passes, which ask for structural
//! views ([`SsaOp::as_binary_op`], [`SsaOp::as_unary_op`],
//! [`SsaOp::infer_result_type`], [`SsaOp::stack_effect`]); and the similarity
//! pipeline, which needs *stable* target-generic features
//! ([`SsaOp::opcode_name`], [`SsaOp::coarse_token`], [`SsaOp::feature_token`],
//! [`SsaOp::similarity_class`]). Token spellings are content-addressed
//! downstream, so changing one changes similarity hashes.

use std::borrow::Cow;

use crate::{
    ir::ops::{
        def::SsaOp,
        kinds::{
            CmpKind, ComputeKind, Signedness, SsaFeatureToken, SsaOpClass, SsaSimilarityClass,
        },
        native::{BinaryOpInfo, BinaryOpKind, BlockStringKind, UnaryOpInfo, UnaryOpKind},
        operands::KindedOperands,
        vector::VectorCompareKind,
    },
    target::Target,
};

/// Lowercases and trims a high-cardinality mnemonic for token use.
///
/// Applied only where the input is a *host-supplied* string — a native
/// mnemonic echoed from a decoder. A `kind_str()` is a crate literal already in
/// this form, pinned by the registry-wide assertion in the test module, so
/// normalizing one would allocate to produce the string it was handed.
fn normalize_token_text(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

impl<T: Target> SsaOp<T> {
    /// Returns the high-level operation family for this opcode.
    #[must_use]
    pub const fn class(&self) -> SsaOpClass {
        match self {
            Self::Nop
            | Self::Phi { .. }
            | Self::Copy { .. }
            | Self::Pop { .. }
            | Self::CallClobber { .. } => SsaOpClass::Synthetic,

            Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => SsaOpClass::Boolean,

            Self::FlagAdjust(_) => SsaOpClass::Flags,
            Self::ReadFlags { .. } | Self::BranchFlags { .. } | Self::ComputeFlags { .. } => {
                SsaOpClass::Flags
            }

            Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorLoad { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorGather { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackLoad { .. }
            | Self::VectorPackStore { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorSplat { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorPack { .. }
            | Self::VectorZeroUpper { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorMaskBinary { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. } => SsaOpClass::Vector,

            Self::LoadField { .. }
            | Self::StoreField { .. }
            | Self::LoadStaticField { .. }
            | Self::StoreStaticField { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElement { .. }
            | Self::StoreElement { .. }
            | Self::LoadElementAddr { .. }
            | Self::PtrAdd { .. }
            | Self::ArrayLength { .. }
            | Self::LoadIndirect { .. }
            | Self::StoreIndirect { .. }
            | Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::LocalAlloc { .. }
            | Self::InitBlk { .. }
            | Self::CopyBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::LoadObj { .. }
            | Self::StoreObj { .. }
            | Self::Fence { .. } => SsaOpClass::Memory,

            Self::CmpXchg { .. }
            | Self::AtomicRmw { .. }
            | Self::AtomicLoad { .. }
            | Self::AtomicStore { .. }
            | Self::AtomicPairLoad { .. }
            | Self::AtomicPairStoreConditional { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. }
            | Self::AtomicCmpXchg { .. }
            | Self::AtomicPairCmpXchg { .. } => SsaOpClass::Atomic,

            Self::Call { .. }
            | Self::CallVirt { .. }
            | Self::CallIndirect { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadVirtFunctionPtr { .. } => SsaOpClass::Call,

            Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::InterruptReturn
            | Self::Unreachable
            | Self::Leave { .. }
            | Self::Break(_) => SsaOpClass::Control,

            Self::NativeOpaque(_) => SsaOpClass::NativeOpaque,
            Self::SystemOp(_) => SsaOpClass::Call,
            Self::ComputeOp(_) => SsaOpClass::Scalar,
            Self::BcdAdjust(_) => SsaOpClass::Scalar,
            Self::VectorCrypto(_) => SsaOpClass::Vector,
            Self::TileOp(_) => SsaOpClass::Vector,
            Self::VectorPermute(_) => SsaOpClass::Vector,
            Self::VectorMultiplyAdd(_) => SsaOpClass::Vector,
            Self::VectorPackNarrow(_) => SsaOpClass::Vector,
            Self::VectorNarrowSaturate(_) => SsaOpClass::Vector,
            Self::VectorPredicateWhile(_) => SsaOpClass::Vector,
            Self::VectorPredicateBreak(_) => SsaOpClass::Vector,
            Self::VectorComplexAdd(_) => SsaOpClass::Vector,
            Self::VectorCountAdjust(_) => SsaOpClass::Vector,
            Self::VectorExtendInLane(_) => SsaOpClass::Vector,
            Self::VectorElementCount(_) => SsaOpClass::Vector,
            Self::VectorSveAddressGen(_) => SsaOpClass::Vector,
            Self::VectorStructLoadReplicate(_) => SsaOpClass::Vector,
            Self::VectorSmeMisc(_) => SsaOpClass::Vector,
            Self::VectorPredicateOp(_) => SsaOpClass::Vector,
            Self::VectorSveCompute(_) => SsaOpClass::Vector,
            Self::VectorReverseChunks(_) => SsaOpClass::Vector,
            Self::VectorMatrixMulAcc(_) => SsaOpClass::Vector,
            Self::VectorSmeOuterProduct(_) => SsaOpClass::Vector,
            Self::VectorPredicateGen(_) => SsaOpClass::Vector,
            Self::VectorFpHelper(_) => SsaOpClass::Vector,
            Self::VectorSvePermute(_) => SsaOpClass::Vector,
            Self::VectorTernaryLogic(_) => SsaOpClass::Vector,
            Self::VectorDotProduct(_) => SsaOpClass::Vector,
            Self::VectorMultiSad(_) => SsaOpClass::Vector,
            Self::VectorIntDotProduct(_) => SsaOpClass::Vector,
            Self::VectorStringCompare(_) => SsaOpClass::Vector,
            Self::VectorBitfield(_) => SsaOpClass::Vector,
            Self::VectorIntersect(_) => SsaOpClass::Vector,
            Self::VectorShuffleBits(_) => SsaOpClass::Vector,
            Self::VectorConditionalMove(_) => SsaOpClass::Vector,
            Self::VectorHorizontalMinPos(_) => SsaOpClass::Vector,
            Self::VectorComplexMul(_) => SsaOpClass::Vector,
            Self::VectorClassify(_) => SsaOpClass::Vector,
            Self::VectorHorizontalReduce(_) => SsaOpClass::Vector,
            Self::BlockString(_) => SsaOpClass::Memory,
            Self::WideCompareExchange { .. } => SsaOpClass::Atomic,
            Self::FpTranscendental { .. } | Self::FpuControl { .. } => SsaOpClass::NativeOpaque,

            Self::WideMul { .. } | Self::WideDiv { .. } => SsaOpClass::WideArithmetic,

            Self::FloatCompareFlags { .. } => SsaOpClass::Scalar,

            Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly => SsaOpClass::Prefix,

            Self::Add { flags: Some(_), .. }
            | Self::AddOvf { flags: Some(_), .. }
            | Self::Sub { flags: Some(_), .. }
            | Self::SubOvf { flags: Some(_), .. }
            | Self::Mul { flags: Some(_), .. }
            | Self::MulOvf { flags: Some(_), .. }
            | Self::Div { flags: Some(_), .. }
            | Self::Rem { flags: Some(_), .. }
            | Self::Neg { flags: Some(_), .. }
            | Self::And { flags: Some(_), .. }
            | Self::Or { flags: Some(_), .. }
            | Self::Xor { flags: Some(_), .. }
            | Self::Not { flags: Some(_), .. }
            | Self::Shl { flags: Some(_), .. }
            | Self::Shr { flags: Some(_), .. } => SsaOpClass::Flags,

            Self::Const { .. }
            | Self::Add { .. }
            | Self::AddOvf { .. }
            | Self::Sub { .. }
            | Self::SubOvf { .. }
            | Self::Mul { .. }
            | Self::MulOvf { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::Neg { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Not { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Ceq { .. }
            | Self::Clt { .. }
            | Self::Cgt { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Rcl { .. }
            | Self::Rcr { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::Bitcast { .. }
            | Self::Select { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. } => SsaOpClass::Scalar,
        }
    }

    /// Returns the operand signedness carried by this operation's payload.
    ///
    /// `Some` only for operations whose payload carries an `unsigned` flag
    /// (overflow-checked arithmetic, wide multiply/divide, division and
    /// remainder, shift-right, ordered comparison, conversion,
    /// compare-and-branch, lane-wise vector comparison); `None` for
    /// sign-agnostic operations.
    #[must_use]
    pub const fn arith_signedness(&self) -> Option<Signedness> {
        match self {
            Self::AddOvf { unsigned, .. }
            | Self::SubOvf { unsigned, .. }
            | Self::MulOvf { unsigned, .. }
            | Self::WideMul { unsigned, .. }
            | Self::Div { unsigned, .. }
            | Self::WideDiv { unsigned, .. }
            | Self::Rem { unsigned, .. }
            | Self::Shr { unsigned, .. }
            | Self::Clt { unsigned, .. }
            | Self::Cgt { unsigned, .. }
            | Self::IntConv { unsigned, .. }
            | Self::IntToFloat { unsigned, .. }
            | Self::FloatToInt { unsigned, .. }
            | Self::BranchCmp { unsigned, .. }
            | Self::VectorCompare { unsigned, .. } => Some(Signedness::from_unsigned(*unsigned)),
            _ => None,
        }
    }

    /// Returns the comparison relation carried by this operation's payload.
    ///
    /// Covers scalar compare-and-set (`Ceq`/`Clt`/`Cgt`), compare-and-branch
    /// (`BranchCmp`), and lane-wise vector comparison (`VectorCompare`).
    /// Signedness is reported separately by [`Self::arith_signedness`].
    /// `None` for non-comparison operations and for `FloatCompareFlags`,
    /// which produces a flag bundle rather than a single relation.
    #[must_use]
    pub const fn compare_kind(&self) -> Option<CmpKind> {
        match self {
            Self::Ceq { .. } => Some(CmpKind::Eq),
            Self::Clt { .. } => Some(CmpKind::Lt),
            Self::Cgt { .. } => Some(CmpKind::Gt),
            Self::BranchCmp { cmp, .. } => Some(*cmp),
            Self::VectorCompare { kind, .. } => match kind {
                VectorCompareKind::Eq => Some(CmpKind::Eq),
                VectorCompareKind::Ne => Some(CmpKind::Ne),
                VectorCompareKind::Lt => Some(CmpKind::Lt),
                VectorCompareKind::Le => Some(CmpKind::Le),
                VectorCompareKind::Gt => Some(CmpKind::Gt),
                VectorCompareKind::Ge => Some(CmpKind::Ge),
                // Unordered / ordered / not-lt / not-le / constant predicates
                // have no clean scalar [`CmpKind`] equivalent.
                VectorCompareKind::Unordered
                | VectorCompareKind::Ordered
                | VectorCompareKind::NotLt
                | VectorCompareKind::NotLe
                | VectorCompareKind::NotGe
                | VectorCompareKind::NotGt
                | VectorCompareKind::AlwaysTrue
                | VectorCompareKind::AlwaysFalse => None,
            },
            _ => None,
        }
    }

    /// Returns a stable similarity-oriented operation family.
    ///
    /// Unlike [`Self::class`], this groups scalar operations by semantic role
    /// and separates memory reads, writes, read-write operations, calls,
    /// atomics, fences, native opaque side effects, and vector instructions.
    /// Host crates can use this for deterministic feature extraction without
    /// relying on formatted opcode text.
    #[must_use]
    pub const fn similarity_class(&self) -> SsaSimilarityClass {
        match self {
            Self::Nop
            | Self::Phi { .. }
            | Self::Copy { .. }
            | Self::Pop { .. }
            | Self::CallClobber { .. } => SsaSimilarityClass::Synthetic,

            Self::Const { .. } | Self::SizeOf { .. } | Self::LoadToken { .. } => {
                SsaSimilarityClass::Constant
            }

            Self::Add { flags: Some(_), .. }
            | Self::AddOvf { flags: Some(_), .. }
            | Self::Sub { flags: Some(_), .. }
            | Self::SubOvf { flags: Some(_), .. }
            | Self::Mul { flags: Some(_), .. }
            | Self::MulOvf { flags: Some(_), .. }
            | Self::Div { flags: Some(_), .. }
            | Self::Rem { flags: Some(_), .. }
            | Self::Neg { flags: Some(_), .. }
            | Self::And { flags: Some(_), .. }
            | Self::Or { flags: Some(_), .. }
            | Self::Xor { flags: Some(_), .. }
            | Self::Not { flags: Some(_), .. }
            | Self::Shl { flags: Some(_), .. }
            | Self::Shr { flags: Some(_), .. }
            | Self::ReadFlags { .. }
            | Self::BranchFlags { .. }
            | Self::ComputeFlags { .. }
            | Self::FlagAdjust(_) => SsaSimilarityClass::Flags,

            Self::Add { .. }
            | Self::AddOvf { .. }
            | Self::Sub { .. }
            | Self::SubOvf { .. }
            | Self::Mul { .. }
            | Self::MulOvf { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::FloatCompareFlags { .. }
            | Self::Neg { .. } => SsaSimilarityClass::Arithmetic,

            Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Not { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. } => SsaSimilarityClass::Bitwise,

            Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Rcl { .. }
            | Self::Rcr { .. } => SsaSimilarityClass::ShiftRotate,

            Self::Ceq { .. } | Self::Clt { .. } | Self::Cgt { .. } => SsaSimilarityClass::Compare,

            Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => SsaSimilarityClass::Boolean,

            Self::Select { .. } => SsaSimilarityClass::Select,

            Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::Bitcast { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. } => SsaSimilarityClass::Conversion,

            Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadVirtFunctionPtr { .. } => SsaSimilarityClass::TypeFlow,

            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElement { .. }
            | Self::LoadElementAddr { .. }
            | Self::PtrAdd { .. }
            | Self::ArrayLength { .. }
            | Self::LoadIndirect { .. }
            | Self::LoadObj { .. }
            | Self::VectorLoad { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorGather { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorPackLoad { .. } => SsaSimilarityClass::MemoryRead,

            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::StoreObj { .. }
            | Self::InitObj { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackStore { .. } => SsaSimilarityClass::MemoryWrite,

            Self::CopyBlk { .. } | Self::InitBlk { .. } | Self::CopyObj { .. } => {
                SsaSimilarityClass::MemoryReadWrite
            }

            Self::NewObj { .. } | Self::NewArr { .. } | Self::LocalAlloc { .. } => {
                SsaSimilarityClass::Allocation
            }

            Self::CmpXchg { .. }
            | Self::AtomicRmw { .. }
            | Self::AtomicLoad { .. }
            | Self::AtomicStore { .. }
            | Self::AtomicPairLoad { .. }
            | Self::AtomicPairStoreConditional { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. }
            | Self::AtomicCmpXchg { .. }
            | Self::AtomicPairCmpXchg { .. } => SsaSimilarityClass::Atomic,

            Self::Fence { .. } => SsaSimilarityClass::Fence,

            Self::Call { .. } | Self::CallVirt { .. } | Self::CallIndirect { .. } => {
                SsaSimilarityClass::Call
            }

            Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::InterruptReturn
            | Self::Unreachable
            | Self::Leave { .. }
            | Self::Break(_) => SsaSimilarityClass::Control,

            Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorSplat { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorPack { .. }
            | Self::VectorZeroUpper { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorMaskBinary { .. }
            | Self::VectorReduce { .. }
            | Self::VectorCrypto(_)
            | Self::TileOp(_)
            | Self::VectorPermute(_)
            | Self::VectorMultiplyAdd(_)
            | Self::VectorPackNarrow(_)
            | Self::VectorNarrowSaturate(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorPredicateBreak(_)
            | Self::VectorComplexAdd(_)
            | Self::VectorCountAdjust(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorSveAddressGen(_)
            | Self::VectorStructLoadReplicate(_)
            | Self::VectorSmeMisc(_)
            | Self::VectorPredicateOp(_)
            | Self::VectorSveCompute(_)
            | Self::VectorReverseChunks(_)
            | Self::VectorMatrixMulAcc(_)
            | Self::VectorSmeOuterProduct(_)
            | Self::VectorPredicateGen(_)
            | Self::VectorFpHelper(_)
            | Self::VectorSvePermute(_)
            | Self::VectorTernaryLogic(_)
            | Self::VectorDotProduct(_)
            | Self::VectorMultiSad(_)
            | Self::VectorIntDotProduct(_)
            | Self::VectorStringCompare(_)
            | Self::VectorBitfield(_)
            | Self::VectorIntersect(_)
            | Self::VectorShuffleBits(_)
            | Self::VectorConditionalMove(_)
            | Self::VectorHorizontalMinPos(_)
            | Self::VectorComplexMul(_)
            | Self::VectorClassify(_)
            | Self::VectorHorizontalReduce(_)
            | Self::VectorBitmask { .. } => SsaSimilarityClass::Vector,

            Self::WideMul { .. } | Self::WideDiv { .. } => SsaSimilarityClass::WideArithmetic,

            Self::NativeOpaque(_) => SsaSimilarityClass::NativeOpaque,
            Self::SystemOp(_) => SsaSimilarityClass::Call,
            Self::ComputeOp(data) => match data.kind {
                ComputeKind::BitDeposit | ComputeKind::BitExtract | ComputeKind::PointerAuth(_) => {
                    SsaSimilarityClass::Bitwise
                }
                ComputeKind::Checksum | ComputeKind::MipsDspAccumulate => {
                    SsaSimilarityClass::Arithmetic
                }
                ComputeKind::Random { .. } => SsaSimilarityClass::Call,
            },
            Self::BcdAdjust(_) => SsaSimilarityClass::Arithmetic,
            Self::BlockString(data) => match data.kind {
                BlockStringKind::Load => SsaSimilarityClass::MemoryRead,
                BlockStringKind::Compare | BlockStringKind::Scan => {
                    SsaSimilarityClass::MemoryReadWrite
                }
            },
            Self::WideCompareExchange { .. } => SsaSimilarityClass::Atomic,
            Self::FpTranscendental { .. } | Self::FpuControl { .. } => {
                SsaSimilarityClass::NativeOpaque
            }

            Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly => SsaSimilarityClass::Prefix,
        }
    }

    /// Returns a stable target-generic opcode name.
    ///
    /// The returned name is intentionally independent of operand IDs, target
    /// metadata, and formatting details. Use [`Self::feature_token`] when a
    /// richer canonical feature is needed.
    #[must_use]
    pub const fn opcode_name(&self) -> &'static str {
        match self {
            Self::Const { .. } => "const",
            Self::Add { .. } => "add",
            Self::AddOvf { .. } => "add.ovf",
            Self::Sub { .. } => "sub",
            Self::SubOvf { .. } => "sub.ovf",
            Self::Mul { .. } => "mul",
            Self::MulOvf { .. } => "mul.ovf",
            Self::WideMul { .. } => "wide.mul",
            Self::Div { .. } => "div",
            Self::Rem { .. } => "rem",
            Self::WideDiv { .. } => "wide.div",
            Self::Neg { .. } => "neg",
            Self::And { .. } => "and",
            Self::Or { .. } => "or",
            Self::Xor { .. } => "xor",
            Self::Not { .. } => "not",
            Self::Shl { .. } => "shl",
            Self::Shr { .. } => "shr",
            Self::Rol { .. } => "rol",
            Self::Ror { .. } => "ror",
            Self::Rcl { .. } => "rcl",
            Self::Rcr { .. } => "rcr",
            Self::BSwap { .. } => "bswap",
            Self::BRev { .. } => "brev",
            Self::BitScanForward { .. } => "bsf",
            Self::BitScanReverse { .. } => "bsr",
            Self::Popcount { .. } => "popcount",
            Self::Parity { .. } => "parity",
            Self::Ceq { .. } => "ceq",
            Self::Clt { .. } => "clt",
            Self::Cgt { .. } => "cgt",
            Self::BoolAnd { .. } => "bool.and",
            Self::BoolOr { .. } => "bool.or",
            Self::BoolXor { .. } => "bool.xor",
            Self::BoolNot { .. } => "bool.not",
            Self::IntConv { .. } => "conv",
            Self::IntToPtr { .. } => "inttoptr",
            Self::PtrToInt { .. } => "ptrtoint",
            Self::IntToFloat { .. } => "inttofloat",
            Self::FloatToInt { .. } => "floattoint",
            Self::FloatConv { .. } => "fconv",
            Self::Bitcast { .. } => "bitcast",
            Self::Select { .. } => "select",
            Self::ReadFlags { .. } => "readflags",
            Self::ComputeFlags { .. } => "flags.compute",
            Self::CallClobber { .. } => "call.clobber",
            Self::VectorUnary { .. } => "vector.unary",
            Self::VectorBinary { .. } => "vector.binary",
            Self::VectorTernary { .. } => "vector.ternary",
            Self::VectorPredicatedUnary { .. } => "vector.predicated.unary",
            Self::VectorPredicatedBinary { .. } => "vector.predicated.binary",
            Self::VectorPredicatedTernary { .. } => "vector.predicated.ternary",
            Self::VectorCompare { .. } => "vector.compare",
            Self::VectorLoad { .. } => "vector.load",
            Self::VectorStore { .. } => "vector.store",
            Self::VectorMaskedLoad { .. } => "vector.masked.load",
            Self::VectorMaskedStore { .. } => "vector.masked.store",
            Self::VectorBroadcastLoad { .. } => "vector.broadcast.load",
            Self::VectorGather { .. } => "vector.gather",
            Self::VectorFaultingLoad { .. } => "vector.faulting.load",
            Self::VectorSegmentLoad { .. } => "vector.segment.load",
            Self::VectorScatter { .. } => "vector.scatter",
            Self::VectorSegmentStore { .. } => "vector.segment.store",
            Self::VectorPackLoad { .. } => "vector.pack.load",
            Self::VectorPackStore { .. } => "vector.pack.store",
            Self::VectorExtract { .. } => "vector.extract",
            Self::VectorInsert { .. } => "vector.insert",
            Self::VectorSplat { .. } => "vector.splat",
            Self::VectorShuffle { .. } => "vector.shuffle",
            Self::VectorCast { .. } => "vector.cast",
            Self::VectorReinterpret { .. } => "vector.reinterpret",
            Self::VectorPack { .. } => "vector.pack",
            Self::VectorZeroUpper { .. } => "vector.zero.upper",
            Self::VectorMaskUnary { .. } => "vector.mask.unary",
            Self::VectorMaskBinary { .. } => "vector.mask.binary",
            Self::VectorReduce { .. } => "vector.reduce",
            Self::VectorBitmask { .. } => "vector.bitmask",
            Self::Jump { .. } => "jump",
            Self::Branch { .. } => "branch",
            Self::BranchCmp { .. } => "branch.cmp",
            Self::BranchFlags { .. } => "branch.flags",
            Self::IndirectBranch { .. } => "branch.indirect",
            Self::Switch { .. } => "switch",
            Self::Return { .. } => "return",
            Self::LoadField { .. } => "load.field",
            Self::StoreField { .. } => "store.field",
            Self::LoadStaticField { .. } => "load.static.field",
            Self::StoreStaticField { .. } => "store.static.field",
            Self::LoadFieldAddr { .. } => "load.field.addr",
            Self::LoadStaticFieldAddr { .. } => "load.static.field.addr",
            Self::LoadElement { .. } => "load.element",
            Self::StoreElement { .. } => "store.element",
            Self::LoadElementAddr { .. } => "load.element.addr",
            Self::PtrAdd { .. } => "ptradd",
            Self::ArrayLength { .. } => "array.length",
            Self::LoadIndirect { .. } => "load.indirect",
            Self::StoreIndirect { .. } => "store.indirect",
            Self::NewObj { .. } => "new.obj",
            Self::NewArr { .. } => "new.arr",
            Self::CastClass { .. } => "cast.class",
            Self::IsInst { .. } => "is.inst",
            Self::Box { .. } => "box",
            Self::Unbox { .. } => "unbox",
            Self::UnboxAny { .. } => "unbox.any",
            Self::SizeOf { .. } => "sizeof",
            Self::LoadToken { .. } => "load.token",
            Self::Call { .. } => "call",
            Self::CallVirt { .. } => "call.virt",
            Self::CallIndirect { .. } => "call.indirect",
            Self::LoadFunctionPtr { .. } => "load.function.ptr",
            Self::LoadVirtFunctionPtr { .. } => "load.virt.function.ptr",
            Self::LoadArg { .. } => "load.arg",
            Self::LoadLocal { .. } => "load.local",
            Self::LoadArgAddr { .. } => "load.arg.addr",
            Self::LoadLocalAddr { .. } => "load.local.addr",
            Self::Copy { .. } => "copy",
            Self::Pop { .. } => "pop",
            Self::Throw { .. } => "throw",
            Self::Rethrow => "rethrow",
            Self::EndFinally => "end.finally",
            Self::EndFilter { .. } => "end.filter",
            Self::InterruptReturn => "interrupt.return",
            Self::Unreachable => "unreachable",
            Self::Leave { .. } => "leave",
            Self::InitBlk { .. } => "init.blk",
            Self::CopyBlk { .. } => "copy.blk",
            Self::Fence { .. } => "fence",
            Self::FloatCompareFlags { .. } => "float.compare.flags",
            Self::NativeOpaque(_) => "native.opaque",
            Self::SystemOp(data) => data.kind.kind_str(),
            Self::ComputeOp(data) => data.kind.kind_str(),
            Self::BcdAdjust(data) => data.kind.kind_str(),
            Self::VectorCrypto(data) => data.kind.kind_str(),
            Self::TileOp(data) => data.kind.kind_str(),
            Self::VectorPermute(_) => "vector.permute",
            Self::VectorMultiplyAdd(data) => data.kind.kind_str(),
            Self::VectorPackNarrow(data) => {
                if data.unsigned {
                    "vector.pack.narrow.u"
                } else {
                    "vector.pack.narrow.s"
                }
            }
            Self::VectorNarrowSaturate(data) => {
                if data.unsigned_dst {
                    "vector.narrow.saturate.u"
                } else {
                    "vector.narrow.saturate.s"
                }
            }
            Self::VectorPredicateWhile(_) => "vector.while",
            Self::VectorPredicateBreak(_) => "vector.break",
            Self::VectorComplexAdd(_) => "vector.cadd",
            Self::VectorCountAdjust(_) => "vector.countadj",
            Self::VectorExtendInLane(_) => "vector.xtl",
            Self::VectorElementCount(_) => "vector.cnt",
            Self::VectorSveAddressGen(_) => "vector.adr",
            Self::FlagAdjust(data) => data.kind.kind_str(),
            Self::VectorStructLoadReplicate(_) => "vector.ldNr",
            Self::VectorSmeMisc(_) => "vector.sme.misc",
            Self::VectorPredicateOp(_) => "vector.predop",
            Self::VectorSveCompute(_) => "vector.sve.compute",
            Self::VectorReverseChunks(_) => "vector.revchunks",
            Self::VectorMatrixMulAcc(_) => "vector.mmla",
            Self::VectorSmeOuterProduct(_) => "vector.sme.mopa",
            Self::VectorPredicateGen(_) => "vector.pgen",
            Self::VectorFpHelper(_) => "vector.fphelper",
            Self::VectorSvePermute(_) => "vector.sve.perm",
            Self::VectorTernaryLogic(_) => "vector.ternlog",
            Self::VectorDotProduct(_) => "vector.dotproduct",
            Self::VectorMultiSad(_) => "vector.mpsadbw",
            Self::VectorIntDotProduct(_) => "vector.intdot",
            Self::VectorStringCompare(_) => "vector.pcmpstr",
            Self::VectorBitfield(_) => "vector.bitfield",
            Self::VectorIntersect(_) => "vector.p2intersect",
            Self::VectorShuffleBits(_) => "vector.shufbitqmb",
            Self::VectorConditionalMove(_) => "vector.condmove",
            Self::VectorHorizontalMinPos(_) => "vector.phminposuw",
            Self::VectorComplexMul(data) => data.kind.kind_str(),
            Self::VectorClassify(_) => "vector.fpclass",
            Self::VectorHorizontalReduce(_) => "vector.hreduce",
            Self::BlockString(data) => data.kind.kind_str(),
            Self::WideCompareExchange(data) => {
                if data.wide {
                    "atomic.cmpxchg16b"
                } else {
                    "atomic.cmpxchg8b"
                }
            }
            Self::FpTranscendental { .. } => "fp.transcendental",
            Self::FpuControl { .. } => "fpu.control",
            Self::CmpXchg { .. } => "cmpxchg",
            Self::AtomicRmw { .. } => "atomic.rmw",
            Self::AtomicLoad { .. } => "atomic.load",
            Self::AtomicStore { .. } => "atomic.store",
            Self::AtomicPairLoad { .. } => "atomic.pair.load",
            Self::AtomicPairStoreConditional { .. } => "atomic.pair.store.conditional",
            Self::AtomicExchange { .. } => "atomic.exchange",
            Self::AtomicLockRmw { .. } => "atomic.lock.rmw",
            Self::AtomicStoreConditional { .. } => "atomic.store.conditional",
            Self::AtomicCmpXchg { .. } => "atomic.cmpxchg",
            Self::AtomicPairCmpXchg { .. } => "atomic.pair.cmpxchg",
            Self::InitObj { .. } => "init.obj",
            Self::CopyObj { .. } => "copy.obj",
            Self::LoadObj { .. } => "load.obj",
            Self::StoreObj { .. } => "store.obj",
            Self::Nop => "nop",
            Self::Break(op) => op.kind_str(),
            Self::Ckfinite { .. } => "ckfinite",
            Self::FpClassify { .. } => "fpclassify",
            Self::LocalAlloc { .. } => "local.alloc",
            Self::Constrained { .. } => "constrained",
            Self::Volatile => "volatile",
            Self::Unaligned { .. } => "unaligned",
            Self::TailPrefix => "tail",
            Self::Readonly => "readonly",
            Self::Phi { .. } => "phi",
        }
    }

    /// Returns a coarse, similarity-oriented token for this operation.
    ///
    /// Deliberately coarser than [`opcode_name`](Self::opcode_name): it merges
    /// variants a similarity fingerprint should treat alike (`Add` and
    /// `AddOvf` both become `"add"`, every scalar load becomes `"load"`) and
    /// folds signedness into the token (`shru` / `shrs`). The match is
    /// exhaustive — no `_` — so a new [`SsaOp`] variant is a compile error
    /// *here*, in the crate that defines the vocabulary, rather than a silent
    /// `"unknown"` in a downstream consumer.
    ///
    /// [`Call`](Self::Call) and [`CallVirt`](Self::CallVirt) return a bare
    /// `"call"`. Classifying the callee needs target-specific reference
    /// metadata this crate is generic over, so a consumer that binds a
    /// concrete target refines those two cases itself.
    #[must_use]
    pub fn coarse_token(&self) -> Cow<'static, str> {
        match self {
            Self::Const { .. } => Cow::Borrowed("const"),
            Self::Copy { .. } => Cow::Borrowed("copy"),
            Self::IntConv { .. } => Cow::Borrowed("conv"),
            Self::IntToPtr { .. } => Cow::Borrowed("inttoptr"),
            Self::PtrToInt { .. } => Cow::Borrowed("ptrtoint"),
            Self::IntToFloat { .. } => Cow::Borrowed("inttofloat"),
            Self::FloatToInt { .. } => Cow::Borrowed("floattoint"),
            Self::FloatConv { .. } => Cow::Borrowed("fconv"),
            Self::PtrAdd { .. } => Cow::Borrowed("ptradd"),
            Self::Bitcast { .. } => Cow::Borrowed("bitcast"),
            Self::Add { .. } | Self::AddOvf { .. } => Cow::Borrowed("add"),
            Self::Sub { .. } | Self::SubOvf { .. } => Cow::Borrowed("sub"),
            Self::Mul { .. } | Self::MulOvf { .. } => Cow::Borrowed("mul"),
            Self::WideMul { .. } => Cow::Borrowed("widemul"),
            Self::Div { .. } => Cow::Borrowed("div"),
            Self::Rem { .. } => Cow::Borrowed("rem"),
            Self::WideDiv { .. } => Cow::Borrowed("widediv"),
            Self::Neg { .. } => Cow::Borrowed("neg"),
            Self::And { .. } => Cow::Borrowed("and"),
            Self::Or { .. } => Cow::Borrowed("or"),
            Self::Xor { .. } => Cow::Borrowed("xor"),
            Self::Not { .. } => Cow::Borrowed("not"),
            Self::Shl { .. } => Cow::Borrowed("shl"),
            Self::Shr { unsigned, .. } => Cow::Borrowed(if *unsigned { "shru" } else { "shrs" }),
            Self::Rol { .. } => Cow::Borrowed("rol"),
            Self::Ror { .. } => Cow::Borrowed("ror"),
            Self::Rcl { .. } => Cow::Borrowed("rcl"),
            Self::Rcr { .. } => Cow::Borrowed("rcr"),
            Self::BSwap { .. } => Cow::Borrowed("bswap"),
            Self::BRev { .. } => Cow::Borrowed("brev"),
            Self::BitScanForward { .. } => Cow::Borrowed("bsf"),
            Self::BitScanReverse { .. } => Cow::Borrowed("bsr"),
            Self::Popcount { .. } => Cow::Borrowed("popcnt"),
            Self::Parity { .. } => Cow::Borrowed("parity"),
            Self::Ceq { .. } | Self::Clt { .. } | Self::Cgt { .. } => {
                Cow::Owned(self.compare_token())
            }
            Self::ReadFlags { .. } | Self::ComputeFlags { .. } => Cow::Borrowed("readflags"),
            Self::CallClobber { .. } => Cow::Borrowed("nop"),
            Self::Select { .. } => Cow::Borrowed("select"),
            Self::Phi { .. } => Cow::Borrowed("phi"),
            Self::BoolAnd { .. } => Cow::Borrowed("booland"),
            Self::BoolOr { .. } => Cow::Borrowed("boolor"),
            Self::BoolXor { .. } => Cow::Borrowed("boolxor"),
            Self::BoolNot { .. } => Cow::Borrowed("boolnot"),
            Self::LoadIndirect { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. } => Cow::Borrowed("load"),
            Self::StoreIndirect { .. } | Self::StoreStaticField { .. } => Cow::Borrowed("store"),
            Self::LoadField { .. } => Cow::Borrowed("load:member"),
            Self::StoreField { .. } => Cow::Borrowed("store:member"),
            Self::LoadElement { .. } => Cow::Borrowed("load:element"),
            Self::StoreElement { .. } => Cow::Borrowed("store:element"),
            // A call classifies to a bare token here; a consumer that can
            // resolve the callee (which needs target-specific metadata this crate
            // does not have) refines it to `call:function` / `call:rva` / etc.
            Self::Call { .. } | Self::CallVirt { .. } => Cow::Borrowed("call"),
            Self::CallIndirect { .. } => Cow::Borrowed("call:indirect"),
            Self::IndirectBranch { .. } => Cow::Borrowed("branch:indirect"),
            Self::AtomicRmw { .. } => Cow::Borrowed("atomic_rmw"),
            Self::AtomicLoad { .. } => Cow::Borrowed("atomic_load"),
            Self::AtomicStore { .. } => Cow::Borrowed("atomic_store"),
            Self::AtomicStoreConditional { .. } => Cow::Borrowed("atomic_store_cond"),
            Self::AtomicPairLoad { .. } => Cow::Borrowed("atomic_pair_load"),
            Self::AtomicPairStoreConditional { .. } => Cow::Borrowed("atomic_pair_store_cond"),
            Self::CmpXchg { .. } | Self::WideCompareExchange { .. } => Cow::Borrowed("cmpxchg"),
            Self::AtomicExchange { .. } => Cow::Borrowed("atomic_xchg"),
            Self::AtomicCmpXchg { .. } => Cow::Borrowed("atomic_cmpxchg"),
            Self::AtomicPairCmpXchg { .. } => Cow::Borrowed("atomic_pair_cmpxchg"),
            Self::AtomicLockRmw { .. } => Cow::Borrowed("atomic_lock_rmw"),
            Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorLoad { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorGather { .. }
            | Self::VectorScatter { .. }
            | Self::VectorReduce { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorSplat { .. }
            | Self::VectorBitmask { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorPack { .. }
            | Self::VectorPackLoad { .. }
            | Self::VectorPackStore { .. }
            | Self::VectorZeroUpper { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorMaskBinary { .. }
            | Self::VectorCrypto(_)
            | Self::TileOp(_)
            | Self::VectorPermute(_)
            | Self::VectorMultiplyAdd(_)
            | Self::VectorPackNarrow(_)
            | Self::VectorNarrowSaturate(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorPredicateBreak(_)
            | Self::VectorComplexAdd(_)
            | Self::VectorCountAdjust(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorSveAddressGen(_)
            | Self::VectorSveCompute(_)
            | Self::VectorPredicateOp(_)
            | Self::VectorSmeMisc(_)
            | Self::VectorStructLoadReplicate(_)
            | Self::VectorReverseChunks(_)
            | Self::VectorMatrixMulAcc(_)
            | Self::VectorSmeOuterProduct(_)
            | Self::VectorPredicateGen(_)
            | Self::VectorFpHelper(_)
            | Self::VectorSvePermute(_)
            | Self::VectorTernaryLogic(_)
            | Self::VectorComplexMul(_)
            | Self::VectorClassify(_)
            | Self::VectorHorizontalReduce(_)
            | Self::VectorDotProduct(_)
            | Self::VectorIntDotProduct(_)
            | Self::VectorMultiSad(_)
            | Self::VectorStringCompare(_)
            | Self::VectorHorizontalMinPos(_)
            | Self::VectorBitfield(_)
            | Self::VectorIntersect(_)
            | Self::VectorShuffleBits(_)
            | Self::VectorConditionalMove(_) => Cow::Borrowed("vector"),
            Self::LoadObj { .. } => Cow::Borrowed("load"),
            Self::StoreObj { .. } => Cow::Borrowed("store"),
            Self::Nop => Cow::Borrowed("nop"),
            Self::Fence { .. } => Cow::Borrowed("fence"),
            Self::NativeOpaque(data) => Cow::Owned(format!(
                "unsupported:{}",
                normalize_token_text(&data.mnemonic)
            )),
            // No `normalize_token_text` on these five: the input is a
            // `kind_str()` literal, and every registered table's keys are
            // pinned equal to their own `trim().to_ascii_lowercase()`, so
            // normalizing is an allocation that cannot change the answer.
            Self::SystemOp(data) => Cow::Owned(format!("system:{}", data.kind.kind_str())),
            Self::ComputeOp(data) => Cow::Owned(format!("compute:{}", data.kind.kind_str())),
            Self::BcdAdjust(data) => Cow::Owned(format!("compute:{}", data.kind.kind_str())),
            Self::FlagAdjust(data) => Cow::Owned(format!("compute:{}", data.kind.kind_str())),
            Self::BlockString(data) => Cow::Owned(format!("blockstring:{}", data.kind.kind_str())),
            Self::FloatCompareFlags { .. }
            | Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::BranchFlags { .. }
            | Self::Switch { .. }
            | Self::Return { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElementAddr { .. }
            | Self::ArrayLength { .. }
            | Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadVirtFunctionPtr { .. }
            | Self::Pop { .. }
            | Self::Throw { .. }
            | Self::EndFilter { .. }
            | Self::Leave { .. }
            | Self::InitBlk { .. }
            | Self::CopyBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::FpTranscendental(_)
            | Self::FpuControl(_)
            | Self::LocalAlloc { .. }
            | Self::Constrained { .. }
            | Self::Unaligned { .. }
            | Self::Break(_)
            | Self::EndFinally
            | Self::InterruptReturn
            | Self::Readonly
            | Self::Rethrow
            | Self::TailPrefix
            | Self::Unreachable
            | Self::Volatile => Cow::Borrowed("unknown"),
        }
    }

    /// Returns the comparison token (`cmp:<rel><u|s>`) for a compare-and-set
    /// op, read straight off the operation.
    fn compare_token(&self) -> String {
        // `None` (an op with no comparison payload) falls back to the `Eq`
        // spelling, matching the historical token for such ops.
        let rel = self.compare_kind().map_or("eq", CmpKind::as_str);
        let unsigned = self.arith_signedness().is_some_and(Signedness::is_unsigned);
        format!("cmp:{rel}{}", if unsigned { "u" } else { "s" })
    }

    /// Returns a canonical target-generic feature token for this operation.
    ///
    /// The token excludes variable identities and target metadata while
    /// retaining the stable opcode name, operation family, effect kind, arity,
    /// definition count, and trap behavior.
    #[must_use]
    pub fn feature_token(&self) -> SsaFeatureToken {
        SsaFeatureToken {
            opcode: self.opcode_name(),
            op_class: self.class(),
            similarity_class: self.similarity_class(),
            effect_kind: self.effects().kind,
            def_count: self.defs().count(),
            use_count: self.use_count(),
            may_throw: self.may_throw(),
        }
    }

    /// Extracts binary operation information if this is a binary operation.
    ///
    /// This method provides a uniform view of all binary operations (arithmetic,
    /// bitwise, comparison, shifts) for optimization passes that need to handle
    /// them generically.
    ///
    /// # Returns
    ///
    /// - `Some(BinaryOpInfo)` if this is a binary operation
    /// - `None` for all other operations
    ///
    /// # Supported Operations
    ///
    /// - Arithmetic: `Add`, `AddOvf`, `Sub`, `SubOvf`, `Mul`, `MulOvf`, `Div`, `Rem`
    /// - Bitwise: `And`, `Or`, `Xor`
    /// - Shifts: `Shl`, `Shr`
    /// - Comparisons: `Ceq`, `Clt`, `Cgt`
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{BinaryOpKind, SsaOp, SsaVarId}};
    ///
    /// let op = SsaOp::<MockTarget>::Add {
    ///     dest: SsaVarId::from_index(2),
    ///     left: SsaVarId::from_index(0),
    ///     right: SsaVarId::from_index(1),
    ///     flags: None,
    /// };
    /// match op.as_binary_op() {
    ///     Some(info) if info.kind == BinaryOpKind::Add => {
    ///         // Handle addition
    ///     }
    ///     Some(info) => {
    ///         // Handle other binary ops
    ///     }
    ///     None => {
    ///         // Not a binary operation
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn as_binary_op(&self) -> Option<BinaryOpInfo> {
        match *self {
            Self::Add {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Add,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::AddOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::AddOvf,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Sub {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Sub,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::SubOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::SubOvf,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Mul {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Mul,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::MulOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::MulOvf,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Div {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Div,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::Rem {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rem,
                dest,
                left,
                right,
                unsigned,
                flags,
            }),
            Self::And {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::And,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::Or {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Or,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::Xor {
                dest,
                left,
                right,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Xor,
                dest,
                left,
                right,
                unsigned: false,
                flags,
            }),
            Self::Shl {
                dest,
                value,
                amount,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Shl,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags,
            }),
            Self::Shr {
                dest,
                value,
                amount,
                unsigned,
                flags,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Shr,
                dest,
                left: value,
                right: amount,
                unsigned,
                flags,
            }),
            Self::Ceq { dest, left, right } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Ceq,
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            }),
            Self::Clt {
                dest,
                left,
                right,
                unsigned,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Clt,
                dest,
                left,
                right,
                unsigned,
                flags: None,
            }),
            Self::Cgt {
                dest,
                left,
                right,
                unsigned,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Cgt,
                dest,
                left,
                right,
                unsigned,
                flags: None,
            }),
            Self::Rol {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rol,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            Self::Ror {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Ror,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            Self::Rcl {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rcl,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            Self::Rcr {
                dest,
                value,
                amount,
            } => Some(BinaryOpInfo {
                kind: BinaryOpKind::Rcr,
                dest,
                left: value,
                right: amount,
                unsigned: false,
                flags: None,
            }),
            _ => None,
        }
    }

    /// Extracts unary operation information if this is a unary operation.
    ///
    /// This method provides a uniform view of all unary operations for
    /// optimization passes that need to handle them generically.
    ///
    /// # Returns
    ///
    /// - `Some(UnaryOpInfo)` if this is a unary operation
    /// - `None` for all other operations
    ///
    /// # Supported Operations
    ///
    /// - `Neg`: Negation
    /// - `Not`: Bitwise NOT
    /// - `Ckfinite`: Check finite
    ///
    /// # Note
    ///
    /// `Conv` is not included because it requires additional type information
    /// that doesn't fit the simple unary pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
    ///
    /// let op = SsaOp::<MockTarget>::Neg {
    ///     dest: SsaVarId::from_index(1),
    ///     operand: SsaVarId::from_index(0),
    ///     flags: None,
    /// };
    /// if let Some(info) = op.as_unary_op() {
    ///     println!("Unary {} on {}", info.kind, info.operand);
    /// }
    /// ```
    #[must_use]
    pub fn as_unary_op(&self) -> Option<UnaryOpInfo> {
        match *self {
            Self::Neg { dest, operand, .. } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Neg,
                dest,
                operand,
            }),
            Self::Not { dest, operand, .. } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Not,
                dest,
                operand,
            }),
            Self::Ckfinite { dest, operand } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Ckfinite,
                dest,
                operand,
            }),
            Self::BSwap { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BSwap,
                dest,
                operand: src,
            }),
            Self::BRev { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BRev,
                dest,
                operand: src,
            }),
            Self::BitScanForward { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BitScanForward,
                dest,
                operand: src,
            }),
            Self::BitScanReverse { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::BitScanReverse,
                dest,
                operand: src,
            }),
            Self::Popcount { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Popcount,
                dest,
                operand: src,
            }),
            Self::Parity { dest, src } => Some(UnaryOpInfo {
                kind: UnaryOpKind::Parity,
                dest,
                operand: src,
            }),
            _ => None,
        }
    }

    /// Returns the stack effect (pops, pushes) for this SSA operation.
    ///
    /// This represents the net effect on the evaluation stack when the operation
    /// is executed, assuming operands have already been loaded. The effect is:
    /// - pops: number of values consumed from the stack
    /// - pushes: number of values produced to the stack
    ///
    /// Note: This tracks the operation's own effect, not the loading of operands
    /// (which is tracked separately during codegen).
    #[must_use]
    pub fn stack_effect(&self) -> (u32, u32) {
        match self {
            // Address computation pops the base and its optional index, pushes
            // the address. (A native-only op; never on the CIL eval stack.)
            Self::PtrAdd { index, .. } => (1u32.saturating_add(u32::from(index.is_some())), 1),
            // Flags computed from N inputs - pop N, push 1.
            Self::ComputeFlags { inputs, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = inputs.len() as u32;
                (pops, 1)
            }
            // Call-clobber defines N caller-saved registers - pop 0, push N.
            Self::CallClobber { outputs } => {
                #[allow(clippy::cast_possible_truncation)]
                let pushes = outputs.len() as u32;
                (0, pushes)
            }
            // Binary arithmetic, comparisons, and array access - pop 2, push 1
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::AddOvf { .. }
            | Self::SubOvf { .. }
            | Self::MulOvf { .. }
            | Self::FloatCompareFlags { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Rcl { .. }
            | Self::Rcr { .. }
            | Self::Ceq { .. }
            | Self::Clt { .. }
            | Self::Cgt { .. }
            | Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::LoadElement { .. }
            | Self::LoadElementAddr { .. }
            | Self::VectorBinary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorMaskBinary { .. }
            | Self::VectorPredicatedUnary {
                passthrough: None, ..
            }
            | Self::VectorPack {
                passthrough: None, ..
            }
            | Self::VectorPackLoad {
                passthrough: None, ..
            } => (2, 1),

            // Pop 3, push 1 (Select, CmpXchg)
            Self::Select { .. }
            | Self::CmpXchg { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary {
                passthrough: Some(_),
                ..
            }
            | Self::VectorPredicatedBinary {
                passthrough: None, ..
            }
            | Self::VectorPack {
                passthrough: Some(_),
                ..
            }
            | Self::VectorPackLoad {
                passthrough: Some(_),
                ..
            } => (3, 1),

            Self::VectorPredicatedBinary {
                passthrough: Some(_),
                ..
            }
            | Self::VectorPredicatedTernary {
                passthrough: None, ..
            } => (4, 1),

            Self::VectorPredicatedTernary {
                passthrough: Some(_),
                ..
            } => (5, 1),

            Self::AtomicCmpXchg { success, .. } => {
                (3, 1_u32.saturating_add(u32::from(success.is_some())))
            }
            Self::AtomicPairCmpXchg { .. } => (5, 2),
            Self::WideMul { .. } => (2, 2),
            Self::WideDiv { .. } => (3, 2),

            // AtomicRmw is pop 2, push 1
            Self::AtomicRmw { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. } => (2, 1),
            Self::AtomicPairStoreConditional { .. } => (3, 1),

            // Control flow
            Self::Return { value } => {
                if value.is_some() {
                    (1, 0) // pop return value
                } else {
                    (0, 0) // void return
                }
            }
            // No stack effect (0, 0)
            Self::Jump { .. }
            | Self::Rethrow
            | Self::Leave { .. }
            | Self::EndFinally
            | Self::Copy { .. }
            | Self::Nop
            | Self::Break(_)
            | Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly
            | Self::Phi { .. }
            | Self::Fence { .. }
            | Self::InterruptReturn
            | Self::VectorZeroUpper { .. }
            | Self::Unreachable => (0, 0),

            // Pop 1, push 0 (1, 0)
            Self::Branch { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. }
            | Self::Throw { .. }
            | Self::EndFilter { .. }
            | Self::Pop { .. }
            | Self::StoreStaticField { .. }
            | Self::InitObj { .. }
            | Self::BranchFlags { .. } => (1, 0),

            // Pop 2, push 0 (2, 0)
            Self::BranchCmp { .. }
            | Self::StoreField { .. }
            | Self::StoreIndirect { .. }
            | Self::AtomicStore { .. }
            | Self::StoreObj { .. }
            | Self::CopyObj { .. }
            | Self::VectorStore { .. } => (2, 0),

            Self::VectorMaskedStore { .. } | Self::VectorPackStore { .. } => (3, 0),
            Self::VectorScatter { .. } => (4, 0),
            Self::VectorSegmentStore { values, mask, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = values.len() as u32;
                (
                    pops.saturating_add(1_u32.saturating_add(u32::from(mask.is_some()))),
                    0,
                )
            }

            // Pop 3, push 0 (3, 0)
            Self::StoreElement { .. } | Self::InitBlk { .. } | Self::CopyBlk { .. } => (3, 0),

            // Pop 0, push 1 (0, 1)
            Self::LoadStaticField { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::Const { .. } => (0, 1),

            // Pop 1, push 1 (1, 1)
            Self::Neg { .. }
            | Self::Not { .. }
            | Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::Bitcast { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. }
            | Self::LoadField { .. }
            | Self::LoadFieldAddr { .. }
            | Self::ArrayLength { .. }
            | Self::NewArr { .. }
            | Self::LoadIndirect { .. }
            | Self::AtomicLoad { .. }
            | Self::LoadObj { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::LoadVirtFunctionPtr { .. }
            | Self::LocalAlloc { .. }
            | Self::ReadFlags { .. }
            | Self::BoolNot { .. }
            | Self::VectorUnary { .. }
            | Self::VectorLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorExtract { .. }
            | Self::VectorSplat { .. }
            | Self::VectorShuffle { right: None, .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. } => (1, 1),

            Self::AtomicPairLoad { .. } => (1, 2),

            Self::VectorInsert { .. } | Self::VectorShuffle { right: Some(_), .. } => (2, 1),
            Self::VectorMaskedLoad {
                passthrough: None, ..
            } => (2, 1),
            Self::VectorMaskedLoad {
                passthrough: Some(_),
                ..
            } => (3, 1),
            Self::VectorGather {
                passthrough: None, ..
            } => (3, 1),
            Self::VectorGather {
                passthrough: Some(_),
                ..
            } => (4, 1),
            Self::VectorFaultingLoad {
                mask, passthrough, ..
            } => (
                1_u32
                    .saturating_add(u32::from(mask.is_some()))
                    .saturating_add(u32::from(passthrough.is_some())),
                1,
            ),
            Self::VectorSegmentLoad { dests, mask, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pushes = dests.len() as u32;
                (1_u32.saturating_add(u32::from(mask.is_some())), pushes)
            }

            // Call operations - stack effect depends on args and return type
            Self::Call { dest, args, .. } | Self::CallVirt { dest, args, .. } => {
                // args.len() will never exceed u32 for CIL methods
                #[allow(clippy::cast_possible_truncation)]
                let pops = args.len() as u32;
                let pushes = u32::from(dest.is_some());
                (pops, pushes)
            }
            Self::CallIndirect { dest, args, .. } => {
                // Indirect call pops args + function pointer
                // args.len() will never exceed u32 for CIL methods
                #[allow(clippy::cast_possible_truncation)]
                let pops = (args.len() as u32).saturating_add(1);
                let pushes = u32::from(dest.is_some());
                (pops, pushes)
            }
            Self::NewObj { args, .. } => {
                // newobj pops constructor args, always pushes new instance
                // args.len() will never exceed u32 for CIL methods
                #[allow(clippy::cast_possible_truncation)]
                let pops = args.len() as u32;
                (pops, 1)
            }
            // A boxed payload pops one operand per input and pushes one per
            // output; the count lives once, in `KindedOperands`.
            Self::NativeOpaque(data) => data.operand_counts(),
            Self::SystemOp(data) => data.operand_counts(),
            Self::ComputeOp(data) => data.operand_counts(),
            Self::BcdAdjust(data) => data.operand_counts(),
            Self::VectorCrypto(data) => data.operand_counts(),
            Self::TileOp(data) => data.operand_counts(),
            Self::VectorPermute(data) => data.operand_counts(),
            Self::VectorMultiplyAdd(data) => data.operand_counts(),
            Self::VectorPackNarrow(data) => data.operand_counts(),
            Self::VectorNarrowSaturate(data) => data.operand_counts(),
            Self::VectorPredicateWhile(data) => data.operand_counts(),
            Self::VectorPredicateBreak(data) => data.operand_counts(),
            Self::VectorComplexAdd(data) => data.operand_counts(),
            Self::VectorCountAdjust(data) => data.operand_counts(),
            Self::VectorExtendInLane(data) => data.operand_counts(),
            Self::VectorElementCount(data) => data.operand_counts(),
            Self::VectorSveAddressGen(data) => data.operand_counts(),
            Self::FlagAdjust(data) => data.operand_counts(),
            Self::VectorStructLoadReplicate(data) => data.operand_counts(),
            Self::VectorSmeMisc(data) => data.operand_counts(),
            Self::VectorPredicateOp(data) => data.operand_counts(),
            Self::VectorSveCompute(data) => data.operand_counts(),
            Self::VectorReverseChunks(data) => data.operand_counts(),
            Self::VectorMatrixMulAcc(data) => data.operand_counts(),
            Self::VectorSmeOuterProduct(data) => data.operand_counts(),
            Self::VectorPredicateGen(data) => data.operand_counts(),
            Self::VectorFpHelper(data) => data.operand_counts(),
            Self::VectorSvePermute(data) => data.operand_counts(),
            Self::VectorTernaryLogic(data)
            | Self::VectorMultiSad(data)
            | Self::VectorClassify(data) => data.operand_counts(),
            Self::VectorDotProduct(data) => data.operand_counts(),
            Self::VectorIntDotProduct(data) => data.operand_counts(),
            Self::VectorStringCompare(data) => data.operand_counts(),
            Self::VectorBitfield(data) => data.operand_counts(),
            Self::VectorIntersect(data) => data.operand_counts(),
            Self::VectorShuffleBits(data) => data.operand_counts(),
            Self::VectorConditionalMove(data) => data.operand_counts(),
            Self::VectorHorizontalMinPos(data) => data.operand_counts(),
            Self::VectorComplexMul(data) => data.operand_counts(),
            Self::VectorHorizontalReduce(data) => data.operand_counts(),
            Self::BlockString(data) => data.operand_counts(),
            Self::WideCompareExchange(data) => data.operand_counts(),
            Self::FpTranscendental(data) => data.operand_counts(),
            Self::FpuControl(data) => data.operand_counts(),
        }
    }
}

impl<T: Target> SsaOp<T> {
    /// Tries to infer the result type of this SSA operation.
    ///
    /// Dispatches per opcode group to small `Target` queries; each host
    /// supplies its own type lattice via `Target::result_type_for_const`,
    /// `arithmetic_result_type`, etc. Returns `None` for ops the host can't
    /// answer or for context-dependent ops resolved later from
    /// `SsaInstruction::result_type()`.
    #[must_use]
    pub fn infer_result_type(&self) -> Option<T::Type> {
        match self {
            Self::Const { value, .. } => T::result_type_for_const(value),
            // Type conversions carry their target.
            Self::IntConv { target, .. }
            | Self::IntToPtr { target, .. }
            | Self::PtrToInt { target, .. }
            | Self::IntToFloat { target, .. }
            | Self::FloatToInt { target, .. }
            | Self::FloatConv { target, .. }
            | Self::Bitcast { target, .. } => Some(target.clone()),
            // Comparisons produce bool (or whatever the host defines).
            Self::Ceq { .. }
            | Self::Clt { .. }
            | Self::Cgt { .. }
            | Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => T::comparison_result_type(),
            // Arithmetic/bitwise ops + SizeOf — per-host arithmetic type.
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Rcl { .. }
            | Self::Rcr { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::Neg { .. }
            | Self::Not { .. }
            | Self::AddOvf { .. }
            | Self::SubOvf { .. }
            | Self::MulOvf { .. }
            | Self::SizeOf { .. } => T::arithmetic_result_type(),
            Self::UnboxAny { value_type, .. } | Self::LoadObj { value_type, .. } => {
                T::value_type_from_ref(value_type)
            }
            // Context-dependent ops — resolved from SsaInstruction::result_type().
            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::Call { dest: Some(_), .. }
            | Self::CallVirt { dest: Some(_), .. }
            | Self::CallIndirect { dest: Some(_), .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. } => None,
            Self::Box { .. }
            | Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. } => T::object_result_type(),
            Self::ArrayLength { .. }
            | Self::LocalAlloc { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. } => T::native_int_result_type(),
            Self::Ckfinite { .. } => T::ckfinite_result_type(),
            Self::FpClassify { .. } => T::native_int_result_type(),
            Self::Parity { .. } | Self::ReadFlags { .. } | Self::ComputeFlags { .. } => {
                T::comparison_result_type()
            }
            Self::VectorLoad { vector_type, .. }
            | Self::VectorMaskedLoad { vector_type, .. }
            | Self::VectorBroadcastLoad { vector_type, .. }
            | Self::VectorGather { vector_type, .. }
            | Self::VectorFaultingLoad { vector_type, .. }
            | Self::VectorSegmentLoad { vector_type, .. }
            | Self::VectorSplat { vector_type, .. }
            | Self::VectorPack { vector_type, .. }
            | Self::VectorPackLoad { vector_type, .. }
            | Self::VectorCast {
                target_type: vector_type,
                ..
            }
            | Self::VectorReinterpret {
                target_type: vector_type,
                ..
            } => Some(vector_type.clone()),
            Self::LoadFunctionPtr { .. } | Self::LoadVirtFunctionPtr { .. } => {
                T::function_ptr_result_type()
            }
            Self::LoadElement { elem_type, .. } => Some(elem_type.clone()),
            Self::LoadIndirect { value_type, .. } => Some(value_type.clone()),
            // LoadToken: assembly-free inference is `None`; the variable's
            // declared type set during SSA construction provides the real one.
            Self::LoadToken { .. } => None,
            Self::Unbox { value_type, .. } => T::byref_value_type_from_ref(value_type),
            Self::LoadElementAddr { elem_type, .. } => T::byref_class_type_from_ref(elem_type),
            Self::PtrAdd { result_type, .. } => Some(result_type.clone()),
            Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. } => None,
            _ => None,
        }
    }
}
