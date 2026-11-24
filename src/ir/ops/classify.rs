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
    ir::{
        ops::{
            calls::{CallData, Callee},
            def::SsaOp,
            execution::Kernel,
            kinds::{
                CmpKind, FloatFunctionKind, MemoryTagKind, RuntimeValueKind, Signedness,
                SsaFeatureToken, SsaOpClass, SsaSimilarityClass,
            },
            native::{BinaryOpInfo, BinaryOpKind, UnaryOpInfo, UnaryOpKind},
            operands::KindedOperands,
            vector::{
                ControlledKind, VectorBinaryKind, VectorCompareKind, VectorElement,
                VectorElementKind, VectorReduceKind, VectorShuffleSelector, VectorTernaryKind,
                VectorUnaryKind,
            },
        },
        variable::SsaVarId,
    },
    target::Target,
};

impl<T: Target> SsaOp<T> {
    /// Whether the operation may compute on floats in a way that raises an
    /// exception, and so is what a [`Self::FloatRaise`] of an instruction is
    /// of: float arithmetic and its square root, rounds, estimates, compares,
    /// conversions and transcendentals, scalar or lane-wise.
    ///
    /// It errs toward including: an operation whose lanes it cannot tell
    /// are integers (an unknown element, a dot product or a complex multiply
    /// of either kind) is included, since a raise of an integer computation
    /// raises nothing, and a float computation left out would raise nothing
    /// it does. What moves, selects, reinterprets or rearranges a value, a
    /// negation, an absolute value and a sign copy (IEEE 754's
    /// non-computational operations) and bitwise lane operations are not; a
    /// negation or absolute value computed as arithmetic
    /// ([`VectorUnaryKind::NegArithmetic`],
    /// [`VectorUnaryKind::AbsArithmetic`]) is.
    ///
    /// # Arguments
    ///
    /// * `is_float` - Whether a scalar variable holds a float, `None` where
    ///   its type is not known.
    #[must_use]
    pub fn may_raise_float_exceptions(&self, is_float: impl Fn(SsaVarId) -> Option<bool>) -> bool {
        let lanes = |element: &VectorElement| element.kind != VectorElementKind::Integer;
        let unary = |kind: VectorUnaryKind| {
            matches!(
                kind,
                VectorUnaryKind::Sqrt
                    | VectorUnaryKind::Round
                    | VectorUnaryKind::RoundExact
                    | VectorUnaryKind::RoundInt32
                    | VectorUnaryKind::RoundInt64
                    | VectorUnaryKind::NegArithmetic
                    | VectorUnaryKind::AbsArithmetic
                    | VectorUnaryKind::Truncate
                    | VectorUnaryKind::Floor
                    | VectorUnaryKind::Ceil
                    | VectorUnaryKind::RoundNearestEven
                    | VectorUnaryKind::RoundNearestAway
                    | VectorUnaryKind::Reciprocal
                    | VectorUnaryKind::ReciprocalSqrt
                    | VectorUnaryKind::GetExponent
                    | VectorUnaryKind::Fraction
                    | VectorUnaryKind::Exp2
                    | VectorUnaryKind::Log2
                    | VectorUnaryKind::HorizontalAdd
            )
        };
        let binary = |kind: VectorBinaryKind| {
            matches!(
                kind,
                VectorBinaryKind::Add
                    | VectorBinaryKind::Sub
                    | VectorBinaryKind::Mul
                    | VectorBinaryKind::Div
                    | VectorBinaryKind::Rem
                    | VectorBinaryKind::Min
                    | VectorBinaryKind::Max
                    | VectorBinaryKind::MinNum
                    | VectorBinaryKind::MaxNum
                    | VectorBinaryKind::MinLess
                    | VectorBinaryKind::MaxGreater
                    | VectorBinaryKind::MinMagnitude
                    | VectorBinaryKind::MaxMagnitude
                    | VectorBinaryKind::MinimumNumber
                    | VectorBinaryKind::MaximumNumber
                    | VectorBinaryKind::MinimumMagnitude
                    | VectorBinaryKind::MaximumMagnitude
                    | VectorBinaryKind::MinimumMagnitudeNumber
                    | VectorBinaryKind::MaximumMagnitudeNumber
                    | VectorBinaryKind::MinAbs
                    | VectorBinaryKind::MaxAbs
                    | VectorBinaryKind::AbsDiff
                    | VectorBinaryKind::AddAbs
                    | VectorBinaryKind::MulNegate
                    | VectorBinaryKind::Scale
                    | VectorBinaryKind::ScaleInteger
                    | VectorBinaryKind::ReciprocalIter1
                    | VectorBinaryKind::ReciprocalIter2
                    | VectorBinaryKind::ReciprocalSqrtIter1
            )
        };
        let ternary = |kind: VectorTernaryKind| {
            kind.fused_signs().is_some() || kind == VectorTernaryKind::FixupNan
        };
        match self {
            Self::Add { dest, .. }
            | Self::Sub { dest, .. }
            | Self::Mul { dest, .. }
            | Self::Div { dest, .. }
            | Self::Rem { dest, .. } => is_float(*dest) != Some(false),
            Self::Ceq { left, .. } | Self::Clt { left, .. } | Self::Cgt { left, .. } => {
                is_float(*left) != Some(false)
            }
            Self::FloatCompareFlags { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::FloatFunction(_) => true,
            Self::VectorUnary { kind, element, .. }
            | Self::VectorPredicatedUnary { kind, element, .. } => lanes(element) && unary(*kind),
            Self::VectorBinary { kind, element, .. }
            | Self::VectorPredicatedBinary { kind, element, .. } => lanes(element) && binary(*kind),
            Self::VectorTernary { kind, element, .. }
            | Self::VectorPredicatedTernary { kind, element, .. } => {
                lanes(element) && ternary(*kind)
            }
            Self::VectorCompare { element, .. } => lanes(element),
            Self::VectorCast { from, to, .. } => from.is_float() || to.is_float(),
            Self::VectorReduce { kind, element, .. } => {
                lanes(element)
                    && matches!(
                        kind,
                        VectorReduceKind::Add
                            | VectorReduceKind::Mul
                            | VectorReduceKind::Min
                            | VectorReduceKind::Max
                            | VectorReduceKind::MinNum
                            | VectorReduceKind::MaxNum
                            | VectorReduceKind::MinimumNumber
                            | VectorReduceKind::MaximumNumber
                            | VectorReduceKind::MinimumMagnitude
                            | VectorReduceKind::MaximumMagnitude
                            | VectorReduceKind::MinimumMagnitudeNumber
                            | VectorReduceKind::MaximumMagnitudeNumber
                    )
            }
            Self::Contract(data) => data.kind.raises_float_exceptions(),
            // A lane computation reports what it raises in its status result.
            Self::Compute(data) => match data.kind.kernel {
                Kernel::FloatHelper(_) => true,
                Kernel::FloatControlled { kind, .. } => kind != ControlledKind::Classify,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns the high-level operation family for this opcode.
    ///
    /// Partitions every [`SsaOp`] variant into one [`SsaOpClass`]. The match is
    /// exhaustive with no `_` arm, so adding a variant is a compile error here,
    /// in the crate that owns the taxonomy, rather than a silent
    /// misclassification downstream. One split is not visible from the variant
    /// name alone: flag-producing arithmetic (an `Add`/`Sub`/`Shl`/... whose
    /// `flags` field is `Some`) classifies as [`SsaOpClass::Flags`], while the
    /// same opcode without a flags destination is [`SsaOpClass::Scalar`].
    #[must_use]
    pub const fn class(&self) -> SsaOpClass {
        match self {
            Self::Nop
            | Self::Phi { .. }
            | Self::Copy { .. }
            | Self::Pop { .. }
            | Self::RuntimeValue { .. }
            | Self::CallClobber { .. } => SsaOpClass::Synthetic,

            Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. } => SsaOpClass::Boolean,

            Self::FloatRaise { .. } => SsaOpClass::Flags,
            Self::ReadFlags { .. }
            | Self::BranchFlags { .. }
            | Self::MergeFlags { .. }
            | Self::InsertFlag { .. } => SsaOpClass::Flags,

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
            | Self::VectorSlice { .. }
            | Self::VectorInsertSlice { .. }
            | Self::VectorConcat { .. }
            | Self::VectorPack { .. }
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
            | Self::ArgNext { .. }
            | Self::ArgStep { .. }
            | Self::StoreIndirect { .. }
            | Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::LocalAlloc { .. }
            | Self::InitBlk { .. }
            | Self::CopyBlk { .. }
            | Self::MoveBlk { .. }
            | Self::CompareBlk(_)
            | Self::ScanBlk { .. }
            | Self::LoadBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::LoadObj { .. }
            | Self::StoreObj { .. }
            | Self::Fence { .. } => SsaOpClass::Memory,

            Self::AtomicCondRmw { .. }
            | Self::AtomicLoad { .. }
            | Self::AtomicStore { .. }
            | Self::AtomicPairLoad { .. }
            | Self::AtomicPairStoreConditional { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. }
            | Self::AtomicCmpXchg { .. }
            | Self::AtomicPairCmpXchg { .. }
            | Self::AtomicPairRmw { .. }
            | Self::AtomicCheckedRmw { .. } => SsaOpClass::Atomic,

            Self::Call(_) | Self::LoadFunctionPtr { .. } | Self::LoadVirtFunctionPtr { .. } => {
                SsaOpClass::Call
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
            | Self::Break(_) => SsaOpClass::Control,

            Self::SystemOp(_) => SsaOpClass::Call,
            Self::Compute(data) => {
                if data.kind.lanes.is_none() && data.kind.element.scalar {
                    SsaOpClass::Scalar
                } else {
                    SsaOpClass::Vector
                }
            }
            Self::Contract(_) => SsaOpClass::Vector,
            Self::VectorPredicateWhile(_) => SsaOpClass::Vector,
            Self::VectorExtendInLane(_) => SsaOpClass::Vector,
            Self::VectorElementCount(_) => SsaOpClass::Vector,
            Self::VectorStructLoadReplicate(_) => SsaOpClass::Vector,
            Self::VectorSveCompute(_) => SsaOpClass::Vector,
            Self::FloatFunction { .. } => SsaOpClass::Scalar,
            Self::FpuControl { .. } => SsaOpClass::Memory,

            Self::WideMul { .. } | Self::WideDiv { .. } => SsaOpClass::WideArithmetic,

            Self::FloatCompareFlags { .. } => SsaOpClass::Scalar,

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
            | Self::ArgList { .. }
            | Self::MakeTypedRef { .. }
            | Self::TypedRefValue { .. }
            | Self::TypedRefType { .. }
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
            | Self::BranchCmp { unsigned, .. } => Some(Signedness::from_unsigned(*unsigned)),
            Self::VectorCompare { element, .. } => match element.signed {
                Some(signed) => Some(Signedness::from_unsigned(!signed)),
                None => None,
            },
            _ => None,
        }
    }

    /// Returns the lanes a lane-wise operation states it computes on, with the
    /// operand they lay out: a vector unary, binary, ternary, compare or
    /// reduction, predicated or not. `None` for any other operation.
    #[must_use]
    pub const fn vector_element(&self) -> Option<(SsaVarId, VectorElement)> {
        match self {
            Self::VectorUnary { value, element, .. }
            | Self::VectorPredicatedUnary { value, element, .. }
            | Self::VectorReduce { value, element, .. } => Some((*value, *element)),
            Self::VectorBinary { left, element, .. }
            | Self::VectorPredicatedBinary { left, element, .. }
            | Self::VectorCompare { left, element, .. } => Some((*left, *element)),
            Self::VectorTernary { first, element, .. }
            | Self::VectorPredicatedTernary { first, element, .. } => Some((*first, *element)),
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
                // Unordered / ordered / not-lt / not-le / unordered-or-equal /
                // ordered-not-equal / constant predicates have no clean scalar
                // [`CmpKind`] equivalent.
                VectorCompareKind::Unordered
                | VectorCompareKind::Ordered
                | VectorCompareKind::UnorderedEq
                | VectorCompareKind::OrderedNe
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
            | Self::RuntimeValue { .. }
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
            | Self::MergeFlags { .. }
            | Self::InsertFlag { .. }
            | Self::FloatRaise { .. } => SsaSimilarityClass::Flags,

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

            Self::Shl { .. } | Self::Shr { .. } | Self::Rol { .. } | Self::Ror { .. } => {
                SsaSimilarityClass::ShiftRotate
            }

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
            | Self::UnboxAny { .. }
            | Self::MakeTypedRef { .. }
            | Self::TypedRefValue { .. }
            | Self::TypedRefType { .. } => SsaSimilarityClass::Conversion,

            Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::ArgList { .. }
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
            | Self::ArgStep { .. }
            | Self::VectorLoad { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorGather { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorPackLoad { .. }
            | Self::CompareBlk(_)
            | Self::ScanBlk { .. }
            | Self::LoadBlk { .. } => SsaSimilarityClass::MemoryRead,

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

            Self::CopyBlk { .. }
            | Self::MoveBlk { .. }
            | Self::InitBlk { .. }
            | Self::ArgNext { .. }
            | Self::CopyObj { .. } => SsaSimilarityClass::MemoryReadWrite,

            Self::NewObj { .. } | Self::NewArr { .. } | Self::LocalAlloc { .. } => {
                SsaSimilarityClass::Allocation
            }

            Self::AtomicCondRmw { .. }
            | Self::AtomicLoad { .. }
            | Self::AtomicStore { .. }
            | Self::AtomicPairLoad { .. }
            | Self::AtomicPairStoreConditional { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. }
            | Self::AtomicCmpXchg { .. }
            | Self::AtomicPairCmpXchg { .. }
            | Self::AtomicPairRmw { .. }
            | Self::AtomicCheckedRmw { .. } => SsaSimilarityClass::Atomic,

            Self::Fence { .. } => SsaSimilarityClass::Fence,

            Self::Call(_) => SsaSimilarityClass::Call,

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
            | Self::VectorSlice { .. }
            | Self::VectorInsertSlice { .. }
            | Self::VectorConcat { .. }
            | Self::VectorPack { .. }
            | Self::VectorReduce { .. }
            | Self::Contract(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorStructLoadReplicate(_)
            | Self::VectorSveCompute(_)
            | Self::VectorBitmask { .. } => SsaSimilarityClass::Vector,

            Self::WideMul { .. } | Self::WideDiv { .. } => SsaSimilarityClass::WideArithmetic,

            Self::SystemOp(_) => SsaSimilarityClass::Call,
            Self::Compute(data) => match data.kind.kernel {
                Kernel::BitDeposit
                | Kernel::BitExtract
                | Kernel::BitGroup
                | Kernel::CounterPredicate { .. }
                | Kernel::PredicateCounter { .. }
                | Kernel::PointerAuth(_)
                | Kernel::ScalarCipher(_)
                | Kernel::MemoryTag(MemoryTagKind::Add) => SsaSimilarityClass::Bitwise,
                Kernel::Checksum(_) | Kernel::CarrylessMultiply { .. }
                    if data.kind.lanes.is_none() && data.kind.element.scalar =>
                {
                    SsaSimilarityClass::Arithmetic
                }
                Kernel::VectorShapeSupported
                | Kernel::VectorLengthChoice
                | Kernel::VectorFloatSupported
                | Kernel::VectorCryptoSupported
                | Kernel::DecimalAdjust { .. } => SsaSimilarityClass::Arithmetic,
                Kernel::FloatToIntModular => SsaSimilarityClass::Conversion,
                Kernel::Random { .. } | Kernel::MemoryTag(MemoryTagKind::Random) => {
                    SsaSimilarityClass::Call
                }
                _ => SsaSimilarityClass::Vector,
            },
            Self::FloatFunction { .. } => SsaSimilarityClass::Arithmetic,
            Self::FpuControl(data) if data.kind.restores_image() => SsaSimilarityClass::MemoryRead,
            Self::FpuControl(_) => SsaSimilarityClass::MemoryWrite,
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
            Self::MergeFlags { .. } => "flags.merge",
            Self::FloatRaise { .. } => "float.raise",
            Self::InsertFlag { .. } => "flags.insert",
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
            Self::VectorShuffle {
                selector: VectorShuffleSelector::Mask(_),
                ..
            } => "vector.shuffle",
            Self::VectorShuffle {
                selector: VectorShuffleSelector::Index(_),
                ..
            } => "vector.shuffle.index",
            Self::VectorCast { .. } => "vector.cast",
            Self::VectorReinterpret { .. } => "vector.reinterpret",
            Self::VectorSlice { .. } => "vector.slice",
            Self::VectorInsertSlice { .. } => "vector.insert.slice",
            Self::VectorConcat { .. } => "vector.concat",
            Self::VectorPack { .. } => "vector.pack",
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
            Self::ArgList { .. } => "arglist",
            Self::ArgNext { .. } => "arg.next",
            Self::ArgStep { .. } => "arg.step",
            Self::MakeTypedRef { .. } => "make.typedref",
            Self::TypedRefValue { .. } => "typedref.value",
            Self::TypedRefType { .. } => "typedref.type",
            Self::Call(call) => match call.callee {
                Callee::Direct(_) => "call",
                Callee::Virtual(_) => "call.virt",
                Callee::Constrained { instance: true, .. } => "call.constrained",
                Callee::Constrained {
                    instance: false, ..
                } => "call.constrained.static",
                Callee::Indirect { .. } => "call.indirect",
            },
            Self::LoadFunctionPtr {
                constraint: None, ..
            } => "load.function.ptr",
            Self::LoadFunctionPtr {
                constraint: Some(_),
                ..
            } => "load.function.ptr.constrained",
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
            Self::CompareBlk(_) => "compare.blk",
            Self::ScanBlk { .. } => "scan.blk",
            Self::LoadBlk { .. } => "load.blk",
            Self::MoveBlk { .. } => "move.blk",
            Self::Fence { .. } => "fence",
            Self::FloatCompareFlags { .. } => "float.compare.flags",
            Self::SystemOp(data) => data.kind.kind_str(),
            Self::Compute(data) => data.kind.kernel.kind_str(),
            Self::Contract(data) => data.kind.kind_str(),
            Self::VectorPredicateWhile(_) => "vector.while",
            Self::VectorExtendInLane(_) => "vector.xtl",
            Self::VectorElementCount(_) => "vector.cnt",
            Self::VectorStructLoadReplicate(_) => "vector.ldNr",
            Self::VectorSveCompute(_) => "vector.sve.compute",
            Self::FloatFunction { .. } => "float.function",
            Self::FpuControl(data) => data.kind.name(),
            Self::AtomicCondRmw { .. } => "atomic.cond.rmw",
            Self::AtomicLoad { .. } => "atomic.load",
            Self::AtomicStore { .. } => "atomic.store",
            Self::AtomicPairLoad { .. } => "atomic.pair.load",
            Self::AtomicPairStoreConditional { .. } => "atomic.pair.store.conditional",
            Self::AtomicExchange { .. } => "atomic.exchange",
            Self::AtomicLockRmw { .. } => "atomic.lock.rmw",
            Self::AtomicStoreConditional { .. } => "atomic.store.conditional",
            Self::AtomicCmpXchg { .. } => "atomic.cmpxchg",
            Self::AtomicPairCmpXchg { .. } => "atomic.pair.cmpxchg",
            Self::AtomicPairRmw { .. } => "atomic.pair.rmw",
            Self::AtomicCheckedRmw { .. } => "atomic.checked.rmw",
            Self::InitObj { .. } => "init.obj",
            Self::CopyObj { .. } => "copy.obj",
            Self::LoadObj { .. } => "load.obj",
            Self::StoreObj { .. } => "store.obj",
            Self::Nop => "nop",
            Self::Break(op) => op.kind_str(),
            Self::Ckfinite { .. } => "ckfinite",
            Self::FpClassify { .. } => "fpclassify",
            Self::LocalAlloc { .. } => "local.alloc",
            Self::Phi { .. } => "phi",
            Self::RuntimeValue {
                kind: RuntimeValueKind::ExceptionObject,
                ..
            } => "runtime.exception",
            Self::RuntimeValue {
                kind: RuntimeValueKind::HandlerEntry,
                ..
            } => "runtime.handler-entry",
            Self::RuntimeValue {
                kind: RuntimeValueKind::Unspecified,
                ..
            } => "runtime.unspecified",
        }
    }

    /// Returns a coarse, similarity-oriented token for this operation.
    ///
    /// Deliberately coarser than [`opcode_name`](Self::opcode_name): it merges
    /// variants a similarity fingerprint should treat alike (`Add` and
    /// `AddOvf` both become `"add"`, every scalar load becomes `"load"`) and
    /// folds signedness into the token (`shru` / `shrs`). The match is
    /// exhaustive - no `_` - so a new [`SsaOp`] variant is a compile error
    /// *here*, in the crate that defines the vocabulary, rather than a silent
    /// `"unknown"` in a downstream consumer.
    ///
    /// A [`Call`](Self::Call) that names its method returns a bare `"call"`.
    /// Classifying the callee needs target-specific reference metadata this
    /// crate is generic over, so a consumer that binds a concrete target
    /// refines it itself.
    #[must_use]
    pub fn coarse_token(&self) -> Cow<'static, str> {
        match self {
            Self::Const { .. } => Cow::Borrowed("const"),
            Self::RuntimeValue { .. } => Cow::Borrowed("runtime"),
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
            Self::BSwap { .. } => Cow::Borrowed("bswap"),
            Self::BRev { .. } => Cow::Borrowed("brev"),
            Self::BitScanForward { .. } => Cow::Borrowed("bsf"),
            Self::BitScanReverse { .. } => Cow::Borrowed("bsr"),
            Self::Popcount { .. } => Cow::Borrowed("popcnt"),
            Self::Parity { .. } => Cow::Borrowed("parity"),
            Self::Ceq { .. } | Self::Clt { .. } | Self::Cgt { .. } => {
                Cow::Owned(self.compare_token())
            }
            Self::ReadFlags { .. } | Self::MergeFlags { .. } | Self::InsertFlag { .. } => {
                Cow::Borrowed("readflags")
            }
            Self::FloatRaise { .. } => Cow::Borrowed("float.raise"),
            Self::CallClobber { .. } => Cow::Borrowed("nop"),
            Self::Select { .. } => Cow::Borrowed("select"),
            Self::Phi { .. } => Cow::Borrowed("phi"),
            Self::BoolAnd { .. } => Cow::Borrowed("booland"),
            Self::BoolOr { .. } => Cow::Borrowed("boolor"),
            Self::BoolXor { .. } => Cow::Borrowed("boolxor"),
            Self::BoolNot { .. } => Cow::Borrowed("boolnot"),
            Self::LoadIndirect { .. }
            | Self::ArgNext { .. }
            | Self::ArgStep { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::ArgList { .. } => Cow::Borrowed("load"),
            Self::StoreIndirect { .. } | Self::StoreStaticField { .. } => Cow::Borrowed("store"),
            Self::LoadField { .. } => Cow::Borrowed("load:member"),
            Self::StoreField { .. } => Cow::Borrowed("store:member"),
            Self::LoadElement { .. } => Cow::Borrowed("load:element"),
            Self::StoreElement { .. } => Cow::Borrowed("store:element"),
            // A call classifies to a bare token here; a consumer that can
            // resolve the callee (which needs target-specific metadata this crate
            // does not have) refines it to `call:function` / `call:rva` / etc.
            Self::Call(call) => Cow::Borrowed(match call.callee {
                Callee::Indirect { .. } => "call:indirect",
                Callee::Direct(_) | Callee::Virtual(_) | Callee::Constrained { .. } => "call",
            }),
            Self::IndirectBranch { .. } => Cow::Borrowed("branch:indirect"),
            Self::AtomicLoad { reserves, .. } => Cow::Borrowed(if *reserves {
                "atomic_load_reserve"
            } else {
                "atomic_load"
            }),
            Self::AtomicStore { .. } => Cow::Borrowed("atomic_store"),
            Self::AtomicStoreConditional { .. } => Cow::Borrowed("atomic_store_cond"),
            Self::AtomicPairLoad { .. } => Cow::Borrowed("atomic_pair_load"),
            Self::AtomicPairStoreConditional { .. } => Cow::Borrowed("atomic_pair_store_cond"),
            Self::AtomicCondRmw { op, .. } => Cow::Owned(format!("atomic_cond_{op}")),
            Self::AtomicExchange { .. } => Cow::Borrowed("atomic_xchg"),
            Self::AtomicCmpXchg { .. } => Cow::Borrowed("atomic_cmpxchg"),
            Self::AtomicPairCmpXchg { .. } => Cow::Borrowed("atomic_pair_cmpxchg"),
            Self::AtomicPairRmw { op, .. } => Cow::Owned(format!("atomic_pair_{op}")),
            Self::AtomicCheckedRmw { op, compare, .. } => match compare {
                Some(_) => Cow::Borrowed("atomic_checked_cmpxchg"),
                None => Cow::Owned(format!("atomic_checked_{op}")),
            },
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
            | Self::VectorSlice { .. }
            | Self::VectorInsertSlice { .. }
            | Self::VectorConcat { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorPack { .. }
            | Self::VectorPackLoad { .. }
            | Self::VectorPackStore { .. }
            | Self::Contract(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorSveCompute(_)
            | Self::VectorStructLoadReplicate(_) => Cow::Borrowed("vector"),
            Self::LoadObj { .. } => Cow::Borrowed("load"),
            Self::StoreObj { .. } => Cow::Borrowed("store"),
            Self::Nop => Cow::Borrowed("nop"),
            Self::Fence { .. } => Cow::Borrowed("fence"),
            // A `kind_str()` is a crate literal, and every registered table's
            // keys are pinned equal to their own `trim().to_ascii_lowercase()`,
            // so the token takes it as it is.
            Self::SystemOp(data) => Cow::Owned(format!("system:{}", data.kind.kind_str())),
            // A lane computation's policies are part of what it computes, so
            // its token is the whole computation.
            Self::Compute(data) => match data.kind.lanes {
                Some(_) => Cow::Owned(format!("lanes:{:?}", data.kind)),
                None => Cow::Owned(format!("compute:{}", data.kind.kernel.kind_str())),
            },
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
            | Self::MoveBlk { .. }
            | Self::CompareBlk(_)
            | Self::ScanBlk { .. }
            | Self::LoadBlk { .. }
            | Self::InitObj { .. }
            | Self::CopyObj { .. }
            | Self::MakeTypedRef { .. }
            | Self::TypedRefValue { .. }
            | Self::TypedRefType { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::FloatFunction(_)
            | Self::FpuControl(_)
            | Self::LocalAlloc { .. }
            | Self::Break(_)
            | Self::EndFinally
            | Self::InterruptReturn
            | Self::Rethrow
            | Self::Unreachable => Cow::Borrowed("unknown"),
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

    /// The call this operation is ([`CallData`]); `None` for every other
    /// operation.
    #[must_use]
    pub fn as_call(&self) -> Option<&CallData<T>> {
        match self {
            Self::Call(call) => Some(call),
            _ => None,
        }
    }

    /// The call this operation is when it names its method (a direct,
    /// virtual or constrained call, [`Callee::method`]); `None` for a call
    /// through a pointer and every other operation.
    #[must_use]
    pub fn as_named_call(&self) -> Option<&CallData<T>> {
        self.as_call().filter(|call| call.callee.method().is_some())
    }

    /// The call this operation is, to rewrite in place; `None` for every
    /// other operation.
    #[must_use]
    pub fn as_call_mut(&mut self) -> Option<&mut CallData<T>> {
        match self {
            Self::Call(call) => Some(call),
            _ => None,
        }
    }

    /// The call this operation is when it names its method, to rewrite in
    /// place ([`Self::as_named_call`]).
    #[must_use]
    pub fn as_named_call_mut(&mut self) -> Option<&mut CallData<T>> {
        self.as_call_mut()
            .filter(|call| call.callee.method().is_some())
    }

    /// Whether the operation's one result is a function of its operands
    /// alone, so that two such operations of one kind over the same values
    /// compute one value: it is pure, defines one value, and is none of the
    /// pure operations whose result depends on more than its operands
    /// (`Const` holds its own value, `CallClobber` defines a fresh unknown,
    /// `Phi` reads the edge control arrived by).
    #[must_use]
    pub fn computes_from_operands(&self) -> bool {
        !matches!(self, Self::Const { .. } | Self::Phi { .. })
            && !self.defines_undefined()
            && self.effects().is_pure()
            && self.defs().count() == 1
    }

    /// Whether every value the operation defines is undefined: fresh, with
    /// nothing in the program deciding it - the registers a call leaves
    /// holding whatever it left (`CallClobber`).
    ///
    /// A path that reads such a value relies on nothing it holds, so a phi
    /// operand that is one may be any value available on its edge. Every other
    /// read keeps it as it is: that a register is read after a call clobbered
    /// it is evidence of its own.
    #[must_use]
    pub const fn defines_undefined(&self) -> bool {
        matches!(self, Self::CallClobber { .. })
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
            Self::Ceq {
                dest, left, right, ..
            } => Some(BinaryOpInfo {
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
                ..
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
                ..
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
    /// (which is tracked separately during codegen). The environment's
    /// rounding mode an operation reads ([`Self::environment`]) is one more
    /// value consumed.
    #[must_use]
    pub fn stack_effect(&self) -> (u32, u32) {
        let (pops, pushes) = self.operand_effect();
        (
            pops.saturating_add(u32::from(self.environment().is_some())),
            pushes,
        )
    }

    /// The stack effect of the operation's own operands
    /// ([`Self::stack_effect`]), the environment's mode aside.
    fn operand_effect(&self) -> (u32, u32) {
        match self {
            Self::VectorSlice { .. } => (3, 1),
            Self::VectorInsertSlice { .. } => (3, 1),
            Self::VectorConcat { .. } => (2, 1),

            // Address computation pops the base and its optional index, pushes
            // the address. (A native-only op; never on the CIL eval stack.)
            Self::PtrAdd { index, .. } => (1u32.saturating_add(u32::from(index.is_some())), 1),
            // Call-clobber defines N caller-saved registers - pop 0, push N.
            // A raise reads the operands of the operation it is of.
            Self::FloatRaise { op, .. } => {
                let mut operation = op.as_ref();
                while let Self::FloatRaise { op, .. } = operation {
                    operation = op;
                }
                (operation.stack_effect().0, 1)
            }
            Self::CallClobber { outputs, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pushes = outputs.len() as u32;
                (0, pushes)
            }
            // Binary arithmetic, comparisons, and array access - pop 2, push 1
            Self::MergeFlags { .. }
            | Self::InsertFlag { .. }
            | Self::Add { .. }
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
            | Self::VectorPredicatedUnary {
                passthrough: None, ..
            }
            | Self::VectorPack {
                passthrough: None, ..
            }
            | Self::VectorPackLoad {
                passthrough: None, ..
            } => (2, 1),

            // Pop 3, push 1 (Select, AtomicCondRmw)
            Self::Select { .. }
            | Self::AtomicCondRmw { .. }
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
            Self::AtomicPairRmw { .. } => (3, 2),
            // The address, the operand and the mask, then the compared value
            // and the soft mask where the form has them; the old value and
            // the status.
            Self::AtomicCheckedRmw {
                compare, rcws_mask, ..
            } => (
                3u32.saturating_add(u32::from(compare.is_some()))
                    .saturating_add(u32::from(rcws_mask.is_some())),
                2,
            ),
            Self::WideMul { .. } => (2, 2),
            Self::WideDiv { .. } => (3, 2),

            Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicStoreConditional { .. } => (2, 1),
            Self::AtomicPairStoreConditional { .. } => (3, 1),

            // Control flow
            Self::Return { value, .. } => {
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
            | Self::Phi { .. }
            | Self::Fence { .. }
            | Self::InterruptReturn
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

            // Pop 3, push 0 (3, 0); a block that reads a direction flag pops it too
            Self::StoreElement { .. } | Self::MoveBlk { .. } => (3, 0),
            Self::InitBlk { backward, .. } | Self::CopyBlk { backward, .. } => {
                (3_u32.saturating_add(u32::from(backward.is_some())), 0)
            }
            Self::CompareBlk(data) => (4_u32.saturating_add(u32::from(data.backward.is_some())), 4),
            Self::ScanBlk { backward, .. } => {
                (4_u32.saturating_add(u32::from(backward.is_some())), 3)
            }
            Self::LoadBlk { backward, .. } => {
                (3_u32.saturating_add(u32::from(backward.is_some())), 2)
            }

            Self::ArgStep { .. } => (1, 2),

            // Pop 0, push 1 (0, 1)
            Self::LoadStaticField { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::SizeOf { .. }
            | Self::LoadToken { .. }
            | Self::ArgList { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::RuntimeValue { .. }
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
            | Self::ArgNext { .. }
            | Self::AtomicLoad { .. }
            | Self::LoadObj { .. }
            | Self::Box { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::CastClass { .. }
            | Self::IsInst { .. }
            | Self::MakeTypedRef { .. }
            | Self::TypedRefValue { .. }
            | Self::TypedRefType { .. }
            | Self::LoadVirtFunctionPtr { .. }
            | Self::LocalAlloc { .. }
            | Self::ReadFlags { .. }
            | Self::BoolNot { .. }
            | Self::VectorUnary { .. }
            | Self::VectorLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorExtract { .. }
            | Self::VectorSplat { .. }
            | Self::VectorShuffle {
                right: None,
                selector: VectorShuffleSelector::Mask(_),
                ..
            }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. } => (1, 1),

            Self::AtomicPairLoad { .. } => (1, 2),

            Self::VectorInsert { .. }
            | Self::VectorShuffle {
                right: Some(_),
                selector: VectorShuffleSelector::Mask(_),
                ..
            }
            | Self::VectorShuffle {
                right: None,
                selector: VectorShuffleSelector::Index(_),
                ..
            } => (2, 1),
            Self::VectorShuffle {
                right: Some(_),
                selector: VectorShuffleSelector::Index(_),
                ..
            } => (3, 1),
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
            // A call pops its arguments, and an indirect one its function
            // pointer too, and pushes its result. `args.len()` never exceeds
            // u32 for a CIL method.
            Self::Call(call) => {
                #[allow(clippy::cast_possible_truncation)]
                let pops = (call.args.len() as u32)
                    .saturating_add(u32::from(call.callee.fptr().is_some()));
                (pops, u32::from(call.dest.is_some()))
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
            Self::SystemOp(data) => data.operand_counts(),
            Self::Compute(data) => data.operand_counts(),
            Self::Contract(data) => data.operand_counts(),
            Self::VectorPredicateWhile(data) => data.operand_counts(),
            Self::VectorExtendInLane(data) => data.operand_counts(),
            Self::VectorElementCount(data) => data.operand_counts(),
            Self::VectorStructLoadReplicate(data) => data.operand_counts(),
            Self::VectorSveCompute(data) => data.operand_counts(),
            Self::FloatFunction(data) => data.operand_counts(),
            Self::FpuControl(data) => data.operand_counts(),
        }
    }

    /// Drops the signedness a vector operation's element states where its
    /// operation does not read it
    /// ([`VectorElement::read_by`](crate::ir::ops::VectorElement::read_by)), so two
    /// operations that compute the same value compare equal: the `Add` a
    /// signed halving add expands into is the `Add` of unsigned lanes.
    pub fn forget_unread_signedness(&mut self) {
        match self {
            Self::VectorBinary { kind, element, .. }
            | Self::VectorPredicatedBinary { kind, element, .. } => {
                *element = element.read_by(kind.reads_signedness());
            }
            // A grouped reduction extends each lane to its wider result lane,
            // so it reads the lanes' signedness whatever it computes.
            Self::VectorReduce {
                kind,
                element,
                groups,
                ..
            } => {
                *element = element.read_by(kind.reads_signedness() || groups.is_some());
            }
            Self::VectorUnary { kind, element, .. }
            | Self::VectorPredicatedUnary { kind, element, .. } => {
                *element = element.read_by(kind.reads_signedness());
            }
            _ => {}
        }
    }

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
            Self::VectorSlice { target_type, .. } | Self::VectorInsertSlice { target_type, .. } => {
                Some(target_type.clone())
            }
            Self::VectorConcat { target_type, .. } => Some(target_type.clone()),

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
            // Arithmetic/bitwise ops + SizeOf - per-host arithmetic type.
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
            // Context-dependent ops - resolved from SsaInstruction::result_type().
            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::Call(_)
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
            Self::Parity { .. } | Self::ReadFlags { .. } => T::comparison_result_type(),
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
            Self::ArgNext { ty, .. } | Self::ArgStep { ty, .. } => Some(ty.clone()),
            // LoadToken: assembly-free inference is `None`; the variable's
            // declared type set during SSA construction provides the real one.
            Self::LoadToken { .. } => None,
            Self::Unbox { value_type, .. } | Self::TypedRefValue { value_type, .. } => {
                T::byref_value_type_from_ref(value_type)
            }
            Self::LoadElementAddr { elem_type, .. } => T::byref_class_type_from_ref(elem_type),
            Self::PtrAdd { result_type, .. } => Some(result_type.clone()),
            Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. } => None,
            _ => None,
        }
    }

    /// Borrows the type this operation states explicitly: a conversion's
    /// target, a vector operation's vector type, an access's value type, the
    /// one a static execution kernel reads.
    ///
    /// Operations without such a field keep ordinary target querying.
    #[must_use]
    pub fn stated_type(&self) -> Option<&T::Type> {
        match self {
            Self::IntConv { target, .. }
            | Self::FloatToInt { target, .. }
            | Self::IntToFloat { target, .. }
            | Self::FloatConv { target, .. }
            | Self::Bitcast { target, .. }
            | Self::IntToPtr { target, .. }
            | Self::PtrToInt { target, .. } => Some(target),
            Self::VectorSplat { vector_type, .. }
            | Self::VectorStore { vector_type, .. }
            | Self::VectorMaskedStore { vector_type, .. }
            | Self::VectorScatter { vector_type, .. }
            | Self::VectorSegmentStore { vector_type, .. }
            | Self::VectorPackStore { vector_type, .. } => Some(vector_type),
            Self::VectorSlice { target_type, .. }
            | Self::VectorInsertSlice { target_type, .. }
            | Self::VectorConcat { target_type, .. }
            | Self::VectorReinterpret { target_type, .. }
            | Self::VectorCast { target_type, .. } => Some(target_type),
            Self::LoadIndirect { value_type, .. }
            | Self::StoreIndirect { value_type, .. }
            | Self::VectorLoad {
                vector_type: value_type,
                ..
            } => Some(value_type),
            _ => None,
        }
    }

    /// Whether `op` is one whose results [`Machine::evaluate_results`](crate::analysis::consts::Machine::evaluate_results) folds
    /// one by one: the double-width arithmetic, the float functions of
    /// several results that are exact (`fxtract`, `fprem`, `fprem1`), and
    /// every computation of more than one result (a computation over lanes
    /// writes its status beside its data). An analysis evaluates such an
    /// operation through it, and every other through [`Machine::evaluate`](crate::analysis::consts::Machine::evaluate).
    #[must_use]
    pub fn folds_results(&self) -> bool {
        match self {
            Self::WideMul { .. } | Self::WideDiv { .. } => true,
            Self::FloatFunction(data) => matches!(
                data.kind,
                FloatFunctionKind::Extract | FloatFunctionKind::Rem | FloatFunctionKind::Rem1
            ),
            Self::Compute(data) => data.outputs.len() > 1,
            _ => false,
        }
    }
}
