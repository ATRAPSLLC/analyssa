//! The [`SsaOp`] enum itself — every operation variant and its operands.
//!
//! Behaviour lives in the sibling modules: operand access in [`super::visit`],
//! effect classification in [`super::effects`], taxonomy and tokens in
//! [`super::classify`], control-flow queries in [`super::control`], and
//! rendering in [`super::display`].

use crate::{
    ir::{
        ops::{
            kinds::{
                AtomicAccessWidth, AtomicOrdering, AtomicRmwOp, BcdAdjustData, BreakpointOp,
                CmpKind, ComputeKind, FenceKind, FpuControlKind, KindedVecData, NativeKindedData,
                NativeOpaqueData, SystemOpKind, TranscendentalKind, VecImm8Data,
            },
            native::{BlockStringOpData, FlagCondition, FlagsMask, WideCmpXchgData},
            vector::{
                ComplexMulKind, FlagAdjustKind, FpHelperKind, PredicateGenKind, SvePermuteKind,
                TileOpKind, VectorBinaryKind, VectorBitfieldData, VectorBitmaskKind,
                VectorCastKind, VectorCompareKind, VectorComplexAddData, VectorConditionalMoveData,
                VectorCountAdjustData, VectorCryptoKind, VectorDotProductData, VectorElement,
                VectorElementCountData, VectorExtendInLaneData, VectorFaultMode,
                VectorHorizontalMinPosData, VectorHorizontalReduceData, VectorIntDotProductData,
                VectorIntersectData, VectorMaddKind, VectorMaskBinaryKind, VectorMaskMode,
                VectorMaskUnaryKind, VectorMatrixMulAccData, VectorNarrowSaturateData,
                VectorPackKind, VectorPackNarrowData, VectorPermuteData, VectorPredicateBreakData,
                VectorPredicateOpData, VectorPredicateWhileData, VectorReduceKind,
                VectorReverseChunksData, VectorSegmentLayout, VectorShuffleBitsData,
                VectorSmeMiscData, VectorSmeOuterProductData, VectorStringCompareData,
                VectorStructLoadReplicateData, VectorSveAddressGenData, VectorSveComputeData,
                VectorTernaryKind, VectorUnaryKind,
            },
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{Target, VectorShuffleMask},
};

/// A decomposed SSA operation.
///
/// Each variant represents a single operation with explicit inputs and outputs.
/// This enables clean pattern matching for optimization and analysis passes.
///
/// # Conventions
///
/// - For operations that produce a result, the first `SsaVarId` is the destination
/// - Operands follow in the order they appear on the CIL stack (first pushed = first operand)
/// - Optional results use `Option<SsaVarId>` (e.g., calls that may not return a value)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T::Type: serde::Serialize, T::TypeRef: serde::Serialize, \
                     T::MethodRef: serde::Serialize, T::FieldRef: serde::Serialize, T::SymbolRef: serde::Serialize, \
                     T::SigRef: serde::Serialize",
        deserialize = "T::Type: serde::Deserialize<'de>, T::TypeRef: serde::Deserialize<'de>, \
                       T::MethodRef: serde::Deserialize<'de>, T::FieldRef: serde::Deserialize<'de>, T::SymbolRef: serde::Deserialize<'de>, \
                       T::SigRef: serde::Deserialize<'de>"
    ))
)]
pub enum SsaOp<T: Target> {
    /// Load a constant value.
    ///
    /// `dest = const value`
    Const {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Constant value assigned to the destination.
        value: ConstValue<T>,
    },

    /// Addition: `dest = left + right`
    Add {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Addition with overflow check: `dest = left + right` (throws on overflow)
    AddOvf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Subtraction: `dest = left - right`
    Sub {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Subtraction with overflow check: `dest = left - right`
    SubOvf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Multiplication: `dest = left * right`
    Mul {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Multiplication with overflow check: `dest = left * right`
    MulOvf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Native wide multiply producing low and high halves.
    WideMul {
        /// Low-half output or dividend variable.
        low: SsaVarId,
        /// High-half output or dividend variable.
        high: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Division: `dest = left / right`
    Div {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Remainder: `dest = left % right`
    Rem {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Floating-point comparison producing a native flags value.
    ///
    /// Models ordered/unordered target flag semantics such as AArch64 `FCMP`
    /// producing `NZCV`. The `signaling` form records exception-sensitive
    /// comparisons such as AArch64 `FCMPE`.
    FloatCompareFlags {
        /// Native flags output variable.
        flags: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether signaling NaNs may raise a floating-point exception.
        signaling: bool,
    },

    /// Native wide divide consuming high:low dividend halves.
    WideDiv {
        /// Quotient output variable.
        quotient: SsaVarId,
        /// Remainder output variable.
        remainder: SsaVarId,
        /// High-half output or dividend variable.
        high: SsaVarId,
        /// Low-half output or dividend variable.
        low: SsaVarId,
        /// Divisor operand variable.
        divisor: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Negation: `dest = -operand`
    Neg {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise AND: `dest = left & right`
    And {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise OR: `dest = left | right`
    Or {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise XOR: `dest = left ^ right`
    Xor {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Bitwise NOT: `dest = ~operand`
    Not {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Shift left: `dest = value << amount`
    Shl {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Shift right: `dest = value >> amount`
    Shr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Optional flags output variable.
        flags: Option<SsaVarId>,
    },

    /// Rotate left: `dest = value <<< amount`
    Rol {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Rotate right: `dest = value >>> amount`
    Ror {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Rotate through carry left
    Rcl {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Rotate through carry right
    Rcr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Byte swap (endian conversion): `dest = bswap(src)`
    BSwap {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Bit reverse: `dest = brev(src)`
    BRev {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Bit scan forward (find first set bit, LSB-based): `dest = bsf(src)`
    BitScanForward {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Bit scan reverse (find first set bit, MSB-based): `dest = bsr(src)`
    BitScanReverse {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Population count: `dest = popcnt(src)`
    Popcount {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Parity: `dest = parity(src)` — 1 if odd number of set bits, 0 if even
    Parity {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Compare equal: `dest = (left == right) ? 1 : 0`
    Ceq {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Compare less than: `dest = (left < right) ? 1 : 0`
    Clt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Compare greater than: `dest = (left > right) ? 1 : 0`
    Cgt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Boolean conjunction: `dest = left && right`.
    BoolAnd {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean disjunction: `dest = left || right`.
    BoolOr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean exclusive-or: `dest = left != right`.
    BoolXor {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean negation: `dest = !value`.
    BoolNot {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Integer→integer conversion: `dest = (target_int)operand`.
    ///
    /// Covers widening (`movzx`/`movsx`, CIL `conv.u*`/`conv.i*` from a narrower
    /// source), narrowing (sub-register extraction, `conv.*` from a wider
    /// source), and equal-width reinterpretation — the physical widen/narrow is a
    /// consequence of the source and `target` widths, not a distinct operation,
    /// so it is derived on demand rather than committed at lift time (the lifter
    /// does not always know the source width). `unsigned` selects zero- vs
    /// sign-extension semantics; `overflow_check` is CIL `conv.ovf.*` (may throw).
    IntConv {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target integer type metadata.
        target: T::Type,
        /// Whether the conversion checks overflow (CIL `conv.ovf.*`).
        overflow_check: bool,
        /// Whether the source is interpreted unsigned (zero- vs sign-extension).
        unsigned: bool,
    },

    /// Integer→pointer conversion: `dest = (target_ptr)operand`. The value crosses
    /// from the integer domain into the pointer domain (an address computation
    /// result, or a `conv` to a pointer-typed slot).
    IntToPtr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target pointer type metadata.
        target: T::Type,
    },

    /// Pointer→integer conversion: `dest = (target_int)operand`. The value crosses
    /// from the pointer domain into the integer domain.
    PtrToInt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target integer type metadata.
        target: T::Type,
    },

    /// Integer→floating-point conversion: `dest = (target_float)operand`
    /// (`sitofp`/`uitofp`, x87 integer load, CIL `conv.r*`). `unsigned` selects
    /// the source interpretation.
    IntToFloat {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target floating-point type metadata.
        target: T::Type,
        /// Whether the integer source is interpreted unsigned.
        unsigned: bool,
    },

    /// Floating-point→integer conversion: `dest = (target_int)operand`
    /// (`fptosi`/`fptoui`, x87 store-as-integer, CIL `conv.ovf.*` from a float).
    /// `unsigned` selects the destination interpretation; `overflow_check` is the
    /// CIL checked form (may throw).
    FloatToInt {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target integer type metadata.
        target: T::Type,
        /// Whether the conversion checks overflow (CIL `conv.ovf.*`).
        overflow_check: bool,
        /// Whether the integer destination is interpreted unsigned.
        unsigned: bool,
    },

    /// Floating-point→floating-point width change: `dest = (target_float)operand`
    /// (`fpext`/`fptrunc`, e.g. `cvtss2sd`).
    FloatConv {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target floating-point type metadata.
        target: T::Type,
    },

    /// Representation-preserving scalar bitcast: `dest = bitcast(target)operand`.
    Bitcast {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
        /// Target type metadata.
        target: T::Type,
    },

    /// Scaled pointer address computation: `dest = base + index*stride + offset`.
    ///
    /// Models a native memory-addressing form (`[base + index*scale + disp]`) as
    /// one structural op instead of a shredded `Shl`/`Mul`/`Add` chain, so a
    /// field access reads as a field access and array indexing as indexing.
    /// `index` is absent for a plain `base + offset`; `stride` is the index
    /// scale in bytes and `offset` the signed byte displacement. `result_type`
    /// carries the pointer type the address evaluates to.
    PtrAdd {
        /// Destination SSA variable (a pointer).
        dest: SsaVarId,
        /// Base address operand.
        base: SsaVarId,
        /// Scaled index operand, when the address adds one.
        index: Option<SsaVarId>,
        /// Scale applied to `index`, in bytes.
        stride: u64,
        /// Constant displacement from the base, in bytes.
        offset: i64,
        /// Pointer type metadata the address evaluates to.
        result_type: T::Type,
    },

    /// Conditional select: `dest = condition ? true_val : false_val`
    Select {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Condition operand variable.
        condition: SsaVarId,
        /// Value selected when the condition is true.
        true_val: SsaVarId,
        /// Value selected when the condition is false.
        false_val: SsaVarId,
    },

    /// Read condition code flags from a flags variable.
    ReadFlags {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Flags operand variable.
        flags: SsaVarId,
        /// Flag mask to read.
        mask: FlagsMask,
    },

    /// Lane-wise unary vector operation.
    VectorUnary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Vector operation kind.
        kind: VectorUnaryKind,
        /// Lane element descriptor (class / width / scalar).
        element: VectorElement,
    },

    /// Lane-wise binary vector operation.
    VectorBinary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Vector operation kind.
        kind: VectorBinaryKind,
        /// Lane element descriptor (class / width / scalar).
        element: VectorElement,
    },

    /// Lane-wise ternary vector operation.
    VectorTernary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// First vector operand variable.
        first: SsaVarId,
        /// Second vector operand variable.
        second: SsaVarId,
        /// Third vector operand variable.
        third: SsaVarId,
        /// Vector operation kind.
        kind: VectorTernaryKind,
    },

    /// Predicated lane-wise unary vector operation.
    VectorPredicatedUnary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorUnaryKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Predicated lane-wise binary vector operation.
    VectorPredicatedBinary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorBinaryKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Predicated lane-wise ternary vector operation.
    VectorPredicatedTernary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// First vector operand variable.
        first: SsaVarId,
        /// Second vector operand variable.
        second: SsaVarId,
        /// Third vector operand variable.
        third: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorTernaryKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Lane-wise vector comparison producing a vector mask.
    VectorCompare {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Vector comparison kind.
        kind: VectorCompareKind,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
    },

    /// Vector load from memory.
    VectorLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Vector store to memory.
    VectorStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Predicated vector load from memory.
    VectorMaskedLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Predicated vector store to memory.
    VectorMaskedStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Loads one scalar value and broadcasts it to all vector lanes.
    VectorBroadcastLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Gathers vector lanes from memory using vector indices.
    VectorGather {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Base address operand variable.
        base: SsaVarId,
        /// Vector index operand variable.
        indices: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Vector load with first-fault or fault-only-first behavior.
    VectorFaultingLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Optional fault/status output variable.
        fault: Option<SsaVarId>,
        /// Address operand variable.
        addr: SsaVarId,
        /// Optional mask operand variable.
        mask: Option<SsaVarId>,
        /// Optional passthrough vector for inactive or fault-suppressed lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Faulting load behavior.
        fault_mode: VectorFaultMode,
        /// Inactive lane behavior.
        mask_mode: VectorMaskMode,
    },

    /// Loads multiple vector segments from memory.
    VectorSegmentLoad {
        /// Destination vector variables, one per segment.
        dests: Vec<SsaVarId>,
        /// Base address operand variable.
        base: SsaVarId,
        /// Optional mask operand variable.
        mask: Option<SsaVarId>,
        /// Vector type metadata for each segment.
        vector_type: T::Type,
        /// Number of segments loaded.
        segments: u32,
        /// Segment memory layout.
        layout: VectorSegmentLayout,
    },

    /// Scatters vector lanes to memory using vector indices.
    VectorScatter {
        /// Base address operand variable.
        base: SsaVarId,
        /// Vector index operand variable.
        indices: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Stores multiple vector segments to memory.
    VectorSegmentStore {
        /// Base address operand variable.
        base: SsaVarId,
        /// Vector values to store, one per segment.
        values: Vec<SsaVarId>,
        /// Optional mask operand variable.
        mask: Option<SsaVarId>,
        /// Vector type metadata for each segment.
        vector_type: T::Type,
        /// Number of segments stored.
        segments: u32,
        /// Segment memory layout.
        layout: VectorSegmentLayout,
    },

    /// Extracts one scalar lane from a vector.
    VectorExtract {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector operand variable.
        vector: SsaVarId,
        /// Zero-based lane index.
        lane: u32,
    },

    /// Inserts one scalar lane into a vector.
    VectorInsert {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector operand variable.
        vector: SsaVarId,
        /// Zero-based lane index.
        lane: u32,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Builds a vector by splatting one scalar value to every lane.
    VectorSplat {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
    },

    /// Shuffles lanes from one or two vector inputs.
    VectorShuffle {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Optional second shuffle input vector.
        right: Option<SsaVarId>,
        /// Vector shuffle lane selector.
        mask: VectorShuffleMask,
    },

    /// Converts vector lane values to another vector type.
    VectorCast {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Target vector type metadata.
        target_type: T::Type,
        /// Vector cast kind.
        kind: VectorCastKind,
    },

    /// Reinterprets vector bits as another vector type of the same total width.
    VectorReinterpret {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Target vector type metadata.
        target_type: T::Type,
    },

    /// Packs or expands vector lanes under a predicate mask.
    VectorPack {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive destination lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Lane element width in bits.
        element_bits: u32,
        /// Packing direction.
        kind: VectorPackKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Loads compact vector lanes and expands them under a predicate mask.
    VectorPackLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Base address operand variable.
        addr: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Optional passthrough vector for inactive destination lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Lane element width in bits.
        element_bits: u32,
        /// Packing direction. Must be [`VectorPackKind::Expand`].
        kind: VectorPackKind,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
    },

    /// Compresses active vector lanes and stores them contiguously to memory.
    VectorPackStore {
        /// Base address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Lane element width in bits.
        element_bits: u32,
        /// Packing direction. Must be [`VectorPackKind::Compress`].
        kind: VectorPackKind,
    },

    /// Clears vector upper lanes according to target vector aliasing rules.
    VectorZeroUpper {
        /// `true` to clear all vector state; `false` to clear upper lanes.
        all: bool,
    },

    /// Applies a unary operation to vector mask lanes.
    VectorMaskUnary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Mask operand variable.
        mask: SsaVarId,
        /// Vector mask operation kind.
        kind: VectorMaskUnaryKind,
    },

    /// Applies a binary operation to vector mask lanes.
    VectorMaskBinary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left mask operand variable.
        left: SsaVarId,
        /// Right mask operand variable.
        right: SsaVarId,
        /// Vector mask operation kind.
        kind: VectorMaskBinaryKind,
    },

    /// Reduces vector lanes to one scalar result.
    VectorReduce {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector or mask operand variable.
        value: SsaVarId,
        /// Reduction operation kind.
        kind: VectorReduceKind,
    },

    /// Extracts lane predicate bits into a scalar integer mask.
    VectorBitmask {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector or mask operand variable.
        value: SsaVarId,
        /// Bitmask extraction kind.
        kind: VectorBitmaskKind,
    },

    /// Unconditional jump to a block.
    Jump {
        /// Target block index.
        target: usize,
    },

    /// Conditional branch: if condition is true, go to true_target, else false_target.
    Branch {
        /// Condition operand variable.
        condition: SsaVarId,
        /// Target block for the true branch.
        true_target: usize,
        /// Target block for the false branch.
        false_target: usize,
    },

    /// Compare and branch: if (left cmp right) goto true_target else false_target.
    ///
    /// This represents CIL comparison branch instructions like `beq`, `blt`, `bgt`, etc.
    /// These are combined compare-and-branch operations that don't produce an intermediate
    /// comparison result.
    BranchCmp {
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Comparison predicate.
        cmp: CmpKind,
        /// Whether operands use unsigned interpretation.
        unsigned: bool,
        /// Target block for the true branch.
        true_target: usize,
        /// Target block for the false branch.
        false_target: usize,
    },

    /// Branch based on condition code flags.
    BranchFlags {
        /// Flags operand variable.
        flags: SsaVarId,
        /// Flag condition predicate.
        condition: FlagCondition,
        /// Target block for the true branch.
        true_target: usize,
        /// Target block for the false branch.
        false_target: usize,
    },

    /// Switch statement: jump to `targets[value]` or default if out of range.
    Switch {
        /// Value operand variable.
        value: SsaVarId,
        /// Switch target block indices.
        targets: Vec<usize>,
        /// Default switch target block.
        default: usize,
    },

    /// Indirect branch through a computed target expression.
    IndirectBranch {
        /// SSA value containing the computed target address.
        target: SsaVarId,
        /// Statically recovered successor block indices, if known.
        resolved_targets: Vec<usize>,
    },

    /// Return from method with optional value.
    Return {
        /// Optional return value variable.
        value: Option<SsaVarId>,
    },

    /// Load instance field: `dest = object.field`
    LoadField {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Store instance field: `object.field = value`
    StoreField {
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Load static field: `dest = ClassName.field`
    LoadStaticField {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Store static field: `ClassName.field = value`
    StoreStaticField {
        /// Field reference metadata.
        field: T::FieldRef,
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Load field address: `dest = &object.field`
    LoadFieldAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Load static field address: `dest = &ClassName.field`
    LoadStaticFieldAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
    },

    /// Load array element: `dest = array[index]`
    LoadElement {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Array operand variable.
        array: SsaVarId,
        /// Index operand variable.
        index: SsaVarId,
        /// Element type metadata.
        elem_type: T::Type,
    },

    /// Store array element: `array[index] = value`
    StoreElement {
        /// Array operand variable.
        array: SsaVarId,
        /// Index operand variable.
        index: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Element type metadata.
        elem_type: T::Type,
    },

    /// Load array element address: `dest = &array[index]`
    LoadElementAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Array operand variable.
        array: SsaVarId,
        /// Index operand variable.
        index: SsaVarId,
        /// Element type metadata.
        elem_type: T::TypeRef,
    },

    /// Get array length: `dest = array.Length`
    ArrayLength {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Array operand variable.
        array: SsaVarId,
    },

    /// Load through pointer: `dest = *ptr`
    LoadIndirect {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value type metadata.
        value_type: T::Type,
        /// Address space the access is qualified by, or `None` for the
        /// target's default (flat) space.
        ///
        /// Opaque and host-defined: on x86 a frontend maps a segment override
        /// (`fs:`/`gs:`) to a distinct id here. It qualifies the *access*, not
        /// the address arithmetic — `lea rax, fs:[0x30]` computes `0x30`,
        /// because segmentation is applied on dereference — which is why this
        /// rides the load/store rather than [`SsaOp::PtrAdd`].
        ///
        /// Alias analysis treats different address spaces as disjoint; without
        /// it, `fs:[0x30]` and a flat `[0x30]` decode to one memory location.
        address_space: Option<u16>,
    },

    /// Store through pointer: `*ptr = value`
    StoreIndirect {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::Type,
        /// Address space the access is qualified by, or `None` for the
        /// target's default (flat) space.
        ///
        /// Opaque and host-defined: on x86 a frontend maps a segment override
        /// (`fs:`/`gs:`) to a distinct id here. It qualifies the *access*, not
        /// the address arithmetic — `lea rax, fs:[0x30]` computes `0x30`,
        /// because segmentation is applied on dereference — which is why this
        /// rides the load/store rather than [`SsaOp::PtrAdd`].
        ///
        /// Alias analysis treats different address spaces as disjoint; without
        /// it, `fs:[0x30]` and a flat `[0x30]` decode to one memory location.
        address_space: Option<u16>,
    },

    /// Create new object: `dest = new Type(args...)`
    NewObj {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Constructor method reference.
        ctor: T::MethodRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Create new array: `dest = new Type[length]`
    NewArr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Element type metadata.
        elem_type: T::TypeRef,
        /// Array length operand variable.
        length: SsaVarId,
    },

    /// Cast object to type (throws if invalid): `dest = (Type)obj`
    CastClass {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Target type metadata.
        target_type: T::TypeRef,
    },

    /// Type check (returns null if invalid): `dest = obj as Type`
    IsInst {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Target type metadata.
        target_type: T::TypeRef,
    },

    /// Box value type: `dest = (object)value`
    Box {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Unbox to pointer: `dest = &((ValueType)obj)`
    Unbox {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Unbox and copy: `dest = (ValueType)obj`
    UnboxAny {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Get size of value type: `dest = sizeof(Type)`
    SizeOf {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Load runtime type token: `dest = typeof(Type).TypeHandle`
    LoadToken {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Runtime token metadata.
        token: T::TypeRef,
    },

    /// Direct method call: `dest = method(args...)`
    Call {
        /// Optional destination SSA variable.
        dest: Option<SsaVarId>,
        /// Method reference metadata.
        method: T::MethodRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Virtual method call: `dest = obj.method(args...)`
    CallVirt {
        /// Optional destination SSA variable.
        dest: Option<SsaVarId>,
        /// Method reference metadata.
        method: T::MethodRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Indirect call through function pointer: `dest = fptr(args...)`
    CallIndirect {
        /// Optional destination SSA variable.
        dest: Option<SsaVarId>,
        /// Function pointer operand variable.
        fptr: SsaVarId,
        /// Indirect call signature metadata.
        signature: T::SigRef,
        /// Call argument variables.
        args: Vec<SsaVarId>,
    },

    /// Load function pointer: `dest = &method`
    LoadFunctionPtr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Method reference metadata.
        method: T::MethodRef,
    },

    /// Load virtual function pointer: `dest = &obj.method`
    LoadVirtFunctionPtr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Method reference metadata.
        method: T::MethodRef,
    },

    /// Load argument value: `dest = argN`
    LoadArg {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Argument index.
        arg_index: u16,
    },

    /// Load local value: `dest = localN`
    LoadLocal {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Local variable index.
        local_index: u16,
    },

    /// Load argument address: `dest = &argN`
    LoadArgAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Argument index.
        arg_index: u16,
    },

    /// Load local address: `dest = &localN`
    LoadLocalAddr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Local variable index.
        local_index: u16,
    },

    /// Copy value (from dup): `dest = src`
    Copy {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source operand variable.
        src: SsaVarId,
    },

    /// Pop value from stack (value is discarded, but we track the use)
    Pop {
        /// Value operand variable.
        value: SsaVarId,
    },

    /// Throw exception: `throw obj`
    Throw {
        /// Exception object variable.
        exception: SsaVarId,
    },

    /// Rethrow current exception (in catch handler)
    Rethrow,

    /// End finally block
    EndFinally,

    /// End filter block with result
    EndFilter {
        /// Filter result variable.
        result: SsaVarId,
    },

    /// Return from interrupt / exception handler
    InterruptReturn,

    /// Unreachable terminator.
    Unreachable,

    /// Leave protected region
    Leave {
        /// Target block index.
        target: usize,
    },

    /// Initialize block of memory to zero
    InitBlk {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
        /// `true` when the fill proceeds from high to low addresses (x86 `rep
        /// stos` with the direction flag set); `false` is the forward default.
        reverse: bool,
    },

    /// Copy block of memory
    CopyBlk {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
        /// `true` when the copy proceeds from high to low addresses (x86 `rep
        /// movs` with the direction flag set), which matters for overlapping
        /// regions; `false` is the forward default.
        reverse: bool,
    },

    /// Memory fence / barrier
    Fence {
        /// Fence ordering kind.
        kind: FenceKind,
    },

    /// Target-specific native instruction with explicit operands, outputs, and
    /// effects.
    ///
    /// Opaque native instruction with explicit inputs/outputs and a
    /// conservative effect summary. The payload is boxed (see
    /// [`NativeOpaqueData`]) to keep `SsaOp` small.
    NativeOpaque(Box<NativeOpaqueData>),

    /// Typed native **system / privileged** operation (`cpuid`, time-stamp reads,
    /// system/control-register access, syscalls, traps, cache/TLB maintenance,
    /// privileged state ops). The first-class typed replacement for the system
    /// cases of `NativeOpaque`: a structured
    /// [`SystemOpKind`] identity drives a precise effect summary and a distinct
    /// similarity class. The payload is boxed (see [`NativeKindedData`]).
    SystemOp(Box<NativeKindedData<SystemOpKind>>),

    /// Typed native **compute** intrinsic (`pdep`/`pext`, `crc32`,
    /// `rdrand`/`rdseed`, pointer authentication). The first-class typed
    /// replacement for the hardware compute intrinsics: a structured
    /// [`ComputeKind`] identity drives a precise effect summary (pure, except
    /// nondeterministic random sources) and a distinct similarity class. The
    /// payload is boxed (see [`NativeKindedData`]).
    ComputeOp(Box<NativeKindedData<ComputeKind>>),

    /// Typed legacy x86 **binary-coded-decimal adjust**
    /// (`daa`/`das`/`aaa`/`aas`/`aam`/`aad`). The [`BcdAdjustKind`] identity
    /// drives a pure effect summary and an arithmetic similarity class; the
    /// accumulator and flags flow through as typed SSA values. The payload is
    /// boxed (see [`BcdAdjustData`]).
    ///
    /// [`BcdAdjustKind`]: crate::ir::ops::kinds::BcdAdjustKind
    BcdAdjust(Box<BcdAdjustData>),

    /// Typed hardware **vector cryptographic** operation (AES / SHA / SM3 /
    /// SM4 / GF(2^8) / carry-less multiply). The first-class, named replacement
    /// for what would otherwise be an opaque crypto blob: a [`VectorCryptoKind`]
    /// identity drives a precise (pure) effect summary and the `Vector`
    /// similarity class. The payload is boxed (see [`KindedVecData`]).
    VectorCrypto(Box<KindedVecData<VectorCryptoKind>>),

    /// Typed AMX **tile** operation (matrix multiply-accumulate, tile load /
    /// store / zero / release, configuration load / store). A [`TileOpKind`]
    /// identity drives a precise effect summary (pure / read / write). The
    /// payload is boxed (see [`KindedVecData`]).
    TileOp(Box<KindedVecData<TileOpKind>>),

    /// Typed vector **lane permute by a runtime index vector** (x86
    /// `vpermd`/`vpermps`/`vpermb`/`vpshufb`/`vpermt2*`/`vpermi2*` and the
    /// variable `vpermilps`/`vpermilpd`). Distinct from [`Self::VectorShuffle`]
    /// (static mask). Pure. The payload is boxed (see [`VectorPermuteData`]).
    VectorPermute(Box<VectorPermuteData>),

    /// Typed vector **fused multiply-then-horizontal-add** (x86 `pmaddwd`/
    /// `pmaddubsw`/`vpdpbusd[s]`/`vpdpwssd[s]`/`vpmadd52luq`/`vpmadd52huq`).
    /// Multiplies two source vectors and horizontally sums adjacent products,
    /// optionally accumulating into a running destination. Pure. The payload is
    /// boxed (see [`KindedVecData`]).
    VectorMultiplyAdd(Box<KindedVecData<VectorMaddKind>>),

    /// Typed vector **two-source saturating narrowing pack** (x86 `packsswb`/
    /// `packssdw`/`packuswb`/`packusdw`). Pure. The payload is boxed (see
    /// [`VectorPackNarrowData`]).
    VectorPackNarrow(Box<VectorPackNarrowData>),
    /// Single-source saturating narrowing (see [`VectorNarrowSaturateData`]).
    VectorNarrowSaturate(Box<VectorNarrowSaturateData>),
    /// Predicate-generation from scalar loop bounds (see [`VectorPredicateWhileData`]).
    VectorPredicateWhile(Box<VectorPredicateWhileData>),
    /// Predicate-break generation (see [`VectorPredicateBreakData`]).
    VectorPredicateBreak(Box<VectorPredicateBreakData>),
    /// Complex-number add with rotation (see [`VectorComplexAddData`]).
    VectorComplexAdd(Box<VectorComplexAddData>),
    /// Adjust a register by an implicit element/predicate count (see [`VectorCountAdjustData`]).
    VectorCountAdjust(Box<VectorCountAdjustData>),
    /// Sign/zero-extend the low field of each lane in place (see [`VectorExtendInLaneData`]).
    VectorExtendInLane(Box<VectorExtendInLaneData>),
    /// Read the symbolic element count `VL / element_bits` into a scalar (see
    /// [`VectorElementCountData`]).
    VectorElementCount(Box<VectorElementCountData>),
    /// SVE vector address generation `base + (extend(index) << shift)` (see
    /// [`VectorSveAddressGenData`]).
    VectorSveAddressGen(Box<VectorSveAddressGenData>),
    /// Manipulate the condition flags directly (see [`KindedVecData`]).
    FlagAdjust(Box<KindedVecData<FlagAdjustKind>>),
    /// Load-and-replicate a 2/3/4-element structure (see [`VectorStructLoadReplicateData`]).
    VectorStructLoadReplicate(Box<VectorStructLoadReplicateData>),
    /// SME ZA-tile accumulate/zero (see [`VectorSmeMiscData`]).
    VectorSmeMisc(Box<VectorSmeMiscData>),
    /// SVE predicate/first-fault-register operation (see [`VectorPredicateOpData`]).
    VectorPredicateOp(Box<VectorPredicateOpData>),
    /// SVE2/NEON compute op named by its kind (see [`VectorSveComputeData`]).
    VectorSveCompute(Box<VectorSveComputeData>),
    /// Reverse chunks within each element (see [`VectorReverseChunksData`]).
    VectorReverseChunks(Box<VectorReverseChunksData>),
    /// Matrix multiply-accumulate over vectors (see [`VectorMatrixMulAccData`]).
    VectorMatrixMulAcc(Box<VectorMatrixMulAccData>),
    /// SME outer-product accumulate into a ZA tile (see [`VectorSmeOuterProductData`]).
    VectorSmeOuterProduct(Box<VectorSmeOuterProductData>),
    /// Predicate generation (const/iterate/ffr/unpack/select; see [`KindedVecData`]).
    VectorPredicateGen(Box<KindedVecData<PredicateGenKind>>),
    /// SVE floating-point transcendental helper (see [`KindedVecData`]).
    VectorFpHelper(Box<KindedVecData<FpHelperKind>>),
    /// SVE data-movement permute/extract (see [`KindedVecData`]).
    VectorSvePermute(Box<KindedVecData<SvePermuteKind>>),

    /// Typed vector **arbitrary three-input bitwise logic** (x86 `vpternlogd`/
    /// `vpternlogq`), selected by an 8-bit truth table. Pure. The payload is
    /// boxed (see [`VecImm8Data`]).
    VectorTernaryLogic(Box<VecImm8Data>),

    /// Typed vector **floating-point dot product** (x86 SSE4.1 `dpps`/`dppd`,
    /// VEX `vdpps`/`vdppd`), selected by an 8-bit lane-participation / result-
    /// broadcast immediate. Pure. The payload is boxed (see
    /// [`VectorDotProductData`]).
    VectorDotProduct(Box<VectorDotProductData>),

    /// Typed vector **multi-block sum of absolute differences** (x86 SSE4.1
    /// `mpsadbw`, VEX `vmpsadbw`, AVX-512 `vdbpsadbw`), selected by an 8-bit
    /// block-offset immediate. Pure. The payload is boxed (see
    /// [`VecImm8Data`]).
    VectorMultiSad(Box<VecImm8Data>),

    /// Typed vector **integer dot-product-accumulate** (ARM/AArch64 `sdot`/
    /// `udot`/`usdot`): each lane accumulates the sum of a group of widened
    /// integer element products. Pure. The payload is boxed (see
    /// [`VectorIntDotProductData`]).
    VectorIntDotProduct(Box<VectorIntDotProductData>),

    /// Typed vector **packed string comparison** (SSE4.2 `pcmpestri`/
    /// `pcmpestrm`/`pcmpistri`/`pcmpistrm` and VEX peers), selected by an 8-bit
    /// format / aggregation / polarity immediate. Pure. The payload is boxed
    /// (see [`VectorStringCompareData`]).
    VectorStringCompare(Box<VectorStringCompareData>),

    /// Typed vector **bit-field extract / insert** over the low 64 bits of a
    /// vector register (SSE4a `extrq`/`insertq`). Pure. The payload is boxed
    /// (see [`VectorBitfieldData`]).
    VectorBitfield(Box<VectorBitfieldData>),

    /// Typed vector **element-intersection to a mask pair** (AVX-512
    /// `vp2intersectd`/`vp2intersectq`). Pure. The payload is boxed (see
    /// [`VectorIntersectData`]).
    VectorIntersect(Box<VectorIntersectData>),

    /// Typed vector **bit-shuffle to a mask** (AVX-512 `vpshufbitqmb`). Pure.
    /// The payload is boxed (see [`VectorShuffleBitsData`]).
    VectorShuffleBits(Box<VectorShuffleBitsData>),

    /// Typed vector **per-byte conditional move** (Cyrix EMMI `pmvzb`/`pmvnzb`/
    /// `pmvlzb`/`pmvgezb`). Pure. The payload is boxed (see
    /// [`VectorConditionalMoveData`]).
    VectorConditionalMove(Box<VectorConditionalMoveData>),

    /// Typed vector **horizontal minimum with position** (SSE4.1 `phminposuw`):
    /// the smallest unsigned 16-bit lane and its source index. Pure. The
    /// payload is boxed (see [`VectorHorizontalMinPosData`]).
    VectorHorizontalMinPos(Box<VectorHorizontalMinPosData>),

    /// Typed vector **complex-number floating-point multiply** (x86 `vfmulcph`/
    /// `vfcmulcph`/`vfmaddcph`/`vfcmaddcph` and `sh` peers). Pure. The payload
    /// is boxed (see [`KindedVecData`]).
    VectorComplexMul(Box<KindedVecData<ComplexMulKind>>),

    /// Typed vector **floating-point lane classification to a mask** (x86
    /// `vfpclass*`), selected by an 8-bit category immediate. Pure. The payload
    /// is boxed (see [`VecImm8Data`]).
    VectorClassify(Box<VecImm8Data>),

    /// Typed vector **grouped widening horizontal add / subtract** (AMD XOP
    /// `vphadd*`/`vphsub*`). Pure. The payload is boxed (see
    /// [`VectorHorizontalReduceData`]).
    VectorHorizontalReduce(Box<VectorHorizontalReduceData>),

    /// Typed native **block-string** operation (`rep`/`repe`/`repne`
    /// `cmps`/`scas`/`lods`). The first-class typed replacement for the
    /// rep-string-stream case of `NativeOpaque`: a structured
    /// [`BlockStringKind`] identity drives a precise effect summary (memory
    /// read / read-write) and a distinct similarity class. The payload is boxed
    /// (see [`BlockStringOpData`]). (`rep movs`/`rep stos` use `CopyBlk`/
    /// `InitBlk`, not this op.)
    ///
    /// [`BlockStringKind`]: crate::ir::ops::native::BlockStringKind
    BlockString(Box<BlockStringOpData>),

    /// Typed native **wide compare-and-swap** (`cmpxchg8b` / `cmpxchg16b`). The
    /// first-class typed replacement for the wide-CAS case of `NativeOpaque`: a
    /// sequentially-consistent atomic op with explicit register-pair inputs /
    /// outputs (see [`WideCmpXchgData`]). Single-width CAS uses [`Self::CmpXchg`].
    WideCompareExchange(Box<WideCmpXchgData>),

    /// Typed native **flags computation** — defines an architectural-flags
    /// value (`EFLAGS` / NZCV) as a pure function of its inputs, for operations
    /// whose precise per-flag semantics the lifter does not decompose (`bsf`/
    /// `bsr`/`popcnt`/`bt` zero-flag side effects). The first-class typed
    /// replacement for the flags-only case of `NativeOpaque`: it is pure and
    /// deterministic, so optimization can still reason about and eliminate it.
    ComputeFlags {
        /// The defined architectural-flags value.
        dest: SsaVarId,
        /// The input values the flags are computed from.
        inputs: Vec<SsaVarId>,
    },

    /// Typed native **call-clobber** marker — defines fresh, undefined values
    /// for the caller-saved registers a preceding [`Self::Call`] clobbers (the
    /// `Call` op carries a single `dest`, so the remaining clobbered registers
    /// need an owning def for the verifier). The first-class typed replacement
    /// for the call-clobber case of `NativeOpaque`: pure (the call already
    /// happened) and freely eliminable when its outputs are unread.
    CallClobber {
        /// The caller-saved register values invalidated by the call.
        outputs: Vec<SsaVarId>,
    },

    /// Compare-and-swap: `old = *addr; if old == expected { *addr = desired; } return old`
    CmpXchg {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Expected compare-exchange value.
        expected: SsaVarId,
        /// Desired compare-exchange value.
        desired: SsaVarId,
    },

    /// Atomic read-modify-write: `old = *addr; *addr = op(old, value); return old`
    AtomicRmw {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Atomic read-modify-write operation.
        op: AtomicRmwOp,
    },

    /// Native atomic load with explicit ordering, width, and volatility.
    AtomicLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Loaded value type.
        value_type: T::Type,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic store with explicit ordering, width, and volatility.
    AtomicStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Stored value type.
        value_type: T::Type,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native store-conditional with explicit status output and ordering.
    AtomicStoreConditional {
        /// Store status output variable.
        status: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Stored value type.
        value_type: T::Type,
        /// Ordering used when the conditional store succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when the conditional store fails.
        failure_ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic pair load with explicit ordering, width, and volatility.
    AtomicPairLoad {
        /// First destination SSA variable.
        first: SsaVarId,
        /// Second destination SSA variable.
        second: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// First loaded value type.
        first_type: T::Type,
        /// Second loaded value type.
        second_type: T::Type,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Total atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native pair store-conditional with one shared status output.
    AtomicPairStoreConditional {
        /// Store status output variable.
        status: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// First value operand variable.
        first_value: SsaVarId,
        /// Second value operand variable.
        second_value: SsaVarId,
        /// First stored value type.
        first_type: T::Type,
        /// Second stored value type.
        second_type: T::Type,
        /// Ordering used when the conditional store succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when the conditional store fails.
        failure_ordering: AtomicOrdering,
        /// Total atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic exchange with explicit ordering, width, and volatility.
    AtomicExchange {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native lock-prefixed read-modify-write operation.
    AtomicLockRmw {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Atomic read-modify-write operation.
        op: AtomicRmwOp,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic compare-exchange with optional success-status output.
    AtomicCmpXchg {
        /// Old memory value output variable.
        old: SsaVarId,
        /// Optional compare-exchange success output variable.
        success: Option<SsaVarId>,
        /// Address operand variable.
        addr: SsaVarId,
        /// Expected compare-exchange value.
        expected: SsaVarId,
        /// Desired compare-exchange value.
        desired: SsaVarId,
        /// Ordering used when compare-exchange succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when compare-exchange fails.
        failure_ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether compare-exchange may fail spuriously.
        weak: bool,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Native atomic pair compare-exchange.
    AtomicPairCmpXchg {
        /// First old memory value output variable.
        old_first: SsaVarId,
        /// Second old memory value output variable.
        old_second: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// First expected compare-exchange value.
        expected_first: SsaVarId,
        /// Second expected compare-exchange value.
        expected_second: SsaVarId,
        /// First desired compare-exchange value.
        desired_first: SsaVarId,
        /// Second desired compare-exchange value.
        desired_second: SsaVarId,
        /// Ordering used when compare-exchange succeeds.
        success_ordering: AtomicOrdering,
        /// Ordering used when compare-exchange fails.
        failure_ordering: AtomicOrdering,
        /// Total atomic memory access width.
        width: AtomicAccessWidth,
        /// Whether compare-exchange may fail spuriously.
        weak: bool,
        /// Whether the access is volatile.
        volatile: bool,
    },

    /// Initialize object (for value types): `*dest = default(T)`
    InitObj {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Copy object (for value types): `*dest = *src`
    CopyObj {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Load object (value type copy): `dest = *src`
    LoadObj {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// Store object (value type copy): `*dest = value`
    StoreObj {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
    },

    /// No operation (for nop instructions)
    Nop,

    /// Breakpoint or undefined-instruction trap, named by its [`BreakpointOp`].
    ///
    /// The payload is what lets a renderer tell `ud2` from `int3` from `brk`:
    /// they lower alike, but they are not the same instruction and the cleaned
    /// IR carries no mnemonic string to fall back on.
    Break(BreakpointOp),

    /// Check for finite floating point: throws if not finite
    Ckfinite {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Operand variable.
        operand: SsaVarId,
    },

    /// Classify a floating-point value, producing an integer mask describing the
    /// operand's IEEE-754 category (RISC-V `fclass`, MIPS `class.fmt`). The exact
    /// bit layout is the native instruction's; the result is a plain integer.
    FpClassify {
        /// Destination SSA variable (integer classification mask).
        dest: SsaVarId,
        /// Floating-point operand variable.
        operand: SsaVarId,
    },

    /// Hardware floating-point transcendental / residue op with no primitive
    /// closed form (x87 `fsin`/`fcos`/`fpatan`/`f2xm1`/`fyl2x`/`fprem`/…). The
    /// payload ([`KindedVecData`]) is boxed to keep `SsaOp` compact.
    FpTranscendental(Box<KindedVecData<TranscendentalKind>>),

    /// Floating-point unit control / state op (x87 `fldcw`/`fnstcw`/`fnstsw`/
    /// `fldenv`/`fnsave`/`frstor`/`fnclex`/`fdecstp`/`ffree`/`fxsave`/…). The
    /// payload ([`KindedVecData`]) is boxed to keep `SsaOp` compact.
    FpuControl(Box<KindedVecData<FpuControlKind>>),

    /// Localloc: allocate stack space
    LocalAlloc {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
    },

    /// Constrained virtual call prefix (affects next callvirt)
    Constrained {
        /// Constrained call type metadata.
        constraint_type: T::TypeRef,
    },

    /// Volatile prefix (next memory access must not be reordered/cached)
    Volatile,

    /// Unaligned prefix (next memory access may be unaligned)
    Unaligned {
        /// Required alignment in bytes.
        alignment: u8,
    },

    /// Tail call prefix (next call is a tail call)
    TailPrefix,

    /// Readonly prefix (next ldelema returns a controlled-mutability managed pointer)
    Readonly,

    /// Phi node: merges values from different predecessors.
    ///
    /// This is placed at the beginning of blocks with multiple predecessors.
    Phi {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Incoming phi operands by predecessor block.
        operands: Vec<(usize, SsaVarId)>,
    },
}
